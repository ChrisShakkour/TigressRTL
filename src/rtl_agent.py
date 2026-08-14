import argparse
import dataclasses
import sys

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from config import DEFAULT_CONFIG_PATH, load_config
from model import build_llm, build_system_prompt, FIX_PLAN_INSTRUCTION, PLANNING_INSTRUCTION
from tools import BASE_TOOLS, BUILD_FAILURE_PREFIXES, BUILD_TOOL_FUNCTIONS, load_project_memory

# Windows terminals default to a codepage that can't render some characters —
# force UTF-8 output so nothing gets garbled.
sys.stdout.reconfigure(encoding="utf-8")

parser = argparse.ArgumentParser(description="Chat with a SystemVerilog RTL design assistant.")
parser.add_argument(
    "--config", default=str(DEFAULT_CONFIG_PATH),
    help="Path to a YAML config file (see configs/default.yaml)",
)
parser.add_argument(
    "--model", default=None,
    help="Override the model from the config, e.g. llama3.2, qwen2.5-coder "
         "(must already be pulled via `ollama pull`)",
)
args = parser.parse_args()


def accumulate_tokens(totals: dict, response) -> None:
    # ChatOllama populates AIMessage.usage_metadata on every non-streaming
    # .invoke() call (confirmed directly against a live response) — the
    # `or {}`/.get(..., 0) guards are defensive only, not expected to
    # actually trigger with this provider.
    usage = getattr(response, "usage_metadata", None) or {}
    totals["input_tokens"] += usage.get("input_tokens", 0)
    totals["output_tokens"] += usage.get("output_tokens", 0)


# One agent process handles exactly one problem in the benchmark harness
# (test/run_benchmark.py runs a fresh subprocess per problem), so a
# process-lifetime total is already a per-problem total — no extra
# scoping needed.
token_totals = {"input_tokens": 0, "output_tokens": 0}

config = load_config(args.config)
if args.model:
    # A one-off override shouldn't require writing a new YAML file — this
    # takes precedence over whatever the config file says.
    config = dataclasses.replace(config, model=args.model)

# config.verilog_build_tool selects exactly one build/lint backend to bind
# to the model — never both, since the point is picking one, not offering
# a confusing choice to the LLM. TOOLS/TOOL_FUNCTIONS are built here rather
# than imported as a fixed list, since the active tool is config-driven.
active_build_tool = BUILD_TOOL_FUNCTIONS[config.verilog_build_tool]
active_build_tool_name = active_build_tool.name
build_failure_prefix = BUILD_FAILURE_PREFIXES[config.verilog_build_tool]
TOOLS = BASE_TOOLS + [active_build_tool]
TOOL_FUNCTIONS = {t.name: t for t in TOOLS}

# `llm` is our handle to the model. `.bind_tools(TOOLS)` returns a new
# runnable that knows about each tool's schema and may respond with
# tool_calls instead of (or alongside) plain text — the model itself never
# executes anything, it only ever *requests* a call.
llm = build_llm(config)
llm_with_tools = llm.bind_tools(TOOLS)

# Fold prior sessions' recorded conventions on this same project (see
# tools.py's update_project_memory) into the system prompt, if any exist
# yet — a fresh project (or a fresh benchmark-harness temp dir, which
# never has one) just gets nothing appended here, no special-casing
# needed. Loaded once at startup, not re-checked mid-session — the model
# updates the in-context copy via update_project_memory the same way it
# updates any other state during a session.
project_memory = load_project_memory()
system_prompt = build_system_prompt(active_build_tool_name)
if project_memory:
    system_prompt += (
        "\n\nProject memory from earlier sessions on this project — "
        "established conventions, follow them unless the user tells you "
        "otherwise:\n\n" + project_memory
    )

# `messages` is the full conversation history sent on every request —
# the API is stateless, so the whole transcript is resent each time. Each
# turn is a typed message object:
#   SystemMessage(...) - instructions, sent once up front
#   HumanMessage(...)  - you
#   AIMessage(...)     - the model's turn (may carry .tool_calls)
#   ToolMessage(...)   - a tool's result, tagged with which call it answers
messages = [SystemMessage(content=system_prompt)]

print("Chatting with", config.model, "— a SystemVerilog RTL design assistant.")
if project_memory:
    print(f"[Memory] Loaded {len(project_memory)} characters of project memory.")
print("Type 'exit' or 'quit' to stop.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in {"exit", "quit"}:
        print(
            f"[Token Usage] input_tokens={token_totals['input_tokens']} "
            f"output_tokens={token_totals['output_tokens']} "
            f"total_tokens={token_totals['input_tokens'] + token_totals['output_tokens']}"
        )
        break
    if not user_input:
        continue

    messages.append(HumanMessage(content=user_input))

    # Planning phase: call the PLAIN `llm` — no tools bound — so the model
    # is structurally incapable of acting yet, no matter what it decides.
    # This is a hard guarantee from the code, not a hope that a "plan
    # before acting" prompt instruction gets followed — local models have
    # repeatedly been observed skipping or misordering such instructions.
    plan_response = llm.invoke(messages + [HumanMessage(content=PLANNING_INSTRUCTION)])
    accumulate_tokens(token_totals, plan_response)
    print("[Plan]", plan_response.content, "\n")
    messages.append(plan_response)

    # Without this, `messages` would end in two consecutive assistant turns
    # (the plan, then immediately another assistant response with no new
    # user turn in between) once the execution loop below calls the model
    # again — an unusual pattern outside normal alternating user/assistant
    # chat structure that some models handle poorly (observed: an entirely
    # empty response with zero tool_calls). A short synthetic user turn
    # restores normal alternation and gives an explicit cue to switch from
    # planning to acting.
    messages.append(HumanMessage(content="Proceed with your plan now, using the available tools."))

    # Tracks whether the most recent auto-build (below) failed and hasn't
    # been fixed yet, and how many times we've already nudged for a fix
    # this turn — reset per user turn. See the retry check further down.
    build_failed = False
    build_retry_count = 0

    # Set whenever an auto-build freshly fails (below), so the very next
    # model turn gets a short, tools-unbound "diagnose and plan the fix"
    # call first — same structural-guarantee pattern as the initial
    # PLANNING_INSTRUCTION (a plain llm.invoke() with no tools bound, not a
    # prompt hope), since local models have been observed diving straight
    # into another guessed edit without pausing to actually read the error.
    # Cleared once that plan call has been made, so a retry nudge (model
    # skipped acting, not a fresh failure) doesn't trigger a second plan
    # for the same error.
    fix_plan_pending = False

    # Inner loop: the ReAct cycle for this one turn — keep calling the
    # model and executing whatever tools it requests until a response has
    # no more tool_calls, which is the model's final answer for this turn.
    while True:
        if fix_plan_pending:
            fix_plan_response = llm.invoke(messages + [HumanMessage(content=FIX_PLAN_INSTRUCTION)])
            accumulate_tokens(token_totals, fix_plan_response)
            print("[Fix Plan]", fix_plan_response.content, "\n")
            messages.append(fix_plan_response)
            messages.append(HumanMessage(content="Now apply that fix using the available tools."))
            fix_plan_pending = False

        try:
            response = llm_with_tools.invoke(messages)
        except Exception as e:
            # Some models have no tool-calling support in Ollama at all and
            # reject any request with tools bound. That 400 comes back as a
            # raw ResponseError traceback by default; recognize it and fail
            # with an actionable message instead, since every subsequent
            # turn would hit the same wall.
            if "does not support tools" in str(e):
                print(
                    f"\nError: model '{config.model}' does not "
                    "support tool calling in Ollama.\nThis agent's tools (write_file, "
                    f"read_file, edit_file_block, list_directory, update_project_memory, "
                    f"{active_build_tool_name}) require a tool-capable model.\nTry --model "
                    "llama3.2 instead — see "
                    "README.md for what's been tested."
                )
                sys.exit(1)
            raise
        accumulate_tokens(token_totals, response)
        messages.append(response)

        # `response.tool_calls` is a list of dicts:
        #   {"name": "write_file", "args": {...}, "id": "..."}
        if not response.tool_calls:
            # A response with no tool_calls is only treated as the real
            # final answer if the last known build actually succeeded (or
            # never ran). A model can correctly diagnose a build failure in
            # prose, say it will fix it, and then simply not call
            # write_file/edit_file_block in that same response — auto-build
            # enforces that a failure is *seen*, but nothing forces it to be
            # *acted on*. So otherwise, nudge for a genuine fix attempt and
            # keep the loop going, up to config.max_build_retries times.
            if build_failed and build_retry_count < config.max_build_retries:
                build_retry_count += 1
                print(f"[Retry] Build is still failing and no fix was applied — "
                      f"forcing attempt {build_retry_count}/{config.max_build_retries}.")
                messages.append(HumanMessage(content=(
                    f"The last {active_build_tool_name} result was a failure, and you "
                    "did not call write_file or edit_file_block to actually apply a "
                    "fix — you only described one. Call the appropriate tool now "
                    "with the corrected content."
                )))
                continue
            break  # model gave its final answer for this turn — inner loop ends

        for call in response.tool_calls:
            name = call["name"]
            print("[Action]", name, call["args"])

            # TOOL_FUNCTIONS[name] is a StructuredTool object (from @tool),
            # not a plain function — it must be called via .invoke(args),
            # passing the whole args dict, not unpacked as **args.
            tool_fn = TOOL_FUNCTIONS[name]
            result = tool_fn.invoke(call["args"])

            # Enforcement, not a request: the system prompt asks the model to
            # always build after writing, but that isn't reliable — models
            # have skipped the write entirely, called the build tool on a
            # file that doesn't exist yet, and narrated fake write_file(...)/
            # build-tool(...) calls as plain text instead of actually
            # invoking them. So instead of trusting the model to remember,
            # the harness runs the build itself right after any successful
            # write/edit, and folds the result into the SAME tool response —
            # the model sees it whether or not it asked.
            if name in ("write_file", "edit_file_block") and result.startswith(("Wrote", "Replaced")):
                file_path = call["args"].get("file_path")
                build_result = active_build_tool.invoke({"file_path": file_path})
                print("[Auto-Build]", build_result)
                # Drives the retry check above: a real, current failure sets
                # this True; a successful build clears it (and resets the
                # retry count) so a later, unrelated failure gets its own
                # fresh set of attempts rather than inheriting an old count.
                build_failed = build_result.startswith(build_failure_prefix)
                if not build_failed:
                    build_retry_count = 0
                else:
                    fix_plan_pending = True
                result = f"{result}\n\n[automatically ran {active_build_tool_name} after {name}]\n{build_result}"

            # Print the tool's actual return value directly — never rely on
            # the model's own later paraphrase of it. Models have been
            # observed confidently misreporting a tool result (e.g.
            # describing a full module when read_file actually returned a
            # single "{"), so this is the ground truth, shown regardless of
            # what the model goes on to say.
            print("[Result]", result)

            # ToolMessage links the result back to the specific call it
            # answers via tool_call_id — the model matches these up itself.
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    final = messages[-1]
    print("Assistant:", final.content or "(no final text — see actions above)", "\n")
