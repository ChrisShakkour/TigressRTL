import argparse
import dataclasses
import re
import sys

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from config import DEFAULT_CONFIG_PATH, load_config
from model import (
    build_llm, build_system_prompt, FIX_PLAN_INSTRUCTION, PLANNING_INSTRUCTION,
    SIM_FIX_PLAN_INSTRUCTION, force_apply_instruction,
)
from tools import (
    BASE_TOOLS, BUILD_FAILURE_PREFIXES, BUILD_TOOL_FUNCTIONS, SIM_FAILURE_PREFIX,
    make_run_simulation_tool,
)

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

# run_simulation is only ever built/bound when explicitly enabled — with the
# flag off, the model never sees it as a callable tool at all (true baseline
# parity with the pre-simulation agent), not just "auto-chain skipped but
# still manually callable". A factory, not a bare tool, because the VCD
# debug window's size is config-driven (sim_debug_window_cycles) but must
# stay out of the tool's LLM-visible schema — see make_run_simulation_tool.
if config.enable_simulation:
    run_simulation = make_run_simulation_tool(config.sim_debug_window_cycles)
    TOOLS = TOOLS + [run_simulation]

TOOL_FUNCTIONS = {t.name: t for t in TOOLS}

# `llm` is our handle to the model. `.bind_tools(TOOLS)` returns a new
# runnable that knows about each tool's schema and may respond with
# tool_calls instead of (or alongside) plain text — the model itself never
# executes anything, it only ever *requests* a call.
llm = build_llm(config)
llm_with_tools = llm.bind_tools(TOOLS)

# `messages` is the full conversation history sent on every request —
# the API is stateless, so the whole transcript is resent each time. Each
# turn is a typed message object:
#   SystemMessage(...) - instructions, sent once up front
#   HumanMessage(...)  - you
#   AIMessage(...)     - the model's turn (may carry .tool_calls)
#   ToolMessage(...)   - a tool's result, tagged with which call it answers
messages = [SystemMessage(content=build_system_prompt(active_build_tool_name, config.enable_simulation))]

_CODE_BLOCK_RE = re.compile(r"```(?:\w+\n)?(.*?)```", re.DOTALL)


def _extract_code_block(text: str):
    match = _CODE_BLOCK_RE.search(text or "")
    return match.group(1).strip() if match else None


def _run_write_auto_chain(file_path: str, write_result: str, action_name: str) -> str:
    # Shared by the normal tool-dispatch loop below and _force_apply_fix —
    # same auto-build/auto-simulate enforcement regardless of whether the
    # write came from a real model tool call or a harness-forced one, so a
    # forced write is held to exactly the same bar as a normal one.
    global build_failed, build_retry_count, sim_failed, sim_retry_count, fix_plan_pending, fix_plan_kind

    build_result = active_build_tool.invoke({"file_path": file_path})
    print("[Auto-Build]", build_result)
    build_failed = build_result.startswith(build_failure_prefix)
    extra = f"\n\n[automatically ran {active_build_tool_name} after {action_name}]\n{build_result}"

    if build_failed:
        fix_plan_pending = True
        fix_plan_kind = "build"
        sim_failed = False
    else:
        build_retry_count = 0
        if config.enable_simulation:
            sim_result = run_simulation.invoke({"file_path": file_path})
            print("[Auto-Simulate]", sim_result)
            sim_failed = sim_result.startswith(SIM_FAILURE_PREFIX)
            if sim_failed:
                fix_plan_pending = True
                fix_plan_kind = "sim"
            else:
                sim_retry_count = 0
            extra += (
                f"\n\n[automatically ran run_simulation after successful "
                f"{active_build_tool_name}]\n{sim_result}"
            )

    return f"{write_result}{extra}"


def _force_apply_fix(file_path: str) -> bool:
    # Last resort when the model has repeatedly failed to act on its own
    # diagnosis (see force_apply_instruction's docstring-style comment in
    # model.py for why this exists at all — Ollama has no way to force a
    # tool call). Returns False (meaning: fall back to the plain text nudge)
    # if the response can't be turned into a real write, so a malformed or
    # refused response never corrupts the file with garbage.
    force_response = llm.invoke(messages + [HumanMessage(content=force_apply_instruction(file_path))])
    accumulate_tokens(token_totals, force_response)
    print("[Force Apply]", force_response.content, "\n")

    code = _extract_code_block(force_response.content)
    if code is None:
        print("[Force Apply] No parseable code block in the response — falling back to a nudge.")
        return False

    write_result = TOOL_FUNCTIONS["write_file"].invoke({"file_path": file_path, "content": code})
    print("[Result]", write_result)
    if not write_result.startswith(("Wrote", "Replaced")):
        return False

    chained_result = _run_write_auto_chain(file_path, write_result, "write_file")
    messages.append(AIMessage(content=force_response.content))
    messages.append(HumanMessage(content=(
        "You did not call a tool, so the harness extracted the code from "
        f"your response above and wrote it to {file_path} directly:\n\n{chained_result}"
    )))
    return True


print("Chatting with", config.model, "— a SystemVerilog RTL design assistant.")
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

    # Same bookkeeping as build_failed/build_retry_count above, but for a
    # simulation that compiled fine and then failed at runtime
    # ($fatal/$error) — a distinct failure mode with its own retry budget
    # (config.max_sim_retries). Both stay permanently False/0 (never
    # touched) when config.enable_simulation is off.
    sim_failed = False
    sim_retry_count = 0

    # Set whenever an auto-build or auto-simulation freshly fails (below),
    # so the very next model turn gets a short, tools-unbound "diagnose and
    # plan the fix" call first — same structural-guarantee pattern as the
    # initial PLANNING_INSTRUCTION (a plain llm.invoke() with no tools
    # bound, not a prompt hope), since local models have been observed
    # diving straight into another guessed edit without pausing to actually
    # read the error. Cleared once that plan call has been made, so a retry
    # nudge (model skipped acting, not a fresh failure) doesn't trigger a
    # second plan for the same error. fix_plan_kind picks which instruction
    # applies — a compile error and a runtime failure need different
    # diagnostic questions, and the two are mutually exclusive at any given
    # moment (simulation only ever runs once a build has just passed).
    fix_plan_pending = False
    fix_plan_kind = None  # "build" or "sim"

    # The file a forced-retry fallback (_force_apply_fix) targets if the
    # model fails to act on its own diagnosis — the most recent file any
    # write/edit (real or forced) touched this turn. None until the first
    # successful write, in which case there's nothing to force-apply to.
    last_write_file_path = None

    # Inner loop: the ReAct cycle for this one turn — keep calling the
    # model and executing whatever tools it requests until a response has
    # no more tool_calls, which is the model's final answer for this turn.
    while True:
        if fix_plan_pending:
            if fix_plan_kind == "sim":
                fix_instruction, fix_label = SIM_FIX_PLAN_INSTRUCTION, "[Sim Fix Plan]"
            else:
                fix_instruction, fix_label = FIX_PLAN_INSTRUCTION, "[Fix Plan]"
            fix_plan_response = llm.invoke(messages + [HumanMessage(content=fix_instruction)])
            accumulate_tokens(token_totals, fix_plan_response)
            print(fix_label, fix_plan_response.content, "\n")
            messages.append(fix_plan_response)
            messages.append(HumanMessage(content="Now apply that fix using the available tools."))
            fix_plan_pending = False
            fix_plan_kind = None

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
                    f"read_file, edit_file_block, list_directory, {active_build_tool_name}) "
                    "require a tool-capable model.\nTry --model llama3.2 instead — see "
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
                if last_write_file_path and _force_apply_fix(last_write_file_path):
                    continue
                messages.append(HumanMessage(content=(
                    f"The last {active_build_tool_name} result was a failure, and you "
                    "did not call write_file or edit_file_block to actually apply a "
                    "fix — you only described one. Call the appropriate tool now "
                    "with the corrected content."
                )))
                continue
            if sim_failed and sim_retry_count < config.max_sim_retries:
                sim_retry_count += 1
                print(f"[Retry] Simulation is still failing and no fix was applied — "
                      f"forcing attempt {sim_retry_count}/{config.max_sim_retries}.")
                if last_write_file_path and _force_apply_fix(last_write_file_path):
                    continue
                messages.append(HumanMessage(content=(
                    "The last run_simulation result was a failure, and you did not "
                    "call write_file or edit_file_block to actually apply a fix — "
                    "you only described one. Call the appropriate tool now with the "
                    "corrected content."
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
                last_write_file_path = file_path
                # Drives the retry check above, and the (im)possibility of
                # a simulation step: see _run_write_auto_chain — a real,
                # current build failure sets build_failed True; a successful
                # build clears it (and resets the retry count) so a later,
                # unrelated failure gets its own fresh set of attempts
                # rather than inheriting an old count. This *is* the "only
                # once the build is passing" gate for simulation, enforced
                # structurally inside the helper, not left to the model's
                # judgment.
                result = _run_write_auto_chain(file_path, result, name)

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
