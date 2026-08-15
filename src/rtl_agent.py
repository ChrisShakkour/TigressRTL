import argparse
import dataclasses
import sys

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from config import DEFAULT_CONFIG_PATH, load_config
from model import build_llm, build_system_prompt, FIX_PLAN_INSTRUCTION, PLANNING_INSTRUCTION
from tools import (
    BASE_TOOLS, BUILD_FAILURE_PREFIXES, BUILD_TOOL_FUNCTIONS,
    SIMULATE_FAILURE_PREFIX, project_has_testbench, simulate_verilog,
)
from verify import VerificationState

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

# simulate_verilog only makes sense with icarus — Slang is a compiler/linter,
# it has no simulator to run at all. Bound as an extra tool alongside
# active_build_tool, not through BUILD_TOOL_FUNCTIONS (that dict picks one
# alternative *build* backend; this is an additional capability layered on
# top of one specific backend, not a competing choice).
can_simulate = config.verilog_build_tool == "icarus"
TOOLS = BASE_TOOLS + [active_build_tool] + ([simulate_verilog] if can_simulate else [])
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
messages = [SystemMessage(content=build_system_prompt(active_build_tool_name, can_simulate))]

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

    # Tracks what's actually been proven about the current on-disk files
    # this turn — build passed? simulated clean, if this project needs
    # that? how many retries used? — reset per user turn. See verify.py.
    verify_state = VerificationState()

    # Inner loop: the ReAct cycle for this one turn — keep calling the
    # model and executing whatever tools it requests until a response has
    # no more tool_calls, which is the model's final answer for this turn.
    while True:
        if verify_state.fix_plan_pending:
            # Same structural-guarantee pattern as the initial
            # PLANNING_INSTRUCTION (a plain llm.invoke() with no tools
            # bound, not a prompt hope) — local models have been observed
            # diving straight into another guessed edit without pausing to
            # actually read the error, whether that error came from a
            # build, a lint, or a simulation attempt.
            fix_plan_response = llm.invoke(messages + [HumanMessage(content=FIX_PLAN_INSTRUCTION)])
            accumulate_tokens(token_totals, fix_plan_response)
            print("[Fix Plan]", fix_plan_response.content, "\n")
            messages.append(fix_plan_response)
            messages.append(HumanMessage(content="Now apply that fix using the available tools."))
            verify_state.fix_plan_pending = False

        try:
            response = llm_with_tools.invoke(messages)
        except Exception as e:
            # Some models have no tool-calling support in Ollama at all and
            # reject any request with tools bound. That 400 comes back as a
            # raw ResponseError traceback by default; recognize it and fail
            # with an actionable message instead, since every subsequent
            # turn would hit the same wall.
            if "does not support tools" in str(e):
                tool_names = ", ".join(t.name for t in TOOLS)
                print(
                    f"\nError: model '{config.model}' does not "
                    f"support tool calling in Ollama.\nThis agent's tools ({tool_names}) "
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
            # final answer if verify_state says nothing's missing: build
            # passed, and simulate ran clean too if this project needs that
            # (see project_has_testbench). A model can correctly diagnose a
            # failure in prose, say it will fix it, and then simply not
            # call the right tool in that same response — auto-build/
            # auto-simulate enforce that a failure is *seen*, but nothing
            # forces it to be *acted on*. Likewise a model can just forget
            # to call simulate_verilog at all once a testbench exists,
            # nothing failed, it just never tried. Either way: nudge and
            # keep the loop going, up to config.max_build_retries times.
            simulate_required = can_simulate and project_has_testbench()
            if verify_state.needs_retry(simulate_required, config.max_build_retries):
                verify_state.retry_count += 1
                reason = "Build/lint is still failing" if not verify_state.build_passed \
                    else "A testbench hasn't been simulated yet"
                print(f"[Retry] {reason} and no fix was applied — "
                      f"forcing attempt {verify_state.retry_count}/{config.max_build_retries}.")
                messages.append(HumanMessage(
                    content=verify_state.nudge_message(active_build_tool_name, simulate_verilog.name)
                ))
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
                build_passed = not build_result.startswith(build_failure_prefix)
                verify_state.record_build(build_passed)
                result = f"{result}\n\n[automatically ran {active_build_tool_name} after {name}]\n{build_result}"

                # Same enforcement, one phase further: a clean build only
                # proves the design elaborates, not that it behaves
                # correctly, so if this project has a testbench and can
                # actually simulate (icarus only — see can_simulate above),
                # auto-run it too and fold its real output into the same
                # tool response the model sees.
                if build_passed and can_simulate and project_has_testbench():
                    sim_result = simulate_verilog.invoke({})
                    print("[Auto-Simulate]", sim_result)
                    sim_ran_clean = not sim_result.startswith(SIMULATE_FAILURE_PREFIX)
                    verify_state.record_simulate(sim_ran_clean)
                    result = f"{result}\n\n[automatically ran simulate_verilog after a clean build]\n{sim_result}"

            # The system prompt also tells the model to call build/lint or
            # simulate itself — an explicit call (e.g. re-verifying without
            # a fresh edit) has to update verify_state the same way an
            # auto-triggered one does, or a failure from a model-initiated
            # call would go unnoticed by the retry/fix-plan enforcement.
            elif name == active_build_tool_name:
                verify_state.record_build(not result.startswith(build_failure_prefix))
            elif name == "simulate_verilog":
                verify_state.record_simulate(not result.startswith(SIMULATE_FAILURE_PREFIX))

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
