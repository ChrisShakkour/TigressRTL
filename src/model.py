from langchain_ollama import ChatOllama

from config import AgentConfig

# The one swappable model/provider point in this project — everything else
# (tools.py, rtl_agent.py) talks to `build_llm()`'s return value, not to
# ChatOllama directly, so switching providers later only means editing here.
# Model choice and context window live in AgentConfig (see config.py) rather
# than as constants here, so they're driven by the YAML config instead of
# requiring a code edit to change.

def build_system_prompt(build_tool_name: str, enable_simulation: bool = False) -> str:
    # A function rather than a plain constant so the build/fix cycle it
    # describes always names whichever tool is actually bound
    # (config.verilog_build_tool in config.py) — build_verilog or
    # lint_verilog — instead of hardcoding one. enable_simulation likewise
    # controls whether the simulation paragraph below is even mentioned —
    # when it's off, run_simulation isn't bound to the model at all (see
    # rtl_agent.py), so the prompt must never reference a tool that isn't
    # actually there.
    simulation_paragraph = (
        f"Once {build_tool_name} reports no errors, if the project already "
        "contains a self-checking testbench (any file that calls $finish), "
        "the harness automatically compiles and runs a simulation for you "
        "with run_simulation — you don't need to call it yourself unless "
        "you want to re-check something. A runtime $fatal or $error means "
        "the simulation failed: read the signal-value table around the "
        "failure time (when one is shown) to see the actual values leading "
        "into it, fix the RTL with write_file or edit_file_block, and it "
        "will be re-verified automatically. Do not give your final answer "
        "while a simulation failure is unresolved. If the project has no "
        "testbench yet, run_simulation has nothing to check and this "
        "doesn't apply — writing one isn't required unless asked for."
    ) if enable_simulation else ""

    return (
        "You are an expert SystemVerilog RTL designer. Write clean, "
        "synthesizable RTL: use always_ff for sequential logic and "
        "always_comb or assign for combinational logic, and non-blocking assignments "
        "(<=) inside always_ff blocks (never inside always_comb). "
        "Declare every port and every signal assigned inside a procedural "
        "block as logic — never wire or bare output, and never declare a "
        "wire inside an always block. For a SYNCHRONOUS reset, put the "
        "clock alone in the sensitivity list (@(posedge clk)) and check "
        "the reset signal only inside the block body; only put reset in "
        "the sensitivity list (@(posedge clk or posedge/negedge reset)) "
        "for an explicitly ASYNCHRONOUS reset. Use the exact module name "
        "given in the request, character-for-character — never rename it, "
        "even when fixing or rewriting existing code. If a request "
        "includes a Karnaugh map or truth table, read the row/column "
        "labels carefully before deriving the logic equation. If a "
        "request is ambiguous — missing bit widths, reset polarity, clock "
        "domain, or similar details — ask a clarifying question or "
        "clearly state the assumptions you're making, rather than "
        "silently guessing, systemverilog files all end with .sv. "
        f"Whenever you write or edit a file, you must call {build_tool_name} on it "
        "afterward. If the compilation log reports any errors, fix them with "
        f"write_file or edit_file_block and call {build_tool_name} again — repeat "
        "this build-and-fix cycle until the log is clean. Do not give your "
        "final answer, and do not claim the code is correct or complete, "
        f"until {build_tool_name} has actually reported no errors."
        + (f" {simulation_paragraph}" if simulation_paragraph else "")
    )

# Used for the planning-phase call only (see rtl_agent.py) — no tools are
# bound for that call, so the model is structurally unable to act yet no
# matter what it decides; this just shapes what it writes in that turn.
PLANNING_INSTRUCTION = (
    "Before any tool is available to you, write a short plan (3-6 bullet "
    "points) for how you will fulfill this request: what module(s) you'll "
    "create, their ports, and the order of write/build steps. Do not write "
    "SystemVerilog code yet — just the plan."
)

# Used for the fix-plan call only (see rtl_agent.py), triggered every time
# an auto-build freshly fails — same structural guarantee as
# PLANNING_INSTRUCTION (no tools bound for this call), so the model can't
# skip straight to another guessed edit without first diagnosing the
# specific error in front of it.
FIX_PLAN_INSTRUCTION = (
    "The build/lint attempt above just failed. Before any tool is "
    "available to you, write a short plan (2-4 bullet points): what "
    "specifically the error log says is wrong, and the exact change "
    "you'll make to fix it. Do not write SystemVerilog code yet — just "
    "the plan, make sure to keep the signal names consistent with the original code"
)

# Used for the simulation fix-plan call only (see rtl_agent.py), triggered
# every time an auto-simulation freshly fails — same structural guarantee as
# FIX_PLAN_INSTRUCTION, but aimed at a *runtime* failure (the design compiled
# fine, the RTL just behaves wrong) rather than a compile error, so it points
# the model at the signal-value table instead of a compiler diagnostic.
SIM_FIX_PLAN_INSTRUCTION = (
    "The simulation above just failed at runtime — the design compiled "
    "fine, but a $fatal/$error fired during simulation. Before any tool is "
    "available to you, write a short plan (2-4 bullet points): read the "
    "signal-value table around the failure time (when one is shown) and "
    "state which signal had the wrong value, what it should have been "
    "instead, and the exact RTL change you'll make to fix it. Do not write "
    "SystemVerilog code yet — just the plan, and keep signal names "
    "consistent with the original code."
)


def force_apply_instruction(file_path: str) -> str:
    # Last-resort structural fallback (see rtl_agent.py) for when the model
    # has already diagnosed a fix correctly but then repeatedly fails to
    # actually call a tool to apply it — confirmed directly to happen, not
    # hypothetical: one real run produced a correct root-cause diagnosis and
    # then skipped acting on it through 6 consecutive forced nudges. Ollama
    # has no tool_choice/forced-function-calling support (confirmed against
    # langchain_ollama directly — the parameter exists for OpenAI-API
    # compatibility but is explicitly ignored), so nothing can make the
    # model itself emit a tool call. This call has no tools bound at all —
    # the only thing left for the model to do is output the file content as
    # plain text, which the harness then writes on its own, so the action
    # happens regardless of whether the model "cooperates" with the tool
    # protocol.
    return (
        f"You correctly diagnosed the problem above but have not applied a "
        f"fix through any tool call — tools are not available in this "
        f"response. Output ONLY the complete, corrected content of "
        f"{file_path}, nothing else, between a single pair of triple "
        f"backticks. No explanation before or after. The harness will write "
        f"this content to {file_path} directly."
    )


def build_llm(config: AgentConfig) -> ChatOllama:
    return ChatOllama(
        model=config.model,
        num_ctx=config.num_ctx,
    )
