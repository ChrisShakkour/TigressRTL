from langchain_ollama import ChatOllama

from config import AgentConfig

# The one swappable model/provider point in this project — everything else
# (tools.py, rtl_agent.py) talks to `build_llm()`'s return value, not to
# ChatOllama directly, so switching providers later only means editing here.
# Model choice and context window live in AgentConfig (see config.py) rather
# than as constants here, so they're driven by the YAML config instead of
# requiring a code edit to change.

def build_system_prompt(build_tool_name: str, can_simulate: bool = False) -> str:
    # A function rather than a plain constant so the build/fix cycle it
    # describes always names whichever tool is actually bound
    # (config.verilog_build_tool in config.py) — build_verilog or
    # lint_verilog — instead of hardcoding one. can_simulate is True only
    # when icarus is active (see rtl_agent.py) — Slang has no simulator, so
    # there's nothing to instruct the model to do with simulate_verilog when
    # it isn't even bound.
    prompt = (
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
    )
    if can_simulate:
        prompt += (
            " If you write a testbench for this design (a second file that "
            "instantiates it and checks its behavior), you must also call "
            "simulate_verilog on the whole project and read its real output "
            "before giving your final answer — a clean compile only proves "
            "the code elaborates, not that the design behaves correctly. "
            "simulate_verilog does not tell you pass or fail by itself; you "
            "must read the testbench's own printed output and judge "
            "correctness from that, the same way you already judge whether "
            "compiler diagnostics are actually fixed."
        )
    return prompt

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
# a build/lint/simulate attempt freshly fails — same structural guarantee as
# PLANNING_INSTRUCTION (no tools bound for this call), so the model can't
# skip straight to another guessed edit without first diagnosing the
# specific error in front of it. Deliberately worded to cover all three —
# src/verify.py's VerificationState sets fix_plan_pending from either a
# failed build/lint or a failed simulate, so this single instruction has to
# make sense for whichever one actually just happened.
FIX_PLAN_INSTRUCTION = (
    "The build, lint, or simulation attempt above just failed. Before any "
    "tool is available to you, write a short plan (2-4 bullet points): what "
    "specifically the error or output above says is wrong, and the exact "
    "change you'll make to fix it. Do not write SystemVerilog code yet — "
    "just the plan, make sure to keep the signal names consistent with the original code"
)


def build_llm(config: AgentConfig) -> ChatOllama:
    return ChatOllama(
        model=config.model,
        num_ctx=config.num_ctx,
    )
