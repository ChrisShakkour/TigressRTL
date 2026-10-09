from dataclasses import dataclass
from pathlib import Path

import yaml

# Repo root, not src/ — configs/ sits alongside src/, not inside it.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


@dataclass
class AgentConfig:
    # qwen2.5-coder *claims* tool support (`ollama show qwen2.5-coder` lists
    # completion, tools, insert) but empirically does not reliably use it:
    # tested directly against both the raw `ollama` library and
    # langchain_ollama, it dumps the tool call as a raw JSON string in the
    # response's plain text content instead of populating the structured
    # tool_calls field — so response.tool_calls comes back empty/None and
    # nothing actually executes. Declared capability metadata isn't proof
    # of working behavior — verify empirically before trusting it.
    #
    # llama3.2 reliably produces structured tool_calls, but empirically
    # fails in other ways: garbled/truncated multi-line content in
    # tool-call arguments, narrating fake pseudo-tool-calls as plain text
    # instead of invoking them, and confidently hallucinating tool results
    # that contradict what the tool actually returned in the same response.
    #
    # devstral (Mistral's model purpose-built for agentic coding, via the
    # OpenHands scaffold) is the first model tested here with none of those
    # problems in a full end-to-end run: real structured tool_calls with
    # complete, untruncated, valid SystemVerilog content; build succeeded
    # on the first try; and its final answer was a byte-for-byte accurate
    # description of what was actually written to disk. It's ~14GB and
    # noticeably slower per turn on limited-GPU hardware (mostly CPU
    # inference), but the reliability difference is large enough to make
    # it the default. --model llama3.2 still works if speed matters more
    # than reliability for a given session.
    model: str = "devstral"

    # Sized up from a typical 4096 default — generated SystemVerilog
    # modules tend to run longer than plain prose/chat content that value
    # is tuned for.
    num_ctx: int = 8192

    # Bounded retry cap for the build-failure nudge in rtl_agent.py —
    # without a cap, a model that can never actually fix a given error
    # would loop forever.
    max_build_retries: int = 3

    # Which build/lint backend the agent's build tool uses — "icarus"
    # (Icarus Verilog, src/tools.py's build_verilog) or "slang"
    # (sv-lang.com, build_verilog's structural sibling lint_verilog).
    # Exactly one is bound to the model at a time (see rtl_agent.py);
    # they're never both exposed together.
    verilog_build_tool: str = "icarus"

    # Whether run_simulation (src/tools.py) is bound to the model at all and
    # auto-chained after a successful build (rtl_agent.py). Defaults to
    # False so every existing config file reproduces today's exact
    # behavior — build-only, no simulation — with zero changes; set True in
    # a separate config to A/B this feature against that unchanged baseline.
    enable_simulation: bool = False

    # Bounded retry cap for the simulation-failure nudge in rtl_agent.py,
    # same role as max_build_retries but for a runtime (not compile)
    # failure. Unused when enable_simulation is False.
    max_sim_retries: int = 3

    # How many clock cycles of context (before and after) to show around a
    # simulation failure's timestamp, when run_simulation renders a
    # VCD-derived signal-value window (src/vcd.py's render_window). Unused
    # when enable_simulation is False.
    sim_debug_window_cycles: int = 3

    def __post_init__(self):
        # Not imported from tools.py's BUILD_TOOL_FUNCTIONS keys on purpose —
        # that would pull langchain_core/subprocess into config.py just for
        # a two-string validation check. A typo'd value (e.g. "islang")
        # should fail loudly here, same discipline as load_config()'s
        # unknown-key check below.
        if self.verilog_build_tool not in ("icarus", "slang"):
            raise ValueError(
                f"verilog_build_tool must be 'icarus' or 'slang', got {self.verilog_build_tool!r}"
            )


def load_config(path: str | Path | None) -> AgentConfig:
    # `path=None` means "use the built-in defaults" — kept as a parameter
    # (rather than always reading DEFAULT_CONFIG_PATH directly) so callers
    # can skip the config file entirely.
    if path is None:
        return AgentConfig()

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    try:
        return AgentConfig(**data)
    except TypeError as e:
        # Passing an unknown key raises TypeError from the dataclass
        # constructor itself — a typo'd key (e.g. "num_ctx_" instead of
        # "num_ctx") fails loudly here instead of silently falling back to
        # a default the user didn't intend.
        raise ValueError(f"Invalid config file {path}: {e}") from e
