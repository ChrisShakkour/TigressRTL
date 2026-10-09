# TigressRTL

An agent that generates, writes, and compiles SystemVerilog RTL. It runs a
ReAct-style loop against a local LLM (via [Ollama](https://ollama.com)) with
tools to write, read, and edit files, and to compile them with
[Icarus Verilog](http://iverilog.icarus.com/) — so generated code is checked
against a real compiler, not just eyeballed.

## Requirements

- **Python 3.10+**
- **Ollama**, running locally, with a tool-capable model pulled — see
  [Installing Ollama](#installing-ollama) below.
- **Icarus Verilog**, for the `build_verilog` tool: `apt install iverilog`
  on Debian/Ubuntu, or `winget install Icarus.Verilog` on Windows (the
  Windows installer does not add itself to `PATH`; `src/tools.py` falls
  back to the default install location if `iverilog` isn't found on
  `PATH`).
- **Slang** (optional) — a second build/lint backend, selected via
  `verilog_build_tool: slang` in your config (default is `icarus`, so this
  isn't required unless you opt in). See
  [Installing Slang](#installing-slang) below and
  [Build tools](configs/README.md#build-tools) for what it catches that
  Icarus doesn't.

These are all system-level installs — a Python virtual environment (below)
only isolates the Python packages (`langchain-core`, `langchain-ollama`),
not Ollama or Icarus Verilog, so both still need to be set up on whatever
machine you're actually running on, local or remote.

## Setup

Works the same whether you're setting this up locally or on a freshly
cloned remote server:

```bash
git clone <this-repo-url>
cd TigressRTL

python -m venv .venv

# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install --upgrade pip
pip install -e .
```

## Installing Ollama

The agent talks to a local Ollama server for all model inference — install
it before running the agent.

**Windows:** download and run the installer from
[ollama.com/download/windows](https://ollama.com/download/windows) (or
`winget install Ollama.Ollama`). It puts `ollama` on `PATH` and runs as a
background service automatically — no separate `ollama serve` step needed.

**Linux:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

This is the official install script — it installs the `ollama` binary and
sets it up as a `systemd` service that starts automatically.

### Pulling a model

Once Ollama is installed and running, pull a tool-capable model:

```bash
ollama pull devstral          # default in configs/default.yaml — see Design notes for why
ollama pull devstral-small-2  # newer 24B model, larger 384K context window (--model devstral-small-2)
ollama pull llama3.2          # faster, less reliable fallback (--model llama3.2)
```

A much larger alternative is also available if you have the RAM/VRAM for it:

```bash
ollama pull devstral-2   # 123B flagship, 75GB — not verified locally, see configs/README.md
```

Run `ollama list` to confirm what's pulled locally.

## Installing Slang

[Slang](https://sv-lang.com) is a second SystemVerilog compiler/linter — the
`lint_verilog` tool in `src/tools.py` wraps it, used when a config sets
`verilog_build_tool: slang` (see [Build tools](configs/README.md#build-tools)).
The default is `icarus`, so this isn't required unless you opt in.

Slang has no installer — grab a prebuilt binary from
[GitHub releases](https://github.com/MikePopoloski/slang/releases)
(`slang-windows-x86_64.zip` or `slang-linux-x86_64.tar.gz`), extract it, and
put `slang`/`slang.exe` on `PATH`. `src/tools.py` falls back to
`C:\slang\slang.exe` on Windows if `slang` isn't found on `PATH`. No
official `winget`/`apt` package exists; `conda-forge` has `slang-verilog`
as an alternative. Building from source is documented at
[sv-lang.com/building.html](https://sv-lang.com/building.html).

## Run

```powershell
python src/rtl_agent.py                                    # uses configs/default.yaml
python src/rtl_agent.py --config configs/my-experiment.yaml # a different config file
python src/rtl_agent.py --model llama3.2                    # override just the model for one run
```

Settings (model, context window, build-retry cap, and more to come) are
loaded from a YAML config file — `configs/default.yaml` unless `--config`
points elsewhere. `--model` always wins over whatever the config says, so a
one-off override doesn't require creating a new file. See
[Config options](#config-options) below for the full list of fields.

Type a request (e.g. *"Write a SystemVerilog module for a 4-bit synchronous
up-counter with active-low reset and enable."*) and the agent will plan,
write the file under `src/generated/`, and compile it automatically. Type
`exit` or `quit` to stop.

While it runs, three prefixes tell you what's actually happening:

- `[Plan]` — the model's stated plan, produced before any tool is available
  to it.
- `[Action]` / `[Result]` — a tool call and its real return value.
- `[Auto-Build]` — the compilation result, run automatically after every
  write/edit.

Always trust `[Result]`/`[Auto-Build]` over the `Assistant:` text that
follows — see [Design notes](#design-notes) below.

## Config options

Settings (model, context window, build-retry cap), the full list of fields,
and a table of usable models with their maximum context length are
documented in [`configs/README.md`](configs/README.md).

## Benchmarking

Two scripts, kept separate since generation and validation are different
concerns run at different times: `test/run_benchmark.py` runs the agent
against the spec-to-rtl problems from
[NVlabs/verilog-eval](https://github.com/NVlabs/verilog-eval) and saves
whatever code it produced; `test/run_validation.py` takes that output and
actually checks it — compiling and simulating each generated module against
the dataset's own reference solution and testbench.

Clone the dataset anywhere (it's not vendored into this repo, same as
Ollama/Icarus Verilog):

```bash
git clone https://github.com/NVlabs/verilog-eval.git
```

Run it:

```bash
# One config, all 156 problems
python test/run_benchmark.py --dataset-dir verilog-eval/dataset_spec-to-rtl --configs configs/default.yaml

# Sweep multiple configs (each gets the full problem set)
python test/run_benchmark.py --dataset-dir verilog-eval/dataset_spec-to-rtl \
    --configs configs/default.yaml configs/devstral-small-2.yaml

# Quick smoke test — first 2 problems only
python test/run_benchmark.py --dataset-dir verilog-eval/dataset_spec-to-rtl --configs configs/default.yaml --limit 2
```

Each invocation runs every problem as a **fresh `rtl_agent.py` subprocess**
(its own context, no history carried over between problems) and writes into
a new timestamped directory under `benchmark_runs/` (gitignored):

```
benchmark_runs/2026-08-04_23-45-28/
  command.txt              # the exact CLI invocation, for reproducibility
  configs/default.yaml      # a copy of every --configs file used in this run
  default/                  # config-stem = Path(config).stem
    Prob001_zero/
      prompt.txt            # the single-lined prompt piped into rtl_agent.py
      transcript.log        # full agent output ([Plan]/[Action]/[Result]/[Auto-Build]/Assistant)
      status.json           # {problem, model, num_ctx, status, sv_file_count, duration_s,
                             #  input_tokens, output_tokens, total_tokens, ...}
      generated/             # whatever ended up in src/generated/ after this run
```

If a long sweep gets interrupted, resume it with
`--resume-dir benchmark_runs/<timestamp>`, which skips any (config, problem)
pair already marked `"completed"` and retries anything that timed out or
errored. Without `--resume-dir`, every invocation always starts a fresh
timestamped directory. See `--help` for `--problems` (filter by name/glob),
`--timeout` (per-problem subprocess timeout, none by default), and `--python`
(interpreter to launch `rtl_agent.py` with).

`rtl_agent.py` prints a `[Token Usage]` line on exit (accumulated across every
LLM call in that process, i.e. that one problem), which `run_benchmark.py`
parses into each problem's `status.json` and sums into the end-of-run
summary — `  default: {'completed': 103}  tokens: input=... output=... total=...`.
Skipped (already-completed) problems contribute their previously-recorded
tokens too, so the total reflects the whole run directory, not just what
this particular invocation freshly ran. Problems from before this feature
existed show `0` rather than crashing.

### Validating generated code

`test/run_validation.py` takes a `run_benchmark.py` output directory and, for
every problem, compiles its `generated/*.sv` files together with the
dataset's `<problem>_ref.sv` and `<problem>_test.sv`, runs the simulation,
and parses the testbench's own `Mismatches: N in M samples` summary:

```bash
python test/run_validation.py --run-dir benchmark_runs/2026-08-04_23-45-28 \
    --dataset-dir verilog-eval/dataset_spec-to-rtl
```

Writes `validation.json` (`status`, `mismatches`, `samples`, `duration_s`)
and `validation.log` (raw compile + simulation output) alongside each
problem's existing `transcript.log`. `status` is one of: `pass`, `fail`
(compiled and ran, but mismatched the reference), `compile_error`,
`sim_timeout` (either the subprocess itself or the testbench's own built-in
simulated-time cutoff), `no_generated_code`, or `missing_dataset_files`.
Already-validated problems are skipped on a re-run unless `--overwrite` is
passed. `--configs`/`--problems` filter which config-stems/problems to
validate, same as `run_benchmark.py`.

At the end, a categorized summary (pass rate, build failures, functional
failures with mismatch counts, other/inconclusive) is both printed and
written to `<run-dir>/validation_summary.json` — it reflects the true
current state of the whole run directory (including already-validated,
skipped problems), not just what changed in that particular invocation.

**This is the trustworthy pass/fail signal — not `run_benchmark.py`'s own
`transcript.log`.** `build_verilog` (the tool the agent itself calls) only
compiles one file at a time, so if the agent writes an extra file alongside
`TopModule.sv` (e.g. its own self-authored testbench), that file's
`build_verilog` call fails with a spurious "Unknown module type" error
(it can't see `TopModule`, which lives in a different file) even when
`TopModule.sv` itself is completely correct. Only `run_validation.py`'s real
multi-file compile against the actual dataset testbench reflects the truth.

## Architecture

- **`src/config.py`** / **`configs/default.yaml`** — `AgentConfig`, a small
  dataclass (`model`, `num_ctx`, `max_build_retries`, `verilog_build_tool`,
  `enable_simulation`, `max_sim_retries`, `sim_debug_window_cycles`) loaded
  from YAML via `load_config()`. The single source of truth for experiment
  settings — see [Run](#run) above.
- **`src/model.py`** — the one swappable model/provider point (`build_llm()`
  takes an `AgentConfig`), plus
  `build_system_prompt(build_tool_name, enable_simulation)` — a function
  rather than a constant, since the build-and-fix (and, when enabled,
  simulate-and-fix) instructions it produces need to name whichever tools
  are actually bound.
- **`src/tools.py`** — `BASE_TOOLS` (`write_file`, `read_file`,
  `edit_file_block`, `list_directory` — always available) plus
  `BUILD_TOOL_FUNCTIONS`/`BUILD_FAILURE_PREFIXES`, keyed by
  `"icarus"`/`"slang"`: `build_verilog` and its structural sibling
  `lint_verilog` (Slang-backed). `AgentConfig.verilog_build_tool` selects
  exactly one to bind to the model — see
  [Build tools](configs/README.md#build-tools). Also
  `make_run_simulation_tool(debug_window_cycles)`, a factory (not a bare
  `@tool`, since the VCD window size is config-driven but must stay out of
  the tool's LLM-visible schema) for `run_simulation` — compiles the whole
  project to a real runnable target with `iverilog -o` + `vvp` and renders a
  VCD-derived signal-value window around a failure via `src/vcd.py`. Only
  bound at all when `AgentConfig.enable_simulation` is true — see
  [Simulation](configs/README.md#simulation). All file-touching tools are
  sandboxed to `src/generated/` — a model-supplied path is untrusted input,
  normalized and checked so nothing can escape that directory.
  `run_simulation`'s own compile step additionally reads (never through any
  model-facing tool) from `SIM_EXTRAS_DIR` (`src/sim_extras/`), a sibling
  directory `_resolve_safe_path` structurally cannot reach — used by
  `test/run_benchmark.py` to supply a real verilog-eval testbench+reference
  pair without ever exposing the reference solution to the model; see
  [Simulation](configs/README.md#simulation) for how the VCD debug window
  filters out the reference module's own `_ref`-suffixed signals whenever
  this directory is involved, rather than suppressing the window outright.
- **`src/vcd.py`** — a small, dependency-free VCD (waveform dump) parser:
  `parse_vcd()` reads a `.vcd` file into per-signal `(time, value)` traces,
  `find_clock_period()` detects a `clk`/`clock` signal's cycle length, and
  `render_window()` renders every signal's value across N clock cycles
  around a given timestamp — the debug context `run_simulation` hands the
  model on a failure.
- **`src/rtl_agent.py`** — the interactive loop: resolves the active build
  tool from config once at startup, then a planning call with no tools
  bound (so the model is structurally unable to act before planning),
  then a ReAct tool-calling loop that auto-runs the active build tool
  after every write/edit and forces a bounded number of fix attempts
  (`config.max_build_retries`) if a build fails. When
  `config.enable_simulation` is true, a build that just passed also
  auto-chains `run_simulation` (the actual "only once the build is passing"
  gate — the call site is physically inside the build-succeeded branch, not
  left to the model to time correctly), with the same forced-retry
  discipline (`config.max_sim_retries`) for a runtime failure.
- **`test/run_benchmark.py`** — sweeps `rtl_agent.py` (one fresh subprocess
  per problem) over the verilog-eval dataset across one or more configs —
  see [Benchmarking](#benchmarking) above. For a config with
  `enable_simulation: true`, also clears and repopulates `SIM_EXTRAS_DIR`
  with that problem's real `_test.sv`/`_ref.sv` before each subprocess run,
  so `run_simulation` has a genuine testbench to exercise during a sweep
  instead of always skipping.
- **`test/run_validation.py`** — compiles and simulates each generated
  problem against the dataset's reference solution and testbench, recording
  a real pass/fail — see [Validating generated code](#validating-generated-code)
  above.

## Design notes

A few non-obvious things learned building this, worth knowing before
extending it:

- **Tool-calling reliability varies a lot by model**, independent of what
  `ollama show <model>` claims to support. Some models report `tools` as a
  capability but never actually populate `tool_calls` — they dump the call
  as raw JSON text instead. Others populate `tool_calls` but truncate or
  garble multi-line content. Verify empirically before trusting a model's
  declared capabilities.
- **Never trust the model's prose over a tool's actual return value.**
  Models have been observed describing a write or build that never
  happened, or fabricating a plausible-but-wrong description of a file's
  real contents — directly contradicting the tool result they'd just
  received. That's why every tool result is printed unconditionally via
  `[Result]`/`[Auto-Build]`, before the model gets a chance to say anything
  about it.
- **A prompt instruction to "plan before acting" isn't reliable enough on
  its own.** `rtl_agent.py` instead makes a first call with no tools bound
  at all, so planning happens before acting as a structural guarantee, not
  a hope.
- **Path handling needs normalization, not just a naive join.** Models
  reliably pass paths as if they need to name the sandbox directory
  themselves (e.g. `/generated/foo.sv` for "put it under the generated
  directory"), which breaks a naive `base / path` join. `_resolve_safe_path`
  in `src/tools.py` strips a redundant leading anchor/`generated` segment
  before resolving, while still rejecting genuine escape attempts like
  `../../etc/passwd`.
- **`build_verilog`'s single-file compile is a real blind spot, not just a
  simplification.** It only ever compiles the one file it's given. If the
  agent writes a second file that references the first (most commonly, a
  self-authored testbench instantiating `TopModule`), `build_verilog` on
  that second file reports a spurious "Unknown module type" failure — the
  referenced module is completely real, just defined in a file this
  particular compile invocation was never given. Confirmed directly: a
  `transcript.log` showing repeated `build_verilog` failures on
  `TopModule_tb.sv` turned out to have a perfectly correct `TopModule.sv`
  the whole time — `test/run_validation.py`'s real multi-file compile
  against the dataset's actual testbench is what caught this, not anything
  in the agent's own transcript.
- **Only one build tool is ever bound to the model at a time, never both.**
  `lint_verilog` (Slang) was built as a structural sibling of
  `build_verilog` (Icarus) specifically so `verilog_build_tool` could later
  swap between them with no redesign — same sandboxing, same timeout, same
  success/failure shape, just different wording ("Compilation failed" vs.
  "Lint failed", tracked in `BUILD_FAILURE_PREFIXES`) that the auto-build
  enforcement and retry logic in `rtl_agent.py` detect generically rather
  than hardcoding one tool's name. Binding both would just leave the model
  guessing which one to call. `-Weverything` alone already catches real
  issues Icarus misses entirely: it flagged a genuine width-mismatch
  (`arith-op-mismatch`) on a file Icarus compiled clean with no warning at
  all — worth trying `verilog_build_tool: slang` on a config you care about
  the strictness of.
- **Icarus's own runtime error text already carries a failure timestamp for
  free.** `$fatal`/`$error` both print a `Time: N` line as part of their
  standard message — confirmed directly (`$fatal` → nonzero exit + `FATAL:`
  text; `$error` → exit 0 but `ERROR:` text, simulation keeps running; a
  clean `$finish` → exit 0, neither string present) — so `run_simulation`
  can anchor its VCD debug window on that timestamp with a plain regex, with
  no testbench-specific convention required to report *when* something went
  wrong.
- **A VCD's raw vector value is genuinely ambiguous without knowing it came
  from a `b<bits>` dump.** A 4-bit `count` signal at binary `11` renders as
  the bare digit string `"11"` — indistinguishable from decimal eleven
  unless you already know it's binary. Caught directly while building the
  signal-value window: a first version rendered `count=10`/`count=11` for
  what were actually decimal 2 and 3. `src/vcd.py` tags every value with
  whether it came from a vector or scalar dump and converts an all-0/1
  vector to decimal (`0b`-prefixed binary only when it contains `x`/`z`
  bits, which can't be converted) — an easy trap for any future VCD-reading
  code to fall back into.
- **This machine's `iverilog` had a real, silent regression.** conda-forge's
  v13.0 build refuses to elaborate a `$dumpvars(...)` call that
  forward-references a wire declared later in the same module — a pattern
  every verilog-eval testbench uses, so it broke elaboration of the entire
  dataset with no RTL-side cause at all. Confirmed by replaying a
  historical, previously-successful `run_validation.py` result
  byte-for-byte against the same generated code and the same `iverilog`
  path: it now fails to compile. `iverilog=12.0` (also on conda-forge) does
  not have this bug. Worth an empirical recheck if `test/run_validation.py`
  ever reports suspiciously uniform `compile_error`/`unknown` results across
  a whole run — that's what this regression actually looked like, not a
  problem with the generated RTL.
- **"Hide the file" isn't enough to keep a benchmark blind — the VCD debug
  window leaks just as much, but suppressing it outright breaks debugging
  entirely.** Making `run_simulation` useful during `run_benchmark.py`
  sweeps means the testbench needs `RefModule` to even elaborate, but
  verilog-eval's testbenches dump the reference's own output as a plain
  top-level `<port>_ref` wire — not nested under the reference instance's
  own scope, so excluding by instance path (the first approach tried)
  doesn't catch it. The first fix suppressed the whole VCD window whenever
  `SIM_EXTRAS_DIR` had any files in it, which stopped the leak but caused
  real, observed harm: a live benchmark transcript showed the model's own
  `[Sim Fix Plan]` correctly asking to "read the signal-value table,"
  finding none, and burning its entire `max_sim_retries` budget confused
  about a missing testbench instead of fixing an actual NAND-vs-NOR logic
  bug (`assign out = ~a | ~b` for a *NOR* gate). `run_simulation` now
  excludes only the `_ref`-suffixed signals (confirmed this convention
  holds across all 156 dataset problems) rather than the whole window — the
  model still sees every stimulus input and its own DUT output around the
  failure, enough to re-derive the right answer from the spec, just never
  the reference's own computed value. Worth remembering before extending
  the debug window further: hiding a *file* and hiding a *value derived
  from that file* are different problems, and VCD dumps routinely do the
  second without the first.
