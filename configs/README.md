# Config options

`configs/default.yaml` (or any file passed via `--config`) accepts the
fields defined by `AgentConfig` in `src/config.py`:

| Field | Default | What it controls |
|---|---|---|
| `model` | `devstral` | Which pulled Ollama model to use. |
| `num_ctx` | `8192` | Context window size, in tokens, given to Ollama. Can be set anywhere up to the chosen model's maximum context length (see table below) — larger values use more RAM/VRAM and run slower. |
| `max_build_retries` | `3` | How many times the agent is forced to retry after a build failure it didn't actually fix, before giving up for that turn. Any non-negative integer. |
| `verilog_build_tool` | `icarus` | Which build/lint backend the agent's build tool uses: `icarus` (Icarus Verilog, a real compiler — `build_verilog`) or `slang` (sv-lang.com, a stricter linter — `lint_verilog`). Exactly one is bound to the model at a time, never both. An invalid value raises a clear error immediately. See [Build tools](#build-tools) below. |
| `enable_simulation` | `false` | Whether `run_simulation` (Icarus `iverilog`+`vvp`) is bound to the model at all and auto-chained after a build that just passed. `false` reproduces the pre-simulation agent's exact behavior with zero other changes — see [Simulation](#simulation) below. |
| `max_sim_retries` | `3` | Same role as `max_build_retries`, but for a simulation failure ($fatal/$error at runtime) instead of a compile error. Unused when `enable_simulation` is `false`. |
| `sim_debug_window_cycles` | `3` | How many clock cycles of context, before and after, `run_simulation` shows in its VCD-derived signal-value table around a failure. Unused when `enable_simulation` is `false`. |

To run an experiment with different settings, copy `default.yaml`, edit the
copy, and pass it via `--config` (e.g. `--config configs/my-experiment.yaml`).
`--model` always overrides just the model field for a one-off run without
needing a new file. A typo'd key in your YAML raises a clear error instead
of silently falling back to the wrong default — see `load_config()` in
`../src/config.py`.

## Build tools

| `verilog_build_tool` | Real tool | Notes |
|---|---|---|
| `icarus` (default) | Icarus Verilog | Compiles to a real simulation target (`-t null` here — elaboration only, no simulation actually run). Silent on success. |
| `slang` | Slang (sv-lang.com) | `lint_verilog` runs it with `-Weverything`, which catches real issues Icarus misses entirely — e.g. it flagged a genuine width mismatch (`-Warith-op-mismatch`) on a file Icarus compiled clean with zero warnings. Always prints a `Build succeeded/failed: N errors, M warnings` summary, even on success. |

Both share the same sandboxing, 30s timeout, and error handling in
`src/tools.py` — switching `verilog_build_tool` doesn't change anything else
about how the agent runs, only which compiler backs the build-and-fix loop
and what it's strict about.

## Simulation

`enable_simulation: true` adds a second, deeper check on top of the
build-and-fix loop above: once a build passes, and if the project already
contains a self-checking testbench (any `.sv` file that calls `$finish` —
nothing is required to be named a certain way), `run_simulation` compiles
the project to a real runnable target with `iverilog -o` and executes it
with `vvp`, the same mechanics `test/run_validation.py` already uses offline
against the verilog-eval dataset. A runtime `$fatal`/`$error` — Icarus's own
wording, not something this project invents — is treated as a failure and
triggers the same forced-retry discipline `max_build_retries` already
applies to compile errors, bounded by `max_sim_retries`.

On a failure, if the testbench dumped a VCD (`$dumpfile(...)`/`$dumpvars(...)`)
and Icarus's own `Time: N` line could be parsed out of the failure message,
`run_simulation` also renders every dumped signal's value across
`sim_debug_window_cycles` clock cycles on either side of that timestamp
(`src/vcd.py`) — concrete input/output values around the failure, not just
"it failed". Both are best-effort: no VCD, or no clock signal to size a
cycle from, just means a shorter or absent table, never an error.

Writing a testbench isn't required or auto-generated yet — if none exists,
`run_simulation` reports `Simulation skipped: ...` and nothing about the
retry loop is affected.

### Simulation during benchmark sweeps

`test/run_benchmark.py`, for any config with `enable_simulation: true`,
copies each problem's real `_test.sv`/`_ref.sv` from `--dataset-dir` into
`SIM_EXTRAS_DIR` (`src/sim_extras/`, cleared and repopulated per problem)
before launching that problem's `rtl_agent.py` subprocess — otherwise
`run_simulation` would always report `Simulation skipped: ...` during a
sweep, since the model itself never writes a testbench and the dataset's
own testbench isn't in `src/generated/` by default.

`SIM_EXTRAS_DIR` is deliberately outside the model's sandbox — `write_file`/
`read_file`/`edit_file_block`/`list_directory` are all rooted at
`src/generated/` and structurally cannot reach it, so the model can never
list or read the reference solution `_test.sv` needs to elaborate.

That alone isn't enough, though: a verilog-eval testbench also dumps the
reference module's own output to the same VCD used for debug context, as a
plain top-level wire named `<port>_ref` sitting right next to the DUT's
`<port>_dut` — hiding the source file doesn't stop the signal-value window
from showing the reference's *computed output values* directly. An earlier
version of this suppressed the whole VCD window whenever `SIM_EXTRAS_DIR`
had any files in it, which fixed the leak but also meant the model debugging
a benchmark problem got no concrete values at all — confirmed directly to
cause real harm: one transcript showed the model's own fix-plan correctly
asking to "read the signal-value table," finding none, and burning its
entire retry budget confused about a missing testbench instead of fixing an
actual logic bug. `run_simulation` instead excludes just the `_ref`-suffixed
signals from the rendered window (confirmed this convention holds across
all 156 problems in the dataset) — the model still sees every stimulus
input and its own DUT output around the failure, enough to re-derive the
right answer from the original spec, just never the reference's own
computed value.

`enable_simulation` defaults to `false` specifically so it's easy to A/B:
run the same problem set through `configs/default.yaml` (unchanged
behavior) and a copy with `enable_simulation: true`, then compare with
`test/run_benchmark.py --configs ...` and `test/run_validation.py`.

## Models and their maximum context length

Verified locally via `ollama show <model>`:

| Model | Max context | Tool-calling | Notes |
|---|---|---|---|
| `devstral` (default) | 131072 (128K) | Reliable | See the root [README's Design notes](../README.md#design-notes) for why this is the default. |
| `devstral-small-2` | 393216 (384K) | Reliable | Newer 24B model, larger context than `devstral`. Real structured `tool_calls`, correctly self-corrected a build failure in testing. Also has `vision` capability (unused by this agent) and a lower built-in default temperature (0.15) than `devstral`. |
| `llama3.2` | 131072 (128K) | Unreliable | Garbled/truncated tool-call arguments, fabricated results — see Design notes. |
| `qwen2.5-coder` | 32768 (32K) | Claims `tools`, doesn't use them | Dumps the call as plain-text JSON instead of populating `tool_calls`. |
| `llama2` | 4096 (4K) | None | No `tools` capability at all — Ollama rejects any request with tools bound. Not usable with this agent regardless of `num_ctx`. |

Newer alternative from the root README's
[Installing Ollama](../README.md#installing-ollama) section, per
[ollama.com](https://ollama.com/library) (not pulled/verified locally — much
larger, 75GB):

| Model | Max context (per ollama.com) |
|---|---|
| `devstral-2` | ~256K |

Whatever model you choose, `num_ctx` in your config must not exceed its
maximum context length above — Ollama will error or silently clamp it
otherwise.
