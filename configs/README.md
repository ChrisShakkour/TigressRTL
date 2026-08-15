# Config options

`configs/default.yaml` (or any file passed via `--config`) accepts the
fields defined by `AgentConfig` in `src/config.py`:

| Field | Default | What it controls |
|---|---|---|
| `model` | `devstral` | Which pulled Ollama model to use. |
| `num_ctx` | `8192` | Context window size, in tokens, given to Ollama. Can be set anywhere up to the chosen model's maximum context length (see table below) — larger values use more RAM/VRAM and run slower. |
| `max_build_retries` | `3` | How many times the agent is forced to retry after verification isn't satisfied (a build/lint failure it didn't actually fix, or — with icarus — a testbench it never simulated), before giving up for that turn. Any non-negative integer. See [Verification](../README.md#verification) in the root README. |
| `verilog_build_tool` | `icarus` | Which build/lint backend the agent's build tool uses: `icarus` (Icarus Verilog, a real compiler — `build_verilog`) or `slang` (sv-lang.com, a stricter linter — `lint_verilog`). Exactly one is bound to the model at a time, never both. An invalid value raises a clear error immediately. See [Build tools](#build-tools) below. |

To run an experiment with different settings, copy `default.yaml`, edit the
copy, and pass it via `--config` (e.g. `--config configs/my-experiment.yaml`).
`--model` always overrides just the model field for a one-off run without
needing a new file. A typo'd key in your YAML raises a clear error instead
of silently falling back to the wrong default — see `load_config()` in
`../src/config.py`.

## Build tools

| `verilog_build_tool` | Real tool | Notes |
|---|---|---|
| `icarus` (default) | Icarus Verilog | `build_verilog` compiles to `-t null` — elaboration only, no simulation actually run. Silent on success. Also the only setting that binds `simulate_verilog`, which compiles to a real runnable target and executes it with `vvp` — see [Verification](../README.md#verification) in the root README. |
| `slang` | Slang (sv-lang.com) | `lint_verilog` runs it with `-Weverything`, which catches real issues Icarus misses entirely — e.g. it flagged a genuine width mismatch (`-Warith-op-mismatch`) on a file Icarus compiled clean with zero warnings. Always prints a `Build succeeded/failed: N errors, M warnings` summary, even on success. Slang has no simulator, so `simulate_verilog` is never bound under this setting. |

Both share the same sandboxing, 30s timeout, and error handling in
`src/tools.py` — switching `verilog_build_tool` doesn't change anything else
about how the agent runs, only which compiler backs the build-and-fix loop
and what it's strict about.

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
