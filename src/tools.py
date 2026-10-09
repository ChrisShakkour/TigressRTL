import os
import re
import shutil
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from vcd import parse_vcd, render_window

# Where the agent's RTL project lives. Kept separate from the script's own
# directory so it can be freely cleared/gitignored without touching the
# actual agent code.
GENERATED_DIR = os.path.join(os.path.dirname(__file__), "generated")

# A second, model-invisible project directory — deliberately NOT a
# subdirectory of GENERATED_DIR, since _resolve_safe_path roots every
# model-facing tool (write_file/read_file/edit_file_block/list_directory)
# at GENERATED_DIR and rejects any escape from it; a file living here is
# therefore structurally unreachable by those tools, not just hidden by
# convention. Only run_simulation's own compile step reads from it (see
# _sim_extra_sv_files) — used by test/run_benchmark.py to supply a
# verilog-eval problem's real _test.sv/_ref.sv so run_simulation has
# something to simulate against without exposing the reference solution
# (source *or* its computed output values — see run_simulation's own
# comment on why the VCD debug window is suppressed whenever this
# directory is involved) to the model being benchmarked.
SIM_EXTRAS_DIR = os.path.join(os.path.dirname(__file__), "sim_extras")

# Icarus Verilog's installer (unlike Ollama's) does not add itself to PATH,
# so a bare "iverilog" call would fail even in a fresh terminal. Check PATH
# first (covers Linux/macOS package managers, or a user who added it
# manually), then fall back to the default Windows install location.
_IVERILOG_FALLBACK = r"C:\iverilog\bin\iverilog.exe"
IVERILOG_PATH = shutil.which("iverilog") or (
    _IVERILOG_FALLBACK if os.path.exists(_IVERILOG_FALLBACK) else "iverilog"
)

# Slang (https://sv-lang.com) ships as a prebuilt binary with no installer —
# same PATH-then-fallback lookup as Icarus above, since there's no
# guarantee it ended up on PATH after extracting a release zip/tar.gz.
_SLANG_FALLBACK = r"C:\slang\slang.exe"
SLANG_PATH = shutil.which("slang") or (
    _SLANG_FALLBACK if os.path.exists(_SLANG_FALLBACK) else "slang"
)

# vvp (Icarus's simulation runtime) ships alongside iverilog itself — same
# PATH-then-fallback lookup, same installer caveat.
_VVP_FALLBACK = r"C:\iverilog\bin\vvp.exe"
VVP_PATH = shutil.which("vvp") or (
    _VVP_FALLBACK if os.path.exists(_VVP_FALLBACK) else "vvp"
)

# Icarus's own $fatal/$error runtime messages always carry this — e.g.
# "FATAL: foo.sv:4: something broke\n       Time: 1400  Scope: foo_tb" —
# giving run_simulation a failure timestamp for free, with no
# testbench-specific convention required to anchor the VCD debug window on.
_SIM_TIME_RE = re.compile(r"Time:\s*(\d+)")

# Same pattern test/run_validation.py's own MISMATCH_RE matches against
# verilog-eval's testbenches. Confirmed directly this convention is
# necessary, not optional: those testbenches never call $error/$fatal at
# all — they only ever $display a "Mismatches: N in M samples" summary from
# a `final` block, so the runtime-error heuristic alone silently reported
# every verilog-eval problem as a pass regardless of whether TopModule was
# actually right, until this was added.
_SIM_MISMATCH_RE = re.compile(r"Mismatches:\s*(\d+)\s+in\s+(\d+)\s+samples")

# verilog-eval's own Hint text (not an Icarus built-in, unlike _SIM_TIME_RE's
# "Time: N" -- this is specific to the "Mismatches:" convention above) is
# the only place a timestamp appears for a pure functional-mismatch failure,
# since those never call $error/$fatal at all, so _SIM_TIME_RE never matches
# them. Without this, a verilog-eval functional mismatch would get no VCD
# debug window at all, regardless of extra_files filtering.
_SIM_MISMATCH_TIME_RE = re.compile(r"first mismatch occurred at time\s+(\d+)", re.IGNORECASE)

# verilog-eval's testbenches dump the hidden reference module's own output
# through a plain top-level wire named "<port>_ref" sitting right next to
# the DUT's "<port>_dut" -- not nested under the reference instance's own
# scope, so excluding by instance path (the first approach tried) wouldn't
# catch it. Confirmed directly: every one of the 156 test files in the
# dataset follows this exact "_ref"/"_dut" suffix convention, so this is a
# reliable (if dataset-specific) filter, not a generic Verilog-parsing one.
_HIDDEN_REFERENCE_SUFFIXES = ("_ref",)


# Matches the start of one diagnostic from either tool: slang's
# "file:line:col: error: msg" / "...warning: msg", or icarus's
# "file:line: error: msg" / bare "file:line: syntax error" (icarus's
# generic syntax-error line, always followed by a more specific "error:"
# line at the same location — so one real icarus error is two matches
# here, not one; MAX_DIAG_BLOCKS is set to 6 rather than 3 to compensate,
# giving ~3 full icarus errors or 6 full slang diagnostics). Deliberately
# does not match "note:" — a note is auxiliary info tied to the block
# before it (e.g. "previous definition here"), so it stays folded into
# that block instead of eating one of the kept slots.
_DIAG_START_RE = re.compile(r"^\S+:\d+(?::\d+)?:\s*(?:(error|warning)\b|syntax error\b)", re.MULTILINE)
_DIAG_IS_ERROR_RE = re.compile(r"^\S+:\d+(?::\d+)?:\s*(error\b|syntax error\b)")
MAX_DIAG_BLOCKS = 6


def _truncate_diagnostics(stderr_log: str, max_blocks: int = MAX_DIAG_BLOCKS) -> str:
    # Both tools write every actual diagnostic to stderr (slang's stdout is
    # just a short fixed-size summary; icarus's stdout is empty), so
    # truncating stderr alone — before it's concatenated with stdout — caps
    # runaway logs (e.g. -Weverything's dozens of warnings) without
    # touching the cheap summary text. Diagnostics are multi-line for slang
    # (message + source snippet + caret) and single-line for icarus, so
    # this splits on _DIAG_START_RE rather than raw line count to avoid
    # slicing a block in half.
    starts = [m.start() for m in _DIAG_START_RE.finditer(stderr_log)]
    if len(starts) <= max_blocks:
        return stderr_log
    bounds = starts + [len(stderr_log)]
    blocks = [stderr_log[bounds[i]:bounds[i + 1]] for i in range(len(starts))]
    kept, omitted = blocks[:max_blocks], blocks[max_blocks:]
    n_err = sum(1 for b in omitted if _DIAG_IS_ERROR_RE.match(b))
    n_warn = len(omitted) - n_err
    return "".join(kept) + f"... ({n_err} more error(s), {n_warn} more warning(s) omitted) ...\n"


def _project_sv_files() -> list[str]:
    # Every design file currently in the sandbox, not just the one the model
    # named — build_verilog/lint_verilog compile the whole project together
    # so a self-authored testbench in a second file that instantiates
    # TopModule resolves correctly instead of failing with a spurious
    # "unknown module" error (each run_benchmark.py problem gets its own
    # cleared GENERATED_DIR, so this never pulls in another problem's
    # files). .svh headers are excluded — they're meant to be `included,
    # not compiled as standalone top-level units.
    base = Path(GENERATED_DIR).resolve()
    return sorted(str(p) for p in base.rglob("*.sv"))


def _sim_extra_sv_files() -> list[str]:
    # Mirrors _project_sv_files(), rooted at SIM_EXTRAS_DIR instead —
    # returns [] whenever nothing has populated that directory (the normal
    # case for interactive use; only test/run_benchmark.py ever writes here).
    base = Path(SIM_EXTRAS_DIR)
    if not base.exists():
        return []
    return sorted(str(p) for p in base.resolve().rglob("*.sv"))


def _resolve_safe_path(path: str) -> Path:
    # Shared by every tool below. Resolves a model-supplied path against
    # GENERATED_DIR and rejects anything that would escape it (e.g. "../..",
    # or an absolute path) — the model's path arguments are untrusted input,
    # normalized and checked so nothing can escape the sandbox, while still
    # allowing safe subdirectories (needed for list_directory to browse a
    # project with packages/testbenches).
    os.makedirs(GENERATED_DIR, exist_ok=True)
    base = Path(GENERATED_DIR).resolve()

    # Models very predictably pass a path as if they need to name
    # GENERATED_DIR themselves — e.g. "/generated/foo.sv" — because a
    # request like "put it under the generated directory" reads naturally
    # as "include 'generated' in the path". Two things go wrong if we don't
    # correct for this: a leading "/" makes pathlib's `base / path` discard
    # `base` entirely (joining with an absolute path replaces the whole
    # thing, resolving to the filesystem root instead of our sandbox), and
    # even a relative "generated/foo.sv" would nest an extra generated/
    # subfolder inside GENERATED_DIR. So: drop any absolute anchor/drive,
    # and drop one leading "generated" path segment if present, before
    # resolving — since GENERATED_DIR already *is* that directory.
    parts = [p for p in Path(path).parts if p != Path(path).anchor]
    if parts and parts[0] == "generated":
        parts = parts[1:]

    candidate = base.joinpath(*parts).resolve() if parts else base
    if not candidate.is_relative_to(base):
        raise ValueError(f"path '{path}' escapes the generated/ directory")
    return candidate


@tool
def write_file(file_path: str, content: str) -> str:
    """Atomically create or overwrite a SystemVerilog (.sv/.svh) file with complete content."""
    try:
        target = _resolve_safe_path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file in the same directory first, then rename over
        # the real target. os.replace() is atomic on both POSIX and Windows,
        # so a crash or interrupted write can never leave a half-written or
        # corrupted .sv file at the target path — the file either has its
        # old complete contents or its new complete contents, never a mix.
        tmp_path = target.parent / (target.name + ".tmp")
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, target)
        rel = target.relative_to(Path(GENERATED_DIR).resolve())
        return f"Wrote {len(content)} characters to {rel}"
    except (OSError, ValueError) as e:
        return f"Failed to write {file_path}: {e}"


@tool
def read_file(file_path: str) -> str:
    """Read the entire contents of a file so the agent can review existing RTL or testbenches."""
    try:
        target = _resolve_safe_path(file_path)
        return target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except (OSError, ValueError) as e:
        return f"Failed to read {file_path}: {e}"


@tool
def edit_file_block(file_path: str, target_string: str, replacement_string: str) -> str:
    """Replace one exact block of code in a file with new code, without rewriting the whole file — use for small, targeted fixes."""
    try:
        target = _resolve_safe_path(file_path)
        text = target.read_text(encoding="utf-8")
        count = text.count(target_string)
        # Require exactly one match: zero means nothing to edit, more than
        # one means target_string is ambiguous about which occurrence to
        # change — both are reported back instead of guessing.
        if count == 0:
            return f"target_string not found in {file_path} — no changes made."
        if count > 1:
            return (
                f"target_string appears {count} times in {file_path} — "
                "make it more specific so exactly one occurrence matches."
            )
        new_text = text.replace(target_string, replacement_string, 1)
        tmp_path = target.parent / (target.name + ".tmp")
        tmp_path.write_text(new_text, encoding="utf-8")
        os.replace(tmp_path, target)
        return f"Replaced 1 occurrence in {file_path}."
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except (OSError, ValueError) as e:
        return f"Failed to edit {file_path}: {e}"


@tool
def build_verilog(file_path: str) -> str:
    """Compile a SystemVerilog file — together with every other .sv file in the project — with Icarus Verilog and return the real compilation log (errors/warnings), without running a simulation."""
    try:
        target = _resolve_safe_path(file_path)
        if not target.exists():
            return f"File not found: {file_path}"

        # -g2012 enables SystemVerilog-2012 syntax support (iverilog defaults
        # to plain Verilog otherwise). -t null elaborates the design and
        # reports errors/warnings without generating a runnable simulation
        # output — a compile/syntax check, not a simulation run. All project
        # .sv files are passed together (not just `target`) so a second file
        # that references the first — most commonly a self-authored
        # testbench instantiating TopModule — elaborates correctly instead
        # of a spurious "unknown module" error.
        result = subprocess.run(
            [IVERILOG_PATH, "-g2012", "-t", "null", *_project_sv_files()],
            capture_output=True, text=True, timeout=30,
        )
        log = (result.stdout + _truncate_diagnostics(result.stderr)).strip()

        if result.returncode == 0:
            return log or "Compiled successfully — no errors or warnings."
        return f"Compilation failed (exit code {result.returncode}):\n{log}"

    except FileNotFoundError:
        # Raised if IVERILOG_PATH itself can't be executed at all (not just
        # a compile error in the .sv file) — distinct from the two errors
        # above, which mean iverilog ran fine but found a problem in the code.
        return (
            "iverilog is not installed or could not be found. Install it via "
            "`winget install Icarus.Verilog` (Windows) or your package "
            "manager, and note the installer may not add it to PATH."
        )
    except subprocess.TimeoutExpired:
        return f"Compilation of {file_path} timed out after 30s."
    except (OSError, ValueError) as e:
        return f"Failed to compile {file_path}: {e}"


@tool
def lint_verilog(file_path: str) -> str:
    """Compile a SystemVerilog file — together with every other .sv file in the project — with Slang and return the real build/lint log (errors/warnings)."""
    try:
        target = _resolve_safe_path(file_path)
        if not target.exists():
            return f"File not found: {file_path}"

        # -Weverything turns on every warning class, not just the default
        # subset — this is what makes it a genuine lint pass rather than
        # just a compile check. No SV-version flag is needed (unlike
        # Icarus's -g2012): slang parses modern SystemVerilog by default.
        # -Wno-newline-eof suppresses a cosmetic-only warning (missing
        # trailing newline) that write_file's model-supplied content
        # triggers constantly and that carries no signal about RTL
        # correctness. All project .sv files are passed together (not just
        # `target`) so a second file that references the first — most
        # commonly a self-authored testbench instantiating TopModule —
        # resolves correctly instead of a spurious "unknown module" error.
        result = subprocess.run(
            [SLANG_PATH, "-Weverything", "-Wno-newline-eof", *_project_sv_files()],
            capture_output=True, text=True, timeout=30,
        )
        log = (result.stdout + _truncate_diagnostics(result.stderr)).strip()

        # Unlike Icarus, slang always prints a "Build succeeded: N errors,
        # M warnings" summary line even on success, so `log` is virtually
        # never empty here — that's fine, it's more informative than
        # build_verilog's silent-on-success behavior (it surfaces warning
        # counts even when the build passes).
        if result.returncode == 0:
            return log or "Linted successfully — no errors or warnings."
        return f"Lint failed (exit code {result.returncode}):\n{log}"

    except FileNotFoundError:
        # Raised if SLANG_PATH itself can't be executed at all (not just a
        # lint error in the .sv file) — slang has no installer; grab a
        # prebuilt release binary and put it on PATH.
        return (
            "slang is not installed or could not be found. Download a prebuilt "
            "release from https://github.com/MikePopoloski/slang/releases and "
            "put slang(.exe) on PATH, or build from source per "
            "https://sv-lang.com/building.html."
        )
    except subprocess.TimeoutExpired:
        return f"Lint of {file_path} timed out after 30s."
    except (OSError, ValueError) as e:
        return f"Failed to lint {file_path}: {e}"


SIM_FAILURE_PREFIX = "Simulation failed"


def _project_has_testbench() -> bool:
    # Synthesizable RTL never legitimately calls $finish (it's a
    # simulation-only system task) — so any project .sv file that does is
    # reliably "a testbench exists", with no naming convention required.
    # Checks SIM_EXTRAS_DIR too — a benchmark-supplied testbench living
    # there counts just as much as one the model wrote itself.
    return any(
        "$finish" in Path(f).read_text(encoding="utf-8")
        for f in [*_project_sv_files(), *_sim_extra_sv_files()]
    )


def make_run_simulation_tool(debug_window_cycles: int):
    # A factory rather than a bare @tool function because the rendered VCD
    # window's size is config-driven (AgentConfig.sim_debug_window_cycles)
    # but must stay out of the tool's LLM-visible schema, which only ever
    # takes file_path — rtl_agent.py calls this once at startup with the
    # active config, the same place active_build_tool is resolved from
    # config.verilog_build_tool.
    @tool
    def run_simulation(file_path: str) -> str:
        """Compile the whole project — together with every other .sv file in it — into a runnable simulation with Icarus Verilog and execute it with vvp, returning the real simulation log. Only produces a meaningful result once a self-checking testbench (a file that calls $finish) exists in the project."""
        try:
            target = _resolve_safe_path(file_path)
            if not target.exists():
                return f"File not found: {file_path}"

            if not _project_has_testbench():
                return (
                    "Simulation skipped: no testbench found in the project "
                    "(no .sv file calls $finish) — write one that drives the "
                    "design and calls $finish before running a simulation."
                )

            # Unlike build_verilog's -t null (elaborate-only), this needs a
            # real runnable target — same iverilog -o + vvp mechanics
            # test/run_validation.py already uses offline against the
            # verilog-eval dataset, just against whatever testbench is in
            # the project instead of an external one. extra_files (from
            # SIM_EXTRAS_DIR, never build_verilog/lint_verilog's own compile)
            # lets a benchmark-supplied testbench elaborate against its own
            # reference module without that module ever being visible to
            # the model via write_file/read_file/list_directory.
            extra_files = _sim_extra_sv_files()
            sim_exe = Path(GENERATED_DIR) / "sim.vvp"
            compile_result = subprocess.run(
                [IVERILOG_PATH, "-g2012", "-o", str(sim_exe), *_project_sv_files(), *extra_files],
                capture_output=True, text=True, timeout=30,
            )
            if compile_result.returncode != 0:
                log = (compile_result.stdout + _truncate_diagnostics(compile_result.stderr)).strip()
                return f"{SIM_FAILURE_PREFIX} to compile (exit code {compile_result.returncode}):\n{log}"

            # cwd=GENERATED_DIR so a testbench's relative
            # $dumpfile("wave.vcd") lands next to the design instead of
            # wherever this process happened to be launched from.
            sim_result = subprocess.run(
                [VVP_PATH, str(sim_exe)],
                capture_output=True, text=True, timeout=30, cwd=GENERATED_DIR,
            )
            log = (sim_result.stdout + sim_result.stderr).strip()

            # Runtime-error heuristic (empirically verified against this
            # exact iverilog/vvp): $fatal -> nonzero exit + "FATAL:" text;
            # $error -> exit 0 but "ERROR:" text, simulation keeps running;
            # a clean $finish -> exit 0, neither string present.
            is_runtime_error = sim_result.returncode != 0 or "ERROR:" in log or "FATAL:" in log

            # A second, independent failure signal: verilog-eval's
            # testbenches (used via SIM_EXTRAS_DIR — see
            # test/run_benchmark.py) never call $error/$fatal at all, only
            # $display a "Mismatches: N in M samples" summary from a `final`
            # block — the runtime-error heuristic alone can't see this at
            # all, and would otherwise report every one of those as a clean
            # pass regardless of whether the design was actually right.
            mismatch_match = _SIM_MISMATCH_RE.search(log)
            has_mismatches = bool(mismatch_match) and int(mismatch_match.group(1)) > 0

            if not (is_runtime_error or has_mismatches):
                return log or "Simulation ran to completion with no output."

            if is_runtime_error:
                message = f"{SIM_FAILURE_PREFIX} (runtime error):\n{log}"
            else:
                message = f"{SIM_FAILURE_PREFIX} (functional mismatch):\n{log}"

            # Best-effort debug context: the timestamp and the VCD are both
            # optional extras, never required for the failure itself to be
            # reported — a missing/unparseable one just means no window is
            # appended, not an error. Try $fatal/$error's own "Time: N"
            # first; a pure functional mismatch never has one, so fall back
            # to verilog-eval's own "first mismatch occurred at time N" text.
            time_match = _SIM_TIME_RE.search(log) or _SIM_MISMATCH_TIME_RE.search(log)
            vcd_path = Path(GENERATED_DIR) / "wave.vcd"
            if time_match and vcd_path.exists():
                try:
                    trace = parse_vcd(vcd_path)
                    # extra_files means a hidden reference module is part of
                    # this compile (see SIM_EXTRAS_DIR) — its own output is
                    # dumped as a plain "<port>_ref" wire right next to the
                    # DUT's "<port>_dut" (confirmed across the whole
                    # dataset), so exclude it from the rendered window
                    # rather than suppressing the whole window: the model
                    # still gets real stimulus/DUT-output values to debug
                    # from, just never the reference's computed answer.
                    exclude = _HIDDEN_REFERENCE_SUFFIXES if extra_files else None
                    window = render_window(
                        trace, int(time_match.group(1)), debug_window_cycles,
                        exclude_leaf_suffixes=exclude,
                    )
                    message += (
                        f"\n\nSignal values around the failure "
                        f"(+/-{debug_window_cycles} clock cycles):\n{window}"
                    )
                except (OSError, ValueError):
                    pass
            return message

        except FileNotFoundError:
            # Raised if IVERILOG_PATH/VVP_PATH itself can't be executed at
            # all (not just a compile/runtime error in the .sv files).
            return (
                "iverilog/vvp is not installed or could not be found. Install it via "
                "`winget install Icarus.Verilog` (Windows) or your package "
                "manager, and note the installer may not add it to PATH."
            )
        except subprocess.TimeoutExpired:
            return (
                f"{SIM_FAILURE_PREFIX}: timed out after 30s — check for a "
                "missing $finish or an infinite loop."
            )
        except (OSError, ValueError) as e:
            return f"Failed to simulate {file_path}: {e}"

    return run_simulation


@tool
def list_directory(path: str = ".") -> str:
    """List files and subdirectories at a path, to discover project structure, packages (.svh), and testbenches."""
    try:
        target = _resolve_safe_path(path)
        if not target.exists():
            return f"Path not found: {path}"
        if not target.is_dir():
            return f"Not a directory: {path}"
        entries = sorted(target.iterdir())
        if not entries:
            return f"(empty directory: {path})"
        return "\n".join(entry.name + ("/" if entry.is_dir() else "") for entry in entries)
    except (OSError, ValueError) as e:
        return f"Failed to list {path}: {e}"


# Tools always available regardless of which build backend is selected.
# `@tool` turns each function into a StructuredTool object, not a plain
# function — it is NOT directly callable as write_file(**args) (that raises
# "object is not callable"). The correct call is
# TOOL_FUNCTIONS[name].invoke(args), passing the whole args dict rather
# than unpacking it as keyword arguments (see rtl_agent.py).
BASE_TOOLS = [write_file, read_file, edit_file_block, list_directory]

# The two interchangeable build/lint backends — AgentConfig.verilog_build_tool
# (see config.py) selects exactly one of these to bind to the model;
# rtl_agent.py builds its own TOOLS/TOOL_FUNCTIONS from BASE_TOOLS plus
# whichever one is active, rather than importing a fixed list here.
BUILD_TOOL_FUNCTIONS = {
    "icarus": build_verilog,
    "slang": lint_verilog,
}

# Return-value prefixes that mean "the build/lint failed", one per backend —
# build_verilog and lint_verilog intentionally keep their own distinct
# wording ("Compilation failed" vs "Lint failed", more informative than a
# generic message), so rtl_agent.py's auto-build enforcement needs this
# lookup to detect failure generically instead of hardcoding either string.
BUILD_FAILURE_PREFIXES = {
    "icarus": "Compilation failed",
    "slang": "Lint failed",
}
