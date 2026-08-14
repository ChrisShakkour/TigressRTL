import argparse
import fnmatch
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Windows terminals default to a codepage that can't render some characters —
# force UTF-8 output so nothing gets garbled (same fix as rtl_agent.py).
sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
RTL_AGENT = REPO_ROOT / "src" / "rtl_agent.py"

# Matches rtl_agent.py's own [Token Usage] summary line, printed once on
# exit -- parsed out of the subprocess's captured stdout the same way
# run_validation.py already pulls "Mismatches: N in M samples" out of raw
# simulation output, no separate IPC mechanism needed.
TOKEN_USAGE_RE = re.compile(
    r"\[Token Usage\] input_tokens=(\d+) output_tokens=(\d+) total_tokens=(\d+)"
)

# test/run_benchmark.py lives outside src/, so tools.py/config.py aren't on
# sys.path the way they are for scripts run directly from inside src/ —
# there's no active editable install of this package on this machine (every
# other script here is just run directly, never `pip install -e .`'d), so
# don't assume `tools`/`config` are globally importable. Adding src/ here
# makes this script self-contained regardless of install state.
sys.path.insert(0, str(REPO_ROOT / "src"))

from config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the RTL agent over the verilog-eval spec-to-rtl problems, "
                    "one fresh agent process per problem. Generation only — no "
                    "validation/scoring against the dataset's reference solutions yet."
    )
    parser.add_argument(
        "--dataset-dir", required=True,
        help="Path to a cloned verilog-eval repo's dataset_spec-to-rtl/ directory "
             "(git clone https://github.com/NVlabs/verilog-eval.git)",
    )
    parser.add_argument(
        "--configs", nargs="+", default=None,
        help="One or more AgentConfig YAML files to sweep (default: configs/default.yaml). "
             "The full problem set is run once per config.",
    )
    parser.add_argument(
        "--problems", nargs="*", default=None,
        help="Exact problem names or fnmatch-style globs (e.g. 'Prob00*') to filter the "
             "discovered set. Default: all problems in --dataset-dir.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the (post-filter) problem count — useful for a quick smoke test.",
    )
    parser.add_argument(
        "--output-dir", default=str(REPO_ROOT / "benchmark_runs"),
        help="Parent directory for timestamped run directories (default: benchmark_runs/).",
    )
    parser.add_argument(
        "--resume-dir", default=None,
        help="Continue a specific previous timestamped run directory instead of creating a "
             "new one — skips (config, problem) pairs already marked completed in it. "
             "Without this, every invocation always starts fresh in a new directory.",
    )
    parser.add_argument(
        "--timeout", type=float, default=None,
        help="Per-problem subprocess timeout in seconds (default: none — a problem can take "
             "however long the model needs).",
    )
    parser.add_argument(
        "--python", default=sys.executable,
        help="Interpreter used to launch rtl_agent.py (default: the current interpreter, so "
             "an activated venv is respected automatically).",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of (config, problem) pairs to run concurrently, each its own "
             "rtl_agent.py subprocess with its own isolated GENERATED_DIR (see "
             "TIGRESSRTL_GENERATED_DIR in src/tools.py). Default: 1 -- sequential, "
             "identical behavior/timing to before this flag existed. Real concurrency "
             "is bounded by how many simultaneous requests your local Ollama server "
             "can actually serve well, not CPU count -- start small (2-4) and watch "
             "for degraded per-problem latency before raising it further.",
    )
    return parser.parse_args()


def discover_problems(dataset_dir: Path) -> list[str]:
    prompt_files = sorted(dataset_dir.glob("*_prompt.txt"))
    return [p.name[: -len("_prompt.txt")] for p in prompt_files]


def copy_generated_outputs(generated_dir: str, dest: Path) -> int:
    src = Path(generated_dir)
    if not src.exists():
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for entry in src.rglob("*"):
        if entry.is_file():
            rel = entry.relative_to(src)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry, target)
            if entry.suffix in (".sv", ".svh"):
                count += 1
    return count


def parse_token_usage(log: str) -> dict:
    match = TOKEN_USAGE_RE.search(log)
    if not match:
        # Subprocess crashed/timed out before reaching the exit path, or
        # predates this feature -- None, not 0, so it's distinguishable
        # from a run that genuinely used zero tokens.
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    input_tokens, output_tokens, total_tokens = (int(g) for g in match.groups())
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def run_one(python: str, config_path: Path, prompt_single_line: str, timeout: float | None,
            generated_dir: str):
    stdin_text = prompt_single_line + "\nexit\n"
    # A copy of the real environment (not a from-scratch dict) so PATH etc.
    # still resolve python/ollama/iverilog/slang inside the subprocess --
    # only TIGRESSRTL_GENERATED_DIR is added/overridden, pointing this one
    # subprocess at its own isolated directory (see tools.py) rather than
    # the shared default, which is what makes --workers > 1 safe.
    env = os.environ.copy()
    env["TIGRESSRTL_GENERATED_DIR"] = generated_dir
    start = datetime.now(timezone.utc)
    try:
        proc = subprocess.run(
            [python, str(RTL_AGENT), "--config", str(config_path)],
            input=stdin_text, capture_output=True, text=True, encoding="utf-8",
            cwd=REPO_ROOT, timeout=timeout, env=env,
        )
        log = proc.stdout + proc.stderr
        returncode = proc.returncode
        status = "completed"
    except subprocess.TimeoutExpired as e:
        log = (e.stdout or "") + (e.stderr or "")
        returncode = None
        status = "timeout"
    except OSError as e:
        log = f"harness error launching subprocess: {e}"
        returncode = None
        status = "error"
    duration = (datetime.now(timezone.utc) - start).total_seconds()
    return status, returncode, log, duration


def _run_task(task: dict) -> dict:
    # The unit of work dispatched either directly (sequential, --workers 1,
    # the default) or via a multiprocessing.Pool (--workers > 1) -- a
    # single top-level function (not a closure) so it's picklable for
    # Pool's default 'spawn' start method on macOS/Windows. Takes/returns
    # only plain JSON-safe values for the same reason.
    config_stem = task["config_stem"]
    problem = task["problem"]
    problem_dir = Path(task["run_dir"]) / config_stem / problem
    problem_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = Path(task["dataset_dir"]) / f"{problem}_prompt.txt"
    raw = prompt_path.read_text(encoding="utf-8")
    single_line = " ".join(raw.split())
    (problem_dir / "prompt.txt").write_text(single_line, encoding="utf-8")

    # A fresh temp dir per task, not the fixed src/generated/ default --
    # used unconditionally, even at --workers 1, so the sequential and
    # concurrent paths run through identical code and this isolation can
    # never silently regress if --workers is raised later without anyone
    # re-touching this function.
    tmp_generated_dir = tempfile.mkdtemp(prefix=f"tigressrtl_{config_stem}_{problem}_")
    try:
        status, returncode, log, duration = run_one(
            task["python"], Path(task["config_path"]), single_line, task["timeout"],
            tmp_generated_dir,
        )
        (problem_dir / "transcript.log").write_text(log, encoding="utf-8")
        sv_count = copy_generated_outputs(tmp_generated_dir, problem_dir / "generated")
    finally:
        shutil.rmtree(tmp_generated_dir, ignore_errors=True)

    token_usage = parse_token_usage(log)
    (problem_dir / "status.json").write_text(json.dumps({
        "problem": problem,
        "config_path": task["config_path"],
        "model": task["model"],
        "num_ctx": task["num_ctx"],
        "status": status,
        "returncode": returncode,
        "sv_file_count": sv_count,
        "duration_s": round(duration, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **token_usage,
    }, indent=2), encoding="utf-8")

    return {
        "config_stem": config_stem,
        "problem": problem,
        "status": status,
        "sv_count": sv_count,
        "duration": duration,
        **token_usage,
    }


def _record_result(result: dict, summary: dict, token_summary: dict) -> None:
    # Shared by both the sequential and pool dispatch branches in main() so
    # printing/aggregation behaves identically regardless of --workers.
    config_stem = result["config_stem"]
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        token_summary[config_stem][key] += result[key] or 0
    print(f"[{result['status']}] {config_stem}/{result['problem']} "
          f"({result['sv_count']} file(s), {result['duration']:.1f}s, "
          f"{result['total_tokens'] or 0} tokens)")
    summary[config_stem][result["status"]] = summary[config_stem].get(result["status"], 0) + 1


def main():
    args = parse_args()

    if args.workers < 1:
        sys.exit(f"--workers must be at least 1, got {args.workers}.")

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    config_paths = (
        [Path(c).resolve() for c in args.configs]
        if args.configs
        else [DEFAULT_CONFIG_PATH]
    )

    if not dataset_dir.is_dir():
        sys.exit(f"--dataset-dir '{dataset_dir}' is not a directory.")
    all_problems = discover_problems(dataset_dir)
    if not all_problems:
        sys.exit(
            f"No *_prompt.txt files found in '{dataset_dir}'. Did you point at the "
            "verilog-eval repo root instead of its dataset_spec-to-rtl/ subdirectory?"
        )

    # Validate every config up front (raises ValueError on a typo'd key) and
    # reject duplicate stems, before any subprocess runs — a long sweep
    # shouldn't die on config #2 of 5 after config #1 already finished.
    stems_seen = {}
    for config_path in config_paths:
        try:
            load_config(config_path)
        except ValueError as e:
            sys.exit(str(e))
        stem = config_path.stem
        if stem in stems_seen:
            sys.exit(
                f"Two --configs paths both resolve to the stem '{stem}': "
                f"{stems_seen[stem]} and {config_path}. Rename one to avoid an "
                "output directory collision."
            )
        stems_seen[stem] = config_path

    problems = all_problems
    if args.problems:
        problems = [
            p for p in problems
            if any(fnmatch.fnmatch(p, pattern) or p == pattern for pattern in args.problems)
        ]
        if not problems:
            sys.exit(f"--problems filter matched none of the {len(all_problems)} discovered problems.")
    if args.limit is not None:
        problems = problems[: args.limit]

    if args.resume_dir:
        run_dir = Path(args.resume_dir).resolve()
        if not run_dir.is_dir():
            sys.exit(f"--resume-dir '{run_dir}' does not exist.")
        resuming = True
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = output_dir / timestamp
        run_dir.mkdir(parents=True)
        resuming = False

    if not resuming:
        (run_dir / "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")
        configs_copy_dir = run_dir / "configs"
        configs_copy_dir.mkdir()
        for config_path in config_paths:
            shutil.copy2(config_path, configs_copy_dir / config_path.name)

    print(f"Run directory: {run_dir}")
    print(f"Configs: {[str(c) for c in config_paths]}")
    print(f"Problems: {len(problems)} of {len(all_problems)} discovered\n")

    summary = {}  # config_stem -> {status: count}
    # config_stem -> {input_tokens, output_tokens, total_tokens} -- "total
    # for this run" means the whole run directory's current state, so
    # skipped (already-completed) problems contribute their own
    # already-recorded tokens too, not just freshly-run ones.
    token_summary = {}
    tasks = []  # queued (config, problem) pairs that still need to run

    for config_path in config_paths:
        config_stem = config_path.stem
        summary[config_stem] = {}
        token_summary[config_stem] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        cfg = load_config(config_path)

        for problem in problems:
            status_file = run_dir / config_stem / problem / "status.json"

            if resuming and status_file.exists():
                existing = json.loads(status_file.read_text(encoding="utf-8"))
                if existing.get("status") == "completed":
                    print(f"[skip] {config_stem}/{problem} already completed")
                    summary[config_stem]["skipped"] = summary[config_stem].get("skipped", 0) + 1
                    # Older runs predate token tracking and have no
                    # input_tokens/etc. keys at all -- .get(..., 0) with a
                    # `None` fallback handled by `or 0` covers both that
                    # and a value explicitly recorded as None.
                    for key in ("input_tokens", "output_tokens", "total_tokens"):
                        token_summary[config_stem][key] += existing.get(key) or 0
                    continue

            tasks.append({
                "config_stem": config_stem,
                "config_path": str(config_path),
                "problem": problem,
                "dataset_dir": str(dataset_dir),
                "run_dir": str(run_dir),
                "python": args.python,
                "timeout": args.timeout,
                "model": cfg.model,
                "num_ctx": cfg.num_ctx,
            })

    # --workers > 1 dispatches queued tasks (interleaved across every
    # config being swept, not grouped one config at a time) through a
    # process pool -- correctness doesn't depend on ordering since each
    # task now runs in its own isolated temp GENERATED_DIR (_run_task).
    # --workers 1 (the default) calls _run_task directly with no pool
    # overhead, so default behavior/timing is unchanged from before this
    # flag existed.
    if not tasks:
        pass  # everything was already completed (--resume-dir) -- nothing to dispatch
    elif args.workers > 1:
        with multiprocessing.Pool(processes=args.workers) as pool:
            for result in pool.imap_unordered(_run_task, tasks):
                _record_result(result, summary, token_summary)
    else:
        for task in tasks:
            _record_result(_run_task(task), summary, token_summary)

    print(f"\nRun directory: {run_dir}")
    for config_stem, counts in summary.items():
        tokens = token_summary[config_stem]
        print(f"  {config_stem}: {counts}  tokens: input={tokens['input_tokens']} "
              f"output={tokens['output_tokens']} total={tokens['total_tokens']}")


if __name__ == "__main__":
    main()
