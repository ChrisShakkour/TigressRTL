from dataclasses import dataclass

# Scaled-down port of Hermes's agent/verify/ shape: phased evidence (build,
# then simulate) plus a stop-gate that blocks a final answer until the
# phases this project actually needs have fresh evidence, not just a hope
# that the model remembers to check. Replaces the three loose booleans
# rtl_agent.py used to carry directly (build_failed, build_retry_count,
# fix_plan_pending) with one object that also knows how to explain, in
# plain English, exactly what's still missing.
#
# Two phases exist for this domain:
#   build    — did the last build/lint for the CURRENT files pass? Always
#              required.
#   simulate — did the last simulation for the CURRENT files run to
#              completion (no crash, no timeout)? Only required when the
#              active build tool is icarus (Slang has no simulator) and the
#              project actually contains a testbench (tools.py's
#              project_has_testbench()).
#
# "Ran to completion" is deliberately NOT the same claim as "the design is
# correct" — an arbitrary model-authored testbench has no fixed pass/fail
# format this harness can parse (unlike test/run_validation.py's dataset
# testbenches, which all end with a known "Mismatches: N in M samples"
# line). Reading the real simulation output and judging correctness against
# it stays the model's job; this object only guarantees that job wasn't
# skipped entirely.


@dataclass
class VerificationState:
    build_passed: bool = False
    simulate_ran: bool = False
    retry_count: int = 0
    fix_plan_pending: bool = False

    def record_build(self, passed: bool) -> None:
        self.build_passed = passed
        # A fresh build supersedes any earlier simulate evidence — the
        # files it just checked may not be the same files an earlier
        # simulation ran against (this build could follow an edit made
        # after the last simulate call).
        self.simulate_ran = False
        self._record(passed)

    def record_simulate(self, ran_clean: bool) -> None:
        self.simulate_ran = ran_clean
        self._record(ran_clean)

    def _record(self, passed: bool) -> None:
        if passed:
            self.retry_count = 0
            self.fix_plan_pending = False
        else:
            self.fix_plan_pending = True

    def is_satisfied(self, simulate_required: bool) -> bool:
        if not self.build_passed:
            return False
        if simulate_required and not self.simulate_ran:
            return False
        return True

    def needs_retry(self, simulate_required: bool, max_retries: int) -> bool:
        return not self.is_satisfied(simulate_required) and self.retry_count < max_retries

    def nudge_message(self, build_tool_name: str, simulate_tool_name: str) -> str:
        # Two genuinely different situations get two different messages:
        # an actual failure to fix (fix_plan_pending is True, and the
        # FIX_PLAN_INSTRUCTION diagnostic call in rtl_agent.py already fired
        # for it) versus nothing having failed at all, simulate was just
        # never attempted even though this project now has a testbench.
        if not self.build_passed:
            return (
                f"The last {build_tool_name} result was a failure, and you "
                "did not call write_file or edit_file_block to actually "
                "apply a fix — you only described one. Call the appropriate "
                "tool now with the corrected content."
            )
        return (
            f"This project contains a testbench, but you gave a final "
            f"answer without calling {simulate_tool_name} against the "
            "current files. A clean compile does not prove the design "
            f"behaves correctly — call {simulate_tool_name} now and read "
            "its real output before answering."
        )
