import re
from dataclasses import dataclass

# Icarus (and every other simulator) emits $var/$scope/$upscope declarations
# on a single line each, e.g. "$scope module tb $end" / "$var wire 1 ! clk $end"
# — these regexes only need to match the leading tokens, not consume the
# trailing "$end" or a multi-bit signal's "[7:0]" range suffix.
_VAR_RE = re.compile(r"^\$var\s+\S+\s+\d+\s+(\S+)\s+(\S+)")
_SCALAR_CHANGE_RE = re.compile(r"^([01xXzZ])(\S+)$")
_VECTOR_CHANGE_RE = re.compile(r"^[bB]([01xXzZ]+)\s+(\S+)$")


@dataclass
class VcdTrace:
    # Declaration order, as dotted hierarchical names (e.g. "stim1.clk",
    # "tb.zero_ref") — built from the $scope/$upscope nesting so two
    # same-named signals in different scopes don't collide.
    signal_order: list
    # name -> sorted list of (time, value, is_vector) — one entry per
    # recorded change, not one per simulated timestep (VCD only records
    # transitions). is_vector distinguishes a multi-bit "b<bits> <id>" dump
    # from a single-character scalar one, since a bare digit string like
    # "11" is genuinely ambiguous (binary 3, or literally "eleven"?) without
    # knowing which form it came from — see _format_value.
    changes: dict


def parse_vcd(path) -> VcdTrace:
    signal_order = []
    id_to_name = {}
    changes = {}
    scope_stack = []
    current_time = 0

    with open(path, encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("$scope"):
                scope_stack.append(line.split()[2])
                continue
            if line.startswith("$upscope"):
                if scope_stack:
                    scope_stack.pop()
                continue
            if line.startswith("$var"):
                m = _VAR_RE.match(line)
                if m:
                    var_id, var_name = m.group(1), m.group(2)
                    full_name = ".".join([*scope_stack, var_name]) if scope_stack else var_name
                    id_to_name[var_id] = full_name
                    if full_name not in changes:
                        signal_order.append(full_name)
                        changes[full_name] = []
                continue
            if line.startswith("#"):
                try:
                    current_time = int(line[1:])
                except ValueError:
                    pass
                continue
            if line.startswith("$"):
                # $date/$version/$timescale/$comment/$enddefinitions/$dumpvars/
                # $dumpall/$dumpon/$dumpoff/$end — none carry a value change
                # themselves (dumpvars' actual value-change lines are its
                # body, not this command line, and fall through below).
                continue

            m = _VECTOR_CHANGE_RE.match(line)
            if m:
                value, var_id, is_vector = m.group(1), m.group(2), True
            else:
                m = _SCALAR_CHANGE_RE.match(line)
                if not m:
                    continue  # unrecognized (e.g. a real-number "r..." change) — ignore
                value, var_id, is_vector = m.group(1), m.group(2), False

            name = id_to_name.get(var_id)
            if name is not None:
                changes[name].append((current_time, value, is_vector))

    return VcdTrace(signal_order=signal_order, changes=changes)


def find_clock_period(trace: VcdTrace):
    # Looks for a dumped signal literally named clk/clock (matching just the
    # leaf, so "stim1.clk" counts) — the near-universal RTL/testbench
    # convention already used throughout this repo's own testbenches and
    # verilog-eval's. One "cycle" = the time between the first two rising
    # (0->1) edges. Every recorded change on a strictly-toggling clock
    # alternates 0/1, so filtering for value == "1" already yields exactly
    # the rising-edge timestamps.
    for name in trace.signal_order:
        leaf = name.rsplit(".", 1)[-1]
        if leaf.lower() in ("clk", "clock"):
            edges = sorted(t for t, v, _ in trace.changes[name] if v == "1")
            if len(edges) >= 2:
                return edges[1] - edges[0]
    return None


def _format_value(value: str, is_vector: bool) -> str:
    # A bare scalar digit ("0"/"1"/"x"/"z") is unambiguous as-is. A vector's
    # raw bit string is not: "11" from a "b11 <id>" dump is binary 3, but
    # rendered bare it reads exactly like decimal 11 — actively misleading
    # for debugging a mismatch. All-0/1 vectors are converted to decimal;
    # anything with x/z bits can't be, so it's shown as explicit binary
    # instead (e.g. "0b1x01") rather than silently guessing a value.
    if not is_vector:
        return value
    if all(c in "01" for c in value):
        return str(int(value, 2))
    return f"0b{value}"


def _value_at(changes: list, t: int) -> str:
    # changes is time-ordered (VCD guarantees monotonically increasing
    # timestamps) — the value in effect at time t is the last recorded
    # change at or before it (standard VCD replay / forward-fill).
    value, is_vector = "x", False
    for ct, cv, cvec in changes:
        if ct > t:
            break
        value, is_vector = cv, cvec
    return _format_value(value, is_vector)


def _display_names(signal_order: list) -> dict:
    # Bare leaf names (e.g. "clk" instead of "tb.stim1.clk") read far better
    # in the rendered table, but two signals in different scopes can share a
    # leaf (observed directly: a real testbench clock and an unrelated,
    # always-'x' clk port on a stimulus-generator submodule both dumped as
    # "clk") — silently showing two identically-labeled "clk=" columns would
    # be actively misleading. Any leaf that collides falls back to its full
    # dotted path for every signal that shares it, not just the duplicates.
    leaf_counts = {}
    for name in signal_order:
        leaf = name.rsplit(".", 1)[-1]
        leaf_counts[leaf] = leaf_counts.get(leaf, 0) + 1
    return {
        name: name if leaf_counts[name.rsplit(".", 1)[-1]] > 1 else name.rsplit(".", 1)[-1]
        for name in signal_order
    }


def render_window(trace: VcdTrace, center_time: int, cycles: int, exclude_leaf_suffixes=None) -> str:
    """Render every dumped signal's value at each change point within
    +/-`cycles` clock cycles of `center_time` (falling back to a fixed
    absolute window if no clock signal was found in the trace).

    exclude_leaf_suffixes, if given, drops any signal whose leaf name ends
    with one of those strings (case-insensitive) from the rendered table —
    this module has no idea what the suffixes mean (that's the caller's
    concern, e.g. tools.py excluding a hidden reference module's "_ref"
    signals), it just does the mechanical filtering."""
    signal_order = trace.signal_order
    if exclude_leaf_suffixes:
        suffixes = tuple(s.lower() for s in exclude_leaf_suffixes)
        signal_order = [
            name for name in signal_order
            if not name.rsplit(".", 1)[-1].lower().endswith(suffixes)
        ]

    period = find_clock_period(trace)
    note = None
    if period:
        lo, hi = center_time - cycles * period, center_time + cycles * period
    else:
        window = 20
        lo, hi = center_time - window, center_time + window
        note = f"(no clock signal detected in the trace — showing a fixed +/-{window}-unit window instead of {cycles} cycles)"

    times = {center_time}
    for name in signal_order:
        times.update(t for t, _, _ in trace.changes[name] if lo <= t <= hi)
    times = sorted(t for t in times if lo <= t <= hi)
    display = _display_names(signal_order)

    lines = [note] if note else []
    for t in times:
        row = f"time={t}  " + " ".join(
            f"{display[name]}={_value_at(trace.changes[name], t)}"
            for name in signal_order
        )
        if t == center_time:
            row += "   <-- failure time"
        lines.append(row)
    return "\n".join(lines)
