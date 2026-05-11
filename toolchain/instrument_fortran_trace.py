#!/usr/bin/env python3
"""Inject runtime grid-point trace scopes into generated Fortran sources."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

MODULE_RE = re.compile(r"^(\s*)module\s+([a-z_]\w*)\b", re.IGNORECASE)
PROGRAM_RE = re.compile(r"^(\s*)program\s+([a-z_]\w*)\b", re.IGNORECASE)
ROUTINE_RE = re.compile(
    r"^(\s*)(?:(?:(?:impure|recursive|module)\s+)*subroutine\s+([a-z_]\w*)|"
    r"(?:(?:impure|recursive|module)\s+)*function\s+([a-z_]\w*)|"
    r"(?:(?:pure|elemental|recursive|module|impure)\s+)*"
    r"(?:integer|real|logical|complex|character|type\s*\([^)]*\)|class\s*\([^)]*\)|double\s+precision)\b[^!]*?\bfunction\s+([a-z_]\w*))\b",
    re.IGNORECASE,
)
END_ROUTINE_RE = re.compile(r"^\s*end\s+(?:subroutine|function|program)\b", re.IGNORECASE)
INTERFACE_RE = re.compile(r"^\s*(?:abstract\s+)?interface\b", re.IGNORECASE)
END_INTERFACE_RE = re.compile(r"^\s*end\s+interface\b", re.IGNORECASE)
END_DO_RE = re.compile(r"(?:^|;)\s*end\s+do\b", re.IGNORECASE)
DO_CAPTURE_RE = re.compile(r"(?:^|;)\s*do\s+([a-z_]\w*)\s*=\s*([^;!]+)", re.IGNORECASE)

POINT_LOOP_CANDIDATES = (
    ("j", "k", "l"),
    ("x", "y", "z"),
    ("i", "j", "k"),
    ("id1", "id2", "id3"),
    ("j_loop", "k_loop", "l_loop"),
    ("k_loop", "l_loop", "q_loop"),
    ("k_idx", "l_idx", "q_idx"),
    ("q_idx", "k_idx", "l_idx"),
    ("l_idx", "q_idx", "k_idx"),
)

DECLARATION_STARTERS = (
    "use ",
    "import",
    "implicit ",
    "integer",
    "real",
    "double precision",
    "complex",
    "logical",
    "character",
    "type",
    "class",
    "procedure",
    "external",
    "intrinsic",
    "parameter",
    "dimension",
    "allocatable",
    "pointer",
    "target",
    "optional",
    "save",
    "common",
    "equivalence",
    "namelist",
    "data ",
    "private",
    "public",
    "volatile",
    "asynchronous",
    "protected",
    "enumerator",
    "gpu_routine",
)


def strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "!" and not in_single and not in_double:
            return line[:idx]
    return line


def normalized(line: str) -> str:
    return strip_comment(line).strip().lower()


def is_preprocessor_or_empty(line: str) -> bool:
    text = line.strip()
    return not text or text.startswith("#") or text.startswith("!")


def is_continuation(line: str) -> bool:
    return strip_comment(line).rstrip().endswith("&")


def is_specification_line(line: str, spec_block_depth: int) -> bool:
    text = normalized(line)
    if not text:
        return True
    if text.startswith("#"):
        return True
    if text.startswith("&"):
        return True
    if spec_block_depth > 0:
        return True
    if "::" in text:
        return True
    return text.startswith(DECLARATION_STARTERS)


def starts_spec_block(line: str) -> bool:
    text = normalized(line)
    return bool(re.match(r"^(type|enum|interface)\b", text)) and not text.startswith("type(")


def ends_spec_block(line: str) -> bool:
    text = normalized(line)
    return bool(re.match(r"^end\s+(type|enum|interface)\b", text))


def should_skip_routine(line: str) -> bool:
    text = normalized(line)
    return " pure " in f" {text} " or " elemental " in f" {text} "


def routine_name(match: re.Match[str]) -> str:
    return next(group for group in match.groups()[1:] if group is not None)


def is_x_bound(bounds: str) -> bool:
    return bool(
        re.search(
            r"(^|[^a-z0-9_])(m|m_glb|local_m|x_beg|x_end|offset_x|is1[a-z0-9_]*|isx|ix|idwint\(1\)|idwbuff\(1\)|ibounds\(1\)|bounds\(1\)|isc1|id_norm\(1\))([^a-z0-9_]|$)",
            bounds,
        )
    )


def is_y_bound(bounds: str) -> bool:
    return bool(
        re.search(
            r"(^|[^a-z0-9_])(n|n_glb|local_n|y_beg|y_end|offset_y|is2[a-z0-9_]*|isy|iy|idwint\(2\)|idwbuff\(2\)|ibounds\(2\)|bounds\(2\)|isc2|id_norm\(2\))([^a-z0-9_]|$)",
            bounds,
        )
    )


def is_z_bound(bounds: str) -> bool:
    return bool(
        re.search(
            r"(^|[^a-z0-9_])(p|p_glb|local_p|z_beg|z_end|offset_z|is3[a-z0-9_]*|isz|iz|idwint\(3\)|idwbuff\(3\)|ibounds\(3\)|bounds\(3\)|isc3|id_norm\(3\))([^a-z0-9_]|$)",
            bounds,
        )
    )


def split_do_bounds(bounds: str) -> tuple[str, str]:
    parts: list[str] = []
    start = 0
    depth = 0
    in_single = False
    in_double = False

    for idx, char in enumerate(bounds):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if char == "(":
                depth += 1
            elif char == ")" and depth > 0:
                depth -= 1
            elif char == "," and depth == 0:
                parts.append(bounds[start:idx].strip())
                start = idx + 1

    parts.append(bounds[start:].strip())
    if len(parts) >= 2 and parts[0] and parts[1]:
        return parts[0], parts[1]

    return "0", "0"


def active_point_vars(loop_stack: list[dict[str, object]]) -> tuple[str, str, str] | None:
    active = {str(entry["var"]): str(entry["bounds"]) for entry in loop_stack}
    for candidate in POINT_LOOP_CANDIDATES:
        if all(var in active for var in candidate) and is_x_bound(active[candidate[0]]) and is_y_bound(active[candidate[1]]) and is_z_bound(active[candidate[2]]):
            return candidate
    return None


def point_scope_active(loop_stack: list[dict[str, object]]) -> bool:
    return any(bool(entry["point_scope"]) for entry in loop_stack)


def point_begin_line(indent: str, point_vars: tuple[str, str, str], loop_stack: list[dict[str, object]]) -> str:
    active = {str(entry["var"]): entry for entry in loop_stack}
    point_args = ", ".join(f"int({var})" for var in point_vars)
    midpoint_args = []
    for var in point_vars:
        entry = active[var]
        midpoint_args.append(f"int(({entry['lower']}) + ({entry['upper']}))/2")
    point_names = ",".join(point_vars)
    return f"{indent}call s_trace_point_begin({point_args}, '{point_names}', {', '.join(midpoint_args)})\n"


def instrument(lines: list[str], point_traces: bool = True) -> list[str]:
    if any(re.match(r"^\s*module\s+m_trace\b", line, re.IGNORECASE) for line in lines):
        return lines

    output: list[str] = []
    scope_stack: list[str] = []
    pending_routine: dict[str, str | int] | None = None
    routine_stack: list[str] = []
    loop_stack: list[dict[str, object]] = []
    interface_depth = 0
    spec_block_depth = 0

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        text = normalized(line)

        if END_INTERFACE_RE.match(line):
            interface_depth = max(0, interface_depth - 1)
        elif INTERFACE_RE.match(line):
            interface_depth += 1

        if pending_routine is not None:
            if is_preprocessor_or_empty(line):
                output.append(line)
                idx += 1
                continue

            if ends_spec_block(line):
                spec_block_depth = max(0, spec_block_depth - 1)
                output.append(line)
                idx += 1
                continue

            if pending_routine.get("continuation") or is_specification_line(line, spec_block_depth):
                output.append(line)
                pending_routine["continuation"] = is_continuation(line)
                if starts_spec_block(line):
                    spec_block_depth += 1
                idx += 1
                continue

            routine_stack.append(str(pending_routine["name"]))
            pending_routine = None
            spec_block_depth = 0
            continue

        module_match = MODULE_RE.match(line)
        if module_match and not text.startswith("module procedure") and not text.startswith("end module"):
            module_name = module_match.group(2)
            output.append(line)
            if module_name.lower() != "m_trace":
                output.append(f"{module_match.group(1)}    use m_trace, only: s_trace_point_begin, s_trace_point_end\n")
            scope_stack.append(module_name)
            idx += 1
            continue

        program_match = PROGRAM_RE.match(line)
        if program_match and not text.startswith("end program"):
            program_name = program_match.group(2)
            output.append(line)
            output.append(f"{program_match.group(1)}    use m_trace, only: s_trace_point_begin, s_trace_point_end\n")
            pending_routine = {
                "indent": f"{program_match.group(1)}    ",
                "name": program_name,
                "continuation": is_continuation(line),
            }
            idx += 1
            continue

        if text.startswith("end module") and scope_stack:
            scope_stack.pop()

        routine_match = ROUTINE_RE.match(line)
        if routine_match and interface_depth == 0 and not text.startswith("end ") and " procedure " not in f" {text} " and not should_skip_routine(line):
            output.append(line)
            if not scope_stack:
                output.append(f"{routine_match.group(1)}    use m_trace, only: s_trace_point_begin, s_trace_point_end\n")
            pending_routine = {
                "indent": f"{routine_match.group(1)}    ",
                "name": routine_name(routine_match),
                "continuation": is_continuation(line),
            }
            idx += 1
            continue

        if routine_stack and END_ROUTINE_RE.match(line):
            routine_stack.pop()
            loop_stack = []
            output.append(line)
            idx += 1
            continue

        if routine_stack and point_traces:
            code_line = strip_comment(line)
            end_do_count = len(END_DO_RE.findall(code_line))
            if end_do_count:
                end_indent = re.match(r"^(\s*)", line).group(1) + "    "
                for _ in range(min(end_do_count, len(loop_stack))):
                    loop_entry = loop_stack.pop()
                    if loop_entry["point_scope"]:
                        output.append(f"{end_indent}call s_trace_point_end()\n")
            else:
                begin_after_line: list[str] = []
                for do_match in DO_CAPTURE_RE.finditer(code_line):
                    bounds = " ".join(do_match.group(2).lower().split())
                    lower, upper = split_do_bounds(bounds)
                    loop_stack.append(
                        {
                            "var": do_match.group(1).lower(),
                            "bounds": bounds,
                            "lower": lower,
                            "upper": upper,
                            "point_scope": False,
                        }
                    )
                    point_vars = active_point_vars(loop_stack)
                    if point_vars is not None and not point_scope_active(loop_stack):
                        loop_stack[-1]["point_scope"] = True
                        begin_indent = re.match(r"^(\s*)", line).group(1) + "    "
                        begin_after_line.append(point_begin_line(begin_indent, point_vars, loop_stack))

                if begin_after_line:
                    output.append(line)
                    output.extend(begin_after_line)
                    idx += 1
                    continue

        output.append(line)
        idx += 1

    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-point-traces", action="store_true", help="Skip grid-point trace probes in accelerated GPU builds.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    lines = args.input.read_text().splitlines(keepends=True)
    traced = instrument(lines, point_traces=not args.no_point_traces)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text("".join(traced))
    os.replace(tmp, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
