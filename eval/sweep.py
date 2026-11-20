"""Detect mechanical "sweep" commits (typing, imports, decorators, strings, whitespace).

A commit is a sweep when at least ``SWEEP_THRESHOLD`` of the added lines in the files it is given
(non-test ``.py`` files, chosen by the caller) are *trivial*. Session 0D does not plant bugs in
sweep commits, because their changed lines carry no logic to break.

An added line is trivial when:

- it is blank (whitespace), an import line or part of a parenthesised import, a decorator, a bare
  annotation with no value (``conn: Connection``), or inside a triple-quoted string; or
- after ``normalise`` it is empty or only a string literal, equals the normalised form of a
  removed line it replaces, or is a piece of the removed lines joined together (the same code
  re-wrapped over more lines).

"The removed line it replaces" means a removed line in the same change block: a run of removed
lines and the added lines that directly follow them in a hunk. Pure additions have no removed
lines, so only the first rule can make them trivial.

``normalise`` removes, in order: type-checker comments (``# type:``, ``# noqa``, ``# pyright:``,
``# mypy:``, ``# ty:``), string literal contents and quote style, parameter, variable and return
annotations, all whitespace and trailing commas. The annotation removal is a heuristic on single
lines (no parser: diff lines are fragments); it is deliberately conservative outside ``def``
lines, where ``name: value`` could also be a dict entry.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from eval.ground_truth import parse_diff

SWEEP_THRESHOLD = 0.7

_CHECKER_COMMENT = re.compile(r"#\s*(type:|noqa|pyright:|mypy:|ty:).*$")
_IMPORT = re.compile(r"^(import\s+\w|from\s+[\w.]+\s+import\b)")
_DEF = re.compile(r"^(async\s+)?def\s")
_LAMBDA = re.compile(r"\blambda\b")
_BLOCK_KEYWORD = re.compile(
    r"^(if|elif|else|for|while|with|try|except|finally|class|return|lambda|yield|match|case|"
    r"assert|raise|del|global|nonlocal|async\s+for|async\s+with|await)\b"
)
_NAME_BEFORE_COLON = re.compile(r"(^|[\s(,*])\*{0,2}[A-Za-z_][\w.]*\s*$")
_TYPE_LIKE = re.compile(r"^\s*[A-Za-z_][\w.\[\], |]*$")  # S (a string) is a name
_TRIPLE = re.compile(r'"""|\'\'\'')
_BARE_ANNOTATION = re.compile(r"^\s*[A-Za-z_][\w.]*\s*:\s*[A-Za-z_][\w.\[\], |]*$")  # x: T
_STRING_PREFIX = re.compile(r"(?:^|[^\w])([fFrRbBuU]{1,2})$")


@dataclass(frozen=True)
class SweepStats:
    trivial: int
    total: int

    @property
    def ratio(self) -> float:
        return self.trivial / self.total if self.total else 0.0

    @property
    def is_sweep(self) -> bool:
        return self.total > 0 and self.ratio >= SWEEP_THRESHOLD


def _replace_strings(code: str) -> str:
    """Replace each string literal (with its prefix) by ``S``; an unterminated one runs to EOL."""
    out: list[str] = []
    i, n = 0, len(code)
    while i < n:
        c = code[i]
        if c in "\"'":
            prefix = _STRING_PREFIX.search("".join(out[-3:]))
            if prefix:  # drop a string prefix (f, r, b, u, rb, ...) already copied to out
                del out[len(out) - len(prefix.group(1)) :]
            q = code[i : i + 3] if code[i : i + 3] in ('"""', "'''") else c
            j = i + len(q)
            while j < n and code[j : j + len(q)] != q:
                j += 2 if code[j] == "\\" else 1
            out.append("S")
            i = j + len(q)
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _strip_return_annotation(code: str) -> str:
    idx = code.find("->")
    if idx < 0:
        return code
    return code[:idx] + (":" if code.rstrip().endswith(":") else "")


def _skip_annotation(code: str, start: int, depth: int) -> int:
    """Index where the annotation that starts after the colon at ``start - 1`` ends."""
    j, d = start, depth
    while j < len(code):
        ch = code[j]
        if ch in "([{":
            d += 1
        elif ch in ")]}":
            if d == depth:
                break
            d -= 1
        elif d == depth and ch == ",":
            break
        elif d == depth and ch == "=" and code[j + 1 : j + 2] != "=" and code[j - 1] not in "=!<>":
            break
        j += 1
    return j


def strip_annotations(code: str) -> str:
    """Remove parameter, variable and return annotations from one line (strings already ``S``)."""
    stripped = code.lstrip()
    is_def = bool(_DEF.match(stripped))
    if not is_def and _BLOCK_KEYWORD.match(stripped):
        return code
    code = _strip_return_annotation(code)
    target = 1 if is_def else 0
    out: list[str] = []
    i, depth = 0, 0
    while i < len(code):
        c = code[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif (
            c == ":"
            and depth == target
            and code[i + 1 : i + 2] != "="  # walrus
            and code[i + 1 :].strip()  # not a block colon
            and _NAME_BEFORE_COLON.search("".join(out))
            and not _LAMBDA.search("".join(out))  # lambda parameters have no annotations
        ):
            end = _skip_annotation(code, i + 1, depth)
            if is_def or _TYPE_LIKE.match(code[i + 1 : end]):
                i = end
                continue
        out.append(c)
        i += 1
    return "".join(out)


def normalise(line: str) -> str:
    code = _replace_strings(line)
    code = _CHECKER_COMMENT.sub("", code)
    code = strip_annotations(code)
    return re.sub(r"\s+", "", code).rstrip(",")


def _structural_trivial(stripped: str, in_import: bool, in_docstring: bool) -> bool:
    return (
        not stripped
        or in_import
        or in_docstring
        or stripped.startswith("@")
        or bool(_IMPORT.match(stripped))
    )


def _next_state(stripped: str, in_import: bool, in_docstring: bool) -> tuple[bool, bool]:
    """Track parenthesised imports and triple-quoted strings on the new-file side of a hunk."""
    if len(_TRIPLE.findall(stripped)) % 2 == 1:
        in_docstring = not in_docstring
    if in_import:
        in_import = ")" not in stripped
    elif _IMPORT.match(stripped) and "(" in stripped and ")" not in stripped:
        in_import = True
    return in_import, in_docstring


def is_trivial(line: str, removed_block: list[str], in_import: bool, in_docstring: bool) -> bool:
    stripped = line.strip()
    if _structural_trivial(stripped, in_import, in_docstring):
        return True
    code = _CHECKER_COMMENT.sub("", _replace_strings(line))
    if _BARE_ANNOTATION.match(code):  # a declaration with no value is only an annotation
        return True
    norm = normalise(line)
    if norm in ("", "S"):  # nothing left, or only a string literal
        return True
    removed = [normalise(r) for r in removed_block]
    return norm in removed or (bool(removed) and norm in "".join(removed))


def sweep_stats(diff_text: str, include: Callable[[str], bool]) -> SweepStats:
    """Count trivial and total added lines in the files of ``diff_text`` that ``include`` keeps."""
    trivial = total = 0
    for pf in parse_diff(diff_text):
        if pf.is_removed_file or not include(pf.path):
            continue
        for hunk in pf:
            in_import = in_docstring = False
            block: list[str] = []
            last = " "
            for line in hunk:
                text = line.value.rstrip("\r\n")
                if line.is_removed:
                    if last != "-":
                        block = []
                    block.append(text)
                    last = "-"
                    continue
                if line.is_added:
                    total += 1
                    trivial += is_trivial(text, block, in_import, in_docstring)
                    last = "+"
                else:
                    block, last = [], " "
                in_import, in_docstring = _next_state(text.strip(), in_import, in_docstring)
    return SweepStats(trivial, total)
