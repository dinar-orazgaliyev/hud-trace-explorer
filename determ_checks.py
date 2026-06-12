#!/usr/bin/env python3
"""Deterministic prompt/grader consistency checks (task-type agnostic).

Walks task roots (directories containing prompt.md + tests/) and reports
mechanical mismatches between what hidden grader tests assert and what the
agent-visible prompt exposes. Agents do not write tests, but task authors
should still keep prompt specs and grader tests aligned in both directions.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

TASK_MARKERS = ("prompt.md", "tests")


def find_task_roots(search_roots: Iterable[Path]) -> list[Path]:
    """Return dirs that look like a coding task (prompt + tests/)."""
    seen: set[Path] = set()
    roots: list[Path] = []

    for base in search_roots:
        if not base.exists():
            continue
        for prompt in base.rglob("prompt.md"):
            task_dir = prompt.parent
            if not (task_dir / "tests").is_dir():
                continue
            resolved = task_dir.resolve()
            if resolved not in seen:
                seen.add(resolved)
                roots.append(task_dir)
    return sorted(roots)


# ---------------------------------------------------------------------------
# Findings model
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    check: str
    severity: str  # "error" | "warning" | "info"
    task: str
    message: str
    file: str = ""
    line: int = 0


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    def add(self, **kwargs) -> None:
        self.findings.append(Finding(**kwargs))

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    def exit_code(self) -> int:
        return 1 if self.errors else 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REGEX_METACHAR = re.compile(r"[.^$*+?{}[\]|\\()]")

# Explicit error-message pins in the prompt (strict — used for coverage checks)
_PROMPT_PINNED_ERROR_PATTERNS = [
    re.compile(r'with message\s+`"([^"]+)"`', re.I),
    re.compile(r'with message\s+"([^"]+)"', re.I),
    re.compile(r'message\s+`"([^"]+)"`', re.I),
    re.compile(r'Raises\s+\w+\s+with\s+message\s+"([^"]+)"', re.I),
]

# Broader literals for context in error_message_alignment diagnostics
_PROMPT_MSG_PATTERNS = [
    *_PROMPT_PINNED_ERROR_PATTERNS,
    re.compile(r'`"([^"]+)"`'),  # backtick-wrapped literals in spec
]

_MIN_PINNED_MSG_LEN = 12
_PLACEHOLDER_RE = re.compile(r"\{[^}]+\}")


def prompt_corpus(prompt_text: str) -> str:
    return prompt_text


def prompt_declared_messages(prompt_text: str) -> set[str]:
    msgs: set[str] = set()
    for pat in _PROMPT_MSG_PATTERNS:
        msgs.update(pat.findall(prompt_text))
    return msgs


def _is_pinned_error_message(msg: str) -> bool:
    """Filter out repr examples, short tokens, and template placeholders."""
    if len(msg) < _MIN_PINNED_MSG_LEN:
        return False
    if _PLACEHOLDER_RE.search(msg):
        return False
    if "%" in msg:
        return False
    if not re.search(r"[a-zA-Z]{4,}", msg):
        return False
    return True


def prompt_pinned_error_messages(prompt_text: str) -> set[str]:
    msgs: set[str] = set()
    for pat in _PROMPT_PINNED_ERROR_PATTERNS:
        for msg in pat.findall(prompt_text):
            if _is_pinned_error_message(msg):
                msgs.add(msg)
    return msgs


def _significant_literals(text: str, *, min_len: int = 6) -> list[str]:
    chunks = re.split(r"[%{\s]+", text)
    return [c for c in chunks if len(c) >= min_len and re.search(r"[a-zA-Z]", c)]


def test_pattern_covers_message(msg: str, pattern: str) -> tuple[bool, str]:
    """Return (covered, quality) where quality is exact/literal/substring/loose."""
    if not is_probably_regex(pattern):
        if pattern == msg:
            return True, "exact"
        if len(pattern) >= 8 and pattern in msg:
            return True, "literal"
        if msg in pattern:
            return True, "literal"
        if len(pattern) >= 6 and pattern in msg:
            return True, "substring"
        return False, ""

    pat_literals = literal_tokens_from_regex(pattern)
    for pl in pat_literals:
        if len(pl) >= 6 and pl in msg:
            return True, "literal"

    for ml in _significant_literals(msg):
        if ml in pattern:
            return True, "literal"
        if any(ml in pl for pl in pat_literals if len(pl) >= 4):
            return True, "substring"

    if pat_literals and all(len(pl) < 6 for pl in pat_literals):
        return True, "loose"
    if is_probably_regex(pattern) and not pat_literals:
        return False, ""

    return False, ""


def classify_prompt_message_coverage(msg: str, patterns: set[str]) -> str:
    """Return 'covered', 'loose', or 'uncovered'."""
    saw_loose = False
    for pat in patterns:
        covered, quality = test_pattern_covers_message(msg, pat)
        if not covered:
            continue
        if quality in ("exact", "literal", "substring"):
            return "covered"
        saw_loose = True
    return "loose" if saw_loose else "uncovered"


def is_probably_regex(pattern: str) -> bool:
    return bool(_REGEX_METACHAR.search(pattern))


def literal_tokens_from_regex(pattern: str) -> list[str]:
    """Best-effort: pull obvious literal chunks out of a regex pattern."""
    chunks = re.split(r"[.^$*+?{}[\]|\\()]+", pattern)
    return [c for c in chunks if len(c) >= 4]


# ---------------------------------------------------------------------------
# AST extraction from tests
# ---------------------------------------------------------------------------

@dataclass
class MessageAssertion:
    exc_type: str | None
    pattern: str
    source: str
    line: int
    synthetic: bool = False  # test raises inside the with-block itself


class TestMessageExtractor(ast.NodeVisitor):
    def __init__(self, source: str) -> None:
        self.source = source
        self.assertions: list[MessageAssertion] = []

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            call = item.context_expr
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "raises"
                and isinstance(func.value, ast.Name)
                and func.value.id == "pytest"
            ):
                continue

            exc_type = None
            if call.args:
                exc_type = _expr_name(call.args[0])
            pattern = None
            for kw in call.keywords:
                if kw.arg == "match" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    pattern = kw.value.value
            if pattern is None:
                continue

            synthetic = _with_block_synthetic_raise(node.body, exc_type)
            self.assertions.append(
                MessageAssertion(
                    exc_type=exc_type,
                    pattern=pattern,
                    source=self.source,
                    line=node.lineno,
                    synthetic=synthetic,
                )
            )
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        # assert "substring" in str(exc.value)
        if not isinstance(node.test, ast.Compare):
            return
        comp = node.test
        if len(comp.ops) != 1 or not isinstance(comp.ops[0], ast.In):
            return
        if not comp.comparators:
            return
        left = comp.left
        if not (isinstance(left, ast.Constant) and isinstance(left.value, str)):
            return
        if not _is_str_of_exception(comp.comparators[0]):
            return
        self.assertions.append(
            MessageAssertion(
                exc_type=None,
                pattern=left.value,
                source=self.source,
                line=node.lineno,
            )
        )


def _expr_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_str_of_exception(node: ast.expr) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "str"


def _with_block_synthetic_raise(body: list[ast.stmt], exc_type: str | None) -> bool:
    """Detect `with pytest.raises(Foo): raise Foo(...)` — class smoke test."""
    for stmt in body:
        if isinstance(stmt, ast.Raise):
            return True
    return False


def extract_message_assertions(py_file: Path) -> list[MessageAssertion]:
    source = py_file.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source, filename=str(py_file))
    visitor = TestMessageExtractor(source)
    visitor.visit(tree)
    return visitor.assertions


# ---------------------------------------------------------------------------
# Check 1: tested error substrings must appear in prompt
# ---------------------------------------------------------------------------

def check_error_messages_in_prompt(task_dir: Path, report: Report) -> None:
    prompt_path = task_dir / "prompt.md"
    prompt_text = prompt_path.read_text(encoding="utf-8", errors="replace")
    corpus = prompt_corpus(prompt_text)
    declared = prompt_declared_messages(prompt_text)

    for test_file in sorted((task_dir / "tests").glob("test_*.py")):
        for assertion in extract_message_assertions(test_file):
            if assertion.synthetic:
                report.add(
                    check="error_message_alignment",
                    severity="info",
                    task=str(task_dir),
                    file=str(test_file.relative_to(task_dir)),
                    line=assertion.line,
                    message=(
                        f"Skipping synthetic exception smoke test "
                        f"({assertion.exc_type!r}, match={assertion.pattern!r})"
                    ),
                )
                continue

            pattern = assertion.pattern
            if is_probably_regex(pattern):
                literals = literal_tokens_from_regex(pattern)
                if not literals:
                    report.add(
                        check="error_message_alignment",
                        severity="warning",
                        task=str(task_dir),
                        file=str(test_file.relative_to(task_dir)),
                        line=assertion.line,
                        message=(
                            f"Regex match={pattern!r} — cannot verify literally; "
                            "manual review or pin exact text in prompt"
                        ),
                    )
                    continue
                missing = [lit for lit in literals if lit not in corpus]
            else:
                missing = [] if pattern in corpus else [pattern]

            if missing:
                report.add(
                    check="error_message_alignment",
                    severity="error",
                    task=str(task_dir),
                    file=str(test_file.relative_to(task_dir)),
                    line=assertion.line,
                    message=(
                        f"Test pins error text {missing!r} "
                        f"({assertion.exc_type or 'Exception'}) but prompt.md does not contain it. "
                        f"Prompt-declared messages: {sorted(declared)!r}"
                    ),
                )


# ---------------------------------------------------------------------------
# Check 2: explicitly pinned prompt error messages should be tested
# ---------------------------------------------------------------------------

def check_prompt_messages_are_tested(task_dir: Path, report: Report) -> None:
    prompt_text = (task_dir / "prompt.md").read_text(encoding="utf-8", errors="replace")
    pinned = prompt_pinned_error_messages(prompt_text)
    if not pinned:
        return

    tested_patterns: set[str] = set()
    for test_file in (task_dir / "tests").glob("test_*.py"):
        for assertion in extract_message_assertions(test_file):
            if not assertion.synthetic:
                tested_patterns.add(assertion.pattern)

    for msg in sorted(pinned):
        status = classify_prompt_message_coverage(msg, tested_patterns)
        if status == "uncovered":
            report.add(
                check="prompt_message_coverage",
                severity="warning",
                task=str(task_dir),
                message=(
                    f"Prompt explicitly pins error message {msg!r} "
                    f"but no test uses a substantive pytest.raises(..., match=...) "
                    f"or substring assert for it"
                ),
            )
        elif status == "loose":
            report.add(
                check="prompt_message_coverage",
                severity="warning",
                task=str(task_dir),
                message=(
                    f"Prompt explicitly pins error message {msg!r} "
                    f"but test match patterns look too loose to enforce it "
                    f"(review grader strictness)"
                ),
            )


# ---------------------------------------------------------------------------
# Check 3: task.py grader wiring (task-agnostic)
# ---------------------------------------------------------------------------

_INJECT_RE = re.compile(r'_inject_and_run\(\s*[\'"]([^\'"]+)[\'"]\s*\)')
_WEIGHT_RE = re.compile(r'"weight"\s*:\s*([0-9.]+)')


def check_task_py_wiring(task_dir: Path, report: Report) -> None:
    task_py = task_dir / "task.py"
    if not task_py.exists():
        return

    text = task_py.read_text(encoding="utf-8", errors="replace")
    wired = set(_INJECT_RE.findall(text))
    on_disk = {p.name for p in (task_dir / "tests").glob("test_*.py")}

    for missing in sorted(on_disk - wired):
        report.add(
            check="task_py_coverage",
            severity="warning",
            task=str(task_dir),
            message=f"Test file {missing} exists but is not referenced in task.py bash_checks",
        )
    for orphan in sorted(wired - on_disk):
        report.add(
            check="task_py_coverage",
            severity="error",
            task=str(task_dir),
            message=f"task.py references {orphan} but file is missing from tests/",
        )

    weights = [float(w) for w in _WEIGHT_RE.findall(text)]
    if weights:
        total = round(sum(weights), 6)
        if abs(total - 1.0) > 1e-6:
            report.add(
                check="task_py_weights",
                severity="error",
                task=str(task_dir),
                message=f"bash_checks weights sum to {total}, expected 1.0",
            )


# ---------------------------------------------------------------------------
# Check 4: package name in prompt vs tests
# ---------------------------------------------------------------------------

def check_import_package_alignment(task_dir: Path, report: Report) -> None:
    prompt_text = (task_dir / "prompt.md").read_text(encoding="utf-8", errors="replace")
    # first path-like package dir mentioned in a code block or layout section
    pkg_match = re.search(r"(\w+)/\s*\n\s*__init__\.py", prompt_text)
    if not pkg_match:
        return
    expected_pkg = pkg_match.group(1)

    imported: set[str] = set()
    for test_file in (task_dir / "tests").glob("test_*.py"):
        tree = ast.parse(test_file.read_text(encoding="utf-8", errors="replace"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])

    if imported and expected_pkg not in imported:
        report.add(
            check="package_name_alignment",
            severity="warning",
            task=str(task_dir),
            message=(
                f"Prompt specifies package {expected_pkg!r} but tests import {sorted(imported)!r}"
            ),
        )


# ---------------------------------------------------------------------------
# Check 5: public re-exports in prompt should appear in grader tests
# ---------------------------------------------------------------------------

def check_prompt_public_api_smoke(task_dir: Path, report: Report) -> None:
    prompt_text = (task_dir / "prompt.md").read_text(encoding="utf-8", errors="replace")
    block = re.search(r"from \.[\s\S]+?from \.", prompt_text)
    if not block:
        return
    names = re.findall(r"^\s*(\w+)\s*,", block.group(0), re.M)
    names += re.findall(r"^\s*(\w+)\s*\)", block.group(0), re.M)
    names = [n for n in set(names) if len(n) >= 3 and not n.startswith("_")]
    if not names:
        return

    test_blob = "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in (task_dir / "tests").glob("test_*.py")
    )
    for name in sorted(names):
        if name not in test_blob:
            report.add(
                check="public_api_coverage",
                severity="info",
                task=str(task_dir),
                message=(
                    f"Prompt re-exports {name!r} but name never appears in grader tests "
                    f"(may be OK if tested indirectly)"
                ),
            )


# ---------------------------------------------------------------------------
# Runner / CLI
# ---------------------------------------------------------------------------

ALL_CHECKS = {
    "error_messages": check_error_messages_in_prompt,
    "prompt_coverage": check_prompt_messages_are_tested,
    "task_py": check_task_py_wiring,
    "package_name": check_import_package_alignment,
    "public_api": check_prompt_public_api_smoke,
}


def run_checks(
    roots: list[Path],
    *,
    checks: set[str] | None = None,
) -> Report:
    report = Report()
    selected = checks or set(ALL_CHECKS)
    task_roots = find_task_roots(roots)

    if not task_roots:
        report.add(
            check="discovery",
            severity="error",
            task=".",
            message=f"No task roots found under {roots}",
        )
        return report

    for task_dir in task_roots:
        for name in selected:
            ALL_CHECKS[name](task_dir, report)
    return report


def print_report(report: Report) -> None:
    by_task: dict[str, list[Finding]] = {}
    for f in report.findings:
        by_task.setdefault(f.task, []).append(f)

    for task, findings in sorted(by_task.items()):
        print(f"\n=== {task} ===")
        for f in findings:
            loc = f" ({f.file}:{f.line})" if f.file else ""
            print(f"  [{f.severity.upper()}] {f.check}{loc}: {f.message}")

    errs = len(report.errors)
    warns = sum(1 for f in report.findings if f.severity == "warning")
    print(f"\nSummary: {errs} error(s), {warns} warning(s), {len(report.findings)} total")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path.cwd()],
        help="Directories to search for tasks (default: cwd)",
    )
    parser.add_argument(
        "--check",
        action="append",
        choices=sorted(ALL_CHECKS),
        help="Run only selected checks (repeatable)",
    )
    args = parser.parse_args(argv)

    report = run_checks(args.roots, checks=set(args.check) if args.check else None)
    print_report(report)
    return report.exit_code()


if __name__ == "__main__":
    sys.exit(main())