#!/usr/bin/env python3
"""Keep the README tier table equal to results/status-matrix.json.

Also runs a tiny self-test (`--selftest`) that flips a tier and requires the
comparator to notice. That self-test does not touch the working tree.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import status_matrix as sm  # noqa: E402

FORBIDDEN_README = (
    "production-grade",
    "real implementations end-to-end",
    "production-ready",
)


def _fail(messages: list[str]) -> int:
    print("drift check failed:", file=sys.stderr)
    for message in messages:
        print(f"  {message}", file=sys.stderr)
    return 1


def check_tree(root: Path) -> list[str]:
    problems: list[str] = []
    readme_path = root / "README.md"
    matrix_path = root / "results" / "status-matrix.json"
    md_path = root / "results" / "status-matrix.md"
    if not matrix_path.is_file():
        return ["results/status-matrix.json is missing"]
    readme = readme_path.read_text(encoding="utf-8")
    for phrase in FORBIDDEN_README:
        if phrase in readme:
            problems.append(f"README still contains {phrase!r}")
    if "]()" in readme:
        problems.append("README has an empty-link badge (]())")
    if "17 numbered projects" not in readme:
        problems.append("README is missing the phrase '17 numbered projects'")
    if "MMLU/GSM8K/HumanEval/MT-Bench" not in readme or "Wilson CI" not in readme or "McNemar" not in readme:
        problems.append("README is missing the project-20 benchmark phrase checked by eval/cases.jsonl")
    if "## FAILURE_MODES" not in readme:
        problems.append("README is missing '## FAILURE_MODES'")
    folded = readme.casefold()
    for heading in (
        "mock masking",
        "dependency rot",
        "archived code still importable",
    ):
        if heading not in folded:
            problems.append(f"README is missing {heading!r}")
    if "actions/workflows/ci.yml" not in readme:
        problems.append("README CI badge does not point at .github/workflows/ci.yml")
    if not (root / ".github" / "workflows" / "ci.yml").is_file():
        problems.append(".github/workflows/ci.yml is missing")

    active, archived = sm.discover(root)
    badge = "archived-" + str(len(archived))
    if badge not in readme:
        problems.append(f"README badge does not mention {badge} (archive dirs: {len(archived)})")
    for name in ("08-smart-customer-service", "10-inference-optimization", "13-paper-review-agent"):
        if not (root / "archive" / name).is_dir():
            problems.append(f"archive/{name} is missing")
    if not any(name.startswith("12-") for name in active):
        problems.append("directory 12-* is not an active project")
    if not any(name.startswith("16-") for name in active):
        problems.append("directory 16-* is not an active project")

    payload = sm.load_matrix(matrix_path)
    problems.extend(sm.validate_matrix(payload, root))

    try:
        readme_rows = sm.parse_table(sm.extract_readme_table(readme))
    except ValueError as exc:
        problems.append(str(exc))
        readme_rows = []
    matrix_rows = payload.get("projects") or []
    # README table has no exit_code/timed_out; compare the published columns.
    published = (
        "project",
        "tier",
        "run",
        "passed",
        "skipped",
        "failed",
        "errors",
        "needs_key",
        "mock_default",
        "notes",
    )
    if readme_rows:
        problems.extend(_diff_published(matrix_rows, readme_rows, published, "README"))

    if md_path.is_file():
        try:
            md_rows = sm.parse_table(sm.extract_readme_table("\n".join(_md_table(md_path))))
        except ValueError:
            # status-matrix.md has no HTML markers; parse the whole file.
            try:
                md_rows = sm.parse_table(md_path.read_text(encoding="utf-8"))
            except ValueError as exc:
                problems.append(f"status-matrix.md: {exc}")
                md_rows = []
        if md_rows:
            problems.extend(_diff_published(matrix_rows, md_rows, published, "status-matrix.md"))
    else:
        problems.append("results/status-matrix.md is missing")
    return problems


def _md_table(path: Path) -> list[str]:
    """Yield nothing special; caller falls through. Kept for clarity."""
    return path.read_text(encoding="utf-8").splitlines()


def _diff_published(matrix_rows, other_rows, fields, label: str) -> list[str]:
    diffs: list[str] = []
    if [r["project"] for r in matrix_rows] != [r["project"] for r in other_rows]:
        diffs.append(f"{label} project order != matrix")
        return diffs
    for left, right in zip(matrix_rows, other_rows):
        for key in fields:
            if left.get(key) != right.get(key):
                diffs.append(
                    f"{label} {left.get('project')} {key}: matrix={left.get(key)!r} doc={right.get(key)!r}"
                )
    return diffs


def selftest() -> int:
    """Prove the comparator goes red when a tier is edited, and that the
    tier function demotes mocks and failures."""
    failures: list[str] = []

    def expect(name: str, got: str, want: str) -> None:
        if got != want:
            failures.append(f"{name}: got {got}, want {want}")

    expect(
        "strong",
        sm.assign_tier(exit_code=0, passed=3, failed=0, errors=0, skipped=0, timed_out=False, mock_default=False),
        "Strong",
    )
    expect(
        "skips",
        sm.assign_tier(exit_code=0, passed=3, failed=0, errors=0, skipped=1, timed_out=False, mock_default=False),
        "Solid",
    )
    expect(
        "mock",
        sm.assign_tier(exit_code=0, passed=3, failed=0, errors=0, skipped=0, timed_out=False, mock_default=True),
        "Default-mock",
    )
    expect(
        "red",
        sm.assign_tier(exit_code=1, passed=3, failed=1, errors=0, skipped=0, timed_out=False, mock_default=False),
        "Unverified",
    )
    expect(
        "timeout",
        sm.assign_tier(exit_code=124, passed=0, failed=0, errors=0, skipped=0, timed_out=True, mock_default=False),
        "Unverified",
    )

    parsed = sm.parse_pytest_output("=== 4 passed, 2 skipped, 1 failed, 3 errors in 0.2s ===\n")
    if (parsed["passed"], parsed["skipped"], parsed["failed"], parsed["errors"]) != (4, 2, 1, 3):
        failures.append(f"parse_pytest_output: {parsed}")
    quiet = sm.parse_pytest_output("......\n67 passed in 0.23s\n")
    if quiet["passed"] != 67 or quiet["failed"] != 0:
        failures.append(f"quiet summary not parsed: {quiet}")

    rows = [
        {
            "project": "01-example",
            "tier": "Strong",
            "run": 2,
            "passed": 2,
            "skipped": 0,
            "failed": 0,
            "errors": 0,
            "needs_key": False,
            "mock_default": False,
            "exit_code": 0,
            "timed_out": False,
            "notes": "",
        }
    ]
    table = sm.render_table(rows)
    parsed_rows = sm.parse_table(table)
    if parsed_rows[0]["tier"] != "Strong" or parsed_rows[0]["passed"] != 2:
        failures.append(f"round-trip table failed: {parsed_rows}")
    tampered = table.replace("| Strong |", "| Solid |")
    tampered_rows = sm.parse_table(tampered)
    diffs = sm.diff_rows(rows, [{**tampered_rows[0], "exit_code": 0, "timed_out": False}])
    if not diffs:
        failures.append("tampered tier was not detected")

    readme = f"intro\n{sm.BEGIN}\n{table}\n{sm.END}\n"
    replaced = sm.replace_readme_table(readme, tampered)
    if "Solid" not in sm.extract_readme_table(replaced):
        failures.append("replace_readme_table did not write the new tier")

    if failures:
        return _fail(failures)
    print("drift selftest: tampered tier detected; tier rules hold")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    problems = check_tree(sm.repo_root())
    self_rc = selftest()
    if problems:
        return _fail(problems)
    return self_rc


if __name__ == "__main__":
    sys.exit(main())
