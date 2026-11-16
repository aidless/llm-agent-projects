#!/usr/bin/env python3
"""Run each active project's tests offline and record the status matrix.

Usage:
  python3 scripts/offline_check.py --refresh   # write results/status-matrix.json and README table
  python3 scripts/offline_check.py --compare   # re-run; exit 1 if semantic rows differ

No API keys are forwarded to the child pytest processes. Mock-provider env
vars are set so projects that refuse to boot without an explicit mock flag
can be tested. A green row is that pytest process, not a live model call.
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import status_matrix as sm  # noqa: E402

PER_PROJECT_S = 120
PER_TEST_S = 15

# Packages this check refuses to treat as installed-by-default. Recorded in
# the matrix so a reader can see the boundary. Presence in the ambient
# interpreter is reported; it is not a failure by itself (row drift is).
ABSENT_BY_CONTRACT = (
    "torch",
    "sentence_transformers",
    "transformers",
    "peft",
    "spacy",
)

CHECK_ENV = {
    "ALLOW_MOCK_PROVIDERS": "true",
    "SKIP_HEAVY_TESTS": "1",
    "SKIP_HF_DOWNLOAD": "1",
    "GATEWAY_AUTH_DISABLED": "true",
    "GUARDRAILS_AUTH_DISABLED": "true",
    "SANDBOX_AUTH_DISABLED": "true",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "MPLBACKEND": "Agg",
    "PYTHONUNBUFFERED": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}

_SECRET_BITS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def _scrub(base: dict[str, str], project: Path) -> dict[str, str]:
    env = {}
    for key, value in base.items():
        upper = key.upper()
        if any(bit in upper for bit in _SECRET_BITS):
            continue
        env[key] = value
    env.pop("PYTEST_ADDOPTS", None)
    env.update(CHECK_ENV)
    env["PYTHONPATH"] = str(project)
    return env


def _run(cmd: list[str], cwd: Path, env: dict[str, str], timeout: int) -> tuple[int, str, bool]:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
        return proc.returncode if proc.returncode is not None else 1, out or "", False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, _ = proc.communicate()
        return 124, (out or "") + "\nTIMEOUT\n", True


def _probe_imports() -> dict[str, bool]:
    import importlib.util

    return {name: importlib.util.find_spec(name) is not None for name in ABSENT_BY_CONTRACT}


def _probe_archive(root: Path, name: str) -> str:
    target = root / "archive" / name
    code = (
        "import importlib, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "try:\n"
        "    importlib.import_module('app.main')\n"
        "    print('ok app.main')\n"
        "except Exception as exc:\n"
        "    print(type(exc).__name__ + ': ' + str(exc).splitlines()[0][:180])\n"
    )
    env = _scrub(os.environ, target)
    code_rc, out, timed = _run([sys.executable, "-c", code, str(target)], target, env, 30)
    line = (out or "").strip().splitlines()
    text = line[-1] if line else f"exit {code_rc}"
    if timed:
        text = "timeout"
    return text[:200]


def measure_project(root: Path, name: str) -> dict:
    project = root / name
    started = time.monotonic()
    mock, evidence, hits = sm.detect_mock_default(project)
    needs_key = sm.detect_needs_key(project)
    static = sm.static_notes(project)
    tests = project / "tests"
    if not tests.is_dir():
        duration = time.monotonic() - started
        measurement = {
            "project": name,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "exit_code": 5,
            "timed_out": False,
            "mock_default": mock,
            "needs_key": needs_key,
            "duration_seconds": duration,
            "notes": sm.build_notes(
                timed_out=False,
                per_project_s=PER_PROJECT_S,
                failed_nodes=[],
                error_nodes=[],
                mock_evidence=evidence,
                mock_hits=hits,
                static=static,
                extra="no tests/ directory",
            ),
        }
        return sm.row_from_measurement(measurement)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "-q",
        "--tb=line",
        f"--timeout={PER_TEST_S}",
        "--timeout-method=thread",
        "-p",
        "no:cacheprovider",
        "--override-ini",
        "addopts=",
    ]
    rc, out, timed = _run(cmd, project, _scrub(os.environ, project), PER_PROJECT_S)
    parsed = sm.parse_pytest_output(out)
    if not parsed["summary"] and rc not in (0,):
        # Collection crashes sometimes print no count line.
        if rc == 2 and parsed["errors"] == 0 and not parsed["error_nodes"]:
            parsed["errors"] = 1
        elif rc not in (0, 1, 5) and parsed["failed"] == 0 and parsed["errors"] == 0:
            parsed["errors"] = max(parsed["errors"], 1)
    extra_bits: list[str] = []
    missing = sorted(set(re.findall(r"No module named ['\"]([^'\"]+)['\"]", out)))
    if missing:
        extra_bits.append("missing=" + ",".join(missing[:8]))
    if rc == 5:
        extra_bits.append("no tests collected")
    elif not parsed["summary"] and rc != 0:
        extra_bits.append(f"no pytest summary (exit {rc})")
    extra = "; ".join(extra_bits)
    measurement = {
        "project": name,
        "passed": parsed["passed"],
        "failed": parsed["failed"],
        "skipped": parsed["skipped"],
        "errors": parsed["errors"],
        "exit_code": 124 if timed else int(rc),
        "timed_out": timed,
        "mock_default": mock,
        "needs_key": needs_key,
        "duration_seconds": time.monotonic() - started,
        "notes": sm.build_notes(
            timed_out=timed,
            per_project_s=PER_PROJECT_S,
            failed_nodes=parsed["failed_nodes"],
            error_nodes=parsed["error_nodes"],
            mock_evidence=evidence,
            mock_hits=hits,
            static=static,
            extra=extra,
        ),
    }
    return sm.row_from_measurement(measurement)


def build_payload(root: Path, rows: list[dict], started_at: datetime, duration_s: float) -> dict:
    _active, archived = sm.discover(root)
    return {
        "schema_version": 1,
        "generated_by": "scripts/offline_check.py --refresh",
        "command": "make check-refresh",
        "verified_by": "make check",
        "python": sys.version.split()[0],
        "recorded_at": started_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_seconds": round(duration_s, 3),
        "keys_used": False,
        "per_project_timeout_seconds": PER_PROJECT_S,
        "per_test_timeout_seconds": PER_TEST_S,
        "env": CHECK_ENV,
        "requirements_file": "requirements-check.txt",
        "excluded_stacks": list(ABSENT_BY_CONTRACT),
        "ambient_excluded_imports": _probe_imports(),
        "archived_import": [
            {"project": name, "import_app_main": _probe_archive(root, name)} for name in archived
        ],
        "tier_rules": {
            "Strong": "pytest exit 0, passed>0, failed=0, errors=0, skipped=0, mock_default=false. Offline suite only.",
            "Solid": "pytest exit 0, passed>0, failed=0, errors=0, skipped>0, mock_default=false.",
            "Default-mock": "pytest exit 0 and non-test code defaults to a mock/keyword stand-in.",
            "Unverified": "suite did not exit 0, timed out, or collected nothing. Not Strong.",
        },
        "totals": sm.totals(rows),
        "projects": rows,
    }


def write_markdown(path: Path, payload: dict) -> None:
    rows = payload["projects"]
    archived = payload.get("archived_import") or []
    lines = [
        "# Offline status matrix",
        "",
        "Generated by `make check-refresh`. `make check` re-runs the same pytest",
        "commands and exits 1 if any semantic column changes. `duration_seconds`",
        "is the refresh run that wrote this file; a later check does not rewrite it.",
        "",
        f"- recorded_at: `{payload['recorded_at']}`",
        f"- python: `{payload['python']}`",
        f"- duration_seconds: `{payload['duration_seconds']}`",
        "- keys_used: `false`",
        f"- requirements: `{payload['requirements_file']}`",
        f"- excluded stacks: {', '.join(payload['excluded_stacks'])}",
        "",
        sm.render_table(rows),
        "",
        "## Archived import probe",
        "",
        "Each line is `import app.main` with that archived directory on `sys.path`.",
        "`ok app.main` means the archived tree is still importable in this environment.",
        "",
    ]
    for item in archived:
        lines.append(f"- `{item['project']}`: {item['import_app_main']}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def refresh(root: Path) -> int:
    active, _archived = sm.discover(root)
    if len(active) < 1:
        print("no active projects discovered", file=sys.stderr)
        return 2
    print(f"offline check: {len(active)} active projects", file=sys.stderr)
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    rows = []
    for name in active:
        print(f"-- {name}", file=sys.stderr, flush=True)
        row = measure_project(root, name)
        rows.append(row)
        print(
            f"   {row['tier']} passed={row['passed']} skipped={row['skipped']} "
            f"failed={row['failed']} errors={row['errors']} exit={row['exit_code']}",
            file=sys.stderr,
            flush=True,
        )
    payload = build_payload(root, rows, started, time.monotonic() - t0)
    problems = sm.validate_matrix(payload, root)
    matrix_path = root / "results" / "status-matrix.json"
    md_path = root / "results" / "status-matrix.md"
    readme_path = root / "README.md"
    sm.dump_matrix(matrix_path, payload)
    write_markdown(md_path, payload)
    readme = readme_path.read_text(encoding="utf-8")
    readme_path.write_text(
        sm.replace_readme_table(readme, sm.render_table(rows)),
        encoding="utf-8",
    )
    print(f"wrote {matrix_path.relative_to(root)} duration_seconds={payload['duration_seconds']}")
    if problems:
        print("matrix failed structural checks:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    return 0


def compare(root: Path) -> int:
    matrix_path = root / "results" / "status-matrix.json"
    if not matrix_path.is_file():
        print("results/status-matrix.json is missing; run make check-refresh", file=sys.stderr)
        return 2
    committed = sm.load_matrix(matrix_path)
    active, _archived = sm.discover(root)
    print(f"offline check (compare): {len(active)} active projects", file=sys.stderr)
    live_rows = []
    for name in active:
        print(f"-- {name}", file=sys.stderr, flush=True)
        row = measure_project(root, name)
        live_rows.append(row)
        print(
            f"   {row['tier']} passed={row['passed']} skipped={row['skipped']} "
            f"failed={row['failed']} errors={row['errors']} exit={row['exit_code']}",
            file=sys.stderr,
            flush=True,
        )
    diffs = sm.diff_rows(committed.get("projects") or [], live_rows)
    observed = root / "results" / "status-matrix.observed.json"
    if diffs:
        started = datetime.now(timezone.utc)
        payload = build_payload(root, live_rows, started, 0)
        sm.dump_matrix(observed, payload)
        print("status matrix drifted from the committed file:", file=sys.stderr)
        for line in diffs:
            print(f"  {line}", file=sys.stderr)
        print(f"live rows written to {observed}", file=sys.stderr)
        return 1
    problems = sm.validate_matrix(committed, root)
    if problems:
        print("committed matrix failed structural checks:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    if observed.exists():
        observed.unlink()
    print("offline counts match results/status-matrix.json")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--refresh", action="store_true")
    group.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    root = sm.repo_root()
    try:
        import pytest  # noqa: F401
    except ImportError:
        print(
            "pytest is not installed. From a fresh clone:\n"
            "  python3 -m venv .venv && . .venv/bin/activate\n"
            "  make check-install\n"
            "  make check",
            file=sys.stderr,
        )
        return 2
    if args.refresh:
        return refresh(root)
    return compare(root)


if __name__ == "__main__":
    sys.exit(main())
