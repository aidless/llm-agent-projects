#!/usr/bin/env python3
"""Pure helpers for the collection offline matrix.

No network. No pytest. `offline_check.py` runs the suites; `drift_check.py`
checks that README and `results/status-matrix.json` describe the same rows.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

BEGIN = "<!-- STATUS-MATRIX:BEGIN -->"
END = "<!-- STATUS-MATRIX:END -->"

HEADER = (
    "| project | tier | run | passed | skipped | failed | errors "
    "| needs-key | mock-default | notes |"
)
SEPARATOR = "|---|---|---:|---:|---:|---:|---:|---|---|---|"

TIERS = ("Strong", "Solid", "Default-mock", "Unverified")

# Compared on every `make check`. Wall-clock and timestamps are not compared.
SEMANTIC_FIELDS = (
    "project",
    "tier",
    "run",
    "passed",
    "skipped",
    "failed",
    "errors",
    "needs_key",
    "mock_default",
    "exit_code",
    "timed_out",
    "notes",
)

_COUNT_RE = re.compile(
    r"(\d+)\s+(passed|failed|skipped|errors|error|xfailed|xpassed)\b"
)
_DIR_RE = re.compile(r"^(\d{2})-[a-z0-9-]+$")

# Default-path mocks in non-test code. A hit forces tier Default-mock when
# the suite is otherwise green. Evidence is the first match (path:line).
_MOCK_RES = (
    re.compile(r"""getenv\(\s*['\"]VISION_PROVIDER['\"]\s*,\s*['\"]mock['\"]"""),
    re.compile(r"""getenv\(\s*['\"]ASR_PROVIDER['\"]\s*,\s*['\"]mock['\"]"""),
    re.compile(r"""getenv\(\s*['\"]TTS_PROVIDER['\"]\s*,\s*['\"]mock['\"]"""),
    re.compile(r"""getenv\(\s*['\"]CONSOLIDATOR_PROVIDER['\"]\s*,\s*['\"]keyword['\"]"""),
    re.compile(r"\[LLM Mock\]"),
    re.compile(r"=\s*MockLLMClient\("),
    re.compile(r"or\s+Mock(?:LLM|Vision|VQA|Chart)Client\("),
    re.compile(r"=\s*MockLLM(?:Extractor|Consolidator)\("),
    re.compile(r"return\s+MockLLMConsolidator\("),
    re.compile(r"return\s+Mock(?:ASR|TTS)\("),
)

_SKIP_PARTS = {"tests", "__pycache__", ".venv", "venv", "node_modules", ".git"}


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def discover(root: Path) -> tuple[list[str], list[str]]:
    """Return (active project dir names, archived project dir names), sorted."""
    active = sorted(
        p.name for p in root.iterdir() if p.is_dir() and _DIR_RE.match(p.name)
    )
    archive = root / "archive"
    archived: list[str] = []
    if archive.is_dir():
        archived = sorted(
            p.name
            for p in archive.iterdir()
            if p.is_dir() and _DIR_RE.match(p.name)
        )
    return active, archived


def assign_tier(
    *,
    exit_code: int,
    passed: int,
    failed: int,
    errors: int,
    skipped: int,
    timed_out: bool,
    mock_default: bool,
) -> str:
    """Map one measured run to a tier.

    Strong: the whole `tests/` suite exited 0 with zero skips and the
    default path scan found no mock stand-in.
    Solid: exited 0 with at least one skip (heavy, HF, or sample data).
    Default-mock: exited 0 and non-test code still defaults to a mock.
    Unverified: the suite did not exit 0. Never Strong.
    """
    green = (
        exit_code == 0
        and not timed_out
        and failed == 0
        and errors == 0
        and passed > 0
    )
    if not green:
        return "Unverified"
    if mock_default:
        return "Default-mock"
    if skipped > 0:
        return "Solid"
    return "Strong"


def parse_pytest_output(text: str) -> dict:
    """Pull counts from pytest's summary line. Missing counts stay 0.

    Only banner lines (they start with ``=``) are trusted. A traceback can
    contain the words "failed in" and must not become the summary.
    """
    summary = ""
    fallback = ""
    for line in text.splitlines():
        stripped = line.strip()
        interesting = any(
            token in stripped
            for token in (" passed", " failed", " error", " skipped", " no tests ran")
        )
        if not interesting or " in " not in stripped:
            continue
        # `-q` prints `67 passed in 0.23s` with no banner. A banner is preferred
        # when pytest emits one, because tracebacks can contain the same words.
        if stripped.startswith("="):
            summary = stripped
        elif _COUNT_RE.search(stripped):
            fallback = stripped
    if not summary:
        summary = fallback
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for match in _COUNT_RE.finditer(summary):
        kind = match.group(2)
        if kind == "error":
            kind = "errors"
        if kind in counts:
            counts[kind] = int(match.group(1))
    failed_nodes = re.findall(r"^FAILED\s+(\S+)", text, re.M)
    error_nodes = re.findall(r"^ERROR\s+(\S+)", text, re.M)
    return {
        "summary": summary.strip(),
        "failed_nodes": failed_nodes,
        "error_nodes": error_nodes,
        **counts,
    }


def _iter_py(project: Path) -> Iterable[Path]:
    for path in project.rglob("*.py"):
        if _SKIP_PARTS.intersection(path.parts):
            continue
        yield path


def detect_mock_default(project: Path) -> tuple[bool, str, int]:
    """Return (is_mock, first evidence 'rel:line', hit count)."""
    hits: list[str] = []
    for path in sorted(_iter_py(project), key=lambda item: item.as_posix()):
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        rel = path.relative_to(project).as_posix()
        for lineno, line in enumerate(lines, 1):
            if any(pattern.search(line) for pattern in _MOCK_RES):
                hits.append(f"{rel}:{lineno}")
    if not hits:
        return False, "", 0
    return True, hits[0], len(hits)


def detect_needs_key(project: Path) -> bool:
    """True when non-test code mentions an API key setting."""
    for path in _iter_py(project):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if re.search(r"API_KEY|api_key", text):
            return True
    return False


def static_notes(project: Path) -> list[str]:
    """Notes that describe this check's contract, not a live measurement."""
    notes: list[str] = []
    extractor = project / "kg" / "extractor.py"
    if extractor.is_file():
        text = extractor.read_text(encoding="utf-8", errors="ignore")
        if re.search(r'provider:\s*str\s*=\s*"regex"', text):
            notes.append("default NER=regex; spaCy excluded by requirements-check.txt")
    bert = project / "metrics" / "bertscore.py"
    if bert.is_file() and "NOT the original" in bert.read_text(encoding="utf-8", errors="ignore"):
        notes.append("semantic_similarity is not Zhang BERTScore; SKIP_HF_DOWNLOAD=1")
    # SKIP_HEAVY_TESTS is part of the check env for every project. Mention it
    # only where the suite actually looks at the variable.
    for path in _iter_py(project):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "SKIP_HEAVY_TESTS" in text:
            notes.append("SKIP_HEAVY_TESTS=1; torch/peft excluded by requirements-check.txt")
            break
    return notes


def build_notes(
    *,
    timed_out: bool,
    per_project_s: int,
    failed_nodes: list[str],
    error_nodes: list[str],
    mock_evidence: str,
    mock_hits: int,
    static: list[str],
    extra: str = "",
) -> str:
    bits: list[str] = []
    if timed_out:
        bits.append(f"killed after {per_project_s}s")
    if failed_nodes:
        bits.append("failed_node=" + failed_nodes[0])
    elif error_nodes:
        bits.append("error_node=" + error_nodes[0])
    if mock_evidence:
        more = f" (+{mock_hits - 1})" if mock_hits > 1 else ""
        bits.append("mock:" + mock_evidence + more)
    bits.extend(static)
    if extra:
        bits.append(extra)
    note = "; ".join(bits).replace("|", "/").replace("\n", " ")
    return note[:220]


def row_from_measurement(measurement: dict) -> dict:
    passed = int(measurement["passed"])
    failed = int(measurement["failed"])
    skipped = int(measurement["skipped"])
    errors = int(measurement["errors"])
    row = {
        "project": measurement["project"],
        "tier": assign_tier(
            exit_code=int(measurement["exit_code"]),
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            timed_out=bool(measurement["timed_out"]),
            mock_default=bool(measurement["mock_default"]),
        ),
        "run": passed + failed + skipped + errors,
        "passed": passed,
        "skipped": skipped,
        "failed": failed,
        "errors": errors,
        "needs_key": bool(measurement["needs_key"]),
        "mock_default": bool(measurement["mock_default"]),
        "exit_code": int(measurement["exit_code"]),
        "timed_out": bool(measurement["timed_out"]),
        "notes": measurement.get("notes") or "",
        "duration_seconds": round(float(measurement.get("duration_seconds") or 0), 3),
    }
    return row


def render_table(rows: list[dict]) -> str:
    lines = [HEADER, SEPARATOR]
    for row in rows:
        lines.append(
            "| {project} | {tier} | {run} | {passed} | {skipped} | {failed} "
            "| {errors} | {needs} | {mock} | {notes} |".format(
                project=row["project"],
                tier=row["tier"],
                run=row["run"],
                passed=row["passed"],
                skipped=row["skipped"],
                failed=row["failed"],
                errors=row["errors"],
                needs="yes" if row["needs_key"] else "no",
                mock="yes" if row["mock_default"] else "no",
                notes=row.get("notes") or "",
            )
        )
    return "\n".join(lines)


def parse_table(table: str) -> list[dict]:
    rows: list[dict] = []
    for raw in table.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells or cells[0] in {"project", "---"} or set(cells[0]) <= {"-", ":"}:
            continue
        if len(cells) != 10:
            raise ValueError(f"expected 10 cells, got {len(cells)}: {line}")
        project, tier, run, passed, skipped, failed, errors, needs, mock, notes = cells
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r} for {project}")
        if needs not in {"yes", "no"} or mock not in {"yes", "no"}:
            raise ValueError(f"needs-key/mock-default must be yes or no: {line}")
        rows.append(
            {
                "project": project,
                "tier": tier,
                "run": int(run),
                "passed": int(passed),
                "skipped": int(skipped),
                "failed": int(failed),
                "errors": int(errors),
                "needs_key": needs == "yes",
                "mock_default": mock == "yes",
                "notes": notes,
            }
        )
    return rows


def extract_readme_table(readme: str) -> str:
    if BEGIN not in readme or END not in readme:
        raise ValueError("README is missing STATUS-MATRIX markers")
    pre, rest = readme.split(BEGIN, 1)
    body, _post = rest.split(END, 1)
    if BEGIN in _post or END in body:
        raise ValueError("README has more than one STATUS-MATRIX block")
    del pre
    return body.strip()


def replace_readme_table(readme: str, table: str) -> str:
    if BEGIN not in readme or END not in readme:
        raise ValueError("README is missing STATUS-MATRIX markers")
    pre, rest = readme.split(BEGIN, 1)
    _body, post = rest.split(END, 1)
    return pre + BEGIN + "\n" + table.strip() + "\n" + END + post


def totals(rows: list[dict]) -> dict:
    by_tier = {tier: 0 for tier in TIERS}
    acc = {"projects": len(rows), "run": 0, "passed": 0, "skipped": 0, "failed": 0, "errors": 0}
    for row in rows:
        by_tier[row["tier"]] = by_tier.get(row["tier"], 0) + 1
        for key in ("run", "passed", "skipped", "failed", "errors"):
            acc[key] += int(row[key])
    acc["by_tier"] = by_tier
    return acc


def semantic(row: dict) -> dict:
    return {key: row[key] for key in SEMANTIC_FIELDS if key in row}


def diff_rows(committed: list[dict], live: list[dict]) -> list[str]:
    """Human-readable differences. Empty list means the semantic rows match."""
    diffs: list[str] = []
    c_map = {row["project"]: row for row in committed}
    l_map = {row["project"]: row for row in live}
    for name in sorted(set(c_map) | set(l_map)):
        if name not in c_map:
            diffs.append(f"only in live run: {name}")
            continue
        if name not in l_map:
            diffs.append(f"missing from live run: {name}")
            continue
        a, b = semantic(c_map[name]), semantic(l_map[name])
        # Live rows include exit_code and timed_out; README rows do not.
        # Compare the intersection, then the runner-only fields when both have them.
        keys = [k for k in SEMANTIC_FIELDS if k in a and k in b]
        for key in keys:
            if a[key] != b[key]:
                diffs.append(f"{name} {key}: committed={a[key]!r} live={b[key]!r}")
    return diffs


def validate_matrix(payload: dict, root: Path) -> list[str]:
    """Structural honesty checks. Returns a list of problems."""
    problems: list[str] = []
    active, archived = discover(root)
    rows = payload.get("projects") or []
    names = [row.get("project") for row in rows]
    if names != active:
        problems.append(f"matrix projects {names} != active dirs {active}")
    if len(names) != len(set(names)):
        problems.append("duplicate project rows")
    arch_ids = {name[:2] for name in archived}
    act_ids = {name[:2] for name in active}
    if not act_ids.isdisjoint(arch_ids):
        problems.append(f"id overlap between active and archive: {act_ids & arch_ids}")
    for needle in ("12-", "16-"):
        if not any(name.startswith(needle) for name in names):
            problems.append(f"active matrix missing {needle}*")
    strong_ok = True
    for row in rows:
        tier = row.get("tier")
        if tier not in TIERS:
            problems.append(f"{row.get('project')} has unknown tier {tier}")
            continue
        if tier == "Strong":
            if not (
                row.get("exit_code") == 0
                and not row.get("timed_out")
                and row.get("failed") == 0
                and row.get("errors") == 0
                and row.get("skipped") == 0
                and int(row.get("passed") or 0) > 0
                and not row.get("mock_default")
            ):
                strong_ok = False
                problems.append(f"Strong is not backed for {row.get('project')}: {semantic(row)}")
        expected = assign_tier(
            exit_code=int(row.get("exit_code", 1)),
            passed=int(row.get("passed") or 0),
            failed=int(row.get("failed") or 0),
            errors=int(row.get("errors") or 0),
            skipped=int(row.get("skipped") or 0),
            timed_out=bool(row.get("timed_out")),
            mock_default=bool(row.get("mock_default")),
        )
        if tier != expected:
            problems.append(f"{row.get('project')} tier {tier} != rule {expected}")
        run = int(row.get("passed") or 0) + int(row.get("failed") or 0) + int(row.get("skipped") or 0) + int(row.get("errors") or 0)
        if int(row.get("run") or 0) != run:
            problems.append(f"{row.get('project')} run column {row.get('run')} != {run}")
    del strong_ok
    claimed = payload.get("totals")
    if claimed is not None:
        fresh = totals(rows)
        if claimed != fresh:
            problems.append(f"totals {claimed} != recomputed {fresh}")
    return problems


def load_matrix(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_matrix(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
