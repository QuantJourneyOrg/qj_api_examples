#!/usr/bin/env python3
"""Reject credentials, local paths, and stored exceptions in public notebooks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PRIVATE_OUTPUT_PATTERNS = (
    "/Users/",
    "~/Library/",
    "Dropbox/QJ_Repo",
    "_repo_qj_",
    "C:\\Users\\",
)
SECRET_PATTERNS = (
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    re.compile(r"qj_[A-Za-z0-9_-]{24,}"),
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{24,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
)


def _source_lines(cell: dict[str, Any]) -> list[str]:
    source = cell.get("source", [])
    return source if isinstance(source, list) else source.splitlines(keepends=True)


def _safe_token_line(line: str) -> str:
    indent = line[: len(line) - len(line.lstrip())]
    if "tokens.access_token" in line and "print" in line:
        return f'{indent}print("  Access token received (value hidden)")\n'
    if "tokens.refresh_token" in line and "print" in line:
        return f'{indent}print("  Refresh token received (value hidden)")\n'
    return line


def inspect_notebook(path: Path, *, fix: bool) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid notebook JSON ({exc})"]
    issues: list[str] = []
    changed = False

    for cell_index, cell in enumerate(payload.get("cells", [])):
        lines = _source_lines(cell)
        safe_lines = [_safe_token_line(line) for line in lines]
        if safe_lines != lines:
            if fix:
                cell["source"] = safe_lines
                changed = True
            else:
                issues.append(f"cell {cell_index}: prints an access or refresh token")

        outputs = cell.get("outputs", [])
        safe_outputs = []
        for output_index, output in enumerate(outputs):
            if output.get("output_type") == "error":
                if fix:
                    changed = True
                    continue
                issues.append(f"cell {cell_index}, output {output_index}: stored exception")
            serialized = json.dumps(output, ensure_ascii=False)
            if any(marker in serialized for marker in PRIVATE_OUTPUT_PATTERNS):
                issues.append(f"cell {cell_index}, output {output_index}: private local path")
            if any(pattern.search(serialized) for pattern in SECRET_PATTERNS):
                issues.append(f"cell {cell_index}, output {output_index}: credential-like value")
            safe_outputs.append(output)
        if fix and safe_outputs != outputs:
            cell["outputs"] = safe_outputs

    if fix and changed:
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return inspect_notebook(path, fix=False)
    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    failures: list[str] = []
    for path in sorted(args.root.rglob("*.ipynb")):
        if ".ipynb_checkpoints" in path.parts:
            continue
        issues = inspect_notebook(path, fix=args.fix)
        failures.extend(f"{path.relative_to(args.root)}: {issue}" for issue in issues)

    if failures:
        print("\n".join(f"ERROR: {failure}" for failure in failures))
        return 1
    print("OK: public notebooks contain no stored exceptions, local paths, or credentials")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
