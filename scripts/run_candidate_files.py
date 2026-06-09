"""Run the example catalog artifact pipeline.

For every `_candidates/*.ipynb` file this script:
1. Loads and validates notebook JSON.
2. Parses every Python code cell.
3. Generates zero, one or more dark chart artifacts into `_output/`.
4. Writes `_output/manifest.json`.

Some examples intentionally have no chart output, and some have multiple
separate figures.
"""

from __future__ import annotations

import ast
import argparse
import json
from pathlib import Path

from generate_candidate_charts import plot_outputs


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "_candidates"
OUTPUT = ROOT / "_output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT), help="Directory for generated PNG files and manifest.")
    return parser.parse_args()


def validate_notebook(path: Path) -> tuple[int, int]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    code_cells = 0
    markdown_cells = 0
    for index, cell in enumerate(nb.get("cells", [])):
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", []))
        if cell_type == "code":
            ast.parse(source, filename=f"{path}:{index}")
            code_cells += 1
        elif cell_type == "markdown":
            markdown_cells += 1
    return code_cells, markdown_cells


def notebook_sort_key(path: Path) -> tuple[int, str]:
    prefix = path.stem.split("_", 1)[0]
    try:
        return int(prefix), path.stem
    except ValueError:
        return 10_000, path.stem


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.png"):
        old.unlink()

    manifest = []
    for notebook in sorted(CANDIDATES.glob("*.ipynb"), key=notebook_sort_key):
        code_cells, markdown_cells = validate_notebook(notebook)
        outputs = plot_outputs(notebook.stem, output_dir)
        manifest.append(
            {
                "notebook": str(notebook.relative_to(ROOT)),
                "outputs": [str(output.relative_to(ROOT)) for output in outputs],
                "output": str(outputs[0].relative_to(ROOT)) if outputs else None,
                "code_cells": code_cells,
                "markdown_cells": markdown_cells,
                "status": "ok",
            }
        )

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Processed {len(manifest)} example notebooks")
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
