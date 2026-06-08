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
import json
from pathlib import Path

from generate_candidate_charts import plot_outputs


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "_candidates"
OUTPUT = ROOT / "_output"


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


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for old in OUTPUT.glob("*.png"):
        old.unlink()

    manifest = []
    for notebook in sorted(CANDIDATES.glob("*.ipynb")):
        code_cells, markdown_cells = validate_notebook(notebook)
        outputs = plot_outputs(notebook.stem, OUTPUT)
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

    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Processed {len(manifest)} example notebooks")
    print(f"Wrote plots to {OUTPUT}")


if __name__ == "__main__":
    main()
