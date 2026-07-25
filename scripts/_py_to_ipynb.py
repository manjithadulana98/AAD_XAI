"""Convert notebooks/kaggle_run_xai_aadnet.py (jupytext-style # %% cells) to .ipynb.

Standalone: parses the .py without needing jupytext installed.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import nbformat


CELL_MARKER = re.compile(r"^# %%(?:\s+\[(?P<kind>[a-zA-Z]+)\])?\s*$")


def parse_py_notebook(py_text: str) -> list[dict]:
    lines = py_text.splitlines()
    cells = []
    cur_kind = None
    cur_lines: list[str] = []

    def flush():
        if cur_kind is None:
            return
        src_lines = cur_lines[:]
        # trim trailing blank lines
        while src_lines and src_lines[-1].strip() == "":
            src_lines.pop()
        if not src_lines:
            return
        if cur_kind == "markdown":
            # strip leading '# ' from each line
            stripped = []
            for ln in src_lines:
                if ln.startswith("# "):
                    stripped.append(ln[2:])
                elif ln == "#":
                    stripped.append("")
                else:
                    stripped.append(ln)
            cells.append(nbformat.v4.new_markdown_cell("\n".join(stripped)))
        else:
            cells.append(nbformat.v4.new_code_cell("\n".join(src_lines)))

    for ln in lines:
        m = CELL_MARKER.match(ln)
        if m:
            flush()
            cur_kind = "markdown" if m.group("kind") == "markdown" else "code"
            cur_lines = []
            continue
        cur_lines.append(ln)
    flush()
    return cells


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: py_to_ipynb.py <input.py> <output.ipynb>")
        return 2
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    py_text = src.read_text(encoding="utf-8")
    cells = parse_py_notebook(py_text)
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
            "mimetype": "text/x-python",
            "file_extension": ".py",
            "pygments_lexer": "ipython3",
            "codemirror_mode": {"name": "ipython", "version": 3},
        },
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [
                {"datasetId": 0, "sourceId": 0, "sourceType": "datasetVersion"}
            ],
            "dockerImageVersionId": 30000,
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
        },
    }
    nbformat.write(nb, dst)
    n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"Wrote {dst}  ({n_md} markdown, {n_code} code cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
