"""Notebook-Zellen in separaten Python-Prozessen prüfen, optional Ausgaben speichern.

Aufruf im Projektordner: python scripts/run_notebooks.py [--write]
Die Notebooks erzeugen ihre dokumentierten CSV-Dateien und Grafiken neu.
Ein Jupyter-Server ist für diese Prüfung nicht erforderlich.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT = Path(__file__).resolve().parents[1]


def run_notebook(path: Path, write: bool) -> None:
    """Rich Outputs sammeln und bei der ersten fehlerhaften Zelle abbrechen."""
    notebook = nbformat.read(path, as_version=4)
    shell = InteractiveShell.instance()
    matplotlib.use("module://matplotlib_inline.backend_inline")
    execution_count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        execution_count += 1
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=False)
        result.raise_error()
        outputs = []
        for name in ["stdout", "stderr"]:
            text = getattr(captured, name)
            if text:
                outputs.append(nbformat.v4.new_output("stream", name=name, text=text))
                print(text, end="", flush=True)
        for output in captured.outputs:
            outputs.append(
                nbformat.v4.new_output(
                    "display_data",
                    data=output.data,
                    metadata=output.metadata,
                )
            )
        cell.outputs = outputs
        cell.execution_count = execution_count
    if write:
        nbformat.write(notebook, path)
    print(f"OK: {path.relative_to(ROOT)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Notebook-Ausgaben speichern")
    parser.add_argument("--notebook", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.notebook:
        run_notebook(arguments.notebook, arguments.write)
        return
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        command = [sys.executable, str(Path(__file__).resolve()), "--notebook", str(path)]
        if arguments.write:
            command.append("--write")
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
