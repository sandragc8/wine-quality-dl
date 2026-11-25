"""Convert a '# %%' script into a notebook: '# %% [markdown]' cells have every line starting with '# '."""
import sys

import nbformat

src, dst = sys.argv[1], sys.argv[2]
cells, kind, lines = [], None, []


def flush():
    if kind is None:
        return
    text = "\n".join(lines).strip("\n")
    if kind == "markdown":
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.split("\n"))
        cells.append(nbformat.v4.new_markdown_cell(text))
    else:
        cells.append(nbformat.v4.new_code_cell(text))


for line in open(src).read().split("\n"):
    if line.startswith("# %%"):
        flush()
        kind, lines = ("markdown" if "[markdown]" in line else "code"), []
    else:
        lines.append(line)
flush()

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11.4"}}
nbformat.write(nb, dst)
print("written", dst, len(cells), "cells")
