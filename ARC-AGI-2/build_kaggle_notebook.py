#!/usr/bin/env python3
"""Build a self-contained, offline Kaggle submission notebook for ARC-AGI-2.

The notebook embeds pipeline.py and every verified program in
gemini/results/programs/, so it needs no internet and no attached datasets
beyond the competition data. It finds the test challenges file under
/kaggle/input, runs the solver, and writes /kaggle/working/submission.json.

Usage:  python build_kaggle_notebook.py   ->  kaggle_submission.ipynb
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROG_DIR = os.path.join(HERE, "gemini", "results", "programs")


def main():
    with open(os.path.join(HERE, "pipeline.py")) as f:
        pipeline_src = f.read()
    programs = {}
    for fn in sorted(os.listdir(PROG_DIR)):
        if fn.endswith(".py"):
            with open(os.path.join(PROG_DIR, fn)) as f:
                programs[fn] = f.read()

    cells = []

    def md(s):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(True)})

    def code(s):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": s.strip().splitlines(True)})

    md(f"""
# ARC-AGI-2 offline solver

DSL program search plus a library of {len(programs)} verified programs. Runs fully offline.
Built from https://github.com/algilani2008/Kaggle-Quantum-AI (`ARC-AGI-2/`).
""")
    code("import json, os\nPIPELINE_SRC = " + json.dumps(pipeline_src) + "\n"
         "PROGRAMS = " + json.dumps(programs) + "\n"
         "os.makedirs('/kaggle/working/arc/library', exist_ok=True)\n"
         "open('/kaggle/working/arc/pipeline.py', 'w').write(PIPELINE_SRC)\n"
         "for name, src in PROGRAMS.items():\n"
         "    open(os.path.join('/kaggle/working/arc/library', name), 'w').write(src)\n"
         "print(len(PROGRAMS), 'library programs written')")
    code("""
import glob
cands = sorted(glob.glob('/kaggle/input/**/*test_challenges*.json', recursive=True))
assert cands, 'test challenges file not found under /kaggle/input'
CHALLENGES = cands[0]
print('Using', CHALLENGES)
""")
    code("""
!python /kaggle/working/arc/pipeline.py --challenges "$CHALLENGES" --out-dir /tmp/arc_out \\
    --time-per-task 8 --program-library /kaggle/working/arc/library
""")
    code("""
import shutil
shutil.copy('/tmp/arc_out/submission.json', '/kaggle/working/submission.json')
sub = json.load(open('/kaggle/working/submission.json'))
print('submission.json written for', len(sub), 'tasks')
""")

    nb = {"nbformat": 4, "nbformat_minor": 4, "cells": cells,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}}}
    out = os.path.join(HERE, "kaggle_submission.ipynb")
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {out} with {len(programs)} programs")


if __name__ == "__main__":
    main()
