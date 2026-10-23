# ARC-AGI-2 × Gemini program search

Gemini writes a Python `transform()` for each ARC-AGI-2 task. A program is kept only if it reproduces **every** train output exactly. If it fails, Gemini is shown the failing example and asked to fix it.

> **Not submittable to Kaggle.** ARC Prize evaluations run with no internet and forbid API-based systems. The verified programs here are research material: port the useful transformations into the offline DSL solver (`../pipeline.py`) by hand.

## Files

| File | Purpose |
|---|---|
| `program_search.py` | the search loop: prompt, sandboxed execution, verification, feedback rounds, resume, scoring |
| `ARC_AGI2_Gemini.ipynb` | Colab notebook that runs it and saves results to Drive and GitHub |
| `results/` | written by the notebook: `programs/<task_id>.py`, `state.json`, `report.md` |

## Run in Colab

1. Open `ARC_AGI2_Gemini.ipynb` in Colab (from Drive: `Kaggle_Projects/Kaggle_Quantum_Data/`, or from GitHub).
2. In **Secrets** (key icon), add `GEMINI_API_KEY` and `GITHUB_TOKEN` (fine-grained, *Contents: read and write* on this repo only) and enable notebook access.
3. Adjust the settings cell if needed, then **Runtime → Run all**.

Each run attempts `MAX_TASKS` new tasks and stops early at `MAX_CALLS`, so cost per run is bounded. Progress is kept in `state.json` on Drive, so the next run continues where the last one stopped.

Results go to:
- Drive: `Kaggle_Projects/Kaggle_Quantum_Data/ARC-AGI-2_gemini/` (plus a dated copy of each report in `reports/`)
- GitHub: `ARC-AGI-2/gemini/results/`

## Safety

- The API key is read from the environment and sent in a request header, never in the URL; it is not printed or saved.
- The GitHub token is used as an in-memory git header; nothing is written to `.git/config`.
- Generated code runs in a separate `python -I` process with a 10 s timeout, a 1 GB memory cap, a temporary working directory and an empty environment (no API key).

## Local test without an API key

```bash
python program_search.py --tasks-dir path/to/ARC-AGI-2/data/training --out-dir out --mock --max-tasks 5
```
