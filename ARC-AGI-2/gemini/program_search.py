#!/usr/bin/env python3
"""
ARC-AGI-2 program synthesis with Gemini
=======================================

For each ARC task, Gemini is asked to write a Python function

    def transform(grid: list[list[int]]) -> list[list[int]]

The function is run in a separate, time- and memory-limited process on every
train input. It is accepted only if it reproduces every train output exactly.
If it fails, Gemini gets the failing pair (expected vs. actual) and tries
again, up to --rounds times per sample. Accepted programs are applied to the
test inputs; when the tasks include test outputs (public evaluation set), the
predictions are scored.

IMPORTANT: ARC Prize Kaggle evaluations have no internet access and forbid
API-based systems, so nothing produced here can be submitted to Kaggle. The
verified programs are research material: they show which transformations a
strong model finds, which can then be hand-ported into the offline DSL
(../pipeline.py).

The API key is read from the GEMINI_API_KEY environment variable and is never
printed, logged or written to disk.

Outputs (in --out-dir)
---------------------
- programs/<task_id>.py   every verified program, with a header comment
- state.json              per-task status, used to resume and skip work
- report.md / report.json run summary: verified, test-correct, calls, tokens

Usage
-----
  export GEMINI_API_KEY=...            # Colab: from userdata, never hard-coded
  python program_search.py --tasks-dir ARC-AGI-2/data/evaluation --out-dir out \
      --max-tasks 20 --samples 2 --rounds 2
  python program_search.py ... --mock   # offline test of the whole loop
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

VERSION = "1.0"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

PROMPT_HEADER = """You are solving an ARC-AGI puzzle. Each grid is a list of rows; each cell is an integer colour 0-9.
Study the training examples, find the single rule that maps every input to its output, and write it as Python.

Requirements:
- Define exactly one function: def transform(grid: list[list[int]]) -> list[list[int]]
- Use only the Python standard library and numpy (import numpy as np is allowed).
- The function must work for every training input and for new inputs of other sizes.
- Do not hard-code the training outputs.
- Reply with a short explanation of the rule, then one ```python code block containing the full function.
"""


# --------------------------------------------------------------------------- #
# Prompting
# --------------------------------------------------------------------------- #
def grid_text(g) -> str:
    return "\n".join(" ".join(str(v) for v in row) for row in g)


def task_prompt(task) -> str:
    parts = [PROMPT_HEADER]
    for n, p in enumerate(task["train"], 1):
        parts.append(f"## Example {n}\nInput ({len(p['input'])}x{len(p['input'][0])}):\n"
                     f"{grid_text(p['input'])}\nOutput ({len(p['output'])}x{len(p['output'][0])}):\n"
                     f"{grid_text(p['output'])}\n")
    for n, t in enumerate(task["test"], 1):
        parts.append(f"## Test input {n} ({len(t['input'])}x{len(t['input'][0])}) — for reference only:\n"
                     f"{grid_text(t['input'])}\n")
    return "\n".join(parts)


def feedback_prompt(failure) -> str:
    n, expected, actual, error = failure
    msg = [f"Your function is wrong on training example {n}."]
    if error:
        msg.append(f"It raised or returned an invalid result:\n{error[:1500]}")
    else:
        msg.append(f"Expected output:\n{grid_text(expected)}\nYour output:\n{grid_text(actual)}")
    msg.append("Find the correct rule and reply with the full corrected function in one ```python block.")
    return "\n\n".join(msg)


CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str | None:
    blocks = CODE_RE.findall(text or "")
    blocks = [b for b in blocks if "def transform" in b]
    return blocks[-1] if blocks else None


# --------------------------------------------------------------------------- #
# Gemini client (REST, no SDK dependency)
# --------------------------------------------------------------------------- #
class GeminiClient:
    def __init__(self, model, thinking, max_output_tokens, timeout=600):
        import requests  # imported lazily so --mock needs nothing extra
        self.requests = requests
        self.key = os.environ.get("GEMINI_API_KEY")
        if not self.key:
            raise SystemExit("GEMINI_API_KEY is not set. In Colab, add it under Secrets (the key icon).")
        self.model, self.thinking = model, thinking
        self.max_output_tokens, self.timeout = max_output_tokens, timeout
        self.calls = self.in_tokens = self.out_tokens = 0

    def chat(self, messages):
        """messages: list of (role, text) with role in {'user','model'}."""
        body = {
            "contents": [{"role": r, "parts": [{"text": t}]} for r, t in messages],
            "generationConfig": {"maxOutputTokens": self.max_output_tokens,
                                 "thinkingConfig": {"thinkingLevel": self.thinking}},
        }
        url = API_URL.format(model=self.model)
        for attempt in range(5):
            # The key goes in a header, never in the URL, so it cannot leak into logs.
            r = self.requests.post(url, json=body, timeout=self.timeout,
                                   headers={"x-goog-api-key": self.key})
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            if r.status_code != 200:
                raise RuntimeError(f"Gemini API error {r.status_code}: {r.text[:500]}")
            data = r.json()
            self.calls += 1
            usage = data.get("usageMetadata", {})
            self.in_tokens += usage.get("promptTokenCount", 0)
            self.out_tokens += usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
            cands = data.get("candidates") or []
            if not cands:
                return ""
            return "".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts", [])
                           if not p.get("thought"))
        raise RuntimeError("Gemini API kept returning rate-limit or server errors")


class MockClient:
    """Offline stand-in: first answer is the identity, second a transpose.

    It lets the whole loop (sandbox, verification, feedback, scoring, resume)
    be tested without an API key.
    """
    def __init__(self, *a, **k):
        self.calls = self.in_tokens = self.out_tokens = 0
        self.model = "mock"

    def chat(self, messages):
        self.calls += 1
        if len(messages) == 1:
            return "Identity.\n```python\ndef transform(grid):\n    return [row[:] for row in grid]\n```"
        return ("Transpose.\n```python\nimport numpy as np\n\ndef transform(grid):\n"
                "    return np.array(grid).T.tolist()\n```")


# --------------------------------------------------------------------------- #
# Sandboxed execution
# --------------------------------------------------------------------------- #
RUNNER = r'''
import json, sys, resource
resource.setrlimit(resource.RLIMIT_AS, (MEM, MEM))
src = open(sys.argv[1]).read()
grids = json.load(sys.stdin)
ns = {"__name__": "candidate"}
try:
    exec(compile(src, "candidate.py", "exec"), ns)
    fn = ns["transform"]
except Exception as e:
    print(json.dumps({"error": f"load: {type(e).__name__}: {e}"})); sys.exit(0)
out = []
for g in grids:
    try:
        r = fn([row[:] for row in g])
        if hasattr(r, "tolist"):
            r = r.tolist()
        r = [[int(v) for v in row] for row in r]
        ok = (0 < len(r) <= 30 and all(0 < len(row) <= 30 and len(row) == len(r[0]) for row in r)
              and all(0 <= v <= 9 for row in r for v in row))
        out.append(r if ok else {"error": "invalid grid (must be 1-30 x 1-30, colours 0-9)"})
    except Exception as e:
        out.append({"error": f"{type(e).__name__}: {e}"})
print(json.dumps({"outputs": out}))
'''


def run_program(code, grids, timeout=10, mem_mb=1024):
    """Run transform() on each grid in an isolated Python process."""
    with tempfile.TemporaryDirectory() as d:
        cand = os.path.join(d, "candidate.py")
        runner = os.path.join(d, "runner.py")
        with open(cand, "w") as f:
            f.write(code)
        with open(runner, "w") as f:
            f.write(RUNNER.replace("MEM", str(mem_mb * 1024 * 1024)))
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"}  # no API key in the child
        try:
            p = subprocess.run([sys.executable, "-I", runner, cand], input=json.dumps(grids),
                               capture_output=True, text=True, timeout=timeout, cwd=d, env=env)
        except subprocess.TimeoutExpired:
            return {"error": f"timeout after {timeout}s"}
        try:
            return json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            return {"error": (p.stderr or "no output")[-1500:]}


def verify(code, task):
    """Return (ok, failure) where failure=(example_no, expected, actual, error)."""
    res = run_program(code, [p["input"] for p in task["train"]])
    if "error" in res:
        return False, (1, None, None, res["error"])
    for n, (p, out) in enumerate(zip(task["train"], res["outputs"]), 1):
        if isinstance(out, dict):
            return False, (n, p["output"], None, out["error"])
        if out != p["output"]:
            return False, (n, p["output"], out, None)
    return True, None


# --------------------------------------------------------------------------- #
# Search loop
# --------------------------------------------------------------------------- #
def solve(task, client, samples, rounds):
    """Return dict with verified programs (deduplicated by test predictions)."""
    found, seen, log = [], set(), []
    for s in range(samples):
        messages = [("user", task_prompt(task))]
        for r in range(rounds + 1):
            reply = client.chat(messages)
            code = extract_code(reply)
            if code is None:
                log.append(f"sample {s + 1} round {r}: no code block")
                messages += [("model", reply), ("user", "Reply with the full function in one ```python block.")]
                continue
            ok, failure = verify(code, task)
            if ok:
                res = run_program(code, [t["input"] for t in task["test"]])
                preds = res.get("outputs") if "outputs" in res else None
                key = json.dumps(preds)
                log.append(f"sample {s + 1} round {r}: VERIFIED")
                if key not in seen:
                    seen.add(key)
                    found.append({"code": code, "test_predictions": preds, "sample": s + 1, "round": r})
                break
            log.append(f"sample {s + 1} round {r}: failed on example {failure[0]}")
            messages += [("model", reply), ("user", feedback_prompt(failure))]
        if len(found) >= 2:
            break
    return found, log


def load_tasks(tasks_dir):
    tasks = {}
    for fn in sorted(os.listdir(tasks_dir)):
        if fn.endswith(".json"):
            with open(os.path.join(tasks_dir, fn)) as f:
                tasks[fn[:-5]] = json.load(f)
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks-dir", required=True, help="directory of one-task-per-file ARC JSONs")
    ap.add_argument("--out-dir", default="gemini_out")
    ap.add_argument("--model", default="gemini-4-argon")
    ap.add_argument("--thinking", default="high", choices=["low", "medium", "high"])
    ap.add_argument("--max-output-tokens", type=int, default=32768)
    ap.add_argument("--max-tasks", type=int, default=20, help="new tasks to attempt this run (cost control)")
    ap.add_argument("--max-calls", type=int, default=200, help="hard cap on API calls this run")
    ap.add_argument("--samples", type=int, default=2, help="independent attempts per task")
    ap.add_argument("--rounds", type=int, default=2, help="feedback rounds per attempt")
    ap.add_argument("--retry-failed", action="store_true", help="also retry tasks that failed before")
    ap.add_argument("--mock", action="store_true", help="offline test without the API")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(os.path.join(args.out_dir, "programs"), exist_ok=True)
    state_path = os.path.join(args.out_dir, "state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}

    client = (MockClient if args.mock else GeminiClient)(args.model, args.thinking, args.max_output_tokens)
    tasks = load_tasks(args.tasks_dir)
    todo = [t for t in tasks if t not in state or (args.retry_failed and state[t]["status"] == "failed")]
    todo = todo[: args.max_tasks]
    print(f"{len(tasks)} tasks, {len(state)} already in state, attempting {len(todo)}", flush=True)

    run = []
    for n, tid in enumerate(todo, 1):
        if client.calls >= args.max_calls:
            print(f"Reached --max-calls {args.max_calls}; stopping.", flush=True)
            break
        task = tasks[tid]
        try:
            found, log = solve(task, client, args.samples, args.rounds)
        except RuntimeError as e:
            print(f"{tid}: API error, stopping run: {e}", flush=True)
            break
        rec = {"status": "verified" if found else "failed", "log": log,
               "model": client.model, "at": datetime.now(timezone.utc).isoformat()}
        truths = [t.get("output") for t in task["test"]]
        if found and all(x is not None for x in truths):
            # pass@2 over the (up to two) distinct verified programs, per test input
            per_input = [any(f["test_predictions"] and f["test_predictions"][i] == truths[i] for f in found[:2])
                         for i in range(len(truths))]
            rec["test_correct"] = sum(per_input) / len(per_input)
        elif all(x is not None for x in truths):
            rec["test_correct"] = 0.0
        for k, f in enumerate(found[:2], 1):
            path = os.path.join(args.out_dir, "programs", f"{tid}{'' if k == 1 else '_2'}.py")
            with open(path, "w") as fh:
                fh.write(f"# ARC-AGI-2 task {tid}: verified on all train pairs\n"
                         f"# model={client.model} sample={f['sample']} round={f['round']}"
                         f" test_correct={rec.get('test_correct')}\n\n{f['code'].strip()}\n")
        state[tid] = rec
        run.append(tid)
        with open(state_path, "w") as fh:
            json.dump(state, fh, indent=1)
        print(f"[{time.time() - t0:7.1f}s] {n}/{len(todo)} {tid}: {rec['status']}"
              + (f", test_correct={rec['test_correct']:.2f}" if "test_correct" in rec else ""), flush=True)

    def stats(ids):
        recs = [state[i] for i in ids]
        scored = [r for r in recs if "test_correct" in r]
        return dict(attempted=len(recs), verified=sum(r["status"] == "verified" for r in recs),
                    scored=len(scored), test_correct=round(sum(r["test_correct"] for r in scored), 3))

    this_run, overall = stats(run), stats(list(state))
    report = dict(version=VERSION, run_at=datetime.now(timezone.utc).isoformat(), model=client.model,
                  thinking=args.thinking, samples=args.samples, rounds=args.rounds,
                  this_run=this_run, overall=overall, api_calls=client.calls,
                  input_tokens=client.in_tokens, output_tokens=client.out_tokens,
                  runtime_sec=round(time.time() - t0, 1), tasks_this_run=run)
    with open(os.path.join(args.out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=1)

    def line(s):
        acc = f"{s['test_correct'] / s['scored']:.1%}" if s["scored"] else "n/a"
        return (f"attempted {s['attempted']}, verified on train {s['verified']}, "
                f"test-correct {s['test_correct']}/{s['scored']} ({acc})")

    md = [f"# Gemini ARC-AGI-2 program search — {report['run_at']}", "",
          f"- Model: `{client.model}` (thinking {args.thinking}), {args.samples} samples x {args.rounds} feedback rounds",
          f"- This run: {line(this_run)}",
          f"- All runs so far: {line(overall)}",
          f"- API calls: {client.calls}; tokens in/out: {client.in_tokens:,} / {client.out_tokens:,}",
          f"- Runtime: {report['runtime_sec']}s", "",
          "Not submittable to Kaggle (no internet or API use allowed there). Use the verified programs",
          "in `programs/` as ideas for new families in the offline DSL solver."]
    with open(os.path.join(args.out_dir, "report.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
