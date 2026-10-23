#!/usr/bin/env python3
"""
ARC-AGI-2 DSL Program-Search Solver
===================================

For each task, a library of candidate "programs" (DSL families) is fitted on
the task's train pairs. A program is accepted only if it reproduces every
train output exactly (the per-task equivalent of cross-validation). Accepted
programs are ranked by simplicity and the top two produce attempt_1 and
attempt_2 for each test input.

DSL families
------------
- Geometry: identity, rotations, flips, transposes
- Crop: crop to the non-background bounding box
- Scaling: up-scale / down-scale by an integer factor
- Pattern Replication:
    * tile the input into the output with a learned transform per tile
    * repair a periodic pattern (fill a masked colour using the detected period)
    * repair a mirror-symmetric pattern
- Grid Reconstruction: split the input on separator lines into cells, then
    * combine cells pixel-wise with a learned truth table (AND/OR/XOR/...)
    * pick the odd-one-out cell, or the cell with the most / fewest coloured pixels
- Dynamic Shape Matching: find connected objects and
    * extract the object selected by a property (largest, smallest, unique colour,
      unique shape)
    * recolour every object by a learned feature -> colour rule (size, size rank,
      shape signature, bounding-box size)
- Post-processing: any family can be followed by a learned colour mapping, and
  geometry can be composed after the other families (depth-2 search)

Inputs
------
- A Kaggle-style challenges JSON ({task_id: {"train": [...], "test": [...]}}),
  optionally with a matching solutions JSON for scoring, or
- A directory of one-task-per-file JSONs (the public ARC-AGI-2 repo layout),
  where test outputs are used for scoring when present.

Outputs (in --out-dir)
---------------------
- submission.json   {task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}
- report.json / report.md  accuracy, per-family hit counts, timing

Usage
-----
  python pipeline.py --tasks-dir data/evaluation --out-dir output
  python pipeline.py --challenges arc-agi_test_challenges.json --out-dir output
  python pipeline.py --challenges ch.json --solutions sol.json --out-dir output
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from itertools import product

import numpy as np

VERSION = "1.0"


# --------------------------------------------------------------------------- #
# Grid helpers
# --------------------------------------------------------------------------- #
def G(x) -> np.ndarray:
    return np.array(x, dtype=np.int16)


def eq(a, b) -> bool:
    return a is not None and b is not None and a.shape == b.shape and bool((a == b).all())


def bg_color(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(counts)])


def bbox(mask: np.ndarray):
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return rows.min(), rows.max() + 1, cols.min(), cols.max() + 1


GEOMS = {
    "identity": lambda g: g,
    "rot90": lambda g: np.rot90(g, 1),
    "rot180": lambda g: np.rot90(g, 2),
    "rot270": lambda g: np.rot90(g, 3),
    "flip_ud": lambda g: g[::-1, :],
    "flip_lr": lambda g: g[:, ::-1],
    "transpose": lambda g: g.T,
    "anti_transpose": lambda g: np.rot90(g, 2).T,
}


def components(g: np.ndarray, bg: int, diag: bool, by_color: bool):
    """Connected components of non-background cells. Returns list of dicts."""
    h, w = g.shape
    seen = np.zeros((h, w), bool)
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diag:
        nbrs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    objs = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or g[r, c] == bg:
                continue
            color = g[r, c]
            stack, cells = [(r, c)], []
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in nbrs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and g[ny, nx] != bg:
                        if by_color and g[ny, nx] != color:
                            continue
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys, xs = zip(*cells)
            r0, r1, c0, c1 = min(ys), max(ys) + 1, min(xs), max(xs) + 1
            mask = np.zeros((r1 - r0, c1 - c0), bool)
            mask[np.array(ys) - r0, np.array(xs) - c0] = True
            sub = g[r0:r1, c0:c1].copy()
            sub[~mask] = bg
            colors = Counter(int(g[y, x]) for y, x in cells)
            objs.append(dict(cells=cells, box=(r0, r1, c0, c1), mask=mask, grid=sub,
                             size=len(cells), color=colors.most_common(1)[0][0],
                             shape_sig=mask.tobytes() + bytes(mask.shape)))
    return objs


# --------------------------------------------------------------------------- #
# DSL families. Each: fit(pairs) -> bool, predict(grid) -> grid | None
# --------------------------------------------------------------------------- #
class Family:
    name = "base"
    cost = 1

    def fit(self, pairs) -> bool:
        raise NotImplementedError

    def predict(self, g):
        raise NotImplementedError

    def check(self, pairs) -> bool:
        try:
            return all(eq(self.predict(i), o) for i, o in pairs)
        except Exception:
            return False


class Geometry(Family):
    cost = 1

    def __init__(self, key):
        self.key, self.name = key, f"geom:{key}"

    def fit(self, pairs):
        return self.check(pairs)

    def predict(self, g):
        return GEOMS[self.key](g)


class CropNonBg(Family):
    name, cost = "crop_nonbg", 2

    def fit(self, pairs):
        return self.check(pairs)

    def predict(self, g):
        b = bbox(g != bg_color(g))
        return None if b is None else g[b[0]:b[1], b[2]:b[3]]


class Scale(Family):
    name, cost = "scale", 2

    def fit(self, pairs):
        i, o = pairs[0]
        self.up = None
        if o.shape[0] % i.shape[0] == 0 and o.shape[1] % i.shape[1] == 0:
            self.up = (o.shape[0] // i.shape[0], o.shape[1] // i.shape[1])
            self.name = f"scale_up:{self.up}"
        elif i.shape[0] % o.shape[0] == 0 and i.shape[1] % o.shape[1] == 0:
            self.down = (i.shape[0] // o.shape[0], i.shape[1] // o.shape[1])
            self.name = f"scale_down:{self.down}"
        else:
            return False
        if self.up == (1, 1):
            return False
        return self.check(pairs)

    def predict(self, g):
        if self.up:
            return np.kron(g, np.ones(self.up, dtype=g.dtype))
        return g[::self.down[0], ::self.down[1]]


# ---- Pattern Replication --------------------------------------------------- #
class TileReplication(Family):
    """Output = (a x b) tiles of the input, each tile a learned geometric transform."""
    name, cost = "pattern:tile", 3

    def fit(self, pairs):
        i, o = pairs[0]
        if o.shape[0] % i.shape[0] or o.shape[1] % i.shape[1]:
            return False
        self.a, self.b = o.shape[0] // i.shape[0], o.shape[1] // i.shape[1]
        if (self.a, self.b) == (1, 1):
            return False
        h, w = i.shape
        self.plan = {}
        for r in range(self.a):
            for c in range(self.b):
                tile = o[r * h:(r + 1) * h, c * w:(c + 1) * w]
                found = None
                for k, f in GEOMS.items():
                    t = f(i)
                    if t.shape == tile.shape and (t == tile).all():
                        found = k
                        break
                if found is None:
                    if (tile == bg_color(o)).all():
                        found = "blank"
                    else:
                        return False
                self.plan[(r, c)] = found
        self.name = f"pattern:tile{self.a}x{self.b}"
        return self.check(pairs)

    def predict(self, g):
        h, w = g.shape
        out = np.zeros((self.a * h, self.b * w), g.dtype)
        for (r, c), k in self.plan.items():
            if k == "blank":
                out[r * h:(r + 1) * h, c * w:(c + 1) * w] = bg_color(g)
            else:
                t = GEOMS[k](g)
                if t.shape != (h, w):
                    return None
                out[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
        return out


def _mask_color(pairs):
    """Colour present in every input but absent from every output (the 'hole')."""
    cands = None
    for i, o in pairs:
        s = set(np.unique(i).tolist()) - set(np.unique(o).tolist())
        cands = s if cands is None else cands & s
    return min(cands) if cands else None


class PeriodicRepair(Family):
    """Fill a masked colour using the smallest 2-D period consistent with the rest."""
    name, cost = "pattern:periodic_repair", 3

    def fit(self, pairs):
        if any(i.shape != o.shape for i, o in pairs):
            return False
        self.mask = _mask_color(pairs)
        if self.mask is None:
            return False
        return self.check(pairs)

    @staticmethod
    def _fill(g, known):
        h, w = g.shape
        for py in range(1, h + 1):
            for px in range(1, w + 1):
                if py == h and px == w:
                    continue
                tile = -np.ones((py, px), np.int16)
                ok = True
                for y in range(h):
                    for x in range(w):
                        if not known[y, x]:
                            continue
                        v = tile[y % py, x % px]
                        if v == -1:
                            tile[y % py, x % px] = g[y, x]
                        elif v != g[y, x]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok and (tile >= 0).all():
                    return np.tile(tile, (h // py + 1, w // px + 1))[:h, :w]
        return None

    def predict(self, g):
        known = g != self.mask
        if known.all():
            return None
        return self._fill(g, known)


class SymmetryRepair(Family):
    """Fill a masked colour from mirror / rotational copies of the grid."""
    name, cost = "pattern:symmetry_repair", 3

    def fit(self, pairs):
        if any(i.shape != o.shape for i, o in pairs):
            return False
        self.mask = _mask_color(pairs)
        if self.mask is None:
            return False
        return self.check(pairs)

    def predict(self, g):
        out = g.copy()
        hole = out == self.mask
        for key in ["flip_lr", "flip_ud", "rot180", "transpose", "anti_transpose", "rot90", "rot270"]:
            t = GEOMS[key](out)
            if t.shape != out.shape:
                continue
            fill = hole & (t != self.mask)
            out[fill] = t[fill]
            hole = out == self.mask
        return None if hole.any() else out


# ---- Grid Reconstruction --------------------------------------------------- #
def split_cells(g):
    """Split on full-length separator rows/cols of one colour. Returns (cells, sep_color)."""
    h, w = g.shape
    for color in np.unique(g):
        rows = [r for r in range(h) if (g[r] == color).all()]
        cols = [c for c in range(w) if (g[:, c] == color).all()]
        if not rows and not cols:
            continue
        rcuts = [-1] + rows + [h]
        ccuts = [-1] + cols + [w]
        cells = []
        for r0, r1 in zip(rcuts, rcuts[1:]):
            for c0, c1 in zip(ccuts, ccuts[1:]):
                if r1 - r0 > 1 and c1 - c0 > 1:
                    cells.append(g[r0 + 1:r1, c0 + 1:c1])
        if len(cells) >= 2 and len({c.shape for c in cells}) == 1:
            return cells, int(color)
    return None, None


class GridCombine(Family):
    """Combine equal-size cells pixel-wise with a learned truth table."""
    name, cost = "grid:combine_cells", 4

    def fit(self, pairs):
        self.table = {}
        for i, o in pairs:
            cells, sep = split_cells(i)
            if cells is None or cells[0].shape != o.shape:
                return False
            bg = self._bg(cells, sep)
            for idx in np.ndindex(o.shape):
                key = tuple(int(c[idx] != bg) for c in cells)
                v = int(o[idx])
                if self.table.setdefault(key, v) != v:
                    return False
        return self.check(pairs)

    @staticmethod
    def _bg(cells, sep):
        return bg_color(np.concatenate([c.ravel() for c in cells]).reshape(1, -1))

    def predict(self, g):
        cells, sep = split_cells(g)
        if cells is None:
            return None
        bg = self._bg(cells, sep)
        out = np.zeros(cells[0].shape, np.int16)
        for idx in np.ndindex(out.shape):
            key = tuple(int(c[idx] != bg) for c in cells)
            if key not in self.table:
                return None
            out[idx] = self.table[key]
        return out


class GridSelect(Family):
    """Pick one cell: odd-one-out, most or fewest coloured pixels."""
    cost = 4

    def __init__(self, rule):
        self.rule, self.name = rule, f"grid:select_{rule}"

    def fit(self, pairs):
        return self.check(pairs)

    def predict(self, g):
        cells, sep = split_cells(g)
        if cells is None:
            return None
        if self.rule == "odd":
            keys = [c.tobytes() for c in cells]
            cnt = Counter(keys)
            odd = [c for c, k in zip(cells, keys) if cnt[k] == 1]
            return odd[0] if len(odd) == 1 else None
        bg = bg_color(np.concatenate([c.ravel() for c in cells]).reshape(1, -1))
        counts = [int((c != bg).sum()) for c in cells]
        target = max(counts) if self.rule == "most" else min(counts)
        hits = [c for c, n in zip(cells, counts) if n == target]
        return hits[0] if len(hits) == 1 else None


# ---- Dynamic Shape Matching ------------------------------------------------ #
class ObjectSelect(Family):
    """Extract the object chosen by a property."""
    cost = 4

    def __init__(self, rule, diag, by_color):
        self.rule, self.diag, self.by_color = rule, diag, by_color
        self.name = f"shape:select_{rule}{'_8c' if diag else ''}{'_mc' if not by_color else ''}"

    def fit(self, pairs):
        return self.check(pairs)

    def predict(self, g):
        objs = components(g, bg_color(g), self.diag, self.by_color)
        if not objs:
            return None
        if self.rule in ("largest", "smallest"):
            sizes = [o["size"] for o in objs]
            t = max(sizes) if self.rule == "largest" else min(sizes)
            hits = [o for o in objs if o["size"] == t]
        elif self.rule == "unique_color":
            cnt = Counter(o["color"] for o in objs)
            hits = [o for o in objs if cnt[o["color"]] == 1]
        else:  # unique_shape
            cnt = Counter(o["shape_sig"] for o in objs)
            hits = [o for o in objs if cnt[o["shape_sig"]] == 1]
        return hits[0]["grid"] if len(hits) == 1 else None


def _features(objs):
    sizes = sorted({o["size"] for o in objs})
    for o in objs:
        yield o, {
            "size": o["size"],
            "size_rank": sizes.index(o["size"]),
            "size_rank_desc": len(sizes) - 1 - sizes.index(o["size"]),
            "shape": o["shape_sig"],
            "box": o["mask"].shape,
        }


class ObjectRecolor(Family):
    """Recolour each object using a learned feature -> colour rule."""
    cost = 5

    def __init__(self, feature, diag):
        self.feature, self.diag = feature, diag
        self.name = f"shape:recolor_by_{feature}{'_8c' if diag else ''}"

    def fit(self, pairs):
        self.rule = {}
        for i, o in pairs:
            if i.shape != o.shape:
                return False
            bg = bg_color(i)
            if ((i == bg) != (o == bg)).any():
                return False
            for obj, f in _features(components(i, bg, self.diag, False)):
                ys, xs = zip(*obj["cells"])
                colors = set(o[list(ys), list(xs)].tolist())
                if len(colors) != 1:
                    return False
                c = colors.pop()
                if self.rule.setdefault(f[self.feature], c) != c:
                    return False
        return len(self.rule) > 0 and self.check(pairs)

    def predict(self, g):
        bg = bg_color(g)
        out = g.copy()
        for obj, f in _features(components(g, bg, self.diag, False)):
            key = f[self.feature]
            if key not in self.rule:
                return None
            ys, xs = zip(*obj["cells"])
            out[list(ys), list(xs)] = self.rule[key]
        return out


# ---- Wrappers -------------------------------------------------------------- #
class WithColorMap(Family):
    """Run a base family, then apply a learned per-colour mapping."""

    def __init__(self, base_factory):
        self.base_factory = base_factory
        self.base = base_factory()
        self.name = f"{self.base.name}+colormap"
        self.cost = self.base.cost + 2

    def fit(self, pairs):
        # Let the base learn its shape parameters; its own exact-match check is
        # expected to fail here because the colours differ.
        try:
            self.base.fit(pairs)
        except Exception:
            return False
        mids = []
        for i, o in pairs:
            try:
                m = self.base.predict(i)
            except Exception:
                return False
            if m is None or m.shape != o.shape:
                return False
            mids.append((m, o))
        self.cmap = {}
        for m, o in mids:
            for a, b in zip(m.ravel().tolist(), o.ravel().tolist()):
                if self.cmap.setdefault(a, b) != b:
                    return False
        if all(k == v for k, v in self.cmap.items()):
            return False
        self.name = f"{self.base.name}+colormap"
        return self.check(pairs)

    def predict(self, g):
        m = self.base.predict(g)
        if m is None:
            return None
        out = m.copy()
        for a, b in self.cmap.items():
            out[m == a] = b
        return out


class Compose(Family):
    """Geometry applied after a fitted family (depth-2 search)."""

    def __init__(self, first_factory, geom_key):
        self.first = first_factory()
        self.geom_key = geom_key
        self.name = f"{self.first.name}>{geom_key}"
        self.cost = self.first.cost + 2

    def fit(self, pairs):
        inv = {"rot90": "rot270", "rot270": "rot90"}.get(self.geom_key, self.geom_key)
        back = [(i, GEOMS[inv](o)) for i, o in pairs]
        if not self.first.fit(back):
            return False
        self.name = f"{self.first.name}>{self.geom_key}"
        return self.check(pairs)

    def predict(self, g):
        m = self.first.predict(g)
        return None if m is None else GEOMS[self.geom_key](m)


def library():
    """Factories for every candidate program, cheapest first."""
    fs = [lambda k=k: Geometry(k) for k in GEOMS]
    fs += [CropNonBg, Scale, TileReplication, PeriodicRepair, SymmetryRepair, GridCombine]
    fs += [lambda r=r: GridSelect(r) for r in ("odd", "most", "fewest")]
    for rule, diag, by_color in product(("largest", "smallest", "unique_color", "unique_shape"),
                                        (False, True), (True, False)):
        fs.append(lambda r=rule, d=diag, b=by_color: ObjectSelect(r, d, b))
    for feat, diag in product(("size", "size_rank", "size_rank_desc", "shape", "box"), (False, True)):
        fs.append(lambda f=feat, d=diag: ObjectRecolor(f, d))
    base_for_cmap = [lambda k=k: Geometry(k) for k in GEOMS] + [CropNonBg, Scale]
    base_for_cmap += [lambda r=r: GridSelect(r) for r in ("odd", "most", "fewest")]
    fs += [lambda b=b: WithColorMap(b) for b in base_for_cmap]
    composable = [CropNonBg, TileReplication, GridCombine] + \
                 [lambda r=r: ObjectSelect(r, False, True) for r in ("largest", "smallest", "unique_color", "unique_shape")]
    fs += [lambda f=f, k=k: Compose(f, k) for f in composable for k in GEOMS if k != "identity"]
    return fs


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
def solve_task(task, time_budget=10.0):
    """Return (list of attempts per test input, list of fitted family names)."""
    pairs = [(G(p["input"]), G(p["output"])) for p in task["train"]]
    t0 = time.time()
    fitted = []
    for factory in library():
        if time.time() - t0 > time_budget:
            break
        fam = factory()
        try:
            if fam.fit(pairs):
                fitted.append(fam)
        except Exception:
            continue
    fitted.sort(key=lambda f: f.cost)

    results = []
    for t in task["test"]:
        g = G(t["input"])
        attempts, seen = [], set()
        for fam in fitted:
            try:
                p = fam.predict(g)
            except Exception:
                p = None
            if p is None or p.size == 0 or p.shape[0] > 30 or p.shape[1] > 30:
                continue
            key = p.tobytes() + bytes(p.shape)
            if key in seen:
                continue
            seen.add(key)
            attempts.append((p.tolist(), fam.name))
            if len(attempts) == 2:
                break
        while len(attempts) < 2:
            attempts.append((g.tolist(), "fallback:input"))
        results.append(attempts)
    return results, [f.name for f in fitted]


# --------------------------------------------------------------------------- #
# I/O and scoring
# --------------------------------------------------------------------------- #
def load_tasks(args):
    if args.challenges:
        with open(args.challenges) as f:
            tasks = json.load(f)
        sols = None
        if args.solutions:
            with open(args.solutions) as f:
                sols = json.load(f)
        return tasks, sols
    tasks, sols = {}, {}
    for fn in sorted(os.listdir(args.tasks_dir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(args.tasks_dir, fn)) as f:
            t = json.load(f)
        tid = fn[:-5]
        tasks[tid] = {"train": t["train"], "test": [{"input": x["input"]} for x in t["test"]]}
        if all("output" in x for x in t["test"]):
            sols[tid] = [x["output"] for x in t["test"]]
    return tasks, (sols or None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tasks-dir", help="directory of one-task-per-file JSONs")
    src.add_argument("--challenges", help="Kaggle-style challenges JSON")
    ap.add_argument("--solutions", help="Kaggle-style solutions JSON (for scoring)")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--time-per-task", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N tasks (0 = all)")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    tasks, sols = load_tasks(args)
    ids = list(tasks)[: args.limit or None]

    submission, per_task, fam_hits, fam_fits = {}, {}, Counter(), Counter()
    solved = total = 0
    for n, tid in enumerate(ids, 1):
        results, fitted = solve_task(tasks[tid], args.time_per_task)
        fam_fits.update(set(fitted))
        submission[tid] = [{"attempt_1": a[0][0], "attempt_2": a[1][0]} for a in results]
        rec = {"fitted": fitted, "attempts": [[a[0][1], a[1][1]] for a in results]}
        if sols and tid in sols:
            ok = []
            for a, truth in zip(results, sols[tid]):
                hit = next((name for grid, name in a if grid == truth), None)
                ok.append(hit is not None)
                if hit:
                    fam_hits[hit] += 1
            # ARC scoring: each test input of a task counts as a fraction of the task
            solved += sum(ok) / len(ok)
            total += 1
            rec["correct"] = ok
        per_task[tid] = rec
        if n % 50 == 0:
            print(f"[{time.time() - t0:6.1f}s] {n}/{len(ids)} tasks", flush=True)

    with open(os.path.join(args.out_dir, "submission.json"), "w") as f:
        json.dump(submission, f)

    acc = solved / total if total else None
    any_fit = sum(1 for r in per_task.values() if r["fitted"])
    report = dict(
        version=VERSION, run_at=datetime.now(timezone.utc).isoformat(),
        source=args.challenges or args.tasks_dir, n_tasks=len(ids),
        tasks_with_a_fitting_program=any_fit,
        scored=bool(total), pass_at_2=acc, tasks_solved=round(solved, 3),
        correct_by_family=dict(fam_hits.most_common()),
        fitted_by_family=dict(fam_fits.most_common()),
        library_size=len(library()), runtime_sec=round(time.time() - t0, 1),
        per_task=per_task,
    )
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=1)

    md = [f"# ARC-AGI-2 solver report — {report['run_at']}", "",
          f"- Source: `{report['source']}`",
          f"- Tasks: {len(ids)}; DSL programs in library: {report['library_size']}",
          f"- Tasks where at least one program fit all train pairs: {any_fit} ({any_fit / max(len(ids), 1):.1%})"]
    if total:
        md.append(f"- **pass@2 accuracy: {acc:.2%}** ({solved:.2f} / {total} tasks)")
        md += ["", "## Correct predictions by DSL family", ""] + \
              [f"- {k}: {v}" for k, v in fam_hits.most_common()] or ["- none"]
    else:
        md.append("- No solutions provided, so accuracy was not scored.")
    md += ["", "## Programs that fit train pairs (top 15)", ""] + \
          [f"- {k}: {v}" for k, v in fam_fits.most_common(15)]
    md.append(f"\nRuntime: {report['runtime_sec']}s")
    with open(os.path.join(args.out_dir, "report.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
