"""cells_table.py - factor counts and per-router aggregates of the Table 6 sweep, printed from data/results/q6_sweep.json.
Run:  python cells_table.py"""
import json, os, itertools
from collections import defaultdict

d = json.load(open(os.path.join(os.path.dirname(__file__), "data", "results", "q6_sweep.json")))
rows = d["rows"]
cells = {(r["topo"], r["n"], r["n_audit"], r["cond"], r["mode"]) for r in rows}
print("rows:", len(rows), " cells (topo,n,n_audit,cond,mode):", len(cells))
for name, idx in (("topology", 0), ("size n", 1), ("n_audit", 2), ("condition", 3), ("relay mode", 4)):
    vals = sorted({c[idx] for c in cells}, key=str)
    print(f"  {name}: {len(vals)} levels {vals}")
trials = {r["trials"] for r in rows}
print("trials per cell:", trials)


def agg(sel, label):
    by = defaultdict(list)
    for r in rows:
        if sel(r):
            by[r["router"]].append(r)
    ncell = len({(r["topo"], r["n"], r["n_audit"], r["cond"], r["mode"]) for rs in by.values() for r in rs})
    print(f"\n{label}  ({ncell} cells)")
    print(f"{'router':8s} {'full %':>7s} {'budget-only %':>14s} {'mean rate':>10s} {'budget-only rate':>17s} {'refusal %':>10s}")
    for rn, rs in by.items():
        m = lambda k: sum(x[k] for x in rs) / len(rs)
        print(f"{rn:8s} {m('compl_pct'):7.1f} {m('compl_budget_only_pct'):14.1f} {m('mean_rate'):10.4f} "
              f"{m('mean_rate_budget_only'):17.4f} {m('refusal_pct'):10.2f}")
    return by


agg(lambda r: True, "ALL cells")
agg(lambda r: r["n_audit"] == 500, "n_audit = 500 (Delta = 0.5)")
by = agg(lambda r: r["n_audit"] == 500 and r["mode"] == "sealed", "n_audit = 500, sealed (phi = 0.99)")
agg(lambda r: r["n_audit"] == 500 and r["mode"] == "sealed" and r["topo"] == "ring", "n_audit = 500, sealed, ring")

import numpy as np
b = [r for r in rows if r["router"] == "BMR"]
print("\nBMR refusal %: ring (all cells) =", round(float(np.mean([r["refusal_pct"] for r in b if r["topo"] == "ring"])), 2),
      "; sealed mode =", round(float(np.mean([r["refusal_pct"] for r in b if r["mode"] == "sealed"])), 2),
      "; all =", round(float(np.mean([r["refusal_pct"] for r in b])), 2))
