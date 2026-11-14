"""e31_analyze.py - Table 8b: category counts and paired differences from e31_rows.csv.gz."""
import gzip, json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(HERE), "e31_out")
df = pd.read_csv(os.path.join(OUT, "e31_rows.csv.gz"))
rng = np.random.default_rng(31)

# --- reproduction check against the published Table 8 / 12 store (post-1 window epochs 25..59)
orig = json.load(open(os.path.join(HERE, "_orig", "out_E_A2", "RR4_dynamic_alarm.json")))["rows"]
chk = []
for r in orig:
    sub = df[(df.topo == r["topo"]) & (df.n_audit == r["n_audit"]) & (df.S1 == r["S1"]) & (df.epoch.between(25, 59))]
    chk.append(max(abs(sub.true_live.mean() - r["bmr_true_secure_post1"]), abs(sub.true_alarm.mean() - r["bma_true_secure_post1"])))
print("reproduces published true-secure (post-1, 24 cells): max abs diff = %.3e" % max(chk))


def boot_ci(x_by_rep, B=10000):
    """paired bootstrap over repetitions (x_by_rep: per-rep mean paired difference)."""
    n = len(x_by_rep)
    idx = rng.integers(0, n, size=(B, n))
    m = x_by_rep[idx].mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


W = df[df.epoch.between(25, 34)].copy()          # first ten epochs of stage 1 (guide)
W["diff"] = W.true_alarm - W.true_live
cats = sorted(W.category.unique())
rows = []
for (topo, na, S1), g in W.groupby(["topo", "n_audit", "S1"]):
    N = len(g)
    row = dict(topo=topo, n_audit=na, S_dag=S1, trials=N)
    for c in cats:
        row[c] = int((g.category == c).sum())
    byrep = g.groupby("rep")["diff"].mean().values
    row["mean_true_live"] = g.true_live.mean()
    row["mean_true_alarm"] = g.true_alarm.mean()
    row["paired_diff"] = g["diff"].mean()
    lo, hi = boot_ci(byrep)
    row["ci_lo"], row["ci_hi"] = lo, hi
    row["live_deliv"] = g.live_delivered.mean()
    row["alarm_deliv"] = g.alarm_delivered.mean()
    row["live_exposed_given_deliv"] = (g.live_has_attacked.sum() / max(g.live_delivered.sum(), 1))
    row["alarm_exposed_given_deliv"] = (g.alarm_has_attacked.sum() / max(g.alarm_delivered.sum(), 1))
    row["mean_flagged_attacked"] = g.n_flagged_attacked.mean()
    row["n_att"] = g.n_attacked.mean()
    rows.append(row)
T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, "e31_table8b_cells.csv"), index=False)
pd.set_option("display.width", 250, "display.max_columns", 40)
print(T[["topo", "n_audit", "S_dag", "trials"] + cats + ["paired_diff", "ci_lo", "ci_hi"]].to_string(index=False))
print("\nall-cell totals over epochs 25-34 (%d trials):" % len(W))
print(W.category.value_counts().sort_index().to_string())
print("mean paired diff (all cells) = %.6f" % W["diff"].mean())

# per n_audit/S_dag pooled over topologies (matches Table 8 layout)
pool = []
for (na, S1), g in W.groupby(["n_audit", "S1"]):
    byrep = g.groupby(["topo", "rep"])["diff"].mean().values
    lo, hi = boot_ci(byrep)
    pool.append(dict(n_audit=na, S_dag=S1, trials=len(g), **{c: int((g.category == c).sum()) for c in cats},
                     true_live=g.true_live.mean(), true_alarm=g.true_alarm.mean(), diff=g["diff"].mean(), ci_lo=lo, ci_hi=hi))
P = pd.DataFrame(pool)
P.to_csv(os.path.join(OUT, "e31_table8b_pooled.csv"), index=False)
print(P.to_string(index=False))

# longer horizon: whole post-attack period (epochs 25..79)
A = df[df.epoch >= 25].copy()
print("\nepochs 25-79 category counts:", A.category.value_counts().sort_index().to_dict())
print("epochs 25-79 mean paired diff = %.6f; trials with true_alarm>true_live: %d; true_alarm<true_live: %d" % (
    (A.true_alarm - A.true_live).mean(), int((A.true_alarm > A.true_live).sum()), int((A.true_alarm < A.true_live).sum())))
# example rows of category c / e2 / f if any
for c in ("c_reroute_to_clean", "e2_leaves_clean_for_attacked", "f_other", "d_reroute_still_attacked"):
    x = A[A.category == c]
    print(c, len(x))
    if len(x):
        print(x.head(5)[["topo", "n_audit", "S1", "rep", "epoch", "live_path", "alarm_path", "live_rate", "alarm_rate", "true_live", "true_alarm"]].to_string(index=False))
# why same_route dominates: in how many trials did the alarm flag an attacked link on the live route?
sr = A[(A.live_has_attacked == 1)]
print("\nepochs 25-79: live route contains an attacked link in %d trials; of these alarm route is identical in %d, drops delivery in %d, other %d" % (
    len(sr), int(sr.same_route.sum()), int(((sr.alarm_delivered == 0)).sum()), int(((sr.same_route == 0) & (sr.alarm_delivered == 1)).sum())))
print("flag counts: mean attacked links flagged per epoch (epochs 25-34) by n_audit,S1:")
print(W.groupby(["n_audit", "S1"])[["n_flagged_attacked", "n_flagged_honest", "n_attacked"]].mean().round(3).to_string())

# ---------------- ground-truth diagnostic: why is category (c) empty?
print("\n=== DIAGNOSTIC (epochs 25-79): trials in which the live route contains an attacked link ===")
X = A[A.live_has_attacked == 1].copy()
X["clean_ge_live"] = (X.clean_exists == 1) & (X.clean_rate >= X.live_rate - 1e-12)
print("n =", len(X))
print("clean feasible route exists (ground truth):", int(X.clean_exists.sum()))
print("  ... of which clean bottleneck >= live bottleneck (max-min would have chosen it):", int(X.clean_ge_live.sum()))
print("  ... clean exists but clean bottleneck < live bottleneck (only case where alarm COULD help):", int(((X.clean_exists == 1) & ~X.clean_ge_live).sum()))
print("no clean feasible route at all:", int((X.clean_exists == 0).sum()))
print(X.groupby(["topo", "n_audit"]).agg(n=("rep", "size"), clean_exists=("clean_exists", "sum"),
      clean_lt_live=("clean_ge_live", lambda v: 0)).to_string())
# among trials with a clean feasible route (any), how often does live route contain attacked link?
Y = A[A.clean_exists == 1]
print("\ntrials with a clean feasible route:", len(Y), "; live route attacked in", int(Y.live_has_attacked.sum()),
      "; alarm route attacked in", int(Y.alarm_has_attacked.sum()))
Z = A[(A.clean_exists == 1) & (A.live_has_attacked == 1)]
if len(Z):
    print("with clean route AND live attacked: rows", len(Z))
    print(Z.head(10)[["topo", "n_audit", "S1", "rep", "epoch", "live_rate", "clean_rate", "live_path", "alarm_path"]].to_string(index=False))
# overall share of attacked-live trials without a clean route, by topology
print("\nshare of post-attack trials (epochs 25-79) with NO clean feasible route, by topology/n:")
print(A.groupby(["topo", "n_audit"]).apply(lambda g: pd.Series(dict(no_clean=1 - g.clean_exists.mean(), live_attacked=g.live_has_attacked.mean(),
      live_deliv=g.live_delivered.mean()))).round(3).to_string())
