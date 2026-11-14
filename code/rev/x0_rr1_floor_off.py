"""X0b: run_pending_fast.py RR1 block only (identical seeds/trials) with the hop floor switched off."""
"""
run_pending_fast.py
===================
Faster version: 200 trials per cell (sufficient for stable estimates at 4 decimal places).
Focuses on key cells only: ER N=30, compromised (f=0.20), S† in {2.418, 2.55, 2.62}, Delta in {1e-3, 3e-5}.

RR1: decomposition
RR2: ring flagship S†=2.55 and S†=2.62 (with alarm tracking)
RR5: 5000-instance cross-validation

Run from: REVISION_OSN_12OCT2026/
  python3 run_pending_fast.py
"""
import sys, os, json, math, time
import numpy as np
import itertools

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(HERE, "out", "X0_rr1")
os.makedirs(OUTDIR, exist_ok=True)

R2_PKG = HERE
sys.path.insert(0, R2_PKG)
from bmr.bounds import eta_tight, h_pironio

# ─── constants ───────────────────────────────────────────────────────────────
Z_ALPHA = 4.7534
S0_REF  = 2.687
ELL     = 10_000
PHI     = 0.99
DELTA   = 1e-6
N_AUDIT = 10_000
R_MIN   = 0.01
CLASS_S = np.array([2.720, 2.687, 2.640])
CLASS_P = np.array([0.3, 0.5, 0.2])

def sigma_w(S0, n):
    return math.sqrt(max(0.0, 4.0 - S0**2 / 4.0) / n)

def certified_rate(S_hat, n=N_AUDIT, delta=DELTA, ell=ELL, phi=PHI):
    eta = eta_tight(n, delta)
    S_lb = S_hat - eta
    if S_lb <= 2.0:
        return 0.0, 1.0, 0.0, False
    h = h_pironio(S_lb)
    H = ell * h
    k = math.floor(phi * H)
    if k <= 0 or H <= k:
        return 0.0, 1.0, h, True
    log2_eps = -1.0 - (H - k) / 2.0
    eps = 0.0 if log2_eps < -1000 else 2.0**log2_eps
    r = k / ell
    return r, eps, h, True

def alarm_fires(S_hat, S0=S0_REF, n=N_AUDIT):
    theta = S0 - Z_ALPHA * sigma_w(S0, n)
    return S_hat < theta

# ─── pre-compute derived ε for the (R) parameter table ────────────────────
print("=== Derived ε bridge (n=10^4, phi=0.99) ===")
for S in [2.720, 2.687, 2.640, 2.62, 2.55, 2.48, 2.418]:
    r, eps, h, gate = certified_rate(S, n=N_AUDIT, delta=DELTA, ell=ELL, phi=PHI)
    w = eps + DELTA
    print(f"  S={S:.3f}: r={r:.4f}, eps={eps:.3e}, w={w:.3e}, gate={gate}")


# ════════════════════════════════════════════════════════════════════════════
#  RR1 — DECOMPOSITION UNDER (R) — key cells only
# ════════════════════════════════════════════════════════════════════════════
print("\n=== RR1: Decomposition under (R) ===")

N_TRIALS = 200  # per cell (fast)
N_NODES  = 30
ER_P     = 4.0 / (N_NODES - 1)
F_COMP   = 0.20
S_ATTACK_LIST = [2.418, 2.55, 2.62]
DELTA_LIST    = [1e-3, 3e-5]


def make_er_graph(n, p, rng):
    iu, ju = np.triu_indices(n, 1)
    keep = rng.random(len(iu)) < p
    return list(zip(iu[keep].tolist(), ju[keep].tolist()))

def make_ba_graph(n, m_ba, rng):
    import networkx as nx
    G = nx.barabasi_albert_graph(n, m_ba, seed=int(rng.integers(2**31)))
    return sorted((min(a,b), max(a,b)) for a,b in G.edges())

def sample_links_R(edges, n_audit, f, rng, S_att):
    E = len(edges)
    cls = rng.choice(3, size=E, p=CLASS_P)
    S_true = CLASS_S[cls].copy()
    k_comp = max(1, int(round(f * E)))
    comp_idx = rng.choice(E, size=k_comp, replace=False)
    comp = np.zeros(E, dtype=bool)
    comp[comp_idx] = True
    S_true[comp] = S_att
    p_win = (1.0 + S_true / 4.0) / 2.0
    K = rng.binomial(4 * n_audit, p_win)
    S_hat = 2.0 * K / n_audit - 4.0
    r_arr = np.zeros(E); eps_arr = np.ones(E); gate_arr = np.zeros(E, dtype=bool); alarm_arr = np.zeros(E, dtype=bool)
    for i in range(E):
        r_arr[i], eps_arr[i], _, gate_arr[i] = certified_rate(S_hat[i])
        alarm_arr[i] = alarm_fires(S_hat[i])
    w_arr = eps_arr + DELTA
    valid_arr = gate_arr & ~alarm_arr
    return dict(r=r_arr, eps=eps_arr, w=w_arr, gate=gate_arr, alarm=alarm_arr, valid=valid_arr)

def best_path_variant(edges, n_nodes, s, d, lk, Delta, mode):
    """DFS over all simple s-d paths (Lmin=3, Hmax=8), return bottleneck rate."""
    from collections import defaultdict
    adj_list = defaultdict(list)
    for (u, v) in edges:
        adj_list[u].append(v); adj_list[v].append(u)
    edge_idx = {(u,v): i for i,(u,v) in enumerate(edges)}
    for i,(u,v) in enumerate(edges):
        edge_idx[(v,u)] = i
    r, w, valid = lk["r"], lk["w"], lk["valid"]
    best = [-1.0, math.inf, math.inf]  # [rate, hops, sw]

    def dfs(node, visited, links_on_path, hops):
        if hops > 8: return
        if node == d:
            # D-1(a): a direct link (hops == 1) is admissible for every router.
            # SPF is scored with the hop floor waived (Table 7 caption), so it is not floor-limited.
            pass  # X0: hop floor OFF for every router
            sw = sum(w[li] for li in links_on_path)
            rv = min(r[li] for li in links_on_path)
            av = all(valid[li] for li in links_on_path)
            bud = sw <= Delta * (1 + 1e-9)
            rte = rv >= R_MIN
            if mode == "spf":
                if hops < best[1] or (hops == best[1]):
                    if hops < best[1]: best[0] = rv; best[1] = hops; best[2] = sw
            elif mode == "spf_valid":
                if av and (hops < best[1] or (hops == best[1] and rv > best[0])):
                    best[0] = rv; best[1] = hops; best[2] = sw
            elif mode == "spf_budget":
                if bud and rte and (hops < best[1] or (hops == best[1] and rv > best[0])):
                    best[0] = rv; best[1] = hops; best[2] = sw
            elif mode == "spf_vb":
                if av and bud and rte and (hops < best[1] or (hops == best[1] and rv > best[0])):
                    best[0] = rv; best[1] = hops; best[2] = sw
            elif mode == "bmr":
                if av and bud and rte and (rv > best[0] or (rv == best[0] and hops < best[1])):
                    best[0] = rv; best[1] = hops; best[2] = sw
            return
        for nb in adj_list[node]:
            if nb in visited: continue
            ei = edge_idx.get((node, nb), -1)
            if ei < 0: continue
            visited.add(nb)
            links_on_path.append(ei)
            dfs(nb, visited, links_on_path, hops+1)
            links_on_path.pop()
            visited.discard(nb)

    dfs(s, {s}, [], 0)
    return max(0.0, best[0])

def decomp_trial(edges, n_nodes, s, d, lk, Delta):
    r_bmr = best_path_variant(edges, n_nodes, s, d, lk, Delta, "bmr")
    r_spf = best_path_variant(edges, n_nodes, s, d, lk, Delta, "spf")
    r_d2  = best_path_variant(edges, n_nodes, s, d, lk, Delta, "spf_budget")
    r_d3  = best_path_variant(edges, n_nodes, s, d, lk, Delta, "spf_vb")
    return r_bmr, r_spf, r_d2, r_d3

rr1_results = {}
np.random.seed(20261019)   # bootstrap CIs reproducible
t0_rr1 = time.time()
for topo in ["er", "ba"]:
    for S_att in S_ATTACK_LIST:
        for Delta in DELTA_LIST:
            key = f"{topo}_N30_f020_Satt{S_att}_D{Delta:.0e}"
            rates = {"bmr": [], "spf": [], "d2": [], "d3": []}
            for trial in range(N_TRIALS):
                trial_rng = np.random.default_rng(np.random.SeedSequence([20261012, trial, int(S_att*1000), int(Delta*1e6)]))
                if topo == "er":
                    edges = make_er_graph(N_NODES, ER_P, trial_rng)
                else:
                    edges = make_ba_graph(N_NODES, 2, trial_rng)
                if not edges: continue
                lk = sample_links_R(edges, N_AUDIT, F_COMP, trial_rng, S_att)
                s, d_node = 0, N_NODES - 1
                r_bmr, r_spf, r_d2, r_d3 = decomp_trial(edges, N_NODES, s, d_node, lk, Delta)
                rates["bmr"].append(r_bmr); rates["spf"].append(r_spf)
                rates["d2"].append(r_d2); rates["d3"].append(r_d3)

            bm = np.array(rates["bmr"]); sp = np.array(rates["spf"])
            d2 = np.array(rates["d2"]); d3 = np.array(rates["d3"])
            n = len(bm)
            ord1_budget   = float((d2 - sp).mean())
            ord1_validity = float((d3 - d2).mean())
            objective     = float((bm - d3).mean())
            total         = float((bm - sp).mean())
            feasible      = int((bm > 0).sum())

            def boot_ci(a, b, B=1000):
                diffs = a - b
                if len(diffs) == 0: return [0.0, 0.0]
                ms = [diffs[np.random.randint(0, len(diffs), len(diffs))].mean() for _ in range(B)]
                return np.percentile(ms, [2.5, 97.5]).tolist()

            ci = boot_ci(bm, sp)
            rr1_results[key] = {
                "topo": topo, "N": N_NODES, "f": F_COMP, "S_att": S_att, "Delta": Delta,
                "n_trials": n, "n_feasible": feasible,
                "BMR_mean": float(bm.mean()), "SPF_mean": float(sp.mean()),
                "D2_mean": float(d2.mean()), "D3_mean": float(d3.mean()),
                "total_BMR_SPF": total, "BMR_SPF_ci95": ci,
                "ord1_budget": ord1_budget, "ord1_validity": ord1_validity, "objective": objective,
            }
            print(f"  {key:45s}: total={total:+.4f} [budget={ord1_budget:+.4f}, valid={ord1_validity:+.4f}, obj={objective:+.4f}] feasible={feasible}/{n}")

json.dump(rr1_results, open(os.path.join(OUTDIR, "RR1_decomposition_R_family.json"), "w"), indent=2)
print(f"RR1 done ({time.time()-t0_rr1:.1f}s) -> RR1_decomposition_R_family.json")


