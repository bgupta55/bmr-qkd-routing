"""
bmr/bmr_mp.py  (revision 13 Nov 2026, X4)
=========================================
BMR-MP: routing a *plan* = t+1 internally vertex-disjoint s-d paths, EACH with sum(w) <= Delta (per-path budget),
every link rate >= rho; maximise the bottleneck rho over the whole plan.  A usable direct link s-d is a plan on its own
(no relay, nothing to collude with).  No hop floor is applied to plans.

Three feasibility tests for a fixed rho (all share the same binary search over the distinct link rates):
  plan_feasible_enum   exact; candidate simple paths by non-decreasing weight (Yen) up to Delta, then DFS for t+1
                       pairwise relay-disjoint candidates.  Exact unless the candidate cap / deadline is hit (status).
  plan_feasible_ilp    exact; CP-SAT, t+1 unit flows, node capacity 1, per-flow budget constraint (integer-scaled weights).
  plan_feasible_agg    NECESSARY filter; min-cost flow of t+1 units on the node-split graph, total weight <= (t+1)*Delta.
Weights are scaled by SCALE=1e12 for the integer solvers; the budget test is  w_int <= floor(Delta*SCALE + 0.5).
"""
import math
import time
import networkx as nx

_TOL = 1e-15
SCALE = 10 ** 12


class Timeout(Exception):
    pass


def _simple(G):
    """work graph: just the attributes we need"""
    H = nx.Graph()
    for u, v, a in G.edges(data=True):
        H.add_edge(u, v, w=a["w"], r=a["r"])
    return H


def filter_rate(G, R):
    H = nx.Graph()
    for u, v, a in G.edges(data=True):
        if a.get("r", 0) >= R:
            H.add_edge(u, v, **a)
    return H


def _direct(H, s, d, Delta):
    return H.has_edge(s, d) and H[s][d]["w"] <= Delta + _TOL


# ---------------------------------------------------------------------------------------------- enumeration (exact)
def plan_feasible_enum(H, s, d, t, Delta, cap=100_000, deadline=None):
    """-> (plan or None, status) with status in {"ok", "cap", "timeout"}; "cap"/"timeout" mean a None is NOT a proof."""
    if s not in H or d not in H:
        return None, "ok"
    if _direct(H, s, d, Delta):
        return [[s, d]], "ok"
    K = t + 1
    Hn = H.copy()
    if Hn.has_edge(s, d):
        Hn.remove_edge(s, d)
    cand = []
    status = "ok"
    try:
        for path in nx.shortest_simple_paths(Hn, s, d, weight="w"):
            if deadline is not None and time.perf_counter() > deadline:
                status = "timeout"; break
            wt = sum(Hn[u][v]["w"] for u, v in zip(path[:-1], path[1:]))
            if wt > Delta + _TOL:
                break
            mask = 0
            for v in path[1:-1]:
                mask |= 1 << _idx(v)
            cand.append((wt, mask, path))
            if len(cand) >= cap:
                status = "cap"; break
    except nx.NetworkXNoPath:
        return None, "ok"
    except nx.NodeNotFound:
        return None, "ok"
    if len(cand) < K:
        return None, status

    def dfs(start, chosen, used):
        if len(chosen) == K:
            return chosen
        for i in range(start, len(cand)):
            if deadline is not None and (i & 1023) == 0 and time.perf_counter() > deadline:
                raise Timeout()
            _, m, p = cand[i]
            if m & used:
                continue
            res = dfs(i + 1, chosen + [p], used | m)
            if res:
                return res
        return None
    try:
        res = dfs(0, [], 0)
    except Timeout:
        return None, "timeout"
    if res:
        return res, "ok"
    return None, status


_IDX = {}


def _idx(v):
    if v not in _IDX:
        _IDX[v] = len(_IDX)
    return _IDX[v]


# ---------------------------------------------------------------------------------------------- ILP (exact)
def plan_feasible_ilp(H, s, d, t, Delta, time_limit=10.0, workers=1):
    from ortools.sat.python import cp_model
    if s not in H or d not in H:
        return None, "ok"
    if _direct(H, s, d, Delta):
        return [[s, d]], "ok"
    K = t + 1
    Hn = H.copy()
    if Hn.has_edge(s, d):
        Hn.remove_edge(s, d)
    if s not in Hn or d not in Hn:
        return None, "ok"
    D_int = int(math.floor(Delta * SCALE + 0.5))
    m = cp_model.CpModel()
    arcs = []
    for u, v, a in Hn.edges(data=True):
        wi = int(round(a["w"] * SCALE))
        for (x, y) in ((u, v), (v, u)):
            if y == s or x == d:
                continue
            arcs.append((x, y, wi))
    x = {(k, i): m.NewBoolVar(f"x{k}_{i}") for k in range(K) for i in range(len(arcs))}
    nodes = list(Hn.nodes())
    for k in range(K):
        out = {v: [] for v in nodes}; inn = {v: [] for v in nodes}
        for i, (a_, b_, wi) in enumerate(arcs):
            out[a_].append(x[k, i]); inn[b_].append(x[k, i])
        m.Add(sum(out[s]) == 1); m.Add(sum(inn[d]) == 1)
        for v in nodes:
            if v != s and v != d:
                m.Add(sum(inn[v]) == sum(out[v]))
        m.Add(sum(wi * x[k, i] for i, (a_, b_, wi) in enumerate(arcs)) <= D_int)
    for v in nodes:
        if v != s and v != d:
            m.Add(sum(x[k, i] for k in range(K) for i, (a_, b_, wi) in enumerate(arcs) if b_ == v) <= 1)
    sol = cp_model.CpSolver()
    sol.parameters.max_time_in_seconds = time_limit
    sol.parameters.num_workers = workers
    st = sol.Solve(m)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        plan = []
        for k in range(K):
            nxt = {arcs[i][0]: arcs[i][1] for i in range(len(arcs)) if sol.Value(x[k, i])}
            p = [s]
            while p[-1] != d:
                p.append(nxt[p[-1]])
            plan.append(p)
        return plan, "ok"
    if st == cp_model.INFEASIBLE:
        return None, "ok"
    return None, "timeout"


# ---------------------------------------------------------------------------------------------- aggregate filter
def plan_feasible_agg(H, s, d, t, Delta):
    """Necessary condition: t+1 internally disjoint paths exist AND the cheapest such family has total weight <= (t+1)*Delta.
    Returns True if the filter ACCEPTS (i.e. the instance is not ruled out)."""
    if s not in H or d not in H:
        return False
    if _direct(H, s, d, Delta):
        return True
    K = t + 1
    Hn = H.copy()
    if Hn.has_edge(s, d):
        Hn.remove_edge(s, d)
    if s not in Hn or d not in Hn:
        return False
    F = nx.DiGraph()
    for v in Hn.nodes():
        cap = K if v in (s, d) else 1
        F.add_edge(("i", v), ("o", v), capacity=cap, weight=0)
    for u, v, a in Hn.edges(data=True):
        wi = int(round(a["w"] * SCALE))
        for (x, y) in ((u, v), (v, u)):
            if y == s or x == d:
                continue
            F.add_edge(("o", x), ("i", y), capacity=1, weight=wi)
    F.nodes[("o", s)]["demand"] = -K
    F.nodes[("i", d)]["demand"] = K
    try:
        cost = nx.min_cost_flow_cost(F)
    except nx.NetworkXUnfeasible:
        return False
    return cost <= K * int(math.floor(Delta * SCALE + 0.5))


# ---------------------------------------------------------------------------------------------- binary search over rho
def route_bmr_mp(G, s, d, t, Delta, R, solver="enum", cap=100_000, deadline_s=None, ilp_time=10.0):
    """-> dict(rho, plan, status) ; rho=None when refused.  status 'ok' means the answer is a proof (no cap/timeout hit)."""
    H = filter_rate(G, R)
    if s not in H or d not in H:
        return dict(rho=None, plan=None, status="ok", calls=0)
    rhos = sorted({a["r"] for _, _, a in H.edges(data=True)})
    lo, hi, best, bad = 0, len(rhos) - 1, None, "ok"
    calls = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        Gp = nx.Graph()
        Gp.add_edges_from((u, v, a) for u, v, a in H.edges(data=True) if a["r"] >= rhos[mid])
        calls += 1
        if solver == "enum":
            dl = None if deadline_s is None else time.perf_counter() + deadline_s
            plan, st = plan_feasible_enum(Gp, s, d, t, Delta, cap=cap, deadline=dl)
        else:
            plan, st = plan_feasible_ilp(Gp, s, d, t, Delta, time_limit=ilp_time)
        if plan is None and st != "ok":
            bad = st
        if plan:
            best = (rhos[mid], plan)
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        return dict(rho=None, plan=None, status=bad, calls=calls)
    return dict(rho=best[0], plan=best[1], status=bad, calls=calls)


# ---------------------------------------------------------------------------------------------- brute force (independent)
def brute_force_best_plan(G, s, d, t, Delta, R):
    """Ground truth: enumerate ALL simple s-d paths (nx.all_simple_paths, no Yen), keep those with sum(w) <= Delta and all
    rates >= R, then search all families of t+1 relay-disjoint ones; return best bottleneck (or None).  A direct link alone is a
    plan.  Independent of every routine above."""
    H = filter_rate(G, R)
    if s not in H or d not in H:
        return None
    best = None
    if H.has_edge(s, d) and H[s][d]["w"] <= Delta + _TOL:
        best = H[s][d]["r"]
    Hn = H.copy()
    if Hn.has_edge(s, d):
        Hn.remove_edge(s, d)
    if s not in Hn or d not in Hn:
        return best
    paths = []
    nodes = {v: i for i, v in enumerate(Hn.nodes())}
    for p in nx.all_simple_paths(Hn, s, d):
        wt = sum(Hn[u][v]["w"] for u, v in zip(p[:-1], p[1:]))
        if wt <= Delta + _TOL:
            rate = min(Hn[u][v]["r"] for u, v in zip(p[:-1], p[1:]))
            mask = 0
            for v in p[1:-1]:
                mask |= 1 << nodes[v]
            paths.append((mask, rate))
    K = t + 1
    res = [None]

    def rec(start, used, cnt, rmin):
        if cnt == K:
            if res[0] is None or rmin > res[0]:
                res[0] = rmin
            return
        for i in range(start, len(paths)):
            m, r = paths[i]
            if m & used:
                continue
            nr = min(rmin, r)
            if res[0] is not None and nr <= res[0]:
                continue
            rec(i + 1, used | m, cnt + 1, nr)
    rec(0, 0, 0, float("inf"))
    if res[0] is not None and (best is None or res[0] > best):
        best = res[0]
    return best


def plan_valid(G, plan, s, d, t, Delta, R):
    """independent check of a returned plan"""
    if plan is None:
        return False
    if len(plan) == 1 and len(plan[0]) == 2:
        a = G[plan[0][0]][plan[0][1]]
        return a["r"] >= R and a["w"] <= Delta + _TOL
    if len(plan) != t + 1:
        return False
    seen = set()
    for p in plan:
        if p[0] != s or p[-1] != d or len(set(p)) != len(p) or len(p) < 3:
            return False
        for v in p[1:-1]:
            if v in seen:
                return False
            seen.add(v)
        ws = 0.0
        for u, v in zip(p[:-1], p[1:]):
            if not G.has_edge(u, v):
                return False
            a = G[u][v]
            if a["r"] < R:
                return False
            ws += a["w"]
        if ws > Delta + _TOL:
            return False
    return True


def plan_rate(G, plan):
    return min(G[u][v]["r"] for p in plan for u, v in zip(p[:-1], p[1:]))
