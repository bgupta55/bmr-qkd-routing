"""
symbolic_exposure.py  -  X1: GF(2) knowledge-closure verifier for key-delivery protocols
=======================================================================================

MODEL (information-theoretic, linear, passive coalitions)
---------------------------------------------------------
* Every link e has an independent uniform hop-key variable k_e. Every hop use of a one-time
  pad draws a FRESH independent variable that only the two endpoints of the link hold
  (an arbitrarily long hop-key stream, never reused). Every node may hold private variables.
* Every message is a public ciphertext (content XOR pad) on a public authenticated channel
  that every node and the adversary can read. Messages sent in the clear are public vectors.
* A message / key is a GF(2)-linear combination of variables, stored as a Python int bitmask.
* Knowledge of a set C of nodes = span( own variables of every member of C  U  all public
  vectors ). "Decryption" is automatic: the pad is in the span of both endpoints.
* A linear secret is exposed to C iff it lies in that span. For independent uniform
  variables this is exact: if the secret is not in the span it is perfectly hidden
  (every linear functional not in the span is uniform given the view).
* A sender can only put into a message something that lies in ITS OWN current span
  (own variables + public vectors); this is asserted when a message is sent.

Only the nodes on the used path(s) matter: variables of other links are independent of the key.
"""
import itertools
import random


# ----------------------------------------------------------------------------- linear algebra
class Space:
    """GF(2) vector space with an echelon basis keyed by the highest set bit."""

    def __init__(self, basis=None):
        self.basis = dict(basis) if basis else {}

    def reduce(self, v):
        b = self.basis
        while v:
            p = v.bit_length() - 1
            w = b.get(p)
            if w is None:
                return v
            v ^= w
        return 0

    def add(self, v):
        v = self.reduce(v)
        if v:
            self.basis[v.bit_length() - 1] = v
            return True
        return False

    def contains(self, v):
        return self.reduce(v) == 0

    def copy(self):
        return Space(self.basis)

    def vectors(self):
        return list(self.basis.values())

    def dim(self):
        return len(self.basis)


def intersect(A, B, width):
    """Basis of span(A) ∩ span(B) (Zassenhaus). A, B: lists of ints < 2**width."""
    S = Space()
    for a in A:
        S.add((a << width) | a)
    for b in B:
        S.add(b << width)
    # echelon rows whose high half vanished carry the intersection in the low half
    return [v for v in S.basis.values() if (v >> width) == 0]


# ----------------------------------------------------------------------------- world
class World:
    """Variables, per-node knowledge and the public transcript."""

    def __init__(self, nodes):
        self.nodes = list(nodes)
        self.nvars = 0
        self.own = {n: [] for n in self.nodes}         # variables each node holds from the start / by pads
        self.space = {n: Space() for n in self.nodes}   # own + public so far
        self.public = []
        self.pub_space = Space()
        self.link_key = {}
        self.n_base = None                              # set by freeze_base()

    # -- variables
    def new_var(self):
        v = 1 << self.nvars
        self.nvars += 1
        return v

    def grant(self, node, vec):
        self.own[node].append(vec)
        self.space[node].add(vec)

    def private(self, node):
        v = self.new_var()
        self.grant(node, v)
        return v

    def link(self, u, w):
        k = frozenset((u, w))
        if k not in self.link_key:
            v = self.new_var()
            self.link_key[k] = v
            self.grant(u, v)
            self.grant(w, v)
        return self.link_key[k]

    def freeze_base(self):
        """Call after all link keys / private variables exist and before any pad is created."""
        self.n_base = self.nvars

    # -- messages
    def publish(self, vec):
        self.public.append(vec)
        self.pub_space.add(vec)
        for n in self.nodes:
            self.space[n].add(vec)

    def send(self, u, w, content):
        """u sends `content` to w over their link under a FRESH pad (pad bookkeeping, pitfall 3)."""
        assert frozenset((u, w)) in self.link_key, "no link between %r and %r" % (u, w)
        assert self.space[u].contains(content), "sender %r does not know the content it sends" % (u,)
        pad = self.new_var()
        self.grant(u, pad)
        self.grant(w, pad)
        self.publish(content ^ pad)
        return pad

    def send_clear(self, u, content):
        assert self.space[u].contains(content), "sender %r does not know the content it sends" % (u,)
        self.publish(content)

    # -- knowledge
    def knows(self, node, vec):
        return self.space[node].contains(vec)

    def coalition_space(self, C):
        S = self.pub_space.copy()
        for n in C:
            for v in self.own[n]:
                S.add(v)
        return S

    def exposed(self, C, vec):
        return self.coalition_space(C).contains(vec)

    def external_exposed(self, vec):
        return self.pub_space.contains(vec)


# ----------------------------------------------------------------------------- protocols
class Instance:
    def __init__(self, world, key, s, d, paths, name):
        self.world, self.key, self.s, self.d, self.paths, self.name = world, key, s, d, paths, name
        self.relays = sorted({v for p in paths for v in p[1:-1]}, key=str)


def _build(paths):
    nodes = sorted({v for p in paths for v in p}, key=str)
    W = World(nodes)
    for p in paths:
        for a, b in zip(p[:-1], p[1:]):
            W.link(a, b)
    return W


def P0_xor_chain(path):
    """Paper's XOR chain as written: v_j sends P_j = P_{j-1} xor K_{j+1} to v_{j+1} under a pad of link j+1;
    d receives P_{m-1} = K_1 xor ... xor K_m. Nothing is returned to s."""
    W = _build([path])
    m = len(path) - 1
    K = [None] + [W.link(path[j - 1], path[j]) for j in range(1, m + 1)]
    if m == 1:
        return Instance(W, K[1], path[0], path[-1], [path], "P0")
    P = K[1]
    for j in range(1, m):
        P = P ^ K[j + 1]
        W.send(path[j], path[j + 1], P)
    return Instance(W, P, path[0], path[-1], [path], "P0")


def P1_chain_with_return(path):
    """P0 plus best-effort hop-by-hop return of K to s under fresh pads."""
    inst = P0_xor_chain(path)
    W, K, m = inst.world, inst.key, len(path) - 1
    for j in range(m, 0, -1):
        W.send(path[j], path[j - 1], K)
    inst.name = "P1"
    return inst


def P2_trusted_forward(path):
    """s draws r_s and forwards it hop by hop under fresh pads; key = r_s."""
    W = _build([path])
    r = W.private(path[0])
    for j in range(1, len(path)):
        W.send(path[j - 1], path[j], r)
    return Instance(W, r, path[0], path[-1], [path], "P2")


def P3_public_xor(path):
    """Public-XOR: z_i = K_i xor K_{i+1} broadcast in the clear by relay v_i (i = 1..m-1); key = XOR of all K_j."""
    W = _build([path])
    m = len(path) - 1
    K = [None] + [W.link(path[j - 1], path[j]) for j in range(1, m + 1)]
    for i in range(1, m):
        W.send_clear(path[i], K[i] ^ K[i + 1])
    key = 0
    for j in range(1, m + 1):
        key ^= K[j]
    return Instance(W, key, path[0], path[-1], [path], "P3")


def P4_xor_sharing(paths):
    """XOR sharing: s draws free shares Y_1..Y_n (key = XOR Y_i); share Y_i is forwarded hop by hop
    on path i under fresh pads. Paths may (wrongly) share relays; the checker then shows the leak."""
    W = _build(paths)
    s, d = paths[0][0], paths[0][-1]
    Y = [W.private(s) for _ in paths]
    key = 0
    for y in Y:
        key ^= y
    for y, p in zip(Y, paths):
        for a, b in zip(p[:-1], p[1:]):
            W.send(a, b, y)
    return Instance(W, key, s, d, [list(p) for p in paths], "P4")


# ----------------------------------------------------------------------------- theory predicates
# (independent of the encoding: derived by hand from the protocol definitions; see revisionplan Lemma A / Thm B / Prop 1')
def theory(inst):
    """-> dict(s_knows, d_knows, external, exposed=callable(C))"""
    name = inst.name
    paths = inst.paths
    path = paths[0]
    m = len(path) - 1
    if name == "P0":
        # Independent small model (no pads simulated): in the K-basis, a coalition C learns K_i for every link
        # incident to C, and Q_j = K_1^..^K_{j+1} for every public ciphertext it can decrypt, i.e. j with v_j or
        # v_{j+1} in C (v_m = d is never in C). Q_0 = K_1 is known to v_1.
        def exposed(C):
            C = set(C)
            v = lambda i: path[i]
            S = Space()
            for i in range(1, m + 1):                       # link i joins path[i-1], path[i]
                if v(i - 1) in C or v(i) in C:
                    S.add(1 << (i - 1))
            for j in range(1, m):                           # P_j sent by v_j to v_{j+1}, pad of link j+1
                if v(j) in C or (j + 1 < m and v(j + 1) in C):
                    q = 0
                    for i in range(1, j + 2):
                        q |= 1 << (i - 1)
                    S.add(q)
            return S.contains((1 << m) - 1)
        return dict(s_knows=(m == 1), d_knows=True, external=False, exposed=exposed)
    if name in ("P1", "P2"):
        return dict(s_knows=True, d_knows=True, external=False,
                    exposed=lambda C: (m >= 2 and len(C) > 0))
    if name == "P3":
        # external eavesdropper: z_1 ^ z_3 ^ ... ^ z_{m-1} = K_1 ^ ... ^ K_m for even m
        return dict(s_knows=True, d_knows=True, external=(m >= 2 and m % 2 == 0),
                    exposed=lambda C: (m >= 2 and len(C) > 0))
    if name == "P4":
        rel = [set(p[1:-1]) for p in paths]
        return dict(s_knows=True, d_knows=True, external=False,
                    exposed=lambda C: all(len(r & set(C)) > 0 for r in rel))
    raise ValueError(name)


# ----------------------------------------------------------------------------- analysis
def analyze(inst, max_coal=3):
    """Machine check of one instance. Returns a dict (one CSV row) incl. disagreements with `theory`."""
    W, key = inst.world, inst.key
    th = theory(inst)
    s_knows = W.knows(inst.s, key)
    d_knows = W.knows(inst.d, key)
    ext = W.external_exposed(key)
    relays = inst.relays
    n_coal = n_exp = 0
    by_size = {1: 0, 2: 0, 3: 0}
    min_size = None
    single = False
    dis = 0
    dis += int(s_knows != th["s_knows"]) + int(d_knows != th["d_knows"]) + int(ext != th["external"])
    base = W.pub_space
    # DFS over coalitions with incremental spaces
    def rec(start, chosen, S):
        nonlocal n_coal, n_exp, min_size, single, dis, by_size
        if chosen:
            n_coal += 1
            e = S.contains(key)
            if e != th["exposed"](set(chosen)):
                dis += 1
            if e:
                n_exp += 1
                by_size[len(chosen)] = by_size.get(len(chosen), 0) + 1
                if min_size is None or len(chosen) < min_size:
                    min_size = len(chosen)
                if len(chosen) == 1:
                    single = True
        if len(chosen) == max_coal:
            return
        for i in range(start, len(relays)):
            S2 = S.copy()
            for v in W.own[relays[i]]:
                S2.add(v)
            rec(i + 1, chosen + [relays[i]], S2)
    rec(0, [], base.copy())
    return dict(protocol=inst.name, s_knows=int(s_knows), d_knows=int(d_knows), external_exposed=int(ext),
                any_single_relay_exposes=int(single),
                min_exposing_coalition_size=("" if min_size is None else min_size),
                n_coalitions_checked=n_coal, n_exposed=n_exp, n_exposed_size1=by_size[1], n_exposed_size2=by_size[2], n_exposed_size3=by_size[3], disagreements=dis,
                n_paths=len(inst.paths), n_relays=len(relays))


# ----------------------------------------------------------------------------- Lemma A (linear instance) checks
def separators(adj, s, d, nodes, max_size=3):
    """All sets C ⊆ nodes∖{s,d}, |C| ≤ max_size whose removal disconnects s from d."""
    cand = [v for v in nodes if v not in (s, d)]

    def connected(removed):
        seen, stack = {s}, [s]
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y in removed or y in seen:
                    continue
                if y == d:
                    return True
                seen.add(y)
                stack.append(y)
        return False
    out = []
    for k in range(1, max_size + 1):
        for C in itertools.combinations(cand, k):
            if not connected(set(C)):
                out.append(C)
    return out


def lemmaA_check(edges, s, d, steps=40, seed=0, max_sep=3, privates=True, p_clear=0.1):
    """Random linear protocol on a graph; returns (#separators checked, #violations of  K_s ∩ K_d ⊆ K_C).

    K_x = span(own variables of x, public ciphertexts). Each step a random node u sends a random linear
    combination of its CURRENT knowledge to a random neighbour w under a fresh pad. After every step we
    verify, for every separator C (|C| ≤ max_sep), that the vectors both s and d can compute are
    computable by C (Lemma A in the linear model)."""
    rng = random.Random(seed)
    nodes = sorted({x for e in edges for x in e}, key=str)
    adj = {n: set() for n in nodes}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    W = World(nodes)
    for a, b in edges:
        W.link(a, b)
    if privates:
        for n in nodes:
            W.private(n)
    seps = separators(adj, s, d, nodes, max_sep)
    n_checked = n_viol = 0
    for _ in range(steps):
        u = rng.choice(nodes)
        w = rng.choice(sorted(adj[u], key=str))
        vecs = W.space[u].vectors()
        content = 0
        for v in vecs:
            if rng.random() < 0.5:
                content ^= v
        if content == 0:
            continue
        if rng.random() < p_clear:
            W.send_clear(u, content)          # public discussion in the clear
        else:
            W.send(u, w, content)
        width = W.nvars
        I = intersect(W.space[s].vectors(), W.space[d].vectors(), width)
        for C in seps:
            S = W.coalition_space(C)
            n_checked += 1
            if not all(S.contains(v) for v in I):
                n_viol += 1
    return len(seps), n_checked, n_viol


def maximal_flow_check(edges, s, d, relays_to_check):
    """Guide item 4: every node forwards everything (base-variable knowledge) over every link under a fresh pad,
    to a fixpoint. Checks (a) K_s ⊆ K_{v1} for each neighbour v1 of s that is a relay, (b) after every round
    K_s ∩ K_d ⊆ K_v for every separating single relay v. Returns (n_checks, n_violations)."""
    nodes = sorted({x for e in edges for x in e}, key=str)
    adj = {n: set() for n in nodes}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    W = World(nodes)
    for a, b in edges:
        W.link(a, b)
    for n in nodes:
        W.private(n)
    W.freeze_base()
    nb = W.n_base
    seps1 = separators(adj, s, d, nodes, 1)
    n_chk = n_bad = 0
    for _ in range(len(nodes) + 2):
        changed = False
        for u in nodes:
            for w in sorted(adj[u], key=str):
                for b in [v for p, v in W.space[u].basis.items() if p < nb]:
                    if not W.space[w].contains(b):
                        W.send(u, w, b)
                        changed = True
        width = W.nvars
        I = intersect(W.space[s].vectors(), W.space[d].vectors(), width)
        for C in seps1:
            S = W.coalition_space(C)
            n_chk += 1
            n_bad += int(not all(S.contains(v) for v in I))
        if not changed:
            break
    # K_s ⊆ K_v1 for relay neighbours of s (restricted to base variables)
    for v1 in adj[s]:
        if v1 == d:
            continue
        n_chk += 1
        S = W.coalition_space([v1])
        n_bad += int(not all(S.contains(v) for v in W.space[s].vectors()))
    return n_chk, n_bad
