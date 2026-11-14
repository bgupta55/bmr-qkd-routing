"""D-1 decision test: does the deposit router admit a direct link (m = 1)?

Instance: G = {s-d (strong), s-a, a-d}, t = 1  ->  hop floor m_min = 2t+1 = 3 as the deposit calls it.
Corollary 1 (exact): a route resists a coalition of <= t relays iff m = 1 or m >= 2t+1.
D-1(a) requires bmr_route to return the direct link s-d when the floor is "m = 1 or m >= 2t+1".
"""
import os
import sys

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bmr.route import bmr_route  # noqa: E402


def build():
    G = nx.Graph()
    G.add_edge("s", "d", r=0.30, w=1e-5, T=0)   # strong direct link
    G.add_edge("s", "a", r=0.10, w=1e-5, T=0)
    G.add_edge("a", "d", r=0.10, w=1e-5, T=0)
    return G


def test_current_router_with_deposit_floor():
    """Deposit call: m_min = 2t+1 = 3. Reports what the router does; D-1 is decided by this outcome."""
    res = bmr_route(build(), "s", "d", 0.01, 1e-3, 0, 10, m_min=3)
    print("bmr_route(m_min=3) ->", res)
    assert res is None or isinstance(res, tuple)


def test_no_floor_returns_direct():
    res = bmr_route(build(), "s", "d", 0.01, 1e-3, 0, 10, m_min=0)
    assert res is not None and res[1] == ["s", "d"], res


def test_direct_link_admitted():
    """D-1(a): with floor 'm = 1 or m >= 3' the direct link must be returned."""
    res = bmr_route(build(), "s", "d", 0.01, 1e-3, 0, 10, m_min=3)
    assert res is not None and res[1] == ["s", "d"], f"router refuses m = 1: {res}"


if __name__ == "__main__":
    test_no_floor_returns_direct()
    r = bmr_route(build(), "s", "d", 0.01, 1e-3, 0, 10, m_min=3)
    ok = r is not None and r[1] == ["s", "d"]
    print("D-1 RESULT:", "(a) admit - router returns direct link" if ok
          else "FAIL -> router refuses m = 1: patch (D-1(a)) or use policy-exclusion text (D-1(b))")
    sys.exit(0 if ok else 1)
