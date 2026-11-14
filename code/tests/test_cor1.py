"""Corollary 1 by brute force: a simple m-hop route is exposed to a coalition of <= t relays
iff 2 <= m <= 2t (m = 1 has no relay; m >= 2t+1 needs more than t relays to cover every link)."""
import itertools


def min_relay_cover(m):
    relays = list(range(1, m))                 # v1..v_{m-1}; s = v0 and d = v_m honest
    for k in range(len(relays) + 1):
        for c in itertools.combinations(relays, k):
            s = set(c)
            if all((i - 1 in s) or (i in s) for i in range(1, m + 1)):   # link i joins v_{i-1}, v_i
                return k
    return None


def exposed(m, t):
    return 2 <= m <= 2 * t


def test_closed_form():
    for t in range(0, 6):
        for m in range(1, 14):
            c = min_relay_cover(m)
            assert exposed(m, t) == (c is not None and c <= t), (m, t)
    assert min_relay_cover(1) is None
    for m in range(2, 14):
        assert min_relay_cover(m) == (m + 1) // 2
