"""Validity of the tight Hoeffding slack eta_tight and the ratio eta_union / eta_tight."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bmr.bounds import eta_tight  # noqa: E402

rng = np.random.default_rng(20261019)


def violation_rate(S, n, delta, reps=400_000):
    E = S / 4.0
    p = (1 + E) / 2
    k = rng.binomial(n, p, size=(reps, 4))
    S_hat = ((2 * k - n) / n).sum(axis=1)
    return (S_hat <= S - math.sqrt(8 * math.log(1 / delta) / n)).mean()


def test_valid():
    for n in (100, 500, 2000):
        assert violation_rate(2.687, n, 0.1) < 0.1


def test_ratio():
    r = math.sqrt(32 * math.log(8 / 1e-6)) / math.sqrt(8 * math.log(1 / 1e-6))
    assert math.isclose(r, 2.145, abs_tol=2e-3)


def test_library_matches_formula():
    for n in (100, 235, 500, 10_000):
        assert math.isclose(eta_tight(n, 1e-6), math.sqrt(8 * math.log(1 / 1e-6) / n), rel_tol=1e-12)
