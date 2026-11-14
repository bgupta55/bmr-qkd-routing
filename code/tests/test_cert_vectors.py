"""Five test vectors of the certificate (delta = 1e-6, ell = 1e4, phi = 0.99, lambda_EC = 0, floor k)."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bmr.certificate import build_certificate  # noqa: E402


def cert(S, n):
    return build_certificate("x", S, n, 1e-6, 10_000, 0.99, 0, 1)


def test_vectors():
    c = cert(2.687, 500);   assert c.k == 896 and math.isclose(c.eps, 1.6969e-2, rel_tol=1e-3), (c.k, c.eps)
    c = cert(2.687, 10_000); assert c.k == 3389 and math.isclose(c.eps, 2.94e-6, rel_tol=5e-3), (c.k, c.eps)
    c = cert(2.418, 10_000); assert c.k == 1393 and math.isclose(c.eps, 2.811e-3, rel_tol=1e-3), (c.k, c.eps)
    c = cert(2.687, 235);   assert c.k == 4 and math.isclose(c.eps, 0.4432, rel_tol=1e-3), (c.k, c.eps)
    assert cert(2.687, 100) is None
