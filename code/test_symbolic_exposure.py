"""test_symbolic_exposure.py - unit tests for the GF(2) verifier (run: pytest test_symbolic_exposure.py)."""
import itertools
import symbolic_exposure as X


def path(m):
    return ["s"] + [f"v{i}" for i in range(1, m)] + ["d"]


def test_space_basic():
    S = X.Space()
    assert S.add(0b101) and S.add(0b011) and not S.add(0b110)
    assert S.contains(0b110) and not S.contains(0b100)


def test_pad_not_reused_one_time():
    # sending the same content twice under fresh pads must give two independent ciphertexts
    W = X.World(["a", "b", "c"])
    W.link("a", "b")
    r = W.private("a")
    p1 = W.send("a", "b", r)
    p2 = W.send("a", "b", r)
    assert p1 != p2 and not W.external_exposed(r)


def test_p0_direct_link():
    inst = X.P0_xor_chain(path(1))
    assert X.analyze(inst)["disagreements"] == 0
    assert inst.world.knows("s", inst.key) and inst.world.knows("d", inst.key)


def test_p0_last_relay_exposes_and_s_cannot():
    for m in range(2, 8):
        inst = X.P0_xor_chain(path(m))
        a = X.analyze(inst)
        assert a["s_knows"] == 0 and a["d_knows"] == 1
        assert inst.world.exposed([f"v{m-1}"], inst.key)
        assert not inst.world.exposed(["v1"], inst.key) or m == 2
        assert a["disagreements"] == 0


def test_p2_every_relay_exposes():
    for m in range(2, 8):
        inst = X.P2_trusted_forward(path(m))
        a = X.analyze(inst)
        assert a["any_single_relay_exposes"] == 1 and a["min_exposing_coalition_size"] == 1
        assert a["disagreements"] == 0


def test_p4_threshold():
    paths = [["s", "a", "d"], ["s", "b", "d"]]
    inst = X.P4_xor_sharing(paths)
    assert not inst.world.exposed(["a"], inst.key) and not inst.world.exposed(["b"], inst.key)
    assert inst.world.exposed(["a", "b"], inst.key)


def test_p4_shared_relay_leaks():
    paths = [["s", "a", "x", "d"], ["s", "b", "x", "d"]]    # not internally disjoint (hmm: also shares link x-d)
    inst = X.P4_xor_sharing(paths)
    assert inst.world.exposed(["x"], inst.key)


def test_p3_parity_external():
    for m in range(2, 9):
        inst = X.P3_public_xor(path(m))
        assert inst.world.external_exposed(inst.key) == (m % 2 == 0)
        assert X.analyze(inst)["disagreements"] == 0


def test_lemma_a_random_linear_protocols():
    edges = [("s", "a"), ("a", "d"), ("s", "b"), ("b", "c"), ("c", "d")]
    n, chk, viol = X.lemmaA_check(edges, "s", "d", steps=30, seed=1, max_sep=2)
    assert chk > 0 and viol == 0
