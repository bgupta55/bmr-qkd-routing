"""
bmr/verify_plan.py (revision 13 Nov 2026, X5): plan-level ledger = K (= t+1) per-path ledgers + ONE assembler signature binding all of them
(plus nonce, s, d, epoch). Per-path checks are those of verify_ledger_v2 (continuity, simplicity, attestor signatures, freshness, rate floor,
database check over [T_i, now]) with the BUDGET applied PER PATH and NO hop floor. Plan-level checks: number of paths == t+1 (a single
direct-link ledger is accepted on its own), internal vertex-disjointness.
"""
import hashlib
from .verify import ledger_digest, cert_message


def plan_digest(paths):
    h = hashlib.sha256()
    for entries in paths:
        h.update(ledger_digest(entries)); h.update(b"#")
    return h.digest()


def plan_session_message(paths, nonce, s, d, epoch):
    return plan_digest(paths) + b"|" + nonce + f"|{s}|{d}|{epoch}".encode()


def bind_plan(registry, paths, nonce, s, d, epoch, signer="__assembler__"):
    return {"nonce": nonce, "s": s, "d": d, "epoch": epoch, "sig": registry.sign[signer](plan_session_message(paths, nonce, s, d, epoch))}


def verify_plan(paths, session, registry, s, d, R, Delta, W, t, now, cert_db, expected_nonce):
    if not paths or any(not p for p in paths):
        return False, "empty ledger"
    if session["nonce"] != expected_nonce:
        return False, "session: nonce mismatch (replay)"
    if (session["s"], session["d"]) != (s, d):
        return False, "session: bound to another (s,d)"
    if not registry.verify["__assembler__"](plan_session_message(paths, session["nonce"], s, d, session["epoch"]), session["sig"]):
        return False, "session: signature/plan-digest mismatch"
    direct = len(paths) == 1 and len(paths[0]) == 1
    if not direct and len(paths) != t + 1:
        return False, "plan: wrong number of paths"
    relays_seen = set()
    for entries in paths:
        nodes = [entries[0]["link"][0]]
        for e in entries:
            u, v = e["link"]
            if nodes[-1] != u:
                return False, "continuity"
            nodes.append(v)
        if nodes[0] != s or nodes[-1] != d:
            return False, "endpoints"
        if len(set(nodes)) != len(nodes):
            return False, "simplicity"
        if not direct and len(nodes) < 3:
            return False, "plan: direct link inside a multi-path plan"
        for v in nodes[1:-1]:
            if v in relays_seen:
                return False, "plan: paths share a relay"
            relays_seen.add(v)
        for e in entries:
            if e["signer"] != registry.attestor(e["link"]):
                return False, "signer is not the link attestor"
            if not registry.verify[e["signer"]](cert_message(e["link"], e["r"], e["eps"], e["delta"], e["T"]), e["sig"]):
                return False, "certificate signature"
            age = now - e["T"]
            if age < 0 or age > W:
                return False, "stale"
        if any(e["r"] < R for e in entries):
            return False, "rate floor"
        if sum(e["eps"] + e["delta"] for e in entries) > Delta:
            return False, "budget (per path)"
        for e in entries:
            rec = cert_db.get((e["link"], e["T"]))
            if rec is not None and (abs(rec[0] - e["r"]) > 1e-8 or abs(rec[1] - e["eps"]) > 1e-15):
                return False, "db: record mismatch"
            for (lnk, Tp), (dr, de, dd) in cert_db.items():
                if lnk == e["link"] and e["T"] < Tp <= now and dr == 0.0:
                    return False, "db: superseded by DEGRADED in (T_i, now]"
    return True, "accepted"
