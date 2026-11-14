"""check_text.py - gate checks of blueprint 7.1 / 7.2 on a manuscript .tex (or .md) file.

Usage:  python check_text.py manuscript.tex [response_or_letter.tex ...]

1. Abstract word count (<= 250) between  % ABSTRACT_START / % ABSTRACT_END  (or <!-- ABSTRACT_START --> / <!-- ABSTRACT_END -->).
2. Highlights (<= 85 characters per \\item) between  % HIGHLIGHTS_START / % HIGHLIGHTS_END.
3. Red-line grep of 7.1 (every pattern must return 0 hits; hits are listed with line numbers).
4. "Never appear" strings of 7.2.
5. Numeric literals in the abstract, highlights, section 8 and section 10 that are not produced by a macro (section numbers and N values excepted).
6. Macros used in the text that results_macros.tex does not define.
Exit status 1 if any check fails.
"""
import re, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))

RED = [r"\bfirst\b", r"native", r"faithful", r"fairness", r"8,600", r"24[–-]29", r"9[–-]16", r"confirms", r"sub-millisecond",
       r"under one millisecond", r"within a single audit epoch", r"submitted response", r"earlier version", r"internal review",
       r"silently", r"counter-intuitive", r"⟨", r"TODO", r"\bx %", r"τ", r"53 B", r"117 B", r"δ_max", r"stricter test",
       r"corresponds to no standard state", r"Qpath", r"Qcast", r"Q-PATH-style", r"Q-CAST-style", r"Table S5", r"exact reduction",
       r"as expected", r"fewer than 2t\+1 relays", r"range 8", r"fits.*MTU", r"±21", r"100\s?\\?%"]
NEVER = [r"92\.8\s?\\?%\s*(vs|against)\s*78\.1", r"0\.064\s*/\s*0\.328", r"\+0\.19 to \+0\.26", r"4,334", r"14,473",
         r"SE = 0\.0000", r"S ≲ 2\.218", r"all hardware .{0,10}below 2\.687", r"Day 2 .{0,3} Day 1 = 0\.236", r"≈ 222 µs", r"3\.5×",
         r"weaker post-quantum", r"exact identity", r"ten shot-noise SD", r"2\.148", r"41\.2\s?\\?%", r"39\.6\s?\\?%", r"4\.66\s?\\?%",
         r"3\.05\s?\\?%", r"54\.8\s?\\?%", r"\[2\.467, 2\.482\]", r"0\.0003 / 0\.0038 / 0\.2435", r"Krenn et al\. removed",
         r"Q-CAST motivated", r"10\.5281/zenodo\.10826019"]

fail = 0


def report(ok, what, detail=""):
    global fail
    print(("PASS  " if ok else "FAIL  ") + what + (("  " + detail) if detail else ""))
    fail += (not ok)


def between(text, tag):
    m = re.search(rf"(?:%|<!--)\s*{tag}_START\s*(?:-->)?(.*?)(?:%|<!--)\s*{tag}_END", text, re.S)
    return m.group(1) if m else None


def plain(s):
    s = re.sub(r"(?<!\\)%.*", "", s)
    s = re.sub(r"\\[A-Za-z]+\*?(\[[^\]]*\])?", " ", s)
    return re.sub(r"[{}$~]", " ", s)


def macros_defined():
    p = os.path.join(HERE, "results_macros.tex")
    return set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", open(p).read())) if os.path.exists(p) else set()


def sections(text):
    parts = re.split(r"(?=\\section\*?\{)", text)
    return [p for p in parts if p.startswith("\\section")]


def numeric_literals(raw):
    """numbers left after removing macro calls, section/table/figure references and labels."""
    s = re.sub(r"(?<!\\)%.*", "", raw)
    s = re.sub(r"\\(?:ref|label|cite[a-z]*|eqref|input|include[a-z]*)\{[^}]*\}", " ", s)
    s = re.sub(r"\\(?:section|subsection|subsubsection|caption|textbf|emph)\*?\{", "{", s)
    s = re.sub(r"\\[A-Za-z]+\*?", " ", s)
    out = []
    for m in re.finditer(r"(?<![A-Za-z\\])(\d[\d,]*\.?\d*)", s):
        tok = m.group(1).rstrip(".,")
        ctx = s[max(0, m.start() - 12):m.start()]
        if re.search(r"(§|Section|Sec\.|Table|Tables|Fig\.|Figure|Theorem|Corollary|Proposition|Algorithm|Appendix|Definition)\s*$", ctx):
            continue
        if re.fullmatch(r"\d", tok):      # single digits: t, k, m, d, section numbers
            continue
        out.append(tok)
    return out


def main(paths):
    macros = macros_defined()
    for path in paths:
        print(f"\n=== {path} ===")
        text = open(path, encoding="utf-8").read()
        lines = text.split("\n")
        ab = between(text, "ABSTRACT")
        if ab is None:
            print("note  no ABSTRACT_START / ABSTRACT_END markers in this file")
        else:
            n = len(plain(ab).split())
            report(n <= 250, f"abstract words = {n} (limit 250)")
            lit = numeric_literals(ab)
            report(not lit, "abstract numeric literals not produced by a macro", str(lit) if lit else "")
        hl = between(text, "HIGHLIGHTS")
        if hl is not None:
            items = [i.strip() for i in re.split(r"\\item", hl)[1:]]
            bad = [(len(plain(i).strip()), plain(i).strip()[:40]) for i in items if len(plain(i).strip()) > 85]
            report(not bad, f"highlights: {len(items)} items, each <= 85 characters", str(bad) if bad else "")
            lit = numeric_literals(hl)
            report(not lit, "highlights numeric literals not produced by a macro", str(lit) if lit else "")
        secs = sections(text)
        for k in (8, 10):
            if len(secs) >= k:
                lit = numeric_literals(secs[k - 1])
                report(not lit, f"section {k} numeric literals not produced by a macro", (str(sorted(set(lit))[:30])) if lit else "")
        for label, pats in (("red-line (7.1)", RED), ("never-appear (7.2)", NEVER)):
            hits = []
            for p in pats:
                for i, l in enumerate(lines, 1):
                    if re.search(p, l, re.I if label.startswith("red") else 0):
                        hits.append((p, i, l.strip()[:80]))
            report(not hits, f"{label}: {len(pats)} patterns", "" if not hits else f"{len(hits)} hits")
            for h in hits[:40]:
                print("      ", h)
        used = set(re.findall(r"\\([A-Za-z]{4,})\b", text))
        guess = {u for u in used if re.match(r"(qSix|qNine|qEight|qEleven|qFifteen|qFive|qFour|expo|gap|alarm|lag|sens|drift|tamper|cells|collusion|nMin|eta|hw|fez|falseAlarm|firstDet|gate|rrFive|yen|trueSecure)", u)}
        miss = sorted(g for g in guess if g not in macros)
        report(not miss, "macros used but not defined in results_macros.tex", str(miss) if miss else "")
    print("\nRESULT:", "FAIL" if fail else "PASS")
    return 1 if fail else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
