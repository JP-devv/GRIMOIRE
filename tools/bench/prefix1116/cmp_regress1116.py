#!/usr/bin/env python3
"""cmp_regress1116.py NEW_DIR BASE_DIR -- compare two regress_all_g0 result folders run by run (2026-11-16).

Answer text = the lines after the 'norms' banner line, without indented / prompt / id lines (regress_all_g0.sh's txt()).
Classes: SAME, DIFF, MISSING_MODEL (the model folder is gone: 'cannot open model directory'), FAILED (no answer and no
missing model), NOBASE.  Last line: the counts.
"""
import pathlib, re, sys

A, Bd = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])


def txt(p):
    out, on = [], False
    for line in p.read_text(errors='replace').splitlines():
        if not on:
            on = 'norms ' in line
            continue
        if not line.strip() or line[0].isspace() or re.search(r'prompt=|prompt ids|greedy ids', line):
            continue
        out.append(line)
    return out


def el(p):
    m = re.findall(r'elapsed=([0-9.]+)', p.read_text(errors='replace'))
    return float(m[-1]) if m else None


cnt = {}
for f in sorted(A.glob('*.log')):
    s = f.read_text(errors='replace')
    if 'cannot open model directory' in s:
        k = 'MISSING_MODEL'
    elif not (Bd / f.name).exists():
        k = 'NOBASE'
    elif not txt(f):
        k = 'FAILED'
    else:
        k = 'SAME' if txt(f) == txt(Bd / f.name) else 'DIFF'
    cnt[k] = cnt.get(k, 0) + 1
    a, b = el(f), el(Bd / f.name) if (Bd / f.name).exists() else None
    t = '' if a is None or b is None else '  %.3f s vs %.3f s (%+.1f%%)' % (a, b, 100 * (a - b) / b)
    print('%-14s %s%s' % (k, f.name, t))
print(' '.join('%s=%d' % kv for kv in sorted(cnt.items())))
