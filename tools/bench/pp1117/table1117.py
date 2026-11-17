#!/usr/bin/env python3
"""Build Ian's full table from the tab1117-* llama-benchy results (run on the Tower, logs dir as argv[1])."""
import pathlib, re, sys

L = pathlib.Path(sys.argv[1])
SETUPS = [
    ('tab1117-q4-one', 'Qwen3.8-27B 4-bit', '1'),
    ('tab1117-q4-tp', 'Qwen3.8-27B 4-bit', '2'),
    ('tab1117-q8-one', 'Qwen3.8-27B 8-bit', '1'),
    ('tab1117-q8-tp', 'Qwen3.8-27B 8-bit (your DUAL)', '2'),
    ('tab1117b-fn4-one', 'Flash-Next 4-bit', '1'),
    ('tab1117b-fn4-tp', 'Flash-Next 4-bit', '2'),
    ('tab1117b-fn8-tp', 'Flash-Next 8-bit (your DUAL)', '2'),
]
# best published number from another engine on a B70: (pp, tg, source); None = nobody publishes it
OTHER = {
    ('Qwen3.8-27B 4-bit', '1', 4096): (2772, 121, 'pp inferzero / tg urakozz @0'),
    ('Qwen3.8-27B 4-bit', '1', 16384): (1845, None, 'urakozz'),
    ('Qwen3.8-27B 4-bit', '1', 32768): (1763, 61, 'urakozz'),
    ('Qwen3.8-27B 4-bit', '1', 65536): (1301, None, 'urakozz'),
    ('Qwen3.8-27B 4-bit', '1', 131072): (939, None, 'urakozz'),
    ('Qwen3.8-27B 4-bit', '2', 4096): (3217, 112, 'urakozz'),
    ('Qwen3.8-27B 4-bit', '2', 16384): (2937, None, 'urakozz'),
    ('Qwen3.8-27B 4-bit', '2', 32768): (None, 88, 'urakozz'),
    ('Qwen3.8-27B 8-bit (your DUAL)', '2', 4096): (1400, 21, 'vLLM FP8 TP2, this Tower 10-24'),
    ('Flash-Next 8-bit (your DUAL)', '2', 4096): (1358, 45, 'urakozz (4-bit)'),
    ('Flash-Next 8-bit (your DUAL)', '2', 16384): (1204, None, 'urakozz (4-bit, 17K)'),
    ('Flash-Next 8-bit (your DUAL)', '2', 60000): (954, 44, 'urakozz (4-bit, 62K)'),
    ('Flash-Next 4-bit', '2', 4096): (1358, 45, 'urakozz'),
    ('Flash-Next 4-bit', '2', 16384): (1204, None, 'urakozz (17K)'),
    ('Flash-Next 4-bit', '2', 60000): (954, 44, 'urakozz (62K)'),
    ('Flash-Next 4-bit', '1', 4096): (1414, None, 'inferzero'),
}


def rows(d):
    f = L / ('copy1114-' + d) / 'benchy.md'
    if not f.exists():
        return {}
    out, cur = {}, None
    for line in f.read_text().splitlines():
        c = [x.strip() for x in line.split('|')]
        if len(c) < 4:
            continue
        m = re.match(r'pp(\d+)', c[2])
        if m:
            cur = int(m.group(1))
            out.setdefault(cur, {})['pp'] = float(c[3].split()[0])
        elif c[2].startswith('tg') and cur:
            out[cur]['tg'] = float(c[3].split()[0])
    return out


def ctx(n):
    return f'{n // 1024}K' if n % 1024 == 0 else f'{round(n / 1000)}K'


print('| Model | GPUs | Prompt length | Reading (pp) Grimoire | Reading best other | Writing (tg) Grimoire | Writing best other | Who wins |')
print('|---|---|---|---|---|---|---|---|')
for tag, model, gpus in SETUPS:
    r = rows(tag)
    if not r:
        print(f'| {model} | {gpus} | -- | did not run | | | | |')
        continue
    for n in sorted(r):
        pp, tg = r[n].get('pp'), r[n].get('tg')
        o = OTHER.get((model, gpus, n))
        opp, otg = (o[0], o[1]) if o else (None, None)
        wins = []
        if opp: wins.append('pp ' + ('Grimoire' if pp and pp >= opp else 'them'))
        if otg: wins.append('tg ' + ('Grimoire' if tg and tg >= otg else 'them'))
        f = lambda v: f'{v:,.0f}' if v else '--'
        print(f'| {model} | {gpus} | {ctx(n)} | {f(pp)} | {f(opp)} | {f(tg)} | {f(otg)} | {", ".join(wins) or "nobody else publishes this"} |')
