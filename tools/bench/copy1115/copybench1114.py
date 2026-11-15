#!/usr/bin/env python3
"""Copy / edit / new-text speed client (2026-11-14), modelled on the Discord "B70 inference: version -26" card:
edit-docstring, copy-code, copy-prose, edit-rename, code-new, review, essay, story.  Greedy (temperature 0),
thinking off, streaming.  Per request: TTFT, decode tok/s (first to last token), tokens, text.  With --users N
every category runs N requests at once (distinct tag in front) and the total is the sum of the users' own
decode rates (each first to last token), the card's "aggregate output while every user decodes".
Usage: copybench1114.py ENDPOINT MODEL SRC_TREE OUT.json [--users 1,4] [--reps 2]"""
import argparse, json, re, threading, time, urllib.request, pathlib

P = argparse.ArgumentParser()
P.add_argument('ep'); P.add_argument('model'); P.add_argument('tree'); P.add_argument('out')
P.add_argument('--users', default='1')
P.add_argument('--reps', type=int, default=2)
P.add_argument('--max-tokens', type=int, default=1536)
P.add_argument('--only', default='')
P.add_argument('--long-chars', type=int, default=0)   # >0: long-context set (book text of this many chars first)
O = P.parse_args()
T = pathlib.Path(O.tree)


def fence(code):
    return '```python\n' + code.rstrip('\n') + '\n```'


src_copy = (T / 'tools/dflash_compare.py').read_text()
src_rename = (T / 'tools/dflash_reference_dump.py').read_text()
src_doc = (T / 'tests/gguf/gen_golden.py').read_text()
src_review = (T / 'tools/bench/run_bb1112.py').read_text()
book = pathlib.Path('/mnt/storage/isos/grimoire-runs/bench-1003/book/1661-0.txt').read_text(errors='replace')
i = book.find('ADVENTURE II.')
prose = book[i:i + 5200] if i > 0 else book[200000:205200]
prose = prose[:prose.rfind('\n\n')]
fn = re.findall(r'^def ([a-z_][a-z0-9_]*)\(', src_rename, re.M)
old = max(fn, key=lambda f: src_rename.count(f)) if fn else 'main'
new = old + '_renamed' if not old.startswith('load') else 'fetch' + old[4:]

CATS = {
    'edit-docstring': 'Add a short one-line docstring to every function in this file that does not have one. Change '
                      'nothing else. Output the complete updated file only, no commentary.\n\n' + fence(src_doc),
    'copy-code': 'Repeat the following file exactly, character for character. Output only the file, no '
                 'commentary.\n\n' + fence(src_copy),
    'copy-prose': 'Copy the following text exactly, word for word, keeping the line breaks. Output only the '
                  'text.\n\n' + prose,
    'edit-rename': f'Rename the function `{old}` to `{new}` everywhere in this file (definition and every call). '
                   'Change nothing else. Output the complete updated file only, no commentary.\n\n' + fence(src_rename),
    'code-new': 'Write a complete Python module that implements a thread-safe LRU cache with per-entry time-to-live, '
                'a decorator that memoizes function calls with it, type hints, docstrings, and pytest unit tests.',
    'review': 'Review the following script. List the ten most important problems (bugs, robustness, style), each '
              'with the line it concerns and a concrete fix.\n\n' + fence(src_review),
    'essay': 'Write a 900-word essay on how the printing press changed science in Europe between 1450 and 1700.',
    'story': 'Write a 900-word short story about a lighthouse keeper who finds a message in a bottle.',
}
if O.long_chars:
    # the card's "users x 63K" rows: a long document first, then copy a passage of it / write new text
    j = max(0, book.find('I. A SCANDAL IN BOHEMIA'))
    doc = book[j:j + O.long_chars]
    assert len(doc) > O.long_chars // 2, 'book too short for --long-chars'
    k = doc.rfind('\n\n', 0, len(doc) - 6000)
    passage = doc[k:k + 4500]
    passage = passage[:passage.rfind('\n\n')].strip()
    CATS = {
        'copy-long': 'Here is a document.\n\n' + doc + '\n\nEnd of document.  Copy the passage that begins with "' +
                     passage[:80] + '" exactly, word for word, keeping its line breaks, for about 900 words.  Output only the '
                     'passage.',
        'write-long': 'Here is a document.\n\n' + doc + '\n\nEnd of document.  Write a new 900-word story in the same '
                      'style, with Holmes and Watson and a case that does not appear in the document.',
    }
    prose = passage
if O.only:
    CATS = {k: v for k, v in CATS.items() if k in O.only.split(',')}


def one(prompt, tag, res, k):
    body = {'model': O.model, 'stream': True, 'max_tokens': O.max_tokens, 'temperature': 0,
            'stream_options': {'include_usage': True},
            'chat_template_kwargs': {'enable_thinking': False},
            'messages': [{'role': 'user', 'content': f'[{tag}] ' + prompt}]}
    req = urllib.request.Request(O.ep + '/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.monotonic(); first = last = None; usage = {}; parts = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b'data: ') or b'[DONE]' in line:
                continue
            d = json.loads(line[6:])
            if d.get('usage'):
                usage = d['usage']
            for c in d.get('choices') or []:
                dl = c.get('delta') or {}
                s = (dl.get('reasoning_content') or '') + (dl.get('content') or '')
                if s:
                    now = time.monotonic(); first = first or now; last = now; parts.append(s)
    n = usage.get('completion_tokens', 0)
    res[k] = {'tag': tag, 'tokens': n, 'prompt_tokens': usage.get('prompt_tokens'), 'text': ''.join(parts),
              'ttft_ms': 1000 * (first - t0) if first else None, 'first': first, 'last': last,
              'decode_tps': (n - 1) / (last - first) if first and last and last > first else None}


def strip_fence(t):
    m = re.search(r'```[a-z]*\n(.*?)(```|$)', t, re.S)
    return m.group(1) if m else t


def exact(cat, text):
    want = {'copy-code': src_copy, 'copy-prose': prose, 'copy-long': prose}.get(cat)
    if want is None:
        return None
    got = strip_fence(text) if cat == 'copy-code' else text
    a, b = want.strip().split(), got.strip().split()
    same = sum(1 for x, y in zip(a, b) if x == y)
    return round(same / max(1, len(a)), 4)


rows = []
for users in [int(u) for u in O.users.split(',')]:
    for rep in range(O.reps):
        for cat, prompt in CATS.items():
            res = [None] * users
            th = [threading.Thread(target=one, args=(prompt, f'{cat} u{u} r{rep}', res, u)) for u in range(users)]
            for t in th:
                t.start()
            for t in th:
                t.join()
            firsts = [r['first'] for r in res if r['first']]
            lasts = [r['last'] for r in res if r['last']]
            row = {'category': cat, 'users': users, 'rep': rep,
                   'tokens': [r['tokens'] for r in res],
                   'decode_tps_each': [r['decode_tps'] for r in res],
                   'ttft_ms': [r['ttft_ms'] for r in res],
                   'prompt_tokens': res[0]['prompt_tokens'],
                   'exact': [exact(cat, r['text']) for r in res],
                   'texts': [r['text'] for r in res]}
            row['total_tps'] = sum(x for x in row['decode_tps_each'] if x)
            print(json.dumps({k: v for k, v in row.items() if k != 'texts'}), flush=True)
            rows.append(row)
            json.dump(rows, open(O.out, 'w'), indent=1)
print('COPYBENCH_DONE', flush=True)
