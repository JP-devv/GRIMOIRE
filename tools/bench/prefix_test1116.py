#!/usr/bin/env python3
"""prefix_test1116.py ENDPOINT MODEL OUT.json [--chars N] -- the prompt cache on what agents do (2026-11-16).

  A: a long document + question, B: the SAME request again, C: the next turn (A's history + A's answer + a new
  question), D: C again.  Greedy, thinking off, 160 tokens.  Records each request's time to first token and text.
  Run once with GRIMOIRE_PREFIX_CACHE=0 and once with =1: every text must be identical across the two runs (the cache
  must not change an answer), and with the cache B, C and D should start in a fraction of A's time.
"""
import argparse, json, pathlib, time, urllib.request

P = argparse.ArgumentParser()
P.add_argument('ep'); P.add_argument('model'); P.add_argument('out')
P.add_argument('--chars', type=int, default=120000)
O = P.parse_args()
book = pathlib.Path('/mnt/storage/isos/grimoire-runs/bench-1003/book/1661-0.txt').read_text(errors='replace')
j = max(0, book.find('I. A SCANDAL IN BOHEMIA'))
doc = book[j:j + O.chars]


def chat(messages):
    body = {'model': O.model, 'messages': messages, 'max_tokens': 160, 'temperature': 0, 'stream': True,
            'stream_options': {'include_usage': True}, 'chat_template_kwargs': {'enable_thinking': False}}
    req = urllib.request.Request(O.ep.rstrip('/') + '/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time(); ttft = None; text = ''; usage = None
    for line in urllib.request.urlopen(req, timeout=3600):
        s = line.decode().strip()
        if not s.startswith('data: ') or s == 'data: [DONE]':
            continue
        d = json.loads(s[6:])
        if d.get('usage'):
            usage = d['usage']
        for c in d.get('choices') or []:
            dl = c.get('delta', {})
            piece = (dl.get('content') or '') + (dl.get('reasoning_content') or '')
            if piece and ttft is None:
                ttft = time.time() - t0
            text += piece
    return {'ttft_s': ttft, 'total_s': time.time() - t0, 'text': text, 'usage': usage}


q1 = [{'role': 'user', 'content': 'Here is a document.\n\n' + doc +
       '\n\nEnd of document.  In three sentences, who is Irene Adler and what does Holmes think of her?'}]
res = {}
res['A'] = chat(q1)
res['B'] = chat(q1)
q2 = q1 + [{'role': 'assistant', 'content': res['A']['text']},
           {'role': 'user', 'content': 'Now name two other cases from the document and say in one sentence each how '
                                       'Holmes solved them.'}]
res['C'] = chat(q2)
res['D'] = chat(q2)
for k, v in res.items():
    pt = (v['usage'] or {}).get('prompt_tokens')
    print('%s: prompt %s tokens, first token %.2f s, total %.2f s, text %d chars' % (k, pt, v['ttft_s'] or -1,
                                                                                     v['total_s'], len(v['text'])),
          flush=True)
print('B text == A text:', res['B']['text'] == res['A']['text'], ' D text == C text:', res['D']['text'] == res['C']['text'])
pathlib.Path(O.out).write_text(json.dumps(res, indent=1))
