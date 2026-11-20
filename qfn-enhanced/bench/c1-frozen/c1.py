#!/usr/bin/env python3
"""CPU-only request rendering and receipt validation. No HTTP or GPU access."""
import json
import math
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
PROMPT = 2706
OUTPUT = 32
MODEL = 'qwen38-flash-next-c1'


def ids(words, n):
    if len(words) != n or any(re.fullmatch(r'[0-9]+', word) is None for word in words):
        raise ValueError(f'Expected exactly {n} token IDs')
    return [int(word) for word in words]


def fixture():
    prompt = ids((HERE / 'prompt.ids').read_text().split(), PROMPT)
    prefix, frozen = (HERE / 'frozen.ids').read_text().strip().split(':', 1)
    if prefix != f'greedy ids ({OUTPUT})':
        raise ValueError('Invalid frozen token receipt')
    return prompt, ids(frozen.split(), OUTPUT)


def result(prefill_s, decode_s):
    if not all(math.isfinite(s) and s > 0 for s in (prefill_s, decode_s)):
        raise ValueError('Invalid phase duration')
    return {'prefill_s': prefill_s, 'decode_s': decode_s,
            'cold_prefill_tok_s': PROMPT / prefill_s, 'decode_tok_s': (OUTPUT - 1) / decode_s}


def native(logs):
    if len(logs) != 4:
        raise ValueError('Expected four rank logs, in rank order')
    pp, tg = [], []
    prompt, frozen = fixture()
    prompt_line = f'prompt ids ({PROMPT}): ' + ' '.join(map(str, prompt))
    frozen_line = f'greedy ids ({OUTPUT}): ' + ' '.join(map(str, frozen))
    for name in logs:
        text = Path(name).read_text()
        lines = text.splitlines()
        if lines.count(prompt_line) != 1 or lines.count(frozen_line) != 1:
            raise ValueError(f'Frozen IDs disagree: {name}')
        phases = [line for line in lines if line.startswith('generation phases:')]
        if len(phases) != 1:
            raise ValueError(f'Expected one generation phase line: {name}')
        m = re.fullmatch(rf'generation phases: prefill_tokens={PROMPT} prefill_ms=(\d+\.\d+) '
                         rf'decode_tokens={OUTPUT - 1} decode_ms=(\d+\.\d+)', phases[0])
        if not m:
            raise ValueError(f'Bad phase receipt: {name}: {phases[0]}')
        a, b = (float(s) / 1000 for s in m.groups())
        result(a, b)
        pp.append(a)
        tg.append(b)
    return result(max(pp), max(tg))


def vllm(reply, before, after):
    prompt, frozen = fixture()
    response = json.loads(Path(reply).read_text())
    choices = response['choices']
    usage = response['usage']
    if (len(choices) != 1 or usage['prompt_tokens'] != PROMPT or usage['completion_tokens'] != OUTPUT or
            usage['prompt_tokens_details']['cached_tokens'] != 0 or
            choices[0].get('prompt_token_ids', response.get('prompt_token_ids')) != prompt or
            choices[0]['token_ids'] != frozen or choices[0]['finish_reason'] != 'length'):
        raise ValueError('vLLM reply did not match frozen IDs, lengths and zero cached tokens')

    def sample(text, metric):
        # The dedicated server has exactly this served model and one request.
        rows = re.findall(r'^' + re.escape(metric) + r'\{([^}\n]*)\}\s+([^\s]+)\s*$', text, re.M)
        values = [float(v) for labels, v in rows if f'model_name="{MODEL}"' in labels.split(',')]
        if len(values) != 1 or not math.isfinite(values[0]) or values[0] < 0:
            raise ValueError(f'Expected one {metric} series for {MODEL}')
        return values[0]

    old, new = (Path(name).read_text() for name in (before, after))
    elapsed = []
    for phase in ('prefill', 'decode'):
        m = f'vllm:request_{phase}_time_seconds'
        if sample(new, m+'_count') - sample(old, m+'_count') != 1:
            raise ValueError(f'{phase} histogram must advance by exactly one observation')
        elapsed.append(sample(new, m+'_sum') - sample(old, m+'_sum'))
    return result(*elapsed)


if __name__ == '__main__':
    mode, *args = sys.argv[1:]
    if mode == 'request' and not args:
        prompt, _ = fixture()
        value = {'model': MODEL, 'prompt': prompt, 'max_tokens': OUTPUT, 'temperature': 0,
                 'ignore_eos': True, 'stream': False, 'return_token_ids': True, 'add_special_tokens': False}
    elif mode == 'native':
        value = native(args)
    elif mode == 'vllm' and len(args) == 3:
        value = vllm(*args)
    else:
        sys.exit('Usage: c1.py request | native R0.log R1.log R2.log R3.log | vllm RESPONSE BEFORE AFTER')
    print(json.dumps(value, indent=2))
