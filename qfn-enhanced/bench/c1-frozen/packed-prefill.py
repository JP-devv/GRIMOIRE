#!/usr/bin/env python3
"""Rescore qualified packed-N64 rank logs using the frozen C1 phase parser."""
import csv
import json
from pathlib import Path
import sys

import c1

ARMS = ('1-control', '1-packed', '2-packed', '2-control')


def baseline(archive):
    if (archive / 'receipt').read_text().strip() != 'stage=verify exit=0':
        raise ValueError('Archived C1 baseline did not qualify')
    return c1.native([archive / '1-mtp-off' / f'matched-r{r}.log' for r in range(4)])


def compare(run, archive):
    if (run / 'receipt').read_text().strip() != 'stage=verify exit=0':
        raise ValueError('Packed-N64 parity or timing did not qualify')
    with (run / 'timing.tsv').open() as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        if reader.fieldnames != ['pair', 'backend', 'prefill_ms', 'decode_ms', 'prefill_tok_s', 'decode_tok_s']:
            raise ValueError('Bad packed-N64 timing header')
        rows = list(reader)
    if tuple(f"{row['pair']}-{row['backend']}" for row in rows) != ARMS:
        raise ValueError('Expected a complete control/packed/packed/control run')
    reference = baseline(archive)
    rates = {}
    for arm, row in zip(ARMS, rows):
        measured = c1.native([run / arm / f'r{r}.log' for r in range(4)])
        fields = (('prefill_ms', measured['prefill_s'] * 1000, 0.00051),
                  ('decode_ms', measured['decode_s'] * 1000, 0.00051),
                  ('prefill_tok_s', measured['cold_prefill_tok_s'], 0.00501),
                  ('decode_tok_s', measured['decode_tok_s'], 0.00501))
        if None in row or not all(abs(float(row[key]) - expected) <= rounding
                                  for key, expected, rounding in fields):
            raise ValueError(f'Timing table disagrees with per-rank phase logs: {arm}')
        rates[arm] = measured
    old = reference['cold_prefill_tok_s']
    pairs = []
    for pair in (1, 2):
        control = rates[f'{pair}-control']['cold_prefill_tok_s']
        packed = rates[f'{pair}-packed']['cold_prefill_tok_s']
        pairs.append({'pair': pair, 'control_cold_prefill_tok_s': control, 'packed_cold_prefill_tok_s': packed,
                      'packed_change_pct_vs_same_build_control': 100 * (packed / control - 1),
                      'packed_change_pct_vs_archived_c1_reference': 100 * (packed / old - 1)})
    return {'archived_native_c1_reference': reference, 'same_build_arms': rates, 'prefill_pairs': pairs}


if __name__ == '__main__':
    mode, *args = sys.argv[1:]
    if mode == 'reference' and len(args) == 1:
        result = baseline(Path(args[0]))
    elif mode == 'compare' and len(args) == 2:
        result = compare(*map(Path, args))
    else:
        sys.exit('Usage: packed-prefill.py reference BASELINE_ARCHIVE | compare RUN_DIR BASELINE_ARCHIVE')
    print(json.dumps(result, indent=2, allow_nan=False))
