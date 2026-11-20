#!/usr/bin/env python3
"""Read only checkpoint metadata; never load tensors or query GPUs."""
import hashlib
import json
from pathlib import Path
import sys

MODELS = Path('/home/haxor/models')
ROLES = {'target': 'qwen38-w4a16-snapshot', 'ple': 'qwen38-w4a16',
         'draft': 'qwen38-w4a16-mtp-snapshot'}


def inventory():
    result = {}
    for role, name in ROLES.items():
        root = MODELS / name
        index = root / 'model.safetensors.index.json'
        shards = sorted(set(json.loads(index.read_text())['weight_map'].values()))
        meta = [index.name] + [n for n in ('config.json', 'tokenizer.json',
                 'tokenizer_config.json', 'generation_config.json', 'chat_template.jinja',
                 'hybrid_sources.json', 'compact_sources.json') if (root / n).is_file()]
        files = {}
        for name in meta + shards:
            resolved = (root / name).resolve(strict=True)
            stat = resolved.stat()
            if name in shards:
                with resolved.open('rb') as f:
                    header_size = int.from_bytes(raw_size := f.read(8), 'little')
                    if len(raw_size) != 8 or not 0 < header_size <= min(stat.st_size - 8, 100_000_000):
                        raise ValueError(f'Invalid safetensors header: {resolved}')
                    header = f.read(header_size)
                    if len(header) != header_size:
                        raise ValueError(f'Truncated safetensors header: {resolved}')
                files[name] = {'path': str(resolved), 'size': stat.st_size,
                               'mtime_ns': stat.st_mtime_ns,
                               'header_sha256': hashlib.sha256(raw_size + header).hexdigest()}
            else:
                files[name] = {'path': str(resolved), 'sha256': hashlib.sha256(resolved.read_bytes()).hexdigest()}
        result[role] = {'root': str(root), 'files': files}
    return result


if __name__ == '__main__':
    locked = Path(__file__).with_name('checkpoint-lock.json')
    actual = inventory()
    if sys.argv[1:] == ['--print']:
        print(json.dumps(actual, indent=2, sort_keys=True))
    elif not sys.argv[1:]:
        if json.loads(locked.read_text()) != actual:
            sys.exit('Checkpoint metadata differs from checkpoint-lock.json')
        print('Target, PLE and draft checkpoint metadata match')
    else:
        sys.exit('Usage: checkpoint-lock.py [--print]')
