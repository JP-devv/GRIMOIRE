#!/usr/bin/env python3
"""Capture the installed Qwen3NextAttention q/gate interpretation on CPU."""
import __future__
import argparse
import ast
import hashlib
import json
from pathlib import Path
import runpy
import struct
from types import MethodType, SimpleNamespace

import torch


def installed_methods(path, class_name, names, scope):
    source = path.read_bytes()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == class_name)
    nodes = [next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
             for name in names]
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec',
                   flags=__future__.annotations.compiler_flag)
    exec(code, scope)
    print(f'sha256({path.name})={hashlib.sha256(source).hexdigest()}')
    return {name: scope[name] for name in names}


def split_qgate(code, qkv, qsize, kvsize, heads):
    scope = dict(torch=torch, qkv=qkv, self=SimpleNamespace(q_size=qsize, kv_size=kvsize, num_heads=heads))
    exec(code, scope)
    assert scope['q'].shape == scope['gate'].shape == (qkv.size(0), qsize)
    return torch.stack((scope['q'], scope['gate']), dim=-1)


def capture_loader(vllm, code, q_proj, heads, dim, kv_heads):
    # Only host methods are needed. Stand-ins for parameter types not used by
    # the BF16 checkpoint let the actual loader choose its unquantized branch.
    unused = ['PerTensorScaleParameter', 'RowvLLMParameter', 'BasevLLMParameter',
              'BlockQuantScaleParameter', 'PackedColumnParameter', 'PackedvLLMParameter']
    scope = {name: type(name, (), {}) for name in unused} | {'torch': torch}
    linear = installed_methods(vllm / 'model_executor/layers/linear.py', 'QKVParallelLinear',
                               ['validate_shard_id', '_get_shard_offset_mapping',
                                '_get_shard_size_mapping', 'weight_loader_v2'], scope)
    param_methods = installed_methods(vllm / 'model_executor/parameter.py', '_ColumnvLLMParameter',
                                      ['load_qkv_weight'], scope)
    world, qkv_offset = 4, 0
    qrows, kvrows = 2 * heads * dim // world, dim
    weight = q_proj.T.contiguous()  # [checkpoint row][sentinel column]
    loaded = []
    split = []
    for rank in range(world):
        layer = SimpleNamespace(num_heads=2 * heads // world, head_size=dim, v_head_size=dim,
                                num_kv_heads=1, num_kv_head_replicas=world // kv_heads)
        param = SimpleNamespace(data=weight.new_full((qrows + 2 * kvrows, weight.shape[1]), -1),
                                tp_rank=rank, output_dim=0)
        for name, method in linear.items():
            setattr(layer, name, MethodType(method, layer))
        param.load_qkv_weight = MethodType(param_methods['load_qkv_weight'], param)
        layer.weight_loader_v2(param, weight, 'q')
        assert (param.data[qrows:] == -1).all(), 'Q loader touched K/V rows'
        qkv = param.data.T.contiguous()
        loaded.extend(qkv.narrow(1, qkv_offset, qrows).flatten().tolist())
        split.append(split_qgate(code, qkv, qrows // 2, kvrows, heads // world))
    return loaded, torch.cat(split, dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('vllm', type=Path, help='installed vllm package directory')
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--rank-rows', type=Path, help='also capture four installed loader Q shard results')
    args = parser.parse_args()
    cfg = json.loads((args.checkpoint / 'config.json').read_text())['text_config']
    heads, dim = cfg['num_attention_heads'], cfg['head_dim']
    assert cfg.get('indexer_n_heads') is None and (heads, dim) == (24, 256)
    weights = json.loads((args.checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
    keys = [k for k in weights if k.endswith('.self_attn.q_proj.weight')]
    assert len(keys) == 12
    shape = [2 * heads * dim, cfg['hidden_size']]
    for key in keys:
        with (args.checkpoint / weights[key]).open('rb') as shard:
            info = json.loads(shard.read(int.from_bytes(shard.read(8), 'little')))[key]
        assert info['shape'] == shape and info['dtype'] == 'BF16', key

    path = args.vllm / 'model_executor/models/qwen3_next.py'
    src = path.read_bytes()
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == 'Qwen3NextAttention')
    project = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_project_qkv_gate')
    # The fused branch is CUDA-only in this revision; XPU uses this split.
    split = next(n for n in project.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'self.attn_output_gate')
    code = compile(ast.Module(body=split.body, type_ignores=[]), str(path), 'exec')
    t = torch.arange(2).view(2, 1, 1, 1)
    h = torch.arange(heads).view(1, heads, 1, 1)
    half = torch.arange(2).view(1, 1, 2, 1)
    col = torch.arange(dim).view(1, 1, 1, dim)
    q_proj = (t * 1000000 + h * 10000 + half * 1000 + col).reshape(2, -1)
    qsize, kvsize = heads * dim, cfg['num_key_value_heads'] * dim
    qkv = torch.cat((q_proj, torch.full((2, 2 * kvsize), -1)), dim=1)
    unsharded = split_qgate(code, qkv, qsize, kvsize, heads)
    if args.rank_rows:
        rank_rows, reconstructed = capture_loader(args.vllm, code, q_proj, heads, dim, cfg['num_key_value_heads'])
        assert torch.equal(reconstructed, unsharded), 'installed loader+split differs from unsharded split'
        with args.rank_rows.open('xb') as f:
            f.write(struct.pack(f'<{len(rank_rows)}i', *rank_rows))
        output = reconstructed.flatten().tolist()
    else:
        output = unsharded.flatten().tolist()
    with args.output.open('xb') as f:
        f.write(struct.pack(f'<{len(output)}i', *output))
    version = runpy.run_path(str(args.vllm / '_version.py'))
    print(f"vLLM {version['__version__']} ({version['__commit_id__']}); sha256({path.name})={hashlib.sha256(src).hexdigest()}")
    print(f'{len(keys)} q_proj tensors: BF16 {shape}; {heads} × {dim}, {len(output) // 2} q/gate pairs')
    if args.rank_rows:
        print('four installed loader+split arms reconstruct the unsharded Q/gate result')


if __name__ == '__main__':
    main()
