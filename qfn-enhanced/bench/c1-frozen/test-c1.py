#!/usr/bin/env python3
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import c1


class Receipts(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prompt, self.output = c1.fixture()

    def file(self, path, text):
        f = self.root / path
        f.write_text(text)
        return f

    def test_native_slowest_each_phase(self):
        logs = []
        for rank in range(4):
            logs.append(self.file(str(rank), f'prompt ids (2706): {" ".join(map(str, self.prompt))}\n'
                f'greedy ids (32): {" ".join(map(str, self.output))}\n'
                f'generation phases: prefill_tokens=2706 prefill_ms={997+rank}.000 '
                f'decode_tokens=31 decode_ms={2000-rank}.000\n'))
        self.assertEqual(c1.native(logs), c1.result(1, 2))
        logs[3].write_text(logs[3].read_text().replace('greedy ids', 'wrong IDs'))
        with self.assertRaises(ValueError):
            c1.native(logs)

    def test_vllm_metrics_and_parity(self):
        response = {'choices': [{'prompt_token_ids': self.prompt, 'token_ids': self.output, 'finish_reason': 'length'}],
                    'usage': {'prompt_tokens': 2706, 'completion_tokens': 32,
                              'prompt_tokens_details': {'cached_tokens': 0}}}
        reply = self.file('reply', json.dumps(response))

        def metrics(count, prefill, decode):
            return ''.join(f'vllm:request_{phase}_time_seconds_{suffix}{{engine="0",model_name="{c1.MODEL}"}} {value}\n'
                           for phase, total in [('prefill', prefill), ('decode', decode)]
                           for suffix, value in [('count', count), ('sum', total)])

        before = self.file('before', metrics(2, 1, 2))
        after = self.file('after', metrics(3, 2, 4))
        self.assertEqual(c1.vllm(reply, before, after), c1.result(1, 2))
        response['usage']['prompt_tokens_details']['cached_tokens'] = 5
        reply.write_text(json.dumps(response))
        with self.assertRaises(ValueError):
            c1.vllm(reply, before, after)
        response['usage']['prompt_tokens_details']['cached_tokens'] = 0
        reply.write_text(json.dumps(response))
        after.write_text(metrics(4, 2, 4))
        with self.assertRaises(ValueError):
            c1.vllm(reply, before, after)


if __name__ == '__main__':
    unittest.main()
