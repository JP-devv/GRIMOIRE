#!/usr/bin/env python3
"""CPU fixture for the separate packed-N64 comparison report."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[2]


class PackedPrefillTest(unittest.TestCase):
    def test_comparison_and_rejections(self):
        inputs = 'prompt ids (2706): ' + ' '.join((HERE / 'prompt.ids').read_text().split()) + '\n'
        outputs = (HERE / 'frozen.ids').read_text().rstrip('\n') + '\n'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, run = root / 'archive', root / 'run'
            for path in (archive, run):
                path.mkdir()
                (path / 'receipt').write_text('stage=verify exit=0\n')

            def logs(path, stem, prefill):
                path.mkdir()
                for rank in range(4):
                    (path / f'{stem}r{rank}.log').write_text(
                        inputs + outputs + 'generation phases: prefill_tokens=2706 '
                        f'prefill_ms={prefill-rank}.000 decode_tokens=31 decode_ms={597+rank}.000\n')

            logs(archive / '1-mtp-off', 'matched-', 2706)
            table = 'pair\tbackend\tprefill_ms\tdecode_ms\tprefill_tok_s\tdecode_tok_s\n'
            for arm, ms in [('1-control', 2500), ('1-packed', 2000), ('2-packed', 2000), ('2-control', 2500)]:
                logs(run / arm, '', ms)
                pair, backend = arm.split('-')
                table += f'{pair}\t{backend}\t{ms}.000\t600.000\t{2706000/ms:.2f}\t51.67\n'
            timing = run / 'timing.tsv'
            timing.write_text(table)
            argv = [sys.executable, '-B', str(HERE / 'packed-prefill.py'), 'compare', str(run), str(archive)]
            result = json.loads(subprocess.check_output(argv, text=True))
            self.assertEqual(result['archived_native_c1_reference']['cold_prefill_tok_s'], 1000)
            for pair in result['prefill_pairs']:
                self.assertAlmostEqual(pair['packed_cold_prefill_tok_s'], 1353)
                self.assertAlmostEqual(pair['control_cold_prefill_tok_s'], 1082.4)
                self.assertAlmostEqual(pair['packed_change_pct_vs_same_build_control'], 25)
                self.assertAlmostEqual(pair['packed_change_pct_vs_archived_c1_reference'], 35.3)
            for path, bad, error in [
                (run / 'receipt', 'stage=gate-logit-parity exit=1\n', 'parity or timing did not qualify'),
                (timing, table.rsplit('\n', 2)[0] + '\n', 'Expected a complete'),
                (timing, table.replace('2500.000', '2000.000', 1), 'Timing table disagrees'),
                (run / '2-packed' / 'r3.log', inputs, 'Frozen IDs disagree'),
            ]:
                good = path.read_text()
                path.write_text(bad)
                proc = subprocess.run(argv, capture_output=True, text=True)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn(error, proc.stderr)
                self.assertEqual(proc.stdout, '')
                path.write_text(good)


if __name__ == '__main__':
    unittest.main()
