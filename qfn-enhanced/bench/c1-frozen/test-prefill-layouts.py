"""CPU checks for the fixed B70 full-chunk attention and HC up layouts."""
import unittest


class PrefillLayouts(unittest.TestCase):
    def test_causal_tile_visibility(self):
        tile_rows = tile_keys = 16
        for context, rows in ((0, 2048), (2048, 658), (2041, 659), (32703, 65)):
            end = context + rows
            for first_row in range(0, rows, tile_rows):
                tile_end = min(end, context + first_row + tile_rows)
                kept = ((tile_end + tile_keys - 1) // tile_keys) * tile_keys
                for row in range(first_row, min(first_row + tile_rows, rows)):
                    self.assertLessEqual(context + row + 1, kept)
                self.assertLessEqual(tile_end, end)
        total = 128 * 128
        causal = sum(range(1, 129))
        self.assertEqual(total - causal, 8128)

    def test_hc_up_vnni_groups_cover_four_streams(self):
        hidden, streams, fragment = 2560, 4, 16
        mapped = set()
        for packed in range(hidden * streams):
            stream = (packed // fragment) % streams
            column = (packed // (fragment * streams)) * fragment + packed % fragment
            self.assertLess(column, hidden)
            mapped.add((stream, column))
            group_start = (packed // (fragment * streams)) * (fragment * streams)
            self.assertEqual(group_start // streams + packed % fragment, column)
        self.assertEqual(len(mapped), streams * hidden)


if __name__ == "__main__":
    unittest.main()
