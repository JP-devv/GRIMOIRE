#!/usr/bin/env python3
"""Render pinned raw-ID requests for the paired C1 benchmark."""
import argparse
import json
from pathlib import Path

from c1 import MODEL, fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir()
    prompt, frozen = fixture()
    for name, tokens, output, expected in (
        ("frozen", prompt, 32, frozen),
        ("full-chunk", prompt[:2048], 32, None),
        ("long-decode", prompt, 256, frozen),
    ):
        request = {
            "model": MODEL,
            "prompt": tokens,
            "max_tokens": output,
            "temperature": 0,
            "ignore_eos": True,
            "stream": False,
            "return_token_ids": True,
            "add_special_tokens": False,
        }
        (args.output_dir / f"{name}.request.json").write_text(json.dumps(request) + "\n")
        (args.output_dir / f"{name}.prompt.ids").write_text(" ".join(map(str, tokens)) + "\n")
        if expected is not None:
            (args.output_dir / f"{name}.expected-prefix.ids").write_text(
                " ".join(map(str, expected)) + "\n"
            )


if __name__ == "__main__":
    main()
