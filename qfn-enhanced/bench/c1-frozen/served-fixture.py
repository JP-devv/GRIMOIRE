"""Measure a raw-ID completion on an isolated HTTP endpoint."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("request", type=Path)
    parser.add_argument("expected", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    expected = list(map(int, args.expected.read_text().split()))
    start = perf_counter()
    with urlopen(Request(args.url, json.dumps(request).encode(),
                         {"Content-Type": "application/json"}), timeout=60) as response:
        result = json.load(response)
    completion_s = perf_counter() - start
    choice, = result["choices"]
    if (choice["token_ids"] != expected or choice["prompt_token_ids"] != request["prompt"]
            or choice["finish_reason"] != "length"):
        raise ValueError("Served completion does not match the frozen fixture")

    request.update(stream=True, return_token_ids=False)
    start = perf_counter()
    with urlopen(Request(args.url, json.dumps(request).encode(),
                         {"Content-Type": "application/json"}), timeout=60) as response:
        first_text_s = None
        usage = None
        for line in response:
            if not line.startswith(b"data: "):
                continue
            message = line[6:].strip()
            if message == b"[DONE]":
                break
            chunk = json.loads(message)
            if first_text_s is None and chunk["choices"][0].get("text"):
                first_text_s = perf_counter() - start
            if "usage" in chunk:
                usage = chunk["usage"]
        else:
            raise ValueError("Incomplete streamed response")
    if first_text_s is None or usage is None or usage["completion_tokens"] != len(expected):
        raise ValueError("Stream did not return the expected length and visible text")
    print(json.dumps({"nonstream_completion_s": completion_s,
                      "served_first_text_s": first_text_s,
                      "stream_usage": usage}, indent=2))


if __name__ == "__main__":
    main()
