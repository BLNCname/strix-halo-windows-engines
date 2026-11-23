"""Measure cold prefill and verify three markers using the native loopback HTTP API."""
import argparse
import json
from pathlib import Path
import random
import re
import secrets
import subprocess
import time
from urllib.request import build_opener, ProxyHandler, Request

LOCAL = build_opener(ProxyHandler({}))


def post(port, path, body, timeout=60):
    req = Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with LOCAL.open(req, timeout=timeout) as reply:
        return json.load(reply)


def prompt(root, target, seed, port):
    paths = subprocess.check_output(["git", "-C", root, "ls-files", "-z", "--", "src"], encoding="utf-8").split("\0")
    paths = sorted(p for p in paths if p.endswith((".cpp", ".hpp")) and "/vendor/" not in p)
    random.Random(seed).shuffle(paths)
    chunks, length = [], 0
    for path in paths:
        part = "\n// File: " + path + "\n" + (root / path).read_text(encoding="utf-8")
        chunks.append(part)
        length += len(part)
        if length >= 8 * target: break
    corpus = "".join(chunks)
    if not corpus: raise ValueError("The reference Gufo tree contains no C++ source.")
    labels = ("ALPHA", "BETA", "GAMMA")
    serials = [secrets.token_hex(5).upper() for _ in labels]
    size = min(len(corpus), 4 * target)
    for _ in range(8):
        content = corpus[:size]
        for pct, label, serial in reversed(list(zip((1, 50, 99), labels, serials))):
            pos = size * pct // 100
            content = content[:pos] + f"\n// Dispatch record {label}: {serial}\n" + content[pos:]
        text = ("Read the following C++ source archive.\n" + content
                + "\nEnd of archive. Reply only with the three dispatch serials for ALPHA, BETA, GAMMA in order.")
        actual = len(post(port, "/tokenize", {"content": text})["tokens"])
        if abs(actual - target) <= 24: return text, serials
        size = min(len(corpus), max(1, round(size * target / actual)))
    raise ValueError("Could not size the input to the requested token target.")


def measure(args):
    content, expected = prompt(args.source, args.target, args.seed, args.port)
    body = {"model": args.model, "messages": [{"role": "user", "content": content}], "temperature": 0,
            "max_tokens": 64, "reasoning_effort": "none", "cache_prompt": False, "cache_persist": False}
    start = time.monotonic()
    result = post(args.port, "/v1/chat/completions", body, timeout=1800)
    msg = result["choices"][0]["message"]["content"]
    found = re.findall(r"\b[0-9A-F]{10}\b", msg.upper())
    usage = result["usage"]
    report = {k: usage[k] for k in ("prompt_tokens", "completion_tokens", "cached_tokens", "prompt_tokens_per_second",
                                      "completion_tokens_per_second", "draft_tokens", "draft_tokens_accepted")}
    report.update(seed=args.seed, text_target=args.target, retrieval_passed=found == expected,
                  elapsed_s=round(time.monotonic() - start, 3))
    print(json.dumps(report), flush=True)
    return report["retrieval_passed"] and report["cached_tokens"] == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Gufo v0.35.0 C++ source checkout")
    parser.add_argument("--port", type=int, default=19443)
    parser.add_argument("--model", default="strix-q3-k-xl")
    parser.add_argument("--target", type=int, default=262000)
    parser.add_argument("--seed", type=int, default=43623)
    raise SystemExit(0 if measure(parser.parse_args()) else 1)
