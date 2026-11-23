"""Exercise concurrent independent histories against a loaded native Gufo worker."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import re
import secrets
import threading
import time
from urllib.request import Request, build_opener, ProxyHandler


def run(port, model, sessions):
    base = f"http://127.0.0.1:{port}"
    start = threading.Barrier(sessions)
    def reply():
        marker = secrets.token_hex(6).upper()
        prompt = (f"Ticket {marker}. Read the following inventory.\n" +
                  "\n".join(f"Inventory row {i}. Gadget type {i % 13}, count {i % 17}." for i in range(240)) +
                  "\nReturn only the ticket identifier at the very beginning of this message.")
        payload = {"model": model, "messages": [{"role":"user","content":prompt}], "temperature":0, "max_tokens":32,
                   "reasoning_effort":"none", "cache_prompt":False, "cache_persist":False}
        start.wait(timeout=15)
        request = Request(base+"/v1/chat/completions", json.dumps(payload).encode(), {"Content-Type":"application/json"})
        with build_opener(ProxyHandler({})).open(request, timeout=240) as stream:
            result = json.load(stream)
        answer = result["choices"][0]["message"]["content"].strip()
        identifiers = re.findall(r"\b[0-9A-F]{12}\b", answer.upper())
        return {"correct":identifiers == [marker], "exact_format":answer == marker,
                "prompt":result["usage"]["prompt_tokens"], "output":result["usage"]["completion_tokens"]}
    peak = 0
    opener = build_opener(ProxyHandler({}))
    with ThreadPoolExecutor(max_workers=sessions) as pool:
        futures = [pool.submit(reply) for _ in range(sessions)]
        while not all(f.done() for f in futures):
            with opener.open(base+"/slots", timeout=5) as stream:
                peak = max(peak, sum(slot["is_processing"] for slot in json.load(stream)))
            time.sleep(.08)
        rows = [f.result() for f in futures]
    print(json.dumps({"peak_active_slots":peak, "requests": rows}), flush=True)
    return peak == sessions and all(r["correct"] for r in rows)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=19443)
    parser.add_argument("--model", default="strix-q3-k-xl")
    parser.add_argument("--sessions", type=int, default=4)
    args=parser.parse_args()
    raise SystemExit(0 if run(args.port,args.model,args.sessions) else 1)
