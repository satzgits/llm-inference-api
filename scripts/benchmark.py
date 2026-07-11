"""
Latency benchmark for the LLM Inference API.
Sends multiple prompts and reports p50/p95/p99 latency.
"""

import time
import json
import statistics
import httpx

BASE_URL = "http://localhost:8008"
PROMPTS = [
    "What is machine learning?",
    "Explain gradient descent.",
    "What is a transformer in NLP?",
    "Write a Python function to sort a list.",
    "Summarize: The quick brown fox jumps over the lazy dog.",
]
NUM_RUNS = 3


def benchmark():
    latencies = []
    for prompt in PROMPTS:
        for _ in range(NUM_RUNS):
            payload = {"prompt": prompt, "max_tokens": 50, "temperature": 0.1}
            start = time.perf_counter()
            try:
                resp = httpx.post(
                    f"{BASE_URL}/generate",
                    json=payload,
                    timeout=120
                )
                elapsed = (time.perf_counter() - start) * 1000
                if resp.status_code == 200:
                    data = resp.json()
                    latencies.append(elapsed)
                    content = data.get("response", "")[:60]
                    print(f"[{elapsed:8.0f}ms] {content}...")
                else:
                    print(f"[FAIL {resp.status_code}] {prompt[:40]}")
            except Exception as e:
                print(f"[ERROR] {e}")

    if latencies:
        latencies.sort()
        p50 = statistics.median(latencies)
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        print(f"\n--- Results ---")
        print(f"Requests: {len(latencies)}")
        print(f"p50: {p50:.0f}ms")
        print(f"p95: {p95:.0f}ms")
        print(f"p99: {p99:.0f}ms")
        print(f"Min: {min(latencies):.0f}ms")
        print(f"Max: {max(latencies):.0f}ms")


if __name__ == "__main__":
    benchmark()
