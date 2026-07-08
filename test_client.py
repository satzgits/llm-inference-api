import httpx

BASE = "http://localhost:8000"
client = httpx.Client(timeout=120.0)

def test_health():
    r = client.get(f"{BASE}/health")
    print(f"Health: {r.status_code} -> {r.json()}")

def test_generate():
    r = client.post(f"{BASE}/generate", json={
        "prompt": "What is quantum computing? Answer in 2 sentences.",
        "max_tokens": 100
    })
    print(f"Generate: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  Response: {data['response'][:150]}...")
        print(f"  Latency: {data['inference_time_ms']}ms")
        print(f"  Tokens: {data['tokens_used']}")

def test_embed():
    r = client.post(f"{BASE}/embed", json={"input": "Hello world"})
    print(f"Embed: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  Embedding dims: {len(data['embedding'])}")
        print(f"  Latency: {data['inference_time_ms']}ms")

if __name__ == "__main__":
    print("=== Testing LLM Inference API ===\n")
    test_health()
    test_generate()
    test_embed()
    print("\n=== Done ===")
