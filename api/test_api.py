r"""
test_api.py
-----------
Tests the FastAPI endpoints and saves results to api_test_results.txt

Run with:
    python test_api.py

Make sure uvicorn is running first:
    .\env\Scripts\uvicorn api.main:app --reload
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"
OUTPUT_FILE = "api_test_results.txt"

results = []

def log(text: str):
    print(text)
    results.append(text)

def separator(title: str):
    line = "=" * 60
    log(f"\n{line}")
    log(f"  {title}")
    log(line)


# ── Header ────────────────────────────────────────────────────
separator("Agriculture Agent — API Test Results")
log(f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Base URL  : {BASE_URL}")


# ── Test 1: GET /docs (health check) ─────────────────────────
separator("Test 1: Server Health Check (GET /docs)")
try:
    r = requests.get(f"{BASE_URL}/docs", timeout=5)
    log(f"Status Code : {r.status_code}")
    log(f"Result      : {'PASS - Server is running' if r.status_code == 200 else 'FAIL'}")
except Exception as e:
    log(f"Result      : FAIL - {e}")


# ── Test 2: POST /chat ────────────────────────────────────────
separator("Test 2: POST /chat — Crop Recommendation Query")

payload1 = {
    "message": "What is the best crop for Punjab in Rabi season with loamy soil?",
    "thread_id": "test-thread-001"
}

log(f"Endpoint    : POST {BASE_URL}/chat")
log(f"Request     : {json.dumps(payload1, indent=2)}")

try:
    r = requests.post(
        f"{BASE_URL}/chat",
        json=payload1,
        timeout=60
    )
    log(f"Status Code : {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        log(f"Response    :")
        log(json.dumps(data, indent=2))
        log(f"Result      : PASS")
    else:
        log(f"Error       : {r.text}")
        log(f"Result      : FAIL")
except Exception as e:
    log(f"Result      : FAIL - {e}")


# ── Test 3: POST /chat — Yield Calculation ────────────────────
separator("Test 3: POST /chat — Yield Calculation Query")

payload2 = {
    "message": "Calculate yield for 50 acres of wheat with good soil quality in Punjab.",
    "thread_id": "test-thread-002"
}

log(f"Endpoint    : POST {BASE_URL}/chat")
log(f"Request     : {json.dumps(payload2, indent=2)}")

try:
    r = requests.post(
        f"{BASE_URL}/chat",
        json=payload2,
        timeout=60
    )
    log(f"Status Code : {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        log(f"Response    :")
        log(json.dumps(data, indent=2))
        log(f"Result      : PASS")
    else:
        log(f"Error       : {r.text}")
        log(f"Result      : FAIL")
except Exception as e:
    log(f"Result      : FAIL - {e}")


# ── Test 4: POST /stream ──────────────────────────────────────
separator("Test 4: POST /stream — Streaming Response")

payload3 = {
    "message": "What fertilizer should I use for cotton in sandy soil in Sindh?",
    "thread_id": "test-thread-003"
}

log(f"Endpoint    : POST {BASE_URL}/stream")
log(f"Request     : {json.dumps(payload3, indent=2)}")
log(f"Format      : Server-Sent Events (SSE)")
log(f"Stream chunks received:")

try:
    with requests.post(
        f"{BASE_URL}/stream",
        json=payload3,
        stream=True,
        timeout=60
    ) as r:
        log(f"Status Code : {r.status_code}")
        chunk_count = 0
        full_stream = []
        for line in r.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                full_stream.append(decoded)
                chunk_count += 1
                # Show first 5 chunks and last chunk
                if chunk_count <= 5:
                    log(f"  chunk {chunk_count:>3}: {decoded[:120]}")
                elif chunk_count == 6:
                    log(f"  ... (streaming) ...")

        log(f"  Total chunks received: {chunk_count}")
        if full_stream:
            log(f"  Last chunk: {full_stream[-1][:200]}")
        log(f"Result      : PASS" if chunk_count > 0 else "Result      : FAIL - No chunks received")

except Exception as e:
    log(f"Result      : FAIL - {e}")


# ── Test 5: Thread persistence check ─────────────────────────
separator("Test 5: Thread Persistence — Same thread_id follow-up")

payload4 = {
    "message": "Based on my previous question, what irrigation schedule do you recommend?",
    "thread_id": "test-thread-001"   # same as Test 2
}

log(f"Endpoint    : POST {BASE_URL}/chat")
log(f"thread_id   : test-thread-001 (reused from Test 2)")
log(f"Request     : {json.dumps(payload4, indent=2)}")

try:
    r = requests.post(
        f"{BASE_URL}/chat",
        json=payload4,
        timeout=60
    )
    log(f"Status Code : {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        log(f"Response    :")
        log(json.dumps(data, indent=2))
        log(f"Result      : PASS")
    else:
        log(f"Error       : {r.text}")
        log(f"Result      : FAIL")
except Exception as e:
    log(f"Result      : FAIL - {e}")


# ── Summary ───────────────────────────────────────────────────
separator("Test Summary")
log("All endpoint tests completed.")
log("Check above for PASS/FAIL status of each test.")


# ── Save to file ──────────────────────────────────────────────
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(results))

print(f"\n[DONE] Results saved to {OUTPUT_FILE}")