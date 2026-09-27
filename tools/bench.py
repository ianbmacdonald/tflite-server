"""Latency per request for repeated identical /classify calls (warm-up vs steady state)."""
import json
import sys
import time
import urllib.request

BASE, TEXT, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
times = []
for _ in range(N):
    req = urllib.request.Request(BASE + "/classify", data=json.dumps({"input": TEXT}).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    urllib.request.urlopen(req, timeout=300).read()
    times.append((time.time() - t0) * 1000)
steady = sorted(times[1:])
print(f"first={times[0]:.0f}ms  steady median={steady[len(steady) // 2]:.0f}ms  min={steady[0]:.0f}ms  ({N} calls)")
