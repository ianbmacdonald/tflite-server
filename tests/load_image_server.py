#!/usr/bin/env python3
"""Concurrency and memory check for a running tflite-server image model.

Usage: load_image_server.py <port> <pid> <big.jpg> <grace_hopper.jpg> [clients] [requests]

Phase 1: `clients` threads each send `requests` copies of <big.jpg> (a ~4 MP
progressive JPEG) while `clients` more threads send PNG zip bombs, and /health
is polled. Every big-image reply must be 200 with a stable top-1, or 503 busy;
every bomb must be 400. Phase 2: 100 sequential grace_hopper requests, checking
that resident memory does not grow. Reports VmHWM/VmRSS from /proc/<pid>/status.
"""

import http.client
import json
import statistics
import struct
import sys
import threading
import time
import uuid
import zlib


def post(port, data):
    b = uuid.uuid4().hex
    body = (
        f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="x"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + data + f"\r\n--{b}--\r\n".encode()
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    c.request("POST", "/classify/image", body, {"Content-Type": f"multipart/form-data; boundary={b}"})
    r = c.getresponse()
    out = r.read()
    c.close()
    return r.status, json.loads(out)


def zip_bomb(inflated=272 << 20):
    c = zlib.compressobj(9)
    z = b"".join(c.compress(bytes(1 << 20)) for _ in range(inflated >> 20)) + c.flush()

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    ihdr = struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", z) + chunk(b"IEND", b"")


def mem(pid):
    out = {}
    for line in open(f"/proc/{pid}/status"):
        k, _, v = line.partition(":")
        if k in ("VmHWM", "VmRSS"):
            out[k] = int(v.split()[0]) // 1024
    return out


def main():
    port, pid, big_path, gh_path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    clients = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    requests = int(sys.argv[6]) if len(sys.argv) > 6 else 50
    big, gh, bomb = open(big_path, "rb").read(), open(gh_path, "rb").read(), zip_bomb()
    print(f"before: {mem(pid)} MiB")

    _, ref = post(port, big)
    ref_top = ref["predictions"][0]["index"]
    results = {"big_ok": 0, "big_busy": 0, "big_bad": [], "bomb_400": 0, "bomb_bad": [], "health_bad": 0, "health": 0}
    lock = threading.Lock()
    stop = threading.Event()

    def big_client():
        for _ in range(requests):
            try:
                status, body = post(port, big)
            except Exception as e:  # noqa: BLE001 - any transport error is a failure
                status, body = -1, {"error": repr(e)}
            with lock:
                if status == 200 and body["predictions"][0]["index"] == ref_top:
                    results["big_ok"] += 1
                elif status == 503:
                    results["big_busy"] += 1
                else:
                    results["big_bad"].append((status, str(body)[:120]))

    def bomb_client():
        for _ in range(requests):
            try:
                status, body = post(port, bomb)
            except Exception as e:  # noqa: BLE001
                status, body = -1, {"error": repr(e)}
            with lock:
                if status == 400 and body.get("error") == "image decode exceeded memory budget":
                    results["bomb_400"] += 1
                else:
                    results["bomb_bad"].append((status, str(body)[:120]))

    def health_poller():
        while not stop.is_set():
            try:
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
                c.request("GET", "/health")
                ok = c.getresponse().status == 200
            except Exception:  # noqa: BLE001
                ok = False
            with lock:
                results["health"] += 1
                results["health_bad"] += not ok
            time.sleep(0.5)

    t0 = time.time()
    threads = [threading.Thread(target=big_client) for _ in range(clients)]
    threads += [threading.Thread(target=bomb_client) for _ in range(clients)]
    poller = threading.Thread(target=health_poller)
    poller.start()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop.set()
    poller.join()
    after_load = mem(pid)
    print(
        f"phase 1 ({clients}+{clients} clients x {requests}, {time.time() - t0:.0f} s): "
        f"big ok {results['big_ok']} busy {results['big_busy']} bad {len(results['big_bad'])}; "
        f"bombs 400 {results['bomb_400']} bad {len(results['bomb_bad'])}; "
        f"health {results['health'] - results['health_bad']}/{results['health']} ok; mem {after_load} MiB"
    )
    for bad in (results["big_bad"] + results["bomb_bad"])[:5]:
        print("   ", bad)

    rss, inf = [], []
    for i in range(100):
        status, body = post(port, gh)
        if status != 200 or body["predictions"][0]["index"] != 653:
            print("phase 2: bad reply", status, str(body)[:200])
            sys.exit(1)
        inf.append(body["timings"]["inference_ms"])
        if i in (9, 99):
            rss.append(mem(pid)["VmRSS"])
    final = mem(pid)
    print(
        f"phase 2 (100 sequential grace_hopper): inference_ms median {statistics.median(inf):.2f} "
        f"max {max(inf):.2f}; VmRSS after 10 {rss[0]} MiB, after 100 {rss[1]} MiB; {final} MiB"
    )
    ok = (
        not results["big_bad"]
        and not results["bomb_bad"]
        and results["health_bad"] == 0
        and rss[1] - rss[0] <= 2
    )
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
