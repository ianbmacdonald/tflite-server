#!/usr/bin/env python3
"""End-to-end checks for a running tflite-server that hosts an image model.

Usage:
  e2e_image_server.py <port> <grace_hopper.jpg>                  # defaults
  e2e_image_server.py <port> <grace_hopper.jpg> --pixel-cap-only  # server started with --max-image-pixels 1000

Standard library only. Expects the TF MobileNetV2 1.0 224 model directory made
by tools/make_image_model_dir.py and the default limits (16 MiB images, one decode slot,
at least 3 HTTP threads). The busy check holds both request slots with slow uploads and
takes about 32 s.
"""

import base64
import http.client
import io
import json
import socket
import struct
import sys
import uuid
import zlib

REF_INDEX, REF_SCORE = 653, 0.803491
failures = 0


def check(ok, what, detail=""):
    global failures
    print(f"{'ok  ' if ok else 'FAIL'} {what}{(': ' + detail) if detail and not ok else ''}")
    failures += not ok


def request(port, method, path, body=b"", headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    c.request(method, path, body, headers or {})
    r = c.getresponse()
    data = r.read()
    try:
        return r.status, json.loads(data)
    except ValueError:
        return r.status, data


def multipart(parts):
    b = uuid.uuid4().hex
    out = io.BytesIO()
    for name, value, filename in parts:
        out.write(f"--{b}\r\nContent-Disposition: form-data; name=\"{name}\"".encode())
        if filename:
            out.write(f'; filename="{filename}"\r\nContent-Type: application/octet-stream'.encode())
        out.write(b"\r\n\r\n" + value + b"\r\n")
    out.write(f"--{b}--\r\n".encode())
    return out.getvalue(), {"Content-Type": f"multipart/form-data; boundary={b}"}


def post_image(port, data, top_k=None, name="image"):
    parts = [(name, data, "x.bin")]
    if top_k is not None:
        parts.append(("top_k", str(top_k).encode(), None))
    body, headers = multipart(parts)
    return request(port, "POST", "/classify/image", body, headers)


def post_json(port, obj):
    return request(port, "POST", "/classify/image", json.dumps(obj).encode(), {"Content-Type": "application/json"})


def png(width, height, raw_rows=None, idat=None):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    if idat is None:
        raw = raw_rows if raw_rows is not None else b"".join(b"\x00" + bytes(width * 3) for _ in range(height))
        idat = zlib.compress(raw)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def zip_bomb(inflated=272 << 20):
    c = zlib.compressobj(9)
    block = bytes(1 << 20)
    z = b"".join(c.compress(block) for _ in range(inflated >> 20)) + c.flush()
    return png(64, 64, idat=z)


def oversize_raw(port, total):
    """Send a declared-length body over a raw socket; return the HTTP status line."""
    s = socket.create_connection(("127.0.0.1", port), timeout=300)
    s.sendall(
        f"POST /classify/image HTTP/1.1\r\nHost: x\r\nContent-Type: application/octet-stream\r\n"
        f"Content-Length: {total}\r\n\r\n".encode()
    )
    chunk = bytes(1 << 20)
    try:
        sent = 0
        while sent < total:
            n = min(len(chunk), total - sent)
            s.sendall(chunk[:n])
            sent += n
    except (BrokenPipeError, ConnectionResetError):
        pass
    try:
        line = s.recv(4096).split(b"\r\n", 1)[0].decode()
    except ConnectionResetError:
        line = "reset"
    s.close()
    return line


def chunked_raw(port, total, chunk_size=1 << 20):
    """Send a chunked (no Content-Length) JSON body of `total` bytes; return the HTTP status line."""
    s = socket.create_connection(("127.0.0.1", port), timeout=300)
    s.sendall(b"POST /classify/image HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
              b"Transfer-Encoding: chunked\r\n\r\n")
    try:
        sent = 0
        while sent < total:
            n = min(chunk_size, total - sent)
            s.sendall(f"{n:x}\r\n".encode() + b"a" * n + b"\r\n")
            sent += n
        s.sendall(b"0\r\n\r\n")
    except (BrokenPipeError, ConnectionResetError):
        pass
    try:
        line = s.recv(4096).split(b"\r\n", 1)[0].decode()
    except ConnectionResetError:
        line = "reset"
    s.close()
    return line


def trickle(port, stop):
    """Hold a request slot: send headers, then one body byte every 2 s (under the 5 s read timeout)."""
    s = socket.create_connection(("127.0.0.1", port), timeout=300)
    s.sendall(b"POST /classify/image HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
              b"Content-Length: 1000\r\n\r\n")
    sent = 0
    while not stop.is_set() and sent < 999:
        s.sendall(b" ")
        sent += 1
        stop.wait(2)
    s.close()


def busy_503(port, gh, holders=2):
    """With the default --max-concurrent-decodes 1 there are 2 request slots; hold both, then a
    third request waits 30 s for one and gets 503."""
    import threading
    import time
    stop = threading.Event()
    threads = [threading.Thread(target=trickle, args=(port, stop)) for _ in range(holders)]
    for t in threads:
        t.start()
    time.sleep(2)
    t0 = time.monotonic()
    status, body = post_image(port, gh)
    waited = time.monotonic() - t0
    stop.set()
    for t in threads:
        t.join()
    return status, body, waited


def main():
    port, gh_path = int(sys.argv[1]), sys.argv[2]
    gh = open(gh_path, "rb").read()

    if "--pixel-cap-only" in sys.argv:
        status, body = post_image(port, png(64, 64))
        check(status == 400 and "limit is 1000" in body.get("error", ""), "64x64 PNG over --max-image-pixels 1000 -> 400", str(body))
        status, body = post_image(port, png(20, 20))
        check(status == 200, "20x20 PNG under the cap -> 200", str(body)[:200])
        sys.exit(1 if failures else 0)

    status, health = request(port, "GET", "/health")
    check(status == 200 and health.get("task") == "image-classification", "/health task", str(health))

    status, body = post_image(port, gh)
    top = body["predictions"][0] if status == 200 else {}
    check(status == 200 and top.get("index") == REF_INDEX and abs(top.get("score", 0) - REF_SCORE) < 0.02,
          f"multipart grace_hopper top-1 653 near {REF_SCORE}", str(body)[:300])
    if status == 200:
        print(f"     top-1 {top['index']} {top['label']} {top['score']:.6f}; timings {body['timings']}")
        check(len(body["predictions"]) == 5, "default top_k is 5 (manifest top_k_default)")
        check(body["input"] == {"width": 512, "height": 600}, "input size reported")

    status, body = post_image(port, gh, top_k=3, name="file")
    scores = [p["score"] for p in body.get("predictions", [])]
    check(status == 200 and len(scores) == 3 and scores == sorted(scores, reverse=True),
          "part named 'file', top_k=3, sorted", str(body)[:200])
    check(status == 200 and set(body["labels"]) == {p["label"] for p in body["predictions"]}, "labels map matches predictions")

    status, body = post_image(port, gh, top_k=5000)
    check(status == 200 and len(body["predictions"]) == 1001, "top_k 5000 clamps to 1001 labels", str(status))

    b64 = base64.b64encode(gh).decode()
    status, body = post_json(port, {"image": b64, "top_k": 2})
    check(status == 200 and body["predictions"][0]["index"] == REF_INDEX and len(body["predictions"]) == 2,
          "JSON base64", str(body)[:200])
    status, body = post_json(port, {"image": "data:image/jpeg;base64," + b64[:76] + "\n" + b64[76:]})
    check(status == 200 and body["predictions"][0]["index"] == REF_INDEX, "JSON data URL with a line break", str(body)[:200])

    status, body = request(port, "POST", "/classify", json.dumps({"input": "hi"}).encode(), {"Content-Type": "application/json"})
    check(status == 400 and "POST /classify/image" in body.get("error", ""), "/classify on an image model -> 400", str(body))

    for what, (status, body) in {
        "no image part": request(port, "POST", "/classify/image", *multipart([("top_k", b"3", None)])),
        "two image parts": request(port, "POST", "/classify/image", *multipart([("image", gh, "a.jpg"), ("image", gh, "b.jpg")])),
        "image and file parts": request(port, "POST", "/classify/image", *multipart([("image", gh, "a.jpg"), ("file", gh, "b.jpg")])),
    }.items():
        check(status == 400 and "exactly one image part" in body.get("error", ""), f"{what} -> 400", str(body))

    cases = {
        "GIF bytes": post_image(port, b"GIF89a\x01\x00\x01\x00\x00\x00\x00;"),
        "bad base64": post_json(port, {"image": "aGV*bG8="}),
        "http URL": post_json(port, {"image": "https://example.com/cat.jpg"}),
        "data URL with a non-image type": post_json(port, {"image": "data:text/plain;base64,aGVsbG8="}),
        "top_k 0": post_image(port, gh, top_k=0),
        "top_k abc": post_image(port, gh, top_k="abc"),
        "JSON top_k 1.5": post_json(port, {"image": b64, "top_k": 1.5}),
        "text/plain body": request(port, "POST", "/classify/image", b"hello", {"Content-Type": "text/plain"}),
        "truncated PNG": post_image(port, png(64, 64)[:60]),
        "PNG with a valid signature and garbage": post_image(port, b"\x89PNG\r\n\x1a\n" + bytes(200)),
        "truncated JPEG header": post_image(port, gh[:200]),
        "JSON top-level array": post_json(port, [b64]),
        "JSON nested object": post_json(port, {"image": b64, "options": {"a": 1}}),
        "JSON duplicate image key": request(port, "POST", "/classify/image", b'{"image":"aGVsbG8=","image":"aGVsbG8="}',
                                            {"Content-Type": "application/json"}),
    }
    for what, (status, body) in cases.items():
        detail = body.get("error", "") if isinstance(body, dict) else str(body)[:100]
        check(status == 400, f"{what} -> 400 ({detail})", f"{status} {detail}")

    jpeg12 = bytearray(gh)
    sof = jpeg12.find(b"\xff\xc0")
    jpeg12[sof + 4] = 12
    status, body = post_image(port, bytes(jpeg12))
    check(status == 400, f"12-bit JPEG -> 400 ({body.get('error')})", str(body))

    bomb = zip_bomb()
    status, body = post_image(port, bomb)
    check(status == 400 and body.get("error") == "image decode exceeded memory budget",
          f"PNG zip bomb ({len(bomb)} bytes, inflates to 272 MiB) -> 400", str(body))

    nested = b"[" * (21 << 19) + b"]" * (21 << 19)
    status, body = request(port, "POST", "/classify/image", nested, {"Content-Type": "application/json"})
    check(status == 400 and "must be a JSON object" in body.get("error", ""),
          "21 MiB of nested JSON arrays -> 400", f"{status} {body}")

    # 20 MB is under the derived payload limit (16 MiB * 4/3 + 64 KiB, about
    # 22.4 MB), so it reaches the image-size check; 30 MB is over the limit and
    # is refused from Content-Length before the body is read.
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    body, headers = multipart([("image", bytes(20 * 1000 * 1000), "x.bin")])
    c.request("POST", "/classify/image", body, headers)
    r = c.getresponse()
    status, data = r.status, r.read()
    check(status == 413 and b"byte limit" in data, "20 MB image part -> 413 (image-size limit)", f"{status} {data[:200]}")
    body, headers = multipart([("image", gh, "x.jpg")])
    c.request("POST", "/classify/image", body, headers)
    r = c.getresponse()
    status, data = r.status, r.read()
    check(status == 200, "same keep-alive connection still serves after the 413", f"{status} {data[:200]}")
    c.close()
    line = oversize_raw(port, 30 * 1000 * 1000)
    check(" 413 " in line, "30 MB body over the payload limit -> 413 before the body is read", line)

    line = chunked_raw(port, 30 * 1000 * 1000)
    check(" 413 " in line, "30 MB chunked body (no Content-Length) -> 413", line)

    status, body, waited = busy_503(port, gh)
    check(status == 503 and "busy" in body.get("error", "") and 29 < waited < 60,
          f"both request slots held by slow uploads -> 503 after {waited:.1f} s", f"{status} {body}")

    status, health = request(port, "GET", "/health")
    status2, body = post_image(port, gh)
    check(status == 200 and status2 == 200 and body["predictions"][0]["index"] == REF_INDEX, "still healthy and correct afterwards")

    print(f"{failures} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
