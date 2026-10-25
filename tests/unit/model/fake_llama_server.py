"""Recording stand-in for ``llama-server --embedding`` over a Unix socket.

It exists only to observe what the provider sends across the process boundary
and to inject malformed outputs; it is never evidence of real inference.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import socketserver
import sys
import time
import wave
from http.server import BaseHTTPRequestHandler
from typing import Any


def _vector(seed: bytes, dimension: int = 768) -> list[float]:
    values = [((seed[i % 32] + i) % 17) - 8.0 for i in range(dimension)]
    values[0] += 0.5
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


def main() -> None:
    arguments = sys.argv[1:]
    mode = arguments[arguments.index("--fake-mode") + 1]
    log = arguments[arguments.index("--fake-log") + 1]
    host = arguments[arguments.index("--host") + 1]
    with open(log, "a", encoding="utf-8") as stream:
        stream.write(json.dumps({"argv": arguments}) + "\n")
    if mode == "crash":
        sys.exit(3)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            return

        def address_string(self) -> str:
            return "unix"

        def do_GET(self) -> None:
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            items: list[dict[str, Any]] = []
            for item in body["content"]:
                if isinstance(item, str):
                    items.append({"text": item, "seed": hashlib.sha256(item.encode()).digest()})
                    continue
                payload = base64.b64decode(item["multimodal_data"][0])
                record: dict[str, object] = {
                    "prompt": "<marker>"
                    if item["prompt_string"] == os.environ["LLAMA_MEDIA_MARKER"]
                    else item["prompt_string"]
                }
                if payload.startswith(b"RIFF"):
                    with wave.open(io.BytesIO(payload)) as audio:
                        frames = audio.readframes(audio.getnframes())
                        record.update(
                            rate=audio.getframerate(),
                            channels=audio.getnchannels(),
                            width=audio.getsampwidth(),
                            pcm_sha256=hashlib.sha256(frames).hexdigest(),
                        )
                else:
                    record["image_sha256"] = hashlib.sha256(payload).hexdigest()
                record["seed"] = hashlib.sha256(payload).digest()
                items.append(record)
            with open(log, "a", encoding="utf-8") as stream:
                safe = [{k: v for k, v in item.items() if k != "seed"} for item in items]
                stream.write(
                    json.dumps({"normalize": body.get("embd_normalize"), "items": safe}) + "\n"
                )
            if mode == "slow":
                time.sleep(5)
            if mode == "http500":
                self.send_response(500)
                self.end_headers()
                return
            rows: list[dict[str, Any]] = [
                {"index": index, "embedding": [_vector(item["seed"])]}
                for index, item in enumerate(items)
            ]
            rows.reverse()
            if mode == "nan":
                rows[0]["embedding"][0][3] = float("nan")
            elif mode == "dimension":
                rows[0]["embedding"][0] = rows[0]["embedding"][0][:767]
            elif mode == "count":
                rows = rows[:-1]
            elif mode == "zero":
                rows[0]["embedding"][0] = [0.0] * 768
            elif mode == "unnormalized":
                rows[0]["embedding"][0] = [value * 2 for value in rows[0]["embedding"][0]]
            elif mode == "duplicate":
                rows[-1]["index"] = rows[0]["index"]
            encoded = json.dumps(rows, allow_nan=True).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    class Server(socketserver.UnixStreamServer):
        def get_request(self):  # type: ignore[no-untyped-def]
            request, _ = super().get_request()
            return request, ("unix", 0)

    with Server(host, Handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
