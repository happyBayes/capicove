#!/usr/bin/env python3
"""Build portable image/video requests and demonstrate a bounded tool round trip."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from capicove.model.vlm import VlmMedia, VlmMessage, VlmRequest, VlmText


def post(base_url: str, route: str, request: dict[str, object]) -> dict:
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("CAPICOVE_VLM_HTTP_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    payload = json.dumps(request, ensure_ascii=False).encode()
    req = urllib.request.Request(base_url.rstrip("/") + route, data=payload, headers=headers)
    with opener.open(req, timeout=180) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("request", "generate", "count", "tools"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8769/v1")
    parser.add_argument("--model", default="qwen35-9b")
    parser.add_argument("--prompt", default="Describe the media in order.")
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--video", type=Path, action="append", default=[])
    args = parser.parse_args()
    parts: list[VlmText | VlmMedia] = [VlmText(args.prompt)]
    for modality, paths in (("image", args.image), ("video", args.video)):
        for path in paths:
            media_type = mimetypes.guess_type(path.name)[0]
            if media_type is None or not media_type.startswith(modality + "/"):
                raise ValueError("unsupported media extension")
            parts.append(VlmMedia(path.read_bytes(), media_type))
    body = VlmRequest((VlmMessage("user", tuple(parts)),), max_tokens=512).to_openai(args.model)
    if args.mode == "request":
        print(json.dumps(body, ensure_ascii=False))
        return 0
    if args.mode == "tools":
        body["messages"] = [
            {
                "role": "user",
                "content": "Use multiply to calculate 6 times 7, then explain the result.",
            }
        ]
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "multiply",
                    "description": "Multiply two integers.",
                    "parameters": {
                        "type": "object",
                        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                        "required": ["a", "b"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        body["tool_choice"] = "required"
        body["parallel_tool_calls"] = False
        for _ in range(4):
            result = post(args.base_url, "/chat/completions", body)
            choice = result["choices"][0]
            if choice["finish_reason"] == "stop":
                print(json.dumps(result, ensure_ascii=False))
                return 0
            if choice["finish_reason"] != "tool_calls":
                raise ValueError("tool round trip did not finish")
            body["tool_choice"] = "auto"
            message = choice["message"]
            body["messages"].append(message)
            for call in message["tool_calls"]:
                function = call["function"]
                arguments = json.loads(function["arguments"])
                if (
                    function["name"] != "multiply"
                    or set(arguments) != {"a", "b"}
                    or any(type(v) is not int or abs(v) > 1000000 for v in arguments.values())
                ):
                    raise ValueError("unapproved tool or arguments")
                body["messages"].append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": str(arguments["a"] * arguments["b"]),
                    }
                )
        raise ValueError("tool turn limit exceeded")
    result = post(
        args.base_url, "/count_tokens" if args.mode == "count" else "/chat/completions", body
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError):
        print(
            "VLM example failed: check media, endpoint, model alias and credential environment.",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
