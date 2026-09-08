"""Offline CLI integration test against a local OpenAI-compatible HTTP server."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from conftest import write_clip
from v_cropper.cli import main

_EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval")
sys.path.insert(0, _EVAL_DIR)
import run_eval  # noqa: E402

POINT = json.dumps({"x": 500, "y": 500})


class _OpenAIHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        request = {
            "path": self.path,
            "headers": dict(self.headers),
            "body": json.loads(self.rfile.read(length)),
        }
        self.server.requests.append(request)
        if self.server.status != 200:
            payload = json.dumps({"error": {"message": "mock authorization failure"}}).encode()
            self.send_response(self.server.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        response = {
            "id": "mock-1",
            "object": "chat.completion",
            "created": 0,
            "model": request["body"]["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": POINT},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
        payload = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format, *_args):
        pass


@pytest.fixture
def openai_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAIHandler)
    server.requests = []
    server.status = 200
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join()


def test_cli_sends_real_multimodal_request_to_custom_endpoint(tmp_path, monkeypatch, openai_server):
    monkeypatch.setattr("v_cropper.cli.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("VCROPPER_API_KEY", "unit-key")
    monkeypatch.setenv("VCROPPER_MODEL", "local-vision")
    monkeypatch.setenv("VCROPPER_EXTRA_HEADERS", '{"X-Gateway-Client": "v-cropper-test"}')
    clip = write_clip(tmp_path / "in.mp4", n_frames=10, w=160, h=90)
    out = tmp_path / "out.mp4"
    base_url = f"http://127.0.0.1:{openai_server.server_port}/v1"

    assert main([str(clip), "-o", str(out), "--base-url", base_url,
                 "--sample-every", "10"]) == 0
    assert out.exists() and out.stat().st_size > 0
    assert len(openai_server.requests) == 1

    request = openai_server.requests[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer unit-key"
    assert request["headers"]["X-Gateway-Client"] == "v-cropper-test"
    assert request["body"]["model"] == "local-vision"
    content = request["body"]["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "football" in content[1]["text"].lower()


def test_cli_401_from_custom_endpoint_does_not_log_api_key(tmp_path, monkeypatch, caplog, openai_server):
    monkeypatch.setattr("v_cropper.cli.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("VCROPPER_API_KEY", "unit-key-must-not-log")
    openai_server.status = 401
    clip = write_clip(tmp_path / "in.mp4", n_frames=10, w=160, h=90)
    base_url = f"http://127.0.0.1:{openai_server.server_port}/v1"
    out = tmp_path / "out.mp4"

    assert main([str(clip), "-o", str(out), "--base-url", base_url,
                 "--sample-every", "10"]) == 0

    assert len(openai_server.requests) == 1
    assert out.exists() and out.stat().st_size > 0
    assert "VLM call failed" in caplog.text
    assert "unit-key-must-not-log" not in caplog.text


def test_run_eval_uses_custom_http_endpoint_end_to_end(tmp_path, monkeypatch, openai_server):
    monkeypatch.setenv("VCROPPER_API_KEY", "unit-key")
    monkeypatch.setenv("VCROPPER_MODEL", "local-vision")
    data = tmp_path / "data"
    raw = data / "raw"
    raw.mkdir(parents=True)
    write_clip(raw / "a.mp4", n_frames=20, w=160, h=90)
    (data / "splits.json").write_text(json.dumps({"val": ["a"]}))
    (data / "val").mkdir()
    (data / "val" / "ground_truth.jsonl").write_text(
        json.dumps({"item_id": "a", "frame_idx": 10, "x": 80, "y": 45}) + "\n")
    out = tmp_path / "out"
    base_url = f"http://127.0.0.1:{openai_server.server_port}/v1"

    result = run_eval.run(argparse.Namespace(
        split="val", data=str(data), out=str(out), model="local-vision", base_url=base_url,
        sample_every=10, sample_fps=2.0, concurrency=1, send_width=768, sport=None, prompt=None,
        prompt_file=None, spring_k=0.05, no_cache=True,
    ))

    assert result["structural_ok"] is True
    assert len(openai_server.requests) == 2
    assert all(r["path"] == "/v1/chat/completions" for r in openai_server.requests)
    assert (out / "val.jsonl").exists()
    assert (out / "metrics-val.json").exists()
