"""llama.cpp planner: llama-server is started only for storyboarding and stopped before drawing."""

import json
import socket
import stat
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from webtoon.cli import main
from webtoon.project import ProjectDir

FAKE_SERVER = r'''#!{python}
"""Stand-in for llama.cpp's llama-server: same CLI flags, /health and /v1/chat/completions."""
import json, os, re, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
with open(os.environ["FAKE_LLAMA_LOG"], "a") as log:
    log.write("start " + " ".join(args) + "\n")
if os.environ.get("FAKE_LLAMA_OLD") and "--reasoning" in args:
    print("error: invalid argument: --reasoning", flush=True)
    sys.exit(1)
port = int(args[args.index("--port") + 1])
assert args[args.index("-m") + 1].endswith(".gguf") and "-c" in args
started = time.time()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            loading = time.time() - started < 0.6
            return self.send(503 if loading else 200, {"error": "Loading model"} if loading else {"status": "ok"})
        self.send(404, {})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user = req["messages"][-1]["content"]
        text = user.split("<text", 1)[1]
        numbers = [int(n) for n in re.findall(r"^\[(\d+)\] ", text, re.M)]
        panels = [{"shot": "medium", "location": "hedge maze", "time_of_day": "day", "characters": ["Jason Asano"],
                   "action": "Jason stands between tall hedges", "mood": "bright", "narration": "", "dialogue": [],
                   "sfx": "", "source_paragraphs": [n]} for n in numbers]
        plan = {"title": "Strange Business", "character_updates": [], "panels": panels, "summary": "Jason wakes up.",
                "new_characters": [{"name": "Jason Asano", "aliases": ["Jason"], "role": "protagonist",
                                    "appearance": "completely bald young man", "outfit": "nothing"}]}
        content = "<think>planning</think>" + json.dumps(plan)
        self.send(200, {"id": "x", "object": "chat.completion", "created": 0, "model": "local",
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": content}}]})

HTTPServer(("127.0.0.1", port), Handler).serve_forever()
'''


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_open(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture
def setup(tmp_path, monkeypatch):
    server = tmp_path / "llama-server"
    server.write_text(FAKE_SERVER.replace("{python}", sys.executable))
    server.chmod(server.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "llama.log"
    monkeypatch.setenv("FAKE_LLAMA_LOG", str(log))
    monkeypatch.setattr("webtoon.llm_server.LOG_PATH", tmp_path / "server-output.log")
    model = tmp_path / "Qwen3-27B-Q4_K_M.gguf"
    model.write_bytes(b"gguf")
    chapter = tmp_path / "Chapter1.txt"
    chapter.write_text("Chapter 1: Strange Business\n\n" + "\n\n".join(
        f"Jason walked through the hedge maze, part {i}, looking for a way out." for i in range(12)))
    port = free_port()
    project = tmp_path / "story"
    main(["init", str(project), "--planner", "llamacpp", "--llm-model", str(model), "--llama-server", str(server),
          "--planner-base-url", f"http://127.0.0.1:{port}/v1", "--image-backend", "mock", "--segment-words", "60"])
    return project, chapter, log, port


def starts(log: Path) -> int:
    return log.read_text().count("start ") if log.exists() else 0


def test_llm_runs_only_while_storyboarding(setup):
    project, chapter, log, port = setup
    main(["chapter", str(project), str(chapter)])
    state = ProjectDir(project).load()
    assert [c.name for c in state.characters] == ["Jason Asano"]
    assert starts(log) == 1  # one load for all segments of the chapter
    assert "--reasoning off" in log.read_text() and "-c 32768" in log.read_text()
    assert not port_open(port)  # stopped before drawing
    assert (project / "chapter_01" / "reader.html").exists()
    report = json.loads((project / "chapter_01" / "coverage.json").read_text())
    assert len(report["segments"]) > 1 and report["patched"] == 0

    # Re-running with every segment already storyboarded doesn't load the LLM at all.
    main(["chapter", str(project), str(chapter), "--number", "1"])
    assert starts(log) == 1


def test_batch_uses_one_llm_session(setup, tmp_path):
    project, chapter, log, port = setup
    second = tmp_path / "Chapter2.txt"
    second.write_text("Chapter 2\n\n" + "\n\n".join(f"Jason fought hamster number {i}." for i in range(8)))
    main(["chapter", str(project), str(chapter), str(second), "--plan-only"])
    assert starts(log) == 1
    assert [c.number for c in ProjectDir(project).load().chapters] == [1, 2]
    assert not port_open(port)


def test_older_llama_cpp_without_reasoning_flag(setup, monkeypatch):
    project, chapter, log, port = setup
    monkeypatch.setenv("FAKE_LLAMA_OLD", "1")
    main(["chapter", str(project), str(chapter), "--plan-only"])
    lines = log.read_text().splitlines()
    assert len(lines) == 2 and "--reasoning" in lines[0] and "--reasoning" not in lines[1]


def test_comfyui_is_asked_to_unload_first(setup):
    project, chapter, log, port = setup
    calls = []

    class Comfy(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, body, starts(log)))
            self.send_response(200)
            self.end_headers()

    comfy = ThreadingHTTPServer(("127.0.0.1", 0), Comfy)
    threading.Thread(target=comfy.serve_forever, daemon=True).start()
    try:
        main(["init", str(project), "--image-backend", "comfyui",
              "--image-base-url", f"http://127.0.0.1:{comfy.server_address[1]}"])
        main(["chapter", str(project), str(chapter), "--plan-only"])
    finally:
        comfy.shutdown()
    assert calls == [("/free", {"unload_models": True, "free_memory": True}, 0)]  # before the LLM started


def test_missing_binary_is_explained(setup, tmp_path):
    project, chapter, log, port = setup
    main(["init", str(project), "--llama-server", str(tmp_path / "nope")])
    with pytest.raises(RuntimeError, match="llama-server"):
        main(["chapter", str(project), str(chapter), "--plan-only"])
