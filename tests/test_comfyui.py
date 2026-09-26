"""ComfyUI backend: file detection/linking, workflow building and the HTTP round trip (fake server)."""

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pytest
from PIL import Image

from webtoon.comfyui import (ComfyUIBackend, ComfyUIError, build_workflow, detect_model_files,
                             fill_custom_workflow, link_into_comfy, memory_report)
from webtoon.models import ImageSettings

USER_FILES = ["qwen-image-2.1-UC-BF16.gguf", "qwen3vl_8b_bf16.safetensors", "qwen_image_2.1_vae_bf16.safetensors"]


@pytest.fixture
def model_dir(tmp_path):
    d = tmp_path / "qwen-image-2.1"
    d.mkdir()
    for name in USER_FILES:
        (d / name).write_bytes(b"x")
    return d


def test_detects_the_three_files(model_dir):
    found = detect_model_files(model_dir)
    assert {k: v.name for k, v in found.items()} == {
        "diffusion_model": "qwen-image-2.1-UC-BF16.gguf",
        "text_encoder": "qwen3vl_8b_bf16.safetensors",
        "vae": "qwen_image_2.1_vae_bf16.safetensors",
    }


def test_link_into_comfy_is_idempotent(model_dir, tmp_path):
    comfy = tmp_path / "ComfyUI"
    src = model_dir / USER_FILES[0]
    assert link_into_comfy(comfy, "diffusion_model", src) == USER_FILES[0]
    link = comfy / "models" / "diffusion_models" / USER_FILES[0]
    assert link.is_symlink() and link.resolve() == src.resolve()
    assert link_into_comfy(comfy, "diffusion_model", src) == USER_FILES[0]


def test_workflow_matches_qwen_image_21_template():
    names = dict(zip(("diffusion_model", "text_encoder", "vae"), USER_FILES))
    wf = build_workflow(names, "a panel", "blurry", 1344, 928, 7, ImageSettings(), ["webtoon/mira.png", "webtoon/joon.png"])
    assert wf["1"]["class_type"] == "UnetLoaderGGUF"
    assert wf["2"]["inputs"] == {"clip_name": USER_FILES[1], "type": "qwen_image", "device": "default"}
    enc = wf["4"]
    assert enc["class_type"] == "TextEncodeQwenImage21"
    assert enc["inputs"]["images.image_1"] == ["101", 0] and enc["inputs"]["images.image_2"] == ["102", 0]
    assert wf["101"] == {"class_type": "LoadImage", "inputs": {"image": "webtoon/mira.png"}}
    ks = wf["6"]["inputs"]
    assert (ks["steps"], ks["cfg"], ks["sampler_name"], ks["scheduler"]) == (25, 1.0, "euler", "simple")
    assert wf["5"]["inputs"]["width"] == 1344

    safetensors = build_workflow({**names, "diffusion_model": "q.safetensors"}, "p", "n", 1024, 1024, 1, ImageSettings())
    assert safetensors["1"] == {"class_type": "UNETLoader", "inputs": {"unet_name": "q.safetensors", "weight_dtype": "default"}}
    assert not any(k.startswith("images.") for k in safetensors["4"]["inputs"])


def test_custom_workflow_placeholders():
    template = {
        "3": {"class_type": "KSampler", "inputs": {"seed": "{{seed}}", "steps": "{{steps}}", "cfg": "{{cfg}}"}},
        "4": {"class_type": "MyEncode", "inputs": {"text": "{{prompt}}, masterpiece", "ref_a": ["10", 0], "ref_b": ["11", 0]}},
        "10": {"class_type": "LoadImage", "inputs": {"image": "{{ref_1}}"}},
        "11": {"class_type": "LoadImage", "inputs": {"image": "{{ref_2}}"}},
    }
    wf = fill_custom_workflow(template, {"seed": 5, "steps": 20, "cfg": 1.0, "prompt": "hero"}, ["a.png"])
    assert wf["3"]["inputs"] == {"seed": 5, "steps": 20, "cfg": 1.0}
    assert wf["4"]["inputs"]["text"] == "hero, masterpiece"
    assert wf["10"]["inputs"]["image"] == "a.png"
    assert "11" not in wf and "ref_b" not in wf["4"]["inputs"]


# ---------------------------------------------------------------------------
# Fake ComfyUI server
# ---------------------------------------------------------------------------


class FakeComfy(BaseHTTPRequestHandler):
    nodes = {"TextEncodeQwenImage21": {}, "UnetLoaderGGUF": {"unet_name": [USER_FILES[0]]},
             "CLIPLoader": {"clip_name": [USER_FILES[1]]}, "VAELoader": {"vae_name": [f"sub/{USER_FILES[2]}"]}}
    submitted = []

    def log_message(self, *args):
        pass

    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/system_stats":
            return self._json({"system": {}})
        if path.startswith("/object_info/"):
            node = path.rsplit("/", 1)[1]
            if node not in self.nodes:
                return self._json({})
            return self._json({node: {"input": {"required": {k: [v] for k, v in self.nodes[node].items()}}}})
        if path.startswith("/history/"):
            return self._json({"p1": {"status": {"status_str": "success", "completed": True},
                                      "outputs": {"8": {"images": [{"filename": "x.png", "subfolder": "", "type": "temp"}]}}}})
        if path == "/view":
            buf = io.BytesIO()
            Image.new("RGB", (64, 32), "red").save(buf, format="PNG")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(buf.getvalue())
            return
        self._json({}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        if self.path == "/upload/image":
            return self._json({"name": "ref.png", "subfolder": "webtoon", "type": "input"})
        if self.path == "/prompt":
            self.submitted.append(json.loads(body)["prompt"])
            return self._json({"prompt_id": "p1", "number": 1, "node_errors": {}})
        self._json({}, 404)


@pytest.fixture
def fake_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeComfy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    FakeComfy.submitted.clear()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def settings_for(model_dir, url, **kw):
    return ImageSettings(backend="comfyui", base_url=url, diffusion_model=str(model_dir / USER_FILES[0]),
                         text_encoder=str(model_dir / USER_FILES[1]), vae=str(model_dir / USER_FILES[2]), **kw)


def test_round_trip_with_references(model_dir, fake_server, tmp_path):
    backend = ComfyUIBackend(settings_for(model_dir, fake_server))
    assert backend.names["vae"] == f"sub/{USER_FILES[2]}"  # found in a subfolder of models/vae
    ref = tmp_path / "ref.png"
    Image.new("RGB", (8, 8)).save(ref)
    img = backend.generate("prompt", "neg", 1340, 930, 3, [ref])
    assert img.size == (64, 32)
    wf = FakeComfy.submitted[-1]
    assert (wf["5"]["inputs"]["width"], wf["5"]["inputs"]["height"]) == (1344, 928)  # rounded to 32
    assert wf["4"]["inputs"]["images.image_1"] == ["101", 0]
    assert wf["101"]["inputs"]["image"] == "webtoon/ref.png"


def test_missing_gguf_node_explains_how_to_install(model_dir, fake_server, monkeypatch):
    monkeypatch.setattr(FakeComfy, "nodes", {k: v for k, v in FakeComfy.nodes.items() if k != "UnetLoaderGGUF"})
    with pytest.raises(ComfyUIError, match="ComfyUI-GGUF"):
        ComfyUIBackend(settings_for(model_dir, fake_server))


def test_unreachable_server_without_comfy_dir(model_dir):
    with pytest.raises(ComfyUIError, match="not reachable"):
        ComfyUIBackend(settings_for(model_dir, "http://127.0.0.1:9"))


def test_ui_format_workflow_rejected(model_dir, fake_server, tmp_path):
    wf = tmp_path / "ui.json"
    wf.write_text(json.dumps({"nodes": [], "links": []}))
    with pytest.raises(ComfyUIError, match="Export \\(API\\)"):
        ComfyUIBackend(settings_for(model_dir, fake_server, workflow_file=str(wf)))


def test_diffusers_rejects_single_file_checkpoint(model_dir):
    pytest.importorskip("diffusers")
    from webtoon.image_backends import DiffusersBackend

    with pytest.raises(ValueError, match="comfyui"):
        DiffusersBackend(ImageSettings(backend="diffusers", model=str(model_dir / USER_FILES[0])))


def test_cli_model_dir_detection(model_dir, tmp_path):
    from webtoon.cli import main
    from webtoon.project import ProjectDir

    main(["init", str(tmp_path / "p"), "--image-backend", "comfyui", "--model-dir", str(model_dir)])
    img = ProjectDir(tmp_path / "p").load().image
    assert Path(img.diffusion_model).name == USER_FILES[0]
    assert Path(img.text_encoder).name == USER_FILES[1]
    assert Path(img.vae).name == USER_FILES[2]


def _sized(path: Path, gb: float) -> Path:
    with open(path, "wb") as fh:  # sparse file: reports the size without using the disk
        fh.truncate(int(gb * 1024 ** 3))
    return path


@pytest.mark.parametrize("dit_gb, expect", [(10, None), (20, "reload one of them"), (40, "alone")])
def test_memory_report_for_48gb_mac(tmp_path, dit_gb, expect):
    files = {"diffusion_model": _sized(tmp_path / "dit.gguf", dit_gb),
             "text_encoder": _sized(tmp_path / "te.safetensors", 16.5),
             "vae": _sized(tmp_path / "vae.safetensors", 0.3)}
    info, warnings = memory_report(files, ram_gb=48, unified=True)
    assert any("48 GB unified memory" in line for line in info)
    if expect is None:
        assert warnings == []
    else:
        assert len(warnings) == 1 and expect in warnings[0]
    assert memory_report(files, ram_gb=48, unified=False)[1] == []  # discrete GPU: no unified-memory check


def test_panel_sizes_target_megapixels():
    from webtoon.image_backends import size_for_shot

    for shot in ("establishing", "medium", "action"):
        w, h = size_for_shot(shot)
        assert w % 32 == 0 and h % 32 == 0
        assert 0.85 < w * h / 1024 ** 2 < 1.15
    w2, h2 = size_for_shot("medium", 2.0)
    assert 1.85 < w2 * h2 / 1024 ** 2 < 2.15


def test_comfyui_desktop_folder(model_dir, tmp_path):
    from webtoon.tui import _valid_comfy_dir

    desktop = tmp_path / "Documents" / "ComfyUI"
    (desktop / "models").mkdir(parents=True)
    assert _valid_comfy_dir(str(desktop)) is True
    assert _valid_comfy_dir(str(tmp_path)) is not True
    with pytest.raises(ComfyUIError, match="ComfyUI Desktop"):
        ComfyUIBackend(settings_for(model_dir, "http://127.0.0.1:9", comfy_dir=str(desktop)))
    assert link_into_comfy(desktop, "vae", model_dir / USER_FILES[2]) == USER_FILES[2]
    assert (desktop / "models" / "vae" / USER_FILES[2]).is_symlink()
