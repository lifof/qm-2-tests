"""Drives the interactive app in a real pseudo-terminal (mock image model + mock planner)."""

import json
import os
import sys
from pathlib import Path

import pytest

from webtoon.tui import parse_panel_list

pexpect = pytest.importorskip("pexpect")

ROOT = Path(__file__).resolve().parent.parent
DOWN, UP, ENTER, BACKSPACE = "\x1b[B", "\x1b[A", "\r", "\x7f"


def test_parse_panel_list():
    assert parse_panel_list("3, 7-9,12") == {3, 7, 8, 9, 12}


@pytest.mark.skipif(sys.platform == "win32", reason="needs a pty")
def test_interactive_session(tmp_path):
    sys.path.insert(0, str(ROOT / "tests"))
    from test_long_chapters import long_chapter

    chapter = tmp_path / "long_chapter.txt"
    chapter.write_text(long_chapter(80), encoding="utf-8")
    env = dict(os.environ, WEBTOON_CONFIG=str(tmp_path / "app.json"), PYTHONPATH=str(ROOT), BROWSER="true")
    child = pexpect.spawn(sys.executable, ["-m", "webtoon"], cwd=str(tmp_path), env=env, encoding="utf-8",
                          timeout=120, dimensions=(50, 140))

    def step(prompt, keys):
        child.expect(prompt)
        child.send(keys)

    step("Project", ENTER)  # create a new project
    step("Story title", "Test Story" + ENTER)
    step("Where should the project live", ENTER)
    step("How do you run Qwen-Image", DOWN * 4 + ENTER)  # mock
    step("Which LLM", DOWN * 3 + ENTER)  # mock planner
    step("Save these model settings", ENTER)
    step("What do you want to do", ENTER)  # add chapter 1
    step("What now\\?", DOWN + ENTER)  # mock planner warning: continue anyway
    step("Chapter text file", str(chapter))
    child.send(ENTER)
    step("Chapter number", ENTER)
    step("Go\\?", ENTER)  # storyboard and draw
    step("What do you want to do", DOWN * 2 + ENTER)  # draw / redraw
    step("Which chapter", ENTER)
    step("Draw what", DOWN + ENTER)  # specific panels
    step("Panel numbers", "999" + ENTER)
    child.expect("No such panel")
    child.send(BACKSPACE * 3 + "2, 4-5" + ENTER)
    step("What do you want to do", DOWN * 6 + ENTER)  # settings
    step("Settings", DOWN * 6 + ENTER)  # segment size
    step("Words per segment", BACKSPACE * 6 + "500" + ENTER)
    step("Settings", UP + ENTER)  # back
    step("What do you want to do", UP + ENTER)  # quit
    child.expect("Bye!")
    child.expect(pexpect.EOF)

    project = tmp_path / "test_story"
    state = json.loads((project / "story.json").read_text())
    assert state["image"]["backend"] == "mock"
    assert state["planner"]["segment_words"] == 500
    report = json.loads((project / "chapter_01" / "coverage.json").read_text())
    assert len(report["segments"]) == 3 and report["patched"] == 0
    assert (project / "chapter_01" / "reader.html").exists()
    defaults = json.loads((tmp_path / "app.json").read_text())["defaults"]
    assert defaults["image"]["backend"] == "mock"


@pytest.mark.skipif(sys.platform == "win32", reason="needs a pty")
def test_comfyui_setup_detects_model_files(tmp_path):
    models = tmp_path / "qwen-image-2.1"
    models.mkdir()
    for name in ("qwen-image-2.1-UC-BF16.gguf", "qwen3vl_8b_bf16.safetensors", "qwen_image_2.1_vae_bf16.safetensors"):
        (models / name).write_bytes(b"x")
    env = dict(os.environ, WEBTOON_CONFIG=str(tmp_path / "app.json"), PYTHONPATH=str(ROOT), BROWSER="true")
    child = pexpect.spawn(sys.executable, ["-m", "webtoon"], cwd=str(tmp_path), env=env, encoding="utf-8",
                          timeout=60, dimensions=(50, 160))

    def step(prompt, keys):
        child.expect(prompt)
        child.send(keys)

    step("Project", ENTER)
    step("Story title", "Comfy Story" + ENTER)
    step("Where should the project live", ENTER)
    step("How do you run Qwen-Image", ENTER)  # comfyui (default)
    step("Folder with your Qwen-Image 2.1 model files", str(models))
    child.send(ENTER)
    child.expect("qwen3vl_8b_bf16.safetensors")  # detection table
    step("Use these files", ENTER)
    step("Your ComfyUI folder", ENTER)  # run ComfyUI yourself
    step("ComfyUI address", ENTER)
    step("official Qwen-Image 2.1 sampling settings", ENTER)
    step("reference sheet", ENTER)
    step("Generate a test image now", ENTER)  # default: no
    step("Which LLM", DOWN * 3 + ENTER)
    step("Save these model settings", ENTER)
    child.expect("Qwen-Image 2.1 via ComfyUI: qwen-image-2.1-UC-BF16.gguf")
    step("What do you want to do", UP + ENTER)  # quit
    child.expect(pexpect.EOF)

    img = json.loads((tmp_path / "comfy_story" / "story.json").read_text())["image"]
    assert img["backend"] == "comfyui"
    assert img["diffusion_model"] == str((models / "qwen-image-2.1-UC-BF16.gguf").resolve())
    assert img["text_encoder"].endswith("qwen3vl_8b_bf16.safetensors")
    assert img["vae"].endswith("qwen_image_2.1_vae_bf16.safetensors")
    assert (img["steps"], img["cfg"], img["use_references"]) == (25, 1.0, True)


@pytest.mark.skipif(sys.platform == "win32", reason="needs a pty")
def test_old_diffusers_project_is_moved_to_comfyui_and_reads_mac_roman(tmp_path):
    from webtoon.cli import main as cli_main

    models = tmp_path / "qwen-image-2.1"
    models.mkdir()
    for name in ("qwen-image-2.1-UC-BF16.gguf", "qwen3vl_8b_bf16.safetensors", "qwen_image_2.1_vae_bf16.safetensors"):
        (models / name).write_bytes(b"x")
    project = tmp_path / "he_who_fights_with_monsters"
    cli_main(["init", str(project), "--image-backend", "diffusers", "--image-model", str(models), "--planner", "mock"])
    chapter = project / "Chapter1.txt"
    chapter.write_bytes("Chapter1— Jason\n\n“Where am I?” Jason asked.".encode("mac_roman"))

    env = dict(os.environ, WEBTOON_CONFIG=str(tmp_path / "app.json"), PYTHONPATH=str(ROOT), BROWSER="true")
    child = pexpect.spawn(sys.executable, ["-m", "webtoon", "app", str(project)], cwd=str(tmp_path), env=env,
                          encoding="utf-8", timeout=60, dimensions=(50, 160))

    def step(prompt, keys):
        child.expect(prompt)
        child.send(keys)

    child.expect("diffusers can't load")
    step("Set up ComfyUI for this project now", ENTER)
    step("Folder with your Qwen-Image 2.1 model files", ENTER)  # prefilled with the project's model folder
    step("Use these files", ENTER)
    step("Your ComfyUI folder", ENTER)
    step("ComfyUI address", ENTER)
    step("official Qwen-Image 2.1 sampling settings", ENTER)
    step("reference sheet", ENTER)
    step("Generate a test image now", ENTER)
    step("default for new projects", ENTER)
    child.expect("planner is 'mock'")
    step("What do you want to do", ENTER)  # add chapter 1
    child.expect("No real storyboarder")
    step("What now\\?", DOWN + ENTER)  # continue with mock
    step("Chapter text file", str(chapter) + ENTER)
    child.expect("read it as Mac OS Roman")
    child.expect("Chapter1— Jason")
    step("Chapter number", ENTER)
    step("Go\\?", DOWN * 2 + ENTER)  # cancel
    step("What do you want to do", UP + ENTER)  # quit
    child.expect(pexpect.EOF)

    img = json.loads((project / "story.json").read_text())["image"]
    assert img["backend"] == "comfyui" and img["diffusion_model"].endswith(".gguf")


@pytest.mark.skipif(sys.platform == "win32", reason="needs a pty")
def test_declined_setup_is_offered_again_before_drawing(tmp_path):
    from webtoon.cli import main as cli_main

    models = tmp_path / "qwen-image-2.1"
    models.mkdir()
    for name in ("qwen-image-2.1-UC-BF16.gguf", "qwen3vl_8b_bf16.safetensors", "qwen_image_2.1_vae_bf16.safetensors"):
        (models / name).write_bytes(b"x")
    project = tmp_path / "story"
    cli_main(["init", str(project), "--image-backend", "diffusers", "--image-model", str(models), "--planner", "mock",
              "--image-base-url", "http://127.0.0.1:9"])
    env = dict(os.environ, WEBTOON_CONFIG=str(tmp_path / "app.json"), PYTHONPATH=str(ROOT), BROWSER="true")
    child = pexpect.spawn(sys.executable, ["-m", "webtoon", "app", str(project)], cwd=str(tmp_path), env=env,
                          encoding="utf-8", timeout=60, dimensions=(50, 160))

    def step(prompt, keys):
        child.expect(prompt)
        child.send(keys)

    step("Set up ComfyUI for this project now", "n")  # decline at startup
    step("What do you want to do", DOWN * 4 + ENTER)  # Test the image model
    step("Set up ComfyUI for this project now", ENTER)  # offered again instead of a ValueError
    step("Folder with your Qwen-Image 2.1 model files", ENTER)
    step("Use these files", ENTER)
    step("Your ComfyUI folder", ENTER)
    step("ComfyUI address", ENTER)  # keeps the (unreachable) address from the project
    step("official Qwen-Image 2.1 sampling settings", ENTER)
    step("reference sheet", ENTER)
    step("Generate a test image now", ENTER)
    step("default for new projects", "n")
    child.expect("not reachable")  # now it's a ComfyUI problem with a clear message, not a diffusers error
    step("What do you want to do", UP + ENTER)
    child.expect(pexpect.EOF)
    assert json.loads((project / "story.json").read_text())["image"]["backend"] == "comfyui"


@pytest.mark.skipif(sys.platform == "win32", reason="needs a pty")
def test_batch_storyboards_all_then_draws_all(tmp_path):
    import stat

    from test_llamacpp import FAKE_SERVER, free_port, port_open

    from webtoon.cli import main as cli_main

    server = tmp_path / "llama-server"
    server.write_text(FAKE_SERVER.replace("{python}", sys.executable))
    server.chmod(server.stat().st_mode | stat.S_IEXEC)
    model = tmp_path / "Qwen3-27B-Q4_K_M.gguf"
    model.write_bytes(b"gguf")
    port = free_port()
    project = tmp_path / "story"
    cli_main(["init", str(project), "--planner", "llamacpp", "--llm-model", str(model), "--llama-server", str(server),
              "--planner-base-url", f"http://127.0.0.1:{port}/v1", "--image-backend", "mock"])
    for n in (1, 2, 10):  # natural order: 1, 2, 10
        (project / f"Chapter{n}.txt").write_text(f"Chapter {n}\n\n" + "\n\n".join(
            f"Jason explored part {i} of the maze." for i in range(4)))
    log = tmp_path / "llama.log"
    env = dict(os.environ, WEBTOON_CONFIG=str(tmp_path / "app.json"), PYTHONPATH=f"{ROOT}:{ROOT / 'tests'}",
               BROWSER="true", FAKE_LLAMA_LOG=str(log))
    child = pexpect.spawn(sys.executable, ["-m", "webtoon", "app", str(project)], cwd=str(tmp_path), env=env,
                          encoding="utf-8", timeout=90, dimensions=(50, 160))

    def step(prompt, keys):
        child.expect(prompt)
        child.send(keys)

    child.expect("loaded only while storyboarding")
    step("What do you want to do", DOWN + ENTER)  # add several chapters
    step("Folder with the chapter .txt files", ENTER)  # defaults to the project folder
    child.expect("Chapter10.txt")
    step("Chapters to add", ENTER)  # all pre-selected
    step("Number of the first one", ENTER)
    step("Go\\?", ENTER)  # storyboard all, then draw all
    child.expect("storyboarding")
    child.expect("Starting the storyboard LLM")
    child.expect("Stopping the storyboard LLM")
    child.expect("drawing")
    step("What do you want to do", UP + ENTER)  # quit
    child.expect(pexpect.EOF)

    state = json.loads((project / "story.json").read_text())
    assert [(c["number"], Path(c["source_file"]).name) for c in state["chapters"]] == [
        (1, "Chapter1.txt"), (2, "Chapter2.txt"), (3, "Chapter10.txt")]
    assert log.read_text().count("start ") == 1  # one LLM load for all three chapters
    assert not port_open(port)
    assert all((project / f"chapter_0{n}" / "reader.html").exists() for n in (1, 2, 3))
