# Webtoon generator (Qwen-Image)

Turns a story, one chapter at a time, into a vertical-scroll webtoon. Characters stay consistent from chapter to chapter.

```
chapter1.txt ──► planner LLM ──► plan.json (panels, dialogue, cast) ──► Qwen-Image ──► lettering ──► strip
                     ▲                                                       │
                     └──────────── story.json (character bible + story so far) ◄┘
```

1. **Plan.** An LLM storyboards the chapter into panels: shot type, location, visible characters, action, mood, dialogue, narration and SFX. It gets the **character bible** and a **story-so-far** summary, so it reuses existing characters by their canonical names. Long chapters are storyboarded in segments, and nothing is skipped (see [Long chapters](#long-chapters)).
2. **Lock the cast.** New characters get a permanent visual description, a fixed seed and a generated **character sheet** (`characters/<name>.png`). Later chapters can't redesign them. Only outfit changes and permanent changes the story states (a scar, a haircut) are applied.
3. **Draw.** Each panel prompt includes the locked description of every character in that panel. With `--use-references` and an edit-capable Qwen model, the character sheets are also passed to the model as image references.
4. **Letter and assemble.** Speech, thought, shout and whisper bubbles, caption boxes and SFX are drawn in code, so text is always readable and easy to edit. The panels are then stacked top-down with wider gaps at scene changes and cut into 800×1280 slices.

## Install

```bash
pip install -r requirements.txt      # or: pip install -e .
```

For Qwen-Image 2.1 you also need [ComfyUI](https://github.com/comfyanonymous/ComfyUI), a recent version with Qwen-Image 2.1 support. See [Connecting your Qwen-Image model](#connecting-your-qwen-image-model).

## Interactive app

```bash
python -m webtoon            # or: pip install -e . && webtoon
```

The app walks you through everything with arrow-key menus:

- **Create or open a project.** Recent projects are remembered.
- **Settings.** Pick how you run Qwen-Image (local model folder, your own server URL, DashScope or mock) and which LLM storyboards the chapters. You can also set steps, CFG, character references, segment size, pacing, art style, negative prompt, strip width and font. API keys can be entered for the session. Settings can be saved as defaults for new projects (`~/.config/webtoon/app.json`).
- **Add chapter N.** Choose the text file. The app shows the word count, number of segments and estimated panels, then storyboards and draws the chapter, or storyboards only so you can edit `plan.json` first.
- **Draw or redraw a chapter.** Draw missing panels, specific panels (`3, 7-9`), everything, or only redo the lettering.
- **Review a chapter's storyboard.** A table of every panel plus the coverage report.
- **Characters.** Edit a character's appearance, outfit or aliases, or redraw their character sheet.
- **Test the image model or planner.** Generate one image, or storyboard a two-paragraph sample, to check your setup.

The header shows what the project is using and warns about problems, such as a missing model path, an unset server URL or a missing API key.

## Command line

Everything is also scriptable:

```bash
# 1. create a project and say how to reach your Qwen-Image model (settings are saved in story.json)
python -m webtoon init my_story --title "The Lantern Shop" \
    --image-backend comfyui --model-dir ~/Projects/qwen-image-2.1 --comfy-dir ~/ComfyUI

# 2. first chapter
python -m webtoon chapter my_story chapter1.txt

# 3. next chapter: the same characters carry over automatically
python -m webtoon chapter my_story chapter2.txt
```

Output:

```
my_story/
  story.json                  character bible, settings, chapter summaries
  index.html                  chapter list + cast gallery
  characters/mira_han.png     character sheets
  chapter_01/
    plan.json                 the storyboard (editable)
    art/panel_001.png         raw model output (+ .prompt.txt with the exact prompt)
    panels/panel_001.png      lettered panel
    strip/ch01_001.jpg ...    800x1280 upload-ready slices
    chapter_01_full.png       the whole strip as one image
    reader.html               open in a browser to scroll through the chapter
```

### Editing and redoing

```bash
python -m webtoon chapter my_story chapter3.txt --plan-only   # stop after storyboarding
#   ...edit my_story/chapter_03/plan.json (dialogue, shots, a character's look)...
python -m webtoon render my_story 3                           # draw it
python -m webtoon render my_story 3 --only 4,9                # redraw panels you don't like
python -m webtoon render my_story 3 --letter-only             # re-letter only, no image generation
python -m webtoon chapter my_story chapter3.txt --number 3    # re-plan and redraw chapter 3 from scratch
python -m webtoon characters my_story                         # show the character bible
```

You can also edit `story.json` by hand, for example to tweak a character's `appearance`. Delete their `reference_image` entry to get a new character sheet.

Runs can be resumed. Panels that already have art are skipped, so an interrupted run continues where it stopped.

## Connecting your Qwen-Image model

### Qwen-Image 2.1 model files (recommended: `comfyui` backend)

Qwen-Image 2.1 comes as three files:

```
qwen-image-2.1-UC-BF16.gguf            diffusion model (.gguf or .safetensors)
qwen3vl_8b_bf16.safetensors            text encoder (Qwen3-VL 8B)
qwen_image_2.1_vae_bf16.safetensors    VAE
```

diffusers can't load these files; it has no Qwen-Image 2.x pipeline. ComfyUI supports the model natively, so the app drives ComfyUI for you:

1. **Install ComfyUI**, following its README (on a Mac, the Apple Silicon instructions). A `.gguf` diffusion model also needs the **ComfyUI-GGUF** custom node. The app offers to install it, or run:
   ```bash
   cd ~/ComfyUI/custom_nodes && git clone https://github.com/city96/ComfyUI-GGUF
   ~/ComfyUI/.venv/bin/python -m pip install -r ComfyUI-GGUF/requirements.txt   # ComfyUI's own Python
   ```
2. **Point the app at your files.** In the app, go to *Settings → Backend and model files → Qwen-Image 2.1 model files*. Give it the folder holding the three files (they are detected automatically) and your ComfyUI folder. Or from the command line:
   ```bash
   python -m webtoon init my_story --image-backend comfyui \
       --model-dir /Users/you/Projects/qwen-image-2.1 --comfy-dir ~/ComfyUI
   ```

Then just add chapters. The app:
- symlinks the three files into `ComfyUI/models/{diffusion_models,text_encoders,vae}`, leaving the originals where they are;
- starts ComfyUI if it isn't running (using ComfyUI's own `.venv`/`venv` if it has one; log in `~/.cache/webtoon/comfyui.log`) and stops it when you're done;
- renders each panel with the same graph as ComfyUI's official Qwen-Image 2.1 template (`UnetLoaderGGUF`/`UNETLoader`, `CLIPLoader` type `qwen_image`, `VAELoader`, `TextEncodeQwenImage21`, `KSampler` with 25 steps, CFG 1.0, euler/simple);
- passes the character sheets of everyone in the panel as reference images (`TextEncodeQwenImage21` accepts up to 16).

If ComfyUI runs elsewhere, or you start it yourself, leave the ComfyUI folder empty and set `--image-base-url http://host:8188`. Then put the files in that ComfyUI's model folders yourself.

Options: `--comfy-args "--lowvram"` passes extra launch flags, and `--sampler`/`--scheduler` change the sampler. `--workflow my_api.json` swaps in your own graph, built in ComfyUI with LoRAs, upscaling and so on, and exported with *Workflow → Export (API)*. Use `{{prompt}}` `{{negative}}` `{{seed}}` `{{width}}` `{{height}}` `{{steps}}` `{{cfg}}` as input values, and `{{ref_1}}`…`{{ref_16}}` as `LoadImage` file names.

### Running on a Mac (Apple Silicon)

The steps above work as-is on Apple Silicon. ComfyUI uses the GPU through Metal (MPS), and the app starts it with `PYTORCH_ENABLE_MPS_FALLBACK=1`, so an operation Metal lacks runs on the CPU instead of crashing. ComfyUI Desktop works too: pick its folder (the one with `models/`, usually `~/Documents/ComfyUI`) and open the app before drawing. Its server usually runs on port 8000.

**Memory.** The GPU shares the Mac's memory but may only use about 75% of it by default: about 36 GB on a 48 GB machine. To draw panels quickly, ComfyUI needs the text encoder and the diffusion model loaded at the same time. The text encoder, Qwen3-VL-8B in BF16, is already about 16–17 GB. When you select your files, the app shows their sizes against your Mac's memory, and the header warns you if they won't fit together. If they don't:

- Use a **Q8_0 GGUF** of the diffusion model instead of BF16: about half the size, with near-identical quality. (Q6_K / Q5_K_M are smaller again.)
- Use a **GGUF text encoder**. A `.gguf` text encoder is loaded with `CLIPLoaderGGUF` automatically.
- Or raise the GPU limit until the next reboot, leaving several GB for macOS: `sudo sysctl iogpu.wired_limit_mb=40960`.

The int8 "convrot" files that ComfyUI's own template links to ([Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)) are aimed at NVIDIA GPUs. On a Mac, GGUF is the safer choice.

**Speed.** Panels are generated at about 1 MP (`--megapixels`, or *Settings → Panel resolution*) and shrunk to the 800 px strip width. Going higher mostly costs time. Use *Test the image model* to see how long one image takes on your machine before starting a long chapter.

**Planner.** The Claude planner runs in the cloud and uses no local memory. If you use a local LLM through Ollama on the same Mac, set `OLLAMA_KEEP_ALIVE=0` so Ollama unloads the model after storyboarding instead of holding memory while ComfyUI draws.

### Other backends

| Backend | When | Example |
|---|---|---|
| `diffusers` | A diffusers-format model folder (Qwen-Image 1.x); needs `pip install torch diffusers transformers accelerate` | `--image-backend diffusers --image-model /models/Qwen-Image` (or a Hugging Face repo id) |
| `openai` | You serve the model behind an OpenAI-style `/v1/images/generations` API (vLLM-Omni, LocalAI, your own FastAPI…) | `--image-backend openai --image-base-url http://gpu-box:8000/v1 --image-model qwen-image-2.1` (optional `IMAGE_API_KEY`) |
| `dashscope` | Alibaba Cloud Model Studio | `--image-backend dashscope --image-model <model name>` with `DASHSCOPE_API_KEY` set (`--image-base-url` to use a different region endpoint) |
| `mock` | Testing without a GPU | draws placeholder panels |

Other knobs: `--steps`, `--cfg`, `--megapixels`, `--style`, `--negative`, `--width`, `--font`. For diffusers, set `WEBTOON_CPU_OFFLOAD=1` to use CPU offload if VRAM is tight.

**Character references (`--use-references`).** Turn this on if your model or server accepts input images (Qwen-Image-Edit-style pipelines, `/images/edits`, or DashScope image inputs). Each panel is then conditioned on the sheets of the characters in it, which gives the strongest likeness. If it's off, consistency comes from the locked text descriptions alone.

## The planner

The planner defaults to Claude (`claude-opus-5`) with structured JSON output. Set `ANTHROPIC_API_KEY`, or log in with `ant auth login`. Server-side refusal fallback is enabled, so a false-positive safety decline on a dark chapter is retried on another model instead of failing.

To keep everything local, point it at any OpenAI-compatible chat server instead, for example a Qwen LLM on vLLM or Ollama:

```bash
python -m webtoon init my_story --planner openai \
    --planner-base-url http://localhost:11434/v1 --planner-model qwen3:32b
```

## Long chapters

Chapters of any length are handled without dropping content:

1. **Segments.** The chapter is split at paragraph boundaries into segments of about `--segment-words` words (default 1200; very long paragraphs are split at sentence ends). Segments are storyboarded one after another. Each one gets the character bible, which now includes characters introduced in earlier segments, a summary of the chapter so far and the last panels drawn. Scenes therefore continue seamlessly.
2. **Pacing scales with length.** Each segment is asked for about `--density` panels per 1000 words (default 10), and more if needed. There is no fixed panel cap.
3. **Coverage check.** Paragraphs are numbered, and every panel must list the paragraphs it adapts. After each segment the app verifies that every paragraph is cited and every quoted line of dialogue appears verbatim in a bubble or caption.
4. **Retry, then patch.** If anything is missing, the segment is re-planned with the gaps listed (`--retries`, default 2). If the planner still skips something, the missing paragraph is inserted as a caption panel in story order, or the missing line is added to the caption of the panel that adapts its paragraph. Nothing is lost, and `chapter_NN/coverage.json` records what happened.

Segment plans are saved in `chapter_NN/segments/`, so a run interrupted mid-chapter resumes without paying for the finished segments again. With a small local LLM, lower the segment size (for example `--segment-words 600`) so each request fits its context window.

## Tests

```bash
python -m pytest -q
```

The tests use canned plans, the mock image backend and the mock planner. They cover:

- **Character continuity.** Chapter 2 keeps chapter 1's designs even when the planner tries to redesign someone, resolves aliases (`Mimi` → `Mira Han`) and applies outfit changes.
- **Long chapters.** Splitting loses no words, skipped paragraphs and dialogue are detected, retries fill the gaps, stubborn gaps are patched in story order, and saved segments are reused on resume.
- **The interactive app.** A full session and the ComfyUI setup flow, driven through a pseudo-terminal.
- **ComfyUI backend.** Detection of the three model files, linking, the Qwen-Image 2.1 graph (with reference images), custom-workflow placeholders, and the HTTP round trip against a fake ComfyUI server. The generated graph was also checked against a real ComfyUI with ComfyUI-GGUF: the server accepted it, and only failed when it tried to load the stand-in model files.
