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
# to run Qwen-Image locally with diffusers, also:
pip install torch "diffusers>=0.36" transformers accelerate
```

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
    --image-backend diffusers --image-model /path/to/qwen-image-2.1

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

Pick the backend that matches how you run the model. You can change it any time with `init` on an existing project.

| Backend | When | Example |
|---|---|---|
| `diffusers` (default) | Weights on this machine (GPU) | `--image-backend diffusers --image-model /models/qwen-image-2.1` (or a Hugging Face repo id) |
| `openai` | You serve it behind an OpenAI-style `/v1/images/generations` API (vLLM-Omni, LocalAI, your own FastAPI…) | `--image-backend openai --image-base-url http://gpu-box:8000/v1 --image-model qwen-image-2.1` (optional `IMAGE_API_KEY`) |
| `dashscope` | Alibaba Cloud Model Studio | `--image-backend dashscope --image-model <model name>` with `DASHSCOPE_API_KEY` set (`--image-base-url` to use a different region endpoint) |
| `mock` | Testing without a GPU | draws placeholder panels |

Other knobs: `--steps`, `--cfg` (passed as `true_cfg_scale`), `--style`, `--negative`, `--width`, `--font`. Set `WEBTOON_CPU_OFFLOAD=1` to use diffusers CPU offload if VRAM is tight.

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
- **The interactive app.** A full session driven through a pseudo-terminal.
