# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CharacterVoiceCloning extracts clean voice clips from YouTube and uses them as reference audio for TTS voice cloning via Qwen3-TTS on Apple Silicon. Built for personal, non-commercial use. The audio extraction pipeline uses subprocess calls to `yt-dlp` and `ffmpeg`. Voice cloning uses [mlx-audio](https://github.com/Blaizzy/mlx-audio) (MLX port of Qwen3-TTS).

## Commands

```bash
# Full workflow: search YouTube, download, extract clips interactively
uv run python data_extractor.py "Data Star Trek voice lines"
uv run python data_extractor.py -n 10 -o data_output "query here"

# Extract clips from already-downloaded files
uv run python data_extractor.py --clip -o data_output

# Benchmark CustomVoice (predefined speaker) inference speed
uv run python benchmark_tts.py 0.6B-8bit 1.7B-8bit

# Test voice cloning with reference clips (default: data_ref, the chosen voice)
uv run python test_voice_clone.py 0.6B-8bit --clips data_ref

# Generate speech directly via CLI
mlx_audio.tts.generate \
  --model mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit \
  --text "Hello" \
  --ref_audio data_output/cmdr_data_voice/data_ref.wav \
  --ref_text "$(cat data_output/cmdr_data_voice/data_ref.txt)" \
  --temperature 0.5 --play --stream

# Voice hook: check dependencies, or speak a test reply
python3 claude-code-hook/speak_pipeline.py --check
echo "Reply text" | python3 claude-code-hook/speak_pipeline.py --test
```

External tools required: `yt-dlp`, `ffmpeg` (both via `brew install`).

## Architecture

- **`data_extractor.py`** -- YouTube search/download and interactive clip extraction. Downloads go to `<output-dir>/downloads/`, clips to `<output-dir>/clips/`. Clips get audio filtering (highpass, lowpass, noise reduction, loudness normalization) and are resampled to 22.05kHz mono.

- **`benchmark_tts.py`** -- Benchmarks Qwen3-TTS CustomVoice models (predefined speakers like "Ryan") on MLX. Measures RTF, generation time, audio duration.

- **`test_voice_clone.py`** -- Tests voice cloning with Qwen3-TTS Base models using reference audio clips. Supports both x_vector mode (no transcript) and transcript mode. Outputs to `clone_outputs/`.

- **`import_clips.py`** -- Imports soundboard MP3s as cleaned clips and can join several into one reference.

- **`claude-code-hook/`** -- The Claude Code Stop hook that speaks replies in the cloned voice (see Hook Setup).

## Winning Configuration

- **Model**: `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit` (0.25x RTF on mlx-audio 0.5.8; 0.62x on 0.3)
- **Mode**: Transcript mode (`ref_audio` + `ref_text`) -- significantly better than x_vector-only
- **Reference**: A single clean 10-15 second clip is sufficient
- **Platform**: MLX on Apple Silicon. PyTorch MPS was ~3-4x slower and produced poor quality audio.

## Hook Setup

The Claude Code voice hook runs from `claude-code-hook/` in this repo (see its README):
- `speak.sh` -- the Stop hook; exits immediately if `~/.claude/speak-on` is absent, otherwise backgrounds the pipeline and exits 0
- `speak_pipeline.py` -- last reply -> `rewrite.py` (Data rewrite by Claude Haiku via `claude -p`, thinking off; local :8082 two-pass rewriter as fallback and for local-model sessions) -> `prepare_text.py` (markdown, contractions, cap 3 sentences / 60 words) -> Qwen3-TTS on the resident voice server (:8880), one sentence at a time -> `afplay`
- `haiku_system.txt` -- Haiku prompt. Real Data lines are appended at runtime from `DATA_VOICE_EXAMPLES` (default `~/AI-Workspace/data-persona/haiku_examples.txt`), kept out of the repo for copyright. Haiku answers `SKIP` when there is nothing to say.
- `voice_config.json` -- TTS model and settings; reference files live in `DATA_VOICE_DIR` (default `data_output/cmdr_data_voice/`, git-ignored)
- Settings are environment variables, listed at the top of `speak_pipeline.py` and `rewrite.py`
- Log: `~/.claude/speak/speak.log`; process-group id in `~/.claude/speak/pid`

Keep the hook standard-library only: it runs under the system `python3`, not the project venv.

Registered in `~/.claude/settings.json` under `hooks.Stop` (timeout 10).

## Next Steps

- Add a demo WAV sample to the README so visitors can hear the cloned voice output
- Add this project to personal website's project portfolio section
