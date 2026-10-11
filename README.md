# CharacterVoiceCloning

Clone a fictional character's voice from YouTube clips and use it for text-to-speech on Apple Silicon. Built with [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS) via [mlx-audio](https://github.com/Blaizzy/mlx-audio) for fast local inference.

**This project is intended exclusively for personal use.** It is not designed for, nor should it be used for, commercial purposes, impersonation, or any use that would infringe on the rights of content creators or voice actors.

## Demo

https://github.com/user-attachments/assets/7314e629-4f60-4e36-8897-465eca991890


**Try it yourself** — the Colab notebook below lets you clone any voice from a short reference clip (no Apple Silicon required):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jcmd13/CommanderData_TNG_TrainedVoice/blob/master/voice_cloning_demo.ipynb)

## How It Works

1. **Extract clips** -- Search YouTube for videos featuring a character's voice, download them, and interactively extract clean speech clips with noise reduction and normalization
2. **Clone the voice** -- Use a reference clip + transcript with Qwen3-TTS Base models to generate new speech in the character's voice
3. **Use it** -- Integrate with tools like Claude Code hooks to have your AI assistant speak in the cloned voice

## Why MLX?

We tried PyTorch on Apple Silicon (MPS backend) first. It was ~3-4x slower than real-time and produced poor audio quality. Switching to [mlx-audio](https://github.com/Blaizzy/mlx-audio) gave us:

| Model | RTF (real-time factor) | Notes |
|-------|------------------------|-------|
| 0.6B-8bit | **0.62x** | Faster than real-time. Best speed/quality tradeoff. |
| 0.6B-bf16 | 2.71x | |
| 1.7B-8bit | 0.72x | Audio generations overly crisp for target character (Commander Data); Likely a good choice for other characters|
| 1.7B-bf16 | 4.25x | Slightly crisper audio, much slower. |

RTF < 1.0 means faster than real-time. The **0.6B-8bit** model is the sweet spot. These figures are from mlx-audio 0.3; on mlx-audio 0.5.8 (M4 Max), 0.6B-8bit with the chosen reference measures **0.25x** in transcript mode.

## Prerequisites

- macOS with Apple Silicon (M1+)
- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) and [ffmpeg](https://ffmpeg.org/) -- `brew install yt-dlp ffmpeg`

## Quick Start

### Extract voice clips from YouTube

```bash
uv run python data_extractor.py "Data Star Trek voice lines"
```

This searches YouTube, downloads selected videos, and walks you through extracting time ranges as individual clips. Clips are saved to `data_output/clips/`.

### Generate speech with the cloned voice

```bash
# Using the CLI (works anywhere if mlx-audio is installed globally: uv tool install mlx-audio)
mlx_audio.tts.generate \
  --model mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit \
  --text "I am functioning within normal parameters." \
  --ref_audio data_output/cmdr_data_voice/data_ref.wav \
  --ref_text "$(cat data_output/cmdr_data_voice/data_ref.txt)" \
  --temperature 0.5 \
  --play --stream
```

Or compare models and reference clips (outputs go to `clone_outputs/`):

```bash
uv run python test_voice_clone.py 0.6B-8bit 1.7B-bf16 --clips data_ref darmok_b
```

The reference audio is not in this repo (`data_output/` is git-ignored): it is copyrighted dialogue. Cut your own 10–15 s clip of the character speaking alone, in a calm working voice, with no music, echo or background noise, and write its exact transcript next to it.

### Claude Code hook

A Claude Code [Stop hook](https://code.claude.com/docs/en/hooks) speaks a short version of every Claude reply in the cloned voice. Claude Haiku rewrites the reply as Data would say it (at most 3 sentences), and the voice server starts playing after the first sentence, about 3 s after the reply ends. Run `python3 claude-code-hook/speak_pipeline.py --check` to see what is set up. See [`claude-code-hook/`](claude-code-hook/) for setup instructions.

## Key Findings

- **Transcript mode matters**: Providing a text transcript of the reference clip (`--ref_text`) significantly improves voice cloning quality vs. x_vector-only mode (no transcript).
- **Single clip is enough**: A single clean 10-15 second clip works well. We didn't need to concatenate multiple clips.
- **Quantization is fine**: The 8-bit quantized model sounds nearly as good as bf16 at 4x the speed.

## Audio Processing

Extracted clips are automatically processed with:
- High-pass filter (80Hz) to remove low rumble
- Low-pass filter (8kHz) to remove hiss
- Adaptive noise reduction
- Loudness normalization (ITU-R BS.1770)
- Downsampled to 22.05kHz mono

## Acknowledgments

- [mlx-audio](https://github.com/Blaizzy/mlx-audio) by Prince Canuma -- MLX inference library that makes this project possible. MIT License.
- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) by Alibaba Qwen -- the underlying TTS model
- [mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit) -- quantized MLX weights used in this project
