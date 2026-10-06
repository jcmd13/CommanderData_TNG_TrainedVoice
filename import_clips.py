#!/usr/bin/env python3
"""
Import pre-cut voice clips (e.g. from a soundboard site) for TTS voice cloning.

Each file in the input directory is one line of dialogue whose filename is its
transcript (e.g. "may-i-ask-you-a-question-sir.mp3"). A transcripts.json in the
same directory, mapping filename stem -> corrected transcript, takes precedence
over the filename.

Clips are run through the same filter chain as data_extractor.py and written to
<output-dir>/clips/ alongside a transcripts.json. Optionally, several clips can
be joined into a single reference clip (ref_audio + ref_text) for cloning.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from data_extractor import extract_clip

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}


def transcript_from_filename(stem):
    """Best-effort transcript from a slugified filename."""
    text = stem.replace("-", " ").replace("_", " ")
    for slug, word in [(" i m ", " I'm "), (" you ve ", " you've "), (" don t ", " don't ")]:
        text = f" {text} ".replace(slug, word).strip()
    text = " ".join("I" if w == "i" else w for w in text.split())
    return text[0].upper() + text[1:] + "."


def probe_duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def trim_silence(audio, sr, threshold_db=-40.0, pad=0.05):
    """Trim leading/trailing silence, keeping a short pad on each side."""
    frame = int(sr * 0.01)
    n_frames = len(audio) // frame
    if n_frames == 0:
        return audio
    rms = np.sqrt(np.mean(audio[: n_frames * frame].reshape(n_frames, frame) ** 2, axis=1))
    loud = np.where(20 * np.log10(rms + 1e-10) > threshold_db)[0]
    if len(loud) == 0:
        return audio
    start = max(0, loud[0] * frame - int(pad * sr))
    end = min(len(audio), (loud[-1] + 1) * frame + int(pad * sr))
    return audio[start:end]


def build_reference(clips_dir, transcripts, stems, output_path, gap=0.4):
    """Join processed clips into one reference WAV and write its transcript next to it."""
    pieces, texts, sr = [], [], None
    for stem in stems:
        audio, clip_sr = sf.read(str(clips_dir / f"{stem}.wav"))
        sr = sr or clip_sr
        pieces.append(trim_silence(audio, sr))
        texts.append(transcripts[stem])

    silence = np.zeros(int(gap * sr))
    joined = []
    for i, piece in enumerate(pieces):
        if i:
            joined.append(silence)
        joined.append(piece)
    audio = np.concatenate(joined)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sr)
    text = " ".join(texts)
    output_path.with_suffix(".txt").write_text(text + "\n")
    print(f"✓ Reference: {output_path} ({len(audio) / sr:.1f}s)")
    print(f"  Transcript: {text}")


def main():
    parser = argparse.ArgumentParser(description="Import pre-cut voice clips for TTS voice cloning")
    parser.add_argument("input_dir", help="Directory of clips whose filenames are their transcripts")
    parser.add_argument(
        "-o", "--output-dir",
        default="data_output",
        help="Base output directory (default: data_output). Clips go to <dir>/clips/",
    )
    parser.add_argument(
        "--reference",
        nargs="+",
        metavar="STEM",
        help="Join these clips (by filename stem) into <dir>/references/<--ref-name>.wav",
    )
    parser.add_argument("--ref-name", default="reference", help="Name for the joined reference clip")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    clips_dir = Path(args.output_dir) / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    if not shutil.which("ffmpeg"):
        print("✗ ffmpeg not found. Install with: brew install ffmpeg")
        return 1

    overrides_path = input_dir / "transcripts.json"
    overrides = json.loads(overrides_path.read_text()) if overrides_path.exists() else {}

    sources = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)
    if not sources:
        print(f"No audio files found in {input_dir}")
        return 1

    manifest = {}
    for src in sources:
        out_path = clips_dir / f"{src.stem}.wav"
        if not extract_clip(str(src), 0, probe_duration(src), str(out_path)):
            continue
        manifest[src.stem] = {
            "path": str(out_path),
            "duration": round(sf.info(str(out_path)).duration, 2),
            "transcript": overrides.get(src.stem) or transcript_from_filename(src.stem),
        }

    manifest_path = clips_dir / "transcripts.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(m["duration"] for m in manifest.values())
    print(f"\n✓ Processed {len(manifest)}/{len(sources)} clips ({total:.1f}s total) -> {clips_dir}/")
    print(f"✓ Transcripts: {manifest_path}")

    if args.reference:
        transcripts = {stem: m["transcript"] for stem, m in manifest.items()}
        missing = [s for s in args.reference if s not in transcripts]
        if missing:
            print(f"Unknown clip(s): {', '.join(missing)}")
            return 1
        build_reference(
            clips_dir, transcripts, args.reference,
            Path(args.output_dir) / "references" / f"{args.ref_name}.wav",
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
