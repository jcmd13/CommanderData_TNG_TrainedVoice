#!/usr/bin/env python3
"""
Test voice cloning with Qwen3-TTS Base models on MLX.
Tries different reference clips and model sizes to compare quality and speed.
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model


MODELS = {
    "0.6B-8bit": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
    "0.6B-bf16": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
    "1.7B-8bit": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
    "1.7B-bf16": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
}

# Reference clips with hand-written transcripts (longest clips first)
# Update these transcripts to match the actual audio content
REFERENCE_CLIPS = {
    "clip_14": {
        "path": "data_output/clips/clip_14.wav",
        "transcript": (
            "I feel nothing at all. That is part of my dilemma. "
            "I have the curiosity of humans, but there are questions that I will never have the answers to: "
            "what it is like to laugh, to cry, or to experience any hu-"
        ),
        "duration": "14s",
    },
    # Joined soundboard clips built by: import_clips.py commander_data_audio_files --reference ...
    "data_ref_a": {
        "path": "data_output/references/data_ref_a.wav",
        "transcript": (
            "I have an ultimate storage capacity of eight hundred quadrillion bits. "
            "My total linear computational speed has been rated at sixty trillion operations per second. "
            "The anomaly is two hundred million kilometers in diameter."
        ),
        "duration": "12.5s",
    },
    "data_ref_b": {
        "path": "data_output/references/data_ref_b.wav",
        "transcript": (
            "Although I do not speak from personal experience, I have seen it have a profound psychological impact. "
            "I have observed that the selection of food is often influenced by the mood of the person ordering."
        ),
        "duration": "11.0s",
    },
    "data_ref_c": {
        "path": "data_output/references/data_ref_c.wav",
        "transcript": (
            "In almost all societies, it is traditional to say a ritual farewell to those you call friends. "
            "When one of my friends is distraught, I have learned that the thoughtful thing to do "
            "is to attempt to make him feel more comfortable."
        ),
        "duration": "13.4s",
    },
    # Data-only windows from "The Offspring" (YouTube Short D0IY_t2itPE), cut with extract_clip
    "short_a": {
        "path": "data_output/references/short_a.wav",
        "transcript": (
            "I have asked myself that many times, as I have struggled to be more human. "
            "Until I realized, it is the struggle itself that is most important."
        ),
        "duration": "13.3s",
    },
    "short_b": {
        "path": "data_output/references/short_b.wav",
        "transcript": (
            "We must strive to be more than we are, Lal. "
            "It does not matter that we will never reach our ultimate goal. "
            "The effort yields its own rewards."
        ),
        "duration": "12.0s",
    },
    "short_full": {
        "path": "data_output/references/short_full.wav",
        "transcript": (
            "I have asked myself that many times, as I have struggled to be more human. "
            "Until I realized, it is the struggle itself that is most important. "
            "We must strive to be more than we are, Lal. "
            "It does not matter that we will never reach our ultimate goal. "
            "The effort yields its own rewards."
        ),
        "duration": "25.3s",
    },
    # Data-only windows from "Darmok" (YouTube 4-O5Fr-jDe0) and the holodeck Short (R7JfEfIX9As)
    "darmok_a": {
        "path": "data_output/references/darmok_a.wav",
        "transcript": (
            "The Tamarian ego structure does not seem to allow what we normally think of as self-identity. "
            "Their ability to abstract is highly unusual. They seem to communicate through narrative imagery, "
            "a reference to the individuals and places which appear in their mytho-historical accounts."
        ),
        "duration": "15.9s",
    },
    "darmok_b": {
        "path": "data_output/references/darmok_b.wav",
        "transcript": (
            "No, sir. The situation is analogous to understanding the grammar of a language, "
            "but none of the vocabulary. "
            "It is necessary for us to learn the narrative from which the Tamarians draw their imagery."
        ),
        "duration": "12.4s",
    },
    "holo_a": {
        "path": "data_output/references/holo_a.wav",
        "transcript": (
            "Through deduction, sir. Lieutenant Barkley and I tried to transport a simulated object off the holodeck, "
            "something that has never been attempted. Since the transporter itself is a simulation, "
            "the computer had no real data from which to create the transport logs."
        ),
        "duration": "12.1s",
    },
    # Data's voiceover narration from "Data's Day" (TNG S04E11), center channel of the Blu-ray 7.1 mix
    "ve_crusher": {
        "path": "data_output/references/ve_crusher.wav",
        "transcript": (
            "I am rarely in need of Doctor Beverly Crusher's professional services, "
            "as my biomechanical maintenance program is self-sufficient. "
            "But I often observe as she practices medicine on others, "
            "and have learned a great deal about human interaction from her."
        ),
        "duration": "18.8s",
    },
    "ve_worf": {
        "path": "data_output/references/ve_worf.wav",
        "transcript": (
            "I find Lieutenant Worf to be what is called a kindred spirit. "
            "We were both orphans rescued by Starfleet officers. "
            "In many ways, we are both still outsiders in human society."
        ),
        "duration": "11.7s",
    },
    "ve_vulcan": {
        "path": "data_output/references/ve_vulcan.wav",
        "transcript": (
            "Since I am not affected by emotional considerations, I am closer to being Vulcan than human. "
            "However, while their devotion to logic does have a certain appeal, "
            "I find their stark philosophy to be somewhat... limited."
        ),
        "duration": "12.8s",
    },
}

TEST_SENTENCES = [
    "I am functioning within normal parameters.",
    "Captain, I believe I have found an anomaly in the sensor readings that warrants further investigation.",
    (
        "It is curious. I am apparently motivated by a desire to understand human behavior, "
        "and yet I frequently find it baffling."
    ),
]

OUTPUT_DIR = Path("clone_outputs")


def run_clone_test(model, model_key, clip_key, clip_info, use_transcript=True):
    """Run voice cloning with a single reference clip and generate test sentences."""
    ref_audio = clip_info["path"]
    ref_text = clip_info["transcript"] if use_transcript else None
    mode = "transcript" if (use_transcript and ref_text) else "xvector"

    print(f"\n  --- Ref: {clip_key} ({clip_info['duration']}) | Mode: {mode} ---")

    if use_transcript and ref_text is None:
        print(f"    Skipping transcript mode (no transcript provided for {clip_key})")
        return None

    out_dir = OUTPUT_DIR / model_key / f"{clip_key}_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, text in enumerate(TEST_SENTENCES, 1):
        print(f"    Sentence {i}: \"{text[:60]}{'...' if len(text) > 60 else ''}\"")

        gen_results = list(model.generate(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            verbose=False,
        ))

        audio_arrays = []
        total_proc_time = 0.0
        sr = None
        for r in gen_results:
            audio_arrays.append(np.array(r.audio, copy=False))
            total_proc_time += r.processing_time_seconds
            sr = r.sample_rate

        audio = np.concatenate(audio_arrays) if len(audio_arrays) > 1 else audio_arrays[0]
        audio_duration = len(audio) / sr
        rtf = total_proc_time / audio_duration

        out_path = out_dir / f"sentence_{i}.wav"
        sf.write(str(out_path), audio, sr)

        results.append({
            "gen_time": total_proc_time,
            "audio_duration": audio_duration,
            "rtf": rtf,
        })

        print(f"      {total_proc_time:.2f}s -> {audio_duration:.2f}s audio (RTF: {rtf:.2f}x) -> {out_path}")

    avg_rtf = sum(r["gen_time"] for r in results) / sum(r["audio_duration"] for r in results)
    print(f"    Average RTF: {avg_rtf:.2f}x")
    return {"clip": clip_key, "mode": mode, "avg_rtf": avg_rtf, "results": results}


def benchmark_model(model_key, clips_to_test):
    model_id = MODELS[model_key]
    print(f"\n{'=' * 60}")
    print(f"Model: {model_id}")
    print(f"{'=' * 60}")

    print("Loading model...")
    load_start = time.time()
    model = load_model(model_id)
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.1f}s")

    all_test_results = []

    for clip_key in clips_to_test:
        clip_info = REFERENCE_CLIPS[clip_key]
        if not Path(clip_info["path"]).exists():
            print(f"\n  Skipping {clip_key}: file not found at {clip_info['path']}")
            continue

        # x_vector mode (no transcript needed)
        result = run_clone_test(model, model_key, clip_key, clip_info, use_transcript=False)
        if result:
            all_test_results.append(result)

        # With transcript (if available)
        if clip_info["transcript"]:
            result = run_clone_test(model, model_key, clip_key, clip_info, use_transcript=True)
            if result:
                all_test_results.append(result)

    del model
    mx.clear_cache()

    return {"model": model_key, "load_time": load_time, "tests": all_test_results}


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-TTS voice cloning on Apple Silicon (MLX)")
    parser.add_argument(
        "models",
        nargs="*",
        default=["0.6B-8bit"],
        choices=list(MODELS.keys()),
        help=f"Which model(s) to test (default: 0.6B-8bit). Options: {list(MODELS.keys())}",
    )
    parser.add_argument(
        "--clips",
        nargs="+",
        default=["clip_14"],
        choices=list(REFERENCE_CLIPS.keys()),
        help=f"Which reference clips to use (default: clip_14). Options: {list(REFERENCE_CLIPS.keys())}",
    )
    args = parser.parse_args()

    print("Qwen3-TTS Voice Cloning Test (MLX)")
    print(f"Output directory: {OUTPUT_DIR}/")

    all_results = []
    for model_key in args.models:
        try:
            result = benchmark_model(model_key, args.clips)
            all_results.append(result)
        except Exception as e:
            print(f"\nError testing {model_key}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if all_results:
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        for r in all_results:
            print(f"\n  {r['model']} (load: {r['load_time']:.1f}s):")
            for t in r["tests"]:
                print(f"    {t['clip']} [{t['mode']}]: avg RTF {t['avg_rtf']:.2f}x")

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")
    print("Listen and compare quality across models and clips!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
