#!/usr/bin/env python3
"""Speak Claude's last reply as Commander Data. Started in the background by speak.sh.

Pipeline:
  last reply -> prepare_text.py (strip markdown, expand contractions)
             -> Data rewrite by Claude Haiku (claude -p, ~2.5 s; ~/AI-Workspace/data-persona/rewrite.py)
             -> Qwen3-TTS voice clone on the :8880 voice server (data_ref.wav, temperature 0.5)
             -> afplay
Fallbacks: Haiku fails, or the session runs on the local model (ANTHROPIC_BASE_URL set) -> local
two-pass rewriter on :8082; that fails too -> first 3 prepared sentences; voice server down -> `say`.
Standard library only.
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
VOICE_DIR = os.path.join(HERE, "..", "data_output", "cmdr_data_voice")
REWRITER_DIR = os.path.expanduser("~/AI-Workspace/data-persona")
STATE = os.path.expanduser("~/.claude/speak")
PIDFILE = os.path.join(STATE, "pid")
TTS_URL = "http://127.0.0.1:8880/v1/audio/speech"
VOICE_HEALTH = "http://127.0.0.1:8880/docs"
VOICE_START = os.path.expanduser("~/models/tools/voice.sh")
VOICE_LOG = os.path.expanduser("~/models/tools/voice.log")
TTS_CLI = os.path.expanduser("~/.local/bin/mlx_audio.tts.generate")

sys.path[:0] = [VOICE_DIR, REWRITER_DIR]
from prepare_text import prepare  # noqa: E402


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def stop_previous():
    """Kill the previous reply's pipeline (and its afplay) via its process group, then claim the pidfile."""
    try:
        old = int(open(PIDFILE).read())
        if old != os.getpid():
            os.killpg(old, signal.SIGTERM)
    except (OSError, ValueError):
        pass
    open(PIDFILE, "w").write(str(os.getpid()))


def last_reply(hook):
    """Text of the final reply: assistant text blocks after the last user/tool-result entry."""
    if hook.get("last_assistant_message"):
        return hook["last_assistant_message"]
    path = hook.get("transcript_path")
    if not path or not os.path.exists(path):
        return ""
    time.sleep(0.5)  # let Claude Code finish flushing the transcript
    texts = []
    for line in open(path, encoding="utf-8").readlines()[-300:]:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("isSidechain") or e.get("type") not in ("user", "assistant"):
            continue
        if e["type"] == "user":
            texts = []
            continue
        content = e.get("message", {}).get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts += [b.get("text", "") for b in content if b.get("type") == "text"]
    return "\n\n".join(t for t in texts if t.strip())


def to_data(text):
    if not os.environ.get("ANTHROPIC_BASE_URL"):  # local-model sessions stay local
        try:
            from rewrite import rewrite_haiku
            out = rewrite_haiku(text)
            log(f"haiku: {out}")
            return prepare(out)  # safety net: contractions
        except Exception as e:
            log(f"haiku unavailable ({e}); using local rewriter")
    try:
        from rewrite import rewrite_both
        plain, out = rewrite_both(text)
        log(f"condensed: {plain}")
        log(f"rewritten: {out}")
        return prepare(out)  # safety net: contractions
    except Exception as e:
        log(f"rewriter unavailable ({e}); using first sentences")
        return prepare(text, max_sentences=3)


def speak(text):
    """Data's voice by the fastest route that works; macOS `say` only if every route fails."""
    cfg = json.load(open(os.path.join(VOICE_DIR, "voice_config.json")))
    wav = os.path.join(STATE, f"reply-{os.getpid()}.wav")
    try:
        for route in (tts_server, tts_cli):
            try:
                route(text, cfg, wav)
                subprocess.run(["afplay", wav])
                return
            except Exception as e:
                log(f"{route.__name__} failed ({e})")
        log("all voice routes failed; using say")
        subprocess.run(["say", text])
    finally:
        if os.path.exists(wav):
            os.remove(wav)


def voice_server_up():
    try:
        urllib.request.urlopen(VOICE_HEALTH, timeout=2)
        return True
    except Exception:
        return False


def tts_server(text, cfg, wav):
    """Resident voice server (~1.5 s). Starts it if the local AI stack is down."""
    if not voice_server_up():
        log("voice server down; starting it")
        # own session, so the next reply's stop_previous() does not kill the server
        subprocess.Popen([VOICE_START], stdin=subprocess.DEVNULL, stdout=open(VOICE_LOG, "a"),
                         stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(90):
            time.sleep(1)
            if voice_server_up():
                break
        else:
            raise RuntimeError("voice server did not come up within 90 s")
    body = {"model": cfg["model"], "input": text, "response_format": "wav",
            "ref_audio": os.path.join(VOICE_DIR, cfg["ref_audio"]),
            "ref_text": open(os.path.join(VOICE_DIR, cfg["ref_text_file"])).read().strip(),
            "temperature": cfg["temperature"], "lang_code": cfg["lang_code"]}
    req = urllib.request.Request(TTS_URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        open(wav, "wb").write(r.read())


def tts_cli(text, cfg, wav):
    """No server: load the model directly (~12-20 s), still in Data's voice."""
    prefix = wav[:-4]
    subprocess.run([TTS_CLI, "--model", cfg["model"], "--text", text,
                    "--ref_audio", os.path.join(VOICE_DIR, cfg["ref_audio"]),
                    "--ref_text", open(os.path.join(VOICE_DIR, cfg["ref_text_file"])).read().strip(),
                    "--temperature", str(cfg["temperature"]), "--lang_code", cfg["lang_code"],
                    "--output_path", STATE, "--file_prefix", os.path.basename(prefix),
                    "--audio_format", "wav", "--join_audio"],
                   check=True, capture_output=True, timeout=180)
    made = sorted(f for f in os.listdir(STATE) if f.startswith(os.path.basename(prefix)) and f.endswith(".wav"))
    if not made:
        raise RuntimeError("CLI produced no audio")
    if made[0] != os.path.basename(wav):
        os.replace(os.path.join(STATE, made[0]), wav)


def main():
    os.setsid()  # own process group, so the next reply can stop this one cleanly
    os.makedirs(STATE, exist_ok=True)
    stop_previous()
    try:
        text = last_reply(json.loads(sys.stdin.read() or "{}"))
        if not text.strip():
            log("no reply text")
            return
        spoken = to_data(text)
        if spoken:
            speak(spoken)
    finally:
        try:
            if open(PIDFILE).read() == str(os.getpid()):
                os.remove(PIDFILE)
        except OSError:
            pass


if __name__ == "__main__":
    main()
