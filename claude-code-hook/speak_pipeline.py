#!/usr/bin/env python3
"""Speak Claude's last reply as Commander Data. Started in the background by speak.sh.

Pipeline:
  last reply -> Data rewrite by Claude Haiku (claude -p, ~2.5 s; rewrite.py)
             -> prepare_text.py (strip markdown, expand contractions, cap at 3 sentences / 60 words)
             -> Qwen3-TTS voice clone on the voice server, one sentence at a time
             -> afplay, playing each sentence while the next one is generated
Fallbacks: Haiku fails, or the session runs on a local model (ANTHROPIC_BASE_URL set) -> local two-pass
rewriter on :8082; that fails too -> first sentences of the reply. Voice server down -> start it and
wait; still down -> mlx_audio.tts.generate CLI; that fails too -> macOS `say`.

  python3 speak_pipeline.py --check          check every dependency, speak nothing
  echo "reply" | python3 speak_pipeline.py --test   speak a reply without a hook event

Standard library only. Settings (environment variables, all optional):
  DATA_VOICE_DIR            voice package with data_ref.wav / data_ref.txt (../data_output/cmdr_data_voice)
  DATA_VOICE_TTS_URL        mlx-audio voice server (http://127.0.0.1:8880)
  DATA_VOICE_SERVER_CMD     command that starts the voice server (~/models/tools/voice.sh)
  DATA_VOICE_MAX_SENTENCES  longest spoken reply, in sentences (3)
  DATA_VOICE_MAX_WORDS      ... and in words, stopping at a sentence end (60)
  DATA_VOICE_HAIKU_MODEL, DATA_VOICE_REWRITER_URL, DATA_VOICE_EXAMPLES: see rewrite.py
"""
import glob
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
env = os.environ.get
VOICE_DIR = os.path.normpath(os.path.expanduser(env("DATA_VOICE_DIR", os.path.join(HERE, "..", "data_output", "cmdr_data_voice"))))
CONFIG = os.path.join(HERE, "voice_config.json")
STATE = os.path.expanduser("~/.claude/speak")
PIDFILE = os.path.join(STATE, "pid")
TTS_BASE = env("DATA_VOICE_TTS_URL", "http://127.0.0.1:8880")
VOICE_START = os.path.expanduser(env("DATA_VOICE_SERVER_CMD", "~/models/tools/voice.sh"))
VOICE_LOG = os.path.splitext(VOICE_START)[0] + ".log"
TTS_CLI = (shutil.which("mlx_audio.tts.generate")
           or os.path.expanduser("~/.local/bin/mlx_audio.tts.generate"))
MAX_SENTENCES = int(env("DATA_VOICE_MAX_SENTENCES", "3"))
MAX_WORDS = int(env("DATA_VOICE_MAX_WORDS", "60"))  # ~20 s of speech
MIN_CHUNK = 25  # shorter sentences are merged with the next, so each TTS request has enough context

sys.path.insert(0, HERE)
from prepare_text import prepare  # noqa: E402
import rewrite  # noqa: E402


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def stop_previous():
    """Kill the previous reply's pipeline (and its afplay) via its process group, then claim the pidfile.
    Only kills if that pid is still a speak pipeline: a stale pid may since belong to something else."""
    try:
        old = int(open(PIDFILE).read())
        cmd = subprocess.run(["ps", "-o", "command=", "-p", str(old)], capture_output=True, text=True).stdout
        if old != os.getpid() and "speak_pipeline.py" in cmd:
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
    """Data's spoken version of the reply, at most MAX_SENTENCES long; "" when there is nothing to say."""
    if not env("ANTHROPIC_BASE_URL"):  # local-model sessions stay local
        try:
            out = rewrite.rewrite_haiku(text)
            log(f"haiku: {out or '(skip)'}")
            return prepare(out, max_sentences=MAX_SENTENCES, max_words=MAX_WORDS)  # safety net: contractions, length
        except Exception as e:
            log(f"haiku unavailable ({e}); using local rewriter")
    try:
        plain, out = rewrite.rewrite_both(text)
        log(f"condensed: {plain}")
        log(f"rewritten: {out}")
        return prepare(out, max_sentences=MAX_SENTENCES, max_words=MAX_WORDS)
    except Exception as e:
        log(f"rewriter unavailable ({e}); using first sentences")
        return prepare(text, max_sentences=MAX_SENTENCES, max_words=MAX_WORDS)


def chunks(text):
    """Split into sentences for streaming, merging short ones into their neighbour."""
    out, buf = [], ""
    for s in re.split(r"(?<=[.!?])\s+", text.strip()):
        buf = f"{buf} {s}".strip()
        if len(buf) >= MIN_CHUNK:
            out.append(buf)
            buf = ""
    if buf:
        if out and len(buf) < MIN_CHUNK:
            out[-1] += " " + buf
        else:
            out.append(buf)
    return out


def speak(text):
    """Data's voice by the fastest route that works; macOS `say` only if every route fails."""
    cfg = json.load(open(CONFIG))
    try:
        stream_server(text, cfg)
        return
    except Exception as e:
        log(f"voice server failed ({e})")
    wav = os.path.join(STATE, f"reply-{os.getpid()}.wav")
    try:
        tts_cli(text, cfg, wav)
        subprocess.run(["afplay", wav])
        return
    except Exception as e:
        log(f"voice CLI failed ({e}); using say")
    subprocess.run(["say", text])


def voice_server_up():
    try:
        urllib.request.urlopen(TTS_BASE + "/docs", timeout=2)
        return True
    except Exception:
        return False


def ensure_voice_server():
    """Start the voice server if it is down (e.g. after `localai stop` or a reboot) and wait for it."""
    if voice_server_up():
        return
    if not os.path.exists(VOICE_START):
        raise RuntimeError(f"voice server down and {VOICE_START} not found")
    log("voice server down; starting it")
    # own session, so the next reply's stop_previous() does not kill the server
    subprocess.Popen([VOICE_START], stdin=subprocess.DEVNULL, stdout=open(VOICE_LOG, "a"),
                     stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(90):
        time.sleep(1)
        if voice_server_up():
            return
    raise RuntimeError("voice server did not come up within 90 s")


def tts_request(text, cfg, wav):
    body = {"model": cfg["model"], "input": text, "response_format": "wav",
            "ref_audio": os.path.join(VOICE_DIR, cfg["ref_audio"]),
            "ref_text": open(os.path.join(VOICE_DIR, cfg["ref_text_file"])).read().strip(),
            "temperature": cfg["temperature"], "lang_code": cfg["lang_code"]}
    req = urllib.request.Request(TTS_BASE + "/v1/audio/speech", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        open(wav, "wb").write(r.read())


def stream_server(text, cfg):
    """Generate one chunk at a time on the voice server and play each as soon as it is ready, so speech
    starts after the first sentence (~0.5 s) instead of after the whole reply. Raises only if nothing
    was played, so the caller can fall back to the CLI."""
    ensure_voice_server()
    ready = queue.Queue()

    def produce():
        for i, part in enumerate(chunks(text)):
            wav = os.path.join(STATE, f"reply-{os.getpid()}-{i}.wav")
            try:
                tts_request(part, cfg, wav)
            except Exception as e:
                ready.put(e)
                return
            ready.put(wav)
        ready.put(None)

    threading.Thread(target=produce, daemon=True).start()
    played = False
    while (item := ready.get()) is not None:
        if isinstance(item, Exception):
            if not played:
                raise item
            log(f"voice server failed mid-reply ({item})")
            return
        if not played:
            log(f"speaking ({len(chunks(text))} parts)")
        subprocess.run(["afplay", item])
        os.remove(item)
        played = True


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


def check():
    """Report on every dependency without speaking. Exit 1 if a required one is missing."""
    cfg = json.load(open(CONFIG))
    settings = os.path.expanduser("~/.claude/settings.json")
    hooked = os.path.exists(settings) and os.path.join(HERE, "speak.sh") in open(settings).read()
    items = [  # (required, ok, label)
        (True, os.path.exists(os.path.join(VOICE_DIR, cfg["ref_audio"])), f"reference audio in {VOICE_DIR}"),
        (True, os.path.exists(os.path.join(VOICE_DIR, cfg["ref_text_file"])), "reference transcript"),
        (True, hooked, f"Stop hook registered in {settings}"),
        (False, os.path.exists(os.path.expanduser("~/.claude/speak-on")), "speech switched on (~/.claude/speak-on)"),
        (False, bool(shutil.which("claude")), "claude CLI, for the Haiku rewrite"),
        (False, os.path.exists(rewrite.EXAMPLES), f"private style examples ({rewrite.EXAMPLES})"),
        (False, _reachable(rewrite.URL.rsplit("/v1/", 1)[0] + "/health"), "local rewriter (fallback)"),
        (False, voice_server_up(), f"voice server at {TTS_BASE}"),
        (False, os.path.exists(VOICE_START), f"voice server start command ({VOICE_START})"),
        (False, os.path.exists(TTS_CLI), "mlx_audio.tts.generate (fallback when the server is down)"),
    ]
    for required, ok, label in items:
        print(f"{'ok  ' if ok else ('MISSING' if required else 'off ')}  {label}")
    voice_ok = voice_server_up() or os.path.exists(VOICE_START) or os.path.exists(TTS_CLI)
    if not voice_ok:
        print("MISSING  any way to run the voice: start an mlx-audio server or install mlx-audio")
    return 0 if voice_ok and all(ok for required, ok, _ in items if required) else 1


def _reachable(url):
    try:
        urllib.request.urlopen(url, timeout=2)
        return True
    except Exception:
        return False


def main():
    if "--check" in sys.argv:
        sys.exit(check())
    try:
        os.setsid()  # own process group, so the next reply can stop this one cleanly
    except OSError:
        pass  # already a group leader (started from an interactive shell)
    # stop_previous() sends SIGTERM; exit through the finally blocks so temp audio is cleaned up
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    os.makedirs(STATE, exist_ok=True)
    stop_previous()
    try:
        stdin = sys.stdin.read()
        hook = {"last_assistant_message": stdin} if "--test" in sys.argv else json.loads(stdin or "{}")
        text = last_reply(hook)
        if not text.strip():
            log("no reply text")
            return
        spoken = to_data(text)
        if spoken:
            speak(spoken)
    finally:
        for f in glob.glob(os.path.join(STATE, f"reply-{os.getpid()}*.wav")):
            os.remove(f)
        try:
            if open(PIDFILE).read() == str(os.getpid()):
                os.remove(PIDFILE)
        except OSError:
            pass


if __name__ == "__main__":
    main()
