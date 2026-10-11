"""Rewrite an assistant reply into a short spoken line in Data's voice.

Primary: Claude Haiku through the logged-in Claude Code CLI (`claude -p`, no API key), prompt in
haiku_system.txt plus optional private style examples (DATA_VOICE_EXAMPLES).

Fallback: two passes on one llama-server (DATA_VOICE_REWRITER_URL, default 127.0.0.1:8082) with the
base model and the Data LoRA loaded:
  1. base model (LoRA off): condense the reply into 1-3 plain sentences
  2. Data LoRA on:          restyle those sentences in Data's voice

Usage: python3 rewrite.py [--local] < reply.txt
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
URL = os.environ.get("DATA_VOICE_REWRITER_URL", "http://127.0.0.1:8082") + "/v1/chat/completions"
HAIKU_MODEL = os.environ.get("DATA_VOICE_HAIKU_MODEL", "claude-haiku-4-5")
HAIKU_SYSTEM = os.path.join(HERE, "haiku_system.txt")
# Real Data lines used as style examples. Kept outside the repo: they are copyrighted dialogue.
EXAMPLES = os.path.expanduser(os.environ.get("DATA_VOICE_EXAMPLES", "~/AI-Workspace/data-persona/haiku_examples.txt"))
SKIP = "SKIP"  # Haiku's answer when the reply holds nothing worth speaking

SUMMARIZE = """Condense the AI assistant's reply below into 1 to 3 short plain sentences that will be read aloud.

Rules:
- Keep the facts exactly: what was done, what failed, what is being asked. Never invent anything.
- Lead with the main outcome. Keep exact numbers.
- If the reply asks the user a question, end with that question and keep every option it offers.
  If it does not ask a question, do not add one.
- No lists, no markdown, no file names, paths, commands, or code.
- Output only the sentences."""

# Must match the system prompt used in training (build_train.py in the data-persona workspace)
DATA_STYLE = ("Rewrite the user's text as a short spoken report in the voice of Lieutenant Commander Data "
              "from Star Trek: The Next Generation. Keep every fact. Never use contractions.")


def speakable(reply):
    """Strip things that should never be read aloud before the model sees them."""
    r = re.sub(r"```.*?```", " ", reply, flags=re.S)                  # code blocks
    r = re.sub(r"https?://\S+", "a link", r)                          # URLs
    r = re.sub(r"`-{0,2}([A-Za-z][\w.-]{0,30})`", r"\1", r)             # inline names: `yt-dlp`, `--check`
    r = re.sub(r"`[^`]*\(\)`", "a function", r)                        # inline function names
    r = re.sub(r"`[^`]*`", "the code", r)                              # other inline code
    r = re.sub(r"[\w./~-]*\.(py|js|ts|json|jsonl|sh|md|txt|ini|gguf|yaml|toml)\b(:\d+)?", "a file", r)
    r = re.sub(r"(?<![\w/])(~/|/)[A-Za-z][\w./~-]*", "a folder", r)    # remaining absolute/home paths
    r = re.sub(r"\b(\d+(?:\.\d+)?)B\b", r"\1-billion-parameter", r)    # model sizes ("4B")
    r = re.sub(r"[*_#>|]", "", r)                                      # markdown
    return re.sub(r"[ \t]+", " ", r)[:4000]


def drop_sir(text):
    """Remove "sir" as a form of address ("Yes, sir." / "Sir, I have..."); Data's real lines are full of it."""
    text = re.sub(r"(^|(?<=[.!?] ))[Ss]ir, *", "", text)        # "Sir, I have" -> "I have"
    text = re.sub(r",? +[Ss]ir\b(?=[,.!?;:]|$)", "", text)          # "Yes, sir." / "push this, sir, or"
    text = re.sub(r",(?=[.!?])", "", text)                       # leftover "x,." from the line above
    return re.sub(r"(^|[.!?] )([a-z])", lambda m: m.group(1) + m.group(2).upper(), text).strip()


def chat(system, text, lora_scale, temperature):
    body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
            "temperature": temperature, "max_tokens": 200,
            "lora": [{"id": 0, "scale": lora_scale}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


def rewrite_both(reply):
    """Return (plain, data) tuple: condensed reply and Data-voiced version."""
    text = speakable(reply)
    plain = chat(SUMMARIZE, text, lora_scale=0.0, temperature=0.0)
    if "?" not in text:  # the small model likes to tack on "What's next?"; drop invented questions
        plain = " ".join(s for s in re.split(r"(?<=[.!?])\s+", plain) if not s.endswith("?")) or plain
    data = chat(DATA_STYLE, plain, lora_scale=1.0, temperature=0.3)
    # Guard: if plain ends with ? but data doesn't have ?, use plain as output
    if plain.rstrip().endswith("?") and "?" not in data:
        data = plain
    return (plain, drop_sir(data))


def rewrite(reply):
    return rewrite_both(reply)[1]


def haiku_system():
    system = open(HAIKU_SYSTEM).read().rstrip()
    if os.path.exists(EXAMPLES):
        system += " Real lines, for style only:\n" + open(EXAMPLES).read().strip()
    return system


def rewrite_haiku(reply, timeout=20):
    """One-step Data rewrite by Claude Haiku via the logged-in Claude Code CLI (no API key).
    Returns "" when Haiku judges there is nothing worth speaking.
    DATA_VOICE_CHILD tells the speak hook to ignore this child session's own Stop event."""
    cmd = ["claude", "-p", "--model", HAIKU_MODEL, "--system-prompt", haiku_system(),
           "--tools", "", "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence",
           "--output-format", "text"]
    # thinking off: ~2.5 s per call instead of 10-60 s
    env = {**os.environ, "DATA_VOICE_CHILD": "1", "MAX_THINKING_TOKENS": "0"}
    text = speakable(reply).strip()
    # Labelled as Data's own message: bare, the text reads as a request to Haiku and it answers it
    prompt = f"Say this message of yours to the user:\n<report>\n{text}\n</report>"
    out = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, env=env,
                         cwd=tempfile.gettempdir(), check=True).stdout.strip()
    if not out:
        raise RuntimeError("empty reply from Haiku")
    if out.strip(" .") == SKIP:
        return ""
    questions = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.endswith("?")]
    if questions and "?" not in out:  # Haiku occasionally turns the question into a statement
        out += " " + questions[-1]
    return drop_sir(out)


if __name__ == "__main__":
    reply = sys.stdin.read()
    print(rewrite(reply) if "--local" in sys.argv else rewrite_haiku(reply))
