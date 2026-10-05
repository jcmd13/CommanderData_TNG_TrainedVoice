# Claude Code Voice Hook

Speaks a short Commander Data version of each Claude Code reply. Uses the [Stop hook](https://docs.anthropic.com/en/docs/claude-code/hooks), which fires after each response.

## How it works

`speak.sh` checks the on/off flag, starts `speak_pipeline.py` in the background and exits 0 at once, so it never holds up Claude Code. The pipeline:

1. Takes the final reply from the transcript: the assistant text after the last tool result or user message.
2. Cleans it with `../data_output/cmdr_data_voice/prepare_text.py`: removes markdown and code, says file names as "session dot py", and expands contractions.
3. Rewrites it in Data's voice with **Claude Haiku** (`claude-haiku-4-5`), through the logged-in Claude Code CLI (`claude -p`), using the prompt in `~/AI-Workspace/data-persona/haiku_system.txt`. This takes about 2.5 s and counts toward your Claude plan's usage, with no API key needed. Thinking is off: with it on, calls took 10–60 s. `DATA_VOICE_CHILD=1` stops that `claude -p` session from triggering this hook itself.
4. Clones the voice with Qwen3-TTS on the resident mlx-audio voice server on `127.0.0.1:8880`. It uses `data_ref.wav` and the settings in `voice_config.json`.
5. Plays the result with `afplay`.

**Fallbacks:** if Haiku fails, or the session runs on a local model (`ANTHROPIC_BASE_URL` set), the local two-pass rewriter on `127.0.0.1:8082` is used instead. If that is down too, the first 3 cleaned sentences are spoken. If the voice server is down, macOS `say` reads the text.

Each new reply stops the previous one's speech by killing its process group.

## Requirements

- macOS on Apple Silicon. The script uses only the Python standard library.
- The local AI stack running (`localai`). It starts the rewriter (`~/models/tools/rewrite.sh`) and the voice server (`~/models/tools/voice.sh`).
- The voice package in `../data_output/cmdr_data_voice/`.

## Setup

Register `speak.sh` as a Stop hook in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "Stop": [
      { "hooks": [ { "type": "command",
                     "command": "/Users/admin/Development/CharacterVoiceCloning/CharacterVoiceCloning/claude-code-hook/speak.sh",
                     "timeout": 10 } ] }
    ]
  }
}
```

## Use

- **On:** `touch ~/.claude/speak-on`
- **Off:** `rm ~/.claude/speak-on`
- **Log:** `~/.claude/speak/speak.log`

## Notes

- **Delay:** about 2.5 s for the Haiku rewrite (about 1 s with the local rewriter) plus about 1.5 s of voice generation for a typical summary. The first voice request after the server starts takes about 5 s while it loads the model.
- **Memory:** the rewriter uses about 3–4 GB and the TTS model about 1 GB in the voice server, alongside the main chat model.
