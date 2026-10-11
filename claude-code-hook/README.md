# Claude Code Voice Hook

Speaks a short Commander Data version of each Claude Code reply. Uses the [Stop hook](https://code.claude.com/docs/en/hooks), which fires after each response.

## How it works

`speak.sh` checks the on/off flag, starts `speak_pipeline.py` in the background and exits 0 at once, so it never holds up Claude Code. The pipeline:

1. Takes the final reply: the hook's `last_assistant_message`, or else the assistant text after the last user message in the transcript.
2. Rewrites it in Data's voice with **Claude Haiku** (`claude-haiku-4-5`) through the logged-in Claude Code CLI (`claude -p`), using `rewrite.py` and the prompt in `haiku_system.txt`. This takes about 2.5 s and counts toward your Claude plan's usage. No API key is needed. Thinking is off: with it on, calls took 10–60 s. `DATA_VOICE_CHILD=1` stops that `claude -p` session from triggering this hook itself. If the reply is only pasted documentation or tool output, Haiku answers `SKIP` and nothing is spoken.
3. Cleans the result with `prepare_text.py`: removes markdown and code, says file names as "session dot py", expands contractions, and caps it at 3 sentences and about 60 words. A question at the end is always kept.
4. Clones the voice with Qwen3-TTS on the resident mlx-audio voice server (`127.0.0.1:8880`), using the reference clip and the settings in `voice_config.json`. It generates one sentence at a time and plays each with `afplay` while the next is generated, so speech starts about 0.5–1 s after the rewrite.

Each new reply stops the previous one's speech by killing its process group.

**Fallbacks:**
- If Haiku fails, or the session runs on a local model (`ANTHROPIC_BASE_URL` set), the local two-pass rewriter on `127.0.0.1:8082` is used instead. If that is down too, the first sentences of the cleaned reply are spoken.
- If the voice server is down (for example, after `localai stop` or a reboot), the pipeline starts it with `DATA_VOICE_SERVER_CMD` in its own process session and waits up to 90 s. Later replies are fast again.
- If the server still fails, the pipeline generates the audio with `mlx_audio.tts.generate` directly (about 7–20 s), still in Data's voice. macOS `say` is used only if both fail.

## Requirements

- macOS on Apple Silicon. The hook uses only the Python standard library and runs under the system `python3`.
- The voice reference: a folder holding `data_ref.wav` and its exact transcript `data_ref.txt`. The default is `../data_output/cmdr_data_voice/`, which is git-ignored because the audio is copyrighted. See the main README for how to make one.
- An mlx-audio voice server on `:8880`, or at least `mlx_audio.tts.generate` (`uv tool install mlx-audio`). Any `mlx_audio.server` works; set `DATA_VOICE_SERVER_CMD` to a script that starts it so the hook can bring it back up.
- Optional: the `claude` CLI, logged in, for the Haiku rewrite. Without it the local rewriter (or plain first sentences) is used.
- Optional: the local rewriter on `:8082` (Qwen3-4B-Instruct with the Data LoRA, built in a separate private workspace).

## Setup

1. Register `speak.sh` as a Stop hook in `~/.claude/settings.json`, with this repo's path (see `settings-snippet.json`):

   ```json
   {
     "hooks": {
       "Stop": [
         { "hooks": [ { "type": "command",
                        "command": "/path/to/CharacterVoiceCloning/claude-code-hook/speak.sh",
                        "timeout": 10 } ] }
       ]
     }
   }
   ```

2. Turn it on: `touch ~/.claude/speak-on`
3. Check everything: `python3 claude-code-hook/speak_pipeline.py --check`
4. Hear a test: `echo "I have finished the task." | python3 claude-code-hook/speak_pipeline.py --test`

## Settings

All optional, as environment variables. To set them for Claude Code, put them in the `env` block of `~/.claude/settings.json`.

| Variable | Default | What it sets |
|---|---|---|
| `DATA_VOICE_DIR` | `../data_output/cmdr_data_voice` | Folder with the reference clip and transcript |
| `DATA_VOICE_TTS_URL` | `http://127.0.0.1:8880` | mlx-audio voice server |
| `DATA_VOICE_SERVER_CMD` | `~/models/tools/voice.sh` | Script that starts the voice server |
| `DATA_VOICE_MAX_SENTENCES` | `3` | Longest spoken reply, in sentences |
| `DATA_VOICE_MAX_WORDS` | `60` | Longest spoken reply, in words (about 20 s) |
| `DATA_VOICE_HAIKU_MODEL` | `claude-haiku-4-5` | Model for the rewrite |
| `DATA_VOICE_EXAMPLES` | `~/AI-Workspace/data-persona/haiku_examples.txt` | Real Data lines added to the Haiku prompt as style examples (kept private: copyrighted dialogue) |
| `DATA_VOICE_REWRITER_URL` | `http://127.0.0.1:8082` | Local fallback rewriter |

`claude-haiku-5-5` also works, but in testing (October 2026) it was about 0.8 s slower per reply and no more accurate.

## Use

- **On:** `touch ~/.claude/speak-on`
- **Off:** `rm ~/.claude/speak-on`
- **Log:** `~/.claude/speak/speak.log`. It shows each rewrite, and "speaking (N parts)" when playback starts.

## Notes

- **Delay:** about 2.5 s for the Haiku rewrite (about 1 s with the local rewriter), then about 0.5–1 s until the first sentence plays. The first voice request after the server starts takes about 5 s while it loads the model.
- **Memory:** the TTS model uses about 1 GB in the voice server, and the local rewriter about 3–4 GB.
