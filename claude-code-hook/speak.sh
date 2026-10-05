#!/bin/bash
# Claude Code "Stop" hook: speak Claude's last reply as Commander Data.
#
# On/off:  touch ~/.claude/speak-on   |   rm ~/.claude/speak-on
# Log:     ~/.claude/speak/speak.log
#
# All the work happens in speak_pipeline.py, started in the background so this hook
# returns immediately and never holds up Claude Code. Always exits 0.

[ -f "$HOME/.claude/speak-on" ] || exit 0
[ -z "$DATA_VOICE_CHILD" ] || exit 0   # the Haiku rewrite's own claude -p session: don't recurse

DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HOME/.claude/speak"
INPUT="$(cat)"

nohup /usr/bin/env python3 "$DIR/speak_pipeline.py" <<< "$INPUT" \
    >> "$HOME/.claude/speak/speak.log" 2>&1 &

exit 0
