#!/usr/bin/env python3
"""
Turn a Claude Code reply into text that reads well in Commander Data's cloned voice.

    echo "$TEXT" | python3 prepare_text.py            # stdin -> stdout
    python3 prepare_text.py --max-sentences 4 < reply.md

Standard library only, so it runs under any python3 without a venv.

What it does:
  - Drops fenced code blocks, tables, URLs and markdown syntax (headings,
    bullets, emphasis, inline-code backticks, link targets).
  - Expands contractions ("don't" -> "do not", "I'm" -> "I am"), because
    Data does not use contractions. Possessives ("Data's") are left alone.
  - Optionally keeps only the first N sentences and/or about N words, so long
    replies stay short (a trailing question is kept).
"""

import argparse
import re
import sys

# Specific forms first; the generic n't / 're / 've / 'll rules come after.
CONTRACTIONS = [
    (r"\bcan't\b", "cannot"),
    (r"\bwon't\b", "will not"),
    (r"\bshan't\b", "shall not"),
    (r"\bain't\b", "is not"),
    (r"\blet's\b", "let us"),
    (r"\bI'm\b", "I am"),
    (r"\b(it|that|there|here|what|who|where|he|she)'s\b", r"\1 is"),
    (r"\b(\w+)n't\b", r"\1 not"),
    (r"\b(\w+)'re\b", r"\1 are"),
    (r"\b(\w+)'ve\b", r"\1 have"),
    (r"\b(\w+)'ll\b", r"\1 will"),
    (r"\b(I|you|he|she|we|they|it|that|there)'d\b", r"\1 would"),
]


def expand_contractions(text):
    text = text.replace("’", "'")
    for pattern, repl in CONTRACTIONS:
        text = re.sub(
            pattern,
            lambda m, r=repl: _match_case(m.group(0), m.expand(r)),
            text,
            flags=re.IGNORECASE,
        )
    return text


def _match_case(original, replacement):
    if original[0].isupper() and replacement[0].islower():
        return replacement[0].upper() + replacement[1:]
    return replacement


def strip_markdown(text):
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)            # fenced code
    text = re.sub(r"^\s*\|.*\|\s*$", " ", text, flags=re.MULTILINE)    # table rows
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)             # [text](url)
    text = re.sub(r"https?://\S+", " ", text)                          # bare URLs
    text = re.sub(r"`([^`]*)`", r"\1", text)                           # inline code
    text = re.sub(r"(?:[\w.~-]+/)+([\w-]+)\.(\w{1,5})(?::\d+)?",       # dir/file.py:42
                  r"\1 dot \2", text)                                  #   -> file dot py
    text = re.sub(r"\b([\w-]+)\.(py|js|ts|tsx|jsx|sh|md|json|yaml|yml|toml|go|rs|rb|java|swift|css|html)(?::\d+)?\b",
                  r"\1 dot \2", text)                                  # file.py -> file dot py
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)  # headings
    text = re.sub(r"^\s*(?:[-*+]|\d+\.)\s+", "", text, flags=re.MULTILINE)  # bullets
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)           # quotes
    text = re.sub(r"(\*\*|__|\*|_)(\S(?:.*?\S)?)\1", r"\2", text)      # emphasis
    text = text.replace("→", " to ").replace("—", ", ").replace("–", " to ")
    # A line without closing punctuation (a bullet or heading) becomes its own sentence.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lines = [ln if re.search(r"[.!?:;,]$", ln) else ln + "." for ln in lines]
    text = re.sub(r"\s+", " ", " ".join(lines))
    return re.sub(r"\s+([,.!?;:])", r"\1", text).strip()


def first_sentences(text, n=None, max_words=None):
    """Keep the first n sentences, stopping early once max_words is reached (always at least one).
    A question that would be cut off replaces the last kept sentence (or follows a lone one), so the
    listener still hears what is being asked."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept, words = [], 0
    for s in sentences[:n]:
        words += len(s.split())
        if kept and max_words and words > max_words:
            break
        kept.append(s)
    if len(kept) == len(sentences):
        return text
    questions = [s for s in sentences[len(kept):] if s.endswith("?")]
    if questions and not any(s.endswith("?") for s in kept):
        kept[-1:] = kept[-1:] + [questions[-1]] if len(kept) == 1 else [questions[-1]]
    return " ".join(kept)


def prepare(text, max_sentences=None, contractions=True, max_words=None):
    text = strip_markdown(text)
    if contractions:
        text = expand_contractions(text)
    if max_sentences or max_words:
        text = first_sentences(text, max_sentences, max_words)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--max-sentences", type=int, help="Keep only the first N sentences")
    parser.add_argument("--max-words", type=int, help="Stop adding sentences once N words are reached")
    parser.add_argument("--keep-contractions", action="store_true", help="Skip contraction expansion")
    args = parser.parse_args()
    print(prepare(sys.stdin.read(), args.max_sentences, not args.keep_contractions, args.max_words))


if __name__ == "__main__":
    main()
