"""Strip tool-call shaped text leaks out of model transcripts.

The native audio Gemini Live models sometimes call a tool AND ALSO narrate
the call as prose, producing transcript chunks like:

    response:setMood{emotion:excited,level:7,reason:User said hello!...

The tool itself still fires correctly, but the leaked text pollutes the
chatbox, the conversation log, and (worst of all) future recallMemories
output, which then feeds the next session and reinforces the pattern.

This module is a best-effort regex scrubber. It runs at turn boundaries
on the fully accumulated transcript, not mid-stream, so it doesn't have to
worry about partial matches across chunk boundaries.
"""

from __future__ import annotations

import re

# response:funcName{ ... }   or just funcName({...})
# the brace block has to look structurish to avoid eating legit prose
_PAREN_CALL = re.compile(
    r"\b(?:response\s*[:\-]\s*)?[a-zA-Z_]\w{2,}\s*\(\s*\{[^{}]{2,800}?\}\s*\)",
    re.DOTALL,
)
_RESPONSE_PREFIX_BRACE = re.compile(
    r"\bresponse\s*[:\-]\s*[a-zA-Z_]\w{2,}\s*\{[^{}]{2,800}?\}",
    re.DOTALL,
)
# bare camelCase or snake_case identifier followed directly by a brace
# block that contains at least two key:value-looking entries. The two-pair
# requirement keeps us from nuking legit sentences that happen to contain
# a single "word{thing}" shape.
_BARE_FUNC_BRACE = re.compile(
    r"\b[a-zA-Z_]\w{2,}\s*\{(?:\s*[a-zA-Z_]\w*\s*:[^{}\n]{1,200},\s*){1,}[^{}\n]{1,200}\}",
    re.DOTALL,
)

# Worst case from observed real-world leaks: the model emits
#   "response:setMood{emotion:excited,level:7,reason:User said hello! ..."
# and NEVER closes the brace. The rest of the buffer is the function args
# bleeding into the spoken reply with no separator. We can't reliably
# split arg-text from reply-text, so when we see this prefix at the very
# start of the transcript we nuke from the prefix up to the first
# clean sentence boundary (capital letter starting a new word right after
# lowercase, e.g. "danceOh") and fall back to nuking the whole buffer if
# we can't find one. Better to log nothing than to log poisoned text that
# recallMemories will feed back into the next session.
_LEAK_PREFIX_UNCLOSED = re.compile(
    r"^\s*(?:response\s*[:\-]\s*)?[a-zA-Z_]\w{2,}\s*\{",
    re.DOTALL,
)
_SENTENCE_RESTART = re.compile(r"(?<=[a-z])([A-Z][a-z])")


def strip_tool_call_leaks(text: str) -> tuple[str, bool]:
    """Returns (cleaned_text, was_modified).

    Empty or None input is returned unchanged.
    """
    if not text:
        return text, False
    original = text
    for pat in (_PAREN_CALL, _RESPONSE_PREFIX_BRACE, _BARE_FUNC_BRACE):
        text = pat.sub("", text)

    # unclosed leak prefix at start of buffer (no `}` ever shows up). Look for
    # the first sentence restart AFTER the open brace, cut to there. If we
    # can't find one, drop the whole buffer - better to log nothing than
    # poison recall with leaked function args.
    m = _LEAK_PREFIX_UNCLOSED.match(text)
    if m:
        after_brace = text[m.end():]
        restart = _SENTENCE_RESTART.search(after_brace)
        if restart:
            text = after_brace[restart.start(1):]
        else:
            text = ""

    # collapse the extra whitespace the deletions leave behind
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text, text != original
