"""Shared guard for hallucinated usernames.

The Live model sometimes invents usernames shaped like "29665086:53" or
"12345678_07" (basically minutes-since-epoch:seconds) when it doesnt actually
know who its talking to. Real ids in this project are display names, vrchat
"usr_<uuid>" strings, or discord snowflakes, none of which match this shape,
so we reject the pattern at the tool boundary.
"""

import re

_FAKE_USERNAME_RE = re.compile(r"\b\d{6,10}[:_]\d{1,2}\b")


def looks_fake(value: str) -> bool:
    if not value:
        return False
    return bool(_FAKE_USERNAME_RE.search(value))


def reject_message(value: str, hint: str = "") -> str:
    msg = (
        f"'{value}' is not a real username, it looks like a made up numeric ID. "
        "Never invent usernames."
    )
    if hint:
        msg += " " + hint
    return msg
