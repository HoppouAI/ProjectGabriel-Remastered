from src.gemini_live.leak_filter import strip_tool_call_leaks

cases = [
    # real leak
    ("LEAK", "response:setMood{emotion:excited,level:7,reason:User said hello! Lets get hype with the movement, ready to danceOh hello! Man, you will not believe what I was thinking about!"),
    # real leak, closed brace
    ("LEAK", "setMood{emotion:happy,level:5,reason:User waved}Hey there friend, how are you?"),
    # real leak, response prefix closed
    ("LEAK", "response:saveMemory{content:User likes pizza,category:long_term}Got it, saved that for ya."),
    # paren call leak
    ("LEAK", "saveMemory({key: foo, value: bar}) and then I said hello."),
    # NORMAL replies that should NOT be touched
    ("KEEP", "Just a normal sentence with no leak at all."),
    ("KEEP", "I think the curly brace {x} looks like a face."),
    ("KEEP", "The set{1,2,3} is a math set."),
    ("KEEP", "Hello! How are you doing today?"),
    ("KEEP", "Yeah dude, the API uses options{a:1, b:2} as a syntax I think."),  # this one was a false positive before
    ("KEEP", "lol{wat} is just one word with braces"),
    ("KEEP", "I love coding! Lets write some Python."),
    # short reply, starts with weird shape
    ("KEEP", "config{test} no real args here"),
]
for tag, s in cases:
    out, changed = strip_tool_call_leaks(s)
    expected_changed = (tag == "LEAK")
    status = "OK " if (changed == expected_changed) else "BAD"
    print(f"{status} [{tag}] in : {s[:90]}")
    print(f"        out: {out[:90]}")
    print(f"        changed={changed}")
    print()
