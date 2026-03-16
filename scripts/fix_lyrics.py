import re
import sys
from difflib import SequenceMatcher


def parse_srt(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    blocks = content.split("\n\n")
    entries = []

    for block in blocks:
        lines = block.split("\n")

        if len(lines) >= 3:
            index = lines[0]
            start, end = lines[1].split(" --> ")
            word = lines[2].strip()

            entries.append({
                "index": index,
                "start": start,
                "end": end,
                "word": word
            })

    return entries


def tokenize(text):
    return re.findall(r"\S+", text)


def load_lyrics(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    words = tokenize(text)

    return lines, words


def align_words(bad_words, correct_words):

    matcher = SequenceMatcher(None, bad_words, correct_words)

    fixed = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():

        if tag == "equal":
            fixed.extend(correct_words[j1:j2])

        elif tag == "replace":
            fixed.extend(correct_words[j1:j2])

        elif tag == "insert":
            fixed.extend(correct_words[j1:j2])

        elif tag == "delete":
            continue

    return fixed


def apply_fixed_words(entries, fixed_words):

    for i in range(min(len(entries), len(fixed_words))):
        entries[i]["word"] = fixed_words[i]

    return entries


def build_line_srt(entries, lyric_lines):

    idx = 0
    out = []
    counter = 1

    for line in lyric_lines:

        words = tokenize(line)
        wc = len(words)

        if idx >= len(entries):
            break

        start = entries[idx]["start"]
        end = entries[min(idx + wc - 1, len(entries)-1)]["end"]

        block = f"{counter}\n{start} --> {end}\n{line}\n"

        out.append(block)

        idx += wc
        counter += 1

    return "\n".join(out)


def main():

    if len(sys.argv) != 4:
        print("Usage:")
        print("python fix_lyrics_from_srt.py bad.srt lyrics.txt fixed.srt")
        return

    input_srt = sys.argv[1]
    lyrics_file = sys.argv[2]
    output_srt = sys.argv[3]

    entries = parse_srt(input_srt)

    bad_words = [e["word"] for e in entries]

    lyric_lines, correct_words = load_lyrics(lyrics_file)

    fixed_words = align_words(bad_words, correct_words)

    entries = apply_fixed_words(entries, fixed_words)

    new_srt = build_line_srt(entries, lyric_lines)

    with open(output_srt, "w", encoding="utf-8") as f:
        f.write(new_srt)

    print("Finished!")
    print("Fixed subtitles saved to:", output_srt)


if __name__ == "__main__":
    main()