"""Text clean-up: user dictionary, filler/stutter removal and tidy-up."""
import re  # regular expressions for finding text patterns

# Filler sounds like "um", "uh", "er", "ah", "hmm" (plus a trailing comma or full stop)
FILLERS = re.compile(r"\b(?:u+m+|u+h+|e+r+m*|a+h+m*|h+m+)\b[,.]?\s*", re.I)
# A word repeated straight away, e.g. "I I think"
STUTTER = re.compile(r"\b(\w+)(\s+\1\b)+", re.I)


def load_corrections(path):
    """Read `wrong => right` lines from a file ('#' starts a comment)."""
    rules = []  # the list of (wrong, right) pairs
    if path.exists():  # no file means no rules
        for line in path.read_text(encoding="utf-8").splitlines():  # go through each line
            line = line.split("#", 1)[0].strip()  # drop comments and spaces
            if "=>" in line:  # only lines with an arrow are rules
                a, b = (s.strip() for s in line.split("=>", 1))  # split into wrong and right
                if a:  # the wrong part must not be empty
                    rules.append((a, b))  # keep the rule
    return rules  # hand back all the rules


def vocab_terms(rules):
    """Right-hand sides worth hinting to Whisper: names/jargon (a capital or digit), not phrases like 'going to'."""
    return list(dict.fromkeys(r for _, r in rules if r and re.search(r"[A-Z0-9]", r)))  # unique, in order


def apply_rules(text, rules):
    """Replace each wrong word or phrase with the right one (ignoring case)."""
    for wrong, right in sorted(rules, key=lambda r: -len(r[0])):  # longest phrases first
        text = re.sub(rf"(?<!\w){re.escape(wrong)}(?!\w)", lambda _: right, text, flags=re.I)  # whole words only
    return text  # the corrected text


def tidy(text):
    """Fix spacing, punctuation and capital letters."""
    text = re.sub(r"[ \t]+", " ", text)  # collapse repeated spaces
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)  # no space before punctuation
    text = re.sub(r"([,.!?;:])(?=[A-Za-z])", r"\1 ", text)  # a space after punctuation
    text = re.sub(r"\bi\b(?=[\s',.!?]|$)", "I", text)  # a lone "i" becomes "I"
    # Capitalise the first letter of the text and of each sentence
    text = re.sub(r"((?:^|[.!?]\s+|\n)\s*)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return "\n".join(l.strip() for l in text.split("\n")).strip()  # trim every line


def clean(text, rules, remove_fillers=True):
    """Full clean-up: fillers, stutters, dictionary, then tidy."""
    if remove_fillers:  # the user can turn this off
        text = FILLERS.sub("", text)  # remove "um", "uh" ...
    text = STUTTER.sub(r"\1", text)  # keep one copy of repeated words
    return tidy(apply_rules(text, rules))  # apply the dictionary and tidy up
