"""Transcript clean-up: user dictionary, filler/stutter removal, tidy-up, optional local-LLM polish."""
import json
import re
import urllib.request
from pathlib import Path

FILLERS = re.compile(r"\b(?:u+m+|u+h+|e+r+m*|a+h+m*|h+m+)\b[,.]?\s*", re.I)
STUTTER = re.compile(r"\b(\w+)(\s+\1\b)+", re.I)


def load_corrections(path: Path) -> list[tuple[str, str]]:
    """Parse `wrong => right` lines ('#' comments allowed)."""
    rules = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if "=>" in line:
                a, b = (s.strip() for s in line.split("=>", 1))
                if a:
                    rules.append((a, b))
    return rules


def vocab_terms(rules: list[tuple[str, str]]) -> list[str]:
    """Right-hand sides worth hinting to Whisper: names/jargon (contain a capital or digit), not phrases like 'going to'."""
    return list(dict.fromkeys(r for _, r in rules if r and re.search(r"[A-Z0-9]", r)))


def apply_rules(text: str, rules: list[tuple[str, str]]) -> str:
    for wrong, right in sorted(rules, key=lambda r: -len(r[0])):
        text = re.sub(rf"(?<!\w){re.escape(wrong)}(?!\w)", lambda _: right, text, flags=re.I)
    return text


def tidy(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([,.!?;:])(?=[A-Za-z])", r"\1 ", text)
    text = re.sub(r"\bi\b(?=[\s',.!?]|$)", "I", text)  # lone "i" -> "I"

    def cap(m):
        return m.group(1) + m.group(2).upper()

    text = re.sub(r"((?:^|[.!?]\s+|\n)\s*)([a-z])", cap, text)
    return "\n".join(l.strip() for l in text.split("\n")).strip()


def clean(text: str, rules, remove_fillers=True, remove_stutters=True) -> str:
    if remove_fillers:
        text = FILLERS.sub("", text)
    if remove_stutters:
        text = STUTTER.sub(r"\1", text)
    return tidy(apply_rules(text, rules))


def llm_polish(text: str, model: str, host: str = "http://localhost:11434", vocab: str = "") -> str:
    """Fix punctuation/mis-heard words with a local Ollama model, paragraph by paragraph.
    Returns the input unchanged for any paragraph the model fails on or rewrites too heavily."""
    system = (
        "You proofread speech-to-text transcripts. Fix spelling, punctuation, capitalisation and "
        "obvious mis-heard words. Do NOT paraphrase, summarise, add or remove content. "
        "Reply with only the corrected text." + (f" Likely vocabulary: {vocab}." if vocab else "")
    )
    out = []
    for para in text.split("\n"):
        if len(para.strip()) < 3:
            out.append(para)
            continue
        body = json.dumps({"model": model, "stream": False, "options": {"temperature": 0},
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": para}]}).encode()
        try:
            req = urllib.request.Request(f"{host}/api/chat", body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                new = json.load(r)["message"]["content"].strip()
        except Exception:
            out.append(para)
            continue
        ratio = len(new) / max(len(para), 1)
        out.append(new if new and 0.7 <= ratio <= 1.4 else para)
    return "\n".join(out)
