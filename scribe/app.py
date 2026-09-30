"""Local web app: record or upload audio, get a transcript that is auto-corrected. Runs on 127.0.0.1 only."""
import os
import tempfile
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from . import autocorrect, engine

HOME = Path(os.environ.get("SCRIBE_HOME", Path.home() / "Transcripts"))
CORRECTIONS = HOME / "corrections.txt"
STATIC = Path(__file__).parent / "static"
DEFAULT_CORRECTIONS = Path(__file__).parent.parent / "corrections.example.txt"

app = Flask(__name__, static_folder=None)


def _rules():
    HOME.mkdir(parents=True, exist_ok=True)
    if not CORRECTIONS.exists() and DEFAULT_CORRECTIONS.exists():
        CORRECTIONS.write_text(DEFAULT_CORRECTIONS.read_text(encoding="utf-8"), encoding="utf-8")
    return autocorrect.load_corrections(CORRECTIONS)


def _vocab(rules):
    """Correct spellings double as a Whisper prompt so they're heard right in the first place."""
    return ", ".join(dict.fromkeys(r for _, r in rules if r))[:600]


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/api/corrections")
def get_corrections():
    _rules()
    return jsonify(text=CORRECTIONS.read_text(encoding="utf-8") if CORRECTIONS.exists() else "")


@app.put("/api/corrections")
def put_corrections():
    HOME.mkdir(parents=True, exist_ok=True)
    CORRECTIONS.write_text(request.get_json()["text"], encoding="utf-8")
    return jsonify(ok=True)


@app.post("/api/transcribe")
def transcribe():
    f = request.files["audio"]
    opt = request.form
    rules = _rules()
    suffix = Path(f.filename or "a.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        f.save(tmp.name)
    try:
        segs = engine.transcribe(tmp.name, model=opt.get("model", "large-v3-turbo"),
                                 language=opt.get("language") or None, vocab=_vocab(rules))
    finally:
        os.unlink(tmp.name)
    stamps = opt.get("timestamps") == "1"
    raw = engine.to_paragraphs(segs, stamps=stamps)
    text = autocorrect.clean(raw, rules, remove_fillers=opt.get("fillers", "1") == "1")
    if opt.get("llm"):
        text = autocorrect.llm_polish(text, opt["llm"], vocab=_vocab(rules))
    name = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    (HOME / f"{name}.txt").write_text(text + "\n", encoding="utf-8")
    (HOME / f"{name}.raw.txt").write_text(raw + "\n", encoding="utf-8")
    return jsonify(raw=raw, text=text, saved=str(HOME / f"{name}.txt"))


def main():
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5055)))


if __name__ == "__main__":
    main()
