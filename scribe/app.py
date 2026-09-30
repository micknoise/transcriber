"""Local web app: record or upload audio, get a transcript that is auto-corrected. Runs on 127.0.0.1 only."""
import logging
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from . import autocorrect, engine, session

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


class UserError(Exception):
    def __init__(self, msg, code=400):
        super().__init__(msg)
        self.code = code


@app.errorhandler(Exception)
def on_error(e):
    if isinstance(e, UserError):
        return jsonify(error=str(e)), e.code
    app.logger.exception("request failed")
    return jsonify(error=f"{type(e).__name__}: {e}"[:500]), 500


def _to_wav(src: str, dst: str) -> None:
    """Normalise whatever the browser sent to 16 kHz mono wav; gives one clear error if it's unreadable."""
    try:
        r = subprocess.run(["ffmpeg", "-nostdin", "-y", "-i", src, "-ac", "1", "-ar", "16000", dst],
                           capture_output=True, timeout=600)
    except FileNotFoundError:
        raise UserError("ffmpeg isn't installed. Run `brew install ffmpeg` and restart.", 500)
    if r.returncode != 0 or not os.path.getsize(dst):
        raise UserError("Couldn't read that audio: the recording is empty or corrupt. If you were recording, "
                        "the browser likely lost the microphone (quit other apps using the mic, check mic "
                        "permission, or try Chrome). Uploading an audio file also works.", 422)


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/api/model/<path:name>")
def model_status(name):
    return jsonify(engine.model_status(name))


@app.post("/api/model/<path:name>")
def model_load(name):
    return jsonify(engine.load_async(name))


def _session(sid):
    sess = session.get(HOME, sid, _rules)
    if sess is None:
        raise UserError("Unknown session.", 404)
    return sess


@app.post("/api/session")
def session_new():
    o = request.get_json() or {}
    opts = {k: o.get(k) for k in ("model", "language", "fillers", "timestamps", "llm")}
    sess = session.create(HOME, opts, _vocab(_rules()), o.get("ext", "webm"), _rules)
    return jsonify(id=sess.id)


@app.post("/api/session/<sid>/chunk")
def session_chunk(sid):
    try:
        _session(sid).append(int(request.args["seq"]), request.get_data())
    except session.SeqGap as e:
        raise UserError(str(e), 409)
    return jsonify(ok=True)


@app.post("/api/session/<sid>/finish")
def session_finish(sid):
    _session(sid).finish()
    return jsonify(ok=True)


@app.get("/api/session/<sid>")
def session_get(sid):
    return jsonify(_session(sid).snapshot())


@app.get("/api/sessions/unfinished")
def sessions_unfinished():
    return jsonify(session.list_unfinished(HOME))


@app.get("/favicon.ico")
def favicon():
    return "", 204


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
    wav = tmp.name + ".wav"
    try:
        if os.path.getsize(tmp.name) < 1000:
            raise UserError("No audio was received (the recording is empty). Check the microphone and try again.")
        _to_wav(tmp.name, wav)
        segs = engine.transcribe(wav, model=opt.get("model", "large-v3-turbo"),
                                 language=opt.get("language") or None, vocab=_vocab(rules))
    finally:
        for p in (tmp.name, wav):
            if os.path.exists(p):
                os.unlink(p)
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    engine.load_async(os.environ.get("SCRIBE_MODEL", "large-v3-turbo"))  # start downloading immediately
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5055)))


if __name__ == "__main__":
    main()
