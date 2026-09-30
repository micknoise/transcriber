"""Live recording sessions.

The browser streams audio chunks; each is appended to sessions/<id>/audio.<ext> on disk (so a crash loses
seconds, not the meeting). A background worker transcribes the not-yet-committed tail every few seconds,
committing only segments that end at a pause, so words aren't cut in half. State lives on disk, so an
interrupted session can be finished later.
"""
import json
import logging
import re
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from . import autocorrect, engine

log = logging.getLogger(__name__)
MIN_TAIL = 20.0   # seconds of new audio to accumulate before transcribing
TICK = 5.0        # worker poll interval
IDLE_EXIT = 600   # worker gives up after this long without new audio (session stays recoverable)
ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}_[0-9a-f]{4}$")

_registry: dict[str, "Session"] = {}
_reg_lock = threading.Lock()


class SeqGap(Exception):
    pass


def extract(audio: Path, start: float, dst: Path) -> None:
    """Decode audio[start:] to 16 kHz mono wav. Truncated/growing files are fine; we judge by output size."""
    dst.unlink(missing_ok=True)
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-ss", f"{start:.3f}", "-i", str(audio),
                    "-ac", "1", "-ar", "16000", str(dst)], capture_output=True, timeout=900)


def valid_id(sid: str) -> bool:
    return bool(ID_RE.match(sid))


def create(home: Path, opts: dict, vocab: str, ext: str, rules_fn) -> "Session":
    sid = datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
    d = home / "sessions" / sid
    d.mkdir(parents=True)
    meta = dict(opts=opts, vocab=vocab, ext=re.sub(r"\W", "", ext) or "webm", seq=0, cursor=0.0,
                finish=False, done=False, saved=None, created=time.time())
    (d / "meta.json").write_text(json.dumps(meta))
    return get(home, sid, rules_fn)


def get(home: Path, sid: str, rules_fn) -> "Session | None":
    if not valid_id(sid) or not (home / "sessions" / sid / "meta.json").exists():
        return None
    key = f"{home}/{sid}"
    with _reg_lock:
        if key not in _registry:
            _registry[key] = Session(home, sid, rules_fn)
        return _registry[key]


def list_unfinished(home: Path, quiet_for: float = 60.0) -> list[dict]:
    """Sessions never finished whose audio hasn't grown recently (i.e. not actively recording)."""
    out = []
    for d in sorted((home / "sessions").glob("*/meta.json")):
        try:
            m = json.loads(d.read_text())
        except ValueError:
            continue
        audio = d.parent / f"audio.{m['ext']}"
        if m["done"] or not audio.exists() or time.time() - audio.stat().st_mtime < quiet_for:
            continue
        out.append(dict(id=d.parent.name, created=m["created"], bytes=audio.stat().st_size))
    return out


class Session:
    def __init__(self, home: Path, sid: str, rules_fn):
        self.home, self.id, self.rules_fn = home, sid, rules_fn
        self.dir = home / "sessions" / sid
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.audio = self.dir / f"audio.{self.meta['ext']}"
        seg = self.dir / "segments.json"
        self.segments = [engine.Segment(*s) for s in json.loads(seg.read_text())] if seg.exists() else []
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.thread = None
        self.error = None
        self._live = (-1, "")

    # ---- persistence -------------------------------------------------------------------------------
    def _save_meta(self):
        tmp = self.dir / "meta.json.tmp"
        tmp.write_text(json.dumps(self.meta))
        tmp.replace(self.dir / "meta.json")

    def _save_segments(self):
        (self.dir / "segments.json").write_text(json.dumps([[s.start, s.end, s.text] for s in self.segments]))

    # ---- input -------------------------------------------------------------------------------------
    def append(self, seq: int, data: bytes) -> None:
        """Chunks must arrive in order; a repeated (already stored) seq is ignored so client retries are safe."""
        with self.lock:
            if seq < self.meta["seq"]:
                return
            if seq > self.meta["seq"]:
                raise SeqGap(f"expected chunk {self.meta['seq']}, got {seq}")
            with open(self.audio, "ab") as f:
                f.write(data)
            self.meta["seq"] += 1
            self._save_meta()
        self.start()

    def finish(self):
        with self.lock:
            self.meta["finish"] = True
            self._save_meta()
        self.start()
        self.wake.set()

    def start(self):
        with self.lock:
            if self.meta["done"] or (self.thread and self.thread.is_alive()):
                return
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    # ---- worker ------------------------------------------------------------------------------------
    def _run(self):
        last_size, idle_since = -1, time.time()
        while True:
            final = self.meta["finish"]
            size = self.audio.stat().st_size if self.audio.exists() else 0
            if final or size != last_size:
                try:
                    self._step(final)
                    self.error, last_size = None, size
                except Exception as e:
                    log.exception("session %s step failed", self.id)
                    self.error, last_size = f"{type(e).__name__}: {e}"[:300], -1
                idle_since = time.time()
            if final:
                try:
                    self._finalize()
                except Exception as e:
                    log.exception("session %s finalize failed", self.id)
                    self.error = f"{type(e).__name__}: {e}"[:300]
                return
            if time.time() - idle_since > IDLE_EXIT:
                return
            self.wake.wait(TICK)
            self.wake.clear()

    def _step(self, final: bool):
        if not self.audio.exists() or self.audio.stat().st_size == 0:
            return
        start = self.meta["cursor"]
        wav = self.dir / "tail.wav"
        extract(self.audio, start, wav)
        dur = max((wav.stat().st_size if wav.exists() else 0) - 44, 0) / 32000
        if dur < (0.5 if final else MIN_TAIL):
            return
        opts = self.meta["opts"]
        vocab = self.meta["vocab"]
        ctx = " ".join(s.text for s in self.segments)[-(500 - len(vocab)):] if self.segments else ""
        segs = engine.transcribe(str(wav), model=opts.get("model", "large-v3-turbo"),
                                 language=opts.get("language") or None, vocab=f"{ctx} {vocab}".strip())
        segs = [s for s in segs if s.text.strip()]
        # Unless finishing, hold back a last segment that touches the end of the audio: it may be cut off.
        if not final and segs and segs[-1].end > dur - 2.0:
            segs = segs[:-1]
        with self.lock:
            if segs:
                self.segments += [engine.Segment(s.start + start, s.end + start, s.text) for s in segs]
                self.meta["cursor"] = self.segments[-1].end
                self._save_segments()
            elif dur > 40:  # long silence: skip ahead rather than re-transcribing it every tick
                self.meta["cursor"] = start + dur - 2.0
            self._save_meta()

    def _finalize(self):
        opts, rules = self.meta["opts"], self.rules_fn()
        raw = engine.to_paragraphs(self.segments, stamps=bool(opts.get("timestamps")))
        text = autocorrect.clean(raw, rules, remove_fillers=bool(opts.get("fillers", True)))
        if opts.get("llm"):
            text = autocorrect.llm_polish(text, opts["llm"], vocab=self.meta["vocab"])
        (self.home / f"{self.id}.txt").write_text(text + "\n", encoding="utf-8")
        (self.home / f"{self.id}.raw.txt").write_text(raw + "\n", encoding="utf-8")
        with self.lock:
            self.meta.update(done=True, saved=str(self.home / f"{self.id}.txt"))
            self._save_meta()

    # ---- output ------------------------------------------------------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
            m, segs = self.meta, list(self.segments)
            status = "done" if m["done"] else "finalizing" if m["finish"] else "recording"
            out = dict(status=status, error=self.error, chunks=m["seq"], transcribed_seconds=m["cursor"])
        if status == "done":
            out.update(text=(self.home / f"{self.id}.txt").read_text(encoding="utf-8").rstrip("\n"),
                       raw=(self.home / f"{self.id}.raw.txt").read_text(encoding="utf-8").rstrip("\n"),
                       saved=m["saved"])
        else:
            if self._live[0] != len(segs):
                raw = engine.to_paragraphs(segs, stamps=bool(m["opts"].get("timestamps")))
                self._live = (len(segs), autocorrect.clean(raw, self.rules_fn(),
                                                           remove_fillers=bool(m["opts"].get("fillers", True))))
            out["text"] = self._live[1]
        return out
