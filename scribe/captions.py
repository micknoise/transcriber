"""Live captions: turn a stream of 16 kHz PCM into rolling caption text with low latency.

The browser sends raw PCM (no container, so no ffmpeg/seeking and latency stays flat for hours). A worker
repeatedly transcribes the audio since the last committed point and publishes it as tentative text. A simple
energy detector finds pauses: text before a pause is committed as final, leading silence is skipped, and
silent gaps are never sent to Whisper (which otherwise invents "Thank you." out of nothing).
"""
import logging
import re
import threading
import time
from pathlib import Path

import numpy as np

from . import autocorrect, engine

log = logging.getLogger(__name__)
SR = 16000
WIN = SR // 10          # energy window: 100 ms
ENDPOINT = 0.6          # seconds of silence that end an utterance
MIN_SPEECH = 0.4        # seconds of voiced audio before bothering Whisper
MAX_TAIL = 10.0         # continuous speech longer than this: commit the segments that are complete
MAX_FORCE = 25.0        # ...and if even that yields nothing, commit everything
PARA_GAP = 3.0          # wall-clock seconds of quiet that start a new caption paragraph
HALLUCINATIONS = {"thank you", "thanks", "thank you for watching", "thanks for watching", "you", "bye",
                  "bye bye", "please subscribe"}


class SeqGap(Exception):
    pass


def parse_terms(text: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Talk-specific vocabulary, one per line: `Kubernetes` (fixes casing) or `wrong => right`."""
    rules, vocab = [], []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if "=>" in line:
            a, b = (x.strip() for x in line.split("=>", 1))
            if a and b:
                rules.append((a, b))
                vocab.append(b)
        else:
            rules.append((line, line))
            vocab.append(line)
    return rules, vocab


class CaptionStream:
    def __init__(self, opts: dict, rules_fn, out_path: Path | None = None):
        self.opts, self.rules_fn, self.out_path = opts, rules_fn, out_path
        self.term_rules, terms = parse_terms(opts.get("terms", ""))
        self.vocab = ", ".join(dict.fromkeys(terms + autocorrect.vocab_terms(rules_fn())))[:400]
        self.lock = threading.RLock()
        self.buf = bytearray()
        self.base = 0        # absolute sample index of buf[0]
        self.cursor = 0      # absolute sample index where uncommitted audio starts
        self.seq = 0
        self.paras: list[dict] = []
        self.partial = ""
        self.version = 0
        self.noise = None
        self.running = True
        self.error = None
        self.last_infer = 0.0
        self._last_len = 0
        self.event = threading.Event()
        self.thread = None

    # ---- input -------------------------------------------------------------------------------------
    def append(self, seq: int, data: bytes) -> None:
        with self.lock:
            if seq < self.seq:
                return  # client retry of a chunk we already have
            if seq > self.seq:
                raise SeqGap(f"expected chunk {self.seq}, got {seq}")
            self.buf += data[: len(data) // 2 * 2]
            self.seq += 1
        self.event.set()

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.event.set()
        if self.thread:
            self.thread.join(timeout=60)

    # ---- worker ------------------------------------------------------------------------------------
    def _run(self):
        while self.running:
            try:
                worked = self.step()
                self.error = None
            except Exception as e:
                log.exception("caption step failed")
                self.error, worked = f"{type(e).__name__}: {e}"[:300], False
                time.sleep(1)
            if not worked:
                self.event.wait(0.25)
                self.event.clear()
        try:  # flush whatever was still being spoken when the presenter pressed Stop
            self.step(force_end=True)
        except Exception:
            log.exception("caption flush failed")

    def _advance(self, to_abs: int):
        with self.lock:
            self.cursor = max(self.cursor, to_abs)
            drop = (self.cursor - SR) - self.base  # keep 1 s of context before the cursor
            if drop > 0:
                del self.buf[: 2 * drop]
                self.base += drop
        self._last_len = 0

    def _set_partial(self, text: str):
        with self.lock:
            if text != self.partial:
                self.partial = text
                self.version += 1

    def step(self, force_end: bool = False) -> bool:
        """One pass over the uncommitted audio. Returns True if it did work (so the caller shouldn't idle)."""
        with self.lock:
            raw = bytes(self.buf[2 * (self.cursor - self.base):])
            end_abs = self.base + len(self.buf) // 2
        x = np.frombuffer(raw[: len(raw) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768
        n = len(x) // WIN
        if n < 3:
            return False
        w = np.sqrt((x[: n * WIN].reshape(n, WIN) ** 2).mean(axis=1))
        low = float(np.percentile(w, 20))
        # noise floor: drops instantly, rises slowly (so a long stretch of speech doesn't raise it much)
        self.noise = low if self.noise is None else min(self.noise * 1.01, low)
        self.noise = min(self.noise, 0.02)
        voiced = w > min(max(3 * self.noise, 0.006), 0.05)
        dur = len(x) / SR

        if not voiced.any():  # silence: don't transcribe it, and don't let it pile up
            if dur > 0.6:
                self._advance(end_abs - int(0.3 * SR))
            self._set_partial("")
            return False
        first = int(voiced.argmax())
        if first * WIN > SR:  # long leading silence: skip to just before the speech
            self._advance(self.cursor + first * WIN - int(0.3 * SR))
            return True
        last_voiced = n - 1 - int(voiced[::-1].argmax())
        ended = force_end or (n - 1 - last_voiced) * WIN >= ENDPOINT * SR
        vdur = voiced.sum() * WIN / SR
        if vdur < MIN_SPEECH:
            if ended:  # a click or cough, not speech
                self._advance(end_abs - int(0.3 * SR))
                self._set_partial("")
                return True
            return False
        if not ended and len(x) - self._last_len < 0.2 * SR:
            return False  # nothing new worth re-transcribing yet

        t0 = time.time()
        segs = engine.transcribe(x, model=self.opts.get("model", "small.en"),
                                 language=self.opts.get("language") or None, vocab=self._prompt())
        self.last_infer = time.time() - t0
        segs = [s for s in segs if s.text.strip()]
        if segs and voiced.mean() < 0.25 and re.sub(r"[^a-z ]", "", " ".join(s.text for s in segs).lower()).strip() in HALLUCINATIONS:
            segs = []

        if ended or dur >= MAX_FORCE:
            commit, rest, new_cursor = segs, [], end_abs
        elif dur >= MAX_TAIL:
            commit = [s for s in segs if s.end <= dur - 1.5]
            rest = [s for s in segs if s not in commit]
            new_cursor = self.cursor + int(commit[-1].end * SR) if commit else None
        else:
            commit, rest, new_cursor = [], segs, None
        log.info("captions: %.1fs tail -> %d segs (%d final) in %.2fs", dur, len(segs), len(commit), self.last_infer)

        if commit:
            self._commit(" ".join(s.text for s in commit))
        self._set_partial(self._clean(" ".join(s.text for s in rest)))
        if new_cursor is not None:
            self._advance(new_cursor)
        else:
            self._last_len = len(x)
        return True

    # ---- text --------------------------------------------------------------------------------------
    def _prompt(self) -> str:
        ctx = " ".join(p["text"] for p in self.paras[-2:])[-(500 - len(self.vocab)):] if self.paras else ""
        return f"{ctx} {self.vocab}".strip()

    def _clean(self, text: str) -> str:
        if not text.strip():
            return ""
        rules = self.term_rules + self.rules_fn()
        return autocorrect.clean(text, rules, remove_fillers=bool(self.opts.get("fillers", True)))

    def _commit(self, text: str):
        text = self._clean(text)
        if not text:
            return
        now = time.time()
        with self.lock:
            if self.paras and now - self.paras[-1]["t"] < PARA_GAP:
                self.paras[-1]["text"] += " " + text
                self.paras[-1]["t"] = now
            else:
                self.paras.append({"t": now, "text": text})
            self.version += 1
        if self.out_path:
            with open(self.out_path, "a", encoding="utf-8") as f:  # append as we go: nothing lost on a crash
                f.write(text + "\n")

    def snapshot(self, last: int = 40) -> dict:
        with self.lock:
            paras = self.paras[-last:]
            return dict(v=self.version, running=self.running, first=len(self.paras) - len(paras),
                        paras=[p["text"] for p in paras], partial=self.partial, infer=round(self.last_infer, 2), error=self.error)
