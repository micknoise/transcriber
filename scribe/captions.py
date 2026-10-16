"""Live captions: turns a stream of 16 kHz audio into rolling caption text.

The browser sends raw audio. A worker thread keeps transcribing the audio since the last
finished sentence and shows it as tentative text. A pause in speech makes the text final.
"""
import logging  # for writing log messages
import re  # for simple text matching
import threading  # for the background worker
import time  # for timing and pauses
from pathlib import Path  # for file paths

import numpy as np  # for audio maths

from . import autocorrect, engine  # our text cleaner and speech-to-text engine

log = logging.getLogger(__name__)  # logger for this file
SR = 16000  # audio samples per second
WIN = SR // 10  # size of one loudness window: 100 ms
ENDPOINT = 0.6  # seconds of quiet that end a sentence
MIN_SPEECH = 0.4  # seconds of speech needed before we bother transcribing
MAX_TAIL = 10.0  # after this much speech, finish the completed segments
MAX_FORCE = 25.0  # after this much speech, finish everything
PARA_GAP = 3.0  # seconds of quiet that start a new paragraph
# phrases Whisper invents out of silence
HALLUCINATIONS = {"thank you", "thanks", "thank you for watching", "thanks for watching", "you", "bye",
                  "bye bye", "please subscribe"}


class SeqGap(Exception):
    """Raised when an audio chunk arrives out of order."""


def parse_terms(text):
    """Read special words, one per line: `Word` or `wrong => right`. Returns (rules, vocab)."""
    rules, vocab = [], []  # replacement rules and vocabulary words
    for line in (text or "").splitlines():  # go through each line
        line = line.strip()  # remove spaces around the line
        if not line:  # skip empty lines
            continue
        if "=>" in line:  # a "wrong => right" line
            a, b = (x.strip() for x in line.split("=>", 1))  # split into wrong and right
            if a and b:  # only use it if both parts exist
                rules.append((a, b))  # remember the replacement
                vocab.append(b)  # remember the right word as vocabulary
        else:  # a plain word
            rules.append((line, line))  # fixes its capitalisation
            vocab.append(line)  # remember it as vocabulary
    return rules, vocab


class CaptionStream:
    """One live captioning session."""

    def __init__(self, opts, rules_fn, out_path=None):
        self.opts, self.rules_fn, self.out_path = opts, rules_fn, out_path  # save the settings
        self.term_rules, terms = parse_terms(opts.get("terms", ""))  # rules from this talk's terms
        # vocabulary hint for Whisper, without duplicates, limited to 400 characters
        self.vocab = ", ".join(dict.fromkeys(terms + autocorrect.vocab_terms(rules_fn())))[:400]
        self.lock = threading.RLock()  # protects data shared between threads
        self.buf = bytearray()  # raw audio received so far
        self.base = 0  # sample number of the first byte pair in buf
        self.cursor = 0  # sample number where unfinished audio starts
        self.seq = 0  # number of the next chunk we expect
        self.paras = []  # finished paragraphs
        self.partial = ""  # tentative text still being spoken
        self.version = 0  # goes up whenever the text changes
        self.noise = None  # estimated background noise level
        self.running = True  # False once stopped
        self.error = None  # latest error message
        self.last_infer = 0.0  # seconds the last transcription took
        self._last_len = 0  # audio length at the last transcription
        self.event = threading.Event()  # wakes the worker when audio arrives
        self.thread = None  # the worker thread

    def append(self, seq, data):
        """Add one audio chunk, checking it is the next in order."""
        with self.lock:  # one thread at a time
            if seq < self.seq:  # we already have this chunk (a retry)
                return
            if seq > self.seq:  # a chunk is missing
                raise SeqGap(f"expected chunk {self.seq}, got {seq}")
            self.buf += data[: len(data) // 2 * 2]  # keep whole 2-byte samples only
            self.seq += 1  # expect the next chunk
        self.event.set()  # wake the worker

    def start(self):
        """Start the background worker."""
        self.thread = threading.Thread(target=self._run, daemon=True)  # make the thread
        self.thread.start()  # run it

    def stop(self):
        """Stop the worker and wait for it to finish."""
        self.running = False  # ask the loop to end
        self.event.set()  # wake the worker so it notices
        if self.thread:  # if a worker exists
            self.thread.join(timeout=60)  # wait up to a minute

    def _run(self):
        """Worker loop: keep processing audio until stopped."""
        while self.running:  # until stopped
            try:
                worked = self.step()  # process the audio
                self.error = None  # all fine
            except Exception as e:  # something went wrong
                log.exception("caption step failed")  # log the details
                self.error, worked = f"{type(e).__name__}: {e}"[:300], False  # remember a short message
                time.sleep(1)  # pause before retrying
            if not worked:  # nothing to do right now
                self.event.wait(0.25)  # wait for new audio
                self.event.clear()  # reset the signal
        try:  # after Stop, finish whatever was being said
            self.step(force_end=True)
        except Exception:
            log.exception("caption flush failed")  # log the problem

    def _advance(self, to_abs):
        """Move the cursor forward and throw away old audio."""
        with self.lock:  # one thread at a time
            self.cursor = max(self.cursor, to_abs)  # never move backwards
            drop = (self.cursor - SR) - self.base  # keep 1 second before the cursor
            if drop > 0:  # if there is old audio to drop
                del self.buf[: 2 * drop]  # delete it (2 bytes per sample)
                self.base += drop  # update where buf starts
        self._last_len = 0  # force a fresh transcription

    def _set_partial(self, text):
        """Update the tentative text."""
        with self.lock:  # one thread at a time
            if text != self.partial:  # only if it changed
                self.partial = text  # store it
                self.version += 1  # tell viewers there is news

    def step(self, force_end=False):
        """Process the unfinished audio once. Returns True if work was done."""
        with self.lock:  # copy the audio safely
            raw = bytes(self.buf[2 * (self.cursor - self.base):])  # audio since the cursor
            end_abs = self.base + len(self.buf) // 2  # sample number of the end
        x = np.frombuffer(raw[: len(raw) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768  # to -1..1
        n = len(x) // WIN  # number of 100 ms windows
        if n < 3:  # too little audio
            return False
        w = np.sqrt((x[: n * WIN].reshape(n, WIN) ** 2).mean(axis=1))  # loudness of each window
        low = float(np.percentile(w, 20))  # the quiet 20% shows the background noise
        # noise estimate drops instantly but rises slowly
        self.noise = low if self.noise is None else min(self.noise * 1.01, low)
        self.noise = min(self.noise, 0.02)  # cap the noise estimate
        voiced = w > min(max(3 * self.noise, 0.006), 0.05)  # which windows contain speech
        dur = len(x) / SR  # seconds of audio

        if not voiced.any():  # all silence
            if dur > 0.6:  # a little has piled up
                self._advance(end_abs - int(0.3 * SR))  # skip it
            self._set_partial("")  # clear tentative text
            return False
        first = int(voiced.argmax())  # first window with speech
        if first * WIN > SR:  # a long silence before speech
            self._advance(self.cursor + first * WIN - int(0.3 * SR))  # skip to just before it
            return True
        last_voiced = n - 1 - int(voiced[::-1].argmax())  # last window with speech
        ended = force_end or (n - 1 - last_voiced) * WIN >= ENDPOINT * SR  # has the speaker paused?
        vdur = voiced.sum() * WIN / SR  # seconds of speech
        if vdur < MIN_SPEECH:  # too short to be speech
            if ended:  # it was a click or cough
                self._advance(end_abs - int(0.3 * SR))  # skip it
                self._set_partial("")  # clear tentative text
                return True
            return False
        if not ended and len(x) - self._last_len < 0.2 * SR:  # not enough new audio
            return False

        t0 = time.time()  # start the timer
        segs = engine.transcribe(x, model=self.opts.get("model", "small.en"),
                                 language=self.opts.get("language") or None, vocab=self._prompt())  # transcribe
        self.last_infer = time.time() - t0  # record how long it took
        segs = [s for s in segs if s.text.strip()]  # drop empty segments
        spoken = re.sub(r"[^a-z ]", "", " ".join(s.text for s in segs).lower()).strip()  # plain lowercase text
        if segs and voiced.mean() < 0.25 and spoken in HALLUCINATIONS:  # a made-up phrase in near silence
            segs = []  # ignore it

        if ended or dur >= MAX_FORCE:  # speaker paused or spoke very long
            commit, rest, new_cursor = segs, [], end_abs  # finish everything
        elif dur >= MAX_TAIL:  # long speech: finish the completed segments
            commit = [s for s in segs if s.end <= dur - 1.5]  # segments that are surely done
            rest = [s for s in segs if s not in commit]  # the others stay tentative
            new_cursor = self.cursor + int(commit[-1].end * SR) if commit else None  # move past finished audio
        else:  # still speaking
            commit, rest, new_cursor = [], segs, None  # everything is tentative
        log.info("captions: %.1fs tail -> %d segs (%d final) in %.2fs", dur, len(segs), len(commit), self.last_infer)

        if commit:  # if anything is final
            self._commit(" ".join(s.text for s in commit))  # save it
        self._set_partial(self._clean(" ".join(s.text for s in rest)))  # show the rest as tentative
        if new_cursor is not None:  # if audio was finished
            self._advance(new_cursor)  # move on
        else:
            self._last_len = len(x)  # remember how much we transcribed
        return True

    def _prompt(self):
        """Build the hint given to Whisper: recent text plus vocabulary."""
        ctx = " ".join(p["text"] for p in self.paras[-2:])[-(500 - len(self.vocab)):] if self.paras else ""  # recent text
        return f"{ctx} {self.vocab}".strip()

    def _clean(self, text):
        """Tidy text using the term rules and the corrections file."""
        if not text.strip():  # nothing to clean
            return ""
        rules = self.term_rules + self.rules_fn()  # all replacement rules
        return autocorrect.clean(text, rules, remove_fillers=bool(self.opts.get("fillers", True)))

    def _commit(self, text):
        """Save final text as a paragraph and write it to the file."""
        text = self._clean(text)  # tidy it
        if not text:  # nothing left
            return
        now = time.time()  # current time
        with self.lock:  # one thread at a time
            if self.paras and now - self.paras[-1]["t"] < PARA_GAP:  # spoken soon after the last text
                self.paras[-1]["text"] += " " + text  # join to the last paragraph
                self.paras[-1]["t"] = now  # update its time
            else:  # after a longer pause
                self.paras.append({"t": now, "text": text})  # start a new paragraph
            self.version += 1  # tell viewers there is news
        if self.out_path:  # if saving to a file
            with open(self.out_path, "a", encoding="utf-8") as f:  # append so nothing is lost in a crash
                f.write(text + "\n")  # one line per saved text

    def snapshot(self, last=40):
        """Return the current captions for viewers."""
        with self.lock:  # one thread at a time
            paras = self.paras[-last:]  # only the latest paragraphs
            return dict(v=self.version, running=self.running, first=len(self.paras) - len(paras),
                        paras=[p["text"] for p in paras], partial=self.partial,
                        infer=round(self.last_infer, 2), error=self.error)
