# Scribe

Local, private transcription for conversations on a Mac. Record from the mic or drop in an audio/video file; get a cleaned-up, auto-corrected transcript. Nothing leaves the machine.

## Run

```
./run.sh
```

First run installs ffmpeg (Homebrew), creates a venv, and opens http://127.0.0.1:5055. The Whisper model downloads on first use (~1.5 GB for `large-v3-turbo`; pick `small` for a quick start).

Apple Silicon uses `mlx-whisper` (fast, GPU); Intel Macs fall back to `faster-whisper` (CPU).

## Autocorrect

1. **Vocabulary hints**: the right-hand side of your corrections dictionary is passed to Whisper as a prompt, so names/jargon are heard correctly.
2. **Dictionary** (`~/Transcripts/corrections.txt`, editable in the app): `wrong => right`, case-insensitive whole words.
3. **Clean-up**: removes um/uh and stutters, fixes spacing, sentence capitalisation and lone "i".
4. **Optional AI polish**: enter a local [Ollama](https://ollama.com) model (e.g. `llama3.1:8b`) to fix punctuation and mis-heard words. Output that strays far from the original is discarded.

## Live recording (long conversations)

Recording streams to disk as it goes: the browser sends a chunk every second and the app appends it to `~/Transcripts/sessions/<id>/audio.webm`. A background worker transcribes new audio at natural pauses, so the transcript appears live and is nearly finished when you press Stop. If the tab, browser or app dies mid-recording, reopen the app and click **Recover unfinished recording**. Keep the tab open and the Mac awake while recording. The mic needs the tab in the foreground in some browsers; Chrome is the most reliable.

Transcripts are saved to `~/Transcripts/` (`.txt` corrected, `.raw.txt` untouched). New paragraph on pauses >1.5s.

Not included: speaker labelling (diarization); pauses are used as a rough proxy for turn changes.

## Live captions (for lectures / presentations)

A second app in the same repo, tuned for low delay instead of accuracy-after-the-fact:

```
./run-captions.sh          # captions visible on this Mac only
./run-captions.sh --lan    # also shareable with phones/laptops on the same network
```

1. In the presenter console, pick your microphone, press **Start captions**, and watch the level meter.
2. Press **Open always-on-top window** and drag it onto the projector / second screen (full-screen button in its toolbar), and/or let audience members open the **audience link** (QR code shown) on their own devices. Chrome and Edge keep this window above other apps; unsupported browsers open a regular caption window instead. Each viewer sets their own text size, colours (white, yellow, black-on-white, blue), number of lines and font; choices are remembered per device.
3. Put slide vocabulary in **Terms for this talk** (names, jargon, `wrong => right`) before you start; it is fed to Whisper and fixes casing/spelling in the captions.

How it differs from the recorder: the browser streams raw 16 kHz audio (no container files, so delay doesn't grow over a two-hour lecture); an energy detector finds pauses, commits finished phrases, skips silence and stops Whisper inventing "Thank you." in quiet gaps; words appear as soon as about 0.5 s of speech has been heard and are firmed up at pauses. Expect roughly 1-2 s delay on Apple Silicon with `small.en`; use `base.en` if the Mac can't keep up (the console shows how long each update takes). Captions are appended to `~/Transcripts/captions_<time>.txt` as they finalise.

Notes: mic start/stop and all settings only accept requests from the presenter's own machine; only the read-only caption view is reachable over the network. Keep the console tab open (it may be behind your slides) and the Mac awake. First use of `--lan` triggers a macOS firewall prompt. Audience devices need to be on the same network; some venue Wi-Fi isolates clients from each other, in which case use a phone hotspot or show the caption window on the projector instead.

## Test

`pip install -r requirements.txt && pytest`
