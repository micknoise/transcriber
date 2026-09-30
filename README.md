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

## Test

`pip install -r requirements.txt && pytest`
