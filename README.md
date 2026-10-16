# Scribe

Local, private live captions for lectures and presentations on a Mac. Nothing leaves the machine.

## Run

```
./run-captions.sh          # captions visible on this Mac only
./run-captions.sh --lan    # also shareable with phones/laptops on the same network
```

The first run creates a venv, installs requirements and opens http://127.0.0.1:5056. The Whisper model downloads on first use (pick `small.en` for a quick start).

Apple Silicon uses `mlx-whisper` (fast); other machines use `faster-whisper` (CPU). Force one with `SCRIBE_BACKEND`.

## Use

1. In the presenter console, pick your microphone, press **Start captions**, and watch the level meter.
2. Press **Open always-on-top window** for the projector, and/or share the **audience link** (QR code shown). Each viewer sets their own text size, colours and font.
3. Put slide vocabulary in **Terms for this talk** (names, jargon, `wrong => right`) before you start.

## Files

Everything is stored in the `transcriptions/` folder (change it with `SCRIBE_HOME`):
- `captions_<time>.txt`: the captions, one paragraph per line, appended as they finalise.
- `corrections.txt`: your dictionary (`wrong => right`, case-insensitive whole words), created from `corrections.example.txt`. Its right-hand sides are also given to Whisper as vocabulary hints.

Clean-up removes um/uh and stutters and fixes spacing and capitalisation.

## Settings

- `CAPTIONS_PORT` (default 5056), `CAPTIONS_LAN=1` (same as `--lan`).
- Starting, stopping and settings only work from the presenter's own machine; only the read-only caption view is reachable over the network.
- Some venue Wi-Fi isolates devices from each other; use a phone hotspot or the projector window then.

## Code map

- `scribe/engine.py`: loads Whisper and transcribes.
- `scribe/captions.py`: turns audio chunks into live captions.
- `scribe/autocorrect.py`: the corrections and clean-up.
- `scribe/captions_app.py`: the web server.
- `scribe/static/`: the console and viewer pages.

## Test

`pip install -r requirements.txt && python -m pytest`
