#!/usr/bin/env bash
# One-command launcher for macOS: sets up a venv on first run, then opens the app.
set -e
cd "$(dirname "$0")"
command -v ffmpeg >/dev/null || { echo "Installing ffmpeg…"; brew install ffmpeg; }
[ -d .venv ] || { python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; }
(sleep 2; open "http://127.0.0.1:${PORT:-5055}") &
exec .venv/bin/python -m scribe.app
