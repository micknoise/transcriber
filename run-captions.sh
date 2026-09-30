#!/usr/bin/env bash
# Live captions launcher. Add --lan to let phones/laptops on the same network view the captions.
set -e
cd "$(dirname "$0")"
command -v ffmpeg >/dev/null || { echo "Installing ffmpeg…"; brew install ffmpeg; }
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
(sleep 2; open "http://127.0.0.1:${CAPTIONS_PORT:-5056}") &
exec .venv/bin/python -m scribe.captions_app "$@"
