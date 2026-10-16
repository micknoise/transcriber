#!/usr/bin/env bash
# Start live captions. Add --lan to share the captions with other devices on your network.
# Stop the script if any command fails
set -e
# Work from the folder this script lives in
cd "$(dirname "$0")"
# Create the virtual environment the first time
[ -d .venv ] || python3 -m venv .venv
# Install the needed packages quietly
.venv/bin/pip install -q -r requirements.txt
# After 2 seconds, open the console in the browser (in the background)
(sleep 2; open "http://127.0.0.1:${CAPTIONS_PORT:-5056}") &
# Run the server, passing along any options such as --lan
exec .venv/bin/python -m scribe.captions_app "$@"
