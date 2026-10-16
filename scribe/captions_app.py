"""Live captions server.

The presenter console and mic control only accept requests from this machine.
The read-only caption view (/view, /api/captions) can be shared on the LAN with --lan.
"""
import io  # in-memory buffer for the QR image
import logging  # prints request logs to the terminal
import os  # reads environment variables
import socket  # used to find this Mac's network address
import sys  # reads command-line arguments
import threading  # lock so start/stop don't clash
from datetime import datetime  # timestamps for file names
from pathlib import Path  # file paths

import segno  # makes QR codes
from flask import Flask, abort, jsonify, redirect, request, send_file, send_from_directory  # web server tools

from . import autocorrect, captions, engine  # our own modules

STATIC = Path(__file__).parent / "static"  # folder holding the web pages
# Folder where transcripts and corrections.txt are saved
HOME = Path(os.environ.get("SCRIBE_HOME", Path(__file__).parent.parent / "transcriptions"))
EXAMPLE = Path(__file__).parent.parent / "corrections.example.txt"  # starter corrections


def _rules():
    """Load the correction rules, creating the folder and file if needed."""
    HOME.mkdir(parents=True, exist_ok=True)  # make sure the folder exists
    mine = HOME / "corrections.txt"  # the user's rules file
    if not mine.exists() and EXAMPLE.exists():  # first run: no rules yet
        mine.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")  # copy the starter rules
    return autocorrect.load_corrections(mine)  # read the rules


LOCAL = {"127.0.0.1", "::1"}  # addresses that mean "this machine"
PORT = int(os.environ.get("CAPTIONS_PORT", 5056))  # port to listen on
LAN = False  # whether sharing on the network is on (set in serve)
app = Flask(__name__, static_folder=None)  # the web app
_stream = None  # the current caption stream, if any
_lock = threading.Lock()  # protects _stream


def local_only():
    """Refuse requests that don't come from this machine."""
    if request.remote_addr not in LOCAL:  # request came from elsewhere
        abort(403, "This control is only available on the presenter's machine.")  # forbidden


def lan_url():
    """Return the address audience members can open, or None."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP socket (sends nothing)
    try:
        s.connect(("10.255.255.255", 1))  # pick the network interface in use
        ip = s.getsockname()[0]  # our address on it
        return f"http://{ip}:{PORT}/view"  # the audience link
    except OSError:  # no network
        return None  # nothing to share
    finally:
        s.close()  # always close the socket


@app.errorhandler(Exception)
def on_error(e):
    """Return every error as JSON."""
    code = getattr(e, "code", None)  # HTTP status if there is one
    if isinstance(code, int) and code < 500:  # a client error
        return jsonify(error=getattr(e, "description", str(e))), code  # pass it on
    app.logger.exception("request failed")  # log real crashes
    return jsonify(error=f"{type(e).__name__}: {e}"[:500]), 500  # generic server error


@app.get("/")
def console():
    """Presenter console (others get redirected to the caption view)."""
    if request.remote_addr not in LOCAL:  # not the presenter
        return redirect("/view")  # send them to the audience page
    return send_from_directory(STATIC, "captions_console.html")  # show the console


@app.get("/view")
def view():
    """Audience caption page."""
    return send_from_directory(STATIC, "captions_view.html")  # show the page


@app.get("/favicon.ico")
def favicon():
    """Silence browser icon requests."""
    return "", 204  # empty response


@app.get("/api/captions")
def api_captions():
    """Return the latest captions."""
    st = _stream  # current stream
    if st is None:  # nothing started yet
        return jsonify(v=0, running=False, paras=[], partial="")  # empty captions
    snap = st.snapshot()  # copy of the current state
    if request.args.get("since") == str(snap["v"]):  # client is already up to date
        return jsonify(v=snap["v"], same=True, running=snap["running"])  # tiny reply
    return jsonify(snap)  # full reply


@app.get("/api/info")
def info():
    """Tell the console whether sharing is on and the link."""
    local_only()  # presenter only
    return jsonify(lan=lan_url() if LAN else None, lan_enabled=LAN)  # share details


@app.get("/qr.svg")
def qr():
    """Make a QR code for a link."""
    local_only()  # presenter only
    buf = io.BytesIO()  # in-memory file
    segno.make(request.args["u"][:300]).save(buf, kind="svg", scale=5, border=2)  # draw the QR
    buf.seek(0)  # rewind to the start
    return send_file(buf, mimetype="image/svg+xml")  # send the image


@app.get("/api/model/<path:name>")
def model_status(name):
    """Check whether a model is downloaded."""
    local_only()  # presenter only
    return jsonify(engine.model_status(name))  # its status


@app.post("/api/model/<path:name>")
def model_load(name):
    """Start downloading/loading a model."""
    local_only()  # presenter only
    return jsonify(engine.load_async(name))  # its status


@app.post("/api/start")
def start():
    """Start a new caption session."""
    local_only()  # presenter only
    global _stream  # we replace the stream
    o = request.get_json() or {}  # settings from the console
    opts = {k: o[k] for k in ("model", "language", "terms", "fillers") if k in o}  # keep known settings
    with _lock:  # one start/stop at a time
        if _stream:  # a session is already running
            _stream.stop()  # end it
        HOME.mkdir(parents=True, exist_ok=True)  # make sure the folder exists
        out = HOME / f"captions_{datetime.now():%Y-%m-%d_%H%M%S}.txt"  # new transcript file
        _stream = captions.CaptionStream(opts, _rules, out)  # create the stream
        _stream.start()  # begin work
    return jsonify(ok=True, saving_to=str(out))  # tell the console where it saves


@app.post("/api/audio")
def audio():
    """Receive a chunk of microphone audio."""
    local_only()  # presenter only
    st = _stream  # current stream
    if st is None or not st.running:  # nothing to feed
        abort(410, "Captions aren't running. Press Start.")  # gone
    try:
        st.append(int(request.args["seq"]), request.get_data())  # add the chunk
    except captions.SeqGap as e:  # a chunk went missing
        abort(409, str(e))  # conflict
    return jsonify(ok=True)  # accepted


@app.post("/api/stop")
def stop():
    """Stop the caption session."""
    local_only()  # presenter only
    with _lock:  # one start/stop at a time
        if _stream:  # something is running
            _stream.stop()  # end it
    return jsonify(ok=True)  # done


def serve():
    """Run the server."""
    global LAN  # we set it here
    LAN = "--lan" in sys.argv or os.environ.get("CAPTIONS_LAN") == "1"  # sharing switch
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")  # log style
    print(f"\n  Presenter console:  http://127.0.0.1:{PORT}")  # console link
    if LAN:  # sharing on
        print(f"  Audience captions:  {lan_url()}   (same Wi-Fi/network)\n")  # audience link
    else:  # sharing off
        print("  Captions are visible on this Mac only. Run with --lan to share them with other devices.\n")  # hint
    app.run(host="0.0.0.0" if LAN else "127.0.0.1", port=PORT, threaded=True)  # start listening


if __name__ == "__main__":  # run directly
    serve()  # start the server
