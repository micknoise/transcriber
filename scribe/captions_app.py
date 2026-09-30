"""Live-captions app. The presenter console and mic control only accept requests from this machine;
the read-only caption view (/view, /api/captions) can be shared with other devices on the LAN (--lan)."""
import io
import logging
import os
import socket
import sys
import threading
from datetime import datetime

import segno
from flask import Flask, abort, jsonify, redirect, request, send_file, send_from_directory

from . import app as main
from . import captions, engine

STATIC = main.STATIC
LOCAL = {"127.0.0.1", "::1"}
PORT = int(os.environ.get("CAPTIONS_PORT", 5056))
LAN = False  # set by serve()

app = Flask(__name__, static_folder=None)
_stream: captions.CaptionStream | None = None
_lock = threading.Lock()


def local_only():
    if request.remote_addr not in LOCAL:
        abort(403)


def lan_url() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packets sent; just selects the outbound interface
            return f"http://{s.getsockname()[0]}:{PORT}/view"
    except OSError:
        return None


@app.errorhandler(Exception)
def on_error(e):
    code = getattr(e, "code", None)
    if isinstance(code, int) and code < 500:
        return jsonify(error=getattr(e, "description", str(e))), code
    app.logger.exception("request failed")
    return jsonify(error=f"{type(e).__name__}: {e}"[:500]), 500


@app.get("/")
def console():
    if request.remote_addr not in LOCAL:
        return redirect("/view")
    return send_from_directory(STATIC, "captions_console.html")


@app.get("/view")
def view():
    return send_from_directory(STATIC, "captions_view.html")


@app.get("/favicon.ico")
def favicon():
    return "", 204


@app.get("/api/captions")
def get_captions():
    st = _stream
    if st is None:
        return jsonify(v=0, running=False, paras=[], partial="")
    snap = st.snapshot()
    if request.args.get("since") == str(snap["v"]):
        return jsonify(v=snap["v"], same=True, running=snap["running"])
    return jsonify(snap)


@app.get("/api/info")
def info():
    local_only()
    return jsonify(lan=lan_url() if LAN else None, lan_enabled=LAN)


@app.get("/qr.svg")
def qr():
    local_only()
    buf = io.BytesIO()
    segno.make(request.args["u"][:300]).save(buf, kind="svg", scale=5, border=2)
    buf.seek(0)
    return send_file(buf, mimetype="image/svg+xml")


@app.get("/api/model/<path:name>")
def model_status(name):
    local_only()
    return jsonify(engine.model_status(name))


@app.post("/api/model/<path:name>")
def model_load(name):
    local_only()
    return jsonify(engine.load_async(name))


@app.post("/api/start")
def start():
    global _stream
    local_only()
    o = request.get_json() or {}
    opts = {k: o[k] for k in ("model", "language", "terms", "fillers") if k in o}
    with _lock:
        if _stream:
            _stream.stop()
        main.HOME.mkdir(parents=True, exist_ok=True)
        out = main.HOME / f"captions_{datetime.now():%Y-%m-%d_%H%M%S}.txt"
        _stream = captions.CaptionStream(opts, main._rules, out)
        _stream.start()
    return jsonify(ok=True, saving_to=str(out))


@app.post("/api/audio")
def audio():
    local_only()
    st = _stream
    if st is None or not st.running:
        abort(410, "Captions aren't running. Press Start.")
    try:
        st.append(int(request.args["seq"]), request.get_data())
    except captions.SeqGap as e:
        abort(409, str(e))
    return jsonify(ok=True)


@app.post("/api/stop")
def stop():
    local_only()
    with _lock:
        if _stream:
            _stream.stop()
    return jsonify(ok=True)



def serve():
    global LAN
    LAN = "--lan" in sys.argv or os.environ.get("CAPTIONS_LAN") == "1"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    print(f"\n  Presenter console:  http://127.0.0.1:{PORT}")
    if LAN:
        print(f"  Audience captions:  {lan_url()}   (same Wi-Fi/network)\n")
    else:
        print("  Captions are visible on this Mac only. Run with --lan to share them with other devices.\n")
    app.run(host="0.0.0.0" if LAN else "127.0.0.1", port=PORT, threaded=True)


if __name__ == "__main__":
    serve()
