import io

from scribe import app as appmod, autocorrect, engine


def test_clean():
    rules = [("open ai", "OpenAI"), ("gonna", "going to")]
    out = autocorrect.clean("um so i i think open ai is gonna win .it works", rules)
    assert out == "So I think OpenAI is going to win. It works"


def test_paragraphs():
    segs = [engine.Segment(0, 2, "Hi."), engine.Segment(2.2, 3, "There."), engine.Segment(6, 7, "Yes.")]
    assert engine.to_paragraphs(segs) == "Hi. There.\nYes."
    assert engine.to_paragraphs(segs, stamps=True).splitlines()[1].startswith("[00:06]")


def test_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(appmod, "HOME", tmp_path)
    monkeypatch.setattr(appmod, "CORRECTIONS", tmp_path / "corrections.txt")
    monkeypatch.setattr(appmod, "_to_wav", lambda src, dst: open(dst, "wb").close())
    monkeypatch.setattr(engine, "transcribe", lambda *a, **k: [engine.Segment(0, 1, "um open ai is gonna ship")])
    c = appmod.app.test_client()
    r = c.post("/api/transcribe", data={"audio": (io.BytesIO(b"x" * 2000), "a.wav")}).get_json()
    assert r["text"] == "OpenAI is going to ship"
    assert len(list(tmp_path.glob("*.txt"))) == 3  # corrections + transcript + raw


def test_empty_and_corrupt_audio_give_clear_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(appmod, "HOME", tmp_path)
    monkeypatch.setattr(appmod, "CORRECTIONS", tmp_path / "corrections.txt")
    c = appmod.app.test_client()
    r = c.post("/api/transcribe", data={"audio": (io.BytesIO(b"x"), "rec.webm")})
    assert r.status_code == 400 and "empty" in r.get_json()["error"]
    r = c.post("/api/transcribe", data={"audio": (io.BytesIO(b"garbage" * 500), "rec.webm")})
    assert r.status_code in (422, 500) and "error" in r.get_json()


def test_live_session_streams_commits_and_recovers(tmp_path, monkeypatch):
    import time
    from scribe import session

    monkeypatch.setattr(appmod, "HOME", tmp_path)
    monkeypatch.setattr(appmod, "CORRECTIONS", tmp_path / "corrections.txt")
    monkeypatch.setattr(session, "TICK", 0.05)
    monkeypatch.setattr(session, "MIN_TAIL", 5)
    monkeypatch.setattr(session, "MIN_PARTIAL", 1)

    def fake_extract(audio, start, dst):  # pretend 1 byte of audio == 1 second
        secs = max(audio.stat().st_size - start, 0)
        dst.write_bytes(b"\0" * (44 + int(secs * 32000)))

    def fake_transcribe(path, **kw):  # one 4s utterance every 6s of tail audio
        dur = (__import__("os").path.getsize(path) - 44) / 32000
        return [engine.Segment(t, min(t + 4, dur), f"um line {int(t)}") for t in range(0, int(dur), 6)]

    monkeypatch.setattr(session, "extract", fake_extract)
    monkeypatch.setattr(engine, "transcribe", fake_transcribe)
    c = appmod.app.test_client()
    sid = c.post("/api/session", json={"ext": "webm", "fillers": True}).get_json()["id"]
    for i in range(3):  # 3s of audio: too short to commit, but tentative text should already show
        assert c.post(f"/api/session/{sid}/chunk?seq={i}", data=b"x").status_code == 200
    for _ in range(60):
        early = c.get(f"/api/session/{sid}").get_json()
        if early.get("partial"):
            break
        time.sleep(0.05)
    assert early["partial"] and early["transcribed_seconds"] == 0
    for i in range(3, 20):
        assert c.post(f"/api/session/{sid}/chunk?seq={i}", data=b"x").status_code == 200
    assert c.post(f"/api/session/{sid}/chunk?seq=5", data=b"x").status_code == 200  # retry: ignored
    assert c.post(f"/api/session/{sid}/chunk?seq=25", data=b"x").status_code == 409  # gap: rejected
    assert (tmp_path / "sessions" / sid / "audio.webm").stat().st_size == 20

    time.sleep(0.5)
    live = c.get(f"/api/session/{sid}").get_json()
    assert live["status"] == "recording" and live["transcribed_seconds"] > 0 and "Line" in live["text"]

    session._registry.clear()  # simulate the app restarting mid-recording
    c.post(f"/api/session/{sid}/finish")
    for _ in range(100):
        done = c.get(f"/api/session/{sid}").get_json()
        if done["status"] == "done":
            break
        time.sleep(0.05)
    assert done["status"] == "done" and "um" not in done["text"].lower().split()
    assert (tmp_path / f"{sid}.txt").exists() and (tmp_path / f"{sid}.raw.txt").exists()
    assert c.get("/api/session/not-a-real-id").status_code == 404


def test_session_id_is_validated():
    from scribe import session
    assert not session.valid_id("../../etc")
