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
