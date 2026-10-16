import numpy as np
import pytest

from scribe import captions, captions_app as capp, engine

SR = captions.SR


def speech(sec):  # loud tone, well above the noise gate
    t = np.arange(int(sec * SR)) / SR
    return (0.2 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)


def quiet(sec):
    return (np.random.default_rng(0).normal(0, 0.001, int(sec * SR)) * 32767).astype(np.int16)


@pytest.fixture
def stream(tmp_path, monkeypatch):
    calls = []

    def fake(audio, **kw):  # one "sentence" per 3 s of audio it is handed
        calls.append(len(audio) / SR)
        d = len(audio) / SR
        return [engine.Segment(t, min(t + 3, d), f"um sentence {i}") for i, t in enumerate(np.arange(0, d, 3))]

    monkeypatch.setattr(engine, "transcribe", fake)
    s = captions.CaptionStream({"fillers": True}, lambda: [], tmp_path / "out.txt")
    s.calls = calls
    return s


def feed(s, *parts):
    pcm = np.concatenate(parts).tobytes()
    for i in range(0, len(pcm), 8000):
        s.append(s.seq, pcm[i:i + 8000])
    for _ in range(40):  # run the worker logic synchronously
        if not s.step():
            break


def test_silence_is_never_transcribed(stream):
    feed(stream, quiet(6))
    assert stream.calls == [] and stream.snapshot()["paras"] == [] and stream.snapshot()["partial"] == ""


def test_tentative_text_then_committed_at_pause(stream):
    feed(stream, quiet(0.5), speech(2))
    snap = stream.snapshot()
    assert snap["partial"].startswith("Sentence 0") and snap["paras"] == []  # tentative, fillers removed
    feed(stream, quiet(1.0))  # pause ends the utterance
    snap = stream.snapshot()
    assert snap["partial"] == "" and snap["paras"] and "Sentence 0" in snap["paras"][0]
    assert (stream.out_path.read_text()).strip() == snap["paras"][0]
    feed(stream, quiet(2))
    assert len(stream.snapshot()["paras"]) == 1  # the quiet after a pause adds nothing


def test_continuous_speech_commits_complete_sentences(stream):
    feed(stream, speech(12))  # no pause at all
    snap = stream.snapshot()
    assert snap["paras"], "long unbroken speech must still produce final text"
    assert stream.cursor > 0 and len(stream.buf) // 2 < 12 * SR  # buffer trimmed, doesn't grow forever


def test_hallucinated_thank_you_in_near_silence_is_dropped(stream, monkeypatch):
    monkeypatch.setattr(engine, "transcribe", lambda *a, **k: [engine.Segment(0, 1, "Thank you.")])
    feed(stream, speech(0.4), quiet(2.5))
    assert stream.snapshot()["paras"] == [] and stream.snapshot()["partial"] == ""


def test_term_rules_fix_casing_and_spelling(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "transcribe", lambda *a, **k: [engine.Segment(0, 1, "we deploy kubernetes with open ai")])
    s = captions.CaptionStream({"terms": "Kubernetes\nopen ai => OpenAI"}, lambda: [])
    feed(s, speech(1.5), quiet(1))
    assert s.snapshot()["paras"] == ["We deploy Kubernetes with OpenAI"]
    assert "Kubernetes" in s.vocab


def test_chunk_sequence_rules(stream):
    stream.append(0, b"\0\0")
    stream.append(0, b"\0\0")  # duplicate retry ignored
    assert stream.seq == 1
    with pytest.raises(captions.SeqGap):
        stream.append(5, b"\0\0")


def test_control_endpoints_are_local_only_but_viewing_is_not(monkeypatch):
    c = capp.app.test_client()
    lan = {"REMOTE_ADDR": "192.168.1.50"}
    for method, url in [("post", "/api/start"), ("post", "/api/audio?seq=0"), ("post", "/api/stop"),
                        ("get", "/api/info"), ("post", "/api/model/small.en")]:
        assert getattr(c, method)(url, environ_overrides=lan).status_code == 403, url
    assert c.get("/view", environ_overrides=lan).status_code == 200
    assert c.get("/api/captions", environ_overrides=lan).status_code == 200
    assert c.get("/", environ_overrides=lan).status_code == 302  # audience gets redirected to the viewer
    assert c.get("/", environ_overrides={"REMOTE_ADDR": "127.0.0.1"}).status_code == 200


def test_autocorrect_cleans_text(tmp_path):
    from scribe import autocorrect
    rules = autocorrect.load_corrections(__import__("pathlib").Path("corrections.example.txt"))
    out = autocorrect.clean("um so i i think open ai is gonna win .it works", rules)
    assert "OpenAI" in out and "going to" in out and "um" not in out.split()
