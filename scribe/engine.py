"""Speech-to-text backends: mlx-whisper on Apple Silicon, faster-whisper elsewhere. All local."""
import os
import platform
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

MLX_MODELS = {"tiny.en": "mlx-community/whisper-tiny.en-mlx", "base.en": "mlx-community/whisper-base.en-mlx",
              "small.en": "mlx-community/whisper-small.en-mlx", "tiny": "mlx-community/whisper-tiny-mlx", "base": "mlx-community/whisper-base-mlx",
              "small": "mlx-community/whisper-small-mlx", "medium": "mlx-community/whisper-medium-mlx",
              "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
              "large-v3": "mlx-community/whisper-large-v3-mlx"}


@dataclass
class Segment:
    start: float
    end: float
    text: str


def _use_mlx() -> bool:
    if platform.system() != "Darwin" or platform.machine() != "arm64" or os.environ.get("SCRIBE_BACKEND") == "faster":
        return False
    try:
        import mlx_whisper  # noqa: F401
        return True
    except ImportError:
        return False


_fw_cache = {}
_state: dict[str, dict] = {}  # model -> {"status": idle|loading|ready|error, "error": str}
_lock = threading.Lock()


def ensure_model(model: str) -> None:
    """Download (first time only) and load the model. Progress bars print to the terminal."""
    if _use_mlx():
        from huggingface_hub import snapshot_download
        snapshot_download(MLX_MODELS.get(model, model))
    else:
        from faster_whisper import WhisperModel
        from faster_whisper.utils import download_model
        if model not in _fw_cache:
            _fw_cache[model] = WhisperModel(download_model(model), compute_type="int8")


def model_status(model: str) -> dict:
    return dict(_state.get(model, {"status": "idle"}))


def load_async(model: str) -> dict:
    """Start loading in the background (once); poll model_status()."""
    with _lock:
        if _state.get(model, {}).get("status") in ("loading", "ready"):
            return dict(_state[model])
        _state[model] = {"status": "loading"}

    def run():
        try:
            ensure_model(model)
            _state[model] = {"status": "ready"}
        except Exception as e:  # network down, disk full, unknown model…
            _state[model] = {"status": "error", "error": str(e)}

    threading.Thread(target=run, daemon=True).start()
    return {"status": "loading"}


# MLX arrays/streams are tied to the thread that created them, and Flask serves each request on a new
# thread, so all inference runs on one dedicated thread (which also serialises concurrent jobs).
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")


def transcribe(audio, model: str = "large-v3-turbo", language: str | None = None,
               vocab: str = "") -> list[Segment]:
    return _pool.submit(_transcribe, audio, model, language, vocab).result()


def _transcribe(audio, model: str = "large-v3-turbo", language: str | None = None,
                vocab: str = "") -> list[Segment]:
    """audio: a file path, or a 16 kHz mono float32 numpy array.
    vocab is fed to Whisper as a prompt so names/jargon are spelled correctly."""
    prompt = vocab or None
    if _use_mlx():
        import mlx_whisper
        r = mlx_whisper.transcribe(audio, path_or_hf_repo=MLX_MODELS.get(model, model), language=language,
                                   initial_prompt=prompt, condition_on_previous_text=False)
        return [Segment(s["start"], s["end"], s["text"].strip()) for s in r["segments"]]
    ensure_model(model)
    segs, _ = _fw_cache[model].transcribe(audio, language=language, initial_prompt=prompt,
                                          vad_filter=True, condition_on_previous_text=False)
    return [Segment(s.start, s.end, s.text.strip()) for s in segs]


def to_paragraphs(segments: list[Segment], gap: float = 1.5, stamps: bool = False) -> str:
    """Start a new paragraph on a long pause (a rough proxy for a change of speaker/topic)."""
    paras, cur, last_end, start = [], [], None, 0.0
    for s in segments:
        if cur and last_end is not None and s.start - last_end > gap:
            paras.append((start, " ".join(cur)))
            cur = []
        if not cur:
            start = s.start
        cur.append(s.text)
        last_end = s.end
    if cur:
        paras.append((start, " ".join(cur)))
    fmt = (lambda t, p: f"[{int(t // 60):02d}:{int(t % 60):02d}] {p}") if stamps else (lambda t, p: p)
    return "\n".join(fmt(t, p) for t, p in paras)
