"""Speech-to-text backends: mlx-whisper on Apple Silicon, faster-whisper elsewhere. All local."""
import os
import platform
from dataclasses import dataclass

MLX_MODELS = {"tiny": "mlx-community/whisper-tiny-mlx", "base": "mlx-community/whisper-base-mlx",
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


def transcribe(path: str, model: str = "large-v3-turbo", language: str | None = None,
               vocab: str = "") -> list[Segment]:
    """vocab is fed to Whisper as a prompt so names/jargon are spelled correctly."""
    prompt = vocab or None
    if _use_mlx():
        import mlx_whisper
        r = mlx_whisper.transcribe(path, path_or_hf_repo=MLX_MODELS.get(model, model), language=language,
                                   initial_prompt=prompt, condition_on_previous_text=False)
        return [Segment(s["start"], s["end"], s["text"].strip()) for s in r["segments"]]
    from faster_whisper import WhisperModel
    if model not in _fw_cache:
        _fw_cache[model] = WhisperModel(model, compute_type="int8")
    segs, _ = _fw_cache[model].transcribe(path, language=language, initial_prompt=prompt,
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
