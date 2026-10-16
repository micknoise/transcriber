"""Speech-to-text: mlx-whisper on Apple Silicon, faster-whisper elsewhere. All local."""
import os  # read environment variables
import platform  # find out which kind of computer this is
import threading  # run model downloads in the background
from concurrent.futures import ThreadPoolExecutor  # a single worker thread for the model
from dataclasses import dataclass  # a tiny class for one piece of text

# Names we offer in the app -> the Apple Silicon (MLX) model that goes with each one
MLX_MODELS = {"tiny.en": "mlx-community/whisper-tiny.en-mlx", "base.en": "mlx-community/whisper-base.en-mlx",
              "small.en": "mlx-community/whisper-small.en-mlx", "tiny": "mlx-community/whisper-tiny-mlx",
              "base": "mlx-community/whisper-base-mlx", "small": "mlx-community/whisper-small-mlx",
              "medium": "mlx-community/whisper-medium-mlx", "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
              "large-v3": "mlx-community/whisper-large-v3-mlx"}


@dataclass  # makes the class store its fields automatically
class Segment:
    """One piece of recognised speech."""
    start: float  # seconds from the start of the audio
    end: float  # seconds where it stops
    text: str  # the words


def _use_mlx():
    """True if we should use the fast Apple Silicon backend."""
    # Only Apple Silicon Macs can use MLX (and SCRIBE_BACKEND=faster turns it off)
    if platform.system() != "Darwin" or platform.machine() != "arm64" or os.environ.get("SCRIBE_BACKEND") == "faster":
        return False  # use faster-whisper instead
    try:  # check that mlx_whisper is really installed
        import mlx_whisper  # noqa: F401
        return True  # it is, so use it
    except ImportError:  # it is not installed
        return False  # fall back to faster-whisper


_fw_cache = {}  # faster-whisper models we have already loaded
_state = {}  # model name -> {"status": loading/ready/error}
_lock = threading.Lock()  # protects _state
# MLX only works on the thread that made it, so all transcribing happens on this one thread
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")


def ensure_model(model):
    """Download (first time only) and load the model."""
    if _use_mlx():  # Apple Silicon path
        from huggingface_hub import snapshot_download  # downloads model files
        snapshot_download(MLX_MODELS.get(model, model))  # fetch the files if needed
    else:  # faster-whisper path
        from faster_whisper import WhisperModel  # the model class
        from faster_whisper.utils import download_model  # downloads model files
        if model not in _fw_cache:  # only load each model once
            _fw_cache[model] = WhisperModel(download_model(model), compute_type="int8")


def model_status(model):
    """Say whether a model is idle, loading, ready or failed."""
    return dict(_state.get(model, {"status": "idle"}))  # idle if we never touched it


def load_async(model):
    """Start loading a model in the background (once). Poll model_status() to follow it."""
    with _lock:  # one caller at a time
        if _state.get(model, {}).get("status") in ("loading", "ready"):  # already started or done
            return dict(_state[model])  # just report where it is
        _state[model] = {"status": "loading"}  # mark it as loading

    def run():
        """Load the model, then record the result."""
        try:  # loading can fail (no network, disk full...)
            ensure_model(model)  # download and load
            _state[model] = {"status": "ready"}  # success
        except Exception as e:  # something went wrong
            _state[model] = {"status": "error", "error": str(e)}  # remember why

    threading.Thread(target=run, daemon=True).start()  # run in the background
    return {"status": "loading"}  # tell the caller it has started


def transcribe(audio, model="large-v3-turbo", language=None, vocab=""):
    """Turn audio (16 kHz mono float32 numpy array) into a list of Segments.
    vocab is given to Whisper as a hint so names and jargon are spelled right."""
    return _pool.submit(_transcribe, audio, model, language, vocab).result()  # run on the worker thread


def _transcribe(audio, model, language, vocab):
    """The real work, always run on the worker thread."""
    prompt = vocab or None  # Whisper wants None rather than an empty hint
    if _use_mlx():  # Apple Silicon path
        import mlx_whisper  # the fast backend
        r = mlx_whisper.transcribe(audio, path_or_hf_repo=MLX_MODELS.get(model, model), language=language,
                                   initial_prompt=prompt, condition_on_previous_text=False)
        return [Segment(s["start"], s["end"], s["text"].strip()) for s in r["segments"]]  # dicts -> Segments
    ensure_model(model)  # make sure the model is loaded
    segs, _ = _fw_cache[model].transcribe(audio, language=language, initial_prompt=prompt,
                                          vad_filter=True, condition_on_previous_text=False)
    return [Segment(s.start, s.end, s.text.strip()) for s in segs]  # objects -> Segments
