"""Voice input that runs entirely on this machine.

Speech-to-text is local: faster-whisper (the Whisper models, running on CPU via
CTranslate2). No API key, no quota, and no audio leaves the computer. The model is
loaded once and reused, so only the first call pays for loading it.

Every transcription is appended to ``logs/voice.log`` so a bad run can be
diagnosed afterwards: what was heard, how long the clip was, and how long it took.
"""

import asyncio
import json
import logging
import os
import re
import sysconfig
import tempfile
import time
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, settings

LOG_FILE = PROJECT_ROOT / "logs" / "voice.log"
MAX_AUDIO_BYTES = 25 * 1024 * 1024

logger = logging.getLogger(__name__)

_cache: dict[str, Any] = {"model": None, "name": "", "device": ""}


class VoiceUnavailable(RuntimeError):
    """Raised when speech cannot be transcribed on this machine."""


def _prepare_cuda() -> bool:
    """
    Let CTranslate2 find the CUDA libraries installed by pip.

    Windows looks for cublas64_12.dll and the cuDNN DLLs on the loader path, and
    the wheels from nvidia-cublas-cu12 / nvidia-cudnn-cu12 are not on it.
    """
    try:
        site = sysconfig.get_paths()["purelib"]
    except Exception:
        return False

    dirs: list[str] = []
    for package in ("cublas", "cudnn"):
        candidate = Path(site) / "nvidia" / package / "bin"
        if candidate.is_dir():
            dirs.append(str(candidate))
    if not dirs:
        return False

    os.environ["PATH"] = os.pathsep.join(dirs) + os.pathsep + os.environ.get("PATH", "")
    for directory in dirs:
        try:
            os.add_dll_directory(directory)
        except Exception as e:  # already on the path, or not a directory
            logger.debug("add_dll_directory(%s) skipped: %s", directory, e)
    return True


def _cuda_devices() -> int:
    """How many CUDA devices CTranslate2 can see."""
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def _choose_device() -> tuple[str, str]:
    """Resolve the configured device and compute type into real values."""
    wanted = (settings.whisper_device or "auto").lower()
    if wanted == "auto":
        _prepare_cuda()
        wanted = "cuda" if _cuda_devices() > 0 else "cpu"

    compute = settings.whisper_compute or ("float16" if wanted == "cuda" else "int8")
    return wanted, compute


def _load_model() -> Any:
    """Load (once) and return the local Whisper model."""
    name = settings.whisper_model
    if _cache["model"] is not None and _cache["name"] == name:
        return _cache["model"]

    try:
        from faster_whisper import WhisperModel
    except Exception as e:  # pragma: no cover - depends on the install
        raise VoiceUnavailable(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        ) from e

    started = time.time()
    device, compute = _choose_device()

    try:
        model = WhisperModel(name, device=device, compute_type=compute)
    except Exception as first:
        if device != "cuda":
            raise VoiceUnavailable(f"cannot load the {name} model: {first}") from first
        # A broken CUDA install must not break voice entirely.
        logger.warning("CUDA load failed (%s); falling back to CPU", first)
        device, compute = "cpu", "int8"
        try:
            model = WhisperModel(name, device=device, compute_type=compute)
        except Exception as e:  # pragma: no cover
            raise VoiceUnavailable(f"cannot load the {name} model: {e}") from e

    _cache["model"] = model
    _cache["name"] = name
    _cache["device"] = device
    logger.info(
        "whisper %s on %s (%s) loaded in %.1fs", name, device, compute, time.time() - started
    )
    return model


def warm_up() -> None:
    """Load the model now, so the first utterance does not pay for it."""
    _load_model()


def forget_model() -> None:
    """Drop the loaded model so the next call picks up a new setting."""
    _cache["model"] = None
    _cache["name"] = ""
    _cache["device"] = ""


def _log(entry: dict[str, Any]) -> None:
    """Append one line of transcription history for later diagnosis."""
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _transcribe_file(path: str) -> str:
    """Run the model on a file, returning cleaned text."""
    model = _load_model()
    language = settings.whisper_language or None

    segments, _info = model.transcribe(
        path,
        language=language,
        beam_size=5,
        vad_filter=True,  # drop silence before it reaches the decoder
        condition_on_previous_text=False,
    )
    return clean_transcript(" ".join(segment.text.strip() for segment in segments))


async def transcribe(audio: bytes, filename: str = "speech.webm") -> str:
    """Turn recorded audio into text, on this machine."""
    if not audio:
        raise VoiceUnavailable("no audio received")
    if len(audio) > MAX_AUDIO_BYTES:
        raise VoiceUnavailable(f"audio is too large ({len(audio)} bytes)")

    suffix = Path(filename).suffix or ".webm"
    handle, path = tempfile.mkstemp(suffix=suffix)
    os.close(handle)
    started = time.time()

    try:
        Path(path).write_bytes(audio)
        text = await asyncio.to_thread(_transcribe_file, path)
    finally:
        try:
            Path(path).unlink()
        except OSError:
            pass

    elapsed = time.time() - started
    _log(
        {
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": settings.whisper_model,
            "bytes": len(audio),
            "seconds": round(elapsed, 2),
            "text": text,
        }
    )

    if not text:
        raise VoiceUnavailable("nothing was understood in that clip — try again closer to the mic")
    return text


def voice_available() -> bool:
    """Whether local speech-to-text can run here."""
    try:
        import faster_whisper  # noqa: F401
    except Exception:
        return False
    return True


def voice_status() -> dict[str, Any]:
    """What the UI and the settings sheet need to know about voice."""
    device, compute = _choose_device()
    return {
        "available": voice_available(),
        "engine": "faster-whisper (local)",
        "model": settings.whisper_model,
        "language": settings.whisper_language or "auto",
        "device": device,
        "compute": compute,
        "loaded_on": _cache["device"] or None,
    }


NOISE_TAG = re.compile(r"\[[^\]]*\]|\([^)]*\)")

# Spoken commands that should not be sent to a model at all.
STOP_WORDS = re.compile(
    r"^(?:ok(?:ay)?[,.! ]+)?(?:stop|cancel|never ?mind|quiet|shut up|that'?s (?:all|it|enough)|"
    r"dismiss|go away)[.!]?$",
    re.I,
)


def clean_transcript(text: str) -> str:
    """Strip the noise tags Whisper invents, and drop silent clips."""
    stripped = NOISE_TAG.sub(" ", str(text or ""))
    collapsed = " ".join(stripped.split())
    return collapsed if len(collapsed) > 1 else ""


def is_stop_command(text: str) -> bool:
    """Whether an utterance means "stop talking" rather than a question."""
    return bool(STOP_WORDS.match(clean_transcript(text)))
