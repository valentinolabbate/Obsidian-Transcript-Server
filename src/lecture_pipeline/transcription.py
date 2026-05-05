from __future__ import annotations

import gc
import importlib
from pathlib import Path
from threading import Lock

from .models import TranscriptSegment

_TRANSCRIPTION_MODEL_LOCK = Lock()


class TranscriptionUnavailableError(RuntimeError):
    pass


def unload_transcription_model() -> None:
    try:
        transcribe_module = importlib.import_module("mlx_whisper.transcribe")
        model_holder = getattr(transcribe_module, "ModelHolder", None)
        if model_holder is not None:
            model_holder.model = None
            model_holder.model_path = None
    except Exception:
        pass

    gc.collect()
    try:
        import mlx.core as mx  # type: ignore

        mx.clear_cache()
    except Exception:
        pass


def transcribe_audio(audio_path: Path, model_name: str) -> tuple[list[TranscriptSegment], str]:
    try:
        import mlx_whisper  # type: ignore
    except ImportError as exc:
        raise TranscriptionUnavailableError(
            "mlx-whisper ist nicht installiert. Installiere die Audio-Abhaengigkeiten zuerst."
        ) from exc

    with _TRANSCRIPTION_MODEL_LOCK:
        try:
            result = mlx_whisper.transcribe(str(audio_path), path_or_hf_repo=model_name)
        finally:
            unload_transcription_model()

    raw_segments = result.get("segments") or []
    segments = [
        TranscriptSegment(
            start=float(segment.get("start", 0.0)),
            end=float(segment.get("end", 0.0)),
            text=str(segment.get("text", "")).strip(),
        )
        for segment in raw_segments
        if str(segment.get("text", "")).strip()
    ]

    if not segments:
        raise RuntimeError("Die Transkription lieferte keine Segmente.")

    return segments, model_name
