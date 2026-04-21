from __future__ import annotations

from pathlib import Path

from .models import TranscriptSegment


class TranscriptionUnavailableError(RuntimeError):
    pass


def transcribe_audio(audio_path: Path, model_name: str) -> tuple[list[TranscriptSegment], str]:
    try:
        import mlx_whisper  # type: ignore
    except ImportError as exc:
        raise TranscriptionUnavailableError(
            "mlx-whisper ist nicht installiert. Installiere die Audio-Abhaengigkeiten zuerst."
        ) from exc

    result = mlx_whisper.transcribe(str(audio_path), path_or_hf_repo=model_name)
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
