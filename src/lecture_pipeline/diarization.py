from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Callable
import warnings

from .models import SpeakerProfile, TranscriptSegment


_DIARIZATION_PIPELINE = None
_DIARIZATION_DEVICE = None
_DIARIZATION_PIPELINE_LOCK = Lock()


def _resolve_device(configured_device: str) -> str:
    if configured_device == "auto":
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    return configured_device


def _segment_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _get_diarization_pipeline(
    hf_token: str,
    device: str = "cpu",
    progress_callback: Callable[[int, str], None] | None = None,
):
    global _DIARIZATION_PIPELINE, _DIARIZATION_DEVICE
    if _DIARIZATION_PIPELINE is not None and _DIARIZATION_DEVICE == device:
        return _DIARIZATION_PIPELINE

    with _DIARIZATION_PIPELINE_LOCK:
        if _DIARIZATION_PIPELINE is not None and _DIARIZATION_DEVICE == device:
            return _DIARIZATION_PIPELINE
        if progress_callback:
            progress_callback(58, f"Speaker-Modell wird geladen ({device.upper()}).")
        from pyannote.audio import Pipeline  # type: ignore
        import torch

        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
        pipeline.to(torch.device(device))
        _DIARIZATION_PIPELINE = pipeline
        _DIARIZATION_DEVICE = device
        return pipeline


def _clear_diarization_cache() -> None:
    global _DIARIZATION_PIPELINE, _DIARIZATION_DEVICE
    with _DIARIZATION_PIPELINE_LOCK:
        _DIARIZATION_PIPELINE = None
        _DIARIZATION_DEVICE = None


def diarize_audio(
    audio_path: Path,
    hf_token: str | None,
    configured_device: str = "auto",
    progress_callback: Callable[[int, str], None] | None = None,
) -> list[tuple[float, float, str]]:
    if not hf_token:
        return []

    try:
        from pyannote.audio import Pipeline  # type: ignore  # noqa: F401
    except ImportError:
        return []

    device = _resolve_device(configured_device)

    try:
        pipeline = _get_diarization_pipeline(hf_token, device=device, progress_callback=progress_callback)
        if progress_callback:
            progress_callback(62, f"Speaker-Diarization-Inferenz laeuft ({device.upper()}).")
        diarization = pipeline(str(audio_path))
        annotation = getattr(diarization, "exclusive_speaker_diarization", None)
        if annotation is None:
            annotation = getattr(diarization, "speaker_diarization", diarization)
        results: list[tuple[float, float, str]] = []
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            results.append((float(segment.start), float(segment.end), str(speaker)))
        if progress_callback:
            progress_callback(64, f"Speaker-Diarization erkannte {len(results)} Segmente.")
        return results
    except Exception as exc:
        if device != "cpu":
            warnings.warn(f"Diarization auf {device.upper()} fehlgeschlagen, fallback auf CPU: {exc}", stacklevel=2)
            _clear_diarization_cache()
            try:
                pipeline = _get_diarization_pipeline(hf_token, device="cpu", progress_callback=progress_callback)
                if progress_callback:
                    progress_callback(62, "Speaker-Diarization-Inferenz laeuft (CPU Fallback).")
                diarization = pipeline(str(audio_path))
                annotation = getattr(diarization, "exclusive_speaker_diarization", None)
                if annotation is None:
                    annotation = getattr(diarization, "speaker_diarization", diarization)
                results: list[tuple[float, float, str]] = []
                for segment, _, speaker in annotation.itertracks(yield_label=True):
                    results.append((float(segment.start), float(segment.end), str(speaker)))
                if progress_callback:
                    progress_callback(64, f"Speaker-Diarization (CPU) erkannte {len(results)} Segmente.")
                return results
            except Exception as exc2:
                warnings.warn(f"Diarization CPU fallback auch fehlgeschlagen: {exc2}", stacklevel=2)
                return []
        warnings.warn(f"Diarization fallback aktiv: {exc}", stacklevel=2)
        return []


def merge_speakers(
    transcript_segments: list[TranscriptSegment],
    diarization_segments: list[tuple[float, float, str]],
) -> tuple[list[TranscriptSegment], list[SpeakerProfile]]:
    if not transcript_segments:
        return [], []

    if not diarization_segments:
        diarization_segments = [
            (segment.start, segment.end, "SPEAKER_00") for segment in transcript_segments
        ]

    diarization_segments = sorted(diarization_segments, key=lambda item: (item[0], item[1]))
    durations: dict[str, float] = defaultdict(float)
    diarization_index = 0
    previous_label = diarization_segments[0][2]

    for transcript_segment in transcript_segments:
        while (
            diarization_index < len(diarization_segments)
            and diarization_segments[diarization_index][1] <= transcript_segment.start
        ):
            previous_label = diarization_segments[diarization_index][2]
            diarization_index += 1

        best_label = previous_label
        best_overlap = 0.0
        candidate_index = max(diarization_index - 1, 0)
        while candidate_index < len(diarization_segments):
            start, end, raw_label = diarization_segments[candidate_index]
            if start >= transcript_segment.end and best_overlap > 0:
                break
            if start > transcript_segment.end and candidate_index > diarization_index:
                break
            overlap = _segment_overlap(transcript_segment.start, transcript_segment.end, start, end)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = raw_label
            candidate_index += 1

        if best_overlap == 0 and diarization_index < len(diarization_segments):
            best_label = diarization_segments[diarization_index][2]

        transcript_segment.raw_speaker = best_label
        durations[best_label] += max(transcript_segment.end - transcript_segment.start, 0.0)

    ranked_labels = [label for label, _ in sorted(durations.items(), key=lambda item: item[1], reverse=True)]
    label_map: dict[str, SpeakerProfile] = {}
    for index, raw_label in enumerate(ranked_labels, start=1):
        is_prof = index == 1
        label_map[raw_label] = SpeakerProfile(
            raw_label=raw_label,
            base_label=f"Speaker {index}",
            display_label=f"Speaker {index} (Prof)" if is_prof else f"Speaker {index}",
            role="professor" if is_prof else "speaker",
            duration_seconds=durations[raw_label],
            reason="highest_speaking_time" if is_prof else "speaker_diarization_duration_ranking",
        )

    for transcript_segment in transcript_segments:
        profile = label_map[transcript_segment.raw_speaker or "SPEAKER_00"]
        transcript_segment.base_label = profile.base_label
        transcript_segment.display_label = profile.display_label

    return transcript_segments, list(label_map.values())
