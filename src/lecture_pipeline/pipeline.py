from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

from .audio import preprocess_audio
from .config import Settings
from .diarization import diarize_audio, merge_speakers
from .lm_studio import LMStudioClient
from .markdown import render_note_markdown, render_transcript_markdown
from .models import ChunkSummary, LectureRequest, NoteSections, PipelineResult, SpeakerProfile, TranscriptSegment
from .paths import build_pipeline_paths
from .storage import (
    delete_path,
    find_job_file_for_id,
    ingest_audio,
    initial_job_payload,
    remove_empty_parent_dirs,
    read_job_status,
    resolve_audio_path,
    resolve_transcript_source,
    resolve_vault_path,
    write_job_status,
    write_segments_json,
)
from .transcription import TranscriptionUnavailableError, transcribe_audio


class JobCancelledError(RuntimeError):
    pass


def _chunk_segments(segments: list[TranscriptSegment], target_chars: int) -> list[str]:
    chunks: list[str] = []
    current_lines: list[str] = []
    current_chars = 0
    for segment in segments:
        label = segment.display_label or segment.base_label or segment.raw_speaker or "Speaker 1 (Prof)"
        line = f"[{segment.start:.2f}-{segment.end:.2f}] {label}: {segment.text}"
        if current_lines and current_chars + len(line) > target_chars:
            chunks.append("\n".join(current_lines))
            current_lines = []
            current_chars = 0
        current_lines.append(line)
        current_chars += len(line)
    if current_lines:
        chunks.append("\n".join(current_lines))
    return chunks


def _fallback_note_sections(segments: list[TranscriptSegment]) -> NoteSections:
    preview = [segment.text for segment in segments[:8]]
    return NoteSections(
        zusammenfassung=["Automatische Zusammenfassung war nicht verfuegbar. Rohtranskript verlinkt."],
        notizen=preview or ["Keine Inhalte erkannt."],
        wichtige_begriffe=[],
        pruefungsrelevanz=[],
        offene_fragen=["LM-Studio-Zusammenfassung konnte nicht erzeugt werden."],
        naechste_schritte=["Rohtranskript pruefen und Zusammenfassung erneut ausfuehren."],
    )


def _speaker_profiles_from_segments(segments: list[TranscriptSegment]) -> list[SpeakerProfile]:
    durations: dict[str, float] = {}
    profiles: dict[str, SpeakerProfile] = {}

    for segment in segments:
        label = segment.display_label or segment.base_label or segment.raw_speaker or "Speaker 1"
        duration = max(segment.end - segment.start, 0.0)
        durations[label] = durations.get(label, 0.0) + duration
        if label not in profiles:
            is_professor = "(Prof)" in label
            profiles[label] = SpeakerProfile(
                raw_label=segment.raw_speaker or label,
                base_label=segment.base_label or label.replace(" (Prof)", ""),
                display_label=label,
                role="professor" if is_professor else "speaker",
                duration_seconds=0.0,
                reason="existing_transcript",
            )

    ordered_profiles = sorted(
        profiles.values(),
        key=lambda profile: (0 if profile.role == "professor" else 1, -durations.get(profile.display_label, 0.0), profile.display_label),
    )
    for profile in ordered_profiles:
        profile.duration_seconds = durations.get(profile.display_label, 0.0)
    return ordered_profiles


def prepare_job(
    request: LectureRequest,
    settings: Settings,
    *,
    job_id: str | None = None,
    created_at: datetime | None = None,
) -> tuple[str, datetime, Path | None, Path | None, object]:
    created_at = created_at or datetime.now(timezone.utc)
    source_audio: Path | None = None
    source_transcript: Path | None = None
    if request.audio_path:
        source_audio = resolve_audio_path(request.audio_path, settings.resolved_vault_root)
        if not source_audio.exists():
            raise FileNotFoundError(f"Audio-Datei nicht gefunden: {source_audio}")
    if request.transcript_path:
        source_transcript = resolve_vault_path(request.transcript_path, settings.resolved_vault_root)
        if not source_transcript.exists():
            raise FileNotFoundError(f"Transkript-Datei nicht gefunden: {source_transcript}")

    paths = build_pipeline_paths(settings, request, source_audio)
    job_id = job_id or str(uuid4())
    write_job_status(
        paths.job_path,
        initial_job_payload(
            request,
            paths,
            source_audio,
            job_id=job_id,
            created_at=created_at,
            source_transcript=source_transcript,
        ),
        replace=True,
    )
    return job_id, created_at, source_audio, source_transcript, paths


def load_job_status(job_id: str, settings: Settings, *, job_path: Path | None = None) -> dict | None:
    if job_path and job_path.exists():
        return read_job_status(job_path)

    job_path = find_job_file_for_id(settings.resolved_vault_root, job_id)
    if not job_path:
        return None
    return read_job_status(job_path)


def process_lecture(
    request: LectureRequest,
    settings: Settings,
    *,
    job_id: str | None = None,
    created_at: datetime | None = None,
    source_audio: Path | None = None,
    source_transcript: Path | None = None,
    paths=None,
    should_cancel: Callable[[], bool] | None = None,
) -> PipelineResult:
    if job_id is None or created_at is None or paths is None or (source_audio is None and source_transcript is None):
        job_id, created_at, source_audio, source_transcript, paths = prepare_job(
            request,
            settings,
            job_id=job_id,
            created_at=created_at,
        )

    def update_job(status: str, *, stage: str, progress: int, message: str, **extra: object) -> None:
        write_job_status(
            paths.job_path,
            {
                "job_id": job_id,
                "status": status,
                "stage": stage,
                "progress": progress,
                "message": message,
                **extra,
            },
        )

    cleanup_targets: list[Path] = []

    def register_cleanup_target(path: Path | None) -> None:
        if path is None:
            return
        if path not in cleanup_targets:
            cleanup_targets.append(path)

    def cleanup_created_data() -> None:
        for target in reversed(cleanup_targets):
            delete_path(target)
            remove_empty_parent_dirs(target, stop_at=settings.resolved_vault_root)

    def check_cancel(progress: int, stage: str, message: str) -> None:
        if should_cancel and should_cancel():
            cleanup_created_data()
            update_job(
                "cancelled",
                stage="cancelled",
                progress=progress,
                message="Job abgebrochen. Bereits erzeugte Daten wurden entfernt.",
                cancellation_requested=False,
            )
            raise JobCancelledError(message)

    try:
        transcription_model = settings.transcription_model
        transcript_markdown_path = paths.transcript_markdown_path
        canonical_audio_path = paths.canonical_audio_path

        check_cancel(1, "queued", "Job wurde vor dem Start abgebrochen.")

        if source_transcript is not None:
            update_job("running", stage="transcript_load", progress=12, message="Vorhandenes Transkript wird geladen.")
            check_cancel(12, "transcript_load", "Job wurde waehrend des Transkript-Ladens abgebrochen.")
            transcript_markdown_path, transcript_json_path, merged_segments, transcript_audio_path = resolve_transcript_source(
                source_transcript,
                settings.resolved_vault_root,
            )
            canonical_audio_path = transcript_audio_path
            if transcript_json_path is None or transcript_json_path != paths.transcript_json_path:
                write_segments_json(paths.transcript_json_path, merged_segments)
                register_cleanup_target(paths.transcript_json_path)
            if not transcript_markdown_path.exists():
                transcript_markdown = render_transcript_markdown(
                    vault_root=settings.resolved_vault_root,
                    request=request,
                    audio_path=canonical_audio_path,
                    speakers=_speaker_profiles_from_segments(merged_segments),
                    segments=merged_segments,
                )
                paths.transcript_markdown_path.write_text(transcript_markdown, encoding="utf-8")
                register_cleanup_target(paths.transcript_markdown_path)
                transcript_markdown_path = paths.transcript_markdown_path
            speakers = _speaker_profiles_from_segments(merged_segments)
            transcription_model = "existing_transcript"
            check_cancel(28, "transcript_load", "Job wurde nach dem Laden des vorhandenen Transkripts abgebrochen.")
            update_job(
                "running",
                stage="transcript_load",
                progress=28,
                message="Vorhandenes Transkript erfolgreich geladen.",
                segment_count=len(merged_segments),
                speaker_count=len(speakers),
                transcript_path=str(transcript_markdown_path),
            )
        else:
            update_job("running", stage="ingest", progress=5, message="Audio wird in den Kursordner uebernommen.")
            check_cancel(5, "ingest", "Job wurde vor dem Audio-Import abgebrochen.")
            canonical_audio_path = ingest_audio(source_audio, paths.canonical_audio_path)
            if source_audio is not None and canonical_audio_path.resolve() != source_audio.resolve():
                register_cleanup_target(canonical_audio_path)
            update_job(
                "running",
                stage="ingest",
                progress=12,
                message="Audio wurde uebernommen.",
                audio=str(canonical_audio_path),
            )

            update_job("running", stage="preprocess", progress=18, message="Audio wird fuer ASR vorbereitet.")
            prepared_audio = preprocess_audio(canonical_audio_path, paths.raw_audio_dir / f"{canonical_audio_path.stem}.wav")
            if prepared_audio.resolve() != canonical_audio_path.resolve():
                register_cleanup_target(prepared_audio)
            update_job(
                "running",
                stage="preprocess",
                progress=24,
                message="Audio-Vorverarbeitung abgeschlossen.",
                prepared_audio=str(prepared_audio),
            )

            update_job("running", stage="transcription", progress=30, message="Transkription laeuft.")
            check_cancel(30, "transcription", "Job wurde vor der Transkription abgebrochen.")
            try:
                segments, transcription_model = transcribe_audio(prepared_audio, settings.transcription_model)
            except TranscriptionUnavailableError as exc:
                raise RuntimeError(f"Transkription noch nicht verfuegbar: {exc}") from exc

            update_job(
                "running",
                stage="transcription",
                progress=48,
                message="Transkription abgeschlossen.",
                segment_count=len(segments),
                transcription_model=transcription_model,
            )

            update_job("running", stage="diarization", progress=56, message="Speaker-Diarization wird vorbereitet.")
            check_cancel(56, "diarization", "Job wurde vor der Speaker-Diarization abgebrochen.")
            diarization_segments = diarize_audio(
                prepared_audio,
                settings.hf_token,
                progress_callback=lambda progress, message: update_job(
                    "running",
                    stage="diarization",
                    progress=progress,
                    message=message,
                ),
            )
            update_job("running", stage="diarization", progress=65, message="Speaker werden dem Transkript zugeordnet.")
            check_cancel(65, "diarization", "Job wurde waehrend der Speaker-Zuordnung abgebrochen.")
            merged_segments, speakers = merge_speakers(segments, diarization_segments)
            write_segments_json(paths.transcript_json_path, merged_segments)
            register_cleanup_target(paths.transcript_json_path)
            update_job(
                "running",
                stage="diarization",
                progress=68,
                message="Speaker-Diarization abgeschlossen.",
                speaker_count=len(speakers),
            )

            update_job("running", stage="transcript_render", progress=72, message="Rohtranskript wird geschrieben.")
            check_cancel(72, "transcript_render", "Job wurde vor dem Schreiben des Rohtranskripts abgebrochen.")
            transcript_markdown = render_transcript_markdown(
                vault_root=settings.resolved_vault_root,
                request=request,
                audio_path=canonical_audio_path,
                speakers=speakers,
                segments=merged_segments,
            )
            paths.transcript_markdown_path.write_text(transcript_markdown, encoding="utf-8")
            register_cleanup_target(paths.transcript_markdown_path)
            transcript_markdown_path = paths.transcript_markdown_path

        note_sections = _fallback_note_sections(merged_segments)
        chunk_summaries: list[ChunkSummary] = []
        client = LMStudioClient(settings)
        try:
            chunks = _chunk_segments(merged_segments, settings.chunk_target_chars)
            total_chunks = len(chunks)
            if total_chunks:
                for index, chunk in enumerate(chunks, start=1):
                    check_cancel(74, "summary_chunks", "Job wurde vor der Zusammenfassung abgebrochen.")
                    progress = 74 + int(index / total_chunks * 14)
                    update_job(
                        "running",
                        stage="summary_chunks",
                        progress=progress,
                        message=f"Zusammenfassung Block {index} von {total_chunks}.",
                        current_chunk=index,
                        total_chunks=total_chunks,
                    )
                    chunk_summaries.append(
                        client.summarize_chunk(
                            course=request.course,
                            session_type=request.session_type,
                            theme=request.theme,
                            date=request.date,
                            chunk_text=chunk,
                        )
                    )

            update_job("running", stage="summary_final", progress=90, message="Finale Notiz wird synthetisiert.")
            check_cancel(90, "summary_final", "Job wurde vor der finalen Synthese abgebrochen.")
            note_sections = client.synthesize_note(
                course=request.course.replace("_", " "),
                course_link=request.course,
                session_type=request.session_type,
                theme=request.theme,
                date=request.date,
                chunk_summaries=chunk_summaries,
            )
        finally:
            client.close()

        update_job("running", stage="note_render", progress=95, message="Sitzungsnote wird geschrieben.")
        check_cancel(95, "note_render", "Job wurde vor dem Schreiben der Notiz abgebrochen.")
        note_markdown = render_note_markdown(
            vault_root=settings.resolved_vault_root,
            request=request,
            note_sections=note_sections,
            transcript_path=transcript_markdown_path,
            audio_path=canonical_audio_path,
            speaker_count=len(speakers),
            transcription_model=transcription_model,
            summary_model=settings.lm_studio_model,
            template_path=request.template_path,
        )
        paths.note_path.write_text(note_markdown, encoding="utf-8")
        register_cleanup_target(paths.note_path)

        update_job(
            "completed",
            stage="completed",
            progress=100,
            message="Pipeline abgeschlossen.",
            note_path=str(paths.note_path),
            transcript_path=str(transcript_markdown_path),
            speaker_count=len(speakers),
            segment_count=len(merged_segments),
            summary_model=settings.lm_studio_model,
        )

        return PipelineResult(
            job_id=job_id,
            created_at=created_at,
            input_audio_path=source_audio,
            paths=paths,
            speakers=speakers,
            segment_count=len(merged_segments),
            note_sections=note_sections,
            summary_model=settings.lm_studio_model,
            transcription_model=transcription_model,
        )
    except JobCancelledError:
        raise
    except Exception as exc:
        update_job(
            "failed",
            stage="failed",
            progress=100,
            message="Pipeline fehlgeschlagen.",
            error=str(exc),
        )
        raise
