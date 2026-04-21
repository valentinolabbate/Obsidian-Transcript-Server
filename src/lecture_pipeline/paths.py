from __future__ import annotations

from pathlib import Path

from .config import Settings
from .models import LectureRequest, PipelinePaths
from .utils import canonical_stem, ensure_directory, sanitize_filename_part


def _render_request_path(template: str, request: LectureRequest) -> str:
    replacements = {
        "course": request.course,
        "context": request.course,
        "course_display": request.course.replace("_", " "),
        "date": request.date,
        "session_type": sanitize_filename_part(request.session_type),
        "theme": sanitize_filename_part(request.theme),
        "stem": canonical_stem(request.date, request.session_type, request.theme),
    }
    rendered = template.strip()
    for key, value in replacements.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
        rendered = rendered.replace(f"{{{key}}}", value)
    return rendered


def _resolve_request_directory(vault_root: Path, configured_path: Path | str, request: LectureRequest) -> Path:
    rendered = _render_request_path(str(configured_path), request)
    candidate = Path(rendered).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (vault_root / candidate).resolve()


def build_pipeline_paths(settings: Settings, request: LectureRequest, source_audio: Path | None) -> PipelinePaths:
    if request.storage_dir:
        storage_root = ensure_directory(
            _resolve_request_directory(settings.resolved_vault_root, request.storage_dir, request)
        )
    else:
        storage_root = ensure_directory(
            settings.resolved_study_root / request.course / "Rohdaten"
        )

    if request.output_dir:
        output_dir = ensure_directory(
            _resolve_request_directory(settings.resolved_vault_root, request.output_dir, request)
        )
    else:
        output_dir = ensure_directory(settings.resolved_study_root / request.course / "Sitzungen")

    raw_audio_dir = ensure_directory(storage_root / "Audio")
    raw_transcript_dir = ensure_directory(storage_root / "Transkripte")
    raw_jobs_dir = ensure_directory(storage_root / "Jobs")

    stem = canonical_stem(request.date, request.session_type, request.theme)
    extension = source_audio.suffix if source_audio and source_audio.suffix else ".m4a"
    canonical_audio_path = raw_audio_dir / f"{stem}{extension}" if source_audio else None
    transcript_markdown_path = raw_transcript_dir / f"{stem}.transcript.md"
    transcript_json_path = raw_transcript_dir / f"{stem}.segments.json"
    note_path = output_dir / f"{stem}.md"
    job_path = raw_jobs_dir / f"{stem}.job.json"

    return PipelinePaths(
        storage_root=storage_root,
        raw_audio_dir=raw_audio_dir,
        raw_transcript_dir=raw_transcript_dir,
        raw_jobs_dir=raw_jobs_dir,
        output_dir=output_dir,
        canonical_audio_path=canonical_audio_path,
        transcript_markdown_path=transcript_markdown_path,
        transcript_json_path=transcript_json_path,
        note_path=note_path,
        job_path=job_path,
    )
