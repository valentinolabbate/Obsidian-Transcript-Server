from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import orjson

from .models import LectureRequest, PipelinePaths, TranscriptSegment


def resolve_vault_path(path: Path, vault_root: Path) -> Path:
    candidate = path.expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (vault_root / candidate).resolve()


def resolve_audio_path(path: Path, vault_root: Path) -> Path:
    return resolve_vault_path(path, vault_root)


def ingest_audio(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return destination
    shutil.copy2(source, destination)
    return destination


def write_segments_json(path: Path, segments: list[TranscriptSegment]) -> None:
    payload = [segment.model_dump() for segment in segments]
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def delete_path(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        return
    path.unlink(missing_ok=True)


def remove_empty_parent_dirs(path: Path | None, stop_at: Path | None = None) -> None:
    if path is None:
        return
    current = path.parent
    stop_at_resolved = stop_at.resolve() if stop_at else None
    while current.exists() and current.is_dir():
        if stop_at_resolved and current.resolve() == stop_at_resolved:
            break
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def read_segments_json(path: Path) -> list[TranscriptSegment]:
    payload = orjson.loads(path.read_bytes())
    if not isinstance(payload, list):
        raise ValueError(f"Segmentdatei ist ungueltig: {path}")
    return [TranscriptSegment.model_validate(item) for item in payload]


def _timestamp_to_seconds(timestamp: str) -> float:
    parts = [int(part) for part in timestamp.split(":")]
    if len(parts) != 3:
        raise ValueError(f"Ungueltiger Zeitstempel: {timestamp}")
    hours, minutes, seconds = parts
    return float(hours * 3600 + minutes * 60 + seconds)


def _extract_audio_link(text: str) -> str | None:
    match = re.search(r'QuelleAudio:\s*"\[\[(.+?)\]\]"', text)
    if match:
        return match.group(1)
    match = re.search(r"\*\*Audio:\*\*\s*\[\[(.+?)\]\]", text)
    if match:
        return match.group(1)
    return None


def read_transcript_markdown(path: Path) -> tuple[list[TranscriptSegment], Path | None]:
    text = path.read_text(encoding="utf-8")
    audio_link = _extract_audio_link(text)
    audio_path = Path(audio_link) if audio_link else None

    marker = "## Transkript"
    marker_index = text.find(marker)
    transcript_body = text[marker_index + len(marker):] if marker_index >= 0 else text
    chunks = re.split(r"\n\s*\n", transcript_body.strip())
    segments: list[TranscriptSegment] = []
    pattern = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\s*-\s*(\d{2}:\d{2}:\d{2})\]\s+(.+?):\s*$", re.MULTILINE)

    for chunk in chunks:
        stripped = chunk.strip()
        if not stripped:
            continue
        match = pattern.match(stripped)
        if not match:
            continue
        start, end, speaker = match.groups()
        lines = stripped.splitlines()
        body = "\n".join(lines[1:]).strip()
        if not body:
            continue
        segments.append(
            TranscriptSegment(
                start=_timestamp_to_seconds(start),
                end=_timestamp_to_seconds(end),
                text=body,
                display_label=speaker.strip(),
            )
        )

    if not segments:
        raise ValueError(f"Keine Transkriptsegmente in Markdown-Datei gefunden: {path}")

    return segments, audio_path


def resolve_transcript_source(
    transcript_path: Path,
    vault_root: Path,
) -> tuple[Path, Path | None, list[TranscriptSegment], Path | None]:
    resolved = resolve_vault_path(transcript_path, vault_root)
    if not resolved.exists():
        raise FileNotFoundError(f"Transkript-Datei nicht gefunden: {resolved}")

    if resolved.name.endswith(".segments.json"):
        transcript_json_path = resolved
        transcript_markdown_path = resolved.with_name(resolved.name[: -len(".segments.json")] + ".transcript.md")
        segments = read_segments_json(resolved)
        audio_path = None
        if transcript_markdown_path.exists():
            _, audio_path = read_transcript_markdown(transcript_markdown_path)
            if audio_path is not None:
                audio_path = resolve_vault_path(audio_path, vault_root)
            return transcript_markdown_path, transcript_json_path, segments, audio_path
        return transcript_markdown_path, transcript_json_path, segments, None

    if resolved.name.endswith(".transcript.md"):
        transcript_markdown_path = resolved
        transcript_json_path = resolved.with_name(resolved.name[: -len(".transcript.md")] + ".segments.json")
        segments, audio_path = read_transcript_markdown(resolved)
        if audio_path is not None:
            audio_path = resolve_vault_path(audio_path, vault_root)
        if transcript_json_path.exists():
            segments = read_segments_json(transcript_json_path)
        else:
            transcript_json_path = None
        return transcript_markdown_path, transcript_json_path, segments, audio_path

    raise ValueError("Unterstuetzt werden nur .transcript.md und .segments.json Dateien.")


def read_job_status(path: Path) -> dict:
    if not path.exists():
        return {}
    return orjson.loads(path.read_bytes())


def write_job_status(path: Path, payload: dict, *, replace: bool = False) -> dict:
    existing = {} if replace else read_job_status(path)
    merged = {**existing, **payload}
    merged["updated_at"] = datetime.now(timezone.utc).isoformat()
    if "created_at" not in merged:
        merged["created_at"] = merged["updated_at"]
    path.write_bytes(orjson.dumps(merged, option=orjson.OPT_INDENT_2))
    return merged


def find_job_file_for_id(study_root: Path, job_id: str) -> Path | None:
    for candidate in study_root.rglob("*.job.json"):
        try:
            payload = read_job_status(candidate)
        except orjson.JSONDecodeError:
            continue
        if payload.get("job_id") == job_id:
            return candidate
    return None


def build_note_link(vault_root: Path, target: Path) -> str:
    return target.relative_to(vault_root).as_posix()


def initial_job_payload(
    request: LectureRequest,
    paths: PipelinePaths,
    source_audio: Path | None,
    *,
    job_id: str,
    created_at: datetime,
    source_transcript: Path | None = None,
) -> dict:
    payload = {
        "job_id": job_id,
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "message": "Job wurde angelegt.",
        "created_at": created_at.isoformat(),
        "request": request.model_dump(mode="json"),
        "paths": {key: (str(value) if value is not None else None) for key, value in paths.model_dump().items()},
    }
    if source_audio:
        payload["source_audio_path"] = str(source_audio)
    if source_transcript:
        payload["source_transcript_path"] = str(source_transcript)
    return payload
