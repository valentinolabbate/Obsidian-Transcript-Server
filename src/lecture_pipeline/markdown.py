from __future__ import annotations

from pathlib import Path
import re

from .models import LectureRequest, NoteSections, SpeakerProfile, TranscriptSegment
from .paths import render_request_path
from .storage import build_note_link
from .utils import format_timestamp


def _render_bullets(items: list[str]) -> str:
    if not items:
        return "- "
    return "\n".join(f"- {item}" for item in items)


def _render_term_bullets(items: list[dict[str, str]]) -> str:
    if not items:
        return "- "
    lines: list[str] = []
    for item in items:
        term = item.get("begriff", "").strip()
        explanation = item.get("erklaerung", "").strip()
        if term and explanation:
            lines.append(f"- **{term}:** {explanation}")
        elif term:
            lines.append(f"- **{term}**")
    return "\n".join(lines) or "- "


def _resolve_optional_template_path(vault_root: Path, template_path: Path | None, request: LectureRequest) -> Path | None:
    if not template_path:
        return None
    candidate = Path(render_request_path(str(template_path), request)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (vault_root / candidate).resolve()


def _render_note_template(template: str, context: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        return context.get(key, match.group(0))

    return re.sub(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}", replace, template)


def render_transcript_markdown(
    *,
    vault_root: Path,
    request: LectureRequest,
    audio_path: Path | None,
    speakers: list[SpeakerProfile],
    segments: list[TranscriptSegment],
) -> str:
    audio_link = build_note_link(vault_root, audio_path) if audio_path else None
    speaker_lines = "\n".join(
        f"- `{speaker.display_label}`: {speaker.duration_seconds:.1f}s, Rolle `{speaker.role}`"
        for speaker in speakers
    ) or "- "
    transcript_lines = []
    for segment in segments:
        label = segment.display_label or segment.base_label or segment.raw_speaker or "Speaker 1"
        transcript_lines.append(
            f"[{format_timestamp(segment.start)} - {format_timestamp(segment.end)}] {label}:\n{segment.text}"
        )
    transcript_body = "\n\n".join(transcript_lines)

    audio_frontmatter = f'QuelleAudio: "[[{audio_link}]]"\n' if audio_link else ""
    audio_line = f"**Audio:** [[{audio_link}]]" if audio_link else "**Audio:** Nicht verfuegbar"

    return f"""---
Typ: Rohtranskript
Kurs: "{request.course.replace('_', ' ')}"
KursLink: "[[{request.course}]]"
Datum: {request.date}
Sitzungstyp: {request.session_type}
Thema: "{request.theme}"
{audio_frontmatter}---

# Rohtranskript – {request.session_type} – {request.theme}

**Datum:** {request.date}
**Kurs:** [[{request.course}]]
{audio_line}

## Sprecher
{speaker_lines}

## Transkript

{transcript_body}
"""


def render_note_markdown(
    *,
    vault_root: Path,
    request: LectureRequest,
    note_sections: NoteSections,
    transcript_path: Path,
    audio_path: Path | None,
    speaker_count: int,
    transcription_model: str,
    summary_model: str,
    template_path: Path | None = None,
) -> str:
    transcript_link = build_note_link(vault_root, transcript_path)
    audio_link = build_note_link(vault_root, audio_path) if audio_path else None
    context = {
        "title": f"{request.session_type} – {request.theme}",
        "session_type": request.session_type,
        "theme": request.theme,
        "date": request.date,
        "course": request.course,
        "course_display": request.course.replace("_", " "),
        "course_link": f"[[{request.course}]]",
        "audio_link": f"[[{audio_link}]]" if audio_link else "Nicht verfuegbar",
        "transcript_link": f"[[{transcript_link}]]",
        "transcription_model": transcription_model,
        "summary_model": summary_model,
        "speaker_count": str(speaker_count),
        "zusammenfassung": _render_bullets(note_sections.zusammenfassung),
        "notizen": _render_bullets(note_sections.notizen),
        "wichtige_begriffe": _render_term_bullets(note_sections.wichtige_begriffe),
        "pruefungsrelevanz": _render_bullets(note_sections.pruefungsrelevanz),
        "offene_fragen": _render_bullets(note_sections.offene_fragen),
        "naechste_schritte": _render_bullets(note_sections.naechste_schritte),
    }

    resolved_template_path = _resolve_optional_template_path(vault_root, template_path, request)
    if resolved_template_path:
        template = resolved_template_path.read_text(encoding="utf-8")
        return _render_note_template(template, context)

    audio_frontmatter = f'QuelleAudio: "[[{audio_link}]]"\n' if audio_link else ""
    audio_note_line = f"**Audio:** [[{audio_link}]]" if audio_link else "**Audio:** Nicht verfuegbar"
    audio_raw_line = f"- Audio: [[{audio_link}]]" if audio_link else "- Audio: Nicht verfuegbar"

    return f"""---
Typ: Sitzung
Kurs: "{request.course.replace('_', ' ')}"
KursLink: "[[{request.course}]]"
Datum: {request.date}
Sitzungstyp: {request.session_type}
Thema: "{request.theme}"
{audio_frontmatter}TranskriptLink: "[[{transcript_link}]]"
ASRModell: "{transcription_model}"
LLMModell: "{summary_model}"
SpeakerAnzahl: {speaker_count}
ErstelltDurch: "Automatisierung"
---

# {request.session_type} – {request.theme}

**Datum:** {request.date}
**Kurs:** [[{request.course}]]
{audio_note_line}
**Rohtranskript:** [[{transcript_link}]]

## Zusammenfassung
{_render_bullets(note_sections.zusammenfassung)}

## Notizen
{_render_bullets(note_sections.notizen)}

## Wichtige Begriffe
{_render_term_bullets(note_sections.wichtige_begriffe)}

## Pruefungsrelevanz
{_render_bullets(note_sections.pruefungsrelevanz)}

## Offene Fragen
{_render_bullets(note_sections.offene_fragen)}

## Naechste Schritte
{_render_bullets(note_sections.naechste_schritte)}

## Rohmaterial
{audio_raw_line}
- Rohtranskript: [[{transcript_link}]]
"""
