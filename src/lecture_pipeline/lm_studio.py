from __future__ import annotations

from typing import Any

import httpx

from .config import Settings
from .models import ChunkSummary, NoteSections
from .utils import extract_json_object


def _messages(system: str, user: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


class LMStudioClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = httpx.Client(base_url=settings.lm_studio_base_url, timeout=settings.request_timeout_seconds)

    def close(self) -> None:
        self.client.close()

    def list_models(self) -> list[str]:
        response = self.client.get("/models")
        response.raise_for_status()
        payload = response.json()
        return [item["id"] for item in payload.get("data", [])]

    def chat_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload = {
            "model": self.settings.lm_studio_model,
            "temperature": 0.1,
            "top_p": 0.8,
            "messages": messages,
        }
        response = self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return extract_json_object(content)

    def summarize_chunk(self, *, course: str, session_type: str, theme: str, date: str, chunk_text: str) -> ChunkSummary:
        system = (
            "Du analysierst deutsche Vorlesungstranskripte fuer Obsidian-Notizen. "
            "Arbeite streng quellennah. Erfinde nichts. Speaker 1 ist wahrscheinlich die dozierende Person. "
            "Ignoriere irrelevanten Smalltalk. Gib nur JSON zurueck."
        )
        user = f"""
Kontext:
- Kurs: {course}
- Sitzungstyp: {session_type}
- Thema: {theme}
- Datum: {date}

Gib genau JSON mit diesen Feldern zurueck:
{{
  "kernaussagen": [],
  "begriffe": [{{"begriff": "", "erklaerung": ""}}],
  "pruefungsrelevant": [],
  "offene_fragen": [],
  "beispiele": [],
  "unsicherheiten": []
}}

Transkript:
{chunk_text}
""".strip()
        payload = self.chat_json(_messages(system, user))
        return ChunkSummary.model_validate(payload)

    def synthesize_note(
        self,
        *,
        course: str,
        course_link: str,
        session_type: str,
        theme: str,
        date: str,
        chunk_summaries: list[ChunkSummary],
    ) -> NoteSections:
        system = (
            "Du erstellst praezise deutschsprachige Vorlesungsnotizen fuer Obsidian. "
            "Schreibe sachlich, knapp und fachlich. Erfinde nichts. Gib nur JSON zurueck."
        )
        user = f"""
Metadaten:
- Kurs: {course}
- KursLink: [[{course_link}]]
- Datum: {date}
- Sitzungstyp: {session_type}
- Thema: {theme}

Erstelle JSON mit exakt diesen Feldern:
{{
  "zusammenfassung": [],
  "notizen": [],
  "wichtige_begriffe": [{{"begriff": "", "erklaerung": ""}}],
  "pruefungsrelevanz": [],
  "offene_fragen": [],
  "naechste_schritte": []
}}

Verdichtete Vorarbeit:
{[summary.model_dump() for summary in chunk_summaries]}
""".strip()
        payload = self.chat_json(_messages(system, user))
        return NoteSections.model_validate(payload)
