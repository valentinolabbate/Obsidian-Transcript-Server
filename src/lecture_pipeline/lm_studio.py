from __future__ import annotations

from typing import Any

import httpx

from .config import Settings
from .models import ChunkSummary, NoteSections
from .prompts import PromptProfile, get_profile
from .utils import extract_json_object

CHUNK_SYSTEM_SUFFIX = " Gib nur JSON zurueck."
NOTE_SYSTEM_SUFFIX = " Gib nur JSON zurueck."

CHUNK_USER_TEMPLATE = """Kontext:
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
{chunk_text}"""

NOTE_USER_TEMPLATE = """Metadaten:
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
{chunk_summaries}"""


def _messages(system: str, user: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _derive_native_base_url(openai_base_url: str) -> str:
    if openai_base_url.rstrip("/").endswith("/v1"):
        return openai_base_url.rstrip("/").removesuffix("/v1")
    return openai_base_url.rstrip("/")


class LMStudioRequestError(RuntimeError):
    pass


def _format_http_error(error: httpx.HTTPStatusError, *, action: str) -> str:
    response = error.response
    body = response.text.strip()
    details = f"LM Studio Fehler bei {action}: HTTP {response.status_code} {response.reason_phrase}"
    if body:
        details = f"{details}\n{body}"
    return details


class LMStudioClient:
    def __init__(self, settings: Settings, *, profile: PromptProfile | None = None):
        self.settings = settings
        self.profile = profile or get_profile("vorlesung")
        self.client = httpx.Client(base_url=settings.lm_studio_base_url, timeout=settings.request_timeout_seconds)
        self.native_client = httpx.Client(
            base_url=_derive_native_base_url(settings.lm_studio_base_url),
            timeout=settings.request_timeout_seconds,
        )
        self._loaded_model: str | None = None

    def close(self) -> None:
        self.client.close()
        self.native_client.close()

    def list_models(self) -> list[str]:
        response = self.client.get("/models")
        response.raise_for_status()
        payload = response.json()
        return [item["id"] for item in payload.get("data", [])]

    def ensure_model_loaded(self) -> str:
        model = self.profile.lm_studio_model or self.settings.lm_studio_model
        if self._loaded_model == model:
            return model
        try:
            response = self.native_client.post("/api/v1/models/load", json={"model": model})
            response.raise_for_status()
        except httpx.HTTPStatusError:
            # LM Studio versions differ in native model-loading support. A failed
            # eager load should not prevent the OpenAI-compatible chat request.
            return model
        self._loaded_model = model
        return model

    def chat_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        model = self.ensure_model_loaded()
        payload = {
            "model": model,
            "temperature": self.profile.temperature,
            "top_p": self.profile.top_p,
            "messages": messages,
        }
        try:
            response = self.client.post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LMStudioRequestError(_format_http_error(exc, action="/chat/completions")) from exc
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return extract_json_object(content)

    def summarize_chunk(self, *, course: str, session_type: str, theme: str, date: str, chunk_text: str) -> ChunkSummary:
        system = self.profile.zusammenfassungs_stil + CHUNK_SYSTEM_SUFFIX
        user = CHUNK_USER_TEMPLATE.format(
            course=course,
            session_type=session_type,
            theme=theme,
            date=date,
            chunk_text=chunk_text,
        )
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
        system = self.profile.notiz_stil + NOTE_SYSTEM_SUFFIX
        user = NOTE_USER_TEMPLATE.format(
            course=course,
            course_link=course_link,
            session_type=session_type,
            theme=theme,
            date=date,
            chunk_summaries=[summary.model_dump() for summary in chunk_summaries],
        )
        payload = self.chat_json(_messages(system, user))
        return NoteSections.model_validate(payload)
