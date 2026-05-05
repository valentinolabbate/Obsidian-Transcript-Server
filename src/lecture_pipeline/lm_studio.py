from __future__ import annotations

import json
from threading import Lock
from typing import Any

import httpx

from .config import Settings
from .models import ChunkSummary, NoteSections
from .prompts import PromptProfile, get_profile
from .utils import extract_json_object

JSON_ONLY_SUFFIX = (
    " Gib nur ein gueltiges JSON-Objekt zurueck. "
    "Schreibe keine Begruendung, kein Markdown und keinen Codeblock."
)
CHUNK_SYSTEM_SUFFIX = JSON_ONLY_SUFFIX
NOTE_SYSTEM_SUFFIX = JSON_ONLY_SUFFIX
_LM_STUDIO_MODEL_LOCK = Lock()
_LM_STUDIO_LOADED_MODEL: str | None = None

CHUNK_USER_TEMPLATE = """/no_think

Kontext:
- Kurs: {course}
- Sitzungstyp: {session_type}
- Thema: {theme}
- Datum: {date}

Gib genau ein gueltiges JSON-Objekt mit diesen Feldern zurueck:
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

NOTE_USER_TEMPLATE = """/no_think

Metadaten:
- Kurs: {course}
- KursLink: [[{course_link}]]
- Datum: {date}
- Sitzungstyp: {session_type}
- Thema: {theme}

Erstelle genau ein gueltiges JSON-Objekt mit exakt diesen Feldern:
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
        timeout = httpx.Timeout(settings.request_timeout_seconds, connect=30.0, write=30.0, pool=30.0)
        self.client = httpx.Client(base_url=settings.lm_studio_base_url, timeout=timeout)
        self.native_client = httpx.Client(
            base_url=_derive_native_base_url(settings.lm_studio_base_url),
            timeout=timeout,
        )
        self._loaded_model: str | None = None

    def close(self) -> None:
        try:
            with _LM_STUDIO_MODEL_LOCK:
                self.unload_loaded_model()
        finally:
            self.client.close()
            self.native_client.close()

    def list_models(self) -> list[str]:
        response = self.client.get("/models")
        response.raise_for_status()
        payload = response.json()
        return [item["id"] for item in payload.get("data", [])]

    def ensure_model_loaded(self) -> str:
        global _LM_STUDIO_LOADED_MODEL
        model = self.profile.lm_studio_model or self.settings.lm_studio_model
        if _LM_STUDIO_LOADED_MODEL == model:
            self._loaded_model = model
            return model
        try:
            response = self.native_client.post("/api/v1/models/load", json={"model": model})
            response.raise_for_status()
        except httpx.HTTPStatusError:
            # LM Studio versions differ in native model-loading support. A failed
            # eager load should not prevent the OpenAI-compatible chat request.
            self._loaded_model = model
            return model
        self._loaded_model = model
        _LM_STUDIO_LOADED_MODEL = model
        return model

    def unload_loaded_model(self) -> None:
        global _LM_STUDIO_LOADED_MODEL
        model = self._loaded_model
        if not model:
            return
        try:
            response = self.native_client.post("/api/v1/models/unload", json={"model": model})
            response.raise_for_status()
        except httpx.HTTPError:
            # Some LM Studio versions do not expose native unload support.
            pass
        finally:
            if _LM_STUDIO_LOADED_MODEL == model:
                _LM_STUDIO_LOADED_MODEL = None
            if self._loaded_model == model:
                self._loaded_model = None

    def chat_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        with _LM_STUDIO_MODEL_LOCK:
            model = self.ensure_model_loaded()
            payload = {
                "model": model,
                "temperature": self.profile.temperature,
                "top_p": self.profile.top_p,
                "messages": messages,
                "stream": True,
            }
            content_parts: list[str] = []
            saw_reasoning = False
            try:
                with self.client.stream("POST", "/chat/completions", json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line.removeprefix("data:").strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        choices = event.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content")
                        if content:
                            content_parts.append(content)
                        if delta.get("reasoning_content"):
                            saw_reasoning = True
            except httpx.HTTPStatusError as exc:
                raise LMStudioRequestError(_format_http_error(exc, action="/chat/completions")) from exc

            content = "".join(content_parts).strip()
            if not content and saw_reasoning:
                raise LMStudioRequestError(
                    "LM Studio hat nur reasoning_content ohne normale Antwort geliefert. "
                    "Deaktiviere Thinking/Reasoning im Modell oder nutze ein Instruct-Modell, "
                    "das JSON in message.content ausgibt."
                )
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
