from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SessionType(str, Enum):
    VORLESUNG = "Vorlesung"
    UEBUNG = "Uebung"
    TUTORIUM = "Tutorium"


class LectureRequest(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    audio_path: Path | None = None
    transcript_path: Path | None = None
    course: str
    date: str
    session_type: str
    theme: str
    template_path: Path | None = None
    storage_dir: Path | None = None
    output_dir: Path | None = None
    prompt_profile: str = "vorlesung"
    zusammenfassungs_stil: str | None = None
    notiz_stil: str | None = None
    lm_studio_model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    speaker_label_mode: Literal["professor", "generic"] = "professor"

    @model_validator(mode="after")
    def validate_source(self) -> "LectureRequest":
        if self.audio_path or self.transcript_path:
            return self
        raise ValueError("Entweder audio_path oder transcript_path muss gesetzt sein.")


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    raw_speaker: str | None = None
    base_label: str | None = None
    display_label: str | None = None


class SpeakerProfile(BaseModel):
    raw_label: str
    base_label: str
    display_label: str
    role: Literal["professor", "speaker"]
    duration_seconds: float
    reason: str


class ChunkSummary(BaseModel):
    kernaussagen: list[str] = Field(default_factory=list)
    begriffe: list[dict[str, str]] = Field(default_factory=list)
    pruefungsrelevant: list[str] = Field(default_factory=list)
    offene_fragen: list[str] = Field(default_factory=list)
    beispiele: list[str] = Field(default_factory=list)
    unsicherheiten: list[str] = Field(default_factory=list)


class NoteSections(BaseModel):
    zusammenfassung: list[str] = Field(default_factory=list)
    notizen: list[str] = Field(default_factory=list)
    wichtige_begriffe: list[dict[str, str]] = Field(default_factory=list)
    pruefungsrelevanz: list[str] = Field(default_factory=list)
    offene_fragen: list[str] = Field(default_factory=list)
    naechste_schritte: list[str] = Field(default_factory=list)


class PipelinePaths(BaseModel):
    storage_root: Path
    raw_audio_dir: Path
    raw_transcript_dir: Path
    raw_jobs_dir: Path
    output_dir: Path
    canonical_audio_path: Path | None
    transcript_markdown_path: Path
    transcript_json_path: Path
    note_path: Path
    job_path: Path


class PipelineResult(BaseModel):
    job_id: str
    created_at: datetime
    input_audio_path: Path | None
    paths: PipelinePaths
    speakers: list[SpeakerProfile]
    segment_count: int
    note_sections: NoteSections
    summary_model: str
    transcription_model: str


class JobStatusSnapshot(BaseModel):
    job_id: str
    status: str
    stage: str | None = None
    progress: int = 0
    message: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    request: dict | None = None
    source_audio_path: str | None = None
    source_transcript_path: str | None = None
    paths: dict | None = None
    note_path: str | None = None
    transcript_path: str | None = None
    speaker_count: int | None = None
    segment_count: int | None = None
    current_chunk: int | None = None
    total_chunks: int | None = None
    error: str | None = None
    is_running: bool = False
    cancellation_requested: bool = False


class JobStartResponse(BaseModel):
    job_id: str
    status: str
    stage: str | None = None
    progress: int = 0
    message: str | None = None
    job_path: str


class JobListResponse(BaseModel):
    jobs: list[JobStatusSnapshot] = Field(default_factory=list)
