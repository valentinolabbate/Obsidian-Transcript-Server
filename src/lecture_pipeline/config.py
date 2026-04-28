from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


MIN_REQUEST_TIMEOUT_SECONDS = 1800.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LECTURE_PIPELINE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    vault_root: Path | None = None
    study_root: str = "10_Studium"
    semester_path: str = "1_Semester_Master_WiWi"
    inbox_dir: str = "99_Inbox/Audio"
    lm_studio_base_url: str = "http://127.0.0.1:1234/v1"
    lm_studio_model: str = "qwen/qwen3.6-35b-a3b"
    transcription_model: str = "mlx-community/whisper-large-v3-turbo"
    diarization_device: str = "auto"  # auto | mps | cpu
    chunk_target_chars: int = 14000
    request_timeout_seconds: float = MIN_REQUEST_TIMEOUT_SECONDS
    idle_shutdown_seconds: int = 900
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")

    @field_validator("request_timeout_seconds")
    @classmethod
    def enforce_minimum_request_timeout(cls, value: float) -> float:
        return max(float(value), MIN_REQUEST_TIMEOUT_SECONDS)

    @property
    def resolved_vault_root(self) -> Path:
        if self.vault_root:
            return self.vault_root.expanduser().resolve()
        return Path(__file__).resolve().parents[4]

    @property
    def resolved_inbox_dir(self) -> Path:
        return (self.resolved_vault_root / self.inbox_dir).resolve()

    @property
    def resolved_study_root(self) -> Path:
        return (self.resolved_vault_root / self.study_root / self.semester_path).resolve()


settings = Settings()
