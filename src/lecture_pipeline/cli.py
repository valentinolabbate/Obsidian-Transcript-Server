from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

# ── PATH patch ────────────────────────────────────────────────────────────────
# macOS GUI apps (Obsidian) launch child processes with a minimal PATH that
# omits Homebrew. We extend os.environ['PATH'] here – before any library that
# calls ffmpeg as a subprocess (e.g. mlx_whisper) is imported or executed.
_EXTRA_PATHS = ["/opt/homebrew/bin", "/opt/homebrew/sbin", "/usr/local/bin", "/opt/local/bin"]
_current = os.environ.get("PATH", "")
_missing = [p for p in _EXTRA_PATHS if p not in _current.split(":") and Path(p).is_dir()]
if _missing:
    os.environ["PATH"] = ":".join(_missing) + (":" + _current if _current else "")
# ─────────────────────────────────────────────────────────────────────────────

import typer
import uvicorn

from .config import settings
from .lm_studio import LMStudioClient
from .models import LectureRequest
from .pipeline import process_lecture


app = typer.Typer(no_args_is_help=True)


@app.command()
def health() -> None:
    client = LMStudioClient(settings)
    try:
        models = client.list_models()
    finally:
        client.close()

    mlx_available = importlib.util.find_spec("mlx_whisper") is not None
    pyannote_available = importlib.util.find_spec("pyannote.audio") is not None

    typer.echo(f"Vault root: {settings.resolved_vault_root}")
    typer.echo(f"Inbox: {settings.resolved_inbox_dir}")
    typer.echo(f"FFmpeg: {'ja' if shutil.which('ffmpeg') else 'nein'}")
    typer.echo(f"mlx-whisper: {'ja' if mlx_available else 'nein'}")
    typer.echo(f"pyannote.audio: {'ja' if pyannote_available else 'nein'}")
    typer.echo(f"HF-Token konfiguriert: {'ja' if settings.hf_token else 'nein'}")
    typer.echo(f"LM Studio Modelle: {', '.join(models)}")


@app.command()
def process(
    audio: Path | None = typer.Option(None, exists=False, help="Pfad zur Audio-Datei, absolut oder relativ zum Vault"),
    transcript: Path | None = typer.Option(None, exists=False, help="Pfad zu einem vorhandenen Rohtranskript oder .segments.json"),
    course: str = typer.Option(..., help="Ordnername des Kurses, z. B. Oekonometrie"),
    date: str = typer.Option(..., help="Datum YYYY-MM-DD"),
    session_type: str = typer.Option(..., help="Freier Sitzungstyp, z. B. Vorlesung oder Meeting"),
    theme: str = typer.Option(..., help="Thema der Sitzung"),
    template_path: Path | None = typer.Option(None, help="Optionaler Pfad zu einer Markdown-Vorlage"),
    storage_dir: Path | None = typer.Option(None, help="Optionaler Ordner fuer Audio, Transkripte und Jobs"),
    output_dir: Path | None = typer.Option(None, help="Optionaler Zielordner fuer die finale Notiz"),
) -> None:
    request = LectureRequest(
        audio_path=audio,
        transcript_path=transcript,
        course=course,
        date=date,
        session_type=session_type,
        theme=theme,
        template_path=template_path,
        storage_dir=storage_dir,
        output_dir=output_dir,
    )
    result = process_lecture(request, settings)
    typer.echo(f"Job: {result.job_id}")
    typer.echo(f"Note: {result.paths.note_path}")
    typer.echo(f"Rohtranskript: {result.paths.transcript_markdown_path}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    uvicorn.run("lecture_pipeline.api:app", host=host, port=port, reload=False)
