from __future__ import annotations

import os
from pathlib import Path
from threading import Event
from threading import Lock, Thread
from time import monotonic, sleep

from fastapi import FastAPI
from fastapi import HTTPException

from .config import settings
from .models import JobListResponse, JobStartResponse, JobStatusSnapshot, LectureRequest
from .pipeline import JobCancelledError, load_job_status, prepare_job, process_lecture
from .storage import write_job_status


app = FastAPI(title="Obsidian Lecture Pipeline")
job_threads: dict[str, Thread] = {}
job_paths: dict[str, str] = {}
job_cancel_events: dict[str, Event] = {}
job_threads_lock = Lock()
activity_lock = Lock()
last_activity_monotonic = monotonic()
watchdog_started = False


def _set_job_thread(job_id: str, thread: Thread | None) -> None:
    with job_threads_lock:
        if thread is None:
            job_threads.pop(job_id, None)
        else:
            job_threads[job_id] = thread


def _set_cancel_event(job_id: str, event: Event | None) -> None:
    with job_threads_lock:
        if event is None:
            job_cancel_events.pop(job_id, None)
        else:
            job_cancel_events[job_id] = event


def _job_cancel_requested(job_id: str) -> bool:
    with job_threads_lock:
        event = job_cancel_events.get(job_id)
    return bool(event and event.is_set())


def _job_is_running(job_id: str) -> bool:
    with job_threads_lock:
        thread = job_threads.get(job_id)
    return bool(thread and thread.is_alive())


def _any_job_running() -> bool:
    with job_threads_lock:
        return any(thread.is_alive() for thread in job_threads.values())


def _touch_activity() -> None:
    global last_activity_monotonic
    with activity_lock:
        last_activity_monotonic = monotonic()


def _seconds_since_last_activity() -> float:
    with activity_lock:
        return monotonic() - last_activity_monotonic


def _idle_watchdog() -> None:
    while True:
        sleep(15)
        idle_seconds = settings.idle_shutdown_seconds
        if idle_seconds <= 0:
            continue
        if _any_job_running():
            continue
        if _seconds_since_last_activity() >= idle_seconds:
            os._exit(0)


def _ensure_watchdog() -> None:
    global watchdog_started
    if watchdog_started:
        return
    thread = Thread(target=_idle_watchdog, daemon=True, name="lecture-pipeline-idle-watchdog")
    thread.start()
    watchdog_started = True


def _run_background_job(request: LectureRequest, job_id: str, created_at, source_audio, source_transcript, paths) -> None:
    try:
        _touch_activity()
        cancel_event = job_cancel_events.get(job_id)
        process_lecture(
            request,
            settings,
            job_id=job_id,
            created_at=created_at,
            source_audio=source_audio,
            source_transcript=source_transcript,
            paths=paths,
            should_cancel=cancel_event.is_set if cancel_event else None,
        )
    except JobCancelledError:
        pass
    finally:
        _set_job_thread(job_id, None)
        _set_cancel_event(job_id, None)


def _load_job_snapshot(job_id: str) -> JobStatusSnapshot | None:
    known_job_path = job_paths.get(job_id)
    payload = load_job_status(job_id, settings, job_path=Path(known_job_path) if known_job_path else None)
    if not payload:
        return None
    payload["is_running"] = _job_is_running(job_id)
    payload["cancellation_requested"] = _job_cancel_requested(job_id)
    return JobStatusSnapshot.model_validate(payload)


@app.get("/health")
def health() -> dict:
    _ensure_watchdog()
    _touch_activity()
    mps_available = False
    try:
        import torch
        mps_available = torch.backends.mps.is_available()
    except ImportError:
        pass
    return {
        "status": "ok",
        "vault_root": str(settings.resolved_vault_root),
        "inbox_dir": str(settings.resolved_inbox_dir),
        "lm_studio_base_url": settings.lm_studio_base_url,
        "lm_studio_model": settings.lm_studio_model,
        "diarization_device": settings.diarization_device,
        "mps_available": mps_available,
        "idle_shutdown_seconds": settings.idle_shutdown_seconds,
    }


@app.post("/process")
def process(request: LectureRequest) -> dict:
    _ensure_watchdog()
    _touch_activity()
    try:
        result = process_lecture(request, settings)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.model_dump(mode="json")


@app.post("/jobs", response_model=JobStartResponse)
def create_job(request: LectureRequest) -> JobStartResponse:
    _ensure_watchdog()
    _touch_activity()
    try:
        job_id, created_at, source_audio, source_transcript, paths = prepare_job(request, settings)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    thread = Thread(
        target=_run_background_job,
        args=(request, job_id, created_at, source_audio, source_transcript, paths),
        daemon=True,
        name=f"lecture-job-{job_id}",
    )
    _set_job_thread(job_id, thread)
    _set_cancel_event(job_id, Event())
    job_paths[job_id] = str(paths.job_path)
    thread.start()
    return JobStartResponse(
        job_id=job_id,
        status="queued",
        stage="queued",
        progress=0,
        message="Job wurde gestartet.",
        job_path=str(paths.job_path),
    )


@app.get("/jobs/{job_id}", response_model=JobStatusSnapshot)
def get_job(job_id: str) -> JobStatusSnapshot:
    _ensure_watchdog()
    _touch_activity()
    snapshot = _load_job_snapshot(job_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail=f"Job nicht gefunden: {job_id}")
    return snapshot


@app.get("/jobs", response_model=JobListResponse)
def list_jobs() -> JobListResponse:
    _ensure_watchdog()
    _touch_activity()
    with job_threads_lock:
        job_ids = list(job_threads.keys())
    snapshots = [snapshot for job_id in job_ids if (snapshot := _load_job_snapshot(job_id))]
    snapshots.sort(key=lambda snapshot: snapshot.created_at.isoformat() if snapshot.created_at else "", reverse=True)
    return JobListResponse(jobs=snapshots)


@app.post("/jobs/{job_id}/cancel", response_model=JobStatusSnapshot)
def cancel_job(job_id: str) -> JobStatusSnapshot:
    _ensure_watchdog()
    _touch_activity()
    snapshot = _load_job_snapshot(job_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail=f"Job nicht gefunden: {job_id}")
    if not snapshot.is_running:
        return snapshot

    with job_threads_lock:
        cancel_event = job_cancel_events.get(job_id)
    if cancel_event is None:
        raise HTTPException(status_code=409, detail="Job kann aktuell nicht abgebrochen werden.")

    cancel_event.set()
    known_job_path = job_paths.get(job_id)
    if known_job_path:
        write_payload = load_job_status(job_id, settings, job_path=Path(known_job_path)) or {}
        write_payload.update(
            {
                "status": "cancelling",
                "message": "Abbruch angefordert. Lauf wird beendet, sobald der aktuelle Schritt abgeschlossen ist.",
                "cancellation_requested": True,
            }
        )
        write_job_status(Path(known_job_path), write_payload)

    updated_snapshot = _load_job_snapshot(job_id)
    if not updated_snapshot:
        raise HTTPException(status_code=404, detail=f"Job nicht gefunden: {job_id}")
    return updated_snapshot
