# Obsidian Transcript Server

Backend fuer die Obsidian-Transcript-GUI. Verwaltet die lokale Pipeline fuer Vorlesungsaufnahmen.

## Funktionen

- Audio aus einer Inbox oder einem beliebigen Pfad uebernehmen
- lokal transkribieren
- Speaker-Diarization lokal ausfuehren
- optional Transkription und Speaker-Diarization parallel ausfuehren
- den laengsten Sprecher als `Speaker 1 (Prof)` markieren
- ueber LM Studio strukturierte Vorlesungsnotizen erzeugen
- verwendete ASR-, Diarization- und LM-Studio-Modelle nach Nutzung entladen
- Rohtranskripte und fertige Sitzungsnotizen in den Vault schreiben

## Installation (automatisch)

Das empfohlene Setup ist ueber das [Obsidian-Transcript-GUI](https://github.com/valentinolabbate/Obsidian-Transcript-GUI)-Plugin. Das Plugin laedt dieses Backend automatisch herunter, erstellt eine Python-Umgebung und installiert alles selbststaendig.

## Manuelle Installation

Falls du das Backend eigenstaendig betreiben moechtest:

```bash
git clone https://github.com/valentinolabbate/Obsidian-Transcript-Server.git
cd Obsidian-Transcript-Server
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e .[audio]
cp .env.example .env
```

## Voraussetzungen

- Python 3.11
- `ffmpeg`
- LM Studio Server auf `http://127.0.0.1:1234/v1`
- optional: `HF_TOKEN` fuer `pyannote.audio`
- Apple Silicon Mac fuer MPS-GPU-Beschleunigung (optional)

## Apple Silicon GPU (MPS)

Die Speaker-Diarization nutzt automatisch die Apple Silicon GPU (MPS), wenn verfuegbar:

- `auto` (Default): MPS wird automatisch erkannt und genutzt
- `mps`: Erzwingt MPS (Apple Silicon GPU)
- `cpu`: Erzwingt CPU-Fallback

Ueber die Umgebungsvariable konfigurierbar:

```bash
export LECTURE_PIPELINE_DIARIZATION_DEVICE=mps  # oder cpu, auto
```

Falls MPS fehlschlaegt, wird automatisch auf CPU zurueckgefallen.

## Parallele Audio-Analyse

Nach der Audio-Vorverarbeitung koennen Transkription und Speaker-Diarization optional parallel laufen. Das veraendert die Ergebnisse nicht, kann aber mehr Speicher und Rechenleistung gleichzeitig beanspruchen. Standard ist deshalb deaktiviert.

```bash
export LECTURE_PIPELINE_PARALLEL_AUDIO_ANALYSIS=true
```

## CLI

Healthcheck:

```bash
lecture-pipeline health
```

End-to-End-Verarbeitung:

```bash
lecture-pipeline process \
  --audio "/Pfad/zur/datei.m4a" \
  --course Oekonometrie \
  --date 2026-04-21 \
  --session-type Vorlesung \
  --theme "Paneldaten und Fixed Effects"
```

API starten:

```bash
lecture-pipeline serve --host 127.0.0.1 --port 8765
```

## API

- `GET /health`
- `POST /process`
- `POST /jobs`
- `GET /jobs/{job_id}`
- `POST /jobs/{job_id}/cancel`
