# Obsidian Lecture Pipeline

Lokale Pipeline fuer Vorlesungsaufnahmen im Obsidian Vault:

- Audio aus einer Inbox oder einem beliebigen Pfad uebernehmen
- lokal transkribieren
- Speaker-Diarization lokal ausfuehren
- den laengsten Sprecher als `Speaker 1 (Prof)` markieren
- ueber LM Studio strukturierte Vorlesungsnotizen erzeugen
- Rohtranskripte und fertige Sitzungsnotizen in den Vault schreiben

## Ordnerkonzept

- Aufnahme-Inbox: `99_Inbox/Audio/`
- Pro Kurs:
  - `Rohdaten/Audio/`
  - `Rohdaten/Transkripte/`
  - `Rohdaten/Jobs/`
  - `Sitzungen/`

## Voraussetzungen

- Python 3.11
- `ffmpeg`
- LM Studio Server auf `http://127.0.0.1:1234/v1`
- optional: `HF_TOKEN` fuer `pyannote.audio`

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e .[audio]
cp .env.example .env
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

Alternativ aus der Inbox heraus:

```bash
lecture-pipeline process \
  --audio "99_Inbox/Audio/deine-datei.m4a" \
  --course Oekonometrie \
  --date 2026-04-21 \
  --session-type Vorlesung \
  --theme "Paneldaten und Fixed Effects"
```

API starten:

```bash
lecture-pipeline serve --host 127.0.0.1 --port 8765
```

## Status

Der aktuelle Stand ist ein robuster Backend-MVP mit:

- Konfiguration
- CLI und HTTP-API
- Rohdatenablage
- LM-Studio-Client
- Markdown-Rendering
- Fallback-Logik fuer fehlende ASR-/Diarization-Abhaengigkeiten

Der naechste Schritt ist ein echter Testlauf mit kurzer Audiodatei nach Installation der Laufzeitabhaengigkeiten.
