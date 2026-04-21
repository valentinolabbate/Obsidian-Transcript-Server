from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any


def sanitize_filename_part(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip()
    value = re.sub(r"[\\/:*?\"<>|]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip().strip(".")


def canonical_stem(date: str, session_type: str, theme: str) -> str:
    return f"{date} – {sanitize_filename_part(session_type)} – {sanitize_filename_part(theme)}"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_timestamp(seconds: float) -> str:
    total = max(int(seconds), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _repair_invalid_string_escapes(text: str) -> str:
    repaired: list[str] = []
    in_string = False
    escaped = False
    valid_escapes = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}

    for character in text:
        if not in_string:
            repaired.append(character)
            if character == '"':
                in_string = True
            continue

        if escaped:
            if character not in valid_escapes:
                repaired.append("\\")
            repaired.append(character)
            escaped = False
            continue

        repaired.append(character)
        if character == "\\":
            escaped = True
        elif character == '"':
            in_string = False

    if escaped:
        repaired.append("\\")

    return "".join(repaired)


def _extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError("Kein JSON-Objekt in der Modellantwort gefunden.")

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
            continue
        if character == "{":
            depth += 1
            continue
        if character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("JSON-Objekt in der Modellantwort ist unvollstaendig.")


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    candidates = [text]
    try:
        candidates.append(_extract_first_json_object(text))
    except ValueError:
        pass

    for candidate in candidates:
        for repaired in (candidate, _repair_invalid_string_escapes(candidate)):
            try:
                parsed = json.loads(repaired)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

    raise ValueError("Modellantwort enthielt kein gueltiges JSON-Objekt.")
