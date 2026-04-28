from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace


@dataclass
class PromptProfile:
    name: str
    zusammenfassungs_stil: str
    notiz_stil: str
    lm_studio_model: str = ""
    temperature: float = 0.1
    top_p: float = 0.8


DEFAULT_PROFILES: dict[str, PromptProfile] = {
    "vorlesung": PromptProfile(
        name="Vorlesung",
        zusammenfassungs_stil=(
            "Du analysierst deutsche Vorlesungstranskripte fuer Obsidian-Notizen. "
            "Arbeite streng quellennah. Erfinde nichts. Speaker 1 ist wahrscheinlich die dozierende Person. "
            "Ignoriere irrelevanten Smalltalk."
        ),
        notiz_stil=(
            "Du erstellst praezise deutschsprachige Vorlesungsnotizen fuer Obsidian. "
            "Schreibe sachlich, knapp und fachlich. Erfinde nichts."
        ),
    ),
    "kompakt": PromptProfile(
        name="Kompakt",
        zusammenfassungs_stil=(
            "Du fasst deutsche Vorlesungstranskripte kompakt zusammen. "
            "Konzentriere dich auf Kernaussagen und Definitionen. "
            "Ignoriere Smalltalk und Wiederholungen."
        ),
        notiz_stil=(
            "Du erstellst kompakte deutschsprachige Notizen fuer Obsidian. "
            "Sehr kurz und sachlich. Keine Fuellungen."
        ),
        temperature=0.2,
    ),
    "meeting": PromptProfile(
        name="Meeting",
        zusammenfassungs_stil=(
            "Du analysierst deutsche Meeting- oder Besprechungstranskripte. "
            "Fokussiere auf Entscheidungen, Aktionspunkte und Verantwortliche."
        ),
        notiz_stil=(
            "Du erstellst strukturierte deutsche Meeting-Notizen fuer Obsidian. "
            "Hebe Entscheidungen, Aktionspunkte und Verantwortliche hervor."
        ),
        temperature=0.15,
    ),
}


def get_profile(name: str) -> PromptProfile:
    return replace(DEFAULT_PROFILES.get(name, DEFAULT_PROFILES["vorlesung"]))
