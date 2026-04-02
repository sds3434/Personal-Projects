"""
Medication transition log for cross-reference with symptom scale dates.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MEDICATIONS_PATH = Path(__file__).resolve().parent / "medications.csv"

# Seeded from clinical summary: titration completed on this date.
DEFAULT_EVENTS: list[dict] = [
    {
        "date": "2026-03-03",
        "label": "Titration complete: Lexapro → Wellbutrin",
        "prior_regimen": (
            "Escitalopram (Lexapro) 40 mg; "
            "Quetiapine (Seroquel) — same dosage as before; "
            "Buspirone — same dosage as before"
        ),
        "new_regimen": (
            "Bupropion (Wellbutrin) 150 mg; "
            "Buspirone — same dosage as before; "
            "Quetiapine (Seroquel) — same dosage as before"
        ),
    },
]


def load_medications() -> pd.DataFrame:
    if not MEDICATIONS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(MEDICATIONS_PATH)


def save_medications(df: pd.DataFrame) -> None:
    df.to_csv(MEDICATIONS_PATH, index=False)


def append_event(row: dict) -> None:
    df = load_medications()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_medications(df)


def ensure_seed_medications() -> None:
    """If empty, write default titration event so cross-reference works out of the box."""
    if MEDICATIONS_PATH.exists() and len(load_medications()) > 0:
        return
    save_medications(pd.DataFrame(DEFAULT_EVENTS))


def regimen_as_of(survey_date: pd.Timestamp, med_df: pd.DataFrame) -> str:
    """
    Most recent new_regimen from an event on or before survey_date.
    If no events, returns a placeholder.
    """
    if med_df.empty or "date" not in med_df.columns:
        return "— (no medication events on file)"
    m = med_df.copy()
    m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values("date")
    before = m[m["date"] <= survey_date.normalize()]
    if before.empty:
        return "— (before first logged regimen change)"
    row = before.iloc[-1]
    return str(row.get("new_regimen", row.get("prior_regimen", "")))


def cross_reference_sessions(sessions: pd.DataFrame, med_df: pd.DataFrame) -> pd.DataFrame:
    """Join each survey row with the regimen considered active on that date."""
    if sessions.empty:
        return pd.DataFrame()
    out = sessions.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["regimen_active"] = out["date"].apply(lambda d: regimen_as_of(d, med_df))
    return out.sort_values("date")
