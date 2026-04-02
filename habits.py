"""
Daily habit log: physical activity, sleep, and mindfulness / calming practice.
Stored in habits.csv (one row per calendar day; re-logging the same day overwrites).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

HABITS_PATH = Path(__file__).resolve().parent / "habits.csv"


def load_habits() -> pd.DataFrame:
    if not HABITS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(HABITS_PATH)


def _dedupe_by_date(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    work = df.copy()
    work["_vd"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["_vd"])
    if "logged_at" in work.columns:
        work["_la"] = pd.to_datetime(work["logged_at"], errors="coerce")
    else:
        work["_la"] = pd.NaT
    work["_row"] = list(range(len(work)))
    work = work.sort_values(["_vd", "_la", "_row"], na_position="first")
    work = work.drop_duplicates(subset=["_vd"], keep="last")
    return work.drop(columns=[c for c in ("_vd", "_la", "_row") if c in work.columns]).reset_index(drop=True)


def load_habits_clean() -> pd.DataFrame:
    """Load and dedupe file; rewrite if duplicates removed."""
    raw = load_habits()
    if raw.empty:
        return raw
    deduped = _dedupe_by_date(raw)
    if len(deduped) < len(raw):
        deduped.to_csv(HABITS_PATH, index=False)
    return deduped


def upsert_habit_day(
    day_iso: str,
    activity_minutes: int,
    sleep_hours: float,
    mindfulness_minutes: int,
    notes: str = "",
) -> None:
    df = load_habits_clean()
    vd = pd.to_datetime(day_iso, errors="coerce").normalize()
    if not df.empty and "date" in df.columns and not pd.isna(vd):
        df = df.copy()
        df["_vd"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df[df["_vd"] != vd]
        df = df.drop(columns=["_vd"], errors="ignore")
    row = {
        "date": day_iso,
        "activity_minutes": activity_minutes,
        "sleep_hours": sleep_hours,
        "mindfulness_minutes": mindfulness_minutes,
        "notes": notes.strip(),
        "logged_at": datetime.now().isoformat(timespec="seconds"),
    }
    out = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    out.to_csv(HABITS_PATH, index=False)
