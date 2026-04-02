"""
Interactive GAD-7, PHQ-9, and PCL-5 entry with scoring, four-factor clinical snapshot, and charts.
Run: streamlit run app.py
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from instruments import (
    FUNCTIONAL_LABELS,
    GAD7_INTRO,
    GAD7_QUESTIONS,
    LIKERT_0_3,
    LIKERT_0_4,
    PCL5_INTRO,
    PCL5_QUESTIONS,
    PHQ9_INTRO,
    PHQ9_QUESTIONS,
    FourFactorResult,
    four_factor_assessment,
    gad7_severity,
    pcl5_severity,
    phq9_severity,
)
from medications import (
    MEDICATIONS_PATH,
    append_event,
    cross_reference_sessions,
    ensure_seed_medications,
    load_medications,
)
from habits import HABITS_PATH, load_habits_clean, upsert_habit_day
from wisdom import AFFIRMATIONS, BIBLE_VERSES, LAWS, MOTIVATING_QUOTES

DATA_PATH = Path(__file__).resolve().parent / "sessions.csv"


def _init_survey_session() -> None:
    """First load in this browser session: wizard at step 1, no persisted step answers."""
    if "survey_step" not in st.session_state:
        st.session_state.survey_step = 1
    if "survey_session_started" not in st.session_state:
        st.session_state.survey_session_started = True
        for k in (
            "persist_gad",
            "persist_imp_gad",
            "persist_phq",
            "persist_imp_phq",
        ):
            st.session_state.pop(k, None)


def _clear_all_survey_widget_keys() -> None:
    """Remove radio state so a new visit starts with no Likert option selected."""
    for i in range(len(GAD7_QUESTIONS)):
        st.session_state.pop(f"gad_{i}", None)
    for i in range(len(PHQ9_QUESTIONS)):
        st.session_state.pop(f"phq_{i}", None)
    for i in range(len(PCL5_QUESTIONS)):
        st.session_state.pop(f"pcl_{i}", None)
    st.session_state.pop("imp_gad", None)
    st.session_state.pop("imp_phq", None)


def _likert_labels(likert: list[tuple[int, str]]) -> list[str]:
    return [o[1] for o in likert]


def _int_from_likert_label(key: str, likert: list[tuple[int, str]]) -> int | None:
    lab = st.session_state.get(key)
    if lab is None:
        return None
    try:
        return next(o[0] for o in likert if o[1] == lab)
    except StopIteration:
        return None


def _radio_likert(
    question: str,
    likert: list[tuple[int, str]],
    key: str,
    *,
    horizontal: bool = False,
) -> None:
    """Single Likert row with no option pre-selected (`index=None`)."""
    st.radio(
        question,
        _likert_labels(likert),
        key=key,
        index=None,
        horizontal=horizontal,
        label_visibility="visible",
    )


def _render_gad7_step() -> None:
    st.markdown(f"**{GAD7_INTRO}**")
    for i, q in enumerate(GAD7_QUESTIONS):
        _radio_likert(q, LIKERT_0_3, key=f"gad_{i}")
    st.markdown(
        "**If you checked any problems, how difficult have they made it for you to do your work, "
        "take care of things at home, or get along with other people?**"
    )
    _radio_likert(
        "Functional impairment (GAD-7 follow-up)",
        FUNCTIONAL_LABELS,
        key="imp_gad",
        horizontal=True,
    )


def _render_phq9_step() -> None:
    st.markdown(f"**{PHQ9_INTRO}**")
    for i, q in enumerate(PHQ9_QUESTIONS):
        _radio_likert(q, LIKERT_0_3, key=f"phq_{i}")
    st.markdown(
        "**If you checked off any problems, how difficult have these problems made it for you to do your work, "
        "take care of things at home, or get along with other people?**"
    )
    _radio_likert(
        "Functional impairment (PHQ-9 follow-up)",
        FUNCTIONAL_LABELS,
        key="imp_phq",
        horizontal=True,
    )


def _render_pcl5_step() -> None:
    st.markdown(f"**{PCL5_INTRO}**")
    for i, q in enumerate(PCL5_QUESTIONS):
        _radio_likert(q, LIKERT_0_4, key=f"pcl_{i}")


def _step1_complete() -> bool:
    for i in range(len(GAD7_QUESTIONS)):
        if _int_from_likert_label(f"gad_{i}", LIKERT_0_3) is None:
            return False
    return _int_from_likert_label("imp_gad", FUNCTIONAL_LABELS) is not None


def _step2_complete() -> bool:
    for i in range(len(PHQ9_QUESTIONS)):
        if _int_from_likert_label(f"phq_{i}", LIKERT_0_3) is None:
            return False
    return _int_from_likert_label("imp_phq", FUNCTIONAL_LABELS) is not None


def _step3_complete() -> bool:
    for i in range(len(PCL5_QUESTIONS)):
        if _int_from_likert_label(f"pcl_{i}", LIKERT_0_4) is None:
            return False
    return True


def _persist_step1_and_advance() -> None:
    st.session_state["persist_gad"] = [
        _int_from_likert_label(f"gad_{i}", LIKERT_0_3) for i in range(len(GAD7_QUESTIONS))
    ]
    st.session_state["persist_imp_gad"] = _int_from_likert_label("imp_gad", FUNCTIONAL_LABELS)
    st.session_state.survey_step = 2


def _persist_step2_and_advance() -> None:
    st.session_state["persist_phq"] = [
        _int_from_likert_label(f"phq_{i}", LIKERT_0_3) for i in range(len(PHQ9_QUESTIONS))
    ]
    st.session_state["persist_imp_phq"] = _int_from_likert_label("imp_phq", FUNCTIONAL_LABELS)
    st.session_state.survey_step = 3


def _dedupe_sessions_by_visit_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per calendar visit date: keep the row with the latest scored_at (else latest file order).
    Fixes duplicate rows from re-clicking Score on the same date.
    """
    if df.empty or "date" not in df.columns:
        return df
    work = df.copy()
    work["_vd"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["_vd"])
    if work.empty:
        return df
    if "scored_at" in work.columns:
        work["_sa"] = pd.to_datetime(work["scored_at"], errors="coerce")
    else:
        work["_sa"] = pd.NaT
    work["_row"] = list(range(len(work)))
    work = work.sort_values(["_vd", "_sa", "_row"], na_position="first")
    work = work.drop_duplicates(subset=["_vd"], keep="last")
    drop_cols = [c for c in ("_vd", "_sa", "_row") if c in work.columns]
    return work.drop(columns=drop_cols).reset_index(drop=True)


def _load_sessions() -> pd.DataFrame:
    """Read sessions; enforce one row per visit date and rewrite CSV if duplicates were present."""
    if not DATA_PATH.exists():
        return pd.DataFrame()
    raw = pd.read_csv(DATA_PATH)
    deduped = _dedupe_sessions_by_visit_date(raw)
    if len(deduped) < len(raw):
        deduped.to_csv(DATA_PATH, index=False)
    return deduped


def _save_session(row: dict) -> None:
    """Upsert by visit date: re-scoring the same date replaces that row instead of appending."""
    df = _load_sessions()
    vd = pd.to_datetime(row.get("date"), errors="coerce")
    if not pd.isna(vd):
        vd = vd.normalize()
        df = df.copy()
        df["_vd"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df[df["_vd"] != vd]
        df = df.drop(columns=["_vd"], errors="ignore")
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)


def _optional_int_for_csv(val) -> int | None:
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def _format_visit_datetime(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return ""
    if ts.hour or ts.minute or ts.second:
        return ts.strftime("%Y-%m-%d %H:%M")
    return ts.strftime("%Y-%m-%d")


def _phq9_item9_from_saved_row(row: pd.Series) -> int:
    """Item 9 must match the stored PHQ-9 item list (source of truth), not only the denormalized column."""
    raw = row.get("phq9_items")
    if raw is not None and not (isinstance(raw, float) and pd.isna(raw)):
        try:
            items = json.loads(str(raw))
            if isinstance(items, list) and len(items) >= 9:
                return int(items[8])
        except (json.JSONDecodeError, ValueError, TypeError, IndexError):
            pass
    v = row.get("phq9_item9", 0)
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return 0


def sync_last_scored_from_dataset() -> None:
    """
    Set the four-factor snapshot from the latest visit **by calendar date** (then scored_at).
    PHQ-9 item 9 is taken from `phq9_items` JSON when present so flags match the saved responses.
    """
    df = _load_sessions()
    if df.empty:
        st.session_state.pop("last_scored", None)
        return
    try:
        work = df.copy()
        work["_row"] = list(range(len(work)))
        # Normalize visit dates so ordering is unambiguous (YYYY-MM-DD from the app; avoid MM/DD ambiguity).
        work["_vd"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
        if "scored_at" in work.columns:
            work["scored_at"] = pd.to_datetime(work["scored_at"], errors="coerce")
        else:
            work["scored_at"] = pd.NaT
        work = work.dropna(subset=["_vd"])
        if work.empty:
            st.session_state.pop("last_scored", None)
            return
        uds = sorted(work["_vd"].dropna().unique())
        latest_day = uds[-1]
        candidates = work[work["_vd"] == latest_day].sort_values(["scored_at", "_row"], na_position="first")
        row = candidates.iloc[-1]

        prev_visit_gad = prev_visit_phq = prev_visit_pcl = None
        prev_date_lbl = None
        if len(uds) >= 2:
            prev_day = uds[-2]
            prev_candidates = work[work["_vd"] == prev_day].sort_values(
                ["scored_at", "_row"], na_position="first"
            )
            prev_row = prev_candidates.iloc[-1]
            prev_visit_gad = int(prev_row["gad7"])
            prev_visit_phq = int(prev_row["phq9"])
            prev_visit_pcl = int(prev_row["pcl5"])
            prev_date_lbl = _format_visit_datetime(pd.Timestamp(prev_row["date"]))

        gad = int(row["gad7"])
        phq = int(row["phq9"])
        pcl = int(row["pcl5"])
        ig = int(row.get("impairment_gad", 0))
        ip = int(row.get("impairment_phq", 0))
        q9 = _phq9_item9_from_saved_row(row)
        pg = _optional_int_for_csv(row.get("prior_gad7"))
        pp = _optional_int_for_csv(row.get("prior_phq9"))
        pl = _optional_int_for_csv(row.get("prior_pcl5"))
        cur_date_lbl = _format_visit_datetime(pd.Timestamp(row["date"]))
        ff = four_factor_assessment(
            gad,
            phq,
            pcl,
            ig,
            ip,
            q9,
            prior_gad7=pg,
            prior_phq9=pp,
            prior_pcl5=pl,
            prev_visit_gad7=prev_visit_gad,
            prev_visit_phq9=prev_visit_phq,
            prev_visit_pcl5=prev_visit_pcl,
            prev_visit_date_label=prev_date_lbl,
            current_visit_date_label=cur_date_lbl,
        )
        visit_ts = pd.Timestamp(row["date"])
        scored_raw = row.get("scored_at")
        scored_label = None
        if scored_raw is not None and not (isinstance(scored_raw, float) and pd.isna(scored_raw)):
            st_s = pd.to_datetime(scored_raw, errors="coerce")
            if not pd.isna(st_s):
                scored_label = st_s.strftime("%Y-%m-%d %H:%M:%S")

        st.session_state["last_scored"] = {
            "visit_date": _format_visit_datetime(visit_ts),
            "gad_total": gad,
            "phq_total": phq,
            "pcl_total": pcl,
            "ff": ff,
            "scored_at": scored_label,
        }
    except Exception:
        pass


def _figure_scorebars(gad7: int, phq9: int, pcl5: int) -> plt.Figure:
    """
    One horizontal panel per scale, each with its own 0–max axis and shaded severity bands.
    Raw totals are not comparable across scales; this avoids a single misleading 0–80 axis.
    """
    # (lo, hi) half-open style boundaries between common band cut points
    gad_bands = [
        (0, 4.5, "#e8f5e9", "Minimal"),
        (4.5, 9.5, "#fffde7", "Mild"),
        (9.5, 14.5, "#fff3e0", "Moderate"),
        (14.5, 21.01, "#ffebee", "Severe"),
    ]
    phq_bands = [
        (0, 4.5, "#e8f5e9", "None–minimal"),
        (4.5, 9.5, "#fffde7", "Mild"),
        (9.5, 14.5, "#fff3e0", "Moderate"),
        (14.5, 19.5, "#ffe0b2", "Mod.–severe"),
        (19.5, 27.01, "#ffebee", "Severe"),
    ]
    pcl_bands = [
        (0, 31, "#e8f5e9", "Below ~31–33 cutoff"),
        (31, 40, "#fffde7", "Borderline / elevated"),
        (40, 60, "#fff3e0", "Elevated"),
        (60, 80.01, "#ffebee", "High"),
    ]

    panels = [
        ("GAD-7 (max 21)", gad7, 21, gad_bands, gad7_severity(gad7), "#1b5e20", 10),
        ("PHQ-9 (max 27)", phq9, 27, phq_bands, phq9_severity(phq9), "#0d47a1", 10),
        ("PCL-5 (max 80)", pcl5, 80, pcl_bands, pcl5_severity(pcl5), "#4a148c", 33),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(9, 6.5), gridspec_kw={"hspace": 0.45})
    fig.suptitle(
        "This visit — where your total falls on each scale",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    for ax, (title, score, xmax, bands, severity_label, bar_color, ref_line) in zip(axes, panels):
        for lo, hi, color, _name in bands:
            ax.axvspan(lo, min(hi, xmax), facecolor=color, edgecolor="none", zorder=0, alpha=1.0)
        ref_label = "≈10 (screening)" if ref_line == 10 else f"≈{ref_line} (common cutoff)"
        ax.axvline(
            ref_line,
            color="#424242",
            linestyle="--",
            linewidth=1.1,
            alpha=0.8,
            zorder=1,
        )
        ax.barh(
            0,
            min(score, xmax),
            height=0.42,
            left=0,
            color=bar_color,
            edgecolor="black",
            linewidth=0.9,
            zorder=3,
            alpha=0.92,
        )
        ax.plot(score, 0, marker="D", color="white", markersize=9, markeredgecolor="black", zorder=4)
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.55, 0.55)
        ax.set_yticks([0])
        ax.set_yticklabels([title], fontsize=10, fontweight="bold")
        ax.tick_params(axis="y", length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        step = max(1, int(xmax) // 8)
        ax.set_xticks(list(range(0, int(xmax) + 1, step)))
        tx = min(score + xmax * 0.02, xmax * 0.98)
        ax.text(
            tx,
            0.36,
            f"{score}  ·  {severity_label}",
            ha="right",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#212121",
            zorder=5,
        )
        ax.text(
            ref_line,
            -0.42,
            ref_label,
            ha="center",
            va="top",
            fontsize=7,
            color="#616161",
            zorder=2,
        )

    axes[-1].set_xlabel("Score (each row is its own scale — do not compare bar lengths across rows)")
    fig.text(
        0.5,
        0.02,
        "Background: lighter = lower typical severity band for that instrument; dashed line = common screening/cutoff. "
        "PCL-5 bands are approximate.",
        ha="center",
        fontsize=8,
        style="italic",
        color="#555555",
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    return fig


def _figure_habits_trends(h: pd.DataFrame) -> plt.Figure:
    """Three stacked panels: activity minutes, sleep hours, mindfulness minutes over time."""
    d = h.copy()
    d["date"] = pd.to_datetime(d["date"])
    for col in ("activity_minutes", "sleep_hours", "mindfulness_minutes"):
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.sort_values("date")
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"hspace": 0.35})
    fig.suptitle("Daily habits over time", fontsize=12, fontweight="bold", y=0.995)

    axes[0].plot(d["date"], d["activity_minutes"], color="#2e7d32", marker="o", linewidth=1.5)
    axes[0].set_ylabel("Minutes", fontsize=9)
    axes[0].set_title("Physical activity (moderate +)", fontsize=10, loc="left", color="#1b5e20")

    axes[1].plot(d["date"], d["sleep_hours"], color="#1565c0", marker="o", linewidth=1.5)
    axes[1].set_ylabel("Hours", fontsize=9)
    axes[1].set_title("Sleep (last night, approximate)", fontsize=10, loc="left", color="#0d47a1")

    axes[2].plot(d["date"], d["mindfulness_minutes"], color="#6a1b9a", marker="o", linewidth=1.5)
    axes[2].set_ylabel("Minutes", fontsize=9)
    axes[2].set_title(
        "Mindfulness / calming practice (meditation, breathing, grounding)",
        fontsize=10,
        loc="left",
        color="#4a148c",
    )
    axes[2].set_xlabel("Date")

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def _ax_severity_background_gad(ax) -> None:
    ax.axhspan(0, 4.5, facecolor="#e8f5e9", zorder=0, alpha=0.85)
    ax.axhspan(4.5, 9.5, facecolor="#fffde7", zorder=0, alpha=0.85)
    ax.axhspan(9.5, 21, facecolor="#ffebee", zorder=0, alpha=0.5)


def _ax_severity_background_phq(ax) -> None:
    ax.axhspan(0, 4.5, facecolor="#e8f5e9", zorder=0, alpha=0.85)
    ax.axhspan(4.5, 9.5, facecolor="#fffde7", zorder=0, alpha=0.85)
    ax.axhspan(9.5, 14.5, facecolor="#fff3e0", zorder=0, alpha=0.7)
    ax.axhspan(14.5, 27, facecolor="#ffebee", zorder=0, alpha=0.55)


def _ax_severity_background_pcl(ax) -> None:
    ax.axhspan(0, 31, facecolor="#e8f5e9", zorder=0, alpha=0.75)
    ax.axhspan(31, 80, facecolor="#fff3e0", zorder=0, alpha=0.45)


def _draw_med_vlines(axes, med_df: pd.DataFrame | None, ylims: list[tuple[float, float]]) -> None:
    if med_df is None or med_df.empty or "date" not in med_df.columns:
        return
    m = med_df.copy()
    m["date"] = pd.to_datetime(m["date"])
    for ax, (y0, y1) in zip(axes, ylims):
        for _, row in m.iterrows():
            d = row["date"]
            ax.axvline(d, color="#c0392b", linestyle="--", alpha=0.5, linewidth=1.2, zorder=1)
            lbl = str(row.get("label", "Med change"))[:38]
            ax.text(
                d,
                y1 * 0.98,
                f" {lbl}",
                rotation=90,
                verticalalignment="top",
                fontsize=6.5,
                color="#7b241c",
                clip_on=False,
            )


def _figure_history(h: pd.DataFrame, med_df: pd.DataFrame | None = None) -> plt.Figure:
    """
    Three stacked panels — each scale uses its real range (not a shared 0–80 axis).
    Background shading = common severity bands; dashed horizontal = screening/cutoff lines.
    """
    d = h.copy()
    d["date"] = pd.to_datetime(d["date"])
    for col in ("gad7", "phq9", "pcl5"):
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.sort_values("date")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8.2), sharex=True, gridspec_kw={"hspace": 0.28})
    fig.suptitle(
        "Symptom trajectories — each row is its own scale (lower is generally better)",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )

    # GAD-7
    _ax_severity_background_gad(axes[0])
    axes[0].plot(d["date"], d["gad7"], color="#1b5e20", marker="o", linewidth=2, markersize=7, zorder=3)
    axes[0].axhline(10, color="#424242", linestyle="--", linewidth=1, alpha=0.85, zorder=2)
    axes[0].set_ylim(0, 21)
    axes[0].set_ylabel("GAD-7 total", fontsize=9)
    axes[0].set_title("Anxiety (GAD-7, 0–21) — dashed line ≈10 (common screening)", fontsize=9, loc="left")

    # PHQ-9
    _ax_severity_background_phq(axes[1])
    axes[1].plot(d["date"], d["phq9"], color="#0d47a1", marker="o", linewidth=2, markersize=7, zorder=3)
    axes[1].axhline(10, color="#424242", linestyle="--", linewidth=1, alpha=0.85, zorder=2)
    axes[1].set_ylim(0, 27)
    axes[1].set_ylabel("PHQ-9 total", fontsize=9)
    axes[1].set_title("Depression (PHQ-9, 0–27) — dashed line ≈10 (common screening)", fontsize=9, loc="left")

    # PCL-5
    _ax_severity_background_pcl(axes[2])
    axes[2].plot(d["date"], d["pcl5"], color="#4a148c", marker="o", linewidth=2, markersize=7, zorder=3)
    axes[2].axhline(33, color="#424242", linestyle="--", linewidth=1, alpha=0.85, zorder=2)
    axes[2].set_ylim(0, 80)
    axes[2].set_ylabel("PCL-5 total", fontsize=9)
    axes[2].set_xlabel("Survey date")
    axes[2].set_title(
        "PTSD symptoms (PCL-5, 0–80) — dashed line ≈33 (common probable-PTSD context)",
        fontsize=9,
        loc="left",
    )

    ylims = [(0, 21), (0, 27), (0, 80)]
    _draw_med_vlines(axes, med_df, ylims)

    for ax in axes:
        ax.grid(True, alpha=0.25, zorder=2)
    fig.autofmt_xdate()
    fig.text(
        0.5,
        0.02,
        "Red vertical lines: medication log events (interpret symptom changes alongside dose changes and your clinician’s timeline).",
        ha="center",
        fontsize=8,
        style="italic",
        color="#555555",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    return fig


def _history_clinical_summary(h: pd.DataFrame, med_df: pd.DataFrame | None) -> str:
    """Short patient-facing interpretation for therapy/psychiatry discussion — not a diagnosis."""
    d = h.copy()
    d["date"] = pd.to_datetime(d["date"])
    for col in ("gad7", "phq9", "pcl5"):
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.sort_values("date")
    n = len(d)
    lines: list[str] = []

    def _trend_block(col: str, label: str) -> str:
        s = d[col].dropna()
        if len(s) < 2:
            return f"**{label}:** need at least two visits to describe a trend."
        first, last = float(s.iloc[0]), float(s.iloc[-1])
        delta = last - first
        improving = delta < -1.5
        worsening = delta > 1.5
        stable = not improving and not worsening

        extra_thresh = ""
        if col in ("gad7", "phq9"):
            if last < 5 and first >= 5:
                extra_thresh = " **Threshold note:** you now sit in the **minimal-symptom** band (roughly 0–4) on this form compared with before."
            elif last >= 10 and first < 10:
                extra_thresh = " **Threshold note:** the total moved **above** the common **~10** screening line — worth a focused check-in."
            elif last < 10 and first >= 10:
                extra_thresh = " **Threshold note:** the total fell **below ~10** — still put **context** (sleep, stress, meds) next to the number."
        elif col == "pcl5":
            if last < 33 and first >= 33:
                extra_thresh = (
                    " **Threshold note:** the total crossed **below** the **~31–33** range often used as a **probable PTSD** screen in research — "
                    "your clinician still interprets this with your full history (the PCL-5 is one piece of the picture)."
                )
            elif last >= 33 and first < 33:
                extra_thresh = (
                    " **Threshold note:** the total rose **above** that **~33** context — discuss **which** symptom clusters worsened "
                    "(intrusions, avoidance, mood/cognition, arousal) and whether mood or sleep might be driving the shift."
                )

        if col == "gad7":
            if improving:
                body = (
                    "**Direction:** worry and tension symptoms on this measure look **less intense** at your latest visit than at the first. "
                    "GAD-7 tracks **generalized** anxiety (restlessness, trouble relaxing, “what-if” worry). "
                    "**Depth:** improvement often tracks better sleep, lower caffeine or stimulant load, more predictable routines, or skills you practiced between visits — "
                    "but a better score doesn’t mean you should stop naming anxiety in therapy if it still shows up in relationships or work."
                )
            elif worsening:
                body = (
                    "**Direction:** this anxiety snapshot is **higher** now than at your first visit in this chart. "
                    "**Depth:** spikes often track **life stress**, **poor sleep**, **caffeine**, **medication timing**, or **health anxiety** — "
                    "not necessarily that you are “going backward” as a person. Bring **what changed in your week** as much as the number."
                )
            else:
                body = (
                    "**Direction:** scores are **about the same** across visits — neither a clear gain nor a clear loss on paper. "
                    "**Depth:** stability can mean your baseline is **predictable** (useful for planning) or that symptoms are **stuck** — "
                    "your therapist can help tell the difference and adjust skills or meds if needed."
                )

        elif col == "phq9":
            if improving:
                body = (
                    "**Direction:** depressive symptoms on this measure look **lighter** at the latest visit than at the first. "
                    "PHQ-9 covers mood, interest, sleep, energy, appetite, self-criticism, concentration, psychomotor changes, and suicidal thoughts — "
                    "a lower total usually means **several** of those channels are less active, not that “everything is fine.” "
                    "**Depth:** meaningful improvement still deserves **ongoing safety check-ins** (especially if sleep or hope dipped recently)."
                )
            elif worsening:
                body = (
                    "**Direction:** depressive burden on this form is **higher** now than at your first plotted visit. "
                    "**Depth:** depression scores rise with **sleep debt**, **pain**, **isolation**, **alcohol/cannabis**, **thyroid or other medical issues**, "
                    "and **stress** — not only with “willpower.” If item 9 (self-harm thoughts) is worse, treat that as urgent to discuss, not as a moral failure."
                )
            else:
                body = (
                    "**Direction:** PHQ-9 totals are **roughly flat** across these visits. "
                    "**Depth:** “plateau” can mean symptoms are **stubborn**, or that you are **holding steady** through a hard life season — "
                    "the chart can’t see **function** (work, parenting, connection); tell your team what the week actually felt like."
                )

        else:  # pcl5
            if improving:
                body = (
                    "**Direction:** trauma-related symptom burden on the PCL-5 is **lower** at your latest visit than at the first in this chart. "
                    "**What this scale actually measures:** intrusions and nightmares, avoidance, negative beliefs and mood, detachment, "
                    "irritability, risky behavior, hypervigilance, concentration, and sleep — the **total** does not say *which* cluster moved; "
                    "your clinician may look item-by-item. "
                    "**Depth:** some people feel relief as **avoidance** drops; others as **sleep** or **mood** stabilizes — improvement is still real if it matches your day-to-day life. "
                    "Watch for **temporary bumps** around anniversaries, court dates, medical stress, or med changes; PTSD trajectories are rarely a straight line."
                )
            elif worsening:
                body = (
                    "**Direction:** trauma-related symptoms are **more endorsed** now than at your first visit here. "
                    "**Depth:** flares often track **new reminders**, **loss of safety**, **depression or panic** overlapping the same nervous system, "
                    "**substance use**, **sleep collapse**, or **pausing therapy skills** — not proof that you are “not trying.” "
                    "Ask your team whether **trauma-focused work**, **medication**, **sleep**, or **safety planning** needs to be the next focus."
                )
            else:
                body = (
                    "**Direction:** PCL-5 totals are **about the same** across visits — a mixed picture. "
                    "**Depth:** stability can mean symptoms are **chronic but contained**, or that **some** areas improved while **others** worsened and the total cancelled out. "
                    "Because PTSD symptoms interact heavily with **depression** and **anxiety**, your therapist may compare this trend with your PHQ-9 and GAD-7, not read PCL-5 alone."
                )

        return f"**{label}**\n\n{body}{extra_thresh}"

    if n >= 2:
        lines.append(_trend_block("gad7", "GAD-7 — generalized anxiety"))
        lines.append(_trend_block("phq9", "PHQ-9 — depression & related symptoms"))
        lines.append(_trend_block("pcl5", "PCL-5 — trauma-related symptoms (PTSD symptom clusters)"))
    else:
        lines.append(
            "You have **one** saved survey so far. After another **dated** visit, this box will summarize "
            "whether each scale moved up, down, or stayed stable — and note crossing common screening lines."
        )

    lines.append(
        "**How to use this:** Bring the chart to therapy or psychiatry — **trends** and **context** matter as much as one score. "
        "Worsening does not automatically mean treatment “failed” (stress, sleep, alcohol/cannabis, and how long a med has been at a dose all matter)."
    )

    if med_df is not None and not med_df.empty and "date" in med_df.columns and n >= 2:
        m = med_df.copy()
        m["date"] = pd.to_datetime(m["date"])
        t0, t1 = d["date"].min(), d["date"].max()
        between = m[(m["date"] > t0) & (m["date"] < t1)]
        if not between.empty:
            lines.append(
                "**Medication timing:** A regimen change falls **between** some of your survey dates. "
                "Many psychiatric meds need **several weeks** to judge; your prescriber can help interpret symptoms alongside dose changes."
            )

    return "\n\n".join(lines)


def _render_scored_visit(last: dict) -> None:
    visit_date_str = last["visit_date"]
    gad_total = last["gad_total"]
    phq_total = last["phq_total"]
    pcl_total = last["pcl_total"]
    ff: FourFactorResult = last["ff"]
    st.divider()
    st.subheader(f"Results for **{visit_date_str}**")
    st.caption(
        "Snapshot uses the **most recent saved visit** in "
        f"`{DATA_PATH.name}`. With **two or more** visits, totals are compared to the **previous** saved survey date."
    )
    if last.get("scored_at"):
        st.caption(f"Row recorded at: **{last['scored_at']}**")
    st.subheader("Totals and severity labels")
    c1, c2, c3 = st.columns(3)
    c1.metric("GAD-7", gad_total, help="0–21")
    c1.caption(gad7_severity(gad_total))
    c2.metric("PHQ-9", phq_total, help="0–27")
    c2.caption(phq9_severity(phq_total))
    c3.metric("PCL-5", pcl_total, help="0–80")
    c3.caption(pcl5_severity(pcl_total))

    st.subheader("Four-factor clinical snapshot")
    st.info(
        "**Dominant signal (automated heuristic):** "
        + {
            "positive": "Positive / on track (relative to mild thresholds)",
            "partial": "Trend / partial response (vs prior saved visit or sidebar priors)",
            "uncertain": "Moderate burden / monitor (mixed picture)",
            "revisit": "Escalate or review soon (severity, impairment, and/or safety)",
        }[ff.dominant]
    )
    st.markdown("##### Prior saved survey vs current (most recent)")
    st.markdown(ff.comparison_block)
    st.markdown(ff.trend_vs_prior_survey)
    c_s, c_w = st.columns(2)
    with c_s:
        st.markdown("**Steps to discuss with your clinician** *(not medical orders)*")
        st.markdown(ff.suggested_steps)
    with c_w:
        st.markdown("**What to watch for until next visit**")
        st.markdown(ff.watch_for)
    st.caption(
        "These prompts are educational; your prescriber or therapist personalizes care. "
        "If you might harm yourself or others, seek emergency help immediately."
    )
    with st.expander("1 — Positive / on track", expanded=False):
        st.write(ff.positive)
    with st.expander("2 — Trend vs prior survey / plateau"):
        st.write(ff.partial)
    with st.expander("3 — Unclear / plateau or mixed"):
        st.write(ff.uncertain)
    with st.expander("4 — Something to revisit / safety & supports"):
        st.write(ff.revisit)
        if ff.phq9_item9_flag:
            st.error(
                "PHQ-9 item 9 endorsed: if you are in crisis, contact local emergency services "
                "or a crisis line (e.g. 988 in the U.S.)."
            )

    st.subheader("Charts")
    fig = _figure_scorebars(gad_total, phq_total, pcl_total)
    st.pyplot(fig)
    plt.close(fig)

    hist = _load_sessions()
    med_df = load_medications()
    if not hist.empty:
        st.subheader("Symptom history & medication context")
        st.caption(
            "Each scale has its **own** score range and severity shading. **Lower** totals usually mean fewer "
            "symptoms on that measure. Use this with your therapist or psychiatrist — it does not replace clinical judgment."
        )
        try:
            fig2 = _figure_history(hist, med_df)
            st.pyplot(fig2)
            plt.close(fig2)
        except Exception:
            st.caption("Could not plot history (check CSV format).")
        try:
            st.markdown("##### How to read your trend (for discussion in session)")
            st.info(_history_clinical_summary(hist, med_df))
        except Exception:
            pass
        if len(hist) < 2:
            st.caption("After a second dated visit, the text above will compare first vs most recent survey.")
        try:
            cr = cross_reference_sessions(hist, med_df)
            if not cr.empty:
                st.subheader("Which regimen was active on each survey date")
                st.caption(
                    "Each row shows the **new_regimen** from your medication log that was in effect on or before that survey "
                    "(your prescriber confirms what you were actually taking)."
                )
                show_cols = ["date", "gad7", "phq9", "pcl5", "regimen_active"]
                avail = [c for c in show_cols if c in cr.columns]
                disp = cr[avail].copy()
                if "date" in disp.columns:
                    disp["date"] = pd.to_datetime(disp["date"]).dt.strftime("%Y-%m-%d")
                st.dataframe(disp, use_container_width=True, hide_index=True)
        except Exception:
            pass


def main() -> None:
    st.set_page_config(page_title="GAD-7 · PHQ-9 · PCL-5", layout="wide")
    ensure_seed_medications()
    sync_last_scored_from_dataset()
    _init_survey_session()

    st.title("Symptom scales — single visit")
    st.caption(
        "Educational / self-monitoring tool. Not a diagnosis. Discuss results with a licensed clinician."
    )

    with st.sidebar:
        sess_df = _load_sessions()
        st.caption(
            f"**Dataset:** `{DATA_PATH.name}` — **{len(sess_df)}** saved visit(s). "
            "Scores append to this file and reload automatically."
        )
        if not sess_df.empty:
            preview = sess_df.copy()
            if "date" in preview.columns:
                preview["date"] = pd.to_datetime(preview["date"]).dt.strftime("%Y-%m-%d")
            cols = [c for c in ["date", "gad7", "phq9", "pcl5"] if c in preview.columns]
            with st.expander("All saved visits", expanded=False):
                st.dataframe(preview[cols], use_container_width=True, hide_index=True)

        st.header("Visit")
        visit_date: date = st.date_input("Date", value=date.today())
        st.subheader("Optional: prior totals")
        st.caption("Used for trend language in the four-factor panel.")
        use_prior = st.checkbox("I have scores from a prior date", value=False)
        prior_gad = prior_phq = prior_pcl = None
        if use_prior:
            prior_gad = st.number_input("Prior GAD-7 total", min_value=0, max_value=21, value=10)
            prior_phq = st.number_input("Prior PHQ-9 total", min_value=0, max_value=27, value=12)
            prior_pcl = st.number_input("Prior PCL-5 total", min_value=0, max_value=80, value=30)

        st.divider()
        with st.expander("Affirmations, quotes & laws", expanded=False):
            st.caption("General perspective — not medical advice.")
            st.markdown("### Affirmations")
            for line in AFFIRMATIONS:
                st.markdown(f"- {line}")
            st.markdown("### Motivating quotes")
            for line in MOTIVATING_QUOTES:
                st.markdown(f"- {line}")
            st.markdown("### Bible (WEB) — anxiety, depression, trauma")
            st.caption("World English Bible (public domain). For spiritual support alongside professional care.")
            for verse in BIBLE_VERSES:
                st.markdown(
                    f"**{verse['theme']}** — *{verse['reference']}*  \n{verse['text']}"
                )
            st.markdown("### Laws & principles")
            for title, body in LAWS:
                st.markdown(f"**{title}**  \n{body}")

    tab_surveys, tab_meds, tab_habits = st.tabs(
        ["Surveys (GAD-7 · PHQ-9 · PCL-5)", "Medication tracker", "Daily habits"]
    )

    with tab_surveys:
        if st.session_state.pop("show_saved_banner", None):
            st.success(f"Saved to `{DATA_PATH.name}`. Surveys reset for a new visit.")

        step = int(st.session_state.survey_step)
        st.progress(step / 3.0)
        st.caption(f"**Step {step} of 3** — use **Next** after GAD-7 and PHQ-9; **Score this visit** appears after PCL-5.")

        if step == 1:
            st.subheader("GAD-7")
            _render_gad7_step()
            c1, c2, _ = st.columns([1, 1, 4])
            with c1:
                if st.button("Next", type="primary", key="next_from_gad"):
                    if _step1_complete():
                        _persist_step1_and_advance()
                        st.rerun()
                    else:
                        st.warning("Please answer every GAD-7 question and the impairment follow-up.")
        elif step == 2:
            st.subheader("PHQ-9")
            _render_phq9_step()
            c1, c2, _ = st.columns([1, 1, 4])
            with c1:
                if st.button("Back", key="back_from_phq"):
                    st.session_state.survey_step = 1
                    st.rerun()
            with c2:
                if st.button("Next", type="primary", key="next_from_phq"):
                    if _step2_complete():
                        _persist_step2_and_advance()
                        st.rerun()
                    else:
                        st.warning("Please answer every PHQ-9 question and the impairment follow-up.")
        else:
            st.subheader("PCL-5")
            _render_pcl5_step()
            c1, c2, _ = st.columns([1, 1, 4])
            with c1:
                if st.button("Back", key="back_from_pcl"):
                    st.session_state.survey_step = 2
                    st.rerun()
            with c2:
                score_clicked = st.button("Score this visit", type="primary", key="score_visit")

            if score_clicked:
                if not (
                    st.session_state.get("persist_gad")
                    and st.session_state.get("persist_imp_gad") is not None
                    and st.session_state.get("persist_phq")
                    and st.session_state.get("persist_imp_phq") is not None
                ):
                    st.error("Missing earlier steps. Go back and complete GAD-7 and PHQ-9.")
                elif not _step3_complete():
                    st.warning("Please answer every PCL-5 question.")
                else:
                    gad_vals = list(st.session_state["persist_gad"])
                    imp_gad = int(st.session_state["persist_imp_gad"])
                    phq_vals = list(st.session_state["persist_phq"])
                    imp_phq = int(st.session_state["persist_imp_phq"])
                    pcl_vals = [
                        _int_from_likert_label(f"pcl_{i}", LIKERT_0_4)
                        for i in range(len(PCL5_QUESTIONS))
                    ]
                    gad_total = sum(gad_vals)
                    phq_total = sum(phq_vals)
                    pcl_total = sum(int(x) for x in pcl_vals if x is not None)
                    phq9_q9 = phq_vals[8]

                    prior_tuple = (
                        (prior_gad, prior_phq, prior_pcl)
                        if use_prior
                        else (None, None, None)
                    )

                    row = {
                        "date": visit_date.isoformat(),
                        "scored_at": datetime.now().isoformat(timespec="seconds"),
                        "gad7": gad_total,
                        "phq9": phq_total,
                        "pcl5": pcl_total,
                        "impairment_gad": imp_gad,
                        "impairment_phq": imp_phq,
                        "phq9_item9": phq9_q9,
                        "prior_gad7": prior_tuple[0] if use_prior else "",
                        "prior_phq9": prior_tuple[1] if use_prior else "",
                        "prior_pcl5": prior_tuple[2] if use_prior else "",
                        "gad7_items": json.dumps(gad_vals),
                        "phq9_items": json.dumps(phq_vals),
                        "pcl5_items": json.dumps(pcl_vals),
                    }
                    _save_session(row)
                    sync_last_scored_from_dataset()
                    st.session_state.survey_step = 1
                    for k in ("persist_gad", "persist_imp_gad", "persist_phq", "persist_imp_phq"):
                        st.session_state.pop(k, None)
                    st.session_state.pop("survey_session_started", None)
                    _clear_all_survey_widget_keys()
                    _init_survey_session()
                    st.session_state["show_saved_banner"] = True
                    st.rerun()

    with tab_meds:
        st.markdown(
            "Log **regimen changes** (e.g. titrations). Each row is effective on **`date`**; "
            "symptom charts show vertical markers on those dates, and each survey is matched to the "
            "**most recent** regimen on or before that date."
        )
        med_df = load_medications()
        if med_df.empty:
            st.info("No medication events yet.")
        else:
            st.dataframe(med_df, use_container_width=True, hide_index=True)

        sess_preview = _load_sessions()
        if not sess_preview.empty and not med_df.empty:
            st.subheader("Survey dates × active regimen")
            try:
                cr = cross_reference_sessions(sess_preview, med_df)
                cols = [c for c in ["date", "gad7", "phq9", "pcl5", "regimen_active"] if c in cr.columns]
                st.dataframe(cr[cols], use_container_width=True, hide_index=True)
            except Exception:
                pass

        with st.form("add_medication_event"):
            st.markdown(f"**Add event** (saved to `{MEDICATIONS_PATH.name}`)")
            ev_date = st.date_input("Effective date", value=date.today(), key="med_ev_date")
            ev_label = st.text_input(
                "Short label (for chart)",
                placeholder="e.g. Titration complete: Lexapro → Wellbutrin",
            )
            ev_prior = st.text_area(
                "Prior regimen",
                placeholder="Drug names and doses stopped or reduced…",
                height=100,
            )
            ev_new = st.text_area(
                "New / current regimen after this date",
                placeholder="Drug names and doses…",
                height=100,
            )
            submitted = st.form_submit_button("Save medication event")
            if submitted:
                if not ev_new.strip():
                    st.error("“New / current regimen” is required.")
                else:
                    append_event(
                        {
                            "date": ev_date.isoformat(),
                            "label": ev_label.strip() or "Medication update",
                            "prior_regimen": ev_prior.strip(),
                            "new_regimen": ev_new.strip(),
                        }
                    )
                    st.success("Saved.")
                    st.rerun()

    with tab_habits:
        st.subheader("Behavioral trackers (evidence-informed)")
        st.markdown(
            "Track **three** targets that trials and guidelines most often emphasize alongside "
            "therapy/medication for depression, anxiety, and PTSD: **moving more**, **protecting sleep**, "
            "and **structured calming / mindfulness** (breathing, meditation, grounding). "
            "This is self-monitoring only — not treatment."
        )
        st.caption(
            "Large reviews support **exercise** for mood and anxiety; **sleep** is a core target in PTSD "
            "and depression (e.g. CBT-I); **mindfulness-based** practices show strong effects for anxiety "
            "and are common adjuncts for PTSD and depression."
        )

        habit_df = load_habits_clean()
        if not habit_df.empty:
            disp = habit_df.copy()
            if "date" in disp.columns:
                disp["date"] = pd.to_datetime(disp["date"]).dt.strftime("%Y-%m-%d")
            show_cols = [
                c
                for c in (
                    "date",
                    "activity_minutes",
                    "sleep_hours",
                    "mindfulness_minutes",
                    "notes",
                )
                if c in disp.columns
            ]
            st.dataframe(disp[show_cols], use_container_width=True, hide_index=True)
            try:
                fig_h = _figure_habits_trends(habit_df)
                st.pyplot(fig_h)
                plt.close(fig_h)
            except Exception:
                st.caption("Could not plot habits (check numeric columns in habits.csv).")
        else:
            st.info("Log at least one day below to see trends.")

        with st.form("log_habit_day"):
            st.markdown(f"**Log or update a day** (saved to `{HABITS_PATH.name}`; same date overwrites).")
            hd = st.date_input("Date", value=date.today(), key="habit_log_date")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                act = st.number_input(
                    "Physical activity (minutes)",
                    min_value=0,
                    max_value=600,
                    value=30,
                    help="Brisk walking, cycling, sport, etc. — whatever you count as moderate+ activity.",
                )
            with col_b:
                sleep_h = st.number_input(
                    "Sleep (hours)",
                    min_value=0.0,
                    max_value=16.0,
                    value=7.0,
                    step=0.25,
                    help="Approximate hours slept last night.",
                )
            with col_c:
                mind = st.number_input(
                    "Mindfulness / calming (minutes)",
                    min_value=0,
                    max_value=180,
                    value=10,
                    help="Meditation, guided breathing, body scan, grounding — structured practice.",
                )
            hnote = st.text_input("Optional note (one line)", placeholder="e.g. rough night, walk at lunch")
            habit_saved = st.form_submit_button("Save day")
            if habit_saved:
                upsert_habit_day(
                    hd.isoformat(),
                    int(act),
                    float(sleep_h),
                    int(mind),
                    notes=hnote,
                )
                st.success("Saved.")
                st.rerun()

    last = st.session_state.get("last_scored")
    if last:
        _render_scored_visit(last)


if __name__ == "__main__":
    main()
