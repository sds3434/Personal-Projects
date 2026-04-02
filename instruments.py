"""
GAD-7, PHQ-9, and PCL-5 item text and scoring helpers.
Severity bands follow common clinical use (not the abbreviated labels on some forms).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LIKERT_0_3 = [
    (0, "Not at all"),
    (1, "Several days"),
    (2, "More than half the days"),
    (3, "Nearly every day"),
]

LIKERT_0_4 = [
    (0, "Not at all"),
    (1, "A little bit"),
    (2, "Moderately"),
    (3, "Quite a bit"),
    (4, "Extremely"),
]

GAD7_INTRO = (
    "Over the last 2 weeks, how often have you been bothered by any of the following problems?"
)

GAD7_QUESTIONS = [
    "Feeling nervous, anxious or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless that it is hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid as if something awful might happen",
]

FUNCTIONAL_LABELS = [
    (0, "Not difficult at all"),
    (1, "Somewhat difficult"),
    (2, "Very difficult"),
    (3, "Extremely difficult"),
]

PHQ9_INTRO = (
    "Over the last 2 weeks, how often have you been bothered by any of the following problems?"
)

PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual",
    "Thoughts that you would be better off dead or of hurting yourself in some way",
]

PCL5_INTRO = (
    "In the past month, how much have you been bothered by the following problems?"
)

PCL5_QUESTIONS = [
    "Repeated, disturbing, and unwanted memories of the stressful experience?",
    "Repeated, disturbing dreams of the stressful experience?",
    "Suddenly feeling or acting as if the stressful experience were actually happening again (as if you were actually back there reliving it)?",
    "Feeling very upset when something reminded you of the stressful experience?",
    "Having strong physical reactions when something reminded you of the stressful experience (for example, heart pounding, trouble breathing, sweating)?",
    "Avoiding memories, thoughts, or feelings related to the stressful experience?",
    "Avoiding external reminders of the stressful experience (for example, people, places, conversations, activities, objects, or situations)?",
    "Trouble remembering important parts of the stressful experience?",
    "Having strong negative beliefs about yourself, other people, or the world (for example, having thoughts such as: I am bad, there is something seriously wrong with me, no one can be trusted, the world is completely dangerous)?",
    "Blaming yourself or someone else for the stressful experience or what happened after it?",
    "Having strong negative feelings such as fear, horror, anger, guilt, or shame?",
    "Loss of interest in activities that you used to enjoy?",
    "Feeling distant or cut off from other people?",
    "Trouble experiencing positive feelings (for example, being unable to feel happiness or have loving feelings for people close to you)?",
    "Irritable behavior, angry outbursts, or acting aggressively?",
    "Taking too many risks or doing things that could cause you harm?",
    'Being "superalert" or watchful or on guard?',
    "Feeling jumpy or easily startled?",
    "Having difficulty concentrating?",
    "Trouble falling or staying asleep?",
]


def gad7_severity(total: int) -> str:
    if total <= 4:
        return "Minimal"
    if total <= 9:
        return "Mild"
    if total <= 14:
        return "Moderate"
    return "Severe"


def phq9_severity(total: int) -> str:
    if total <= 4:
        return "None–minimal"
    if total <= 9:
        return "Mild"
    if total <= 14:
        return "Moderate"
    if total <= 19:
        return "Moderately severe"
    return "Severe"


def pcl5_severity(total: int, probable_cutoff: int = 33) -> str:
    """Descriptive label; probable PTSD screening often uses total ≥ 31–33 depending on sample."""
    if total < probable_cutoff:
        return f"Below common probable-PTSD cutoff ({probable_cutoff})"
    return f"At or above common probable-PTSD cutoff ({probable_cutoff})"


FactorKey = Literal["positive", "partial", "uncertain", "revisit"]


@dataclass
class FourFactorResult:
    positive: str
    partial: str
    uncertain: str
    revisit: str
    dominant: FactorKey
    phq9_item9_flag: bool
    comparison_block: str
    trend_vs_prior_survey: str
    suggested_steps: str
    watch_for: str


def four_factor_assessment(
    gad7: int,
    phq9: int,
    pcl5: int,
    impairment_gad: int,
    impairment_phq: int,
    phq9_q9: int,
    *,
    pcl_probable_cutoff: int = 33,
    prior_gad7: int | None = None,
    prior_phq9: int | None = None,
    prior_pcl5: int | None = None,
    prev_visit_gad7: int | None = None,
    prev_visit_phq9: int | None = None,
    prev_visit_pcl5: int | None = None,
    prev_visit_date_label: str | None = None,
    current_visit_date_label: str | None = None,
) -> FourFactorResult:
    """
    Maps a single visit into four clinical-posture narratives, optional comparison to the
    chronologically prior saved visit (prev_visit_*), and actionable-style notes for discussion
    with a clinician — not a substitute for care.
    """
    phq9_flag = phq9_q9 >= 1
    high_impairment = impairment_gad >= 2 or impairment_phq >= 2
    severe_gad = gad7 >= 15
    severe_phq = phq9 >= 20
    severe_pcl = pcl5 >= 50
    moderate_or_worse_gad = gad7 >= 10
    moderate_or_worse_phq = phq9 >= 10
    probable_ptsd = pcl5 >= pcl_probable_cutoff

    has_prev_survey = (
        prev_visit_gad7 is not None
        and prev_visit_phq9 is not None
        and prev_visit_pcl5 is not None
    )
    has_manual_prior = (
        prior_gad7 is not None and prior_phq9 is not None and prior_pcl5 is not None
    )

    revisit_bits: list[str] = []
    if phq9_flag:
        revisit_bits.append(
            "PHQ-9 item 9 is endorsed: discuss safety and follow your crisis plan; seek urgent help if risk is imminent."
        )
    if severe_gad or severe_phq or severe_pcl:
        revisit_bits.append(
            "One or more total scores are in a severe range: review treatment intensity and supports."
        )
    if high_impairment:
        revisit_bits.append(
            "Functional impairment is rated as very or extremely difficult in one or both impairment questions."
        )
    if probable_ptsd and moderate_or_worse_phq:
        revisit_bits.append(
            "PTSD symptoms likely overlap with mood: ensure trauma-informed care is addressed in your plan."
        )

    positive_ok = (
        gad7 <= 9
        and phq9 <= 9
        and pcl5 < pcl_probable_cutoff
        and not phq9_flag
        and impairment_gad <= 1
        and impairment_phq <= 1
    )

    partial_bits: list[str] = []
    if has_prev_survey:
        d_gad = prev_visit_gad7 - gad7
        d_phq = prev_visit_phq9 - phq9
        d_pcl = prev_visit_pcl5 - pcl5
        lbl = prev_visit_date_label or "prior survey"
        if d_gad + d_phq + d_pcl > 0:
            partial_bits.append(
                f"Versus **{lbl}**: GAD-7 Δ {d_gad:+d}, PHQ-9 Δ {d_phq:+d}, PCL-5 Δ {d_pcl:+d} "
                "(positive Δ = lower symptoms now)."
            )
        elif d_gad + d_phq + d_pcl < 0:
            partial_bits.append(
                f"Versus **{lbl}**: total scores are higher now on one or more scales — worth reviewing with your clinician."
            )
        else:
            partial_bits.append(
                f"Versus **{lbl}**: totals unchanged; may reflect stability or plateau."
            )
    elif has_manual_prior:
        d_gad = prior_gad7 - gad7
        d_phq = prior_phq9 - phq9
        d_pcl = prior_pcl5 - pcl5
        if d_gad + d_phq + d_pcl > 0:
            partial_bits.append(
                f"Compared with sidebar “prior” totals: GAD-7 Δ {d_gad:+d}, PHQ-9 Δ {d_phq:+d}, PCL-5 Δ {d_pcl:+d}."
            )
        else:
            partial_bits.append("Compared with sidebar prior totals, scores did not improve overall.")
    else:
        partial_bits.append(
            "No earlier saved survey in your dataset yet — the next visit will show a before/after comparison here."
        )

    uncertain_bits: list[str] = []
    if moderate_or_worse_gad or moderate_or_worse_phq or probable_ptsd:
        if not (severe_gad or severe_phq or phq9_flag):
            uncertain_bits.append(
                "Moderate symptoms or probable PTSD-level burden without clear crisis flags: monitor closely and align therapy/medication with goals."
            )
    if not uncertain_bits:
        uncertain_bits.append(
            "No separate “uncertain/plateau” signal beyond the other rows — use prior scores if available."
        )

    positive_text = (
        "Symptoms and impairment look relatively controlled on this snapshot, without self-harm ideation endorsed on item 9."
        if positive_ok
        else "Not all domains are in the mild/minimal range and/or impairment or item 9 suggests more attention is needed — see other rows."
    )

    partial_text = " ".join(partial_bits)
    uncertain_text = " ".join(uncertain_bits)
    revisit_text = (
        " ".join(revisit_bits)
        if revisit_bits
        else "No automatic “revisit/escalate” flags from totals and impairment alone — still discuss any worsening with your clinician."
    )

    dominant: FactorKey = "positive"
    if revisit_bits or phq9_flag:
        dominant = "revisit"
    elif not positive_ok and (moderate_or_worse_gad or moderate_or_worse_phq or probable_ptsd):
        dominant = "uncertain"
    elif has_prev_survey:
        net = (prev_visit_gad7 - gad7) + (prev_visit_phq9 - phq9) + (prev_visit_pcl5 - pcl5)
        if net > 0:
            dominant = "partial"
    elif has_manual_prior:
        if (prior_gad7 - gad7) + (prior_phq9 - phq9) + (prior_pcl5 - pcl5) > 0:
            dominant = "partial"

    # Comparison block: previous survey vs current (table-like text)
    cur_l = current_visit_date_label or "Current visit"
    if has_prev_survey:
        pl = prev_visit_date_label or "Prior survey"
        comparison_block = (
            f"| Scale | {pl} | {cur_l} |\n"
            f"|---|---:|---:|\n"
            f"| GAD-7 | {prev_visit_gad7} | {gad7} |\n"
            f"| PHQ-9 | {prev_visit_phq9} | {phq9} |\n"
            f"| PCL-5 | {prev_visit_pcl5} | {pcl5} |"
        )
        dg = prev_visit_gad7 - gad7
        dp = prev_visit_phq9 - phq9
        dc = prev_visit_pcl5 - pcl5
        trend_vs_prior_survey = (
            f"**Change (prior → current):** GAD-7 {dg:+d}, PHQ-9 {dp:+d}, PCL-5 {dc:+d}. "
            "Positive numbers mean totals went down (symptom improvement on that scale)."
        )
    else:
        comparison_block = (
            "Only one saved survey is in your dataset so far. After your next scored visit, "
            "this section will compare **that** visit to this one."
        )
        trend_vs_prior_survey = "Add a second dated visit to see trend arrows and side-by-side totals."

    # Suggested steps & watch-for (educational prompts; not individualized medical advice)
    steps: list[str] = []
    watches: list[str] = []
    if phq9_flag:
        steps.append(
            "Prioritize safety: tell your clinician or therapist about these thoughts; "
            "use your crisis plan or local emergency services if risk feels immediate."
        )
        watches.append("Any increase in hopelessness, intent, or plans; substance use; isolation.")
    if severe_gad or severe_phq or severe_pcl:
        steps.append(
            "Schedule or keep contact with your prescriber/clinician — high scores may warrant review of medication, therapy frequency, or supports."
        )
        watches.append("Sleep, appetite, energy, and ability to work or care for yourself day to day.")
    if high_impairment:
        steps.append(
            "Name one concrete support (person, appointment, workplace accommodation) to reduce strain on work, home, or relationships."
        )
        watches.append("Whether difficulty is stable or getting worse week to week.")
    if probable_ptsd and not severe_phq:
        steps.append(
            "If trauma-focused therapy is part of your plan, note whether avoidance or triggers are improving or flaring."
        )
    if has_prev_survey:
        net_improve = (
            (prev_visit_gad7 - gad7)
            + (prev_visit_phq9 - phq9)
            + (prev_visit_pcl5 - pcl5)
        )
        if net_improve > 0:
            steps.append(
                "Trend is favorable: keep logging visits and continue current strategies unless your clinician advises changes."
            )
        elif net_improve < 0:
            steps.append(
                "Trends are up since the last survey: bring a short note about sleep, stress, and medication changes to your next appointment."
            )
            watches.append("New stressors, medication side effects, or skipped doses.")
        else:
            steps.append(
                "Scores are similar to last time: discuss goals with your clinician (symptom control vs side effects vs life stressors)."
            )
    if not steps:
        steps.append(
            "Keep regular follow-up with your care team; bring these scores to discuss what is working and what is not."
        )
    if not watches:
        watches.append(
            "Sudden jumps in scores, new safety concerns, or major life changes — mention them even if not captured on the forms."
        )

    suggested_steps = "\n".join(f"- {s}" for s in steps[:6])
    watch_for = "\n".join(f"- {w}" for w in watches[:5])

    return FourFactorResult(
        positive=positive_text,
        partial=partial_text,
        uncertain=uncertain_text,
        revisit=revisit_text,
        dominant=dominant,
        phq9_item9_flag=phq9_flag,
        comparison_block=comparison_block,
        trend_vs_prior_survey=trend_vs_prior_survey,
        suggested_steps=suggested_steps,
        watch_for=watch_for,
    )
