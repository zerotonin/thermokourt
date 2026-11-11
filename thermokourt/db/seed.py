"""Reference rows that every behaviour database starts with.

Experiments follow the crossing guide v1.2 (Nov 2026); the ethogram is the
candidate list for the December pilot and will be trimmed there.
"""

from __future__ import annotations

import sqlite3

EXPERIMENTS = [
    (
        "PILOT",
        2,
        "December pilot: timeline, ethogram, scoring speed, temperature log, tracking test",
        None,
    ),
    ("E1", 2, "Thermal performance curves, Canton-S at 18/22/25/28/31/34 °C", "H1"),
    ("E2", 2, "Aggression without a female (two males alone) at 22/28/33 °C", "H3"),
    ("E3", 3, "Developmental temperature (reared 25 vs 28 °C), tested 22/28/33 °C", "H1"),
    ("E4", 4, "Peripheral thermosensation: arista-ablated vs sham", "H2"),
    ("E5", 4, "Aggression off, courtship on: TβhnM18 vs w1118", "H0"),
    ("E6", 4, "Central warm sensing: TrpA1 null, deficiency, rescue, driver/UAS controls", "H2"),
    ("E7", 4, "Peripheral hot cells: Gr28b.d-GAL4 > UAS-Kir2.1 and parental controls", "H2"),
]

# Short titles for pickers and reports ("E1 – Thermal performance curve").
EXPERIMENT_TITLES = {
    "PILOT": "December pilot",
    "E1": "Thermal performance curve",
    "E2": "Aggression without a female",
    "E3": "Developmental temperature",
    "E4": "Arista ablation",
    "E5": "Aggression off (TβhnM18)",
    "E6": "Central warm sensing (TrpA1)",
    "E7": "Peripheral hot cells (Gr28b.d > Kir2.1)",
}


def experiment_label(exp_id: str) -> str:
    """``"E1 – Thermal performance curve"``; unknown ids come back unchanged."""
    t = EXPERIMENT_TITLES.get(exp_id)
    return f"{exp_id} – {t}" if t else exp_id


# (name, category, gamethogram_key, description)
BEHAVIOURS = [
    ("lunge", "aggression", None, "Male rears and snaps down onto the opponent"),
    ("wing_threat", "aggression", None, "Both wings raised ~45°, facing the opponent"),
    ("fencing", "aggression", None, "Foreleg strikes at the opponent"),
    ("boxing", "aggression", None, "Mutual rearing with foreleg strikes"),
    ("holding", "aggression", None, "Grabs and holds the opponent"),
    ("tussling", "aggression", None, "Mutual grappling/rolling"),
    ("chasing", "aggression", None, "Sustained pursuit of the other male"),
    ("orientation", "courtship", None, "Faces the female within one body length"),
    ("following", "courtship", None, "Follows the female"),
    ("tapping", "courtship", None, "Foreleg taps on the female"),
    ("wing_extension", "courtship", None, "Unilateral wing extension (song)"),
    ("licking", "courtship", None, "Proboscis contact with female genitalia"),
    ("attempted_copulation", "courtship", None, "Abdominal bending toward the female"),
    ("walking", "locomotion", None, "Walking bout"),
    ("knock_down", "outcome", None, "Loses posture at high temperature"),
    ("immobile", "outcome", None, "No movement for > 10 s"),
]

ANNOTATORS = [
    ("rule", "rules", "v0"),
]


def seed(con: sqlite3.Connection) -> None:
    con.executemany(
        "INSERT OR IGNORE INTO experiments"
        " (experiment_id, chapter, question, hypothesis) VALUES (?,?,?,?)",
        EXPERIMENTS,
    )
    con.executemany(
        "INSERT OR IGNORE INTO behaviours"
        " (name, category, gamethogram_key, description) VALUES (?,?,?,?)",
        BEHAVIOURS,
    )
    con.executemany(
        "INSERT OR IGNORE INTO annotators (kind, name, version) VALUES (?,?,?)",
        ANNOTATORS,
    )
    con.commit()
