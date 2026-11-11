"""Stage 0 — the operator's entry point.

``thermokourt-trial`` turns one Motif recording session into database rows,
arena clips, Aoraki pose jobs and rule-based events, in idempotent stages
that can be re-run after any failure. The GUI (``thermokourt.trial.gui``)
is a thin front end over :mod:`thermokourt.trial.runner`.
"""
