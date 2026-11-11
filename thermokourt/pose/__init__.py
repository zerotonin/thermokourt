"""Pose-based features and rule-based behaviour detection (stage 2.5 / 5-v0).

Input is SLEAP's *analysis* HDF5 export (``sleap-convert --format analysis``)
for one arena clip: ``tracks`` (n_tracks × 2 × n_nodes × n_frames),
``node_names``, ``track_occupancy``, ``point_scores``. From it we derive
per-frame features for the two males and detect wing extension, wing
threat, lunges and following with transparent thresholds. These "rules-v0"
events feed the database and are the baseline that trained classifiers
(SimBA, week 2+) must beat.
"""
