-- ThermoKourt behaviour database (SQLite).
--
-- Generated artefact: rebuilt from files (Motif recordings, temperature log,
-- trials sheet, DrosoDB snapshot, tracker output, scoring exports). Never
-- edited by hand. Star schema: small dimension tables, fact tables keyed by
-- trial / fly / window. Timestamps are ISO-8601 TEXT with offset;
-- temperatures are REAL degrees Celsius; times within a trial are seconds
-- from the first video frame.
--
-- Blinding: the ddb_* tables carry genotype names. The blinded export
-- (thermokourt-db export --blind) drops them and coded_groups.ddb_genotype_id.

PRAGMA foreign_keys = ON;

-- ── schema bookkeeping ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS schema_info (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT OR IGNORE INTO schema_info VALUES ('schema_version', '1');

-- ── provenance ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS source_files (
    file_id     INTEGER PRIMARY KEY,
    path        TEXT NOT NULL UNIQUE,          -- as seen from the recording PC
    kind        TEXT NOT NULL CHECK (kind IN
                ('video','arena_clip','temp_log','tracks_h5','pose_slp',
                 'ethogram','sheet','ddb_snapshot','arenas_json')),
    sha256      TEXT,
    size_bytes  INTEGER,
    ingested_at TEXT NOT NULL
);

-- ── DrosoDB copy (standalone: no live DDB needed after ingest) ────────
CREATE TABLE IF NOT EXISTS ddb_snapshots (
    snapshot_id INTEGER PRIMARY KEY,
    sha256      TEXT NOT NULL UNIQUE,
    source_path TEXT NOT NULL,
    taken_at    TEXT,                          -- mtime of the snapshot file
    ingested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ddb_donor (
    id          INTEGER PRIMARY KEY,           -- DDB donor.id
    name        TEXT,
    snapshot_id INTEGER NOT NULL REFERENCES ddb_snapshots
);
CREATE TABLE IF NOT EXISTS ddb_genotype (
    id              INTEGER PRIMARY KEY,       -- DDB genotype.id
    name            TEXT NOT NULL,
    chromosome_x    TEXT, chromosome_2 TEXT, chromosome_3 TEXT,
    chromosome_4    TEXT, chromosome_y TEXT,
    phenotype       TEXT,
    is_wildtype     INTEGER NOT NULL DEFAULT 0,
    donor_strain_id TEXT,                      -- e.g. BDSC number
    donor_id        INTEGER REFERENCES ddb_donor,
    snapshot_id     INTEGER NOT NULL REFERENCES ddb_snapshots
);
CREATE TABLE IF NOT EXISTS ddb_vial (
    id                INTEGER PRIMARY KEY,     -- DDB vial.id
    print_code        TEXT NOT NULL,
    genotype_id       INTEGER NOT NULL REFERENCES ddb_genotype,
    flipped_from_id   INTEGER,                 -- DDB vial.id (may be absent from the copy)
    generation        INTEGER,
    is_active         INTEGER,
    created_at        TEXT,
    decommissioned_at TEXT,
    notes             TEXT,
    snapshot_id       INTEGER NOT NULL REFERENCES ddb_snapshots
);
CREATE INDEX IF NOT EXISTS ix_ddb_vial_print_code ON ddb_vial (print_code);
CREATE TABLE IF NOT EXISTS ddb_vial_parentage (
    child_id  INTEGER NOT NULL,
    parent_id INTEGER NOT NULL,
    role      TEXT NOT NULL,                   -- 'mother' | 'father' | 'parent'
    snapshot_id INTEGER NOT NULL REFERENCES ddb_snapshots,
    PRIMARY KEY (child_id, parent_id, role)
);

-- ── dimensions ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,            -- 'E1' .. 'E7', 'PILOT'
    chapter       INTEGER,
    question      TEXT,
    hypothesis    TEXT
);
CREATE TABLE IF NOT EXISTS coded_groups (
    group_code      TEXT PRIMARY KEY,          -- opaque, e.g. 'G07'
    experiment_id   TEXT NOT NULL REFERENCES experiments,
    cross_label     TEXT,                      -- 'C1'..'C8' from the crossing guide, or NULL
    ddb_genotype_id INTEGER REFERENCES ddb_genotype,   -- dropped in the blinded export
    description     TEXT
);
CREATE TABLE IF NOT EXISTS behaviours (
    behaviour_id   INTEGER PRIMARY KEY,
    name           TEXT NOT NULL UNIQUE,       -- 'lunge', 'wing_threat', ...
    category       TEXT NOT NULL CHECK (category IN
                   ('aggression','courtship','locomotion','outcome','other')),
    gamethogram_key TEXT,
    description    TEXT
);
CREATE TABLE IF NOT EXISTS annotators (
    annotator_id INTEGER PRIMARY KEY,
    kind         TEXT NOT NULL CHECK (kind IN ('human','rule','model')),
    name         TEXT NOT NULL,                -- initials, or 'rules-v0', 'simba-lunge-v1'
    version      TEXT,
    UNIQUE (kind, name, version)
);

-- ── trials and animals ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trials (
    trial_id          TEXT PRIMARY KEY,        -- coded, also in the video file name
    experiment_id     TEXT NOT NULL REFERENCES experiments,
    group_code        TEXT NOT NULL REFERENCES coded_groups,
    stimulus          TEXT NOT NULL CHECK (stimulus IN ('triad','two_males','single_male')),
    test_temp_c       REAL NOT NULL,
    ramp_c_per_h      REAL,                    -- NULL = static
    rearing_temp_c    REAL,
    treatment         TEXT NOT NULL DEFAULT 'none'
                      CHECK (treatment IN ('none','arista_ablated','sham')),
    start_time        TEXT NOT NULL,           -- ISO-8601 with offset
    duration_s        REAL,
    fps               REAL,
    recording_dir     TEXT NOT NULL,           -- Motif recording folder
    arena_index       INTEGER NOT NULL,        -- which arena on the plate
    arena_cx REAL, arena_cy REAL, arena_r REAL, mm_per_pixel REAL,
    video_file_id     INTEGER REFERENCES source_files,
    clip_file_id      INTEGER REFERENCES source_files,
    temp_log_file_id  INTEGER REFERENCES source_files,
    arenas_file_id    INTEGER REFERENCES source_files,
    operator          TEXT,                    -- initials
    qc_flag           TEXT NOT NULL DEFAULT 'pending'
                      CHECK (qc_flag IN ('pending','ok','exclude')),
    exclusion_reason  TEXT,
    notes             TEXT,
    created_at        TEXT NOT NULL,
    UNIQUE (recording_dir, arena_index)
);
CREATE TABLE IF NOT EXISTS flies (
    fly_id           INTEGER PRIMARY KEY,
    trial_id         TEXT NOT NULL REFERENCES trials ON DELETE CASCADE,
    role             TEXT NOT NULL CHECK (role IN ('male_1','male_2','female_decap')),
    ddb_vial_id      INTEGER REFERENCES ddb_vial,
    ddb_print_code   TEXT,                     -- as typed/scanned at set-up
    eclosion_date    TEXT,
    age_days         REAL,
    marker_scored    TEXT,                     -- 'Sb', 'B', 'w', 'round-eyed', ...
    pcr_tube_label   TEXT,
    track_id         INTEGER,                  -- identity in the pose file after posttrack
    excluded         INTEGER NOT NULL DEFAULT 0,
    exclusion_reason TEXT,
    UNIQUE (trial_id, role)
);

-- ── facts ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS temperature_samples (
    trial_id TEXT NOT NULL REFERENCES trials ON DELETE CASCADE,
    t_s      REAL NOT NULL,
    temp_c   REAL NOT NULL,
    sensor   TEXT NOT NULL DEFAULT 'floor',
    PRIMARY KEY (trial_id, sensor, t_s)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS behaviour_events (
    event_id      INTEGER PRIMARY KEY,
    trial_id      TEXT NOT NULL REFERENCES trials ON DELETE CASCADE,
    annotator_id  INTEGER NOT NULL REFERENCES annotators,
    behaviour_id  INTEGER NOT NULL REFERENCES behaviours,
    actor_fly_id  INTEGER REFERENCES flies,
    target_fly_id INTEGER REFERENCES flies,
    start_s       REAL NOT NULL,
    end_s         REAL NOT NULL,
    confidence    REAL,                        -- model/rule score, NULL for humans
    source_file_id INTEGER REFERENCES source_files,
    CHECK (end_s >= start_s)
);
CREATE INDEX IF NOT EXISTS ix_events_trial ON behaviour_events (trial_id, annotator_id, behaviour_id);

CREATE TABLE IF NOT EXISTS window_summary (
    trial_id        TEXT NOT NULL REFERENCES trials ON DELETE CASCADE,
    fly_id          INTEGER NOT NULL REFERENCES flies,
    window_start_s  REAL NOT NULL,
    window_s        REAL NOT NULL DEFAULT 60,
    temp_c_mean     REAL,
    speed_mm_s      REAL,
    distance_mm     REAL,
    frac_tracked    REAL,                      -- fraction of frames with a valid pose
    n_encounters    INTEGER,                   -- approaches within 1.5 body lengths
    tracks_file_id  INTEGER REFERENCES source_files,
    PRIMARY KEY (trial_id, fly_id, window_start_s)
) WITHOUT ROWID;

-- ── views for the statistics ─────────────────────────────────────────
-- Per trial × annotator: aggression and courtship per encounter, mean speed.
CREATE VIEW IF NOT EXISTS v_trial_rates AS
WITH loco AS (
    SELECT w.trial_id,
           AVG(w.speed_mm_s)     AS speed_mm_s,
           SUM(w.distance_mm)    AS distance_mm,
           SUM(w.n_encounters)   AS n_encounters
    FROM window_summary w
    JOIN flies f ON f.fly_id = w.fly_id
    WHERE f.role IN ('male_1','male_2') AND f.excluded = 0
    GROUP BY w.trial_id
),
ev AS (
    SELECT e.trial_id, e.annotator_id,
           SUM(CASE WHEN b.category = 'aggression' THEN 1 ELSE 0 END)              AS n_aggression,
           SUM(CASE WHEN b.category = 'courtship'  THEN e.end_s - e.start_s ELSE 0 END) AS courtship_s,
           SUM(CASE WHEN b.name = 'lunge' THEN 1 ELSE 0 END)                       AS n_lunges
    FROM behaviour_events e JOIN behaviours b USING (behaviour_id)
    GROUP BY e.trial_id, e.annotator_id
)
SELECT t.trial_id, t.experiment_id, t.group_code, t.test_temp_c, t.ramp_c_per_h,
       t.treatment, t.stimulus, t.duration_s, t.qc_flag,
       ev.annotator_id,
       ev.n_aggression, ev.n_lunges, ev.courtship_s,
       loco.speed_mm_s, loco.distance_mm, loco.n_encounters,
       CASE WHEN loco.n_encounters > 0 THEN 1.0 * ev.n_aggression / loco.n_encounters END AS aggression_per_encounter,
       CASE WHEN loco.n_encounters > 0 THEN ev.courtship_s / loco.n_encounters END        AS courtship_s_per_encounter
FROM trials t
LEFT JOIN ev   ON ev.trial_id = t.trial_id
LEFT JOIN loco ON loco.trial_id = t.trial_id;

-- Human vs model/rule agreement per group × temperature cell: event counts side by side.
CREATE VIEW IF NOT EXISTS v_agreement AS
SELECT t.group_code, t.test_temp_c, b.name AS behaviour,
       a.kind AS annotator_kind, a.name AS annotator_name,
       COUNT(DISTINCT e.trial_id) AS n_trials,
       COUNT(*)                   AS n_events,
       SUM(e.end_s - e.start_s)   AS total_s
FROM behaviour_events e
JOIN trials t      USING (trial_id)
JOIN behaviours b  USING (behaviour_id)
JOIN annotators a  USING (annotator_id)
WHERE t.qc_flag <> 'exclude'
GROUP BY t.group_code, t.test_temp_c, b.name, a.kind, a.name;
