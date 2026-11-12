"""The session pipeline, as idempotent stages with on-disk state.

A *session* is one Motif recording folder (all arenas recorded together).
State lives in ``<recording>/thermokourt_session.json`` so that any stage
can be re-run after a crash, a lost SSH connection or a cancelled job.

Stages
    extract   arena clips via the arena extractor (reusing the rig's arenas JSON)
    register  trials + flies + DrosoDB copy into the database
    upload    rsync clips to the cluster
    submit    sbatch array job (one task per arena clip)
    poll      wait for the job; returns when it has finished
    fetch     rsync the pose .h5 files back
    score     roles, features, rules-v0 events, window summaries -> database
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from thermokourt.db import ddb as ddbmod
from thermokourt.db.core import connect, init_db, now_iso, register_file
from thermokourt.db.seed import seed
from thermokourt.db.trials import FlySpec, TrialSpec, add_trial, ensure_group
from thermokourt.pose.analysis import (
    RuleParams,
    assign_roles,
    compute_features,
    detect_events,
    load_sleap_h5,
    window_summary,
)
from thermokourt.trial.config import RigConfig

STAGES = ("extract", "register", "upload", "submit", "poll", "fetch", "score")


def job_states_ok(sacct_text: str) -> tuple[bool, dict[str, int]]:
    """Parse ``sacct -n -X -o State`` output (one state per task, possibly
    ``CANCELLED+`` or ``CANCELLED by 123``) into counts; ok iff every task
    COMPLETED."""
    counts: dict[str, int] = {}
    for line in sacct_text.splitlines():
        w = line.strip().split()
        if not w:
            continue
        state = w[0].rstrip("+").upper()
        counts[state] = counts.get(state, 0) + 1
    return bool(counts) and set(counts) == {"COMPLETED"}, counts


def cancel_waiting_command(job_ids: str) -> str:
    """Shell command cancelling only the still-pending jobs among ``job_ids``.
    (``scancel`` exits 1 on job ids that already finished.)"""
    return (
        f"ids=$(squeue -j {shlex.quote(job_ids)} -h -t PENDING -o %i 2>/dev/null | paste -sd,); "
        '[ -z "$ids" ] || scancel "$ids" 2>/dev/null; true'
    )


def parse_worker_jobs(submit_output: str) -> str:
    """Comma-separated job ids from sleap_submit.sh's ``JOBS=`` line."""
    for line in reversed(submit_output.splitlines()):
        if line.startswith("JOBS="):
            ids = [j for j in line[5:].strip().split(",") if j.isdigit()]
            if ids:
                return ",".join(ids)
    raise RuntimeError(f"sleap_submit.sh started no workers: {submit_output[-300:]!r}")


Log = Callable[[str], None]


@dataclass
class SessionSpec:
    """What the operator fills in once per recording session."""

    recording_dir: Path
    experiment_id: str
    group_code: str
    test_temp_c: float
    stimulus: str = "triad"
    ramp_c_per_h: float | None = None
    rearing_temp_c: float | None = None
    treatment: str = "none"
    operator: str | None = None
    notes: str | None = None
    cross_label: str | None = None
    temp_log: Path | None = None
    # arena_index -> list of {"role":..., "print_code":..., "age_days":...}
    arenas: dict[int, list[dict]] = field(default_factory=dict)

    def to_json(self) -> dict:
        d = {k: (str(v) if isinstance(v, Path) else v) for k, v in self.__dict__.items()}
        d["arenas"] = {str(k): v for k, v in self.arenas.items()}
        return d

    @classmethod
    def from_json(cls, d: dict) -> SessionSpec:
        d = dict(d)
        d["recording_dir"] = Path(d["recording_dir"])
        d["temp_log"] = Path(d["temp_log"]) if d.get("temp_log") else None
        d["arenas"] = {int(k): v for k, v in d.get("arenas", {}).items()}
        return cls(**d)


class Session:
    def __init__(self, cfg: RigConfig, spec: SessionSpec, log: Log = print):
        self.cfg = cfg
        self.spec = spec
        self.log = log
        self.rec = Path(spec.recording_dir)
        self.name = self.rec.name
        self.state_path = self.rec / "thermokourt_session.json"
        self.clips_dir = self.rec / "clips"
        self.pose_dir = self.rec / "pose"
        self.state = self._load_state()

    # ── state ──────────────────────────────────────────────────────────
    def _load_state(self) -> dict:
        if self.state_path.is_file():
            return json.loads(self.state_path.read_text())
        return {"spec": self.spec.to_json(), "done": {}, "job_id": None, "trials": {}}

    def _save(self):
        self.state["updated"] = now_iso()
        self.state_path.write_text(json.dumps(self.state, indent=2))

    def _mark(self, stage: str, **info):
        self.state["done"][stage] = {"at": now_iso(), **info}
        self._save()

    def is_done(self, stage: str) -> bool:
        return stage in self.state["done"]

    # ── shell helpers ──────────────────────────────────────────────────
    def _run(self, cmd: list[str], **kw) -> subprocess.CompletedProcess:
        self.log("$ " + " ".join(shlex.quote(c) for c in cmd))
        r = subprocess.run(cmd, capture_output=True, text=True, **kw)
        if r.returncode != 0:
            raise RuntimeError(f"command failed ({r.returncode}): {cmd[0]}\n{r.stderr[-2000:]}")
        return r

    def _run_with_progress(self, cmd: list[str], stage: str) -> None:
        """Run a command that prints ``@progress arena=NN of=N phase=P frac=F`` lines
        (the cutter) and log the overall share done as
        ``@progress <stage> <frac> <detail>``, at most once a second."""
        self.log("$ " + " ".join(shlex.quote(c) for c in cmd))
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        arenas: dict[str, float] = {}
        n_total, last, tail = 0, 0.0, []
        for line in proc.stdout:
            line = line.rstrip()
            if not line.startswith("@progress "):
                tail = (tail + [line])[-40:]
                continue
            f = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
            arenas[f["arena"]] = float(f["frac"])
            n_total = max(n_total, int(f.get("of", 0)))
            now = time.time()
            if now - last >= 1.0 or f.get("phase") == "done":
                frac = sum(arenas.values()) / max(n_total, len(arenas), 1)
                busy = [a for a, v in sorted(arenas.items()) if v < 1.0]
                n_done = sum(v >= 1.0 for v in arenas.values())
                detail = f"{n_done}/{n_total} arenas" + (
                    f", working on A{', A'.join(busy[:4])}" if busy else ""
                )
                self.log(f"@progress {stage} {frac:.4f} {detail}")
                last = now
        if proc.wait() != 0:
            raise RuntimeError(
                f"command failed ({proc.returncode}): {cmd[0]}\n" + "\n".join(tail[-20:])
            )

    def _ssh(self, remote_cmd: str) -> str:
        ssh = shlex.split(self.cfg.cluster.ssh_cmd)
        r = self._run([*ssh, self.cfg.cluster.ssh_host, remote_cmd])
        return r.stdout.strip()

    @property
    def _rsync_e(self) -> list[str]:
        """``-e`` for rsync so it uses the same transport as :meth:`_ssh`."""
        return ["-e", self.cfg.cluster.ssh_cmd]

    @property
    def expected_arenas(self) -> int:
        """Arenas on the plate (the triad plate has 10), not the number in use."""
        return 10

    @property
    def remote_clips(self) -> str:
        return f"{self.cfg.cluster.project}/clips/{self.name}"

    @property
    def remote_pose(self) -> str:
        return f"{self.cfg.cluster.project}/pose/{self.name}"

    # ── stages ─────────────────────────────────────────────────────────
    def extract(self):
        if self.is_done("extract"):
            self.log("extract: already done")
            return
        self.clips_dir.mkdir(exist_ok=True)
        rig_arenas = Path(self.cfg.arenas_json) if self.cfg.arenas_json else None
        have_layout = bool(rig_arenas and rig_arenas.is_file())
        cmd = [
            sys.executable,
            "-m",
            "thermokourt.extract.arena_extractor",
            str(self.rec),
            "-o",
            str(self.clips_dir),
            "--no_gui",
            "--crf",
            str(self.cfg.extract.crf),
            "-j",
            str(self.cfg.extract.jobs),
            "--min_size",
            str(self.cfg.extract.min_size),
        ]
        own = self.clips_dir / f"{self.name}_arenas.json"
        used_own = own.is_file()
        if used_own:
            # this recording's own layout (detected or fixed for it, or an older
            # camera position) wins over the rig default
            self.log(f"extract: using this recording's arena layout {own.name}")
            cmd += ["--arenas", str(own)]
        elif have_layout:
            cmd += ["--arenas", str(rig_arenas)]
        else:
            # first session on this rig: detect on this recording's max-projection
            self.log(f"extract: no saved arena layout at {rig_arenas}; detecting on this recording")
            cmd += ["-n", str(self.expected_arenas)]
        try:
            self._run_with_progress(cmd, "cut")
        except RuntimeError as e:
            if not have_layout and "No arenas defined" in str(e):
                raise RuntimeError(
                    "no arenas found in this recording. Is the backlight on, the lens cap off "
                    "and the plate in view? Check the live preview, then record again."
                ) from e
            raise
        found = self.clips_dir / f"{self.name}_arenas.json"
        if not have_layout and not used_own and found.is_file():
            n = len(json.loads(found.read_text(encoding="utf-8")).get("arenas", []))
            if n != self.expected_arenas:
                raise RuntimeError(
                    f"found {n} of {self.expected_arenas} arenas — see "
                    f"{self.clips_dir / (self.name + '_arenas.png')}. Fix zoom, lighting or plate "
                    "position and record again (or run thermokourt-extract with the editor)."
                )
            if rig_arenas:
                rig_arenas.parent.mkdir(parents=True, exist_ok=True)
                rig_arenas.write_text(found.read_text(encoding="utf-8"), encoding="utf-8")
                self.log(f"extract: saved this layout as the rig default → {rig_arenas}")
        clips = sorted(self.clips_dir.glob(f"{self.name}_arena_*.mp4"))
        if not clips:
            raise RuntimeError("extract produced no clips")
        self.log(f"extract: {len(clips)} clips")
        self._mark("extract", n_clips=len(clips))

    def register(self):
        if self.is_done("register"):
            self.log("register: already done")
            return
        con = init_db(self.cfg.db)
        seed(con)
        ensure_group(con, self.spec.group_code, self.spec.experiment_id, self.spec.cross_label)
        codes = sorted(
            {
                f["print_code"]
                for flies in self.spec.arenas.values()
                for f in flies
                if f.get("print_code")
            }
        )
        vials = {}
        if codes:
            if not self.cfg.ddb_snapshot or not Path(self.cfg.ddb_snapshot).is_file():
                raise FileNotFoundError("DrosoDB snapshot not found; check rig.yaml ddb_snapshot")
            vials = ddbmod.import_vials_by_print_code(con, self.cfg.ddb_snapshot, codes)
            for c, v in vials.items():
                self.log(f"vial {c}: {v.genotype_name}")
        arenas_json = self.clips_dir / f"{self.name}_arenas.json"
        if not arenas_json.is_file():
            arenas_json = Path(self.cfg.arenas_json)
        for idx, flies in sorted(self.spec.arenas.items()):
            clip = self.clips_dir / f"{self.name}_arena_{idx:02d}.mp4"
            ts = TrialSpec(
                arena_index=idx,
                experiment_id=self.spec.experiment_id,
                group_code=self.spec.group_code,
                test_temp_c=self.spec.test_temp_c,
                stimulus=self.spec.stimulus,
                ramp_c_per_h=self.spec.ramp_c_per_h,
                rearing_temp_c=self.spec.rearing_temp_c,
                treatment=self.spec.treatment,
                operator=self.spec.operator,
                notes=self.spec.notes,
                flies=[
                    FlySpec(
                        role=f["role"],
                        print_code=f.get("print_code"),
                        ddb_vial_id=vials[f["print_code"]].vial_id if f.get("print_code") else None,
                        eclosion_date=f.get("eclosion_date"),
                        age_days=f.get("age_days"),
                        marker_scored=f.get("marker_scored"),
                        pcr_tube_label=f.get("pcr_tube_label"),
                    )
                    for f in flies
                ],
            )
            try:
                tid = add_trial(
                    con,
                    self.rec,
                    arenas_json,
                    ts,
                    clip_path=clip if clip.is_file() else None,
                    temp_log=self.spec.temp_log,
                    mm_per_pixel=self.cfg.mm_per_pixel,
                )
            except Exception as e:  # already registered
                if "UNIQUE" not in str(e):
                    raise
                row = con.execute(
                    "SELECT trial_id FROM trials WHERE recording_dir=? AND arena_index=?",
                    (str(self.rec), idx),
                ).fetchone()
                tid = row[0]
            self.state["trials"][str(idx)] = tid
            self.log(f"arena {idx}: trial {tid}")
        con.close()
        self._mark("register", n_trials=len(self.spec.arenas))

    def upload(self):
        if self.is_done("upload"):
            self.log("upload: already done")
            return
        self._ssh(f"mkdir -p {shlex.quote(self.remote_clips)} {shlex.quote(self.remote_pose)}")
        self._run(
            [
                "rsync",
                *self._rsync_e,
                "-a",
                "--partial",
                "--include=*_arena_*.mp4",
                "--exclude=*",
                f"{self.clips_dir}/",
                f"{self.cfg.cluster.ssh_host}:{self.remote_clips}/",
            ]
        )
        n = int(self._ssh(f"ls {shlex.quote(self.remote_clips)} | wc -l"))
        self.log(f"upload: {n} clips on the cluster")
        self._mark("upload", n_clips=n)

    def submit(self):
        if self.is_done("submit") and self.state.get("job_id"):
            self.log(f"submit: already done (job {self.state['job_id']})")
            return
        idxs = sorted(self.spec.arenas) or list(range(self.state["done"]["extract"]["n_clips"]))
        todo = self._unfinished(idxs)
        if not todo:
            self.log("submit: every arena already has its poses on the cluster; nothing to send")
            self.state["job_id"] = ""
            self._mark("submit", job_id="", n_tasks=0, n_workers=0)
            return
        if len(todo) < len(idxs):
            self.log(f"submit: {len(idxs) - len(todo)} arenas already done, sending {len(todo)}")
        idxs = todo
        arr = ",".join(str(i) for i in idxs)
        c = self.cfg.cluster
        cmd = (
            f"cd {shlex.quote(c.project)} && N_ANIMALS={self.cfg.n_animals} "
            f"INVERT={1 if c.invert else 0} TIME={c.array_time} "
            f"bash scripts/slurm/sleap_submit.sh {shlex.quote(self.name)} "
            f"{shlex.quote(c.sleap_run)} {arr}"
        )
        out = self._ssh(cmd)
        job_id = parse_worker_jobs(out)
        self.state["job_id"] = job_id
        n_workers = len(job_id.split(","))
        self.log(f"submit: {len(idxs)} clips queued, {n_workers} GPU workers (jobs {job_id})")
        self._mark("submit", job_id=job_id, n_tasks=len(idxs), n_workers=n_workers)

    def poll(self, interval_s: float = 60, timeout_s: float = 12 * 3600) -> str:
        if self.is_done("poll"):
            self.log("poll: job already finished")
            return "done"
        job = self.state.get("job_id")
        if not job and self.is_done("submit"):
            self._mark("poll", states="nothing sent", ok=True)
            return "done"
        if not job:
            raise RuntimeError("no job submitted")
        t0 = time.time()
        while True:
            q = self._ssh(f"squeue -j {job} -h -o '%T' | sort | uniq -c | tr '\\n' ' '")
            left = self._clips_left()
            if left == 0:
                # every clip is tracked, empty or parked: waiting workers have no work
                if q:
                    self._ssh(cancel_waiting_command(job))
                    self.log("poll: all clips finished; cancelled workers still waiting for a GPU")
                else:
                    self.log("poll: all clips finished")
                self._mark("poll", states="clips finished", ok=True)
                return "done"
            if q:
                self.log(
                    f"poll: job {job}: {q.strip()}" + (f", {left} clips to go" if left >= 0 else "")
                )
            else:
                st = self._ssh(f"sacct -j {job} -n -X -o State")
                ok, counts = job_states_ok(st)
                summary = " ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
                self.log(f"poll: job {job} finished: {summary}")
                self._mark("poll", states=summary, ok=ok)
                return "done" if ok else "failed"
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"job {job} still running after {timeout_s / 3600:.0f} h")
            time.sleep(interval_s)

    def _remote_pose_names(self) -> set[str]:
        return set(
            self._ssh(
                f"cd {shlex.quote(self.remote_pose)} 2>/dev/null && "
                "{ ls -1; ls -1 _queue 2>/dev/null | sed 's|^|_queue/|'; } || true"
            ).split()
        )

    def _unfinished(self, idxs: list[int]) -> list[int]:
        """Arenas without poses (.h5) or a no-animals marker (.empty) on the cluster."""
        names = self._remote_pose_names()
        done = {f"{self.name}_arena_{i:02d}{ext}" for i in idxs for ext in (".h5", ".empty")}
        have = names & done
        return [
            i
            for i in idxs
            if not {f"{self.name}_arena_{i:02d}.h5", f"{self.name}_arena_{i:02d}.empty"} & have
        ]

    def _clips_left(self) -> int:
        """Clips of this trial without a .h5, .empty or parked marker on the cluster."""
        idxs = sorted(int(i) for i in self.state.get("trials", {})) or []
        if not idxs:
            return -1
        names = self._remote_pose_names()
        left = 0
        for i in idxs:
            stem = f"{self.name}_arena_{i:02d}"
            if not ({f"{stem}.h5", f"{stem}.empty", f"_queue/parked_{i:02d}"} & names):
                left += 1
        return left

    def fetch(self):
        if self.is_done("fetch"):
            self.log("fetch: already done")
            return
        self.pose_dir.mkdir(exist_ok=True)
        self._run(
            [
                "rsync",
                *self._rsync_e,
                "-a",
                "--include=*.h5",
                "--include=*.slp",
                "--include=*.empty",
                "--exclude=*",
                f"{self.cfg.cluster.ssh_host}:{self.remote_pose}/",
                f"{self.pose_dir}/",
            ]
        )
        h5 = sorted(self.pose_dir.glob("*.h5"))
        stems = [f"{self.name}_arena_{int(i):02d}" for i in self.state["trials"]]
        empty = [s for s in stems if (self.pose_dir / f"{s}.empty").is_file()]
        missing = [
            f"{s}.h5" for s in stems if s not in empty and not (self.pose_dir / f"{s}.h5").is_file()
        ]
        for s in empty:
            self.log(f"fetch: arena {s[-2:]}: no animals found by the pose model")
        self.log(f"fetch: {len(h5)} pose files, {len(empty)} empty, {len(missing)} missing")
        if missing:
            raise RuntimeError(
                f"pose files missing for {len(missing)} arena(s): {missing[:3]}... "
                "check the SLURM logs on the cluster, then re-run from 'submit'"
            )
        self._mark("fetch", n_pose=len(h5), n_empty=len(empty))

    def score(self, params: RuleParams | None = None):
        params = params or RuleParams()
        con = connect(self.cfg.db)
        con.execute(
            "INSERT OR IGNORE INTO annotators (kind, name, version) VALUES ('rule','rules',?)",
            (params.version,),
        )
        ann_id = con.execute(
            "SELECT annotator_id FROM annotators WHERE kind='rule' AND name='rules' AND version=?",
            (params.version,),
        ).fetchone()[0]
        beh = {r[0]: r[1] for r in con.execute("SELECT name, behaviour_id FROM behaviours")}
        n_ev = 0
        skipped, empty = [], []
        for idx_s, tid in self.state["trials"].items():
            idx = int(idx_s)
            h5 = self.pose_dir / f"{self.name}_arena_{idx:02d}.h5"
            if not h5.is_file() and h5.with_suffix(".empty").is_file():
                self.log(
                    f"score: arena {idx}: no animals found by the pose model, nothing to score"
                )
                empty.append(idx)
                continue
            if not h5.is_file():
                self.log(f"score: arena {idx}: no pose file, skipped")
                skipped.append(idx)
                continue
            fps = (
                con.execute("SELECT fps FROM trials WHERE trial_id=?", (tid,)).fetchone()[0] or 25.0
            )
            pose = load_sleap_h5(str(h5), fps)
            roles = assign_roles(pose, self.cfg.n_animals)
            feats = compute_features(pose, roles)
            fly_ids = {
                r[0]: r[1]
                for r in con.execute("SELECT role, fly_id FROM flies WHERE trial_id=?", (tid,))
            }
            for role, tr in roles.items():
                if role in fly_ids:
                    con.execute("UPDATE flies SET track_id=? WHERE fly_id=?", (tr, fly_ids[role]))
            fid = register_file(con, h5, "tracks_h5")
            con.execute(
                "DELETE FROM behaviour_events WHERE trial_id=? AND annotator_id=?", (tid, ann_id)
            )
            events = detect_events(feats, fps, params)
            con.executemany(
                """INSERT INTO behaviour_events
                   (trial_id, annotator_id, behaviour_id, actor_fly_id, target_fly_id,
                    start_s, end_s, confidence, source_file_id) VALUES (?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        tid,
                        ann_id,
                        beh[e.behaviour],
                        fly_ids.get(e.actor),
                        fly_ids.get(e.target),
                        e.start_s,
                        e.end_s,
                        e.confidence,
                        fid,
                    )
                    for e in events
                    if e.behaviour in beh
                ],
            )
            con.execute("DELETE FROM window_summary WHERE trial_id=?", (tid,))
            rows = window_summary(feats, fps, self.cfg.mm_per_pixel)
            con.executemany(
                """INSERT INTO window_summary
                   (trial_id, fly_id, window_start_s, window_s, speed_mm_s, distance_mm,
                    frac_tracked, n_encounters, tracks_file_id) VALUES (?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        tid,
                        fly_ids[r["role"]],
                        r["window_start_s"],
                        r["window_s"],
                        r["speed_mm_s"],
                        r["distance_mm"],
                        r["frac_tracked"],
                        r["n_encounters"],
                        fid,
                    )
                    for r in rows
                    if r["role"] in fly_ids
                ],
            )
            con.commit()
            n_ev += len(events)
            self.log(f"score: trial {tid}: {len(events)} events, roles {roles}")
        con.close()
        if skipped:
            raise RuntimeError(f"score: no pose file for arenas {skipped}; re-run from 'fetch'")
        self._mark("score", n_events=n_ev, n_empty=len(empty))

    # ── driver ─────────────────────────────────────────────────────────
    def run(self, stages: tuple[str, ...] = STAGES, poll_interval_s: float = 60) -> None:
        for st in stages:
            self.log(f"── {st} ── {datetime.now():%H:%M:%S}")
            if st == "poll":
                if self.poll(interval_s=poll_interval_s) == "failed":
                    raise RuntimeError("cluster job failed; see the SLURM logs")
            else:
                getattr(self, st)()
        self.log("session complete")
