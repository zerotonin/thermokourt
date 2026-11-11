"""Pure parts of the session runner."""

from thermokourt.trial.runner import SessionSpec, job_states_ok


def test_job_states_ok_requires_all_completed():
    ok, counts = job_states_ok("COMPLETED\nCOMPLETED\n")
    assert ok and counts == {"COMPLETED": 2}
    ok, counts = job_states_ok("COMPLETED\nCANCELLED+\nCANCELLED by 123\nFAILED\n")
    assert not ok and counts == {"COMPLETED": 1, "CANCELLED": 2, "FAILED": 1}
    assert job_states_ok("") == (False, {})


def test_session_spec_roundtrip(tmp_path):
    spec = SessionSpec(
        recording_dir=tmp_path,
        experiment_id="E1",
        group_code="G01",
        test_temp_c=28.0,
        arenas={0: [{"role": "male_1", "print_code": "Ab13"}]},
    )
    back = SessionSpec.from_json(spec.to_json())
    assert back.recording_dir == tmp_path and back.arenas == {
        0: [{"role": "male_1", "print_code": "Ab13"}]
    }


def test_parse_worker_jobs():
    import pytest

    from thermokourt.trial.runner import parse_worker_jobs

    out = "worker 11 on aoraki_gpu_L40 (12 cpus, 64G)\n10 clips, 2 workers\nJOBS=11,12\n"
    assert parse_worker_jobs(out) == "11,12"
    with pytest.raises(RuntimeError, match="no workers"):
        parse_worker_jobs("sbatch refused a worker on x\nno worker could be submitted")


def test_progress_lines_are_aggregated(tmp_path, monkeypatch):
    import sys

    from thermokourt.trial.runner import Session

    script = tmp_path / "fake_cutter.py"
    script.write_text(
        "print('@progress arena=00 of=2 phase=crop frac=0.500', flush=True)\n"
        "print('some ffmpeg chatter')\n"
        "print('@progress arena=01 of=2 phase=done frac=1.000', flush=True)\n"
        "print('@progress arena=00 of=2 phase=done frac=1.000', flush=True)\n",
        encoding="utf-8",
    )
    logs = []
    fake = Session.__new__(Session)
    fake.log = logs.append
    fake._run_with_progress([sys.executable, str(script)], "cut")
    prog = [line for line in logs if line.startswith("@progress cut")]
    assert prog[-1].startswith("@progress cut 1.0000 2/2 arenas")
    assert any(" 0.7500 " in line for line in prog)


def test_progress_run_raises_with_output_tail(tmp_path):
    import sys

    import pytest

    from thermokourt.trial.runner import Session

    script = tmp_path / "bad.py"
    script.write_text("print('No arenas defined.')\nraise SystemExit(1)\n", encoding="utf-8")
    fake = Session.__new__(Session)
    fake.log = lambda s: None
    with pytest.raises(RuntimeError, match="No arenas defined"):
        fake._run_with_progress([sys.executable, str(script)], "cut")


def test_unfinished_skips_tracked_and_empty_arenas():
    from thermokourt.trial.runner import Session

    s = Session.__new__(Session)
    s.spec = None
    type(s).name = property(lambda self: "rec")
    s._remote_pose_names = lambda: {"rec_arena_00.h5", "rec_arena_02.empty", "rec_arena_03.slp"}
    try:
        assert s._unfinished([0, 1, 2, 3]) == [1, 3]
    finally:
        del type(s).name


def test_cancel_waiting_command_only_touches_pending_jobs():
    from thermokourt.trial.runner import cancel_waiting_command

    cmd = cancel_waiting_command("11,12")
    assert "squeue -j 11,12 -h -t PENDING" in cmd and cmd.endswith("true")
    assert "scancel 11,12" not in cmd


def test_poll_finishes_on_clips_even_if_workers_were_cancelled():
    from thermokourt.trial.runner import Session

    s = Session.__new__(Session)
    logs, marks, calls = [], {}, []
    s.log = logs.append
    s.state = {"job_id": "11,12", "done": {"submit": {}}}
    s.is_done = lambda st: st in s.state["done"] and st != "poll"
    s._mark = lambda st, **kw: marks.update({st: kw})
    s._clips_left = lambda: 0

    def ssh(cmd):
        calls.append(cmd)
        return ""  # nothing left in squeue: finished + cancelled 'not needed' workers

    s._ssh = ssh
    assert s.poll(interval_s=0) == "done"
    assert marks["poll"]["ok"] is True and not any("scancel" in c for c in calls)
