"""Operator window: pick the recording, fill in the session, press Run.

Everything the window does goes through :class:`thermokourt.trial.runner.Session`,
so a session started here can be resumed from the command line and vice
versa. The pipeline runs in a worker thread; the log pane shows the same
lines ``thermokourt-trial run`` would print.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from thermokourt.db import ddb as ddbmod
from thermokourt.trial.config import RigConfig
from thermokourt.trial.runner import STAGES, Session, SessionSpec

EXPERIMENTS = ["PILOT", "E1", "E2", "E3", "E4", "E5", "E6", "E7"]
ROLES = ("male_1", "male_2", "female_decap")
N_ARENAS = 10


class Worker(QThread):
    line = pyqtSignal(str)
    finished_ok = pyqtSignal(bool)

    def __init__(self, cfg, spec, stages):
        super().__init__()
        self.cfg, self.spec, self.stages = cfg, spec, stages

    def run(self):
        try:
            Session(self.cfg, self.spec, log=self.line.emit).run(self.stages)
            self.finished_ok.emit(True)
        except Exception as e:  # shown in the log, never swallowed
            self.line.emit(f"ERROR: {e}\n{traceback.format_exc(limit=3)}")
            self.finished_ok.emit(False)


class MainWindow(QMainWindow):
    def __init__(self, cfg: RigConfig):
        super().__init__()
        self.cfg = cfg
        self.worker: Worker | None = None
        self.setWindowTitle("ThermoKourt — new recording session")
        self.vial_names: dict[str, str] = {}
        self._build()
        self._load_vials()

    # ── layout ─────────────────────────────────────────────────────────
    def _build(self):
        root = QWidget()
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)

        # recording
        rec_box = QGroupBox("1. Recording")
        rl = QHBoxLayout(rec_box)
        self.rec_edit = QLineEdit()
        self.rec_edit.setPlaceholderText(str(self.cfg.data_root))
        btn = QPushButton("Choose folder…")
        btn.clicked.connect(self._choose_rec)
        rl.addWidget(self.rec_edit)
        rl.addWidget(btn)
        lay.addWidget(rec_box)

        # session fields
        sess_box = QGroupBox("2. Session")
        form = QFormLayout(sess_box)
        self.exp = QComboBox()
        self.exp.addItems(EXPERIMENTS)
        self.group = QLineEdit()
        self.group.setPlaceholderText("coded group, e.g. G01 (no genotype names)")
        self.temp = QDoubleSpinBox()
        self.temp.setRange(10, 45)
        self.temp.setValue(25)
        self.temp.setSuffix(" °C")
        self.ramp = QDoubleSpinBox()
        self.ramp.setRange(0, 30)
        self.ramp.setSpecialValueText("static")
        self.ramp.setSuffix(" °C/h")
        self.rearing = QDoubleSpinBox()
        self.rearing.setRange(0, 35)
        self.rearing.setSpecialValueText("–")
        self.rearing.setValue(25)
        self.rearing.setSuffix(" °C")
        self.stimulus = QComboBox()
        self.stimulus.addItems(["triad", "two_males", "single_male"])
        self.treatment = QComboBox()
        self.treatment.addItems(["none", "arista_ablated", "sham"])
        self.operator = QLineEdit()
        self.operator.setPlaceholderText("initials")
        self.notes = QLineEdit()
        for label, w in (
            ("Experiment", self.exp),
            ("Group code", self.group),
            ("Test temperature", self.temp),
            ("Ramp", self.ramp),
            ("Rearing temperature", self.rearing),
            ("Stimulus", self.stimulus),
            ("Treatment", self.treatment),
            ("Operator", self.operator),
            ("Notes", self.notes),
        ):
            form.addRow(label, w)
        lay.addWidget(sess_box)

        # arenas
        ar_box = QGroupBox(
            "3. Arenas — vial print codes from DrosoDB (leave a row unticked if empty)"
        )
        al = QVBoxLayout(ar_box)
        self.table = QTableWidget(N_ARENAS, 5)
        self.table.setHorizontalHeaderLabels(
            ["use", "male 1", "male 2", "female (decap.)", "age (d)"]
        )
        self.completer = QCompleter([])
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        for r in range(N_ARENAS):
            cb = QCheckBox()
            cb.setChecked(True)
            self.table.setCellWidget(r, 0, cb)
            for c in (1, 2, 3):
                e = QLineEdit()
                e.setCompleter(self.completer)
                e.editingFinished.connect(self._refresh_vial_hint)
                self.table.setCellWidget(r, c, e)
            self.table.setItem(r, 4, QTableWidgetItem("3"))
            self.table.setVerticalHeaderItem(r, QTableWidgetItem(f"A{r:02d}"))
        self.table.resizeColumnsToContents()
        al.addWidget(self.table)
        self.vial_hint = QLabel("")
        al.addWidget(self.vial_hint)
        fill = QPushButton("Copy row A00 to all ticked rows")
        fill.clicked.connect(self._fill_down)
        al.addWidget(fill)
        lay.addWidget(ar_box)

        # run
        run_box = QGroupBox("4. Run")
        rl2 = QHBoxLayout(run_box)
        self.stage_from = QComboBox()
        self.stage_from.addItems(STAGES)
        self.stage_to = QComboBox()
        self.stage_to.addItems(STAGES)
        self.stage_to.setCurrentIndex(len(STAGES) - 1)
        self.run_btn = QPushButton("Save && Run")
        self.run_btn.clicked.connect(self._run)
        self.save_btn = QPushButton("Save only")
        self.save_btn.clicked.connect(lambda: self._save(show=True))
        rl2.addWidget(QLabel("from"))
        rl2.addWidget(self.stage_from)
        rl2.addWidget(QLabel("to"))
        rl2.addWidget(self.stage_to)
        rl2.addStretch()
        rl2.addWidget(self.save_btn)
        rl2.addWidget(self.run_btn)
        lay.addWidget(run_box)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        lay.addWidget(self.log, stretch=1)
        self.resize(900, 900)

    # ── DrosoDB ────────────────────────────────────────────────────────
    def _load_vials(self):
        snap = self.cfg.ddb_snapshot
        if not snap or not Path(snap).is_file():
            self._log(f"DrosoDB snapshot not found at {snap} — print codes cannot be checked")
            return
        try:
            ddb = ddbmod.open_snapshot(snap)
            rows = ddb.execute(
                "SELECT v.print_code, g.name FROM vial v JOIN genotype g ON g.id=v.genotype_id "
                "WHERE v.is_active = 1 ORDER BY v.print_code"
            ).fetchall()
            ddb.close()
        except Exception as e:
            self._log(f"could not read DrosoDB snapshot: {e}")
            return
        self.vial_names = {r[0]: r[1] for r in rows}
        self.completer.model().setStringList(list(self.vial_names))  # type: ignore[union-attr]
        self._log(f"DrosoDB snapshot: {len(rows)} active vials")

    def _refresh_vial_hint(self):
        codes = set()
        for r in range(N_ARENAS):
            for c in (1, 2, 3):
                t = self.table.cellWidget(r, c).text().strip()
                if t:
                    codes.add(t)
        known = {c: self.vial_names.get(c) for c in codes}
        parts = [
            f"{c} → {g}" if g else f"{c} → ? (not an active vial)" for c, g in sorted(known.items())
        ]
        self.vial_hint.setText("   ".join(parts))

    # ── helpers ────────────────────────────────────────────────────────
    def _log(self, s: str):
        self.log.appendPlainText(s)

    def _choose_rec(self):
        d = QFileDialog.getExistingDirectory(
            self, "Motif recording folder", str(self.cfg.data_root)
        )
        if d:
            self.rec_edit.setText(d)

    def _fill_down(self):
        src = [self.table.cellWidget(0, c).text() for c in (1, 2, 3)]
        age = self.table.item(0, 4).text()
        for r in range(1, N_ARENAS):
            if self.table.cellWidget(r, 0).isChecked():
                for c, v in zip((1, 2, 3), src, strict=True):
                    self.table.cellWidget(r, c).setText(v)
                self.table.item(r, 4).setText(age)
        self._refresh_vial_hint()

    def _spec(self) -> SessionSpec:
        rec = Path(self.rec_edit.text().strip())
        if not rec.is_dir():
            raise ValueError("choose a recording folder first")
        if not self.group.text().strip():
            raise ValueError("group code is required")
        arenas: dict[int, list[dict]] = {}
        n_roles = {"triad": 3, "two_males": 2, "single_male": 1}[self.stimulus.currentText()]
        for r in range(N_ARENAS):
            if not self.table.cellWidget(r, 0).isChecked():
                continue
            flies = []
            for c, role in zip((1, 2, 3), ROLES, strict=True):
                if c - 1 >= n_roles:
                    break
                code = self.table.cellWidget(r, c).text().strip()
                if not code:
                    raise ValueError(f"arena A{r:02d}: print code for {role} missing")
                if self.vial_names and code not in self.vial_names:
                    raise ValueError(f"arena A{r:02d}: {code} is not an active vial in DrosoDB")
                age = self.table.item(r, 4).text().strip()
                flies.append(
                    {"role": role, "print_code": code, "age_days": float(age) if age else None}
                )
            arenas[r] = flies
        if not arenas:
            raise ValueError("no arena ticked")
        return SessionSpec(
            recording_dir=rec,
            experiment_id=self.exp.currentText(),
            group_code=self.group.text().strip(),
            test_temp_c=self.temp.value(),
            stimulus=self.stimulus.currentText(),
            ramp_c_per_h=self.ramp.value() or None,
            rearing_temp_c=self.rearing.value() or None,
            treatment=self.treatment.currentText(),
            operator=self.operator.text().strip() or None,
            notes=self.notes.text().strip() or None,
            arenas=arenas,
        )

    def _save(self, show=False) -> SessionSpec | None:
        try:
            spec = self._spec()
        except ValueError as e:
            QMessageBox.warning(self, "Incomplete", str(e))
            return None
        out = spec.recording_dir / "session.json"
        out.write_text(json.dumps(spec.to_json(), indent=2))
        self._log(f"saved {out}")
        if show:
            QMessageBox.information(self, "Saved", f"Session saved to\n{out}")
        return spec

    def _run(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Busy", "A session is already running")
            return
        spec = self._save()
        if spec is None:
            return
        i0, i1 = self.stage_from.currentIndex(), self.stage_to.currentIndex() + 1
        if i0 >= i1:
            QMessageBox.warning(self, "Stages", "'from' must come before 'to'")
            return
        self.run_btn.setEnabled(False)
        self.worker = Worker(self.cfg, spec, STAGES[i0:i1])
        self.worker.line.connect(self._log)
        self.worker.finished_ok.connect(self._done)
        self.worker.start()

    def _done(self, ok: bool):
        self.run_btn.setEnabled(True)
        if ok:
            QMessageBox.information(self, "Done", "Session pipeline finished")
        else:
            QMessageBox.critical(
                self,
                "Failed",
                "The pipeline stopped with an error — see the log.\n"
                "Fix the cause and press Run again; finished stages are skipped.",
            )


def main(config_path: str | None = None):
    app = QApplication(sys.argv)
    try:
        cfg = RigConfig.load(config_path)
    except FileNotFoundError as e:
        QMessageBox.critical(None, "No rig configuration", str(e))
        return 1
    w = MainWindow(cfg)
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
