"""Validation helpers for the real BIDS/fMRIPrep dataset."""
from dataclasses import dataclass
from pathlib import Path

from bids import BIDSLayout
import nibabel as nib
import numpy as np
import pandas as pd


SPACE = "MNI152NLin2009cAsym"
MOTION_COLUMNS = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]


@dataclass(frozen=True)
class RunFiles:
    task: str
    run: str
    bold: Path
    events: Path
    confounds: Path
    mask: Path


def make_layout(bids_root):
    bids_root = Path(bids_root).resolve()
    derivatives = bids_root / "derivatives"
    return BIDSLayout(
        str(bids_root),
        derivatives=str(derivatives),
        validate=True,
    )


def _one(layout, label, **query):
    files = layout.get(return_type="file", **query)
    if len(files) != 1:
        raise ValueError(f"Expected exactly one {label}, found {len(files)}: {files}")
    return Path(files[0])


def get_run_files(layout, subject, session, task, run):
    common = {
        "subject": subject,
        "session": session,
        "task": task,
        "run": run,
    }
    return RunFiles(
        task=task,
        run=f"{int(run):02d}",
        bold=_one(
            layout,
            "preprocessed BOLD",
            **common,
            space=SPACE,
            desc="preproc",
            suffix="bold",
            extension=".nii.gz",
        ),
        events=_one(
            layout,
            "events table",
            **common,
            suffix="events",
            extension=".tsv",
        ),
        confounds=_one(
            layout,
            "confounds table",
            **common,
            desc="confounds",
            suffix="timeseries",
            extension=".tsv",
        ),
        mask=_one(
            layout,
            "brain mask",
            **common,
            space=SPACE,
            desc="brain",
            suffix="mask",
            extension=".nii.gz",
        ),
    )


def validate_run(run_files):
    bold_img = nib.load(run_files.bold)
    mask_img = nib.load(run_files.mask)
    events = pd.read_csv(run_files.events, sep="\t")
    confounds = pd.read_csv(run_files.confounds, sep="\t")

    if len(bold_img.shape) != 4:
        raise ValueError(f"BOLD must be 4D: {run_files.bold}")
    if mask_img.shape != bold_img.shape[:3]:
        raise ValueError(f"Mask/BOLD shape mismatch for run {run_files.run}")
    if not np.allclose(mask_img.affine, bold_img.affine):
        raise ValueError(f"Mask/BOLD affine mismatch for run {run_files.run}")
    if len(confounds) != bold_img.shape[3]:
        raise ValueError(
            f"Confounds have {len(confounds)} rows; BOLD has "
            f"{bold_img.shape[3]} volumes for run {run_files.run}"
        )
    required_confounds = set(MOTION_COLUMNS) | {"framewise_displacement"}
    missing_confounds = required_confounds - set(confounds.columns)
    if missing_confounds:
        raise ValueError(f"Missing confounds: {sorted(missing_confounds)}")
    acompcor = [c for c in confounds if c.startswith("a_comp_cor_")]
    if len(acompcor) < 5:
        raise ValueError(f"Expected at least 5 aCompCor columns, found {len(acompcor)}")

    id_columns = (
        ["imageid"] if run_files.task == "recognition" else ["cueid", "targetid"]
    )
    required_events = {"onset", "duration", *id_columns}
    missing_events = required_events - set(events.columns)
    if missing_events:
        raise ValueError(f"Missing event columns: {sorted(missing_events)}")
    if not events.onset.is_monotonic_increasing:
        raise ValueError(f"Events are not chronologically ordered in {run_files.events}")
    scan_duration = bold_img.shape[3] * float(bold_img.header.get_zooms()[3])
    if (events.onset < 0).any() or (
        events.onset + events.duration > scan_duration
    ).any():
        raise ValueError(f"Events fall outside scan bounds in {run_files.events}")

    return {
        "run_files": run_files,
        "bold_img": bold_img,
        "mask_img": mask_img,
        "events": events,
        "confounds": confounds,
        "tr": float(bold_img.header.get_zooms()[3]),
        "n_volumes": bold_img.shape[3],
        "n_fd_outliers": int(
            (confounds.framewise_displacement.fillna(0) > 0.3).sum()
        ),
    }


def validate_retrieval_dataset(bids_root, subject, session, runs=None):
    layout = make_layout(bids_root)
    available = layout.get(
        subject=subject,
        session=session,
        task="retrieval",
        target="run",
        return_type="id",
    )
    selected = available if runs is None else [
        run for run in available if f"{int(run):02d}" in {f"{int(r):02d}" for r in runs}
    ]
    if not selected:
        raise ValueError("No retrieval runs matched the requested selection")

    reports = [
        validate_run(get_run_files(layout, subject, session, "retrieval", run))
        for run in selected
    ]
    pair_sets = []
    for report in reports:
        events = report["events"]
        counts = events.groupby(["cueid", "targetid"]).size()
        if len(counts) != 20 or not (counts == 2).all():
            raise ValueError(
                f"Run {report['run_files'].run} must contain 20 pairs twice each"
            )
        cue_targets = events[["cueid", "targetid"]].drop_duplicates()
        if cue_targets.cueid.nunique() != len(cue_targets):
            raise ValueError("Each cue must map to exactly one target")
        pair_sets.append(set(map(tuple, cue_targets.to_numpy())))
    if len(pair_sets) > 1 and any(pairs != pair_sets[0] for pairs in pair_sets[1:]):
        raise ValueError("Retrieval cue-target pair sets differ across runs")
    return reports
