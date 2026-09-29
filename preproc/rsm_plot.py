import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.image import resample_to_img

SPACE = "MNI152NLin2009cAsym"
ROI_LABELS = [
    "Lateral Occipital Cortex, superior division",
    "Lateral Occipital Cortex, inferior division",
    "Intracalcarine Cortex",
    "Cuneal Cortex",
    "Lingual Gyrus",
    "Temporal Occipital Fusiform Cortex",
    "Occipital Fusiform Gyrus",
    "Supracalcarine Cortex",
    "Occipital Pole",
    "Superior Parietal Lobule",
    "Supramarginal Gyrus, anterior division",
    "Supramarginal Gyrus, posterior division",
    "Angular Gyrus",
]


def run_files(bids_root, beta_root, subject, session, run):
    run = f"{int(run):02d}"
    stem = f"sub-{subject}_ses-{session}_task-retrieval_run-{run}"
    derivative_func = (
        Path(bids_root) / "derivatives" / f"sub-{subject}"
        / f"ses-{session}" / "func"
    )
    return {
        "betas": Path(beta_root) / "beta_maps"
        / f"{stem}_desc-lss_betas.nii.gz",
        "metadata": Path(beta_root) / "beta_maps"
        / f"{stem}_desc-lss_metadata.tsv",
        "mask": derivative_func
        / f"{stem}_space-{SPACE}_desc-brain_mask.nii.gz",
    }


def load_patterns(
        bids_root, beta_root, subject, session, atlas_data_dir=None):
    """Load trial betas within visual and lateral parietal cortex."""
    files_by_run = [
        run_files(bids_root, beta_root, subject, session, run)
        for run in ("01", "02")
    ]
    missing = [
        str(file_path)
        for files in files_by_run
        for file_path in files.values()
        if not file_path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing GLM/RSM inputs: {missing}")

    masks = [nib.load(files["mask"]) for files in files_by_run]
    reference = masks[0]
    for mask in masks[1:]:
        if mask.shape != reference.shape or not np.allclose(
                mask.affine, reference.affine):
            raise ValueError("Retrieval masks do not have matching geometry")
    intersection = np.logical_and.reduce(
        [np.asarray(mask.dataobj) > 0 for mask in masks]
    )
    if not intersection.any():
        raise ValueError("Retrieval masks have an empty intersection")

    atlas = fetch_atlas_harvard_oxford(
        "cort-maxprob-thr25-2mm", data_dir=atlas_data_dir
    )
    missing_labels = set(ROI_LABELS) - set(atlas.labels)
    if missing_labels:
        raise ValueError(f"Atlas labels not found: {sorted(missing_labels)}")
    label_indices = [atlas.labels.index(label) for label in ROI_LABELS]
    resampled_atlas = resample_to_img(
        atlas.maps,
        reference,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    roi_mask = np.isin(
        np.asarray(resampled_atlas.dataobj, dtype=np.int16), label_indices
    )
    roi_mask &= intersection
    if not roi_mask.any():
        raise ValueError("Visual and lateral parietal ROI is empty")

    metadata_frames = []
    pattern_frames = []
    for files in files_by_run:
        metadata = pd.read_csv(files["metadata"], sep="\t")
        beta_img = nib.load(files["betas"])
        if beta_img.shape[:3] != roi_mask.shape:
            raise ValueError(f"Beta/mask shape mismatch: {files['betas']}")
        if beta_img.shape[3] != len(metadata):
            raise ValueError(
                f"Beta count does not match metadata: {files['betas']}"
            )
        patterns = np.asarray(
            beta_img.dataobj, dtype=np.float32
        )[roi_mask, :].T
        if not np.isfinite(patterns).all():
            raise ValueError(f"Non-finite beta values: {files['betas']}")
        metadata_frames.append(metadata)
        pattern_frames.append(patterns)

    metadata = pd.concat(metadata_frames, ignore_index=True)
    patterns = np.vstack(pattern_frames)
    return metadata, patterns, roi_mask, reference.affine


def calculate_rsms(metadata, patterns):
    """Calculate trial, condition-average, and cross-run retrieval RSMs."""
    order = metadata.sort_values(
        ["cueid", "targetid", "run", "trial_idx"]
    ).index.to_numpy()
    trial_metadata = metadata.loc[order].reset_index(drop=True)
    trial_rsm = np.corrcoef(patterns[order])

    labels = (
        metadata[["cueid", "targetid"]]
        .drop_duplicates()
        .sort_values(["cueid", "targetid"])
        .reset_index(drop=True)
    )
    run_patterns = {}
    for run in (1, 2):
        averaged = []
        for condition in labels.itertuples(index=False):
            selected = (
                (metadata.run.astype(int) == run)
                & (metadata.cueid.astype(int) == int(condition.cueid))
                & (metadata.targetid.astype(int) == int(condition.targetid))
            )
            if selected.sum() != 2:
                raise ValueError(
                    f"Expected two run-{run} trials for cue {condition.cueid}"
                )
            averaged.append(patterns[selected.to_numpy()].mean(axis=0))
        run_patterns[run] = np.stack(averaged)

    pooled = (run_patterns[1] + run_patterns[2]) / 2
    condition_rsm = np.corrcoef(pooled)
    stacked = np.vstack([run_patterns[1], run_patterns[2]])
    cross = np.corrcoef(stacked)[:len(labels), len(labels):]
    cross_rsm = (cross + cross.T) / 2
    return trial_rsm, trial_metadata, condition_rsm, cross_rsm, labels


def matrix_frame(matrix, labels):
    names = [
        f"cue-{int(row.cueid):02d}_target-{int(row.targetid)}"
        for row in labels.itertuples(index=False)
    ]
    return pd.DataFrame(matrix, index=names, columns=names)


def plot_trial_rsm(matrix, metadata, output_path):
    cueids = metadata.cueid.astype(int).drop_duplicates().tolist()
    fig, axis = plt.subplots(figsize=(10, 9), constrained_layout=True)
    image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    centers = np.arange(len(cueids)) * 4 + 1.5
    axis.set_xticks(centers, cueids, rotation=90, fontsize=7)
    axis.set_yticks(centers, cueids, fontsize=7)
    for boundary in np.arange(4, len(matrix), 4) - 0.5:
        axis.axhline(boundary, color="black", linewidth=0.3, alpha=0.5)
        axis.axvline(boundary, color="black", linewidth=0.3, alpha=0.5)
    axis.set_xlabel("Cue ID (four trials per cue-target pair)")
    axis.set_ylabel("Cue ID (four trials per cue-target pair)")
    axis.set_title("Trial-level retrieval RSM (visual + lateral parietal ROI)")
    fig.colorbar(image, ax=axis, shrink=0.8, label="Pearson correlation")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_condition_rsms(condition_rsm, cross_rsm, labels, output_path):
    cueids = labels.cueid.astype(int).astype(str).tolist()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    titles = [
        "Condition-averaged retrieval RSM",
        "Cross-run retrieval RSM",
    ]
    for axis, matrix, title in zip(
            axes, [condition_rsm, cross_rsm], titles):
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
        axis.set_xticks(range(len(cueids)), cueids, rotation=90, fontsize=7)
        axis.set_yticks(range(len(cueids)), cueids, fontsize=7)
        axis.set_xlabel("Cue ID")
        axis.set_ylabel("Cue ID")
        axis.set_title(title)
    fig.colorbar(image, ax=axes, shrink=0.8, label="Pearson correlation")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rsms(
        bids_root, beta_root, output_dir, subject="03", session="001",
        atlas_data_dir=None):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, patterns, mask, affine = load_patterns(
        bids_root, beta_root, subject, session,
        atlas_data_dir=atlas_data_dir,
    )
    trial_rsm, trial_metadata, condition_rsm, cross_rsm, labels = (
        calculate_rsms(metadata, patterns)
    )

    trial_names = [
        (
            f"cue-{int(row.cueid):02d}_target-{int(row.targetid)}_"
            f"run-{int(row.run):02d}_trial-{int(row.trial_idx):03d}"
        )
        for row in trial_metadata.itertuples(index=False)
    ]
    pd.DataFrame(
        trial_rsm, index=trial_names, columns=trial_names
    ).to_csv(output_dir / "retrieval_trial_rsm.csv")
    matrix_frame(condition_rsm, labels).to_csv(
        output_dir / "retrieval_condition_rsm.csv"
    )
    matrix_frame(cross_rsm, labels).to_csv(
        output_dir / "retrieval_cross_run_rsm.csv"
    )
    trial_metadata.to_csv(
        output_dir / "retrieval_trial_order.tsv", sep="\t", index=False
    )
    nib.save(
        nib.Nifti1Image(mask.astype(np.uint8), affine),
        output_dir / "retrieval_visual_lateral_parietal_mask.nii.gz",
    )
    plot_trial_rsm(
        trial_rsm, trial_metadata, output_dir / "retrieval_trial_rsm.png"
    )
    plot_condition_rsms(
        condition_rsm,
        cross_rsm,
        labels,
        output_dir / "retrieval_condition_rsms.png",
    )
    print(f"Saved retrieval RSM outputs to {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--bids-root", default=str(script_dir / "bids_data"))
    parser.add_argument(
        "--beta-root",
        default=str(script_dir / "working_glm_results" / "retrieval"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            script_dir / "working_glm_results" / "retrieval" / "rsm_visual_lpc"
        ),
    )
    parser.add_argument(
        "--atlas-data-dir", default=str(script_dir / "atlas_data")
    )
    parser.add_argument("--subject", default="03")
    parser.add_argument("--session", default="001")
    args = parser.parse_args()
    plot_rsms(**vars(args))