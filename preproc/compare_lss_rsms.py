"""Compare retrieval RSMs from working_glm.py and glm.py."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from rsm_plot import load_patterns


KEY_COLUMNS = ["run", "trial_idx", "cueid", "targetid"]


def load_glm_patterns(metadata_path, mask):
    """Load individual glm.py beta maps through the shared ROI mask."""
    metadata = pd.read_csv(metadata_path)
    metadata = metadata.loc[metadata.task == "retrieval"].reset_index(drop=True)
    if len(metadata) != 80:
        raise ValueError(f"Expected 80 glm.py retrieval betas, found {len(metadata)}")

    patterns = np.empty((len(metadata), int(mask.sum())), dtype=np.float32)
    for index, row in metadata.iterrows():
        image = nib.load(row.beta_path)
        if image.shape != mask.shape:
            raise ValueError(f"Beta/mask shape mismatch: {row.beta_path}")
        values = np.asarray(image.dataobj, dtype=np.float32)[mask]
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite beta values: {row.beta_path}")
        patterns[index] = values
    return metadata, patterns


def normalized_keys(metadata):
    keys = metadata[KEY_COLUMNS].copy()
    for column in KEY_COLUMNS:
        keys[column] = keys[column].astype(int)
    if keys.duplicated().any():
        raise ValueError("Trial alignment keys are not unique")
    return pd.MultiIndex.from_frame(keys)


def align_patterns(working_metadata, working_patterns, glm_metadata, glm_patterns):
    """Align both pipelines by run, trial index, cue ID, and target ID."""
    working_keys = normalized_keys(working_metadata)
    glm_keys = normalized_keys(glm_metadata)
    if set(working_keys) != set(glm_keys):
        missing_from_glm = sorted(set(working_keys) - set(glm_keys))
        missing_from_working = sorted(set(glm_keys) - set(working_keys))
        raise ValueError(
            "Pipeline trial keys differ. "
            f"Missing from glm.py: {missing_from_glm}; "
            f"missing from working_glm.py: {missing_from_working}"
        )

    glm_lookup = pd.Series(np.arange(len(glm_keys)), index=glm_keys)
    glm_order = glm_lookup.loc[working_keys].to_numpy()
    aligned_metadata = working_metadata.copy().reset_index(drop=True)
    for column in KEY_COLUMNS:
        aligned_metadata[column] = aligned_metadata[column].astype(int)
    return aligned_metadata, working_patterns, glm_patterns[glm_order]


def trial_index_rsms(metadata, working_patterns, glm_patterns):
    order = metadata.sort_values(["trial_idx", "run"]).index.to_numpy()
    ordered_metadata = metadata.loc[order].reset_index(drop=True)
    return (
        np.corrcoef(working_patterns[order]),
        np.corrcoef(glm_patterns[order]),
        ordered_metadata,
    )


def cue_rsms(metadata, working_patterns, glm_patterns):
    labels = (
        metadata[["cueid", "targetid"]]
        .drop_duplicates()
        .sort_values(["cueid", "targetid"])
        .reset_index(drop=True)
    )
    working_averages = []
    glm_averages = []
    for condition in labels.itertuples(index=False):
        selected = (
            (metadata.cueid == int(condition.cueid))
            & (metadata.targetid == int(condition.targetid))
        ).to_numpy()
        if selected.sum() != 4:
            raise ValueError(
                f"Expected four trials for cue {condition.cueid}; "
                f"found {selected.sum()}"
            )
        working_averages.append(working_patterns[selected].mean(axis=0))
        glm_averages.append(glm_patterns[selected].mean(axis=0))
    return (
        np.corrcoef(np.stack(working_averages)),
        np.corrcoef(np.stack(glm_averages)),
        labels,
    )


def correlate_rsms(first, second):
    if first.shape != second.shape or first.shape[0] != first.shape[1]:
        raise ValueError("RSMs must be square matrices with matching shapes")
    upper = np.triu_indices(first.shape[0], k=1)
    result = pearsonr(first[upper], second[upper])
    return {
        "pearson_r": float(result.statistic),
        "pearson_p_two_sided": float(result.pvalue),
        "n_unique_off_diagonal_cells": int(len(first[upper])),
    }


def save_matrix(matrix, names, output_path):
    pd.DataFrame(matrix, index=names, columns=names).to_csv(output_path)


def plot_side_by_side(
        working_rsm, glm_rsm, tick_positions, tick_labels,
        axis_label, title, correlation, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    for axis, matrix, panel_title in zip(
            axes, [working_rsm, glm_rsm], ["working_glm.py", "glm.py"]):
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
        axis.set_xticks(tick_positions, tick_labels, rotation=90, fontsize=6)
        axis.set_yticks(tick_positions, tick_labels, fontsize=6)
        axis.set_xlabel(axis_label)
        axis.set_ylabel(axis_label)
        axis.set_title(panel_title)
    fig.suptitle(f"{title}; RSM correlation r={correlation['pearson_r']:.4f}")
    fig.colorbar(image, ax=axes, shrink=0.8, label="Pearson correlation")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(
        bids_root, working_beta_root, glm_beta_root, output_dir,
        subject="03", session="001", atlas_data_dir=None):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    working_metadata, working_patterns, mask, affine = load_patterns(
        bids_root,
        working_beta_root,
        subject,
        session,
        atlas_data_dir=atlas_data_dir,
    )
    glm_metadata_path = (
        Path(glm_beta_root)
        / f"beta_maps_metadata_sub-{subject}_ses-{session}.csv"
    )
    glm_metadata, glm_patterns = load_glm_patterns(glm_metadata_path, mask)
    metadata, working_patterns, glm_patterns = align_patterns(
        working_metadata, working_patterns, glm_metadata, glm_patterns
    )

    working_trial, glm_trial, trial_order = trial_index_rsms(
        metadata, working_patterns, glm_patterns
    )
    working_cue, glm_cue, cue_labels = cue_rsms(
        metadata, working_patterns, glm_patterns
    )
    metrics = {
        "trial_index_rsms": correlate_rsms(working_trial, glm_trial),
        "cue_averaged_rsms": correlate_rsms(working_cue, glm_cue),
        "roi_voxels": int(mask.sum()),
        "event_duration_seconds": 4.0,
    }

    trial_names = [
        (
            f"trial-{int(row.trial_idx):02d}_run-{int(row.run):02d}_"
            f"cue-{int(row.cueid):02d}"
        )
        for row in trial_order.itertuples(index=False)
    ]
    cue_names = [
        f"cue-{int(row.cueid):02d}_target-{int(row.targetid)}"
        for row in cue_labels.itertuples(index=False)
    ]
    save_matrix(
        working_trial, trial_names, output_dir / "working_glm_trial_index_rsm.csv"
    )
    save_matrix(glm_trial, trial_names, output_dir / "glm_trial_index_rsm.csv")
    save_matrix(working_cue, cue_names, output_dir / "working_glm_cue_rsm.csv")
    save_matrix(glm_cue, cue_names, output_dir / "glm_cue_rsm.csv")
    trial_order.to_csv(output_dir / "trial_index_order.tsv", sep="\t", index=False)
    cue_labels.to_csv(output_dir / "cue_order.tsv", sep="\t", index=False)
    nib.save(
        nib.Nifti1Image(mask.astype(np.uint8), affine),
        output_dir / "comparison_visual_lateral_parietal_mask.nii.gz",
    )
    with (output_dir / "rsm_correlations.json").open("w") as file:
        json.dump(metrics, file, indent=2)

    trial_indices = trial_order.trial_idx.drop_duplicates().astype(int).tolist()
    plot_side_by_side(
        working_trial,
        glm_trial,
        np.arange(len(trial_indices)) * 2 + 0.5,
        trial_indices,
        "Trial index (run 1 and run 2 adjacent)",
        "Trial-index-ordered retrieval RSMs",
        metrics["trial_index_rsms"],
        output_dir / "trial_index_rsms_side_by_side.png",
    )
    cueids = cue_labels.cueid.astype(int).tolist()
    plot_side_by_side(
        working_cue,
        glm_cue,
        np.arange(len(cueids)),
        cueids,
        "Cue ID",
        "Cue-averaged retrieval RSMs",
        metrics["cue_averaged_rsms"],
        output_dir / "cue_rsms_side_by_side.png",
    )
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--bids-root", default=str(script_dir / "bids_data"))
    parser.add_argument(
        "--working-beta-root",
        default=str(script_dir / "working_glm_results" / "retrieval"),
    )
    parser.add_argument(
        "--glm-beta-root",
        default=str(script_dir / "glm_comparison_results" / "retrieval"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(script_dir / "glm_comparison_results" / "rsm_comparison"),
    )
    parser.add_argument("--subject", default="03")
    parser.add_argument("--session", default="001")
    parser.add_argument(
        "--atlas-data-dir", default=str(script_dir / "atlas_data")
    )
    args = parser.parse_args()
    main(**vars(args))
