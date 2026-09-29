"""Build retrieval RSMs from real-data trial beta maps."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from real_data import get_run_files, make_layout
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.image import resample_to_img


VISUAL_ATLAS_LABELS = [
    "Lateral Occipital Cortex, superior division",
    "Lateral Occipital Cortex, inferior division",
    "Intracalcarine Cortex",
    "Cuneal Cortex",
    "Lingual Gyrus",
    "Temporal Occipital Fusiform Cortex",
    "Occipital Fusiform Gyrus",
    "Supracalcarine Cortex",
    "Occipital Pole",
]

LATERAL_PARIETAL_ATLAS_LABELS = [
    "Superior Parietal Lobule",
    "Supramarginal Gyrus, anterior division",
    "Supramarginal Gyrus, posterior division",
    "Angular Gyrus",
]


def validate_beta_metadata(metadata):
    required = {
        "subject", "session", "task", "run", "trial_idx",
        "cueid", "targetid", "onset", "beta_path",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing beta metadata columns: {sorted(missing)}")
    rows = metadata[metadata.task == "retrieval"].copy()
    if len(rows) != 80:
        raise ValueError(f"Expected 80 retrieval betas, found {len(rows)}")
    rows["run"] = rows.run.astype(int)
    rows["cueid"] = rows.cueid.astype(int)
    rows["targetid"] = rows.targetid.astype(int)
    if sorted(rows.run.unique()) != [1, 2]:
        raise ValueError(f"Expected runs 1 and 2, found {sorted(rows.run.unique())}")
    counts = rows.groupby(["run", "cueid", "targetid"]).size()
    if len(counts) != 40 or not (counts == 2).all():
        raise ValueError("Each cue-target pair must occur twice in each run")
    mapping = rows[["cueid", "targetid"]].drop_duplicates()
    if len(mapping) != 20 or mapping.cueid.nunique() != 20:
        raise ValueError("Expected a one-to-one mapping for 20 cues")
    return rows.sort_values(["run", "trial_idx"]).reset_index(drop=True)


def retrieval_intersection_mask(bids_root, subject, session):
    layout = make_layout(bids_root)
    mask_images = []
    for run in ("01", "02"):
        path = get_run_files(
            layout, subject, session, "retrieval", run
        ).mask
        mask_images.append(nib.load(path))
    reference = mask_images[0]
    for image in mask_images[1:]:
        if image.shape != reference.shape or not np.allclose(
                image.affine, reference.affine):
            raise ValueError("Retrieval masks do not share geometry")
    intersection = np.logical_and.reduce(
        [image.get_fdata() > 0 for image in mask_images]
    )
    if not intersection.any():
        raise ValueError("Retrieval run masks have an empty intersection")
    return intersection, reference.affine, [int((img.get_fdata() > 0).sum())
                                             for img in mask_images]


def retrieval_atlas_mask(
        bids_root, subject, session, atlas_labels, atlas_data_dir=None):
    """Create an anatomical Harvard-Oxford ROI on the retrieval BOLD grid."""
    intersection, affine, run_mask_voxels = retrieval_intersection_mask(
        bids_root, subject, session
    )
    target = nib.Nifti1Image(intersection.astype(np.uint8), affine)
    atlas = fetch_atlas_harvard_oxford(
        "cort-maxprob-thr25-2mm",
        data_dir=atlas_data_dir,
    )
    missing = set(atlas_labels) - set(atlas.labels)
    if missing:
        raise ValueError(f"Atlas labels not found: {sorted(missing)}")
    label_indices = [
        atlas.labels.index(label) for label in atlas_labels
    ]
    resampled = resample_to_img(
        atlas.maps,
        target,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    roi_mask = np.isin(
        np.asarray(resampled.dataobj, dtype=np.int16), label_indices
    )
    roi_mask &= intersection
    if not roi_mask.any():
        raise ValueError("Atlas ROI does not overlap the retrieval masks")
    return roi_mask, affine, run_mask_voxels


def retrieval_visual_mask(
        bids_root, subject, session, atlas_data_dir=None):
    return retrieval_atlas_mask(
        bids_root,
        subject,
        session,
        VISUAL_ATLAS_LABELS,
        atlas_data_dir=atlas_data_dir,
    )


def retrieval_lateral_parietal_mask(
        bids_root, subject, session, atlas_data_dir=None):
    return retrieval_atlas_mask(
        bids_root,
        subject,
        session,
        LATERAL_PARIETAL_ATLAS_LABELS,
        atlas_data_dir=atlas_data_dir,
    )


def retrieval_visual_lpc_mask(
        bids_root, subject, session, atlas_data_dir=None):
    return retrieval_atlas_mask(
        bids_root,
        subject,
        session,
        VISUAL_ATLAS_LABELS + LATERAL_PARIETAL_ATLAS_LABELS,
        atlas_data_dir=atlas_data_dir,
    )


def extract_trial_patterns(metadata, mask):
    patterns = np.empty((len(metadata), int(mask.sum())), dtype=np.float32)
    for index, row in metadata.iterrows():
        image = nib.load(row.beta_path)
        if image.shape != mask.shape:
            raise ValueError(f"Beta/mask shape mismatch: {row.beta_path}")
        values = image.get_fdata(dtype=np.float32)[mask]
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite beta values: {row.beta_path}")
        patterns[index] = values
    return patterns


def normalize_trial_patterns(trial_patterns, method):
    if method == "none":
        return trial_patterns
    if method != "trial-zscore":
        raise ValueError(f"Unknown pattern normalization: {method}")
    means = trial_patterns.mean(axis=1, keepdims=True)
    standard_deviations = trial_patterns.std(axis=1, keepdims=True)
    if np.any(standard_deviations == 0):
        raise ValueError("Cannot z-score a constant trial pattern")
    return (trial_patterns - means) / standard_deviations


def condition_patterns(metadata, trial_patterns):
    labels = (
        metadata[["cueid", "targetid"]]
        .drop_duplicates()
        .sort_values("cueid")
        .reset_index(drop=True)
    )
    by_run = {}
    for run in (1, 2):
        run_patterns = []
        for row in labels.itertuples(index=False):
            selected = (
                (metadata.run == run)
                & (metadata.cueid == row.cueid)
                & (metadata.targetid == row.targetid)
            )
            if selected.sum() != 2:
                raise ValueError(
                    f"Expected two run-{run} betas for cue {row.cueid}"
                )
            run_patterns.append(trial_patterns[selected.to_numpy()].mean(axis=0))
        by_run[run] = np.stack(run_patterns)
    return labels, by_run


def compute_trial_rsm(metadata, trial_patterns):
    """Order four repetitions per stimulus contiguously and correlate trials."""
    order = (
        metadata
        .sort_values(["cueid", "targetid", "run", "trial_idx"])
        .index.to_numpy()
    )
    ordered_metadata = metadata.loc[order].reset_index(drop=True).copy()
    ordered_metadata["stimulus_repeat"] = (
        ordered_metadata.groupby(["cueid", "targetid"]).cumcount() + 1
    )
    ordered_patterns = trial_patterns[order]
    return np.corrcoef(ordered_patterns), ordered_metadata


def compute_stimulus_grouped_rsm(metadata, trial_patterns):
    """Order trials by target stimulus ID, then run and trial index."""
    order = (
        metadata
        .sort_values(["targetid", "run", "trial_idx"])
        .index.to_numpy()
    )
    ordered_metadata = metadata.loc[order].reset_index(drop=True).copy()
    ordered_metadata["stimulus_repeat"] = (
        ordered_metadata.groupby("targetid").cumcount() + 1
    )
    ordered_patterns = trial_patterns[order]
    return np.corrcoef(ordered_patterns), ordered_metadata


def compute_trial_index_rsm(metadata, trial_patterns):
    """Group trials by within-run trial index, with runs adjacent."""
    order = metadata.sort_values(["trial_idx", "run"]).index.to_numpy()
    ordered_metadata = metadata.loc[order].reset_index(drop=True).copy()
    ordered_patterns = trial_patterns[order]
    return np.corrcoef(ordered_patterns), ordered_metadata


def compute_rsms(run_patterns):
    run_1 = run_patterns[1]
    run_2 = run_patterns[2]
    combined = (run_1 + run_2) / 2
    conventional = np.corrcoef(combined)
    all_patterns = np.vstack([run_1, run_2])
    cross = np.corrcoef(all_patterns)[:len(run_1), len(run_1):]
    cross_symmetric = (cross + cross.T) / 2
    return conventional, cross_symmetric


def reliability_metrics(cross_rsm, permutations=10000, seed=42):
    diagonal = np.diag(cross_rsm)
    off_diagonal = cross_rsm[~np.eye(len(cross_rsm), dtype=bool)]
    observed = float(diagonal.mean())
    rng = np.random.default_rng(seed)
    null = np.empty(permutations)
    for index in range(permutations):
        null[index] = np.diag(cross_rsm[:, rng.permutation(len(cross_rsm))]).mean()
    return {
        "matched_mean": observed,
        "matched_median": float(np.median(diagonal)),
        "unmatched_mean": float(off_diagonal.mean()),
        "unmatched_median": float(np.median(off_diagonal)),
        "matched_minus_unmatched": float(observed - off_diagonal.mean()),
        "permutation_p_one_sided": float(
            (1 + np.count_nonzero(null >= observed)) / (permutations + 1)
        ),
        "n_conditions": int(len(cross_rsm)),
        "n_permutations": int(permutations),
    }


def within_run_repeat_metrics(metadata, trial_patterns):
    rows = metadata.copy()
    rows["repeat"] = rows.groupby(
        ["run", "cueid", "targetid"]
    ).cumcount()
    results = {}
    for run in (1, 2):
        repeat_patterns = []
        for repeat in (0, 1):
            indices = (
                rows[(rows.run == run) & (rows["repeat"] == repeat)]
                .sort_values("cueid")
                .index.to_numpy()
            )
            repeat_patterns.append(trial_patterns[indices])
        stacked = np.vstack(repeat_patterns)
        cross = np.corrcoef(stacked)[:20, 20:]
        results[f"run_{run}"] = reliability_metrics((cross + cross.T) / 2)
    return results


def trial_similarity_distributions(metadata, trial_patterns):
    """Collect within- and across-stimulus trial-pattern correlations."""
    correlations = np.corrcoef(trial_patterns)
    records = []
    for first in range(len(metadata)):
        for second in range(first + 1, len(metadata)):
            same_stimulus = (
                metadata.loc[first, "cueid"] == metadata.loc[second, "cueid"]
                and metadata.loc[first, "targetid"]
                == metadata.loc[second, "targetid"]
            )
            scopes = ["all_pairs"]
            if metadata.loc[first, "run"] != metadata.loc[second, "run"]:
                scopes.append("cross_run")
            for scope in scopes:
                records.append({
                    "scope": scope,
                    "pair_type": "within_stimulus"
                    if same_stimulus else "across_stimulus",
                    "correlation": float(correlations[first, second]),
                    "first_run": int(metadata.loc[first, "run"]),
                    "second_run": int(metadata.loc[second, "run"]),
                    "first_cueid": int(metadata.loc[first, "cueid"]),
                    "second_cueid": int(metadata.loc[second, "cueid"]),
                })
    return pd.DataFrame(records)


def summarize_similarity_distributions(distributions):
    summary = {}
    for scope, scope_rows in distributions.groupby("scope"):
        values = {
            pair_type: rows.correlation.to_numpy()
            for pair_type, rows in scope_rows.groupby("pair_type")
        }
        within = values["within_stimulus"]
        across = values["across_stimulus"]
        summary[scope] = {
            "within_n": int(len(within)),
            "within_mean": float(within.mean()),
            "within_median": float(np.median(within)),
            "across_n": int(len(across)),
            "across_mean": float(across.mean()),
            "across_median": float(np.median(across)),
            "within_minus_across_mean": float(within.mean() - across.mean()),
        }
    return summary


def plot_similarity_histograms(distributions, output_path, roi_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, scope, title in zip(
            axes,
            ["all_pairs", "cross_run"],
            [
                f"All trial pairs ({roi_name})",
                f"Cross-run trial pairs only ({roi_name})",
            ]):
        rows = distributions[distributions.scope == scope]
        lower, upper = rows.correlation.min(), rows.correlation.max()
        bins = np.linspace(lower, upper, 55)
        for pair_type, color, label in [
                ("within_stimulus", "#2166ac", "Within stimulus"),
                ("across_stimulus", "#b2182b", "Across stimulus")]:
            values = rows.loc[
                rows.pair_type == pair_type, "correlation"
            ].to_numpy()
            axis.hist(
                values,
                bins=bins,
                density=True,
                alpha=0.55,
                color=color,
                label=f"{label} (n={len(values)})",
            )
            axis.axvline(
                values.mean(), color=color, linestyle="--", linewidth=1.5
            )
        axis.set_title(title)
        axis.set_xlabel("Pearson correlation between trial beta maps")
        axis.set_ylabel("Density")
        axis.legend(fontsize=8)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_matrix(matrix, labels, path):
    names = [
        f"cue-{int(row.cueid):02d}_target-{int(row.targetid)}"
        for row in labels.itertuples(index=False)
    ]
    pd.DataFrame(matrix, index=names, columns=names).to_csv(path)


def save_trial_matrix(matrix, ordered_metadata, path):
    names = [
        (
            f"cue-{int(row.cueid):02d}_target-{int(row.targetid)}_"
            f"run-{int(row.run):02d}_repeat-{int(row.stimulus_repeat)}"
        )
        for row in ordered_metadata.itertuples(index=False)
    ]
    pd.DataFrame(matrix, index=names, columns=names).to_csv(path)


def plot_trial_rsm(matrix, ordered_metadata, output_path, roi_name):
    cueids = (
        ordered_metadata[["cueid"]]
        .drop_duplicates()
        .cueid.astype(int)
        .tolist()
    )
    fig, axis = plt.subplots(figsize=(10, 9), constrained_layout=True)
    image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    centers = np.arange(len(cueids)) * 4 + 1.5
    axis.set_xticks(centers, cueids, rotation=90, fontsize=7)
    axis.set_yticks(centers, cueids, fontsize=7)
    for boundary in np.arange(4, len(matrix), 4) - 0.5:
        axis.axhline(boundary, color="black", linewidth=0.3, alpha=0.5)
        axis.axvline(boundary, color="black", linewidth=0.3, alpha=0.5)
    axis.set_xlabel("Cue ID (four trials per stimulus)")
    axis.set_ylabel("Cue ID (four trials per stimulus)")
    axis.set_title(f"Trial-level retrieval RSM ({roi_name})")
    fig.colorbar(image, ax=axis, shrink=0.8, label="Pearson correlation")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_stimulus_grouped_rsm(
        matrix, ordered_metadata, output_path, roi_name):
    stimulus_ids = (
        ordered_metadata[["targetid"]]
        .drop_duplicates()
        .targetid.astype(int)
        .tolist()
    )
    fig, axis = plt.subplots(figsize=(10, 9), constrained_layout=True)
    image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    centers = np.arange(len(stimulus_ids)) * 4 + 1.5
    axis.set_xticks(centers, stimulus_ids, rotation=90, fontsize=6)
    axis.set_yticks(centers, stimulus_ids, fontsize=6)
    for boundary in np.arange(4, len(matrix), 4) - 0.5:
        axis.axhline(boundary, color="black", linewidth=0.3, alpha=0.5)
        axis.axvline(boundary, color="black", linewidth=0.3, alpha=0.5)
    axis.set_xlabel("Stimulus ID (four trials per stimulus)")
    axis.set_ylabel("Stimulus ID (four trials per stimulus)")
    axis.set_title(f"Stimulus-grouped retrieval RSM ({roi_name})")
    fig.colorbar(image, ax=axis, shrink=0.8, label="Pearson correlation")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_trial_index_rsm(matrix, ordered_metadata, output_path, roi_name):
    trial_indices = (
        ordered_metadata[["trial_idx"]]
        .drop_duplicates()
        .trial_idx.astype(int)
        .tolist()
    )
    fig, axis = plt.subplots(figsize=(10, 9), constrained_layout=True)
    image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    centers = np.arange(len(trial_indices)) * 2 + 0.5
    axis.set_xticks(centers, trial_indices, rotation=90, fontsize=6)
    axis.set_yticks(centers, trial_indices, fontsize=6)
    for boundary in np.arange(2, len(matrix), 2) - 0.5:
        axis.axhline(boundary, color="black", linewidth=0.2, alpha=0.35)
        axis.axvline(boundary, color="black", linewidth=0.2, alpha=0.35)
    axis.set_xlabel("Trial index (run 1, then run 2 within each index)")
    axis.set_ylabel("Trial index (run 1, then run 2 within each index)")
    axis.set_title(f"Trial-index-ordered retrieval RSM ({roi_name})")
    fig.colorbar(image, ax=axis, shrink=0.8, label="Pearson correlation")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rsms(conventional, cross_rsm, labels, output_path, roi_name):
    cue_labels = [str(int(value)) for value in labels.cueid]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    titles = [
        f"Condition-averaged Pearson RSM ({roi_name})",
        f"Cross-run Pearson reliability RSM ({roi_name})",
    ]
    for axis, matrix, title in zip(
            axes, [conventional, cross_rsm], titles):
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
        axis.set_title(title)
        axis.set_xlabel("Cue ID")
        axis.set_ylabel("Cue ID")
        axis.set_xticks(range(len(cue_labels)), cue_labels, rotation=90, fontsize=7)
        axis.set_yticks(range(len(cue_labels)), cue_labels, fontsize=7)
    fig.colorbar(image, ax=axes, shrink=0.8, label="Pearson correlation")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(
        bids_root, beta_root, output_dir, subject="03", session="001",
        roi="whole-brain", atlas_data_dir=None, pattern_normalization="none"):
    bids_root = Path(bids_root).resolve()
    beta_root = Path(beta_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(
        beta_root / f"beta_maps_metadata_sub-{subject}_ses-{session}.csv"
    )
    metadata = validate_beta_metadata(metadata)
    if roi == "visual":
        mask, affine, run_mask_voxels = retrieval_visual_mask(
            bids_root, subject, session, atlas_data_dir=atlas_data_dir
        )
        roi_name = "visual ROI"
    elif roi == "lateral-parietal":
        mask, affine, run_mask_voxels = retrieval_lateral_parietal_mask(
            bids_root, subject, session, atlas_data_dir=atlas_data_dir
        )
        roi_name = "lateral parietal ROI"
    elif roi == "visual-lpc":
        mask, affine, run_mask_voxels = retrieval_visual_lpc_mask(
            bids_root, subject, session, atlas_data_dir=atlas_data_dir
        )
        roi_name = "visual + lateral parietal ROI"
    elif roi == "whole-brain":
        mask, affine, run_mask_voxels = retrieval_intersection_mask(
            bids_root, subject, session
        )
        roi_name = "whole brain"
    else:
        raise ValueError(f"Unknown ROI: {roi}")
    trial_patterns = extract_trial_patterns(metadata, mask)
    trial_patterns = normalize_trial_patterns(
        trial_patterns, pattern_normalization
    )
    trial_rsm, ordered_trial_metadata = compute_trial_rsm(
        metadata, trial_patterns
    )
    stimulus_rsm, stimulus_metadata = compute_stimulus_grouped_rsm(
        metadata, trial_patterns
    )
    trial_index_rsm, trial_index_metadata = compute_trial_index_rsm(
        metadata, trial_patterns
    )
    labels, by_run = condition_patterns(metadata, trial_patterns)
    conventional, cross_rsm = compute_rsms(by_run)
    distributions = trial_similarity_distributions(metadata, trial_patterns)
    metrics = reliability_metrics(cross_rsm)
    metrics.update({
        "analysis_mask_voxels": int(mask.sum()),
        "run_mask_voxels": run_mask_voxels,
        "roi": roi,
        "pattern_normalization": pattern_normalization,
        "within_run_repeat_reliability": within_run_repeat_metrics(
            metadata, trial_patterns
        ),
        "trial_similarity_distributions": summarize_similarity_distributions(
            distributions
        ),
    })
    if roi == "visual":
        metrics.update({
            "atlas": "Harvard-Oxford cortical maxprob thr25 2mm",
            "atlas_space": "MNI152",
            "visual_atlas_labels": VISUAL_ATLAS_LABELS,
        })
    elif roi == "lateral-parietal":
        metrics.update({
            "atlas": "Harvard-Oxford cortical maxprob thr25 2mm",
            "atlas_space": "MNI152",
            "lateral_parietal_atlas_labels": LATERAL_PARIETAL_ATLAS_LABELS,
        })
    elif roi == "visual-lpc":
        metrics.update({
            "atlas": "Harvard-Oxford cortical maxprob thr25 2mm",
            "atlas_space": "MNI152",
            "visual_atlas_labels": VISUAL_ATLAS_LABELS,
            "lateral_parietal_atlas_labels": LATERAL_PARIETAL_ATLAS_LABELS,
        })

    save_matrix(conventional, labels, output_dir / "retrieval_rsm.csv")
    save_matrix(cross_rsm, labels, output_dir / "retrieval_cross_run_rsm.csv")
    save_trial_matrix(
        trial_rsm,
        ordered_trial_metadata,
        output_dir / "retrieval_trial_rsm.csv",
    )
    ordered_trial_metadata.to_csv(
        output_dir / "retrieval_trial_rsm_order.csv", index=False
    )
    save_trial_matrix(
        stimulus_rsm,
        stimulus_metadata,
        output_dir / "retrieval_stimulus_grouped_rsm.csv",
    )
    stimulus_metadata.to_csv(
        output_dir / "retrieval_stimulus_grouped_rsm_order.csv", index=False
    )
    pd.DataFrame(trial_index_rsm).to_csv(
        output_dir / "retrieval_trial_index_rsm.csv", index=False
    )
    trial_index_metadata.to_csv(
        output_dir / "retrieval_trial_index_rsm_order.csv", index=False
    )
    labels.to_csv(output_dir / "retrieval_conditions.csv", index=False)
    distributions.to_csv(
        output_dir / "retrieval_trial_similarity_values.csv", index=False
    )
    nib.save(
        nib.Nifti1Image(mask.astype(np.uint8), affine),
        output_dir / f"retrieval_{roi.replace('-', '_')}_mask.nii.gz",
    )
    with open(output_dir / "retrieval_rsm_metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)
    plot_rsms(
        conventional,
        cross_rsm,
        labels,
        output_dir / "retrieval_rsms.png",
        roi_name,
    )
    plot_similarity_histograms(
        distributions,
        output_dir / "retrieval_trial_similarity_histograms.png",
        roi_name,
    )
    plot_similarity_histograms(
        distributions,
        output_dir / "retrieval_stimulus_within_between.png",
        roi_name,
    )
    plot_trial_rsm(
        trial_rsm,
        ordered_trial_metadata,
        output_dir / "retrieval_trial_rsm.png",
        roi_name,
    )
    plot_stimulus_grouped_rsm(
        stimulus_rsm,
        stimulus_metadata,
        output_dir / "retrieval_stimulus_grouped_rsm.png",
        roi_name,
    )
    plot_trial_index_rsm(
        trial_index_rsm,
        trial_index_metadata,
        output_dir / "retrieval_trial_index_rsm.png",
        roi_name,
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids-root", required=True)
    parser.add_argument("--beta-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subject", default="03")
    parser.add_argument("--session", default="001")
    parser.add_argument(
        "--roi",
        choices=["whole-brain", "visual", "lateral-parietal", "visual-lpc"],
        default="whole-brain",
    )
    parser.add_argument("--atlas-data-dir", default=None)
    parser.add_argument(
        "--pattern-normalization",
        choices=["none", "trial-zscore"],
        default="none",
    )
    args = parser.parse_args()
    print(json.dumps(main(**vars(args)), indent=2))
