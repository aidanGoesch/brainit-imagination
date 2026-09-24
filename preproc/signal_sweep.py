"""Generate one synthetic run, fit the GLM, and quantify pattern recovery."""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

import generate_synthetic as gen
import glm as glm_mod


COND_ORDER = ["A1", "A2", "B1", "B2"]


def resolve_beta_path(raw_path, out_root):
    path = Path(raw_path)
    candidates = [
        path,
        Path(__file__).resolve().parent / path,
        Path(out_root) / "beta_maps" / path.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve beta map {raw_path}")


def evaluate_run(out_root, subject="01", session="001"):
    out_root = Path(out_root).resolve()
    ground_truth = pd.read_csv(
        out_root / "ground_truth_patterns.csv", index_col=0
    )
    roi_coords = pd.read_csv(out_root / "roi_coords.csv").to_numpy(dtype=int)
    metadata = pd.read_csv(
        out_root / f"beta_maps_metadata_sub-{subject}_ses-{session}.csv"
    )
    rows = metadata[metadata.task == "recognition"].reset_index(drop=True)

    recovered = np.zeros((len(rows), len(roi_coords)))
    for i, row in rows.iterrows():
        beta = nib.load(resolve_beta_path(row.beta_path, out_root)).get_fdata()
        recovered[i] = beta[
            roi_coords[:, 0], roi_coords[:, 1], roi_coords[:, 2]
        ]

    own_pattern_r = np.array(
        [
            np.corrcoef(recovered[i], ground_truth.loc[row.imageid])[0, 1]
            for i, row in rows.iterrows()
        ]
    )
    condition_maps = np.stack(
        [recovered[rows.imageid == condition].mean(axis=0)
         for condition in COND_ORDER]
    )
    rsm = np.corrcoef(condition_maps)
    within = np.mean([rsm[0, 1], rsm[2, 3]])
    between = np.mean(rsm[np.ix_([0, 1], [2, 3])])
    return {
        "mean_r": float(np.nanmean(own_pattern_r)),
        "std_r": float(np.nanstd(own_pattern_r)),
        "n_nan": int(np.isnan(own_pattern_r).sum()),
        "within_similarity": float(within),
        "between_similarity": float(between),
        "within_minus_between": float(within - between),
    }, rsm


def main(
        signal_magnitude, noise_level, out_root, subject="01", session="001",
        signal_method="CNR_Amp/Noise-SD"):
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    gen.main(
        signal_magnitude=signal_magnitude,
        noise_level=noise_level,
        signal_method=signal_method,
        out_root=str(out_root),
    )
    glm_mod.main(subject=subject, session=session, out_root=str(out_root))
    metrics, _ = evaluate_run(out_root, subject=subject, session=session)
    pd.DataFrame(
        [{"signal_magnitude": signal_magnitude, "noise_level": noise_level,
          "signal_method": signal_method, **metrics}]
    ).to_csv(out_root / "result.csv", index=False)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-magnitude", type=float, required=True)
    parser.add_argument(
        "--noise-level", choices=list(gen.NOISE_PRESETS), default="medium"
    )
    parser.add_argument(
        "--signal-method",
        choices=["CNR_Amp/Noise-SD", "PSC"],
        default="CNR_Amp/Noise-SD",
    )
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--subject", default="01")
    parser.add_argument("--session", default="001")
    args = parser.parse_args()
    main(
        signal_magnitude=args.signal_magnitude,
        noise_level=args.noise_level,
        out_root=args.out_root,
        subject=args.subject,
        session=args.session,
        signal_method=args.signal_method,
    )