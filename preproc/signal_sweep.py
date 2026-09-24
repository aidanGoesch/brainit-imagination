# run_single_magnitude.py
import argparse, os, sys
import numpy as np, pandas as pd, nibabel as nib

parser = argparse.ArgumentParser()
parser.add_argument("--signal-magnitude", type=float, required=True)
parser.add_argument("--out-root", type=str, required=True)
parser.add_argument("--subject", default="01")
parser.add_argument("--session", default="001")
args = parser.parse_args()

os.makedirs(args.out_root, exist_ok=True)

# call the generator and GLM as importable functions rather than subprocess.
# easiest: wrap each script's body in a main(signal_magnitude, out_root, ...)
# function and import it here.
import generate_synthetic as gen
gen.main(signal_magnitude=args.signal_magnitude, out_root=args.out_root)

import glm as glm_mod
glm_mod.main(subject=args.subject, session=args.session, out_root=args.out_root)

# compute mean r and write ONE row to this magnitude's own results file
gt_df = pd.read_csv(f"{args.out_root}/ground_truth_patterns.csv", index_col=0)
roi_coords = pd.read_csv(f"{args.out_root}/roi_coords.csv").values
patterns = {cond: gt_df.loc[cond].values for cond in gt_df.index}

metadata = pd.read_csv(f"{args.out_root}/beta_maps_metadata_sub-{args.subject}_ses-{args.session}.csv")
recognition_rows = metadata[metadata.task == "recognition"].reset_index(drop=True)

recovered = np.zeros((len(recognition_rows), roi_coords.shape[0]))
for i, row in recognition_rows.iterrows():
    beta_img = nib.load(row.beta_path).get_fdata()
    recovered[i] = beta_img[roi_coords[:, 0], roi_coords[:, 1], roi_coords[:, 2]]

own_pattern_r = [
    np.corrcoef(recovered[i], patterns[recognition_rows.loc[i, "imageid"]])[0, 1]
    for i in range(len(recognition_rows))
]

pd.DataFrame([{
    "signal_magnitude": args.signal_magnitude,
    "mean_r": np.nanmean(own_pattern_r),
    "std_r": np.nanstd(own_pattern_r),
    "n_nan": int(np.sum(np.isnan(own_pattern_r))),
}]).to_csv(f"{args.out_root}/result.csv", index=False)