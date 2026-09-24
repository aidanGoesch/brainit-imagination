import glob
import re
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

SUB, SES = "01", "001"
SWEEP_ROOT = "./sweep_runs"
COND_ORDER = ["A1", "A2", "B1", "B2"]  # controls block order within each RSM

# ---------------------------------------------------------------
# 1. discover every sweep run dir and its signal_magnitude
# ---------------------------------------------------------------
run_dirs = sorted(glob.glob(f"{SWEEP_ROOT}/mag_*"))

runs = []
for run_dir in run_dirs:
    result_path = f"{run_dir}/result.csv"
    try:
        result_df = pd.read_csv(result_path)
        mag = result_df.loc[0, "signal_magnitude"]
    except FileNotFoundError:
        print(f"skipping {run_dir}: no result.csv")
        continue
    runs.append({"run_dir": run_dir, "signal_magnitude": mag})

runs_df = pd.DataFrame(runs).sort_values("signal_magnitude").reset_index(drop=True)
# lower signal_magnitude == relatively more noise (fixed noise params, varying signal),
# so sorting ascending by magnitude orders panels from noisiest -> cleanest
print(runs_df)

# ---------------------------------------------------------------
# 2. build recovered RSA (condition-averaged) for each run
# ---------------------------------------------------------------
rsms = []
titles = []

for _, row in runs_df.iterrows():
    run_dir = row.run_dir
    mag = row.signal_magnitude

    gt_df = pd.read_csv(f"{run_dir}/ground_truth_patterns.csv", index_col=0)
    roi_coords_run = pd.read_csv(f"{run_dir}/roi_coords.csv").values
    patterns_run = {cond: gt_df.loc[cond].values for cond in gt_df.index}

    metadata = pd.read_csv(f"{run_dir}/beta_maps_metadata_sub-{SUB}_ses-{SES}.csv")
    recognition_rows = metadata[metadata.task == "recognition"].reset_index(drop=True)

    n_roi_voxels_run = roi_coords_run.shape[0]
    recovered = np.zeros((len(recognition_rows), n_roi_voxels_run))
    for i, r in recognition_rows.iterrows():
        beta_img = nib.load(r.beta_path).get_fdata()
        recovered[i] = beta_img[roi_coords_run[:, 0], roi_coords_run[:, 1], roi_coords_run[:, 2]]

    # average within condition so each run gives a clean 4x4, in COND_ORDER
    recovered_by_cond = np.stack([
        recovered[recognition_rows.imageid == c].mean(axis=0)
        for c in COND_ORDER
    ])
    rsm = np.corrcoef(recovered_by_cond)

    rsms.append(rsm)
    titles.append(f"mag={mag:.2f}")

# ---------------------------------------------------------------
# 3. plot grid, ordered noisiest -> cleanest
# ---------------------------------------------------------------
n = len(rsms)
ncols = min(5, n)
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows))
axes = np.atleast_1d(axes).flatten()

for ax, rsm, title in zip(axes, rsms, titles):
    im = ax.imshow(rsm, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(COND_ORDER))); ax.set_xticklabels(COND_ORDER, fontsize=7)
    ax.set_yticks(range(len(COND_ORDER))); ax.set_yticklabels(COND_ORDER, fontsize=7)
    ax.set_title(title, fontsize=9)

for ax in axes[len(rsms):]:
    ax.axis("off")

fig.suptitle("Recovered RSMs across signal_magnitude sweep (low signal / high noise -> high signal / low noise)")
fig.colorbar(im, ax=axes[:len(rsms)], shrink=0.7, label="correlation")
plt.show()