from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.io import loadmat

# ── Paths (relative to this script) ───────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
NSD_DATA = SCRIPT_DIR.parent.parent / "nsd_data"

RAW_FMRI_DIR   = str(NSD_DATA / "raw_fmri")
ROI_MASKS_DIR  = str(NSD_DATA / "roi_masks")
EXPERIMENT_DIR = str(NSD_DATA / "experiment")
OUT_DIR        = str(NSD_DATA / "transfer")

# ── Load session 1 fMRI betas per subject ─────────────────────────────────────
print("Loading session 1 fMRI betas...")
sub_fmri = {}
for s in range(1, 9):
    sub = str(s)
    dir_ = RAW_FMRI_DIR + "/subj0" + sub
    print(f"  subject {s}/8")

    space_lh = np.load(ROI_MASKS_DIR + "/sub" + sub + "/lh.all-vertices_fsaverage_space.npy")
    space_rh = np.load(ROI_MASKS_DIR + "/sub" + sub + "/rh.all-vertices_fsaverage_space.npy")

    file = nib.load(dir_ + "/lh.betas_session01.mgh")
    lh = file.get_fdata()[:, 0, 0, :].transpose()
    lh = lh[:, space_lh.astype(bool)]

    file = nib.load(dir_ + "/rh.betas_session01.mgh")
    rh = file.get_fdata()[:, 0, 0, :].transpose()
    rh = rh[:, space_rh.astype(bool)]

    Y_i = np.concatenate([lh, rh], axis=1)
    Y_i = (Y_i - Y_i.mean(axis=0, keepdims=True)) / Y_i.std(axis=0, keepdims=True)
    Y_i = np.nan_to_num(Y_i)

    sub_fmri["subj"+str(s)] = Y_i
    print(f"  subject {s}/8 done: shape {Y_i.shape}")

# ── Experiment design ─────────────────────────────────────────────────────────
print("Loading experiment design...")
mat = loadmat(EXPERIMENT_DIR + "/nsd_expdesign.mat")

ses_to_sub    = mat['masterordering'][0, :] - 1
sub_to_global = mat['subjectim'] - 1

# first 1000 images seen by all subjects (shared/multi-subject images)
mult_inds = sub_to_global[0][:1000]

sub_im_id = {}
for subj in range(1, 9):
    global_inds = []
    for i in range(750):
        global_ind = sub_to_global[subj-1, ses_to_sub[i]]
        global_inds.append(global_ind)
    global_inds = np.array(global_inds)

    # remove shared images
    mask = ~np.isin(global_inds, mult_inds)
    sub_fmri["subj"+str(subj)] = sub_fmri["subj"+str(subj)][mask]
    sub_im_id["subj"+str(subj)] = global_inds[mask]

# ── Save ──────────────────────────────────────────────────────────────────────
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
np.savez(OUT_DIR + "/subjects_single_ses_fmri.npz", **sub_fmri)
np.savez(OUT_DIR + "/subjects_single_ses_imgid.npz", **sub_im_id)
print("Done.")
