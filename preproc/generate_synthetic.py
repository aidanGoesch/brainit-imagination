"""
Synthetic fMRI dataset generator for GLM/LSS pipeline validation.

Generates a BIDS raw tree (events.tsv) plus an fMRIPrep-derivatives-style
tree (preproc bold, brain mask, confounds) in real MNI152NLin2009cAsym
space, with a known ground-truth multivoxel pattern planted in an ROI that
sits entirely inside gray/white matter. Noise level is adjustable so you
can generate a near-noiseless dataset to test whether your GLM correctly
recovers a known signal, independent of how realistic/noisy the data is.

Usage:
    python generate.py --noise-level none   --signal-magnitude 3.0
    python generate.py --noise-level medium --signal-magnitude 1.5
    python generate.py --noise-level high    --signal-magnitude 1.5 --resolution 2

Validate the generated data only (no GLM involved):
    python generate.py --test
or, if you have pytest installed:
    pytest generate.py -v
"""
import os
import json
import argparse

import numpy as np
import pandas as pd
import nibabel as nib
import brainiak.utils.fmrisim as sim
from scipy.ndimage import binary_erosion

try:
    from nilearn import datasets
except ImportError as e:
    raise ImportError(
        "This script requires nilearn for a real MNI152 template/mask "
        "(`pip install nilearn`)."
    ) from e


# ---------------------------------------------------------------------------
# Fixed config (shared by generation and tests)
# ---------------------------------------------------------------------------
SEED = 42
SUB, SES, TASK, RUN = "01", "001", "recognition", "01"

ROI_HALF = 1                 # -> 3x3x3 = 27 voxel ROI
TR = 2.0                     # must match TR_VALUE in glm.py
EVENT_DURATION = 1.0         # seconds
FINE_RES = 10.0              # samples/sec before downsampling to TR

CONFOUND_NOISE_SD = 1e-3     # scale of simulated head-motion jitter (mm / rad)

# Noise presets: (sfnr, snr). "none" -> effectively noiseless, for isolating
# GLM bugs from data problems. "medium" roughly matches the real dataset
# cited in the fmrisim docs (SNR ~23, SFNR ~70).
NOISE_PRESETS = {
    "none":   {"sfnr": 5000, "snr": 5000},
    "low":    {"sfnr": 120,  "snr": 40},
    "medium": {"sfnr": 70,   "snr": 23},
    "high":   {"sfnr": 40,   "snr": 12},
}


def paths(out_root="."):
    bids_root = f"{out_root}/bids_data"
    deriv_root = f"{bids_root}/derivatives"
    func_dir = f"{deriv_root}/sub-{SUB}/ses-{SES}/func"
    prefix = f"sub-{SUB}_ses-{SES}_task-{TASK}_run-{RUN}"
    return {
        "bids_root": bids_root,
        "deriv_root": deriv_root,
        "events": f"{bids_root}/sub-{SUB}/ses-{SES}/func/{prefix}_events.tsv",
        "bold": f"{func_dir}/{prefix}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
        "mask": f"{func_dir}/{prefix}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
        "confounds": f"{func_dir}/{prefix}_desc-confounds_timeseries.tsv",
        "gt_patterns": f"{out_root}/ground_truth_patterns.csv",
        "roi_coords": f"{out_root}/roi_coords.csv",
    }


def check(label, cond):
    status = "OK" if cond else "FAILED"
    print(f"[CHECK] {label}: {status}")
    if not cond:
        raise AssertionError(f"Sanity check failed: {label}")


def unit(v):
    return v / np.linalg.norm(v)


def pad_or_truncate(x, target_len):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    cur_len = x.shape[0]
    if cur_len == target_len:
        return x
    elif cur_len > target_len:
        return x[:target_len]
    pad = np.zeros((target_len - cur_len, x.shape[1]))
    return np.vstack([x, pad])


# ---------------------------------------------------------------------------
# Real brain geometry
# ---------------------------------------------------------------------------
def load_real_brain(resolution_mm=3):
    """Real MNI152NLin2009cAsym template + brain mask (same space fMRIPrep
    outputs to), via nilearn. Downloads/caches on first call -- needs
    network access once."""
    template_img = datasets.load_mni152_template(resolution=resolution_mm)
    mask_img = datasets.load_mni152_brain_mask(resolution=resolution_mm)
    template = template_img.get_fdata()
    mask = (mask_img.get_fdata() > 0).astype(np.uint8)
    affine = template_img.affine
    return mask, template, affine


def pick_roi_center(mask, roi_half, rng):
    """A voxel whose (2*roi_half+1)^3 neighborhood is entirely inside the
    brain mask, so the ROI never touches non-brain / volume-edge voxels."""
    structure = np.ones((2 * roi_half + 1,) * 3)
    eroded = binary_erosion(mask.astype(bool), structure=structure, border_value=0)
    candidates = np.argwhere(eroded)
    check("at least one valid ROI center exists inside the brain mask", len(candidates) > 0)
    idx = rng.integers(len(candidates))
    return tuple(int(c) for c in candidates[idx])


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------
def main(signal_magnitude=1.5, noise_level="medium", resolution_mm=3, out_root="."):
    rng = np.random.default_rng(SEED)
    np_rng = np.random.RandomState(SEED)  # some brainiak calls want legacy RandomState

    P = paths(out_root)

    # -- 1. Real brain geometry -------------------------------------------------
    print("=" * 70); print("1. Loading real brain geometry (MNI152NLin2009cAsym)"); print("=" * 70)
    mask, template, affine = load_real_brain(resolution_mm)
    DIM = mask.shape
    print(f"Volume dims: {DIM}, voxel size ~{resolution_mm}mm, brain voxels: {int(mask.sum())}")
    check("mask is not empty", mask.sum() > 0)
    check("mask is not the entire volume (i.e. actually brain-shaped)",
          mask.sum() < 0.9 * mask.size)

    roi_center = pick_roi_center(mask, ROI_HALF, rng)
    offsets = range(-ROI_HALF, ROI_HALF + 1)
    roi_coords = np.array([
        (roi_center[0] + dx, roi_center[1] + dy, roi_center[2] + dz)
        for dx in offsets for dy in offsets for dz in offsets
    ])
    n_roi_voxels = roi_coords.shape[0]
    check("every ROI voxel is inside the brain mask",
          mask[roi_coords[:, 0], roi_coords[:, 1], roi_coords[:, 2]].all())

    # -- 2. Ground-truth multivoxel patterns ------------------------------------
    print(); print("=" * 70); print("2. Building ground-truth patterns"); print("=" * 70)
    base_A = unit(np_rng.normal(size=n_roi_voxels))
    base_B = unit(np_rng.normal(size=n_roi_voxels))
    patterns = {
        "A1": unit(base_A + 0.05 * np_rng.normal(size=n_roi_voxels)),
        "A2": unit(base_A + 0.05 * np_rng.normal(size=n_roi_voxels)),
        "B1": unit(base_B + 0.05 * np_rng.normal(size=n_roi_voxels)),
        "B2": unit(base_B + 0.05 * np_rng.normal(size=n_roi_voxels)),
    }
    gt_matrix = np.stack([patterns[k] for k in ["A1", "A2", "B1", "B2"]])
    gt_rsa = pd.DataFrame(np.corrcoef(gt_matrix),
                           index=["A1", "A2", "B1", "B2"], columns=["A1", "A2", "B1", "B2"])
    print(gt_rsa.round(2))
    check("A1/A2 and B1/B2 are more similar within- than between-pair",
          gt_rsa.loc["A1", "A2"] > gt_rsa.loc["A1", "B1"]
          and gt_rsa.loc["B1", "B2"] > gt_rsa.loc["B1", "A1"])

    # -- 3. Trial timing ---------------------------------------------------------
    print(); print("=" * 70); print("3. Building trial timing"); print("=" * 70)
    conditions = ["A1"] * 3 + ["A2"] * 3 + ["B1"] * 3 + ["B2"] * 3
    rng.shuffle(conditions)
    isis = rng.uniform(14, 22, size=len(conditions) - 1)
    isis[3] = 3.0  # one tight-SOA pair
    onsets = np.cumsum(np.insert(isis, 0, 10.0))
    onsets = np.round(onsets / TR) * TR
    total_time = float(onsets[-1] + 20.0)
    n_volumes = int(np.ceil(total_time / TR))
    total_time = n_volumes * TR
    trials = pd.DataFrame({"onset": onsets, "duration": EVENT_DURATION, "imageid": conditions})
    print(trials)
    print(f"n_volumes={n_volumes}, total_time={total_time}s, resolution={resolution_mm}mm, noise={noise_level}")
    EXPECTED_FINE_LEN = int(round(total_time * FINE_RES))
    DOWNSAMPLE_STEP = int(round(FINE_RES * TR))

    def downsample_to_tr(x, n_vol):
        x = pad_or_truncate(x, EXPECTED_FINE_LEN)
        return pad_or_truncate(x[::DOWNSAMPLE_STEP], n_vol)

    check("each condition appears exactly 3 times", (trials.imageid.value_counts() == 3).all())

    # -- 4. Noise (generated first, so signal can be CNR-scaled against it) -----
    print(); print("=" * 70); print("4. Generating noise"); print("=" * 70)
    all_stimfunction_fine = sim.generate_stimfunction(
        onsets=trials["onset"].tolist(),
        event_durations=[EVENT_DURATION] * len(trials),
        total_time=total_time, temporal_resolution=FINE_RES,
    )
    all_stimfunction_tr = downsample_to_tr(all_stimfunction_fine, n_volumes).flatten()

    noise_dict = {"voxel_size": [resolution_mm] * 3, "matched": 0}
    noise_dict.update(NOISE_PRESETS[noise_level])
    noise_4d = sim.generate_noise(
        dimensions=np.array(DIM), stimfunction_tr=all_stimfunction_tr,
        tr_duration=TR, template=template, mask=mask, noise_dict=noise_dict,
    )
    check("noise_4d has no NaNs", not np.isnan(noise_4d).any())

    # -- 5. Signal, CNR-scaled against the actual generated ROI noise -----------
    print(); print("=" * 70); print("5. Generating signal"); print("=" * 70)
    noise_roi = noise_4d[roi_coords[:, 0], roi_coords[:, 1], roi_coords[:, 2], :].mean(axis=0)
    signal_4d = np.zeros(DIM + (n_volumes,))
    for cond, pattern in patterns.items():
        cond_onsets = trials.loc[trials.imageid == cond, "onset"].tolist()
        stimfunction_fine = pad_or_truncate(
            sim.generate_stimfunction(
                onsets=cond_onsets, event_durations=[EVENT_DURATION] * len(cond_onsets),
                total_time=total_time, temporal_resolution=FINE_RES,
            ), EXPECTED_FINE_LEN)
        signal_function_fine = sim.convolve_hrf(
            stimfunction=stimfunction_fine, tr_duration=TR, temporal_resolution=FINE_RES)
        signal_function = downsample_to_tr(signal_function_fine, n_volumes)

        scaled = sim.compute_signal_change(
            signal_function, noise_roi[:, None], noise_dict,
            magnitude=[signal_magnitude], method="CNR_Amp/Noise-SD",
        )

        volume_signal = np.zeros(DIM)
        volume_signal[roi_coords[:, 0], roi_coords[:, 1], roi_coords[:, 2]] = pattern
        cond_signal_4d = sim.apply_signal(signal_function=scaled, volume_signal=volume_signal)
        check(f"  [{cond}] applied signal has no NaNs", not np.isnan(cond_signal_4d).any())
        signal_4d += cond_signal_4d

    check("signal_4d is not all-zero", np.abs(signal_4d).sum() > 0)
    bold_4d = signal_4d + noise_4d
    check("bold_4d has no NaNs", not np.isnan(bold_4d).any())

    # -- 6. Realistic (small) simulated head motion, used for FD -----------------
    motion_mm = np_rng.normal(0, CONFOUND_NOISE_SD, size=(n_volumes, 3))
    motion_rad = np_rng.normal(0, CONFOUND_NOISE_SD / 50, size=(n_volumes, 3))

    def framewise_displacement(mm, rad, head_radius_mm=50.0):
        d_mm = np.diff(mm, axis=0)
        d_rad = np.diff(rad, axis=0) * head_radius_mm
        fd = np.abs(d_mm).sum(axis=1) + np.abs(d_rad).sum(axis=1)
        return np.concatenate([[np.nan], fd])

    def dvars_from_bold(bold, mask_):
        in_mask = bold[mask_.astype(bool)]
        d = np.diff(in_mask, axis=1)
        dv = np.sqrt((d ** 2).mean(axis=0))
        dv = np.concatenate([[np.nan], dv])
        std_dv = dv / np.nanstd(dv)
        return dv, std_dv

    fd = framewise_displacement(motion_mm, motion_rad)
    dvars, std_dvars = dvars_from_bold(bold_4d, mask)

    def cosine_basis(n_vol, n_bases=4):
        k = np.arange(1, n_bases + 1)
        t = np.arange(n_vol)
        return np.sqrt(2.0 / n_vol) * np.cos(np.pi * (2 * t[:, None] + 1) * k[None, :] / (2 * n_vol))

    a_comp_cor = np_rng.normal(0, CONFOUND_NOISE_SD, size=(n_volumes, 5))
    cosines = cosine_basis(n_volumes)
    confounds_df = pd.DataFrame({
        "trans_x": motion_mm[:, 0], "trans_y": motion_mm[:, 1], "trans_z": motion_mm[:, 2],
        "rot_x": motion_rad[:, 0], "rot_y": motion_rad[:, 1], "rot_z": motion_rad[:, 2],
        "csf": np_rng.normal(0, CONFOUND_NOISE_SD, n_volumes),
        "white_matter": np_rng.normal(0, CONFOUND_NOISE_SD, n_volumes),
        "global_signal": np_rng.normal(0, CONFOUND_NOISE_SD, n_volumes),
        "framewise_displacement": fd,
        "dvars": dvars,
        "std_dvars": std_dvars,
    })
    for i in range(a_comp_cor.shape[1]):
        confounds_df[f"a_comp_cor_{i:02d}"] = a_comp_cor[:, i]
    for i in range(cosines.shape[1]):
        confounds_df[f"cosine{i:02d}"] = cosines[:, i]

    # -- 7. Save BIDS raw + derivatives ------------------------------------------
    print(); print("=" * 70); print("7. Saving BIDS files"); print("=" * 70)
    os.makedirs(os.path.dirname(P["events"]), exist_ok=True)
    trials[["onset", "duration", "imageid"]].to_csv(P["events"], sep="\t", index=False)
    print(f"Wrote {P['events']}")

    os.makedirs(os.path.dirname(P["bold"]), exist_ok=True)
    img = nib.Nifti1Image(bold_4d.astype(np.float32), affine=affine)
    img.header.set_xyzt_units("mm", "sec")
    pixdim = img.header["pixdim"]; pixdim[4] = TR; img.header["pixdim"] = pixdim
    nib.save(img, P["bold"]); print(f"Wrote {P['bold']}")

    mask_img = nib.Nifti1Image(mask.astype(np.uint8), affine=affine)
    nib.save(mask_img, P["mask"]); print(f"Wrote {P['mask']}")

    confounds_df.to_csv(P["confounds"], sep="\t", index=False, na_rep="n/a")
    print(f"Wrote {P['confounds']}")

    for root, name, is_deriv in [(P["bids_root"], "synthetic-lss-test", False),
                                  (P["deriv_root"], "synthetic-lss-test-fmriprep", True)]:
        desc = {"Name": name, "BIDSVersion": "1.8.0"}
        if is_deriv:
            desc["DatasetType"] = "derivative"
            desc["GeneratedBy"] = [{"Name": "fmrisim synthetic test"}]
        with open(f"{root}/dataset_description.json", "w") as f:
            json.dump(desc, f)

    gt_df = pd.DataFrame(gt_matrix, index=["A1", "A2", "B1", "B2"],
                          columns=[f"voxel_{i:02d}" for i in range(n_roi_voxels)])
    gt_df.to_csv(P["gt_patterns"])
    pd.DataFrame(roi_coords, columns=["x", "y", "z"]).to_csv(P["roi_coords"], index=False)
    print("Wrote ground_truth_patterns.csv and roi_coords.csv")

    print()
    print("All checks passed. Dataset is ready at:", P["bids_root"])
    print("Validate the data alone with: python generate.py --test")


# ---------------------------------------------------------------------------
# Data-validation tests (assert-based; pytest-collectible via `test_` prefix).
# These ONLY check properties of the files generate() wrote to disk -- no
# GLM is run, so a pass here rules out the synthetic data as the cause of a
# downstream GLM problem.
# ---------------------------------------------------------------------------
_OUT_ROOT = "."


def _load():
    P = paths(_OUT_ROOT)
    for key in ("bold", "mask", "confounds", "events", "gt_patterns", "roi_coords"):
        assert os.path.exists(P[key]), f"missing {key} at {P[key]} -- run generation first"
    bold_img = nib.load(P["bold"])
    mask_img = nib.load(P["mask"])
    confounds = pd.read_csv(P["confounds"], sep="\t")
    events = pd.read_csv(P["events"], sep="\t")
    gt = pd.read_csv(P["gt_patterns"], index_col=0)
    roi_coords = pd.read_csv(P["roi_coords"]).values
    return bold_img, mask_img, confounds, events, gt, roi_coords


def test_brain_is_actually_brain_shaped():
    bold_img, mask_img, *_ = _load()
    mask = mask_img.get_fdata()
    assert mask.sum() > 0
    # a filled cuboid mask would have ~100% fill in its bounding box; a real
    # brain mask does not, and its corners are background.
    assert mask.sum() < 0.9 * mask.size
    corner = tuple(np.array(mask.shape) - 1)
    assert mask[0, 0, 0] == 0 and mask[corner] == 0, "volume corners should be non-brain"


def test_affine_matches_real_mni_space():
    bold_img, mask_img, *_ = _load()
    assert np.allclose(bold_img.affine, mask_img.affine)
    assert not np.allclose(bold_img.affine, np.eye(4)), "affine should not be identity for MNI space"
    voxel_sizes = np.abs(np.diag(bold_img.affine)[:3])
    assert np.allclose(voxel_sizes, voxel_sizes[0], atol=0.01), "voxels should be isotropic"


def test_bold_and_mask_share_geometry():
    bold_img, mask_img, *_ = _load()
    assert bold_img.shape[:3] == mask_img.shape


def test_tr_matches_spec():
    bold_img, *_ = _load()
    assert np.isclose(bold_img.header.get_zooms()[3], TR)


def test_no_nans_or_infs_in_bold():
    bold_img, *_ = _load()
    data = bold_img.get_fdata()
    assert not np.isnan(data).any()
    assert not np.isinf(data).any()


def test_roi_entirely_inside_brain_mask():
    _, mask_img, _, _, _, roi_coords = _load()
    mask = mask_img.get_fdata()
    vals = mask[roi_coords[:, 0], roi_coords[:, 1], roi_coords[:, 2]]
    assert (vals > 0).all(), "every ROI voxel must be inside the brain mask"


def test_confounds_have_fmriprep_style_columns():
    _, _, confounds, *_ = _load()
    expected = {"trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z",
                "framewise_displacement", "dvars", "std_dvars",
                "csf", "white_matter", "global_signal"}
    assert expected.issubset(set(confounds.columns))
    assert any(c.startswith("a_comp_cor_") for c in confounds.columns)
    assert any(c.startswith("cosine") for c in confounds.columns)


def test_confounds_first_row_derivatives_are_na():
    # fMRIPrep leaves the first-timepoint derivative confounds blank, since
    # there is no prior volume to difference against.
    _, _, confounds, *_ = _load()
    assert pd.isna(confounds.loc[0, "framewise_displacement"])
    assert pd.isna(confounds.loc[0, "dvars"])


def test_confounds_no_constant_columns():
    _, _, confounds, *_ = _load()
    non_na_std = confounds.std(numeric_only=True, skipna=True)
    assert (non_na_std > 0).all(), "a constant confound column makes the design matrix singular"


def test_confounds_row_count_matches_n_volumes():
    bold_img, _, confounds, *_ = _load()
    assert len(confounds) == bold_img.shape[3]


def test_events_onsets_within_scan_and_on_tr_grid():
    bold_img, _, _, events, *_ = _load()
    total_time = bold_img.shape[3] * TR
    assert events["onset"].min() >= 0
    assert (events["onset"] + events["duration"]).max() <= total_time
    assert np.allclose(events["onset"] % TR, 0, atol=1e-6)


def test_each_condition_has_three_trials():
    *_, events, _, _ = _load()
    assert (events["imageid"].value_counts() == 3).all()


def test_ground_truth_pattern_structure():
    *_, gt, _ = _load()
    corr = np.corrcoef(gt.values)
    labels = list(gt.index)
    a1, a2 = labels.index("A1"), labels.index("A2")
    b1, b2 = labels.index("B1"), labels.index("B2")
    assert corr[a1, a2] > corr[a1, b1]
    assert corr[b1, b2] > corr[b1, a1]


def run_all_tests():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = []
    for fn in fns:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except AssertionError as e:
            failures.append(fn.__name__)
            print(f"[FAIL] {fn.__name__}: {e}")
    print(f"\n{len(fns) - len(failures)}/{len(fns)} tests passed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-magnitude", type=float, default=1.5)
    parser.add_argument("--noise-level", choices=list(NOISE_PRESETS), default="medium")
    parser.add_argument("--resolution", type=int, default=3, help="MNI template resolution in mm")
    parser.add_argument("--out-root", type=str, default=".")
    parser.add_argument("--test", action="store_true", help="only run the data-validation tests")
    args = parser.parse_args()

    _OUT_ROOT = args.out_root
    if args.test:
        run_all_tests()
    else:
        main(signal_magnitude=args.signal_magnitude, noise_level=args.noise_level,
             resolution_mm=args.resolution, out_root=args.out_root)