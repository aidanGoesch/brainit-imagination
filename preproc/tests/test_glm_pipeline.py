from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pandas as pd


PREPROC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PREPROC_DIR))

import generate_synthetic as gen
import glm
import signal_sweep


def test_brainiak_hrf_has_one_sample_per_volume_and_preserves_late_events():
    n_volumes = 110
    signal = gen.build_condition_signal_function(
        onsets=[10.0, 100.0, 194.0],
        total_time=n_volumes * gen.TR,
        n_volumes=n_volumes,
    )

    assert signal.shape == (n_volumes, 1)
    assert np.any(np.abs(signal[50:]) > 0)


def test_lss_transformer_uses_positional_rows_with_nondefault_index():
    events = pd.DataFrame(
        {
            "onset": [10.0, 20.0],
            "duration": [1.0, 1.0],
            "imageid": ["A1", "B1"],
        },
        index=[10, 20],
    )

    transformed, trial_name = glm.lss_transformer(events, 1, ["imageid"])

    assert trial_name == "B1__001"
    assert transformed.iloc[1]["trial_type"] == trial_name
    assert transformed.iloc[0]["trial_type"] == "other"


def test_lss_design_is_full_rank_for_committed_events():
    events_path = (
        PREPROC_DIR
        / "sweep_runs/mag_14/bids_data/sub-01/ses-001/func/"
        "sub-01_ses-001_task-recognition_run-01_events.tsv"
    )
    events = pd.read_csv(events_path, sep="\t")
    frame_times = np.arange(107) * 2.0
    design, trial_name = glm.build_lss_design_matrix(
        events,
        trial_idx=0,
        frame_times=frame_times,
        id_cols=["imageid"],
    )

    assert trial_name in design.columns
    assert design.shape[0] == len(frame_times)
    assert np.linalg.matrix_rank(design.to_numpy()) == design.shape[1]


def test_rsm_metric_recovers_known_group_structure():
    rng = np.random.default_rng(7)
    base_a = rng.normal(size=27)
    base_b = rng.normal(size=27)
    condition_maps = np.stack(
        [
            base_a + 0.01 * rng.normal(size=27),
            base_a + 0.01 * rng.normal(size=27),
            base_b + 0.01 * rng.normal(size=27),
            base_b + 0.01 * rng.normal(size=27),
        ]
    )
    rsm = np.corrcoef(condition_maps)
    within = np.mean([rsm[0, 1], rsm[2, 3]])
    between = np.mean(rsm[np.ix_([0, 1], [2, 3])])

    assert within - between > 0.8


def test_saved_metadata_matches_events_and_beta_paths_resolve():
    root = PREPROC_DIR / "sweep_runs/mag_14"
    events = pd.read_csv(
        root
        / "bids_data/sub-01/ses-001/func/"
        "sub-01_ses-001_task-recognition_run-01_events.tsv",
        sep="\t",
    )
    metadata = pd.read_csv(
        root / "beta_maps_metadata_sub-01_ses-001.csv"
    ).sort_values("trial_idx")

    assert metadata.imageid.tolist() == events.imageid.tolist()
    assert metadata.onset.tolist() == events.onset.tolist()
    for raw_path in metadata.beta_path:
        assert signal_sweep.resolve_beta_path(raw_path, root).is_file()


def test_saved_data_matches_the_historical_double_downsampling_bug():
    root = PREPROC_DIR / "sweep_runs/mag_14"
    events = pd.read_csv(
        root
        / "bids_data/sub-01/ses-001/func/"
        "sub-01_ses-001_task-recognition_run-01_events.tsv",
        sep="\t",
    )
    coords = pd.read_csv(root / "roi_coords.csv").to_numpy(dtype=int)
    ground_truth = pd.read_csv(root / "ground_truth_patterns.csv", index_col=0)
    bold = nib.load(
        root
        / "bids_data/derivatives/sub-01/ses-001/func/"
        "sub-01_ses-001_task-recognition_run-01_space-"
        "MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    ).get_fdata()
    roi_data = bold[coords[:, 0], coords[:, 1], coords[:, 2], :].T

    correct = []
    historical = []
    n_volumes = roi_data.shape[0]
    for condition in ground_truth.index:
        onsets = events.loc[events.imageid == condition, "onset"].tolist()
        regressor = gen.build_condition_signal_function(
            onsets, n_volumes * gen.TR, n_volumes
        )
        correct.append(regressor[:, 0])
        historical.append(
            gen.historical_double_downsample(regressor, n_volumes)[:, 0]
        )

    def mean_pattern_recovery(regressors):
        design = np.column_stack([*regressors, np.ones(n_volumes)])
        coefficients = np.linalg.lstsq(design, roi_data, rcond=None)[0][:-1]
        return np.mean(
            [
                np.corrcoef(coefficients[i], ground_truth.loc[condition])[0, 1]
                for i, condition in enumerate(ground_truth.index)
            ]
        )

    assert mean_pattern_recovery(historical) > mean_pattern_recovery(correct) + 0.3
