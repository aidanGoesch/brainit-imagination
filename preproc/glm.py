import os
import argparse
import json
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nilearn.glm.first_level import FirstLevelModel
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import make_first_level_design_matrix

from bids import BIDSLayout
from sklearn.utils import Bunch

# debug flag - plots design matrix rn
DEBUG = False
NUM_DESIGN_MAT_PLOTS = 3

# default values for the GLM that should hold for all participants and runs
SPACE_NAME = 'MNI152NLin2009cAsym'

HRF_MODEL = 'spm'        
DRIFT_MODEL = 'cosine'   
FD_THRESH = 0.3
DEFAULT_SLICE_TIME_REF = 0.5


# -------------------------------------- functions -------------------------------------- 
def lss_transformer(events_df, row_number, id_cols):
    """Isolate one trial for LSS, building its label from id_cols."""
    events_df = events_df.reset_index(drop=True).copy()
    if not 0 <= row_number < len(events_df):
        raise IndexError(f"trial row {row_number} outside 0..{len(events_df) - 1}")

    events_df["trial_id"] = events_df[id_cols].astype(str).agg("_".join, axis=1)

    trial_name = f"{events_df.iloc[row_number]['trial_id']}__{row_number:03d}"

    events_df["trial_type"] = "other"
    events_df.at[row_number, "trial_type"] = trial_name

    return events_df, trial_name


def build_lss_design_matrix(
        events_df, trial_idx, frame_times, id_cols, add_regs=None,
        add_reg_names=None, hrf_model=HRF_MODEL, drift_model=DRIFT_MODEL,
        event_duration=None):
    """Construct one LS-S design and return it with the target column name."""
    lss_events_df, trial_condition = lss_transformer(
        events_df, trial_idx, id_cols=id_cols
    )
    event_columns = ["onset", "duration", "trial_type"]
    if "modulation" in lss_events_df:
        event_columns.append("modulation")
    design_events = lss_events_df[event_columns].copy()
    if event_duration is not None:
        if event_duration < 0:
            raise ValueError("event_duration must be non-negative")
        design_events["duration"] = float(event_duration)
    design = make_first_level_design_matrix(
        frame_times=frame_times,
        events=design_events,
        hrf_model=hrf_model,
        drift_model=drift_model,
        add_regs=add_regs,
        add_reg_names=add_reg_names,
    )
    return design, trial_condition


def prepare_confounds(confounds_df, num_volumes):
    """Select nuisance regressors and add one-hot regressors for high-FD TRs."""
    motion_cols = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    missing = set(motion_cols + ["framewise_displacement"]) - set(
        confounds_df.columns
    )
    if missing:
        raise ValueError(f"Missing required confounds: {sorted(missing)}")
    motion_params = confounds_df[motion_cols].fillna(0)
    acompcor_cols = [
        c for c in confounds_df.columns if c.startswith("a_comp_cor_")
    ][:5]
    if len(acompcor_cols) < 5:
        raise ValueError(
            f"Expected at least 5 aCompCor columns, found {len(acompcor_cols)}"
        )
    acompcor_params = confounds_df[acompcor_cols].fillna(0)
    all_confounds = pd.concat([motion_params, acompcor_params], axis=1)
    fd = confounds_df["framewise_displacement"].fillna(0)
    for outlier_tr in np.where(fd > FD_THRESH)[0]:
        column = np.zeros(num_volumes)
        column[outlier_tr] = 1.0
        all_confounds[f"motion_outlier_{outlier_tr:03d}"] = column
    if len(all_confounds) != num_volumes:
        raise ValueError(
            f"Confounds have {len(all_confounds)} rows, expected {num_volumes}"
        )
    return all_confounds


def frame_times_from_bold(img, bold_path):
    """Use fMRIPrep's recorded slice-time reference for design frame times."""
    run_tr = float(img.header.get_zooms()[3])
    if not np.isfinite(run_tr) or run_tr <= 0:
        raise ValueError(f"Invalid TR {run_tr} in {bold_path}")
    sidecar_path = str(bold_path).removesuffix(".nii.gz") + ".json"
    start_time = DEFAULT_SLICE_TIME_REF * run_tr
    if os.path.exists(sidecar_path):
        with open(sidecar_path) as file:
            start_time = float(json.load(file).get("StartTime", start_time))
    frame_times = np.arange(img.shape[3]) * run_tr + start_time
    return frame_times, run_tr, start_time


def make_and_save_design_matrix(events_df, trial_idx, frame_times, hrf_model,
        drift_model, add_regs, add_reg_names, id_cols,
        output_dir="./design_matrix_checks", tag="", event_duration=None):
    os.makedirs(output_dir, exist_ok=True)

    lss_design_matrix, trial_condition = build_lss_design_matrix(
        events_df=events_df,
        trial_idx=trial_idx,
        frame_times=frame_times,
        id_cols=id_cols,
        hrf_model=hrf_model,
        drift_model=drift_model,
        add_regs=add_regs,
        add_reg_names=add_reg_names,
        event_duration=event_duration,
    )

    plt.figure(figsize=(6, 8))
    plot_design_matrix(lss_design_matrix)
    plt.title(f"LSS Design Matrix for Isolated Trial: {trial_condition}")

    prefix = f"{tag}_" if tag else ""
    png_filename = f"{prefix}trial-{trial_idx:03d}_{trial_condition}_design_matrix.png"
    png_path = os.path.join(output_dir, png_filename)

    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved {png_path}")

    return lss_design_matrix, trial_condition, png_path


def main(
        subject, session, out_root=".", tasks=None, runs=None,
        bids_root=None, output_root=None, max_trials=None,
        event_duration=None):
    """Run trial-wise LS-S models for selected tasks and runs."""
    sub_id = subject
    session_id = session
    out_root = os.path.abspath(out_root)
    bids_root = os.path.abspath(bids_root or os.path.join(out_root, "bids_data"))
    output_root = os.path.abspath(output_root or out_root)
    selected_tasks = tasks or ["recognition", "retrieval"]
    if isinstance(selected_tasks, str):
        selected_tasks = [selected_tasks]
    invalid_tasks = set(selected_tasks) - {"recognition", "retrieval"}
    if invalid_tasks:
        raise ValueError(f"Unknown tasks: {sorted(invalid_tasks)}")
    selected_runs = None
    if runs:
        selected_runs = {f"{int(run):02d}" for run in runs}

    layout = BIDSLayout(
        bids_root,
        derivatives=os.path.join(bids_root, "derivatives"),
        validate=True,
    )

    output_dir = os.path.join(output_root, "beta_maps")
    os.makedirs(output_dir, exist_ok=True)

    metadata_rows = []  # will accumulate one dict per trial

    for task in selected_tasks:
        if task == 'recognition':
            id_cols = ['imageid']
        elif task == 'retrieval':
            id_cols = ['cueid', 'targetid']
        else:
            raise ValueError(f"Unknown task: {task}")

        num_runs = layout.get(subject=sub_id, session=session_id, task=task,
                           return_type='id', target='run')
        if selected_runs is not None:
            num_runs = [
                run for run in num_runs
                if f"{int(run):02d}" in selected_runs
            ]
        
        for run in num_runs:
            run_label = f"{int(run):02d}"
            # load files specific to each run
            func_files = layout.get(subject=sub_id, session=session_id, task=task, run=run,
                    space=SPACE_NAME, desc='preproc', suffix='bold', extension='.nii.gz',
                    return_type='file')
            events_files = layout.get(subject=sub_id, session=session_id, task=task, run=run,
                    suffix='events', extension='.tsv', return_type='file')
            confounds_files = layout.get(subject=sub_id, session=session_id, task=task, run=run,
                    desc='confounds', suffix='timeseries', extension='.tsv', return_type='file')

            mask_files = layout.get(subject=sub_id, session=session_id, task=task, run=run,
                    space=SPACE_NAME, desc='brain', suffix='mask', extension='.nii.gz',
                    return_type='file')


            for name, files in [
                    ("func", func_files), ("events", events_files),
                    ("mask", mask_files), ("confounds", confounds_files)]:
                assert len(files) == 1, f"Expected exactly 1 {name} file, got {len(files)}: {files}"


            # print("func files (used the first one")
            # print(func_files)
            # print(events_files)
            # print(confounds_files)

            # make dataset
            dataset = Bunch(
                func=[func_files[0]],
                events=[events_files[0]],
                confounds=[confounds_files[0]],
                description="Local BIDS dataset formatted for Nilearn"
            )
            
            # get events
            events_df = pd.read_csv(dataset.events[0], sep="\t").reset_index(drop=True)

            img = nib.load(dataset.func[0])
            num_volumes = img.shape[3]
            frame_times, run_tr, frame_time_start = frame_times_from_bold(
                img, dataset.func[0]
            )
            slice_time_ref = frame_time_start / run_tr

            # get confounds from events df 
            confounds_df = pd.read_csv(dataset.confounds[0], sep="\t")

            all_confounds = prepare_confounds(confounds_df, num_volumes)
            add_regs = all_confounds.values
            add_reg_names = all_confounds.columns.tolist()

            num_trials = len(events_df)
            if max_trials is not None:
                num_trials = min(num_trials, max_trials)

            # iterate through all trials in the block
            for trial_idx in range(num_trials):
                if DEBUG:
                    make_and_save_design_matrix(
                        events_df=events_df,
                        trial_idx=trial_idx,
                        frame_times=frame_times,
                        hrf_model=HRF_MODEL,
                        drift_model=DRIFT_MODEL,
                        add_regs=add_regs,
                        add_reg_names=add_reg_names,
                        id_cols=id_cols,
                        output_dir="./design_matrix_checks",
                        tag="sub-01_ses-001_task-retrieval_run-01",
                        event_duration=event_duration,
                    )

                    if trial_idx >= NUM_DESIGN_MAT_PLOTS:
                        quit(1)

                    continue
                lss_design_matrix, trial_condition = build_lss_design_matrix(
                    events_df=events_df,
                    trial_idx=trial_idx,
                    frame_times=frame_times,
                    id_cols=id_cols,
                    hrf_model=HRF_MODEL,
                    drift_model=DRIFT_MODEL,
                    add_regs=add_regs,
                    add_reg_names=add_reg_names,
                    event_duration=event_duration,
                )
                design_values = lss_design_matrix.to_numpy()
                design_rank = np.linalg.matrix_rank(design_values)
                if design_rank != design_values.shape[1]:
                    raise ValueError(
                        f"Rank-deficient design for task-{task} run-{run_label} "
                        f"trial-{trial_idx:03d}: rank {design_rank}, "
                        f"{design_values.shape[1]} columns"
                    )
                design_condition = float(np.linalg.cond(design_values))
                if not np.isfinite(design_condition) or design_condition > 1e12:
                    raise ValueError(
                        f"Ill-conditioned design for task-{task} run-{run_label} "
                        f"trial-{trial_idx:03d}: {design_condition:.3g}"
                    )

                # breakpoint()
                # fit the model
                glm = FirstLevelModel(
                    t_r=run_tr,
                    slice_time_ref=slice_time_ref,
                    hrf_model=HRF_MODEL,
                    drift_model=DRIFT_MODEL,
                    mask_img=mask_files[0],
                    signal_scaling=False,
                )
                glm.fit(dataset.func[0], design_matrices=lss_design_matrix)

                beta_map = glm.compute_contrast(trial_condition, output_type='effect_size')

                # save the resulting beta map 
                beta_filename = (
                    f"sub-{sub_id}_ses-{session_id}_task-{task}_run-{run_label}_"
                    f"trial-{trial_idx:03d}_beta.nii.gz"
                )
                beta_path = os.path.join(output_dir, beta_filename)
                beta_map.to_filename(beta_path)

                trial_row = events_df.iloc[trial_idx]
                
                if task == 'recognition':
                    stim_info = {"imageid": trial_row["imageid"]}
                elif task == 'retrieval':
                    stim_info = {"cueid": trial_row["cueid"], "targetid": trial_row["targetid"]}
                else:
                    raise ValueError(f"Unknown task: {task}")

                metadata_rows.append({
                    "subject": sub_id,
                    "session": session_id,
                    "task": task,
                    "run": run_label,
                    "trial_idx": trial_idx,
                    **stim_info,
                    "onset": trial_row["onset"],
                    "trial_condition": trial_condition,
                    "design_condition_number": design_condition,
                    "frame_time_start": frame_time_start,
                    "slice_time_ref": slice_time_ref,
                    "event_duration": (
                        float(event_duration)
                        if event_duration is not None
                        else float(trial_row["duration"])
                    ),
                    "beta_path": beta_path,
                })

                print(f"Saved {beta_filename}")

    
    # save the accumulated metadata
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = os.path.join(
        output_root,
        f"beta_maps_metadata_sub-{sub_id}_ses-{session}.csv",
    )
    metadata_df.to_csv(metadata_path, index=False)
    print(f"Done. {len(metadata_df)} trial betas saved.")
    return metadata_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, required=True)
    parser.add_argument('--session', type=str, required=True)
    parser.add_argument('--out-root', type=str, default=".")
    parser.add_argument(
        '--tasks', nargs='+', choices=['recognition', 'retrieval'], default=None)
    parser.add_argument('--runs', nargs='+', default=None)
    parser.add_argument('--bids-root', type=str, default=None)
    parser.add_argument('--output-root', type=str, default=None)
    parser.add_argument('--max-trials', type=int, default=None)
    parser.add_argument(
        '--event-duration',
        type=float,
        default=None,
        help="Override all event durations in each LS-S design (seconds).",
    )
    args = parser.parse_args()
    main(
        subject=args.subject,
        session=args.session,
        out_root=args.out_root,
        tasks=args.tasks,
        runs=args.runs,
        bids_root=args.bids_root,
        output_root=args.output_root,
        max_trials=args.max_trials,
        event_duration=args.event_duration,
    )