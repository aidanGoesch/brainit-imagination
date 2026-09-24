import os
import argparse
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nilearn.datasets import fetch_language_localizer_demo_dataset
from nilearn.glm.first_level import FirstLevelModel, first_level_from_bids
from nilearn.plotting import plot_design_matrix, plot_stat_map, show
from nilearn.glm.first_level import make_first_level_design_matrix

from bids import BIDSLayout
import nilearn.image as nimg
from sklearn.utils import Bunch

# debug flag - plots design matrix rn
DEBUG = False
NUM_DESIGN_MAT_PLOTS = 3

# default values for the GLM that should hold for all participants and runs
SPACE_NAME = 'MNI152NLin2009cAsym'

TR_VALUE = 2.0
HRF_MODEL = 'spm'        
DRIFT_MODEL = 'cosine'   
FD_THRESH = 0.3


# -------------------------------------- functions -------------------------------------- 
def lss_transformer(events_df, row_number, id_cols):
    """Isolate one trial for LSS, building its label from id_cols."""
    events_df = events_df.copy()

    events_df["trial_id"] = events_df[id_cols].astype(str).agg("_".join, axis=1)

    trial_name = f"{events_df.loc[row_number, 'trial_id']}__{row_number:03d}"

    events_df["trial_type"] = "other"
    events_df.loc[row_number, "trial_type"] = trial_name

    return events_df, trial_name


def make_and_save_design_matrix(events_df, trial_idx, frame_times, hrf_model,
        drift_model, add_regs, add_reg_names, id_cols, output_dir="./design_matrix_checks", tag="",):
    os.makedirs(output_dir, exist_ok=True)

    lss_events_df, trial_condition = lss_transformer(events_df, trial_idx, id_cols=id_cols)

    lss_design_matrix = make_first_level_design_matrix(
        frame_times=frame_times,
        events=lss_events_df,
        hrf_model=hrf_model,
        drift_model=drift_model,
        add_regs=add_regs,
        add_reg_names=add_reg_names,
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


def main(subject, session, out_root="."):
    "Runs main GLM Script"
    sub_id = subject
    session_id = session   # renaming to avoid shadowing the param name below if you reuse `session`

    # set up data layout — use out_root instead of hardcoded paths
    layout = BIDSLayout(f"{out_root}/bids_data",
                         derivatives=f"{out_root}/bids_data/derivatives")

    output_dir = f"{out_root}/beta_maps"
    os.makedirs(output_dir, exist_ok=True)

    metadata_rows = []  # will accumulate one dict per trial

    for task in ['recognition', 'retrieval']:
        if task == 'recognition':
            id_cols = ['imageid']
        elif task == 'retrieval':
            id_cols = ['cueid', 'targetid']
        else:
            raise ValueError(f"Unknown task: {task}")

        num_runs = layout.get(subject=sub_id, session=session_id, task=task,
                           return_type='id', target='run')
        
        for run in num_runs:
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


            for name, files in [("func", func_files), ("mask", mask_files), ("confounds", confounds_files)]:
                assert len(files) == 1, f"Expected exactly 1 {name} file, got {len(files)}: {files}"

            if not (func_files and events_files and confounds_files):
                print(f"Skipping sub-{sub_id} ses-{session_id} task-{task} run-{run}: missing files")
                continue


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
            events_df = pd.read_csv(dataset.events[0], sep="\t")

            img = nib.load(dataset.func[0])
            num_volumes = img.shape[3]
            frame_times = np.arange(num_volumes) * TR_VALUE

            # get confounds from events df 
            confounds_df = pd.read_csv(dataset.confounds[0], sep="\t")

            motion_cols = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
            motion_params = confounds_df[motion_cols].fillna(0)

            # aCompCor — top N components
            N_ACOMPCOR = 5  
            acompcor_cols = [c for c in confounds_df.columns if c.startswith('a_comp_cor_')][:N_ACOMPCOR]
            acompcor_params = confounds_df[acompcor_cols].fillna(0)

            fd = confounds_df["framewise_displacement"].fillna(0)
            outlier_trs = np.where(fd > FD_THRESH)[0]

            # construct nuissance regressor mat
            all_confounds = pd.concat([motion_params, acompcor_params], axis=1)
            add_regs = all_confounds.values
            add_reg_names = all_confounds.columns.tolist()

            assert add_regs.shape[0] == num_volumes, (
                f"confounds have {add_regs.shape[0]} rows, expected {num_volumes} "
                f"(sub-{sub_id} ses-{session_id} task-{task} run-{run})"
            )

            num_trials = len(events_df)

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
                    )

                    if trial_idx >= NUM_DESIGN_MAT_PLOTS:
                        quit(1)

                    continue
                lss_events_df, trial_condition = lss_transformer(events_df, trial_idx, id_cols=id_cols)
                # create actual model
                lss_design_matrix = make_first_level_design_matrix(
                    frame_times=frame_times,
                    events=lss_events_df,
                    hrf_model=HRF_MODEL,
                    drift_model=DRIFT_MODEL,
                    add_regs=add_regs,
                    add_reg_names=add_reg_names
                )

                # breakpoint()
                # fit the model
                glm = FirstLevelModel(t_r=TR_VALUE, hrf_model=HRF_MODEL, drift_model=DRIFT_MODEL, mask_img=mask_files[0], signal_scaling=False)
                glm.fit(dataset.func[0], design_matrices=lss_design_matrix)

                beta_map = glm.compute_contrast(trial_condition, output_type='effect_size')

                # save the resulting beta map 
                beta_filename = (
                    f"sub-{sub_id}_ses-{session_id}_task-{task}_run-{run}_"
                    f"trial-{trial_idx:03d}_beta.nii.gz"
                )
                beta_path = os.path.join(output_dir, beta_filename)
                beta_map.to_filename(beta_path)

                trial_row = events_df.loc[trial_idx]
                
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
                    "run": run,
                    "trial_idx": trial_idx,
                    **stim_info,
                    "onset": trial_row["onset"],
                    "beta_path": beta_path,
                })

                print(f"Saved {beta_filename}")

    
    # save the accumulated metadata
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(f"{out_root}/beta_maps_metadata_sub-{sub_id}_ses-{session}.csv", index=False)
    print(f"Done. {len(metadata_df)} trial betas saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, required=True)
    parser.add_argument('--session', type=str, required=True)
    parser.add_argument('--out-root', type=str, default=".")
    args = parser.parse_args()
    main(subject=args.subject, session=args.session, out_root=args.out_root)