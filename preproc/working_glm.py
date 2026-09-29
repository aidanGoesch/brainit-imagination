import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from os.path import join as path
from pathlib import Path
from glob import glob
from nilearn.image import get_data
from nibabel.nifti1 import Nifti1Image
import re

active = [1,  2,  3,  5,  6,  7, 11, 13, 17, 19, 21, 27, 32, 33, 35]
sham =   [8, 10, 15, 16, 18, 20, 22, 24, 26, 28, 31, 36]
sub_args = active + sham
sub_args.sort()


tasks = ['rest_run-1','feedback_run-1','feedback_run-2','feedback_run-3','rest_run-2','tnt_run-1','tnt_run-2','tnt_run-3','tnt_run-4']
alearn_tasks = ['acquisition','extinction','renewal']
mem_order = ['baseline','acquisition','extinction']
rest_tasks = [task for task in tasks if 'rest' in task]

standard = '/scratch/achennings/standard' if '/Users/ah7700' not in sys.base_exec_prefix else '/Users/ah7700/Documents/standard'

work = '/jukebox/norman/achennings'
scratch = '/scratch/achennings'
nilearn_cache = path(scratch,'nilearn_cache') if '/Users/ah7700' not in sys.base_exec_prefix else '/Users/ah7700/Desktop/nilearn_cache'

bids_dir = path(work,'rp-bids')
derivatives = path(bids_dir,'derivatives')
group_masks = path(derivatives,'group_masks') if '/Users/ah7700' not in sys.base_exec_prefix else '/Users/ah7700/Desktop/group_masks'
fmriprep = path(derivatives,'fmriprep')
freesurfer = path(derivatives,'freesurfer')
preproc = path(derivatives,'preproc')
model = path(derivatives,'model')
confounds = path(derivatives,'confounds')
group_model = path(derivatives,'group_model')
exp_dir = os.getcwd().split(os.sep+'analysis')[0]

std_2009c_1mm_brain = f'{standard}/MNI152NLin2009cAsym_T1_1mm_brain.nii.gz'
std_2009c_1mm = f'{standard}/MNI152NLin2009cAsym_T1_1mm.nii.gz'
std_2009c_1mm_brain_mask = f'{standard}/MNI152NLin2009cAsym_T1_1mm_brain_mask.nii.gz'
std_2009c_2mm_brain = f'{standard}/MNI152NLin2009cAsym_T1_2mm_brain.nii.gz'
std_2009c_2mm_brain_mask = f'{standard}/MNI152NLin2009cAsym_T1_2mm_brain_mask.nii.gz'
std_2009c_25mm_brain = f'{standard}/MNI152NLin2009cAsym_T1_25mm_brain.nii.gz'
std_2009c_25mm_brain_mask = f'{standard}/MNI152NLin2009cAsym_T1_25mm_brain_mask.nii.gz'
gm_2mm_thr = f'{standard}/gm_2mm_thr.nii.gz'
gm_25mm_thr = f'{standard}/gm_25mm_thr.nii.gz'
gm_3mm_thr = f'{standard}/gm_3mm_thr.nii.gz'


# pilot_subs = [905,906,907,908,910]
# pilot_subs1 = [905,906,907,908]
# pilot_subs2 = [921,922,923,924,925,926,927,930,931]
# pilot_subs3 = [941,944,950,951,952]#940
# m3_subs = [965,966,967]
# pilot_subs = pilot_subs1 + pilot_subs2 + pilot_subs3
# groups = {'puzzle_active': [905,906,907,908],
#           'induction_active-ppi': [921,922,923,924,930,931],
#           'induction_sham-ppi': [925,926,927],
#           'induction_active-rs': [941,944,950,951,952] }

def lgroup(x):
    if x in active:
        return 'active'
    else:
        return 'sham'
# def lgroup(x):
#     if x in groups['puzzle_active']:
#         return 'puzzle_active'
#     elif x in groups['induction_active-ppi']:
#         return 'induction_active-ppi'
#     elif x in groups['induction_sham-ppi']:
#         return 'induction_sham-ppi'
#     elif x in groups['induction_active-rs']:
#         return 'induction_active-rs'

feedback_tasks = [i for i in tasks if 'feedback' in i]

from scipy.stats import norm
def p2z(p,tail):
    '''returns a z-score corresponding a to the pvalue, given the tail (1 or 2 sided)'''
    return norm.ppf(1-p/tail)

def apply_mask(target=None,mask=None):

    if type(target) == str:
        target = get_data(target)
    elif type(target) == Nifti1Image:
        target = target.get_fdata()

    if type(mask) == str:
        mask = get_data(mask)
    elif type(mask) == Nifti1Image:
        mask = mask.get_fdata()

    coor = np.where(mask == 1)
    values = target[coor]
    if values.ndim > 1:
        values = np.transpose(values) #swap axes to get sample X feature
    return values


def mkdir(path,local=False):
    if not os.path.exists(path) and not local:
        os.makedirs(path)

class bids_meta(object):

    def __init__(self,sub):

        local = False
        if '/Users/ah7700' in sys.base_exec_prefix:
            local = True

        self.num = int(sub)
        self.fsub = f'sub-RP{self.num:03d}'

        if self.num in active:
            self.group = 'active'
        elif self.num in sham:
            self.group = 'sham'

        '''get the csplus category for cc'''
        cc_order_path = f'../../rt-press-tasks/cc-experiment/data/{self.fsub}/inputs/{self.fsub}_ses-4_task-cc-acquisition.csv'
        if os.path.exists(cc_order_path):
            order = pd.read_csv(cc_order_path)
            try:
                self.csplus = order.loc[order.us_order == 'CSUS','cs_stim'].values[0].split('/')[0]
            except:
                self.csplus = 'error'
        else:
            self.csplus = None

        '''folders'''
        self.subj_dir = path(bids_dir,self.fsub)
        self.fmriprep = path(fmriprep,self.fsub)
        self.freesurfer = path(freesurfer,self.fsub)

        self.confounds = path(confounds,self.fsub);mkdir(self.confounds,local)

        '''preproc folder'''
        self.preproc = path(preproc,self.fsub);mkdir(self.preproc,local)
        '''reference folder and things in it'''
        self.reference    = path(self.preproc,'reference');mkdir(self.reference,local)
        self.refvol       = path(self.reference,'boldref.nii.gz')
        self.refvol_mask  = path(self.reference,'boldref_mask.nii.gz')
        self.refvol_brain = path(self.reference,'boldref_brain.nii.gz')
        self.ref2std      = path(self.reference,'ref2std.mat')
        self.std2ref      = path(self.reference,'std2ref.mat')
        '''other things in preproc'''
        self.masks        = path(self.preproc,'masks');mkdir(self.masks,local)
        self.extracted    = path(self.preproc,'extracted');mkdir(self.extracted,local)
        self.betas        = path(self.preproc,'betas');mkdir(self.betas,local)
        self.bold         = path(self.preproc,'bold');mkdir(self.bold,local)
        self.denoised = path(self.preproc,'denoised');mkdir(self.denoised,local)


        '''model folder'''
        self.model = path(model,self.fsub);mkdir(self.model,local)
        # self.decoding = path(self.model,'decoding');mkdir(self.decoding,local)

        if sub < 200:
            self.timing = path(exp_dir,'rp-bids',self.fsub)
        else:
            self.timing = path(exp_dir,'rp-pilot-bids',self.fsub)
        if not os.path.exists(self.timing):os.makedirs(self.timing)



import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import concat_imgs
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SPACE = "MNI152NLin2009cAsym"
MOTION_COLUMNS = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
FD_THRESHOLD = 0.3


def prepare_confounds(confounds, n_volumes):
    """Select nuisance regressors for the non-denoised fMRIPrep BOLD data."""
    required = set(MOTION_COLUMNS + ["framewise_displacement"])
    missing = required - set(confounds.columns)
    if missing:
        raise ValueError(f"Missing confounds: {sorted(missing)}")

    acompcor_columns = [
        column for column in confounds if column.startswith("a_comp_cor_")
    ][:5]
    if len(acompcor_columns) < 5:
        raise ValueError(
            f"Expected at least 5 aCompCor regressors; found {len(acompcor_columns)}"
        )

    nuisance = confounds[MOTION_COLUMNS + acompcor_columns].fillna(0).copy()
    fd = confounds["framewise_displacement"].fillna(0).to_numpy()
    for volume in np.flatnonzero(fd > FD_THRESHOLD):
        outlier = np.zeros(n_volumes)
        outlier[volume] = 1
        nuisance[f"motion_outlier_{volume:03d}"] = outlier
    if len(nuisance) != n_volumes:
        raise ValueError(
            f"Confounds have {len(nuisance)} rows but BOLD has {n_volumes} volumes"
        )
    return nuisance


def retrieval_paths(bids_root, subject, session, run):
    """Return the four BIDS/fMRIPrep files needed for one retrieval run."""
    stem = f"sub-{subject}_ses-{session}_task-retrieval_run-{run}"
    raw_func = (
        Path(bids_root) / f"sub-{subject}" / f"ses-{session}" / "func"
    )
    derivative_func = (
        Path(bids_root) / "derivatives" / f"sub-{subject}"
        / f"ses-{session}" / "func"
    )
    files = {
        "bold": derivative_func
        / f"{stem}_space-{SPACE}_desc-preproc_bold.nii.gz",
        "mask": derivative_func
        / f"{stem}_space-{SPACE}_desc-brain_mask.nii.gz",
        "confounds": derivative_func / f"{stem}_desc-confounds_timeseries.tsv",
        "events": raw_func / f"{stem}_events.tsv",
    }
    missing = [str(file_path) for file_path in files.values()
               if not file_path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing retrieval inputs: {missing}")
    return files


def slice_time_reference(bold_path, t_r):
    """Read fMRIPrep's slice-time reference, falling back to the source default."""
    sidecar = Path(str(bold_path).removesuffix(".nii.gz") + ".json")
    if not sidecar.is_file():
        return 0.5
    with sidecar.open() as file:
        start_time = float(json.load(file).get("StartTime", 0.5 * t_r))
    return start_time / t_r


def lss(
        subject="03", session="001", run="01", bids_root=None,
        output_root=None, max_trials=None):
    """Fit the minimally adapted trial-wise LS-S retrieval model."""
    script_dir = Path(__file__).resolve().parent
    bids_root = Path(bids_root or script_dir / "bids_data").resolve()
    output_root = Path(
        output_root or script_dir / "working_glm_results" / "retrieval"
    ).resolve()
    beta_dir = output_root / "beta_maps"
    beta_dir.mkdir(parents=True, exist_ok=True)

    run = f"{int(run):02d}"
    files = retrieval_paths(bids_root, subject, session, run)
    img = nib.load(files["bold"])
    if len(img.shape) != 4:
        raise ValueError(f"BOLD image must be 4D: {files['bold']}")
    t_r = float(img.header.get_zooms()[3])
    slice_time_ref = slice_time_reference(files["bold"], t_r)

    events = pd.read_csv(files["events"], sep="\t").reset_index(drop=True)
    required_event_columns = {"onset", "duration", "cueid", "targetid"}
    missing = required_event_columns - set(events.columns)
    if missing:
        raise ValueError(f"Missing event columns: {sorted(missing)}")
    events["trial_type"] = "other"

    confounds = pd.read_csv(files["confounds"], sep="\t")
    nuisance = prepare_confounds(confounds, img.shape[3])

    model = FirstLevelModel(
        t_r=t_r,
        slice_time_ref=slice_time_ref,
        hrf_model="spm",
        drift_model=None,
        high_pass=None,
        mask_img=str(files["mask"]),
        signal_scaling=False,
        smoothing_fwhm=None,
        noise_model="ar1",
        n_jobs=1,
        verbose=-1,
        memory=None,
        memory_level=0,
        minimize_memory=True,
    )

    n_trials = len(events) if max_trials is None else min(max_trials, len(events))
    betas = []
    metadata = []
    for trial in range(n_trials):
        print(f"Fitting retrieval run-{run} trial {trial + 1}/{n_trials}")
        beta_events = events.copy()
        beta_events.loc[trial, "trial_type"] = "beta"

        model.fit(run_imgs=img, events=beta_events, confounds=nuisance)
        estimate = model.compute_contrast("beta", output_type="effect_size")
        estimate.header["qform_code"] = 4
        estimate.header["sform_code"] = 4
        betas.append(estimate)

        event = events.iloc[trial]
        metadata.append({
            "subject": subject,
            "session": session,
            "task": "retrieval",
            "run": int(run),
            "trial_idx": trial,
            "cueid": int(event.cueid),
            "targetid": int(event.targetid),
            "onset": float(event.onset),
            "duration": float(event.duration),
            "t_r": t_r,
            "slice_time_ref": slice_time_ref,
        })

    stem = f"sub-{subject}_ses-{session}_task-retrieval_run-{run}"
    beta_path = beta_dir / f"{stem}_desc-lss_betas.nii.gz"
    metadata_path = beta_dir / f"{stem}_desc-lss_metadata.tsv"
    nib.save(concat_imgs(betas), beta_path)
    pd.DataFrame(metadata).to_csv(metadata_path, sep="\t", index=False)
    print(f"Saved {len(betas)} betas to {beta_path}")
    return beta_path, metadata_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="03")
    parser.add_argument("--session", default="001")
    parser.add_argument("--runs", nargs="+", default=["01", "02"])
    parser.add_argument("--bids-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    args = parser.parse_args()
    for run in args.runs:
        lss(
            subject=args.subject,
            session=args.session,
            run=run,
            bids_root=args.bids_root,
            output_root=args.output_root,
            max_trials=args.max_trials,
        )


if __name__ == "__main__":
    main()