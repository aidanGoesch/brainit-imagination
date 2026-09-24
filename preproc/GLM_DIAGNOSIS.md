# Synthetic GLM diagnosis

## Root cause

`brainiak.utils.fmrisim.convolve_hrf` accepts a fine-resolution stimulus
function but returns a time course already sampled once per TR. The generator
previously passed that result through `downsample_to_tr` again. At TR=2 s and
10 samples/s, this retained only every 20th TR sample, compressed responses
into the first few volumes, and padded the rest of the run with zeros. The GLM
then modeled the original event times, so its beta maps could not recover the
planted patterns.

On the old magnitude-3 dataset, regressors reproducing the bug recover the
ground-truth patterns better than correctly timed regressors (mean condition
correlation 0.487 versus 0.038). With the second downsampling removed, the
existing SPM LS-S model recovers the near-noiseless data (mean trial
correlation 0.621; within-minus-between RSM similarity 0.704).

BrainIAK and SPM condition regressors correlate approximately 0.95 on this
design. Exact-HRF and SPM condition models recover similarly (0.832 versus
0.816), so HRF mismatch is not the primary failure. Exact trial-wise LS-A and
the current target-versus-other LS-S model also recover similarly (0.647
versus 0.621).

## Diagnostic environment

The original `requirements.txt` is a Linux/CUDA environment export and includes
build-host paths such as `file:///home/...`; pip cannot install it directly on
macOS. The portable analysis subset is in `requirements-glm-macos.txt`.

```bash
python3 -m venv .venv-glm
.venv-glm/bin/python -m pip install -r preproc/requirements-glm-macos.txt
.venv-glm/bin/python -m pytest preproc/tests -q
```

Run one fixed-signal noise condition with:

```bash
PYTHONPATH=preproc .venv-glm/bin/python preproc/signal_sweep.py \
  --signal-magnitude 3.0 --noise-level medium \
  --signal-method PSC \
  --out-root preproc/diagnostic_runs/noise_sweep/medium
```

Use `PSC` when comparing noise presets at fixed absolute signal. The default
`CNR_Amp/Noise-SD` intentionally rescales the signal with each run's noise
standard deviation, so holding its magnitude fixed does not constitute a
fixed-signal noise sweep.
