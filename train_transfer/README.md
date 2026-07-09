# Transfer Learning

This directory contains scripts for adapting pretrained models to new subjects/datasets.

> **Note:** This transfer learning procedure is performed within the Natural Scenes Dataset (NSD), using held-out subjects as the transfer target. It can be easily modified to transfer to other datasets; the default NSD checkpoints trained on all subjects can be used as the reference model.

## Overview

The transfer learning pipeline has two stages:

1. **Base training** — Train a model on 7 NSD subjects, holding out one subject (e.g. `--remove_sub 1` excludes subject 1). This follows the same procedure as the regular training pipeline in the main [README](../README.md), but uses the scripts in this directory and excludes the held-out subject from training and validation.
2. **Transfer training** — Fine-tune the base model on the held-out subject using a small amount of single-session data.

Throughout this README, `N` denotes the held-out subject number (1–8). Use the same value consistently across all steps.

---

## Base Training (7 subjects)

Base training uses the processed NSD data from the main [README](../README.md) (download and data processing steps). No additional data preparation is needed.

### 1. Train Encoder

```bash
python train_transfer/train_encoder_base.py --remove_sub N
```

### 2. Clustering and External fMRI Prediction

After training the encoder, cluster voxels and generate synthetic fMRI from external images:

```bash
# Cluster voxels (GMM on base encoder embeddings, excluding held-out subject)
python data/scripts/transfer/get_clusters.py --remove_sub N

# Generate synthetic fMRI from external images using the base encoder
python data/scripts/transfer/pred_fmri_ext_transfer.py --base --subject N
```

### 3. Train Decoder

```bash
# VGG-based decoder with contrastive loss
python train_transfer/train_decoder_base.py --VGG --CONT --EXT --SAVE --remove_sub N

# CLIP-guided decoder (stage 1)
python train_transfer/train_decoder_base.py --CLIPG --EXT --SAVE --remove_sub N
```

### 4. Train Decoder Stage 2 (Diffusion)

```bash
python train_transfer/train_decoder_stage2_base.py --EXT --remove_sub N
```

**Note:** Stage 2 training requires significant GPU memory (see main [README](../README.md)).

---

## Transfer Training (held-out subject)

### Data Preparation

Prepare single-session fMRI data for the held-out subject:

```bash
python data/scripts/transfer/prepare_fmri_single_session.py
```

### 1. Train Encoder

```bash
python train_transfer/train_encoder_transfer.py --SUBJECT N
```

### 2. Map Voxel Clusters and External fMRI Prediction

After training the transfer encoder, map the held-out subject's voxels to base-model cluster centers and generate synthetic fMRI:

**Voxel-to-cluster mapping**: Mapping of new voxels to existing cluster centers is established according to nearest neighbor (NN) in the embedding space.

```bash
# Map transfer voxels to base clusters (NN in embedding space)
python data/scripts/transfer/map_clusters_transfer.py --subject N

# Generate synthetic fMRI from external images using the transfer encoder
python data/scripts/transfer/pred_fmri_ext_transfer.py --subject N
```

### 3. Train Decoder

**Voxel embedding initialization**: The initialization of voxel embeddings in transfer learning uses NN voxel embeddings from the reference model (this works better than random initialization).

```bash
# VGG-based decoder with contrastive loss
python train_transfer/train_decoder_transfer.py --VGG --CONT --EXT --SAVE --SUBJECT N

# CLIP-guided decoder (stage 1)
python train_transfer/train_decoder_transfer.py --CLIPG --EXT --SAVE --SUBJECT N
```

### 4. Train Decoder Stage 2 (Diffusion)

By default, stage-2 transfer saves only the fine-tuned **voxel embeddings**. At inference, these are loaded onto the **base** stage-2 checkpoint (functionally the same as a full transfer checkpoint, but using less storage). Optionally, train with `--SAVE_FULL_CKPT` and infer with `--full_stage2` to use a full transfer stage-2 checkpoint instead.

```bash
python train_transfer/train_decoder_stage2_transfer.py --EXT --SUBJECT N
# Optional: add --SAVE_FULL_CKPT to save the full transfer stage-2 model
```

---

## Inference

```bash
python inference/full_inference.py --transfer --subject N
# Optional: add --full_stage2 if trained with --SAVE_FULL_CKPT
```
