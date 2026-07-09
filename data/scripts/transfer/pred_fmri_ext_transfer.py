#!/usr/bin/env python3
"""
Script to predict synthetic fMRI from external images using trained encoder.
Transfer learning version for data/transfer pipeline.
"""

import sys
import os
import argparse
import torch 
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.getcwd())
device = torch.device("cuda")

inner_ch = 128


def trans_imgs_shift(img):
    """
    Transform and normalize images for encoder model.
    
    Args:
        img: numpy array of shape (H, W, 3) with values in [0, 1]
    
    Returns:
        torch tensor of shape (C, H, W) normalized for model input
    """
    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3]).astype(float)
    std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3]).astype(float)
    img = (img - mean) / std
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img.astype(float)).float()
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', dest='base', action='store_const', const=True, default=False, help='use base encoder; default is transfer encoder')
    parser.add_argument('--subject', type=int, default=1, help='subject number (1-8) held out in base training / fine-tuned in transfer')
    args = parser.parse_args()

    TRANSFER_SUB = f"subj{args.subject}"
    base_save_dir = "results/saved_models/transfer/"
    nsd_data_dir = "data/nsd_data/"
    transfer_data_dir = "data/nsd_data/transfer/"

    # Load trained encoder model (base or transfer) and resolve number of voxels
    if args.base:
        print("Loading base encoder model...")
        encoder_model = torch.load(f"{base_save_dir}encoder_ch{inner_ch}_base_remove_sub_{args.subject}.pth").eval()
        fmri_data = np.load(nsd_data_dir + "fmri_v2.npz")
        num_voxels = int(fmri_data['num_voxels_subjects'].astype(int).sum())
        output_name = f"ext_fmri_base_remove_sub_{args.subject}.npy"
    else:
        print("Loading transfer learning encoder model...")
        encoder_model = torch.load(f"{base_save_dir}encoder_transfer_{TRANSFER_SUB}.pth").eval()
        fmri_data = np.load(transfer_data_dir + "subjects_single_ses_fmri.npz")
        num_voxels = fmri_data[TRANSFER_SUB].shape[1]
        output_name = f"ext_fmri_{TRANSFER_SUB}.npy"
    encoder_model = encoder_model.to(device)
    print("  Model loaded!")
    
    # Load external images (224x224)
    ext_imgs_path = nsd_data_dir + "ext_images_224.npy"
    
    if not os.path.exists(ext_imgs_path):
        print(f"Error: {ext_imgs_path} not found!")
        print("Please ensure external images are available for synthetic fMRI generation.")
        return
    
    print(f"\nLoading external images from {ext_imgs_path}...")
    ext_imgs = np.load(ext_imgs_path)
    print(f"  Shape: {ext_imgs.shape}, dtype: {ext_imgs.dtype}")
    
    print(f"\nNumber of voxels: {num_voxels}")
    
    # Predict fMRI for external images
    print(f"\nPredicting fMRI for {ext_imgs.shape[0]} images...")
    embeds = np.zeros([ext_imgs.shape[0], num_voxels], dtype=np.float32)
    
    for i in range(ext_imgs.shape[0]):
        if i % 1000 == 0:
            print(f"  Processing image {i}/{ext_imgs.shape[0]}")
        
        # Preprocess image
        image_tensor = trans_imgs_shift(ext_imgs[i] / 255.0).unsqueeze(0)
        
        # Predict fMRI
        with torch.no_grad():
            pred = encoder_model(image_tensor.cuda(), torch.arange(num_voxels).unsqueeze(0))
        
        embeds[i] = pred.detach().cpu().numpy()
    
    # Save predicted fMRI
    output_dir = "data/derived_data/transfer/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir + output_name
    
    print(f"\nSaving to {output_path}...")
    np.save(output_path, embeds.astype(np.float16))
    print(f"  Saved shape: {embeds.shape}, dtype: float16")
    
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
