"""
Full inference pipeline for fMRI-to-image reconstruction.
Combines low-level (VGG) and semantic (diffusion) decoders.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.io import imsave
from skimage.transform import resize

# Add paths
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src', 'MindEyeV2'))
sys.path.append(os.path.join(os.getcwd(), 'src', 'MindEyeV2', 'generative_models'))

from utils.datasets import EmbedGraphDataset, collate, DatasetExtWraper
from utils.diffusion_utils import load_diffusion_engine, enhance_recons
from utils.low_level_utils import LowLevelREC
from utils.metric_functions import compute_metrics
from models.combined_diffusion_engine import CombinedDiffusionEngine
import src.MindEyeV2.generative_models.sgm

# Directory constants
DATA_DIR = 'data/nsd_data/'
DERIVED_DATA_DIR = 'data/derived_data/'
MODEL_DIR = 'results/saved_models/'
OUTPUT_DIR = 'results/reconstructions/'

# Test mode settings
test_samples = 5


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Full inference pipeline')
    parser.add_argument('--run_name', type=str, default='nsd_rec', 
                       help='subdirectory name inside output directory')
    parser.add_argument('--vgg_model', type=str, default=None,
                       help='VGG decoder model name')
    parser.add_argument('--clipg_model', type=str, default=None,
                       help='CLIP-guided decoder model name')
    parser.add_argument('--stage2_model', type=str, default=None,
                       help='Stage 2 diffusion model name')
    parser.add_argument('--v2c_mapping', type=str, default=None,
                       help='path to v2c mapping file')
    parser.add_argument('--centers', type=int, default=128,
                       help='number of centers')
    parser.add_argument('--subjects', type=int, nargs='+', default=[0,1,4,6],
                       help='subject indices to process')
    parser.add_argument('--transfer', action='store_true',
                       help='transfer learning inference: single held-out subject with subject-specific voxel space')
    parser.add_argument('--subject', type=int, default=1,
                       help='held-out subject number (1-8) for transfer inference')
    parser.add_argument('--test', action='store_true',
                       help='test mode: only process 5 images of subject 0')
    parser.add_argument('--full_stage2', action='store_true',
                       help='transfer: use full transfer stage2 checkpoint instead of base checkpoint + voxel embeddings')
    parser.add_argument('--voxel_embed_path', type=str, default=None,
                       help='transfer: path to stage2 voxel embedding file')
    
    args = parser.parse_args()
    
    # Override settings in test mode
    if args.test:
        args.subjects = [0]
    
    # Transfer inference processes only the held-out subject, indexed as 0 in its own voxel space
    if args.transfer:
        args.subjects = [0]

    # Resolve default model names for the selected pipeline
    if args.transfer:
        sub = f"subj{args.subject}"
        stage2_name = f"decoder_stage2_transfer_{sub}_ext4"
        base_stage2_name = f"decoder_stage2_ext-1_base_remove_sub_{args.subject}"
        args.vgg_model = args.vgg_model or f"transfer/decoder_transfer_{sub}_vgg_cont_ext4_batch64_save.pth"
        args.clipg_model = args.clipg_model or f"transfer/decoder_transfer_{sub}_clipg_ext4_save.pth"
        if args.full_stage2:
            args.stage2_model = args.stage2_model or f"transfer/{stage2_name}/last.ckpt"
        else:
            args.stage2_model = args.stage2_model or f"transfer/{base_stage2_name}/last.ckpt"
            args.voxel_embed_path = args.voxel_embed_path or f"transfer/{stage2_name}_voxel_embed.pth"
    else:
        args.vgg_model = args.vgg_model or 'decoder_vgg_ext-1_batch64_save.pth'
        args.clipg_model = args.clipg_model or 'decoder_clipg_ext-1_save.pth'
        args.stage2_model = args.stage2_model or 'combined_model.ckpt'

    return args


def load_data(args):
    """Load fMRI data and create data structures"""
    print("Loading data...")
    
    # Load fMRI data
    fmri_data = np.load(DATA_DIR + "fmri_v2.npz")
    num_voxels_subjects = fmri_data['num_voxels_subjects'].astype(int)
    type_sample = fmri_data["type_sample"]
    multi_sub_fmri = fmri_data['multi_sub_fmri']
    
    # Transfer learning uses a single held-out subject in its own voxel space:
    # slice the subject's voxels and treat them as the whole (0-indexed) voxel set
    if args.transfer:
        end = np.cumsum(num_voxels_subjects)
        start = end - num_voxels_subjects
        sub = args.subject - 1
        multi_sub_fmri = multi_sub_fmri[:, start[sub]:end[sub]]
        num_voxels_subjects = np.array([num_voxels_subjects[sub]])
    
    N = num_voxels_subjects.sum()
    
    # Load v2c mapping
    if args.v2c_mapping is not None:
        v2c_mapping = np.load(args.v2c_mapping)
    elif args.transfer:
        v2c_mapping = np.load(DERIVED_DATA_DIR + f"transfer/v2c_128_mapping_gmm_subj{args.subject}.npy")
    else:
        v2c_mapping = np.load(DERIVED_DATA_DIR + "v2c_128_mapping_gmm_v2.npy")
    
    gt_imgs = np.load(DATA_DIR + "nsd_images_224.npy")
    gt_imgs = gt_imgs[type_sample == 2]
    
    return {
        'num_voxels_subjects': num_voxels_subjects,
        'N': N,
        'multi_sub_fmri': multi_sub_fmri,
        'v2c_mapping': v2c_mapping,
        'gt_imgs': gt_imgs,
    }


def load_models(args, device):
    """Load all models"""
    print("Loading models...")
    
    # VGG decoder (low-level)
    print(f"  Loading VGG decoder: {args.vgg_model}")
    vgg_model = torch.load(os.path.join(MODEL_DIR, args.vgg_model))
    vgg_model = vgg_model.eval().to(device).float()
    
    # Load diffusion engine
    print(f"  Loading diffusion engine...")
    diffusion_engine = load_diffusion_engine()
    
    # Load CLIP-guided decoder
    print(f"  Loading CLIP-guided decoder: {args.clipg_model}")
    gnn_model = torch.load(os.path.join(MODEL_DIR, args.clipg_model))
    
    # Load combined stage 2 model
    print(f"  Loading stage 2 model: {args.stage2_model}")
    stage2_path = os.path.join(MODEL_DIR, args.stage2_model)
    combined_model = CombinedDiffusionEngine.load_from_checkpoint(
        stage2_path, 
        map_location="cpu",
        diffusion_engine=diffusion_engine, 
        gnn_model=gnn_model
    )

    if args.transfer and not args.full_stage2:
        print(f"  Loading stage 2 voxel embeddings: {args.voxel_embed_path}")
        voxel_embed = torch.load(os.path.join(MODEL_DIR, args.voxel_embed_path))
        combined_model.gnn_model.voxel_embed = nn.Parameter(voxel_embed, requires_grad=False)

    combined_model = combined_model.to(device)
    
    return {
        'vgg_model': vgg_model,
        'combined_model': combined_model
    }


def get_vgg_predictions(vgg_model, data, args, device, num_samples):
    """Get predictions from VGG decoder"""
    print("Getting VGG predictions...")
    
    num_voxels_subjects = data['num_voxels_subjects']
    multi_sub_fmri = data['multi_sub_fmri']
    v2c_mapping = data['v2c_mapping']
    
    end = np.cumsum(num_voxels_subjects)
    start = end - num_voxels_subjects
    
    embeds_sub = {}
    
    for sub in args.subjects:
        print(f"  Processing subject {sub}...")
        
        sub_vector = np.ones(num_samples, dtype=int) * sub
        Y_placeholder = np.zeros((num_samples, 1))
        
        test_loader = EmbedGraphDataset(
            multi_sub_fmri[:num_samples, start[sub]:end[sub]],
            Y_placeholder, 
            v2c_mapping, 
            sub_vector,
            sub_num_voxels=num_voxels_subjects, 
            sample=False, 
            num_centers=args.centers
        )
        
        predicts = []
        for i in range(num_samples):
            fmri, _, sub_idx, indexes = test_loader[i]
            
            with torch.no_grad():
                predict = vgg_model(
                    fmri.unsqueeze(0).to(device), 
                    sub_idx.to(device),
                    indexes.to(device)
                ).detach().cpu()
            predicts.append(predict)
        
        predicts_test = np.stack(predicts, axis=0)
        embeds_sub[f"sub_{sub}"] = predicts_test
        print(f"    Shape: {predicts_test.shape}")
    
    return embeds_sub


def reconstruct_lowlevel_images(embeds_sub, args, num_samples):
    """Reconstruct low-level images from VGG embeddings"""
    print("Reconstructing low-level images...")
    
    lw_rec = LowLevelREC()
    imgs_lw_sub = {}
    
    for sub in args.subjects:
        print(f"  Processing subject {sub}...")
        imgs = []
        for i in range(num_samples):
            if i % 100 == 0:
                print(f"    Sample {i}/{num_samples}")
            img_i = lw_rec.rec_image(embeds_sub[f"sub_{sub}"][i])
            imgs.append(img_i)
        imgs_lw_sub[sub] = np.stack(imgs)
        print(f"    Output shape: {imgs_lw_sub[sub].shape}")
    
    return imgs_lw_sub


def reconstruct_semantic_images(combined_model, imgs_lw_sub, data, args, device, num_samples):
    """Reconstruct semantic images using diffusion model"""
    print("Reconstructing semantic images with diffusion model...")
    
    num_voxels_subjects = data['num_voxels_subjects']
    multi_sub_fmri = data['multi_sub_fmri']
    v2c_mapping = data['v2c_mapping']
    
    end = np.cumsum(num_voxels_subjects)
    start = end - num_voxels_subjects
    
    imgs_semantic_sub = {}
    
    for sub in args.subjects:
        print(f"  Processing subject {sub}...")
        
        sub_vector = np.ones(num_samples, dtype=int) * sub
        Y_placeholder = np.zeros((num_samples, 1))
        
        test_loader = EmbedGraphDataset(
            multi_sub_fmri[:num_samples, start[sub]:end[sub]],
            Y_placeholder,
            v2c_mapping,
            sub_vector,
            sub_num_voxels=num_voxels_subjects,
            sample=False,
            num_centers=args.centers
        )
        
        imgs = []
        for i in range(num_samples):
            if i % 100 == 0:
                print(f"    Sample {i}/{num_samples}")
            
            fmri, _, sub_idx, indexes = test_loader[i]
            
            with torch.no_grad():
                img = combined_model.generate(
                    fmri.unsqueeze(0).cuda(),
                    sub_idx.cuda(),
                    indexes.cuda(),
                    img_init=imgs_lw_sub[sub][i]
                ).detach().cpu()
            
            imgs.append(img)
        imgs = np.concatenate(imgs,axis=0)
        imgs_semantic_sub[sub] = np.transpose(imgs,[0,2,3,1])  
        print(f"    Output shape: {imgs_semantic_sub[sub].shape}")
    
    return imgs_semantic_sub


def enhance_semantic_images(imgs_semantic_sub, args):
    """Enhance semantic images using SDXL"""
    print("Enhancing semantic images with SDXL...")
    
    imgs_enhanced_sub = {}
    
    for sub in args.subjects:
        print(f"  Processing subject {sub}...")
        imgs_enhanced_sub[sub] = enhance_recons(imgs_semantic_sub[sub])
        print(f"    Shape: {imgs_enhanced_sub[sub].shape}")
    
    return imgs_enhanced_sub


def save_results(imgs_lw, imgs_semantic, imgs_enhanced, data, args, num_samples):
    """Save reconstruction results"""
    print("Saving results...")
    
    output_dir = os.path.join(OUTPUT_DIR, args.run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    for sub in args.subjects:
        sub_dir = os.path.join(output_dir, f"subject_{sub}")
        os.makedirs(sub_dir, exist_ok=True)
        
        # Save low-level reconstructions
        if imgs_lw is not None and sub in imgs_lw:
            imgs_uint8 = (imgs_lw[sub] * 255).astype(np.uint8)
            np.save(os.path.join(sub_dir, "low_level_recons.npy"), imgs_uint8)
            lw_dir = os.path.join(sub_dir, "low_level")
            os.makedirs(lw_dir, exist_ok=True)
            for i in range(num_samples):
                imsave(os.path.join(lw_dir, f"img_{i:04d}.png"), imgs_uint8[i])
        
        # Save semantic reconstructions
        if imgs_semantic is not None and sub in imgs_semantic:
            imgs_uint8 = (imgs_semantic[sub] * 255).astype(np.uint8)
            np.save(os.path.join(sub_dir, "semantic_recons.npy"), imgs_uint8)
            semantic_dir = os.path.join(sub_dir, "semantic")
            os.makedirs(semantic_dir, exist_ok=True)
            for i in range(num_samples):
                imsave(os.path.join(semantic_dir, f"img_{i:04d}.png"), imgs_uint8[i])
        
        # Save enhanced reconstructions
        if imgs_enhanced is not None and sub in imgs_enhanced:
            imgs_resized = np.array([resize(imgs_enhanced[sub][i], (224, 224), anti_aliasing=True) for i in range(num_samples)])
            imgs_uint8 = (imgs_resized * 255).astype(np.uint8)
            np.save(os.path.join(sub_dir, "enhanced_recons.npy"), imgs_uint8)
            enhanced_dir = os.path.join(sub_dir, "enhanced")
            os.makedirs(enhanced_dir, exist_ok=True)
            for i in range(num_samples):
                imsave(os.path.join(enhanced_dir, f"img_{i:04d}.png"), imgs_uint8[i])
    
    print(f"Results saved to {output_dir}")


def compute_and_print_metrics(recons_by_sub, gt_imgs, args, num_samples, recon_type):
    """Compute and print reconstruction metrics against ground truth."""
    gt_imgs = gt_imgs[:num_samples]

    print(f"\nComputing metrics ({recon_type})...")
    for sub in args.subjects:
        print(f"\n--- {recon_type} | subject {sub} ---")
        metrics = compute_metrics(recons_by_sub[sub], gt_imgs)
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")


def main():
    """Main inference pipeline"""
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    data = load_data(args)
    gt_imgs = data['gt_imgs']
    
    # Determine number of samples
    if args.test:
        num_samples = test_samples
    else:
        num_samples = data['multi_sub_fmri'].shape[0]
    
    # Load models
    models = load_models(args, device)
    
    # Get VGG predictions
    embeds_sub = get_vgg_predictions(models['vgg_model'], data, args, device, num_samples)
    
    # Reconstruct low-level images
    imgs_lw_sub = reconstruct_lowlevel_images(embeds_sub, args, num_samples)
    
    # Reconstruct semantic images with diffusion model
    imgs_semantic_sub = reconstruct_semantic_images(models['combined_model'], imgs_lw_sub, data, args, device, num_samples)
    
    # Enhance semantic images
    imgs_enhanced_sub = enhance_semantic_images(imgs_semantic_sub, args)
    
    # Save results
    save_results(imgs_lw_sub, imgs_semantic_sub, imgs_enhanced_sub, data, args, num_samples)

    compute_and_print_metrics(imgs_enhanced_sub, gt_imgs, args, num_samples, 'enhanced')
    
    print("Inference complete!")


if __name__ == "__main__":
    main()
