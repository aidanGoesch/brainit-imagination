import os
import sys
import argparse
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch 
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture

sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument('--remove_sub', type=int, default=1, help='subject number (1-8) held out in base training')
args = parser.parse_args()
removed_sub = args.remove_sub
inner_ch = 128

# Load DINOv2 encoder
torch.hub.set_dir("data/external_models/torch_hub/")
encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')

# Load base encoder model (trained without the held-out subject)
base_name = f"encoder_ch{inner_ch}_base_remove_sub_{removed_sub}"
model = torch.load(f"results/saved_models/transfer/{base_name}.pth").eval()

vox_embed = model.voxel_embed.cpu().detach().numpy()
vox_embed.shape[0]

vox_embed_normalized = normalize(vox_embed, axis=1)

data_dir = 'data/nsd_data/'
num_voxels_subjects = np.load(data_dir + 'fmri_v2.npz')['num_voxels_subjects'].astype(int)
end_ind = np.cumsum(num_voxels_subjects)

start_ind = end_ind - num_voxels_subjects

# Ignore the held-out subject's voxels (untrained in the base model), keep original ordering
used_mask = np.ones(vox_embed_normalized.shape[0], dtype=bool)
used_mask[start_ind[removed_sub - 1]:end_ind[removed_sub - 1]] = False
used_inds = np.where(used_mask)[0]

centers_num = 128
gmm = GaussianMixture(n_components=centers_num, verbose=1)
gmm.fit(vox_embed_normalized[used_mask])
labels_used = gmm.predict(vox_embed_normalized[used_mask])
centers = gmm.means_

# Full-length labels aligned to original voxel ordering (-1 for ignored voxels)
labels = np.full(vox_embed_normalized.shape[0], -1, dtype=int)
labels[used_inds] = labels_used

output_dir = Path(__file__).resolve().parent.parent.parent / "derived_data" / "transfer"
output_dir.mkdir(parents=True, exist_ok=True)
np.savez(output_dir / f"gmm_centers_{centers_num}_remove_sub_{removed_sub}.npz", labels=labels, centers=centers)

mapping = np.ones([centers_num, (np.unique(labels_used, return_counts=True)[1]).max()], dtype=int) * -1

for c in range(centers_num):
    inds_c = used_inds[np.where(labels_used == c)[0]]
    mapping[c, :len(inds_c)] = inds_c

np.save(output_dir / f"v2c_{centers_num}_mapping_gmm_remove_sub_{removed_sub}.npy", mapping)
