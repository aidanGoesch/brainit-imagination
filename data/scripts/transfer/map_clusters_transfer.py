import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from sklearn.preprocessing import normalize
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1, help='subject number (1-8) held out in base training')
args = parser.parse_args()

inner_ch = 128
centers_num = 128
TRANSFER_SUB = f"subj{args.subject}"
base_save_dir = "results/saved_models/transfer/"
derived_data_dir = "data/derived_data/transfer/"

model = torch.load(f"{base_save_dir}encoder_transfer_{TRANSFER_SUB}.pth").eval()

vox_embed = model.voxel_embed.detach().cpu()
vox_embed_normalized = normalize(vox_embed, axis=1)

model = torch.load(f"{base_save_dir}encoder_ch{inner_ch}_base_remove_sub_{args.subject}.pth").eval()
vox_embed_ref = model.voxel_embed.detach().cpu()
vox_embed_ref_normalized = normalize(vox_embed_ref, axis=1)

file = np.load(f"{derived_data_dir}gmm_centers_{centers_num}_remove_sub_{args.subject}.npz")

labels = file["labels"]
centers = file["centers"]

with torch.no_grad():
    dis_l2 = torch.cdist(torch.from_numpy(vox_embed_ref_normalized), torch.from_numpy(vox_embed_normalized))

nn_vox = dis_l2.argmin(0)
nn_vox = nn_vox.numpy()
label_vox = labels[nn_vox]

mapping = np.ones([centers_num, (np.unique(label_vox, return_counts=True)[1]).max()], dtype=int) * -1

for c in range(centers_num):
    inds_c = np.where(label_vox == c)[0]
    mapping[c, :len(inds_c)] = inds_c

os.makedirs(derived_data_dir, exist_ok=True)
np.save(f"{derived_data_dir}v2c_128_mapping_gmm_{TRANSFER_SUB}.npy", mapping)
np.save(f"{derived_data_dir}v2c_{TRANSFER_SUB}_nnvox.npy", nn_vox)
