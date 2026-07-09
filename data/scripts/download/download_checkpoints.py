#!/usr/bin/env python3
"""Download pretrained Brain-IT checkpoints and voxel mapping from Hugging Face."""

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "RomanBeliy/Brain-IT"

SCRIPT_DIR = Path(__file__).resolve().parent
DERIVED_DATA_DIR = (SCRIPT_DIR / "../../derived_data").resolve()
MODEL_DIR = (SCRIPT_DIR / "../../../results/saved_models").resolve()

FILES = {
    "derived_data/v2c_128_mapping_gmm_v2.npy": DERIVED_DATA_DIR / "v2c_128_mapping_gmm_v2.npy",
    "checkpoints/encoder_ch128.pth": MODEL_DIR / "encoder_ch128.pth",
    "checkpoints/decoder_vgg_ext-1_batch64_save.pth": MODEL_DIR / "decoder_vgg_ext-1_batch64_save.pth",
    "checkpoints/decoder_clipg_ext-1_save.pth": MODEL_DIR / "decoder_clipg_ext-1_save.pth",
    "checkpoints/combined_model.ckpt": MODEL_DIR / "combined_model.ckpt",
}


def download_file(repo_path, dest):
    if dest.exists():
        print(f"skip (exists): {dest}")
        return

    print(f"Downloading {repo_path} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = hf_hub_download(repo_id=REPO_ID, filename=repo_path)
    shutil.copy2(src, dest)


def main():
    for repo_path, dest in FILES.items():
        download_file(repo_path, dest)

    print(f"Done. Files saved under {DERIVED_DATA_DIR} and {MODEL_DIR}")


if __name__ == "__main__":
    main()
