"""
Encoder inference: predict fMRI from images.

Default images are the shared test set (type_sample == 2), aligned with
multi_sub_fmri. Results are saved per subject.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.getcwd())

DATA_DIR = "data/nsd_data/"
TRANSFER_DATA_DIR = "data/nsd_data/transfer/"
MODEL_DIR = "results/saved_models/"
TRANSFER_MODEL_DIR = "results/saved_models/transfer/"
DEFAULT_OUTPUT_DIR = "results/encoder_predictions/"

INNER_CH = 128

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3]).astype(float)
IMAGENET_STD = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3]).astype(float)


def trans_imgs_shift(img):
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose([2, 0, 1])
    return torch.from_numpy(img.astype(float)).float()


def parse_args():
    parser = argparse.ArgumentParser(description="Encoder inference: image -> fMRI")
    parser.add_argument(
        "--image_set",
        type=str,
        choices=["test", "ext", "all"],
        default="test",
        help="image set to predict (default: test/shared/multi_sub images)",
    )
    parser.add_argument(
        "--images",
        type=str,
        default=None,
        help="path to custom images .npy (overrides --image_set)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="path to encoder checkpoint",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=None,
        help="subject indices (0-7) to save; default: all subjects",
    )
    parser.add_argument(
        "--transfer",
        action="store_true",
        help="use transfer-learning encoder for a single held-out subject",
    )
    parser.add_argument(
        "--subject",
        type=int,
        default=1,
        help="held-out subject number (1-8) for transfer inference",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="directory to save predictions",
    )
    args = parser.parse_args()

    if args.transfer:
        args.subjects = [args.subject - 1]
        args.model = args.model or (
            TRANSFER_MODEL_DIR + f"encoder_transfer_subj{args.subject}.pth"
        )
    else:
        args.model = args.model or MODEL_DIR + f"encoder_ch{INNER_CH}.pth"
        if args.subjects is None:
            args.subjects = list(range(8))

    if args.output_dir is None:
        image_label = "custom" if args.images else args.image_set
        args.output_dir = os.path.join(DEFAULT_OUTPUT_DIR, image_label)

    return args


def load_images(args, fmri_data):
    if args.images is not None:
        images_path = args.images
        print(f"Loading custom images from {images_path}...")
        images = np.load(images_path)
        return images

    if args.image_set == "test":
        type_sample = fmri_data["type_sample"]
        images = np.load(DATA_DIR + "nsd_images_224.npy")
        images = images[type_sample == 2]
        print(f"Loading test/shared images ({images.shape[0]} images)...")
    elif args.image_set == "ext":
        images_path = DATA_DIR + "ext_images_224.npy"
        print(f"Loading external images from {images_path}...")
        images = np.load(images_path)
    elif args.image_set == "all":
        images_path = DATA_DIR + "nsd_images_224.npy"
        print(f"Loading all NSD images from {images_path}...")
        images = np.load(images_path)
    else:
        raise ValueError(f"Unknown image set: {args.image_set}")

    return images


def predict_fmri(encoder_model, images, num_voxels, device):
    num_images = images.shape[0]
    embeds = np.zeros([num_images, num_voxels], dtype=np.float32)

    print(f"Predicting fMRI for {num_images} images...")
    for i in range(num_images):
        if i % 100 == 0:
            print(f"  Processing image {i}/{num_images}")

        image_tensor = trans_imgs_shift(images[i] / 255.0).unsqueeze(0)
        with torch.no_grad():
            pred = encoder_model(
                image_tensor.to(device),
                torch.arange(num_voxels, device=device).unsqueeze(0),
            )
        embeds[i] = pred.detach().cpu().numpy()

    return embeds


def save_predictions(embeds, num_voxels_subjects, subjects, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    end = np.cumsum(num_voxels_subjects)
    start = end - num_voxels_subjects

    print(f"Saving results to {output_dir}...")
    for sub in subjects:
        sub_dir = output_dir / f"subject_{sub}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        sub_embeds = embeds[:, start[sub]:end[sub]].astype(np.float16)
        output_path = sub_dir / "pred_fmri.npy"
        np.save(output_path, sub_embeds)
        print(f"  subject {sub}: {sub_embeds.shape} -> {output_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.transfer:
        fmri_data = np.load(TRANSFER_DATA_DIR + "subjects_single_ses_fmri.npz")
        transfer_sub = f"subj{args.subject}"
        num_voxels_subjects = np.array([fmri_data[transfer_sub].shape[1]])
    else:
        fmri_data = np.load(DATA_DIR + "fmri_v2.npz")
        num_voxels_subjects = fmri_data["num_voxels_subjects"].astype(int)

    num_voxels = int(num_voxels_subjects.sum())
    images = load_images(args, fmri_data)

    print(f"Loading encoder model from {args.model}...")
    encoder_model = torch.load(args.model).eval().to(device)
    embeds = predict_fmri(encoder_model, images, num_voxels, device)
    save_predictions(embeds, num_voxels_subjects, args.subjects, args.output_dir)

    print("Encoder inference complete!")


if __name__ == "__main__":
    main()
