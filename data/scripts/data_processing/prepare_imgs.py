#!/usr/bin/env python3
"""
Script to prepare image arrays at different resolutions (256, 224, 112).
- NSD stimuli are loaded from nsd_stimuli.hdf5
- External images are loaded from a directory of image files
"""

import argparse
import numpy as np
import h5py
import imageio.v2 as iio
from skimage.transform import resize
import os


def load_from_hdf5(hdf5_path):
    """
    Load NSD stimuli images from HDF5 file.

    Args:
        hdf5_path: path to nsd_stimuli.hdf5

    Returns:
        numpy array of shape (N, 256, 256, 3), dtype uint8
    """
    print(f"Loading images from {hdf5_path}...")
    with h5py.File(hdf5_path, "r") as f:
        images = f["imgBrick"][:]
    print(f"  Loaded shape: {images.shape}, dtype: {images.dtype}")
    images = resize_images(images, (256, 256))
    return images


def load_from_dir(img_dir):
    """
    Load images from a directory, center-crop to square and resize to 256x256.

    Args:
        img_dir: directory containing image files

    Returns:
        numpy array of shape (N, 256, 256, 3), dtype uint8
    """
    img_files = sorted([f for f in os.listdir(img_dir) if not f.startswith('.')])
    images = np.zeros((len(img_files), 256, 256, 3), dtype=np.uint8)

    print(f"Loading {len(img_files)} images from {img_dir}...")
    counter = 0
    for i, file in enumerate(img_files):
        if i % 1000 == 0:
            print(f"  Processing image {i}/{len(img_files)}")
        image = iio.imread(os.path.join(img_dir, file))
        if image.ndim == 3:
            height, width, _ = image.shape
            crop_size = min(height, width)
            start_x = (width - crop_size) // 2
            start_y = (height - crop_size) // 2
            cropped = image[start_y:start_y + crop_size, start_x:start_x + crop_size]
            resized = resize(cropped, (256, 256), anti_aliasing=True)
            images[counter] = (resized * 255).astype(np.uint8)
            counter += 1

    images = images[:counter]
    print(f"  Loaded shape: {images.shape}, dtype: {images.dtype}")
    return images


def save_at_resolutions(images_256, base_name, out_dir, resolutions=(224, 112), include_base=True):
    """
    Save images_256 as-is and at each additional resolution.

    Args:
        images_256: numpy array of shape (N, 256, 256, 3), dtype uint8
        base_name: output filename prefix (e.g. 'nsd_images')
        out_dir: directory to save .npy files
        resolutions: additional resolutions to generate besides 256
        include_base: whether to also save the 256 base resolution
    """
    if include_base:
        path_256 = os.path.join(out_dir, f"{base_name}_256.npy")
        print(f"Saving {path_256}...")
        np.save(path_256, images_256)

    for res in resolutions:
        resized = resize_images(images_256, (res, res))
        path = os.path.join(out_dir, f"{base_name}_{res}.npy")
        print(f"Saving {path}...")
        np.save(path, resized)
        print(f"  Saved shape: {resized.shape}, dtype: {resized.dtype}")


def resize_images(images, target_size):
    """
    Resize a batch of images to target size.
    
    Args:
        images: numpy array of shape (N, H, W, 3) in uint8
        target_size: tuple (height, width) for target resolution
        
    Returns:
        numpy array of resized images in uint8
    """
    n_images = images.shape[0]
    resized = np.zeros((n_images, target_size[0], target_size[1], 3), dtype=np.uint8)
    
    print(f"Resizing {n_images} images to {target_size}...")
    for i in range(n_images):
        if i % 1000 == 0:
            print(f"  Processing image {i}/{n_images}")
        resized_img = resize(images[i], target_size, preserve_range=True, anti_aliasing=True)
        resized[i] = resized_img.astype(np.uint8)
    
    return resized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-only", action="store_true",
                        help="Skip COCO images and only generate resolution 224.")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'nsd_data')
    imgs_dir = os.path.join(data_dir, 'imgs')

    if args.inference_only:
        # NSD stimuli from HDF5, only resolution 224
        nsd_images = load_from_hdf5(os.path.join(imgs_dir, 'nsd_stimuli.hdf5'))
        save_at_resolutions(nsd_images, 'nsd_images', data_dir, resolutions=(224,), include_base=False)
    else:
        # # NSD stimuli from HDF5
        # nsd_images = load_from_hdf5(os.path.join(imgs_dir, 'nsd_stimuli.hdf5'))
        # save_at_resolutions(nsd_images, 'nsd_images', data_dir)

        # External images from directory
        ext_images = load_from_dir(os.path.join(imgs_dir, 'unlabeled2017'))
        save_at_resolutions(ext_images, 'ext_images', data_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
