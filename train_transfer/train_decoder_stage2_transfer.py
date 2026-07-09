import sys
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from omegaconf import OmegaConf
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import random
from sklearn.model_selection import train_test_split
from functools import partial
import wandb
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from pytorch_lightning.loggers import TensorBoardLogger

#from finetune_utils import *

# Set GPU configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" #if 4 gpus are used set to 0,1,2,3


sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src', 'MindEyeV2'))
sys.path.append(os.path.join(os.getcwd(), 'src', 'MindEyeV2','generative_models'))

from utils.datasets import EmbedGraphDataset, collate, DatasetExtWraper
from models.combined_diffusion_engine import CombinedDiffusionEngine
from utils.diffusion_utils import load_diffusion_engine

import src.MindEyeV2.generative_models.sgm
from src.MindEyeV2.generative_models.sgm.models.diffusion_no_lightning import DiffusionEngine_nolightning
from src.MindEyeV2.generative_models.sgm.util import instantiate_from_config
from models.decoder_models import dec_param, Decoder
from torch_geometric.nn.conv import TransformerConv

from pytorch_lightning.callbacks import ModelCheckpoint

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    #parser.add_argument('--SAVE', dest='save', action='store_const', const=True, default=False, help='save model')
    parser.add_argument('--EXT', dest='ext', action='store_const', const=True, default=False, help='include external data')
    parser.add_argument('--SAMPLE', dest='ext_sample_factor', type=int, default=4, help='external freq sample')
    parser.add_argument('--GNN_MODEL_PATH', dest='gnn_model_path', type=str, default=None, help='path to trained GNN model')
    parser.add_argument('--MAP_FILE', dest='v2c_mapping', type=str, default=None, help='path to map file')
    parser.add_argument('--NUM_VOX', dest='num_vox', type=int, default=15, help='number of voxels')
    parser.add_argument('--C', dest='centers', type=int, default=128, help='number of centers')
    parser.add_argument('--NUM_EPOCHS', dest='num_epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--NUM_GPUS', dest='num_gpus', type=int, default=2, help='number of GPUs to use')
    parser.add_argument('--SUBJECT', dest='subject', type=int, default=1, help='subject number (1-8) to fine-tune on')
    parser.add_argument('--SAVE_FULL_CKPT', dest='save_full_checkpoint', action='store_const', const=True, default=False,
                        help='save full diffusion checkpoint (default: save only voxel embeddings)')

    #parser.add_argument('--SAVE_EVERY_N_EPOCHS', dest='save_every_n_epochs', type=int, default=1, help='save every n epochs')
    return parser.parse_args()

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

save_dir      = "results/saved_models/transfer/"
data_dir      = "data/nsd_data/transfer/"
nsd_data_dir  = "data/nsd_data/"
derived_data_dir = "data/derived_data/"
transfer_derived_data_dir = "data/derived_data/transfer/"
tensorbaord_dir =  'logs/tensorboard/decoder_stage2/'

args = parse_arguments()

TRANSFER_SUB = f"subj{args.subject}"  # subject held out in base training, now fine-tuned on

# Model checkpoint paths for transfer learning
# Combined model from stage2 base training (trained without this subject)
base_stage2_name = f"decoder_stage2_base_remove_sub_{args.subject}"
if args.ext:
    base_stage2_name = f"decoder_stage2_ext-1_base_remove_sub_{args.subject}"
COMBINED_CHECKPOINT_PATH = save_dir + f"{base_stage2_name}/last.ckpt"
# Decoder transfer model (output of train_decoder_transfer.py --CLIPG --EXT --SAVE)
finetuned_model_name = f"decoder_transfer_{TRANSFER_SUB}_clipg_save.pth"
if args.ext:
    finetuned_model_name = f"decoder_transfer_{TRANSFER_SUB}_clipg_ext{args.ext_sample_factor}_save.pth"
FINETUNED_MODEL_PATH = save_dir + finetuned_model_name

# Training hyperparameters
LEARNING_RATE = 1e-5
BATCH_SIZE = 4 if args.num_gpus == 4 else 8
#SAVE_EVERY_N_EPOCHS = args.save_every_n_epochs
ACCUMULATE_GRAD_BATCHES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transforms
image_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.Normalize([0.5], [0.5])
])



name = f"decoder_stage2_transfer_{TRANSFER_SUB}"
      
if(args.ext):
    name+="_ext"+str(args.ext_sample_factor)

VOXEL_EMBED_PATH = save_dir + f"{name}_voxel_embed.pth"




# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)

def print_gpu_memory_usage():
    """Print GPU memory usage for all available GPUs"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total", flush=True)



# =============================================================================
# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main():
    """Main training function"""
    
    #################################################     load data ###################################################
    
    # Load subject transfer data (produced by prepare_fmri_single_session.py)
    fmri_file = np.load(data_dir + "subjects_single_ses_fmri.npz")
    single_sub_fmri = fmri_file[TRANSFER_SUB]

    imgid_file = np.load(data_dir + "subjects_single_ses_imgid.npz")
    img_ids = imgid_file[TRANSFER_SUB]

    num_voxels_subjects = np.array([single_sub_fmri.shape[1]])
    N = num_voxels_subjects.sum()

    images_train = np.load(nsd_data_dir + "nsd_images_256.npy")[img_ids]

    if(args.ext):  
        fmri_ext = np.load(transfer_derived_data_dir + f"ext_fmri_{TRANSFER_SUB}.npy")
        images_ext = np.load(nsd_data_dir + "ext_images_256.npy")

    # Set up training data
    X_train = single_sub_fmri
    Y_train = images_train
    single_sub_train = np.zeros(single_sub_fmri.shape[0]).astype(int)


    if args.v2c_mapping is not None:
        v2c_mapping = np.load(args.v2c_mapping)
    else:
        v2c_mapping = np.load(transfer_derived_data_dir + f"v2c_128_mapping_gmm_{TRANSFER_SUB}.npy")


    print("load data")
    
    train_dataset = EmbedGraphDataset(X_train, Y_train,v2c_mapping , single_sub_train ,sub_num_voxels = num_voxels_subjects
                                         , sample = True ,num_voxels_to_sample = args.num_vox*1000, 
                                         num_centers = args.centers, transform=image_transform) 
    if(args.ext):
        ext_dataset = EmbedGraphDataset(fmri_ext, images_ext,v2c_mapping , single_sub_train,sub_num_voxels = num_voxels_subjects
                                       , sample = True ,num_voxels_to_sample = args.num_vox*1000,  rand_subject = True, 
                                       num_centers = args.centers, transform=image_transform) 
        full_dataset = DatasetExtWraper(train_dataset,ext_dataset, sample_factor = args.ext_sample_factor)
    
    else:
        full_dataset = train_dataset
    
    custom_collate_fn = partial(collate, N_C=args.centers)
    
    train_dataloader = DataLoader(
        full_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        persistent_workers=True,
    )
    
    print(f"Created dataloader - Train: {len(train_dataloader)} batches", flush=True)
    
    
    #################################################     callbacks ###################################################
     
    
    
    checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(save_dir, name),   # folder to save in
    filename=name,         # name without extension
    save_last=True,          # save only the last epoch
    save_top_k=1            # don't save intermediate/best
)    
        

    logger = TensorBoardLogger(tensorbaord_dir, name=name)

    
    #################################################     load models ###################################################
    # Load models
    diffusion_engine = load_diffusion_engine()
    # Base decoder trained without this subject (used as backbone for combined model)
    gnn_model = torch.load(save_dir + f"decoder_clipg_ext-1_base_remove_sub_{args.subject}_save.pth")

    # Load combined checkpoint
    model = CombinedDiffusionEngine.load_from_checkpoint(
        COMBINED_CHECKPOINT_PATH,
        diffusion_engine=diffusion_engine,
        gnn_model=gnn_model
    )
    
    # Freeze all parameters
    for param in model.diffusion_engine.parameters():
        param.requires_grad = False
    
    for param in model.gnn_model.parameters():
        param.requires_grad = False
    
    print("Froze diffusion_engine and gnn_model after loading checkpoint", flush=True)

    # Load finetuned voxel embeddings
    finetuned_model = torch.load(FINETUNED_MODEL_PATH)
    model.gnn_model.voxel_embed = nn.Parameter(finetuned_model.voxel_embed, requires_grad=True)
    print(f"Loaded voxel embeddings from {FINETUNED_MODEL_PATH}", flush=True)
    
    model.learning_rate = LEARNING_RATE
    print("Created GnnDiffusionEngine with diffusion_engine and gnn_model", flush=True)
        
        
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator='gpu',
        devices=args.num_gpus,
        strategy="ddp",
        precision='16-mixed',
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        check_val_every_n_epoch=1,  # logs at every validation epoch
        enable_model_summary=False,
        logger=logger,
        callbacks=[checkpoint_callback] if args.save_full_checkpoint else [],
       
    )
    
    # Start training
    print("Starting training...", flush=True)
    print("GPU memory before training:", flush=True)
    print_gpu_memory_usage()
    
    
    trainer.fit(model, train_dataloader)
    print("Training complete!", flush=True)

    if not args.save_full_checkpoint:
        trained_model = trainer.model
        if hasattr(trained_model, "module"):
            trained_model = trained_model.module
        torch.save(trained_model.gnn_model.voxel_embed.data, VOXEL_EMBED_PATH)
        print(f"Saved voxel embeddings to {VOXEL_EMBED_PATH}", flush=True)
    
    print("GPU memory after training:", flush=True)
    print_gpu_memory_usage()
    

if __name__ == "__main__":
    main()
