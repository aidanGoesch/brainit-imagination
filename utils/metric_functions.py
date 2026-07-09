# from MindEyev2 git: final_evaluations.ipynb
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import torch
import random
import numpy as np
from torchvision import transforms
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#from finetune_utils import IMAGE_TABLE_DIR,CAPTION_TABLE_DIR,NSD_DATA_PATH

import torch
import numpy as np
import random

import torch
import numpy as np
import random

@torch.no_grad()
def top1_identification_nway(all_recons, gt_images, model, preprocess, feature_layer=None, num_experiments=1,n=1000):
    num_images = len(all_recons)

    # Load distractor pool from NSD
    all_nsd = np.load(NSD_DATA_PATH + "/all_images_v2.npy")
    fmri_files = np.load(NSD_DATA_PATH + "/fmri_v2.npz")
    type_sample = fmri_files["type_sample"]
    distractor_pool = all_nsd[type_sample != 2]
    distractor_pool = torch.from_numpy((distractor_pool/255).astype(np.float16)).permute(0,3,1,2).to(device)
    
    per_image_accuracies = []  # each entry is a list of 0/1 values over experiments
    for i, recon in enumerate(all_recons):
        true_img = gt_images[i]
        image_results = []

        for _ in range(num_experiments):
            # Sample 999 distractors
            distractor_imgs = random.sample(list(distractor_pool), n-1)
            candidates = [true_img] + distractor_imgs

            # Encode features
            recon_feat = model(preprocess(recon).unsqueeze(0).to(device))
            real_feats = model(torch.stack([preprocess(img).to(device) for img in candidates]))

            if feature_layer is not None:
                recon_feat = recon_feat[feature_layer]
                real_feats = real_feats[feature_layer]

            recon_feat = recon_feat.float().flatten(1).cpu().numpy()
            real_feats = real_feats.float().flatten(1).cpu().numpy()

            # Correlation
            scores = np.corrcoef(recon_feat, real_feats)[0, 1:]
            top_idx = np.argmax(scores)

            image_results.append(1 if top_idx == 0 else 0)

        per_image_accuracies.append(image_results)

    # Compute per-image mean and std
    per_image_means = [np.mean(accs) for accs in per_image_accuracies]
    per_image_stds  = [np.std(accs) for accs in per_image_accuracies]

    mean_accuracy = np.mean(per_image_means)
    avg_std = np.mean(per_image_stds) if num_experiments > 1 else None

    return (mean_accuracy, avg_std) if num_experiments > 1 else mean_accuracy



@torch.no_grad()
def two_way_identification(all_recons, all_images, model, preprocess, feature_layer=None, return_avg=True):
    preds = model(torch.stack([preprocess(recon) for recon in all_recons], dim=0).to(device))
    reals = model(torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device))
    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()

    r = np.corrcoef(reals, preds)
    r = r[:len(all_images), len(all_images):]
    congruents = np.diag(r)

    success = r < congruents
    success_cnt = np.sum(success, 0)

    if return_avg:
        perf = np.mean(success_cnt) / (len(all_images)-1)
        return perf
    else:
        return success_cnt, len(all_images)-1
    
def compute_lpips(all_recons,all_images,metric):
    from lpips import LPIPS
    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
    ])
    preprocess_recons = preprocess(all_recons)
    preprocess_images = preprocess(all_images)
    
    # Convert from [0, 1] to [-1, 1] range as expected by LPIPS
    preprocess_recons = preprocess_recons * 2.0 - 1.0
    preprocess_images = preprocess_images * 2.0 - 1.0
    
    lpips = LPIPS(net=metric).to(device).eval()
    lpips_score = lpips(preprocess_recons.to(device),preprocess_images.to(device))
    lpips_score = lpips_score.mean()
    print('lpips',lpips_score)
    return lpips_score.item()

def compute_pixcorr(all_recons,all_images):
    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
    ])

    # Flatten images while keeping the batch dimension
    all_images_flattened = preprocess(all_images).reshape(len(all_images), -1).cpu()
    all_recons_flattened = preprocess(all_recons).reshape(len(all_recons), -1).cpu()

    print(all_images_flattened.shape)
    print(all_recons_flattened.shape)

    corrsum = 0
    for i in tqdm(range(len(all_images))):
        corrsum += np.corrcoef(all_images_flattened[i], all_recons_flattened[i])[0][1]
    corrmean = corrsum / len(all_images)

    pixcorr = corrmean
    print(pixcorr)
    return pixcorr

def compute_ssim(all_recons,all_images):
    # see https://github.com/zijin-gu/meshconv-decoding/issues/3
    from skimage.color import rgb2gray
    from skimage.metrics import structural_similarity as ssim

    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR), 
    ])

    # convert image to grayscale with rgb2grey
    img_gray = rgb2gray(preprocess(all_images).permute((0,2,3,1)).cpu())
    recon_gray = rgb2gray(preprocess(all_recons).permute((0,2,3,1)).cpu())
    print("converted, now calculating ssim...")

    ssim_score=[]
    for im,rec in tqdm(zip(img_gray,recon_gray),total=len(all_images)):
        ssim_score.append(ssim(rec, im, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0))

    ssim = np.mean(ssim_score)
    print(ssim)
    return ssim
def compute_color_ssim(all_recons,all_images):
    # see https://github.com/zijin-gu/meshconv-decoding/issues/3
    from skimage.metrics import structural_similarity as ssim

    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR), 
    ])

    # Preprocess images: resize and change to (N, H, W, C) for skimage
    img = preprocess(all_images).permute(0, 2, 3, 1).cpu().numpy()
    recon = preprocess(all_recons).permute(0, 2, 3, 1).cpu().numpy()

    print("Converted to (N, H, W, C), now calculating SSIM...")

    ssim_scores = []
    for im, rec in tqdm(zip(img, recon), total=len(all_images)):
        score = ssim(
            rec, im,
            channel_axis=-1,  # updated argument in newer versions of skimage
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            data_range=1.0
        )
        ssim_scores.append(score)

    mean_ssim = np.mean(ssim_scores)
    print(f"Mean SSIM: {mean_ssim:.4f}")
    return mean_ssim

def compute_alexnet_2way_id(all_recons,all_images):
    from torchvision.models import alexnet, AlexNet_Weights
    alex_weights = AlexNet_Weights.IMAGENET1K_V1

    alex_model = create_feature_extractor(alexnet(weights=alex_weights), return_nodes=['features.4','features.11']).to(device)
    alex_model.eval().requires_grad_(False)

    # see alex_weights.transforms()
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    layer = 'early, AlexNet(2)'
    print(f"\n---{layer}---")
    all_per_correct = two_way_identification(all_recons.to(device).float(), all_images, 
                                                            alex_model, preprocess, 'features.4')
    alexnet2 = np.mean(all_per_correct)
    print(f"2-way Percent Correct: {alexnet2:.4f}")

    layer = 'mid, AlexNet(5)'
    print(f"\n---{layer}---")
    all_per_correct = two_way_identification(all_recons.to(device).float(), all_images, 
                                                            alex_model, preprocess, 'features.11')
    alexnet5 = np.mean(all_per_correct)
    print(f"2-way Percent Correct: {alexnet5:.4f}")
    return alexnet2,alexnet5

def compute_inceptionv3_2way_id(all_recons,all_images):
    from torchvision.models import inception_v3, Inception_V3_Weights
    weights = Inception_V3_Weights.DEFAULT
    inception_model = create_feature_extractor(inception_v3(weights=weights), 
                                            return_nodes=['avgpool']).to(device)
    inception_model.eval().requires_grad_(False)

    # see weights.transforms()
    preprocess = transforms.Compose([
        transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    all_per_correct = two_way_identification(all_recons, all_images,
                                            inception_model, preprocess, 'avgpool')
            
    inception = np.mean(all_per_correct)
    print(f"2-way Percent Correct: {inception:.4f}")
    return inception

def compute_clip_nway_id(all_recons,all_images,n=1000,num_experiments=1):
    import clip
    clip_model, preprocess = clip.load("ViT-L/14", device=device)

    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    clip_ = top1_identification_nway(all_recons, all_images,
                                            clip_model.encode_image, preprocess, None, num_experiments=num_experiments,n=n) # final layer
    if num_experiments > 1:
        clip_mean = clip_[0]
        clip_std = clip_[1]
    else:
        clip_mean = clip_
        clip_std = -1
    print(f"{n}-way Percent Correct: {clip_mean:.4f} std {clip_std:.4f}")
    return clip_mean,clip_std

def compute_clip_2way_id(all_recons,all_images):
    import clip
    clip_model, preprocess = clip.load("ViT-L/14", device=device)

    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    all_per_correct = two_way_identification(all_recons, all_images,
                                            clip_model.encode_image, preprocess, None) # final layer
    clip_ = np.mean(all_per_correct)
    print(f"2-way Percent Correct: {clip_:.4f}")
    return clip_

def compute_efficientNet_dist(all_recons,all_images):
    import scipy as sp
    from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
    weights = EfficientNet_B1_Weights.DEFAULT
    eff_model = create_feature_extractor(efficientnet_b1(weights=weights), 
                                        return_nodes=['avgpool'])
    eff_model.eval().requires_grad_(False)

    # see weights.transforms()
    preprocess = transforms.Compose([
        transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    gt = eff_model(preprocess(all_images))['avgpool']
    gt = gt.reshape(len(gt),-1).cpu().numpy()
    fake = eff_model(preprocess(all_recons))['avgpool']
    fake = fake.reshape(len(fake),-1).cpu().numpy()

    effnet = np.array([sp.spatial.distance.correlation(gt[i],fake[i]) for i in range(len(gt))]).mean()
    print("Distance:",effnet)
    return effnet

def compute_swav_dist(all_recons,all_images):
    import scipy as sp
    swav_model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    swav_model = create_feature_extractor(swav_model, 
                                        return_nodes=['avgpool'])
    swav_model.eval().requires_grad_(False)

    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    gt = swav_model(preprocess(all_images))['avgpool']
    gt = gt.reshape(len(gt),-1).cpu().numpy()
    fake = swav_model(preprocess(all_recons))['avgpool']
    fake = fake.reshape(len(fake),-1).cpu().numpy()

    swav = np.array([sp.spatial.distance.correlation(gt[i],fake[i]) for i in range(len(gt))]).mean()
    print("Distance:",swav)
    return swav

def add_to_table_img(metrics,model_name):
    import os
    import pandas as pd

    # Create a dictionary to store variable names and their corresponding values
    metrics_dict = {
        "model_name": model_name,
        "Top1-nway_CLIP": metrics["top1_nway"], # imgs
        "Top1-nway_CLIP_std": metrics["top1_nway_std"], # imgs
        "PixCorr": metrics["pixcorr"], # imgs
        "Color SSIM": metrics["color_ssim"], # imgs
        "SSIM": metrics["ssim"], # imgs
        "AlexNet(2)": metrics["alexnet2"], # imgs
        "AlexNet(5)": metrics["alexnet5"], # imgs
        "InceptionV3": metrics["inceptionv3"], # imgs
        "CLIP": metrics["clip_"], # imgs
        "EffNet-B": metrics["effnet"], # imgs
        "SwAV": metrics["swav"], # imgs
        "LPIPS_alex": metrics["lpips_alex"], # imgs
    }

    # Create DataFrame with one row
    df = pd.DataFrame([metrics_dict])
    # Output path
    output_path = IMAGE_TABLE_DIR
    # If the file exists, append the new row
    if os.path.exists(output_path):
        df_existing = pd.read_csv(output_path)
        df_combined = pd.concat([df_existing, df], ignore_index=True)
        df_combined.to_csv(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

def compute_meteor(all_predcaptions,all_git_generated_captions,all_captions):
    import nltk
    from nltk.translate import meteor_score
    from nltk.tokenize import word_tokenize

    # Make sure nltk resources are downloaded
    nltk.download('punkt')
    nltk.download('wordnet')

    # Assuming all_git_generated_captions and all_predcaptions are your predictions
    # and all_captions are the references

    # Tokenize the captions and predictions
    all_captions_tokenized = [word_tokenize(ref) for ref in all_captions]
    all_git_generated_captions_tokenized = [word_tokenize(pred) for pred in all_git_generated_captions]
    all_predcaptions_tokenized = [word_tokenize(pred) for pred in all_predcaptions]

    # Compute METEOR between generated captions and references
    meteor_img_ref = [meteor_score.meteor_score([ref], pred) for ref, pred in zip(all_captions_tokenized, all_git_generated_captions_tokenized)]

    # Compute METEOR between brain-based predictions and references
    meteor_brain_ref = [meteor_score.meteor_score([ref], pred) for ref, pred in zip(all_captions_tokenized, all_predcaptions_tokenized)]

    # Compute METEOR between brain-based predictions and generated captions
    meteor_brain_img = [meteor_score.meteor_score([ref], pred) for ref, pred in zip(all_git_generated_captions_tokenized, all_predcaptions_tokenized)]

    # Calculate relative METEOR
    relative_brain_image_meteor = [b / a if a != 0 else 0 for a, b in zip(meteor_img_ref, meteor_brain_img)]

    # Print results
    meteor_img_ref = {'meteor':sum(meteor_img_ref)/len(meteor_img_ref)}
    meteor_brain_ref = {'meteor':sum(meteor_brain_ref)/len(meteor_brain_ref)}
    meteor_brain_img = {'meteor':sum(meteor_brain_img)/len(meteor_brain_img)}
    relative_brain_image_meteor = {'meteor':sum(relative_brain_image_meteor)/len(relative_brain_image_meteor)}

    print(f"[GROUND] METEOR GIT from images vs captions: {meteor_img_ref['meteor']}")
    print(f"[ABSOLUTE] METEOR GIT from brain vs captions: {meteor_brain_ref['meteor']}")
    print(f"[ABSOLUTE] METEOR GIT from brain vs images: {meteor_brain_img['meteor']}")
    print(f"[RELATIVE] METEOR: {relative_brain_image_meteor['meteor']}")
    return meteor_img_ref['meteor'],meteor_brain_ref['meteor'],meteor_brain_img['meteor'],relative_brain_image_meteor['meteor']

def compute_rouge(all_predcaptions,all_git_generated_captions,all_captions):
    import evaluate
    rouge = evaluate.load('rouge')
    rouge_img_ref=rouge.compute(predictions=all_git_generated_captions,references=all_captions)
    rouge_brain_ref=rouge.compute(predictions=all_predcaptions,references=all_captions)
    rouge_brain_img=rouge.compute(predictions=all_predcaptions,references=all_git_generated_captions)


    relative_brain_image_rouge1 = rouge_brain_img['rouge1']/rouge_img_ref['rouge1']
    relative_brain_image_rougeL = rouge_brain_img['rougeL']/rouge_img_ref['rougeL']

    print(f"[GROUND] ROUGE-1 GIT from images vs captions: {rouge_img_ref['rouge1']}")
    print(f"[ABSOLUTE] ROUGE-1 GIT from brain vs captions: {rouge_brain_ref['rouge1']}")
    print(f"[ABSOLUTE] ROUGE-1 GIT from brain vs images: {rouge_brain_img['rouge1']}")
    print(f"[RELATIVE] ROUGE-1  {relative_brain_image_rouge1}")

    print(f"[GROUND] ROUGE-L GIT from images vs captions: {rouge_img_ref['rougeL']}")
    print(f"[ABSOLUTE] ROUGE-L GIT from brain vs captions: {rouge_brain_ref['rougeL']}")
    print(f"[ABSOLUTE] ROUGE-L GIT from brain vs images: {rouge_brain_img['rougeL']}")
    print(f"[RELATIVE] ROUGE-L  {relative_brain_image_rougeL}")
    return rouge_img_ref['rouge1'],rouge_brain_ref['rouge1'],rouge_brain_img['rouge1'],relative_brain_image_rouge1,rouge_img_ref['rougeL'],rouge_brain_ref['rougeL'],rouge_brain_img['rougeL'],relative_brain_image_rougeL

def compute_sentence_transformers(all_predcaptions,all_git_generated_captions,all_captions):
    from sentence_transformers import SentenceTransformer, util
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    with torch.no_grad():
        embedding_brain= sentence_model.encode(all_predcaptions, convert_to_tensor=True)
        embedding_captions = sentence_model.encode(all_captions, convert_to_tensor=True)
        embedding_images = sentence_model.encode(all_git_generated_captions, convert_to_tensor=True)

        ss_sim_brain_img=util.pytorch_cos_sim(embedding_brain, embedding_images).cpu()
        ss_sim_brain_cap=util.pytorch_cos_sim(embedding_brain, embedding_captions).cpu()
        ss_sim_img_cap=util.pytorch_cos_sim(embedding_images, embedding_captions).cpu()

        relative_brain_image_ss=ss_sim_brain_img.diag().mean()/ss_sim_img_cap.diag().mean()
        print(f"[GROUND] Sentence Transformer Similarity GIT from images vs captions: {ss_sim_img_cap.diag().mean()}")
        print(f"[ABSOLUTE] Sentence Transformer Similarity GIT from brain vs captions: {ss_sim_brain_cap.diag().mean()}")
        print(f"[ABSOLUTE] Sentence Transformer Similarity GIT from brain vs images: {ss_sim_brain_img.diag().mean()}")
        print(f"[RELATIVE] Sentence Transformer Similarity   {relative_brain_image_ss.mean()}")
        return ss_sim_img_cap.diag().mean(),ss_sim_brain_cap.diag().mean(),ss_sim_brain_img.diag().mean(),relative_brain_image_ss.mean()

def compute_clip_b_score(all_predcaptions,all_git_generated_captions,all_captions):
    from transformers import CLIPModel, AutoTokenizer, AutoProcessor
    from sentence_transformers import SentenceTransformer, util
    model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor_clip = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer =  AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    with torch.no_grad():
        input_ids=tokenizer(list(all_predcaptions),return_tensors="pt",padding=True)
        embedding_brain= model_clip.get_text_features(**input_ids)

        input_ids=tokenizer(list(all_captions),return_tensors="pt",padding=True)
        embedding_captions= model_clip.get_text_features(**input_ids)

        input_ids=tokenizer(all_git_generated_captions,return_tensors="pt",padding=True)
        embedding_images= model_clip.get_text_features(**input_ids)

    clip_B_sim_brain_img=util.pytorch_cos_sim(embedding_brain, embedding_images).cpu()
    clip_B_sim_brain_cap=util.pytorch_cos_sim(embedding_brain, embedding_captions).cpu()
    clip_B_sim_img_cap=util.pytorch_cos_sim(embedding_images, embedding_captions).cpu()

    relative_brain_image_clip_B=clip_B_sim_brain_img.diag().mean()/clip_B_sim_img_cap.diag().mean()

    print(f"[GROUND] CLIP Similarity GIT from images vs captions: {clip_B_sim_img_cap.diag().mean()}")
    print(f"[ABSOLUTE] CLIP Similarity GIT from brain vs captions: {clip_B_sim_brain_cap.diag().mean()}")
    print(f"[ABSOLUTE] CLIP Similarity GIT from brain vs images: {clip_B_sim_brain_img.diag().mean()}")
    print(f"[RELATIVE] CLIP Similarity   {relative_brain_image_clip_B.mean()}")
    return clip_B_sim_img_cap.diag().mean(),clip_B_sim_brain_cap.diag().mean(),clip_B_sim_brain_img.diag().mean(),relative_brain_image_clip_B.mean()

def compute_clip_l_score(all_predcaptions,all_git_generated_captions,all_captions):
    from transformers import CLIPModel, AutoTokenizer, AutoProcessor
    from sentence_transformers import SentenceTransformer, util

    model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor_clip = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer =  AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    with torch.no_grad():
        input_ids=tokenizer(list(all_predcaptions),return_tensors="pt",padding=True)
        embedding_brain= model_clip.get_text_features(**input_ids)

        input_ids=tokenizer(list(all_captions),return_tensors="pt",padding=True)
        embedding_captions= model_clip.get_text_features(**input_ids)

        input_ids=tokenizer(all_git_generated_captions,return_tensors="pt",padding=True)
        embedding_images= model_clip.get_text_features(**input_ids)

    clip_L_sim_brain_img=util.pytorch_cos_sim(embedding_brain, embedding_images).cpu()
    clip_L_sim_brain_cap=util.pytorch_cos_sim(embedding_brain, embedding_captions).cpu()
    clip_L_sim_img_cap=util.pytorch_cos_sim(embedding_images, embedding_captions).cpu()

    relative_brain_image_clip_L=clip_L_sim_brain_img.diag().mean()/clip_L_sim_img_cap.diag().mean()

    print(f"[GROUND] CLIP Similarity GIT from images vs captions: {clip_L_sim_img_cap.diag().mean()}")
    print(f"[ABSOLUTE] CLIP Similarity GIT from brain vs captions: {clip_L_sim_brain_cap.diag().mean()}")
    print(f"[ABSOLUTE] CLIP Similarity GIT from brain vs images: {clip_L_sim_brain_img.diag().mean()}")
    print(f"[RELATIVE] CLIP Similarity   {relative_brain_image_clip_L.mean()}")
    return clip_L_sim_img_cap.diag().mean(),clip_L_sim_brain_cap.diag().mean(),clip_L_sim_brain_img.diag().mean(),relative_brain_image_clip_L.mean()

def add_to_table_cap(metrics,model_name):
    import os
    import pandas as pd

    # Caption metrics dictionary
    caption_metrics = { 
        "model_name": model_name,
        "Rouge1_git_coco": metrics['rouge1_img_ref'],
        "Rouge1_pred_coco": metrics['rouge1_brain_ref'],
        "Rouge1_pred_git": metrics['rouge1_brain_img'],
        "Rouge1_relative": metrics['rouge1_relative'],
        "RougeL_git_coco": metrics['rougeL_img_ref'],
        "RougeL_pred_coco": metrics['rougeL_brain_ref'],
        "RougeL_pred_git": metrics['rougeL_brain_img'],
        "RougeL_relative": metrics['rougeL_relative'],
        "Meteor_git_coco": metrics['meteor_img_ref'],
        "Meteor_pred_coco": metrics['meteor_brain_ref'],
        "Meteor_pred_git": metrics['meteor_brain_img'],
        "Meteor_relative": metrics['relative_brain_image_meteor'],
        "Sentence_git_coco": metrics['sentence_img_ref'],
        "Sentence_pred_coco": metrics['sentence_brain_ref'],
        "Sentence_pred_git": metrics['sentence_brain_img'],
        "Sentence_relative": metrics['sentence_relative'],
        "CLIP-B_git_coco": metrics['clip_B_img_ref'],
        "CLIP-B_pred_coco": metrics['clip_B_brain_ref'],
        "CLIP-B_pred_git": metrics['clip_B_brain_img'],
        "CLIP-B_relative": metrics['clip_B_relative'],
        "CLIP-L_git_coco": metrics['clip_L_img_ref'],
        "CLIP-L_pred_coco": metrics['clip_L_brain_ref'],
        "CLIP-L_pred_git": metrics['clip_L_brain_img'],
        "CLIP-L_relative": metrics['clip_L_relative'],
    }

    # Create DataFrame where each key is a column, the values are the dictionary values for the model
    df = pd.DataFrame([caption_metrics])

    # If the file exists, append the new row
    if os.path.exists(CAPTION_TABLE_DIR):
        df_existing = pd.read_csv(CAPTION_TABLE_DIR)
        df_existing = pd.concat([df_existing, df], ignore_index=True)
        df_existing.to_csv(CAPTION_TABLE_DIR, index=False)
    else:
        # If the file doesn't exist, save the DataFrame as a new CSV with comma separation
        df.to_csv(CAPTION_TABLE_DIR, index=False)
        
        
        
def _prepare_images(images):
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)

    images = images.float()

    if images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2)

    if images.max() > 1.0:
        images = images / 255.0

    images = transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR)(images)

    return images

def compute_metrics(all_recons,all_images):
    all_recons = _prepare_images(all_recons)
    all_images = _prepare_images(all_images)

    metrics = {}
   # metrics['lpips_alex'] = compute_lpips(all_recons,all_images,'alex')
    #metrics['top1_nway'],metrics['top1_nway_std'] = compute_clip_nway_id(all_recons,all_images,num_experiments=1,n=1000)
    metrics['pixcorr'] = compute_pixcorr(all_recons,all_images)
    metrics['ssim'] = compute_ssim(all_recons,all_images)
    metrics['color_ssim'] = compute_color_ssim(all_recons,all_images)
    metrics['alexnet2'],metrics['alexnet5'] = compute_alexnet_2way_id(all_recons,all_images)
    metrics['inceptionv3'] = compute_inceptionv3_2way_id(all_recons,all_images)
    metrics['clip_'] = compute_clip_2way_id(all_recons,all_images)
    metrics['effnet'] = compute_efficientNet_dist(all_recons,all_images)
    metrics['swav'] = compute_swav_dist(all_recons,all_images)
    return metrics

