import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats
import warnings
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pickle
from SpatialRelationEncoder import SphereMixScaleSpatialRelationEncoder
from load_pickled_embeddings import load_st_audio_data, generate_text_descriptions
from typing import Dict, List
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from early_fusion_classifier_head import *
from spatiotemporal_encoder import SpatiotemporalEncoder
import librosa
import json
from birdset_preparation import load_birdset_data
SAVE_DIR = "/scratch/e1583377/pickled_birdset_embeds/"

def compute_auroc(model, dataloader, subset_labels, device='cuda', num_classes=None):
    """
    Compute AUROC for multi-class classification.
    
    Args:
        model: Your classifier
        dataloader: Test/validation DataLoader
        device: 'cuda' or 'cpu'
        num_classes: Number of classes (inferred if None)
    
    Returns:
        dict: AUROC metrics
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for audio_data, st_data, labels in dataloader:
            audio_data, st_data, labels = audio_data.to(device), st_data.to(device), labels.to(device)
            
            outputs = model(audio_data, st_data)
            if isinstance(outputs, tuple):
                logits, _ = outputs
            else:
                logits = outputs
            print(logits.shape)
            
            probs = torch.sigmoid(logits)[:,subset_labels]
            
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate all batches
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    if num_classes is None:
        num_classes = all_probs.shape[1]
    
    # Compute AUROC for each class
    auroc_scores = {}
    
    if num_classes == 2:  # Binary classification
        try:
            auroc = roc_auc_score(all_labels, all_probs[:, 1])
            auroc_scores['binary'] = auroc
        except:
            auroc_scores['binary'] = np.nan
    else:  # Multi-class
        # One-vs-Rest AUROC for each class
        for class_idx in range(num_classes):
            try:
                # Create binary labels for this class
                binary_labels = (all_labels == class_idx).astype(int)
                
                # Skip if class has no positive examples
                if np.sum(binary_labels) == 0:
                    auroc_scores[f'class_{class_idx}'] = np.nan
                    continue
                
                # AUROC for this class vs all others
                class_auroc = roc_auc_score(binary_labels, all_probs[:, class_idx])
                auroc_scores[f'class_{class_idx}'] = class_auroc
            except:
                auroc_scores[f'class_{class_idx}'] = np.nan
    
    # Compute macro-average AUROC (ignore NaN)
    valid_scores = [v for v in auroc_scores.values() if not np.isnan(v)]
    if valid_scores:
        auroc_scores['macro_avg'] = np.mean(valid_scores)
    else:
        auroc_scores['macro_avg'] = np.nan
    
    return auroc_scores, all_probs, all_labels

def compute_cmap(model, dataloader, subset_labels, device='cuda', num_classes=None):
    """
    Compute Class-weighted Mean Average Precision (cmAP).
    
    cmAP = mean of per-class AP scores, weighted by class support
    
    Args:
        model: Your classifier
        dataloader: Test/validation DataLoader
        device: 'cuda' or 'cpu'
        num_classes: Number of classes
    
    Returns:
        dict: cmAP metrics
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for audio_data, st_data, labels in dataloader:
            audio_data, st_data, labels = audio_data.to(device), st_data.to(device), labels.to(device)
            
            outputs = model(audio_data, st_data)
            if isinstance(outputs, tuple):
                logits, _ = outputs
            else:
                logits = outputs
            
            probs = torch.sigmoid(logits)[:,subset_labels]
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    if num_classes is None:
        num_classes = all_probs.shape[1]
    
    # Compute Average Precision for each class
    ap_scores = {}
    class_supports = []
    
    for class_idx in range(num_classes):
        try:
            # Binary labels for this class
            binary_labels = (all_labels == class_idx).astype(int)
            class_support = np.sum(binary_labels)
            class_supports.append(class_support)
            
            # Skip if no positive examples
            if class_support == 0:
                ap_scores[f'class_{class_idx}'] = np.nan
                continue
            
            # Average Precision for this class
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ap = average_precision_score(binary_labels, all_probs[:, class_idx])
            
            ap_scores[f'class_{class_idx}'] = ap
        except:
            ap_scores[f'class_{class_idx}'] = np.nan
            class_supports.append(0)
    
    # Compute cmAP (weighted by class support)
    valid_aps = []
    valid_weights = []
    
    for class_idx in range(num_classes):
        ap = ap_scores.get(f'class_{class_idx}', np.nan)
        if not np.isnan(ap):
            valid_aps.append(ap)
            valid_weights.append(class_supports[class_idx])
    
    if valid_aps:
        # Weighted average
        cmap = np.average(valid_aps, weights=valid_weights)
        # Macro average (unweighted)
        macro_map = np.mean(valid_aps)
    else:
        cmap = np.nan
        macro_map = np.nan
    
    return {
        'cmap': cmap,
        'macro_map': macro_map,
        'per_class_ap': ap_scores,
        'class_supports': class_supports
    }
    

def compute_multilabel_metrics(model, dataloader, subset_labels, device='cuda'):
    """
    Compute AUROC and cmAP for MULTI-LABEL classification.
    
    Args:
        model: Your classifier
        dataloader: Test/validation DataLoader
        subset_labels: List of pretraining class indices to evaluate
        device: 'cuda' or 'cpu'
    
    Returns:
        dict: AUROC and cmAP metrics
    """
    model.eval()
    model.to(device)
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for audio_data, st_data, labels in dataloader:
            audio_data = audio_data.to(device)
            st_data = st_data.to(device)
            
            # Forward pass
            outputs = model(audio_data, st_data)
            if isinstance(outputs, tuple):
                logits, _ = outputs
            else:
                logits = outputs
            
            all_probs_batch = torch.sigmoid(logits)
            
            # Extract probabilities for evaluation classes
            probs_subset = all_probs_batch[:, subset_labels]  # [batch_size, num_task_classes]
            
            all_probs.append(probs_subset.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate
    all_probs = np.concatenate(all_probs, axis=0)  # [n_samples, num_task_classes]
    all_labels = np.concatenate(all_labels, axis=0)  # [n_samples, num_task_classes]
    
    num_classes = all_probs.shape[1]
    
    # Initialize results
    auroc_scores = []
    ap_scores = []  # Average Precision scores
    class_info = []
    
    # For each class (column), compute metrics
    for class_idx in range(num_classes):
        y_true = all_labels[:, class_idx]  # Binary labels for this class
        y_score = all_probs[:, class_idx]   # Probabilities for this class
        
        # Check if class has any positive samples
        n_positives = np.sum(y_true)
        
        if n_positives == 0:
            # No positive samples for this class
            print(f"Class {class_idx} (pretrain idx {subset_labels[class_idx]}): "
                  f"No positive samples, skipping")
            auroc_scores.append(np.nan)
            ap_scores.append(np.nan)
            continue
        
        # Check if probabilities are valid (not all NaN or constant)
        if np.all(np.isnan(y_score)) or len(np.unique(y_score)) <= 1:
            print(f"Class {class_idx}: Invalid predictions "
                  f"(all NaN or constant: {len(np.unique(y_score))} unique values)")
            auroc_scores.append(np.nan)
            ap_scores.append(np.nan)
            continue
        
        try:
            # AUROC
            auroc = roc_auc_score(y_true, y_score)
            auroc_scores.append(auroc)
            
            # Average Precision (AP)
            ap = average_precision_score(y_true, y_score)
            ap_scores.append(ap)
            
            class_info.append({
                'class_idx': class_idx,
                'pretrain_idx': subset_labels[class_idx],
                'n_positives': n_positives,
                'auroc': auroc,
                'ap': ap
            })
            
            print(f"Class {class_idx} (pretrain {subset_labels[class_idx]}): "
                  f"AUROC={auroc:.3f}, AP={ap:.3f}, "
                  f"positives={n_positives}")
            
        except Exception as e:
            print(f"Class {class_idx}: Error computing metrics: {e}")
            auroc_scores.append(np.nan)
            ap_scores.append(np.nan)
    
    # Compute macro averages (ignore NaN)
    valid_auroc = [s for s in auroc_scores if not np.isnan(s)]
    valid_ap = [s for s in ap_scores if not np.isnan(s)]
    
    if valid_auroc:
        macro_auroc = np.mean(valid_auroc)
        macro_ap = np.mean(valid_ap)  # This is cmAP
    else:
        macro_auroc = np.nan
        macro_ap = np.nan
    
    # Also compute micro averages if needed
    try:
        # Flatten all predictions and labels
        micro_auroc = roc_auc_score(all_labels.ravel(), all_probs.ravel())
        micro_ap = average_precision_score(all_labels.ravel(), all_probs.ravel())
    except:
        micro_auroc = np.nan
        micro_ap = np.nan
    
    results = {
        'macro_auroc': macro_auroc,
        'cmap': macro_ap,  # Class-wise mean Average Precision
        'micro_auroc': micro_auroc,
        'micro_map': micro_ap,
        'auroc_per_class': auroc_scores,
        'ap_per_class': ap_scores,
        'class_info': class_info,
        'n_valid_classes': len(valid_auroc),
        'all_probs': all_probs,
        'all_labels': all_labels
    }
    
    return results


class BirdsetDataset(Dataset):
    def __init__(self, data_dict):
        # Store as numpy arrays, not tensors
        self.embeddings = data_dict["embeddings"].astype(np.float32)
        self.st_context = data_dict["st_context"].astype(np.float32)
        self.labels = data_dict["labels"].astype(np.float32)
        print(self.embeddings.shape)
        print(self.st_context.shape)
        print(self.labels.shape)
        
        print(f"Dataset using numpy arrays")
    def __len__(self):
        """Return the number of samples in the dataset"""
        # Return the length of any of the data arrays
        # They should all have the same length
        return len(self.labels)  # Assuming self.embeddings is a list/array
    
    def __getitem__(self, idx):
        # Convert to tensor only when accessed
        audio = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        st = torch.tensor(self.st_context[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return audio, st, label
# After flattening your batches
def create_multi_hot_labels(label_lists, num_classes):
    """
    Convert list-of-lists to multi-hot binary matrix
    
    Args:
        label_lists: List of lists, e.g., [[], [3], [1, 7], [2], []]
        num_classes: Total number of species
        
    Returns:
        labels_multi_hot: Binary matrix of shape [n_samples, num_classes]
    """
    n_samples = len(label_lists)
    labels_multi_hot = np.zeros((n_samples, num_classes), dtype=np.int32)
    
    for i, label_list in enumerate(label_lists):
        if label_list:  # If list is not empty
            labels_multi_hot[i, label_list] = 1
    
    return labels_multi_hot

# Complete data preparation function
def prepare_data_for_evaluation(data_dict, num_classes, st_enc):
    """
    Prepare data for multi-label evaluation
    
    Returns:
        audio_data: [n_samples, 1536]
        st_data: [n_samples, 512]  
        labels_multi_hot: [n_samples, num_classes] binary matrix
        labels_list: Original list format (for debugging)
    """
    # Flatten features
    audio_flat = np.concatenate(data_dict["embeddings"], axis=0)
    st_flat = np.concatenate(data_dict["st_context"], axis=0)
    st_encodings = st_enc(np.expand_dims(st_flat.astype(np.float32), axis=1))
    st_encodings = st_encodings.squeeze(1).cpu().numpy()
    # Flatten labels (list of lists)
    all_label_lists = []  # Will be list of lists
    for batch in data_dict["labels"]:
        # batch is a list of lists, e.g., [[3, 7], [], [1], [2, 5, 8], ...]
        all_label_lists.extend(batch)  # Keep each sample's list intact!
    
    # Create multi-hot binary matrix
    labels_multi_hot = create_multi_hot_labels(all_label_lists, num_classes)
    # Add this check

    # They should all have the same first dimension
    if audio_flat.shape[0] != labels_multi_hot.shape[0]:
        print(f"\n??  WARNING: Dimension mismatch!")
        print(f"   Audio samples: {audio_flat.shape[0]}")
        print(f"   Label samples: {labels_multi_hot.shape[0]}")
    
        # Try to understand what happened
        total_labels_from_batches = sum(len(batch) for batch in data_dict["labels"])
        print(f"   Total labels from batches: {total_labels_from_batches}")
        all_label_lists = data_dict["labels"]
        labels_multi_hot = create_multi_hot_labels(all_label_lists, num_classes)
    
    return {
        "embeddings":audio_flat, 
        "st_context":st_encodings, 
        "labels":labels_multi_hot
    }
        


def simple_collate_fn(batch):
    """
    Simple collate function for (audio, st, labels) tuples
    Each element in batch is (audio_tensor, st_tensor, label_tensor)
    """
    # batch is a list of tuples: [(audio1, st1, label1), (audio2, st2, label2), ...]
    
    # Separate the components
    #print("calling_collate_fn")
    audios = [item[0] for item in batch]
    sts = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    
    # Stack them
    audio_batch = torch.stack(audios, dim=0)
    st_batch = torch.stack(sts, dim=0)
    label_batch = torch.stack(labels, dim=0)
    
    return audio_batch, st_batch, label_batch

if __name__ == '__main__':    
    subsets = ["HSN", "SNE", "PER", "UHH", "NES", "SSW"]
    num_epochs = 3
    no_st_epochs = 4
    num_experts = 4
    torch_device = "cuda"
    for subset in subsets:
        prediction_save_path = "/home/svu/e1583377/ST-Geo-Perch/predictions/mixup_balanced_mixture_of_" + str(num_experts) + "_experts_early_fusion_classifier_lr_0006_epoch_" + str(num_epochs - 1) + "_" + subset + ".pth"
        base_classifier = BalancedMoE_ST_EmbeddingClassifierHead(st_embedding_dim = 0, st_hidden_dim = 0, num_experts = num_experts)
        geo_aware_classifier = BalancedMoE_ST_EmbeddingClassifierHead(st_embedding_dim = 165, st_hidden_dim = 512, num_experts = num_experts)
        
        base_classifier_dict = torch.load("/scratch/e1583377/models/mixup_ce_" + str(num_experts) + "_experts_epoch_" + str(no_st_epochs) + "_no_st.pth")
        geo_aware_classifier_dict = torch.load("/scratch/e1583377/models/mixup_ce_" + str(num_experts) + "_experts_epoch_" + str(num_epochs) + ".pth")
        base_classifier.load_state_dict(base_classifier_dict)
        geo_aware_classifier.load_state_dict(geo_aware_classifier_dict)
        base_classifier = base_classifier.to(torch_device)
        geo_aware_classifier = geo_aware_classifier.to(torch_device)
        CLASS_LABELS_FILEPATH = "/home/svu/e1583377/Spatial_Perch_Transfer_Learning/assets/perch_v2_label_mapping.json"
        with open(CLASS_LABELS_FILEPATH) as f:
            perch_label_mapping = json.load(f)
        location_encoder = SphereMixScaleSpatialRelationEncoder(160, frequency_num = 32)
        st_enc = SpatiotemporalEncoder(location_encoder, loc_dim = 2, device = torch_device, temporal_encoding_dim = 5)
        print(list(perch_label_mapping.keys())[:10])
        predicted_ebird_codes = []
        geo_aware_predicted_ebird_codes = []
        ground_truth = []
        base_classifier.eval()
        geo_aware_classifier.eval()
        with open(SAVE_DIR + subset + ".pkl", "rb") as f:
            birdset_data = pickle.load(f)
            
        birdset_test = load_birdset_data(subset)
        subset_label_mapping = birdset_test.features['ebird_code']._int2str  # or whatever the feature name is
        dataset_ebird_codes = [subset_label_mapping[i] for i in range(len(subset_label_mapping))]
        print(dataset_ebird_codes[:10])
    
        # Your model's eBird codes (in the order of model outputs)
        CLASS_LABELS_FILEPATH = "/home/svu/e1583377/Spatial_Perch_Transfer_Learning/assets/perch_v2_label_mapping.json"
        with open(CLASS_LABELS_FILEPATH) as f:
            perch_label_mapping = json.load(f)
        model_ebird_codes = [perch_label_mapping[str(i)] for i in range(len(perch_label_mapping))]
        CLASS_LABELS_FILEPATH = "/home/svu/e1583377/Spatial_Perch_Transfer_Learning/assets/perch_v2_label_mapping.json"
        with open(CLASS_LABELS_FILEPATH) as f:
            perch_label_mapping = json.load(f)
        # Create mapping from dataset eBird codes to model indices
        # map birdset labels to perch labels
        dataset_to_model_indices = []
        dataset_label_mapping = {}
        for idx, code in enumerate(dataset_ebird_codes):
            if code in model_ebird_codes:
                label = model_ebird_codes.index(code)
                dataset_to_model_indices.append(label)
                dataset_label_mapping[str(idx)] = code
            else:
                # Handle case where dataset code isn't in model (if possible)
                print(f"Warning: {code} not found in model outputs")
        print(dataset_to_model_indices)
        prepared_data = prepare_data_for_evaluation(birdset_data, len(dataset_to_model_indices), st_enc)
        
        birdset_dataset = BirdsetDataset(prepared_data)
        print(len(birdset_dataset))
        birdset_dataloader = DataLoader(
            birdset_dataset, 
            batch_size=1024,
            shuffle=False,
            collate_fn=simple_collate_fn,
            num_workers=0
        )
        audio_only_results = compute_multilabel_metrics(base_classifier, birdset_dataloader, subset_labels = dataset_to_model_indices)
        print("Audio-only scores:", audio_only_results)
        st_audio_results = compute_multilabel_metrics(geo_aware_classifier, birdset_dataloader, subset_labels = dataset_to_model_indices)
        print("Spatiotemporally-aware scores:", st_audio_results)
        print("Audio-only macro scores: AUROC", audio_only_results['macro_auroc'], "cmAP", audio_only_results['cmap'])
        print("Spatiotemporally-aware macro scores: AUROC", st_audio_results['macro_auroc'], "cmAP:", st_audio_results['cmap'])
        print(subset)
        print(num_experts)