import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from SpatialRelationEncoder import SphereMixScaleSpatialRelationEncoder
from load_pickled_embeddings import load_st_audio_data, generate_text_descriptions
from typing import Dict, List
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from early_fusion_classifier_head import *
from clasp import SpatiotemporalEncoder
import librosa
import json
import pandas as pd
from pathlib import Path

# Paths (you will need to set these yourself)
BIRDSET_SAVE_DIR = "/path/to/pickled_birdset_embeds/"
WABAD_SAVE_DIR = "/path/to/pickled_wabad_embeds/"
CLASS_LABELS_FILEPATH = "metadata/perch_v2_label_mapping.json"
PREDICTION_SAVE_DIR = "path/to/save/dir"

def load_wabad_embeddings(subset_name, wabad_dir=WABAD_SAVE_DIR):
    """Load WABAD embeddings for a specific subset"""
    wabad_path = os.path.join(wabad_dir, f"{subset_name}_wabad.pkl")
    
    if not os.path.exists(wabad_path):
        print(f"Warning: No embeddings found for {subset_name} at {wabad_path}")
        return None
    
    with open(wabad_path, "rb") as f:
        wabad_data = pickle.load(f)
    
    return wabad_data

def compute_accuracy_per_subset(subset_name, base_classifier, geo_aware_classifier, 
                                st_enc, perch_label_mapping, torch_device, 
                                num_experts, num_epochs, no_st_epochs):
    """Compute accuracy for a single WABAD subset"""
    
    print(f"\n{'='*60}")
    print(f"Processing {subset_name} subset...")
    print(f"{'='*60}")
    
    # Load WABAD embeddings
    wabad_data = load_wabad_embeddings(subset_name)
    if wabad_data is None:
        return None, None, None
    
    # Extract data
    embeddings = wabad_data['embeddings']
    st_context = wabad_data['st_context']
    labels = wabad_data['labels']  # Multi-hot encoded labels
    
    print(f"Loaded {len(embeddings)} samples")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Spatiotemporal context shape: {st_context.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Create mapping from perch label index to ebird code
    idx_to_ebird = {int(k): v for k, v in perch_label_mapping.items()}
    
    # Initialize results
    predicted_ebird_codes = []
    geo_aware_predicted_ebird_codes = []
    ground_truth_labels_list = []  # List of label indices for each sample
    
    # For tracking accuracy
    base_correct = 0
    geo_correct = 0
    total_samples = 0
    
    base_classifier.eval()
    geo_aware_classifier.eval()
    
    # Get label statistics
    label_counts = labels.sum(axis=1)
    single_label_indices = np.where(label_counts == 1)[0]
    multi_label_indices = np.where(label_counts > 1)[0]
    zero_label_indices = np.where(label_counts == 0)[0]
    
    print(f"\nLabel statistics:")
    print(f"  Single-label samples: {len(single_label_indices)}")
    print(f"  Multi-label samples: {len(multi_label_indices)}")
    print(f"  Zero-label samples: {len(zero_label_indices)}")
    
    with torch.no_grad():
        for i in tqdm(range(len(embeddings)), desc=f"Processing {subset_name}"):
            audio_embed = embeddings[i]
            st_context_i = st_context[i]
            
            # Get ground truth labels (list of indices where label=1)
            ground_truth_indices = np.where(labels[i] == 1)[0]
            
            # Skip samples with no labels
            if len(ground_truth_indices) == 0:
                continue
            
            # Prepare audio tensor (float32)
            audio_tensor = torch.from_numpy(audio_embed).float().to(torch_device).unsqueeze(0)
            
            # Prepare spatiotemporal context
            # Ensure it's numpy array
            if isinstance(st_context_i, np.ndarray):
                st_context_np = st_context_i
            else:
                st_context_np = np.array(st_context_i)
            
            # Convert to float32 for the encoder
            st_context_np = st_context_np.astype(np.float32)
            
            # Reshape for encoder: (batch_size, seq_len, input_dim)
            st_context_np_batch = st_context_np.reshape(1, 1, -1)
            
            # Encode spatiotemporal context (encoder expects numpy)
            st_encodings = st_enc(st_context_np_batch)
            
            # Convert encodings to float32 tensor
            if isinstance(st_encodings, np.ndarray):
                st_encodings_tensor = torch.from_numpy(st_encodings).float().to(torch_device)
            else:
                st_encodings_tensor = st_encodings.float().to(torch_device)
            
            # Remove the sequence dimension if needed
            if len(st_encodings_tensor.shape) > 2:
                st_encodings_tensor = st_encodings_tensor.squeeze(1)
            
            # Base classifier prediction
            base_logits = base_classifier(audio_tensor, st_encodings_tensor)
            if isinstance(base_logits, tuple):
                base_logits = base_logits[0]
            base_pred_idx = torch.argmax(base_logits, dim=1).item()
            
            # Geo-aware classifier prediction
            geo_logits = geo_aware_classifier(audio_tensor, st_encodings_tensor)
            if isinstance(geo_logits, tuple):
                geo_logits = geo_logits[0]
            geo_pred_idx = torch.argmax(geo_logits, dim=1).item()
            
            # Store predictions
            predicted_ebird_codes.append(idx_to_ebird.get(base_pred_idx, "UNKNOWN"))
            geo_aware_predicted_ebird_codes.append(idx_to_ebird.get(geo_pred_idx, "UNKNOWN"))
            
            # Store ground truth labels
            ground_truth_codes = [idx_to_ebird.get(idx, "UNKNOWN") for idx in ground_truth_indices]
            ground_truth_labels_list.append(ground_truth_codes)
            
            # Calculate accuracy: correct if top-1 prediction matches ANY ground truth label
            if base_pred_idx in ground_truth_indices:
                base_correct += 1
            
            if geo_pred_idx in ground_truth_indices:
                geo_correct += 1
            
            total_samples += 1
    
    # Compute accuracies
    base_accuracy = (base_correct / total_samples * 100) if total_samples > 0 else 0
    geo_accuracy = (geo_correct / total_samples * 100) if total_samples > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Results for {subset_name}:")
    print(f"{'='*60}")
    print(f"Total valid samples: {total_samples}")
    print(f"\nTop-1 Accuracy (prediction matches any ground truth label):")
    print(f"  Base Classifier: {base_accuracy:.2f}% ({base_correct}/{total_samples})")
    print(f"  Geo-aware Classifier: {geo_accuracy:.2f}% ({geo_correct}/{total_samples})")
    print(f"\nImprovement with spatiotemporal context: {geo_accuracy - base_accuracy:.2f}%")
    
    # Print some example predictions
    print(f"\nSample predictions (first 10):")
    for i in range(min(10, len(predicted_ebird_codes))):
        print(f"  Sample {i}:")
        print(f"    GT: {ground_truth_labels_list[i]}")
        print(f"    Base: {predicted_ebird_codes[i]}")
        print(f"    Geo: {geo_aware_predicted_ebird_codes[i]}")
        print(f"    Correct: Base={'Yes' if predicted_ebird_codes[i] in ground_truth_labels_list[i] else 'No'}, "
              f"Geo={'Yes' if geo_aware_predicted_ebird_codes[i] in ground_truth_labels_list[i] else 'No'}")
    
    return {
        'subset': subset_name,
        'total_samples': total_samples,
        'base_accuracy': base_accuracy,
        'geo_accuracy': geo_accuracy,
        'base_correct': base_correct,
        'geo_correct': geo_correct,
        'predictions': predicted_ebird_codes,
        'geo_predictions': geo_aware_predicted_ebird_codes,
        'ground_truth': ground_truth_labels_list
    }
def evaluate_all_wabad_subsets(base_classifier_path, geo_aware_classifier_path, 
                               num_experts, num_epochs, no_st_epochs, torch_device="cuda"):
    """Evaluate all WABAD subsets"""
    
    # Load models
    print("Loading models...")
    base_classifier = BalancedMoE_ST_EmbeddingClassifierHead(st_embedding_dim=0, st_hidden_dim=0, num_experts=num_experts)
    geo_aware_classifier = BalancedMoE_ST_EmbeddingClassifierHead(st_embedding_dim=165, st_hidden_dim=512, num_experts=num_experts)
    
    base_classifier_dict = torch.load(base_classifier_path)
    geo_aware_classifier_dict = torch.load(geo_aware_classifier_path)
    
    base_classifier.load_state_dict(base_classifier_dict)
    geo_aware_classifier.load_state_dict(geo_aware_classifier_dict)
    
    base_classifier = base_classifier.to(torch_device)
    geo_aware_classifier = geo_aware_classifier.to(torch_device)
    
    # Load class labels
    with open(CLASS_LABELS_FILEPATH) as f:
        perch_label_mapping = json.load(f)
    
    # Initialize spatiotemporal encoder
    location_encoder = SphereMixScaleSpatialRelationEncoder(160, frequency_num=32)
    st_enc = SpatiotemporalEncoder(location_encoder, loc_dim=2, device=torch_device, temporal_encoding_dim=5)
    
    # Find all WABAD subsets
    wabad_files = list(Path(WABAD_SAVE_DIR).glob("*_wabad.pkl"))
    subsets = [f.stem.replace("_wabad", "") for f in wabad_files]
    
    print(f"\nFound {len(subsets)} WABAD subsets: {subsets}")
    
    # Evaluate each subset
    all_results = {}
    for subset in subsets:
        result = compute_accuracy_per_subset(
            subset, base_classifier, geo_aware_classifier, st_enc, 
            perch_label_mapping, torch_device, num_experts, num_epochs, no_st_epochs
        )
        if result:
            all_results[subset] = result
            
            # Save predictions for this subset
            os.makedirs(PREDICTION_SAVE_DIR, exist_ok=True)
            pred_path = os.path.join(PREDICTION_SAVE_DIR, f"wabad_{subset}_predictions.pkl")
            with open(pred_path, "wb") as fw:
                pickle.dump({
                    "perch": result['predictions'],
                    "geo_aware": result['geo_predictions'],
                    "ground_truth": result['ground_truth']
                }, fw)
            print(f"\nSaved predictions to {pred_path}")
    
    # Print summary of all subsets
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL WABAD SUBSETS")
    print(f"{'='*80}")
    print(f"{'Subset':<15} {'Samples':<10} {'Base Acc':<12} {'Geo Acc':<12} {'Improvement':<12}")
    print(f"{'-'*80}")
    
    total_samples_all = 0
    total_base_correct = 0
    total_geo_correct = 0
    
    for subset, result in all_results.items():
        print(f"{subset:<15} {result['total_samples']:<10} "
              f"{result['base_accuracy']:>6.2f}%     {result['geo_accuracy']:>6.2f}%     "
              f"{result['geo_accuracy'] - result['base_accuracy']:>+6.2f}%")
        
        total_samples_all += result['total_samples']
        total_base_correct += result['base_correct']
        total_geo_correct += result['geo_correct']
    
        # Calculate unweighted averages (each subset contributes equally)
    base_accuracies = [result['base_accuracy'] for result in all_results.values()]
    geo_accuracies = [result['geo_accuracy'] for result in all_results.values()]
    
    unweighted_base_acc = np.mean(base_accuracies)
    unweighted_geo_acc = np.mean(geo_accuracies)
    unweighted_improvement = unweighted_geo_acc - unweighted_base_acc
    
    # Also calculate standard deviations to show variability
    base_std = np.std(base_accuracies)
    geo_std = np.std(geo_accuracies)
    
    print(f"{'-'*80}")
    print(f"{'UNWEIGHTED AVG':<15} {'-':<10} "
          f"{unweighted_base_acc:>6.2f}% +/- {base_std:.2f}     {unweighted_geo_acc:>6.2f}% +/- {geo_std:.2f}     "
          f"{unweighted_improvement:>+6.2f}%")
    
    # Also show weighted averages for reference
    overall_base_acc = (total_base_correct / total_samples_all * 100) if total_samples_all > 0 else 0
    overall_geo_acc = (total_geo_correct / total_samples_all * 100) if total_samples_all > 0 else 0
    
    print(f"{'WEIGHTED AVG':<15} {total_samples_all:<10} "
          f"{overall_base_acc:>6.2f}%     {overall_geo_acc:>6.2f}%     "
          f"{overall_geo_acc - overall_base_acc:>+6.2f}%")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS:")
    print("="*80)
    print(f"Total subsets evaluated: {len(all_results)}")
    print(f"Total samples across all subsets: {total_samples_all}")
    print(f"\nUnweighted Averages (each subset contributes equally):")
    print(f"  Base Classifier: {unweighted_base_acc:.2f}% +/- {base_std:.2f}")
    print(f"  Geo-aware Classifier: {unweighted_geo_acc:.2f}% +/- {geo_std:.2f}")
    print(f"  Improvement: {unweighted_improvement:.2f}%")
    print(f"\nWeighted Averages (sample-weighted):")
    print(f"  Base Classifier: {overall_base_acc:.2f}%")
    print(f"  Geo-aware Classifier: {overall_geo_acc:.2f}%")
    print(f"  Improvement: {overall_geo_acc - overall_base_acc:.2f}%")
    
    # Add unweighted metrics to the results dictionary if you want to save them
    all_results['summary'] = {
        'unweighted_base_accuracy': unweighted_base_acc,
        'unweighted_geo_accuracy': unweighted_geo_acc,
        'unweighted_improvement': unweighted_improvement,
        'base_std': base_std,
        'geo_std': geo_std,
        'weighted_base_accuracy': overall_base_acc,
        'weighted_geo_accuracy': overall_geo_acc,
        'weighted_improvement': overall_geo_acc - overall_base_acc,
        'total_subsets': len(all_results),
        'total_samples': total_samples_all
    }
    
    return all_results

if __name__ == '__main__':
    # Configuration
    num_experts = 6
    num_epochs = 1  # Geo-aware epochs
    no_st_epochs = 3  # Base epochs
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model paths
    base_classifier_path = f"/scratch/e1583377/models/geospatial_mixup_ce_{num_experts}_experts_epoch_{no_st_epochs}_no_st.pth"
    geo_aware_classifier_path = f"/scratch/e1583377/models/geospatial_mixup_ce_{num_experts}_experts_epoch_{num_epochs}.pth"
    
    # Evaluate all WABAD subsets
    results = evaluate_all_wabad_subsets(
        base_classifier_path, 
        geo_aware_classifier_path,
        num_experts,
        num_epochs,
        no_st_epochs,
        torch_device
    )
    
    print("\nEvaluation complete!")