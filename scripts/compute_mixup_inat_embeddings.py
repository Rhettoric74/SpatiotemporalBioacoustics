import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
# Remove or comment out these lines:
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# COMPLETELY isolate TensorFlow initialization

import tensorflow as tf
print("=== GPU Status ===")
print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")

# Configure GPU BEFORE any operations
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        # Set memory growth FIRST
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("? TensorFlow GPU configured with memory growth")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
import tensorflow_hub as hub
tf.experimental.numpy.experimental_enable_numpy_behavior()
def load_perch_gpu_model(url = 'https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2/2'):
    return hub.load(url)
with tf.device('/GPU:0'):
    perch_model = load_perch_gpu_model()
print("Warming up PERCH model cuDNN...")
try:
    # Create dummy audio data in the expected format
    # Adjust shape based on your PERCH model's expected input
    waveform = np.zeros(5 * 32000, dtype=np.float32)  # Batch of 1, 16000 samples, 1 channel
    with tf.device('/GPU:0'):
        # Force cuDNN initialization
        _ = perch_model.signatures['serving_default'](inputs=waveform[np.newaxis, :])
    print("? PERCH model cuDNN initialized successfully")
except Exception as e:
    print(f"? PERCH warm-up failed: {e}")
    import traceback
    traceback.print_exc()
import torch


# Test TensorFlow alone first
print("\n=== Testing TensorFlow ===")
try:
    # Simple TF operation
    with tf.device('/GPU:0'):
        a = tf.constant([1.0, 2.0, 3.0])
        b = tf.constant([4.0, 5.0, 6.0])
        c = a + b
    print("? TensorFlow GPU operations work")
except Exception as e:
    print(f"? TensorFlow GPU failed: {e}")

# Test PyTorch alone
print("\n=== Testing PyTorch ===")
try:
    if torch.cuda.is_available():
        a = torch.tensor([1.0, 2.0, 3.0]).cuda()
        b = torch.tensor([4.0, 5.0, 6.0]).cuda()
        c = a + b
        print("? PyTorch GPU operations work")
    else:
        print("? PyTorch cannot access GPU")
except Exception as e:
    print(f"? PyTorch GPU failed: {e}")

# Configure memory growth to avoid conflicts
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Don't do any explicit GPU configuration at all
# Let TensorFlow use its defaults

# Your model loading and inference code here

# Verify both work

#import torchaudio, torch, subprocess, shutil, sys, os, tempfile
from pydub import AudioSegment
from pydub.utils import which
import subprocess
from pathlib import Path
import warnings
# Force PyDub to use the binaries in ~/bin
AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe   = which("ffprobe")
API_TOKEN = "eyJhbGciOiJIUzUxMiJ9.eyJ1c2VyX2lkIjo5OTY4NzEyLCJleHAiOjE3NjM2NDEyNDF9.LBvFA7xqOqOo-voUM5tTWwtA_HwyxTHqOMnvLGAHNxb2-Noz-3AhIxBKFa-VOTg38EYTl1AQy41-_hqi1WThBg"
headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "User-Agent": "Mozilla/5.0 (compatible; MyDownloader/1.0)"
}
DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MyDownloader/1.0)"
}

from SpatialRelationEncoder import SphereMixScaleSpatialRelationEncoder
from spatiotemporal_encoder import SpatiotemporalEncoder
from geo_aware_perch import *
from datasets import load_dataset, Audio
from torch.utils.data import DataLoader
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
import pickle
import librosa
import io
import requests
from inaturalist_preparation import encode_time, encode_time_fourier, prepare_for_spatiotemporal_encoder
import concurrent.futures
from tqdm import tqdm
import time
import random
import tempfile

import numpy as np
import torch
import pickle
import os
from pathlib import Path
from scipy.stats import betabinom
from scipy.spatial import cKDTree
from torch.distributions import Dirichlet
import random
from tqdm import tqdm
import json
from compute_inat_embeddings import *
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great-circle distance between two points 
    on the Earth (specified in decimal degrees).
    
    Returns distance in kilometers.
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    r = 6371  # Radius of earth in kilometers
    
    return r * c


class PrecomputedDatasetMixup:
    """
    Precompute mixing combinations from entire dataset, then process in batches.
    """
    
    def __init__(self, n=2, alpha=91.3, beta=100.0, omega=1, 
                 mixup_ratio=1.0, seed=42):
        """
        Args:
            mixup_ratio: How many mixed samples to create (multiple of original dataset size)
        """
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.omega = omega
        self.mixup_ratio = mixup_ratio
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    def sample_num_components(self):
        """Sample N from BetaBin(n, alpha, beta) + 1"""
        from scipy.stats import betabinom
        return betabinom.rvs(self.n, self.alpha, self.beta, size=1)[0] + 1
    
    def sample_mixing_weights(self, n_components):
        """Sample w ~ SymDir(N, ?)"""
        concentration = torch.ones(n_components) * self.omega
        dirichlet_dist = Dirichlet(concentration)
        return dirichlet_dist.sample().numpy()
    
    def precompute_mixing_plan(self, total_samples, output_path):
        """
        Precompute which samples to mix and their weights.
    
        Returns mixing plan saved to disk.
        """
        num_mixed_samples = int(total_samples * self.mixup_ratio)
        print(f"Precomputing {num_mixed_samples} mixing combinations from {total_samples} originals")
    
        mixing_plan = []
    
        for mix_idx in tqdm(range(num_mixed_samples), desc="Precomputing mixes"):
            # Sample number of components
            n_components = self.sample_num_components()
        
            # Randomly select components from ENTIRE dataset
            component_indices = np.random.choice(total_samples, n_components, replace=True)
        
            # Sample mixing weights
            weights = self.sample_mixing_weights(n_components)
        
            # Calculate gain normalization factor
            gain_norm = np.sqrt(np.sum(weights ** 2))
        
            # CONVERT NUMPY TYPES TO PYTHON NATIVE TYPES
            mixing_plan.append({
                'mixed_sample_id': int(mix_idx),  # Convert to Python int
                'components': component_indices.astype(int).tolist(),  # Convert int64 to int
                'weights': weights.astype(float).tolist(),  # Convert float64 to float
                'gain_norm_factor': float(gain_norm),  # Convert float64 to float
                'num_components': int(n_components)  # Convert to Python int
            })
    
        # Save mixing plan
        plan_path = Path(output_path) / "inat_mixing_plan.json"
        with open(plan_path, 'w') as f:
            json.dump(mixing_plan, f, indent=2)
    
        print(f"Saved mixing plan to {plan_path}")
        return mixing_plan
        



    def precompute_mixing_plan_geospatial(self, total_samples, spatiotemporal_contexts, output_path, 
                                      k=10, max_distance_km=None):
      """
      Fast geospatial mixing using KD-tree for nearest neighbor queries.
      """
      num_mixed_samples = int(total_samples * self.mixup_ratio)
      print(f"Precomputing {num_mixed_samples} geospatial mixes using KD-tree (k={k}, max_dist={max_distance_km}km)")
      
      # Extract coordinates and convert to radians for haversine
      st_contexts = np.array(spatiotemporal_contexts)
      longitudes = st_contexts[:, 0]
      latitudes = st_contexts[:, 1]
      
      # Convert to radians
      lon_rad = np.radians(longitudes)
      lat_rad = np.radians(latitudes)
      
      # Convert to 3D Cartesian coordinates on unit sphere for KD-tree
      # This approximates great-circle distance with Euclidean distance
      x = np.cos(lat_rad) * np.cos(lon_rad)
      y = np.cos(lat_rad) * np.sin(lon_rad)
      z = np.sin(lat_rad)
      points_3d = np.column_stack([x, y, z])
      
      # Build KD-tree (fast for nearest neighbor queries)
      print("Building KD-tree...")
      from scipy.spatial import cKDTree
      tree = cKDTree(points_3d)
      
      mixing_plan = []
      
      for mix_idx in tqdm(range(num_mixed_samples), desc=f"KD-tree {k}-NN mixes"):
          # Sample number of components (limit to k or less)
          n_components = min(self.sample_num_components(), k)
          
          # Randomly select an anchor point
          anchor_idx = np.random.randint(total_samples)
          
          # Query k+1 nearest neighbors (includes self)
          # Use k+2 to be safe in case we need to exclude self
          query_k = min(k + 2, total_samples)
          distances_euclidean, indices = tree.query(
              points_3d[anchor_idx], 
              k=query_k
          )
          
          # Convert Euclidean distance on unit sphere to km (great-circle distance)
          # Euclidean distance = 2*sin(?/2) where ? is angular distance in radians
          # So ? = 2*arcsin(d/2)
          angular_distances = 2 * np.arcsin(np.clip(distances_euclidean / 2, 0, 1))
          distances_km = 6371 * angular_distances  # Convert to km (Earth radius)
          
          # Find and exclude self from results
          self_idx = np.where(indices == anchor_idx)[0]
          if len(self_idx) > 0:
              # Remove self from results
              mask = np.ones(len(indices), dtype=bool)
              mask[self_idx[0]] = False
              neighbor_indices = indices[mask]
              neighbor_distances = distances_km[mask]
          else:
              # Self not in results (shouldn't happen but handle it)
              neighbor_indices = indices
              neighbor_distances = distances_km
          
          # Take k nearest neighbors
          k_nearest = neighbor_indices[:min(k, len(neighbor_indices))]
          k_distances = neighbor_distances[:min(k, len(neighbor_distances))]
          
          # Filter by maximum distance if specified
          if max_distance_km is not None:
              valid_mask = k_distances <= max_distance_km
              if not valid_mask.any():
                  # No points within max distance, use just the closest one
                  k_nearest = k_nearest[:1]
                  k_distances = k_distances[:1]
              else:
                  k_nearest = k_nearest[valid_mask]
                  k_distances = k_distances[valid_mask]
          
          # Sample n_components from k nearest neighbors
          if n_components <= len(k_nearest):
              
              
              # Sample without replacement
              component_indices = np.random.choice(
                  k_nearest,
                  size=n_components,
                  replace=False
              )
          else:
              # Not enough neighbors, use all we have
              component_indices = k_nearest[:n_components]
              # If still not enough, pad with anchor
              if len(component_indices) < n_components:
                  pad_needed = n_components - len(component_indices)
                  # Add anchor to make up the difference
                  component_indices = np.concatenate([
                      component_indices,
                      np.array([anchor_idx] * pad_needed)
                  ])
          
          # Ensure anchor is included (important for realistic mixes)
          if anchor_idx not in component_indices:
              # Replace a random component with anchor
              replace_idx = np.random.randint(len(component_indices))
              component_indices[replace_idx] = anchor_idx
          
          # Sample mixing weights
          weights = self.sample_mixing_weights(len(component_indices))
          
          # Calculate gain normalization factor
          gain_norm = np.sqrt(np.sum(weights ** 2))
          
          # Get distances for the selected components
          component_distances = []
          for idx in component_indices:
              if idx == anchor_idx:
                  component_distances.append(0.0)
              else:
                  # Compute actual haversine distance for these specific pairs
                  d = haversine_distance(
                      np.degrees(lon_rad[anchor_idx]), np.degrees(lat_rad[anchor_idx]),
                      np.degrees(lon_rad[idx]), np.degrees(lat_rad[idx])
                  )
                  component_distances.append(d)
          
          mixing_plan.append({
              'mixed_sample_id': int(mix_idx),
              'anchor_idx': int(anchor_idx),
              'anchor_coords': [float(longitudes[anchor_idx]), float(latitudes[anchor_idx])],
              'components': component_indices.astype(int).tolist(),
              'weights': weights.astype(float).tolist(),
              'gain_norm_factor': float(gain_norm),
              'num_components': int(n_components),
              'distances_km': component_distances,
              'avg_distance_km': float(np.mean(component_distances)),
              'max_distance_km': float(np.max(component_distances))
          })
  
      
      plan_path = Path(output_path) / f"geospatial_top_k_mixup_plan.json"
      with open(plan_path, 'w') as f:
          json.dump(mixing_plan, f, indent=2)
      
      print(f"Saved geospatial KD-tree mixing plan to {plan_path}")
      
      # Print statistics
      if mixing_plan:
          avg_distances = [plan['avg_distance_km'] for plan in mixing_plan]
          max_distances = [plan['max_distance_km'] for plan in mixing_plan]
          
          print(f"\nGeospatial Mix Statistics:")
          print(f"  Average mix distance: {np.mean(avg_distances):.1f} km +- {np.std(avg_distances):.1f} km")
          print(f"  Maximum mix distance: {np.mean(max_distances):.1f} km +- {np.std(max_distances):.1f} km")
          print(f"  Min distance in any mix: {np.min(avg_distances):.1f} km")
          print(f"  Max distance in any mix: {np.max(max_distances):.1f} km")
          
          # Distance distribution
          distance_bins = [0, 10, 50, 100, 500, 1000, float('inf')]
          bin_labels = ['<10km', '10-50km', '50-100km', '100-500km', '500-1000km', '>1000km']
          
          for i in range(len(distance_bins)-1):
              lower = distance_bins[i]
              upper = distance_bins[i+1]
              count = sum(1 for d in avg_distances if lower <= d < upper)
              percentage = 100 * count / len(avg_distances)
              print(f"  {bin_labels[i]}: {count} mixes ({percentage:.1f}%)")
      
      return mixing_plan



    
    def load_mixing_plan(self, plan_path):
        """Load precomputed mixing plan from disk."""
        print("Loading mixing plan from", plan_path)
        with open(plan_path, 'r') as f:
            mixing_plan = json.load(f)
        print(len(mixing_plan))
        return mixing_plan

def create_mixing_batches(mixing_plan, original_metadata, batch_size=128):
    """
    Organize mixing plan into processable batches with robust error handling.
    """
    if not mixing_plan:
        print("ERROR: Empty mixing plan!")
        return []
    
    batches = []
    
    for i in range(0, len(mixing_plan), batch_size):
        batch_mixes = mixing_plan[i:i+batch_size]
        
        component_indices_needed = set()
        valid_mixes = []
        
        for mix in batch_mixes:
            # Ensure mix has required keys
            if 'components' not in mix:
                print(f"WARNING: Skipping mix without 'components' key")
                continue
            
            components = mix['components']
            
            # Convert to list if needed
            if isinstance(components, str):
                try:
                    components = json.loads(components)
                except:
                    print(f"WARNING: Could not parse components string: {components}")
                    continue
            
            # Convert all components to integers
            try:
                components = [int(c) for c in components]
                mix['components'] = components  # Update with integers
            except:
                print(f"WARNING: Could not convert components to integers: {components}")
                continue
            
            component_indices_needed.update(components)
            valid_mixes.append(mix)
        
        if not valid_mixes:
            continue  # Skip empty batch
        
        batches.append({
            'batch_id': i // batch_size,
            'mixed_sample_ids': [mix.get('mixed_sample_id', idx) for idx, mix in enumerate(valid_mixes)],
            'component_indices_needed': list(component_indices_needed),
            'mixes': valid_mixes
        })
    
    print(f"Created {len(batches)} mixing batches from {len(mixing_plan)} total mixes")
    return batches


def process_mixing_batch_with_full_context(batch_spec, original_data, geo_aware_perch, device='cuda'):
    """
    Process mixing batch and save ALL spatiotemporal contexts for later aggregation.
    
    Returns data with:
    - audio_embeddings: Embeddings for mixed audio
    - spatiotemporal_embeddings: Embeddings computed with primary context
    - labels_sparse: Mixed labels (list of lists)
    - component_contexts: ALL spatiotemporal contexts for each mix
    - component_weights: Mixing weights for each component
    - component_indices: Original indices of components
    - primary_context: The context used for embedding computation
    """
    component_indices = batch_spec['component_indices_needed']
    
    print(f"Processing mixing batch {batch_spec['batch_id']}")
    print(f"Need {len(component_indices)} original observations for {len(batch_spec['mixes'])} mixes")
    
    # STEP 1: Get occurrence IDs for needed components
    component_occurrence_ids = original_data['occurrence_ids'][component_indices].tolist()
    
    # STEP 2: Fetch audio URLs for these observations
    print(f"Fetching audio URLs for {len(component_occurrence_ids)} observations...")
    component_audio_urls = fetch_inat_audio_batch_safe(component_occurrence_ids)
    
    # Count valid audio
    valid_count = sum(1 for url in component_audio_urls if url is not None)
    
    if valid_count == 0:
        print(f"? No valid audio URLs found for this batch")
        return None
    
    print(f"? Found {valid_count}/{len(component_occurrence_ids)} valid audio URLs")
    
    # Create mapping from original index to audio URL
    index_to_audio_url = {}
    for idx, (original_idx, audio_url) in enumerate(zip(component_indices, component_audio_urls)):
        if audio_url is not None:
            index_to_audio_url[original_idx] = audio_url
    
    # STEP 3: Load audio for valid URLs
    valid_audio_urls = [url for url in component_audio_urls if url is not None]

    # Also track which original indices these correspond to
    valid_component_indices = []
    for i, comp_idx in enumerate(component_indices):
        if component_audio_urls[i] is not None:
            valid_component_indices.append(comp_idx)

    print(f"Loading {len(valid_audio_urls)} audio files...")

    audio_tensor, segment_mapping = load_audio_from_url_parallel(
        valid_audio_urls,
        strategy="random",
        num_segments=1
    )

    if len(audio_tensor) == 0:
        print(f"? Failed to load any audio")
        return None


    # Create mapping from original index to loaded audio USING segment_mapping
    index_to_audio = {}
    for i, audio in enumerate(audio_tensor):
        # segment_mapping[i] gives position in valid_audio_urls
        pos_in_valid_urls = segment_mapping[i]
    
        # Map back to actual component index
        if pos_in_valid_urls < len(valid_component_indices):
            component_idx = valid_component_indices[pos_in_valid_urls]
            index_to_audio[component_idx] = audio
        else:
            print(f"Warning: segment_mapping[{i}] = {pos_in_valid_urls} out of bounds")

    
    # STEP 4: Create mixed samples with FULL context preservation
    mixed_audio_list = []
    primary_st_list = []  # Context used for embedding computation
    all_component_contexts_list = []  # ALL contexts for each mix
    all_component_weights_list = []  # Weights for each component
    all_component_indices_list = []  # Original indices for each component
    mixed_labels_sparse_list = []
    mixed_ids_list = []
    successful_mix_ids = []
    
    print(f"Creating {len(batch_spec['mixes'])} mixed samples with full context...")
    
    for mix in tqdm(batch_spec['mixes'], desc="Creating mixes"):
        components = mix['components']
        weights = np.array(mix['weights'])
        gain_norm = mix['gain_norm_factor']
        
        # Check if we have audio for ALL components
        if any(idx not in index_to_audio for idx in components):
            continue
        
        # Get audio for all components
        component_audio = [index_to_audio[idx] for idx in components]
        
        if not component_audio:
            continue
        
        # Mix audio
        mixed_audio = torch.zeros_like(component_audio[0])
        for j, audio in enumerate(component_audio):
            mixed_audio += weights[j] * audio
        mixed_audio = mixed_audio / gain_norm
        
        
        mixed_audio_list.append(mixed_audio)
        
        # Mix labels - keep all labels in same order as components
        mixed_labels = []
        for idx in components:
            if idx < len(original_data['labels']):
                # Direct extraction from tensor (no deduplication, no sorting)
                mixed_labels.append(original_data['labels'][idx].item())
    
        mixed_labels_sparse_list.append(mixed_labels)  # This matches component_contexts order
        
        # NEW: Save ALL spatiotemporal contexts for this mix
        component_contexts = []
        for idx in components:
            if idx < len(original_data['st_contexts']):
                # Get the context (already 3D: [1, num_features] from earlier expansion)
                context = original_data['st_contexts'][idx]
                component_contexts.append(context)
        
        # Store all component data
        all_component_contexts_list.append(component_contexts)
        all_component_weights_list.append(weights.tolist())
        all_component_indices_list.append(components)
        
        # For embedding computation, use primary component's context
        primary_idx = components[0]
        if primary_idx < len(original_data['st_contexts']):
            primary_context = original_data['st_contexts'][primary_idx].copy()
        else:
            # Fallback: use first available context
            primary_context = component_contexts[0].copy() if component_contexts else np.zeros_like(original_data['st_contexts'][0])
        
        # Ensure primary_context has correct shape [1, num_features]
        if primary_context.ndim == 2 and primary_context.shape[0] == 1:
            # Already correct shape [1, num_features]
            pass
        elif primary_context.ndim == 1:
            # Need to add batch dimension: [num_features] -> [1, num_features]
            primary_context = np.expand_dims(primary_context, axis=0)
        elif primary_context.ndim == 3 and primary_context.shape[1] == 1:
            # Shape is [1, 1, num_features], remove middle dimension
            primary_context = primary_context[:, 0, :]
        
        primary_st_list.append(primary_context)
        
        # Create ID
        if components[0] < len(original_data['ids']):
            primary_id = original_data['ids'][components[0]]
        else:
            primary_id = f"unknown_{components[0]}"
        
        mixed_ids_list.append(f"mixed_{mix['mixed_sample_id']}_from_{primary_id}")
        successful_mix_ids.append(mix['mixed_sample_id'])
    """# Save first mixed audio and stop
    import soundfile as sf
    mixed_np = mixed_audio_list[1].cpu().numpy()
    sf.write('/scratch/e1583377/debug_mix.wav', mixed_np, 32000)
    print("Saved first mix to /scratch/e1583377/debug_mix.wav - CHECK THIS FILE")
    raise SystemExit("Stopping for audio verification")
    """
    
    if not mixed_audio_list:
        print(f"? No valid mixed samples created for batch {batch_spec['batch_id']}")
        return None
    
    print(f"? Created {len(mixed_audio_list)}/{len(batch_spec['mixes'])} successful mixes")
    
    # STEP 5: Compute embeddings using primary context
    mixed_audio_batch = torch.stack(mixed_audio_list)
    primary_st_batch = np.array(primary_st_list)
    
    print(f"\nDEBUG - Shapes for embedding computation:")
    print(f"  Audio batch shape: {mixed_audio_batch.shape}")
    print(f"  Primary ST batch shape: {primary_st_batch.shape}")
    
    # Ensure correct dtype and shape
    audio_numpy = mixed_audio_batch.cpu().numpy().astype(np.float32)
    
    if primary_st_batch.dtype != np.float32:
        primary_st_batch = primary_st_batch.astype(np.float32)
    
    # Ensure 3D shape: [batch_size, 1, num_features]
    if primary_st_batch.ndim == 2:
        primary_st_batch = np.expand_dims(primary_st_batch, axis=1)
    elif primary_st_batch.ndim == 3 and primary_st_batch.shape[1] != 1:
        # Reshape to [batch_size, 1, num_features]
        batch_size, _, num_features = primary_st_batch.shape
        primary_st_batch = primary_st_batch.reshape(batch_size, 1, num_features)
    
    print(f"  Final ST shape: {primary_st_batch.shape}, dtype: {primary_st_batch.dtype}")
    
    print("Computing embeddings...")
    with torch.no_grad():
        audio_embeds, st_embeds = geo_aware_perch.embed_audio_and_spatiotemporal(
            audio_numpy,
            primary_st_batch
        )
    
    # Convert embeddings to numpy if needed
    if isinstance(audio_embeds, torch.Tensor):
        audio_embeds = audio_embeds.cpu().numpy()
    if isinstance(st_embeds, torch.Tensor):
        st_embeds = st_embeds.cpu().numpy()
    
    # STEP 6: Return data with FULL context information
    return {
        'audio_embeddings': audio_embeds,
        'spatiotemporal_embeddings': st_embeds,
        'labels_sparse': mixed_labels_sparse_list,
        'primary_spatiotemporal_contexts': primary_st_batch,  # Context used for embeddings
        'component_contexts': all_component_contexts_list,    # ALL contexts for each mix
        'component_weights': all_component_weights_list,      # Mixing weights
        'component_indices': all_component_indices_list,      # Original indices
        'ids': mixed_ids_list,
        'mixed_sample_ids': successful_mix_ids,
        'is_mixed': np.ones(len(mixed_audio_list), dtype=bool),
        'batch_id': batch_spec['batch_id'],
        'success_rate': len(mixed_audio_list) / len(batch_spec['mixes']),
        'metadata': {
            'context_preservation': 'full',  # Flag indicating we saved all contexts
            'num_components_per_mix': [len(ctx_list) for ctx_list in all_component_contexts_list],
            'embedding_context': 'primary_component'  # Which context was used for embeddings
        }
    }

# ============================================================================
# POST-PROCESSING FUNCTIONS FOR CONTEXT AGGREGATION
# ============================================================================

def aggregate_contexts_later(data, aggregation_method='weighted_average'):
    """
    Aggregate saved component contexts using different methods.
    
    Args:
        data: The data dict saved by process_mixing_batch_with_full_context
        aggregation_method: One of:
            - 'weighted_average': Weighted average of all contexts
            - 'primary_only': Use only primary component (already computed)
            - 'random_component': Randomly select one component
            - 'weighted_selection': Select based on mixing weights
            - 'circular_time': Circular average for time, spatial from primary
            - 'nearest_to_average': Use component nearest to weighted average
    
    Returns:
        aggregated_st: Aggregated spatiotemporal contexts [batch_size, 1, num_features]
    """
    component_contexts = data['component_contexts']
    component_weights = data['component_weights']
    
    batch_size = len(component_contexts)
    aggregated_st = []
    
    for i in range(batch_size):
        contexts = component_contexts[i]  # List of [1, num_features] arrays
        weights = component_weights[i]
        
        if aggregation_method == 'weighted_average':
            # Weighted average
            contexts_array = np.array(contexts)  # [n_components, 1, num_features]
            # Average across components dimension
            weighted_avg = np.average(contexts_array, axis=0, weights=weights)
            aggregated_st.append(weighted_avg)
            
        elif aggregation_method == 'primary_only':
            # Use first component (already computed in data['primary_spatiotemporal_contexts'])
            aggregated_st.append(contexts[0])
            
        elif aggregation_method == 'random_component':
            # Randomly select one component
            idx = np.random.randint(len(contexts))
            aggregated_st.append(contexts[idx])
            
        elif aggregation_method == 'weighted_selection':
            # Select component based on weights
            idx = np.random.choice(len(contexts), p=weights/np.sum(weights))
            aggregated_st.append(contexts[idx])
            
        elif aggregation_method == 'circular_time':
            # Circular average for time, spatial from primary
            # Assuming spatial features are first 2 (lat, lon), temporal are rest
            primary_context = contexts[0][0]  # [num_features]
            
            # Extract temporal features from all components
            temporal_features = []
            for ctx in contexts:
                temporal_features.append(ctx[0, 2:])  # Skip first 2 spatial features
            
            temporal_features = np.array(temporal_features)  # [n_components, n_temporal]
            
            # Mix temporal features circularly
            mixed_temporal = []
            for j in range(temporal_features.shape[1]):
                if j == 0:  # day_of_year
                    mixed_val = circular_average(temporal_features[:, j], weights, period=365.0)
                elif j == 1:  # hour_of_day
                    mixed_val = circular_average(temporal_features[:, j], weights, period=24.0)
                else:
                    mixed_val = np.average(temporal_features[:, j], weights=weights)
                mixed_temporal.append(mixed_val)
            
            # Combine spatial from primary with mixed temporal
            mixed_context = np.concatenate([
                primary_context[:2],  # Spatial
                mixed_temporal       # Temporal
            ])
            aggregated_st.append(np.expand_dims(mixed_context, axis=0))
            
        elif aggregation_method == 'nearest_to_average':
            # Find component nearest to weighted average
            contexts_array = np.array(contexts)  # [n_components, 1, num_features]
            weighted_avg = np.average(contexts_array, axis=0, weights=weights)[0]  # [num_features]
            
            # Calculate distances
            distances = []
            for ctx in contexts:
                dist = np.sqrt(np.sum((ctx[0] - weighted_avg) ** 2))
                distances.append(dist)
            
            nearest_idx = np.argmin(distances)
            aggregated_st.append(contexts[nearest_idx])
    
    return np.array(aggregated_st)

def circular_average(values, weights, period):
    """Weighted circular average."""
    angles = 2 * np.pi * values / period
    sin_sum = np.sum(weights * np.sin(angles))
    cos_sum = np.sum(weights * np.cos(angles))
    mean_angle = np.arctan2(sin_sum, cos_sum)
    return (mean_angle * period / (2 * np.pi)) % period

# ============================================================================
# SAVE AND LOAD FUNCTIONS FOR FULL CONTEXT DATA
# ============================================================================

def save_full_context_data(data, filepath):
    """
    Save data with full context information.
    
    Optimizes storage by converting lists of arrays to single concatenated arrays.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Create a copy to avoid modifying original
    data_to_save = data.copy()
    
    # Optimize storage of component_contexts
    if 'component_contexts' in data_to_save:
        component_contexts = data_to_save['component_contexts']
        
        # Flatten to single arrays with metadata
        all_contexts = []
        context_lengths = []
        context_shapes = []
        
        for ctx_list in component_contexts:
            # Flatten this mix's contexts
            flattened = []
            for ctx in ctx_list:
                flattened.append(ctx.flatten())
            
            if flattened:
                all_contexts.extend(flattened)
                context_lengths.append(len(flattened))
                context_shapes.append(ctx_list[0].shape)
            else:
                context_lengths.append(0)
                context_shapes.append(None)
        
        # Replace with optimized storage
        data_to_save['component_contexts_flat'] = np.concatenate(all_contexts) if all_contexts else np.array([])
        data_to_save['component_contexts_lengths'] = np.array(context_lengths, dtype=np.int32)
        data_to_save['component_contexts_shapes'] = context_shapes
        
        # Remove original to save space
        del data_to_save['component_contexts']
    
    # Optimize other list fields if needed
    for key in ['component_weights', 'component_indices', 'labels_sparse']:
        if key in data_to_save and isinstance(data_to_save[key], list):
            # Convert to object array for variable-length lists
            data_to_save[key] = np.array(data_to_save[key], dtype=object)
    
    # Save
    with open(filepath, 'wb') as f:
        pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"? Saved full context data to {filepath}")
    print(f"  Original size: {len(data.get('component_contexts', []))} mixes")
    
    return filepath

def load_full_context_data(filepath):
    """
    Load data and reconstruct component contexts.
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Reconstruct component_contexts if saved in optimized format
    if 'component_contexts_flat' in data:
        flat_contexts = data['component_contexts_flat']
        lengths = data['component_contexts_lengths']
        shapes = data['component_contexts_shapes']
        
        # Reconstruct
        component_contexts = []
        idx = 0
        for length, shape in zip(lengths, shapes):
            if length > 0 and shape is not None:
                mix_contexts = []
                for _ in range(length):
                    size = np.prod(shape)
                    ctx_flat = flat_contexts[idx:idx+size]
                    ctx = ctx_flat.reshape(shape)
                    mix_contexts.append(ctx)
                    idx += size
                component_contexts.append(mix_contexts)
            else:
                component_contexts.append([])
        
        data['component_contexts'] = component_contexts
        
        # Remove optimized fields
        del data['component_contexts_flat']
        del data['component_contexts_lengths']
        del data['component_contexts_shapes']
    
    # Convert object arrays back to lists if needed
    for key in ['component_weights', 'component_indices', 'labels_sparse']:
        if key in data and isinstance(data[key], np.ndarray) and data[key].dtype == object:
            data[key] = data[key].tolist()
    
    return data


def main_with_precomputed_mixup(mixing_plan = ""):
    """Main pipeline with precomputed dataset-wide mixup."""
    OUTPUT_PATH = '/scratch/e1583377/pickled_audio_embeds_mixed' + mixing_plan + "/"
    os.makedirs(OUTPUT_PATH, exist_ok = True)
    MIXUP_PLAN_PATH = Path(OUTPUT_PATH) / (mixing_plan + "inat_mixing_plan.json")
    
    # Load your existing data
    prepared_train = prepare_for_spatiotemporal_encoder('all', temporal_encoder=encode_time)
    st_train, audio_paths_train, ids_train, y_train = prepared_train[0], prepared_train[5], prepared_train[2], torch.from_numpy(prepared_train[1]).long()
    d = "cuda"
    ST_MODEL_PATH = "/home/svu/e1583377/Spatial_Perch_Transfer_Learning/sphere2vec/main/perch_v2_xc_non_fourier_spatiotemporal_encoder_lr_0001.pth"
    geo_aware_perch = GeoAwarePerch(load_st_encoder(ST_MODEL_PATH, 5, model_device=d, output_dim=14795), perch_model, output_dim=14795)
    geo_aware_perch.eval()
    
    total_samples = len(y_train)
    print(f"Total samples in dataset: {total_samples}")
    
    
    # Precompute mixing plan using only valid samples
    mixup = PrecomputedDatasetMixup(mixup_ratio=1.0)
    
    print("Precomputing mixing plan...")
    if os.path.exists(MIXUP_PLAN_PATH):
        print("loading existing mixup plan")
        mixing_plan = mixup.load_mixing_plan(MIXUP_PLAN_PATH)
    elif mixing_plan == "":
        # default option, purely random mixing, like in the Perch 2.0 paper
        mixing_plan = mixup.precompute_mixing_plan(
            len(audio_paths_train),  # Use count of valid samples
            OUTPUT_PATH
        )
    elif mixing_plan == "geospatial_top_k_":
        mixing_plan = mixup.precompute_mixing_plan_geospatial(
            len(audio_paths_train),  # Use count of valid samples
            st_train,
            OUTPUT_PATH,
            100
        )
    # Organize original data for easy access
    original_data = {
        'occurrence_ids': np.array(audio_paths_train),
        'st_contexts': np.array(st_train),
        'labels': y_train,
        'ids': np.array(ids_train)
    }
    
    # Process in batches
    mixing_batches = create_mixing_batches(mixing_plan, original_data, batch_size=128)
    
    for batch_spec in mixing_batches:
        batch_id = batch_spec['batch_id']
        outfile = OUTPUT_PATH + f"inat_mixed_full_context_batch_{batch_id:04d}.pkl"
        
        if os.path.exists(outfile):
            print(f"Batch {batch_id} already processed, skipping...")
            continue
        
        print(f"\nProcessing mixing batch {batch_id}...")
        
        # Process with full context preservation
        batch_result = process_mixing_batch_with_full_context(
            batch_spec,
            original_data,
            geo_aware_perch,
            device='cuda'
        )
        
        if batch_result:
            # Save with optimized storage
            save_full_context_data(batch_result, outfile)
            print(f"? Saved batch {batch_id} with full context")
        
        torch.cuda.empty_cache()




# ============================================================================
# UTILITY: Combine Original and Mixed Datasets
# ============================================================================

def combine_original_and_mixed_datasets(original_embeddings_path, mixed_embeddings_path, output_path):
    """
    Combine original embeddings with mixed embeddings for training.
    """
    # Load all original embeddings
    print("Loading original embeddings...")
    original_files = sorted(Path(original_embeddings_path).glob("*.pkl"))
    original_data = load_all_embeddings_from_files(original_files)
    
    # Load all mixed embeddings
    print("Loading mixed embeddings...")
    mixed_files = sorted(Path(mixed_embeddings_path).glob("mixed_batch_*.pkl"))
    mixed_data = load_all_embeddings_from_files(mixed_files)
    
    # Add mix flag to original data
    original_data['is_mixed'] = np.zeros(len(original_data['audio_embeddings']), dtype=bool)
    
    # Combine
    print("Combining datasets...")
    combined_data = {}
    for key in original_data.keys():
        if key in mixed_data:
            combined_data[key] = np.concatenate([original_data[key], mixed_data[key]])
        else:
            combined_data[key] = original_data[key]
    
    # Save combined dataset
    combined_file = Path(output_path) / "combined_original_and_mixed.pkl"
    with open(combined_file, 'wb') as f:
        pickle.dump(combined_data, f)
    
    print(f"Saved combined dataset with {len(combined_data['audio_embeddings'])} samples "
          f"({len(original_data['audio_embeddings'])} original + {len(mixed_data['audio_embeddings'])} mixed)")
    
    return combined_data

def load_all_embeddings_from_files(file_list):
    """Helper to load multiple embedding files."""
    all_data = None
    
    for filepath in tqdm(file_list, desc="Loading files"):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        if all_data is None:
            all_data = {k: [] for k in data.keys()}
        
        for key in data.keys():
            all_data[key].append(data[key])
    
    # Concatenate
    for key in all_data.keys():
        all_data[key] = np.concatenate(all_data[key])
    
    return all_data

if __name__ == '__main__':
    main_with_precomputed_mixup("")