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
OUTPUT_PATH = '/scratch/e1583377/pickled_audio_embeds_3_random_windows_corrected/'
import time
import random
import tempfile

import numpy as np
import torch



def chunked(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

def fetch_inat_audio_batch_safe(occurrence_ids, max_batch_size=200, max_retries=5, delay_between_batches=1):
    """
    Fetch audio URLs for a batch of occurrenceIDs with retries and chunking.
    Preserves batch order, returns one URL per occurrence (or None if unavailable).
    """
    all_urls = []
    
    # Split into safe-size API batches
    for batch_ids in chunked(occurrence_ids, max_batch_size):
        # Extract numeric IDs
        obs_ids = [occ.rstrip("/").split("/")[-1] for occ in batch_ids]
        
        # Retry loop
        for attempt in range(max_retries):
            try:
                # Construct API query
                params = [("id[]", oid) for oid in obs_ids]
                r = requests.get("https://api.inaturalist.org/v1/observations", params=params, headers = headers, timeout=20)
                
                if r.status_code == 429:
                    raise Exception("Rate limited (429)")
                
                if not r.ok:
                    raise Exception(f"HTTP {r.status_code}")
                
                data = r.json().get("results", [])
                
                # Map observation ID ? first audio URL
                id_to_url = {str(obs["id"]): obs["sounds"][0]["file_url"] if obs.get("sounds") else None
                             for obs in data}
                
                # Preserve order
                batch_urls = [id_to_url.get(oid, None) for oid in obs_ids]
                all_urls.extend(batch_urls)
                
                # Delay to be polite
                time.sleep(delay_between_batches + random.random() * 0.2)
                break  # success, move to next batch
            
            except Exception as e:
                sleep_time = 1.5 ** attempt + random.random() * 0.5
                print(f"API batch error: {e}. Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
        
        else:
            # Max retries exceeded ? append None for all batch items
            print("Max retries exceeded for this batch, skipping all items.")
            all_urls.extend([None] * len(batch_ids))
    
    return all_urls



"""
def process_single_audio(args):
    url, idx, target_sr, strategy, num_segments = args
    try:
        # Download audio data
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Create file-like object
        audio_bytes = io.BytesIO(response.content)

        # Load audio using librosa
        waveform, sample_rate = librosa.load(audio_bytes, sr=target_sr)
        
        target_length = 5 * target_sr
        
        # If audio is shorter than target, pad it and return single segment
        if len(waveform) < target_length:
            waveform = np.pad(waveform, (0, target_length - len(waveform)), mode='constant')
            segments = [waveform]  # Single segment for short audio
        else:
            # For audio longer than target_length, extract multiple segments
            max_start = len(waveform) - target_length
            
            if strategy == "random":
                if num_segments > 1:
                    # Generate multiple random start positions
                    start_samples = np.random.randint(0, max_start + 1, size=num_segments)
                    segments = [waveform[start:start + target_length] for start in start_samples]
                else:
                    # Single random segment (original behavior)
                    start_sample = torch.randint(0, max_start + 1, (1,)).item()
                    segments = [waveform[start_sample:start_sample + target_length]]
                    
            elif strategy == "peak_select":
                peak_idx = np.argmax(np.abs(waveform))
                start_idx = peak_idx - target_length // 2
                end_idx = start_idx + target_length
                
                if start_idx < 0:
                    segment = waveform[:target_length]
                elif end_idx > len(waveform):
                    segment = waveform[-target_length:]
                else:
                    segment = waveform[start_idx:end_idx]
                segments = [segment]  # Peak select only returns one segment
            else:
                raise ValueError(f"Invalid strategy: {strategy}")
        
        # Normalize each segment
        normalized_segments = []
        for segment in segments:
            max_val = np.max(np.abs(segment))
            if max_val > 0:  # Avoid division by zero
                normalized_segment = segment / max_val * 0.25
                normalized_segments.append(normalized_segment)
            else:
                normalized_segments.append(segment)  # Keep as-is if all zeros
        
        return normalized_segments, idx, None
    except Exception as e:
        return None, idx, str(e)
"""

def load_audio_from_bytes(data: bytes, target_sr: int, url: str | None = None):
    try:
        # --- choose format for FFmpeg to infer ------------------------------------
        warnings.filterwarnings("ignore", category=UserWarning)
        fmt = None
        if url:
            ext = url.split('?')[0].rsplit('.', 1)[-1].lower()
            if ext in {"m4a", "mp3", "aac", "wav", "flac"}:
                fmt = ext

        # --- write bytes to temporary file ----------------------------------------
        with tempfile.NamedTemporaryFile(suffix=f".{fmt}" if fmt else "", delete=False) as tmp:
            tmp.write(data)
            tmp.flush()
            tmp_path = tmp.name

        # --- load audio from file --------------------------------------------------
        audio = AudioSegment.from_file(tmp_path, format=fmt)
        audio = audio.set_channels(1)           # convert to mono
        audio = audio.set_frame_rate(target_sr) # resample if needed

        # --- convert to numpy array ------------------------------------------------
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        if np.max(np.abs(samples)) > 0:
            samples /= np.max(np.abs(samples))  # normalize to [-1, 1]
        samples *= 0.25                         # scale to [-0.25, 0.25]

        return samples

    except Exception as e:
        print(f"Audio decode failed: {e}")
        return None

    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
RATE_LIMIT_DELAY   = 1.5
JITTER             = 0.5
BACKOFF_EXP        = 2.0
GLOBAL_BAN_FILE = Path("/scratch/e1583377/cdn_ban.flag")
BAN_COOLDOWN    = 45 * 60          # 45 min

def wait_out_ban():
    age = time.time() - GLOBAL_BAN_FILE.stat().st_mtime
    if age < BAN_COOLDOWN:
        print(f"CDN ban still active - sleeping {BAN_COOLDOWN - age:.0f} s ..")
        time.sleep(BAN_COOLDOWN - age)
    GLOBAL_BAN_FILE.unlink(missing_ok=True)
def process_single_audio(args, max_retries=3):
    """
    Process a single audio URL with multiple segments, retrying on network errors or rate limits.
    Supports .m4a files using torchaudio (requires ffmpeg backend).
    
    Returns:
        (list of np.ndarray segments, original index, error message or None)
    """
    url, idx, target_sr, strategy, num_segments = args
    if GLOBAL_BAN_FILE.exists():
        wait_out_ban()

    if url is None:
        return None, idx, "No URL"

    for attempt in range(max_retries):
        try:
            time.sleep(RATE_LIMIT_DELAY + random.uniform(0, JITTER))
            r = requests.get(url, headers = DOWNLOAD_HEADERS, timeout=30)
            if r.status_code == 429:
                raise Exception("Rate limited (429)")
            r.raise_for_status()
            print(f"[{idx}]  HTTP {r.status_code}  "
                  f"Content-Type: {r.headers.get('Content-Type','???')}  "
                  f"size={len(r.content)}")

            audio_bytes = r.content
            waveform = load_audio_from_bytes(audio_bytes, 32000, url)

            target_length = 5 * target_sr

            # Pad short audio
            if len(waveform) < target_length:
                waveform = np.pad(waveform, (0, target_length - len(waveform)), mode='constant')
                segments = [waveform]
            else:
                max_start = len(waveform) - target_length

                if strategy == "random":
                    if num_segments > 1:
                        start_samples = np.random.randint(0, max_start + 1, size=num_segments)
                        segments = [waveform[start:start + target_length] for start in start_samples]
                    else:
                        start_sample = torch.randint(0, max_start + 1, (1,)).item()
                        segments = [waveform[start_sample:start_sample + target_length]]

                elif strategy == "peak_select":
                    peak_idx = np.argmax(np.abs(waveform))
                    start_idx = peak_idx - target_length // 2
                    end_idx = start_idx + target_length
                    if start_idx < 0:
                        segment = waveform[:target_length]
                    elif end_idx > len(waveform):
                        segment = waveform[-target_length:]
                    else:
                        segment = waveform[start_idx:end_idx]
                    segments = [segment]

                else:
                    raise ValueError(f"Invalid strategy: {strategy}")

            # Normalize segments
            normalized_segments = []
            for segment in segments:
                max_val = np.max(np.abs(segment))
                if max_val > 0:
                    normalized_segments.append(segment / max_val * 0.25)
                else:
                    normalized_segments.append(segment)

            return normalized_segments, idx, None

        except Exception as e:
            if "403" in str(e):
                # global ban if we hit **many** 403 in a row
                if attempt >= 2:
                    GLOBAL_BAN_FILE.touch()
                sleep = BACKOFF_EXP ** attempt + random.uniform(0, JITTER)
                print(f"HTTP-403 for {url} - sleeping {sleep:.1f}s")
            else:
                # exponential backoff + jitter
                sleep_time = (1.5 ** attempt) + random.random() * 0.5
                print(f"Audio download error (attempt {attempt+1}/{max_retries}) for {url}: {e}. Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)

    # If all retries fail
    return None, idx, f"Failed after {max_retries} attempts"



def load_audio_from_url_parallel(urls, target_sr=32000, max_workers=3, strategy="random", num_segments=1):
    """
    Load audio from URLs in parallel, extracting multiple segments for long files
    """
    segments = [None] * len(urls)
    errors = [None] * len(urls)
    
    # Prepare arguments
    tasks = [(url, idx, target_sr, strategy, num_segments) for idx, url in enumerate(urls)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(process_single_audio, task): task[1] for task in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_index), 
                          total=len(urls), desc="Processing audio"):
            original_idx = future_to_index[future]
            segment_list, _, error = future.result()
            
            if segment_list is not None:
                segments[original_idx] = segment_list
            else:
                errors[original_idx] = error
    
    # Flatten the segments and track original indices
    all_segments = []
    segment_to_original_idx = []  # Maps each segment back to its original index
    
    for idx, segment_list in enumerate(segments):
        if segment_list is not None:
            for segment in segment_list:
                all_segments.append(segment)
                segment_to_original_idx.append(idx)
    
    # Convert to PyTorch tensor
    if all_segments:
        waveform_tensor = torch.stack([torch.from_numpy(waveform) for waveform in all_segments])
    else:
        waveform_tensor = torch.tensor([])
    
    return waveform_tensor, segment_to_original_idx

def save_embeddings_basic(audio_embeddings, spatiotemporal_embeddings, batch_st, batch_labels, batch_ids, filepath):
    """
    Save embeddings - now handles multiple segments per original sample
    """
    data = {
        'audio_embeddings': audio_embeddings,
        'spatiotemporal_embeddings': spatiotemporal_embeddings,
        'labels': batch_labels,
        'spatiotemporal_contexts': batch_st,
        'ids': batch_ids
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    NUM_SEGMENTS = 3  # Number of random segments to extract from long files

    prepared_train = prepare_for_spatiotemporal_encoder('all', temporal_encoder=encode_time)
    st_train, audio_paths_train, ids_train, y_train = prepared_train[0], prepared_train[5], prepared_train[2], torch.from_numpy(prepared_train[1]).long()
    print("data prepared!")
    d = "cuda"
    ST_MODEL_PATH = "/home/svu/e1583377/Spatial_Perch_Transfer_Learning/sphere2vec/main/perch_v2_xc_non_fourier_spatiotemporal_encoder_lr_0001.pth"
    geo_aware_perch = GeoAwarePerch(load_st_encoder(ST_MODEL_PATH, 5, model_device=d, output_dim=14795), perch_model, output_dim=14795) 
    geo_aware_perch.eval() 
    batch_size = 128
    i = 0
    
    with torch.no_grad():
        while i * batch_size < len(y_train):
            slice_end = min(batch_size * (i + 1), len(y_train))
            outfile = OUTPUT_PATH + "inaturalist_" + str(batch_size * i) + "-" + str(slice_end) + ".pkl"
            
            if not os.path.exists(outfile):
                # Get the full batch first
                batch_st = st_train[batch_size * i: slice_end]
                batch_ids = ids_train[batch_size * i: slice_end]
                batch_st = np.expand_dims(batch_st, axis=1)
                batch_audio_paths = fetch_inat_audio_batch_safe(audio_paths_train[batch_size * i: slice_end])
                batch_labels = y_train[batch_size * i: slice_end]

                # Process audio and get segments with mapping
                batch_audio_tensor, segment_indices = load_audio_from_url_parallel(
                    batch_audio_paths, 
                    strategy="random", 
                    num_segments=NUM_SEGMENTS
                )
                
                if len(segment_indices) == 0:
                    print(f"Batch {i}: No valid audio samples, skipping")
                    i += 1
                    continue

                # Use segment_indices to duplicate the corresponding metadata
                filtered_st = batch_st[segment_indices]
                filtered_labels = batch_labels[segment_indices]
                filtered_ids = batch_ids[segment_indices]

                # Now process all segments
                audio_embeds, st_embeds = geo_aware_perch.embed_audio_and_spatiotemporal(
                    batch_audio_tensor, 
                    filtered_st
                )
                
                try:
                    assert len(audio_embeds) == len(st_embeds) == len(filtered_st) == len(filtered_labels) == len(filtered_ids)
                except:
                    print("Mismatching lengths from modalities")
                    print(len(audio_embeds), len(st_embeds), len(filtered_labels), len(filtered_ids))

                save_embeddings_basic(
                    audio_embeds.cpu(), 
                    st_embeds.cpu(), 
                    filtered_st,
                    filtered_labels, 
                    filtered_ids, 
                    outfile
                )
            i += 1
            #torch.cuda.empty_cache()