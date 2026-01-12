from datasets import Audio, load_dataset
import torch
from torch.utils.data import DataLoader
import tensorflow as tf
import tensorflow_hub as hub
tf.experimental.numpy.experimental_enable_numpy_behavior()
import pickle
import os
import numpy as np
import math
import gc

SAVE_DIR = "/scratch/e1583377/pickled_birdset_embeds/"
MIN_YEAR, MAX_YEAR = 2005, 2025

def load_perch_gpu_model(url = 'https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2/2'):
    return hub.load(url)

def load_birdset_data(subset = "NES"):
    dataset = load_dataset("DBD-research-group/BirdSet", subset, cache_dir = "/scratch/e1583377/huggingface/")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=32_000))
    return dataset["test_5s"]

def encode_time(time, date, min_year, max_year):
    # Cyclic encodings
    hour = int(time[:time.index(":")]) + int(time[time.index(":") + 1:]) / 60
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    year, month, day = date.split("-")
    day_of_year = 30.4167 * int(month) + int(day)
    day_sin = math.sin(2 * math.pi * day_of_year / 365)
    day_cos = math.cos(2 * math.pi * day_of_year / 365)
    
    year_norm = (int(year) - min_year) / (max_year - min_year)
    return np.array([hour_sin, hour_cos, day_sin, day_cos, year_norm])

class BirdsetTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, recording_date, temporal_encoder = encode_time):
        # Store dataset directly instead of pre-loading all data
        self.hf_dataset = hf_dataset
        self.recording_date = recording_date
        self.temporal_encoder = temporal_encoder
        
        # Pre-compute only the spatiotemporal context to save memory
        self.local_times = [sample["local_time"] for sample in hf_dataset]
        self.lats = [sample["lat"] for sample in hf_dataset]
        self.longs = [sample["long"] for sample in hf_dataset]
        
        temporal_encodings = np.array([temporal_encoder(t[:-3], recording_date, MIN_YEAR, MAX_YEAR) for t in self.local_times])
        geo = np.stack([self.longs, self.lats], axis=1)
        self.st_context = np.concatenate([geo, temporal_encodings], axis=1)
        
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        # Load audio on-the-fly
        audio_data = self.hf_dataset[idx]["audio"]
        audio_array = audio_data["array"]
        
        target_length = 5 * 32_000
        if len(audio_array) < target_length:
            audio_array = np.pad(audio_array, (0, target_length - len(audio_array)), mode='constant')
        else:
            audio_array = audio_array[:target_length]
            
        return (
            audio_array / np.max(np.abs(audio_array)) * 0.25,
            self.st_context[idx],
            self.hf_dataset[idx]["ebird_code_multilabel"]
        )

def custom_collate(batch):
    audio_batch = np.array([item[0] for item in batch])
    st_batch = np.array([item[1] for item in batch])
    label_batch = [item[2] for item in batch]
    return np.stack(audio_batch), st_batch, label_batch

def process_and_save_batch(perch_model, audio_batch, st_batch, label_batch, task, batch_idx):
    """Process a single batch and save it immediately"""
    audio_batch = audio_batch.astype(np.float32)
    perch_output = perch_model.signatures['serving_default'](inputs=audio_batch)['embedding'].numpy()
    
    # Save this batch immediately
    batch_save_dir = os.path.join(SAVE_DIR, task, "batches")
    os.makedirs(batch_save_dir, exist_ok=True)
    
    batch_filename = os.path.join(batch_save_dir, f"batch_{batch_idx:06d}.pkl")
    with open(batch_filename, "wb") as f:
        pickle.dump({
            'embeddings': perch_output, 
            'labels': label_batch, 
            "st_context": st_batch
        }, f)
    
    # Clear memory
    del audio_batch, perch_output
    gc.collect()
    
    return batch_filename

def combine_batches(task, total_batches):
    """Combine all batch files into a single file"""
    all_embeddings = []
    all_labels = []
    all_spatiotemporal = []
    
    batch_save_dir = os.path.join(SAVE_DIR, task, "batches")
    
    for i in range(total_batches):
        batch_filename = os.path.join(batch_save_dir, f"batch_{i:06d}.pkl")
        with open(batch_filename, "rb") as f:
            batch_data = pickle.load(f)
            all_embeddings.append(batch_data['embeddings'])
            all_labels.extend(batch_data['labels'])
            all_spatiotemporal.append(batch_data['st_context'])
        
        # Remove batch file to save space
        os.remove(batch_filename)
    
    # Remove the batches directory
    os.rmdir(batch_save_dir)
    
    # Save combined file
    final_filename = os.path.join(SAVE_DIR, task + ".pkl")
    with open(final_filename, "wb") as f:
        pickle.dump({
            'embeddings': all_embeddings, 
            'labels': all_labels, 
            "st_context": all_spatiotemporal
        }, f)
    
    print(f"Saved combined data to {final_filename}")

if __name__ == '__main__':
    task = "SSW"
    print(f"Processing {task} dataset")
    
    # Reduce batch size for large datasets
    batch_size = 128
    
    perch_model = load_perch_gpu_model()
    huggingface_dataset = load_birdset_data(task)
    
    recording_date = "2018-05-01"  # for POW
    torch_dataset = BirdsetTorchDataset(huggingface_dataset, recording_date)
    dataloader = DataLoader(torch_dataset, batch_size=batch_size, collate_fn=custom_collate, shuffle=False, num_workers=0)
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Process in batches and save each batch immediately
    batch_filenames = []
    for batch_idx, (audio_array, st_array, ebird_codes) in enumerate(dataloader):
        print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
        
        batch_filename = process_and_save_batch(
            perch_model, audio_array, st_array, ebird_codes, task, batch_idx
        )
        batch_filenames.append(batch_filename)
        
        # Clear memory after each batch
        del audio_array, st_array, ebird_codes
        gc.collect()
    
    print("All batches processed. Combining into final file...")
    combine_batches(task, len(dataloader))
    print("Done!")