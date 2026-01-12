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
import kagglehub
SAVE_DIR = "/scratch/e1583377/pickled_birdset_embeds/"
MIN_YEAR, MAX_YEAR = 2005, 2025
def load_perch_gpu_model(url = 'https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2/2'):
    return hub.load(url)
def load_perch_cpu_model(url = "google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu"):
    path = kagglehub.model_download(url)
    return hub.load(path)
def load_birdset_data(subset = "NES"):
    dataset = load_dataset("DBD-research-group/BirdSet", subset, cache_dir = "/scratch/e1583377/huggingface/")


    # the dataset comes without an automatic Audio casting, this has to be enabled via huggingface
    # this means that each time a sample is called, it is decoded (which may take a while if done for the complete dataset)
    # in BirdSet, this is all done on-the-fly during training and testing (since the dataset size would be too big if mapping and saving it only once)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=32_000))
    return dataset["test_5s"]
def encode_time(time, date, min_year, max_year):
    # Cyclic encodings
    hour = int(time[:time.index(":")]) + int(time[time.index(":") + 1:]) / 60
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    year, month, day = date.split("-")
    # simple approximation of days per month rather than mapping each month to it's correspoing length
    day_of_year = 30.4167 * int(month) + int(day)
    day_sin = math.sin(2 * math.pi * day_of_year / 365)
    day_cos = math.cos(2 * math.pi * day_of_year / 365)
    
    # Linear normalized year
    year_norm = (int(year) - min_year) / (max_year - min_year)
    return np.array([hour_sin, hour_cos, day_sin, day_cos, year_norm])
class BirdsetTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, recording_date, temporal_encoder = encode_time):
        self.audio = [sample["audio"] for sample in hf_dataset]
        self.labels = [sample["ebird_code_multilabel"] for sample in hf_dataset]
        self.lats = [sample["lat"] for sample in hf_dataset]
        self.longs = [sample["long"] for sample in hf_dataset]
        self.local_times = [sample["local_time"] for sample in hf_dataset]

        temporal_encodings = np.array([temporal_encoder(t[:-3], recording_date, MIN_YEAR, MAX_YEAR) for t in self.local_times])
        geo = np.stack([self.longs, self.lats], axis=1)
        self.st_context = np.concatenate([geo, temporal_encodings], axis=1)
        
    def __len__(self):
        return len(self.audio)
    
    def __getitem__(self, idx):
        audio = self.audio[idx]["array"]
        target_length = 5 * 32_000
        if len(audio) < target_length:
            # pad and append
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            #num_padded += 1
        else:
            # Truncate
           audio = audio[:target_length]
        return (
            #normalize audio
            audio / np.max(np.abs(audio)) * 0.25,
            self.st_context[idx],
            self.labels[idx]
        )
def custom_collate(batch):
    audio_batch = np.array([item[0] for item in batch])  # numpy
    st_batch = np.array([item[1] for item in batch])  # numpy
    label_batch = [item[2] for item in batch] # list
    return np.stack(audio_batch), st_batch, label_batch
    
if __name__ == '__main__':
    task = "SSW"
    print(task)
    perch_model = load_perch_gpu_model()
    huggingface_dataset = load_birdset_data(task)
    # birdset lacks date information, need to manually find date(s)
    # representing the recorded dates of the samples
    # for PER: between January 14th and February 2nd, 2019
    # for NES: around september 14 2019
    # for HSN: between July 9 and 12, 2015
    # for SNE: May-August 2018 (picked 06-15 as midpoint)
    # for POW: April-July 2018 (picked 06-01 as midpoint)
    # for UHH: 2016 and 2022, to wide to pick a useful representative date.
    # for SSW: between Feb and Aug 2017 (picked 05-15 as midpoint)
    recording_date = "2018-05-01" # for POW
    torch_dataset = BirdsetTorchDataset(huggingface_dataset, recording_date)
    dataloader = DataLoader(torch_dataset, batch_size=128, collate_fn=custom_collate, shuffle=False)
    all_embeddings = []
    all_labels = []
    all_spatiotemporal = []
    for audio_array, st_array, ebird_codes in dataloader:
        audio_array = audio_array.astype(np.float32)
        perch_output = perch_model.signatures['serving_default'](inputs=audio_array)['embedding'].numpy()
        all_embeddings.append(perch_output)
        all_spatiotemporal.append(st_array)
        all_labels.append(ebird_codes)
    os.makedirs(SAVE_DIR, exist_ok = True)
    with open(SAVE_DIR + task + ".pkl", "wb") as f:
        pickle.dump({'embeddings':all_embeddings, 'labels':all_labels, "st_context":all_spatiotemporal}, f)
        