import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
import math
METADATA_PATH = "/scratch/e1583377/inat_metadata.csv"
SPECIES_NAMES_FILEPATH = "/home/svu/e1583377/Spatial_Perch_Transfer_Learning/assets/perch_v2_labels.csv"
COLUMNS = ["gbifID", "day", "month", "year", "speciesKey", "decimalLatitude", "decimalLongitude", "species", "occurrenceID", "eventDate"]
def load_inat_data(path = METADATA_PATH, relevant_columns = COLUMNS):
    all_data = pd.read_csv(path, sep="\t")
    all_data = all_data.dropna(subset=relevant_columns)
    return all_data[relevant_columns]
def get_class_labels(data = load_inat_data(), species_names_path = SPECIES_NAMES_FILEPATH):
    species_labels = pd.read_csv(species_names_path)
    species_names_to_labels = {}
    for label, species_name in enumerate(species_labels["inat2024_fsd50k"]):
        species_names_to_labels[species_name] = int(label)
    print(len(species_names_to_labels.keys()))
    data['perch_label'] = data['species'].map(species_names_to_labels)
    data = data.dropna(subset=['perch_label'])
    return data
    
def split_train_and_val(data = get_class_labels(), split_ratio = 0.3):
    train, val = train_test_split(data, test_size=split_ratio, random_state=74)
    return train, val
    
def prepare_for_sphere2vec(split = 'train', data = get_class_labels()):
    if split == 'train':
        data = split_train_and_val(data)[0]
    elif split == 'val':
        data = split_train_and_val(data)[1]
    elif split == 'all':
        print("using all data without splitting into train and validation")
    else:
        print("Invalid split selected, choose 'train', 'val' or 'all'.")
        return
    locs = data[['lon', 'lat']].to_numpy() # (num_train, 2) lon, lat coordinates of each image
    train_classes = np.array(data['perch_label'])             # (num_train,) perch class labels for each sample
    # not actual user ids, just place holders to fit with sphere2vec inputs
    # should not affect training
    train_users = np.array(data['id'])
    train_dates = np.array(data['date']) # (num_train, ), training dates
    # dummy values for now           
    train_inds = np.array(np.ones(len(data)))              # (num_train, ), the indices training data keeps
    # dummy values, am trying to just train the location encoder
    train_imgs = np.array(["no_image" for i in range(len(data))])              # (num_train, ), the train image file path
    return locs, train_classes, train_users, train_dates, train_inds, train_imgs
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
def encode_time_fourier(time, date, min_year, max_year, num_freqs = 16):
    hour_norm = hour = (int(time[:time.index(":")]) + int(time[time.index(":") + 1:]) / 60) / 24
    year, month, day = date.split("-")
    year_norm = int(year) / (max_year - min_year)
    # simple approximation of days per month rather than mapping each month to it's correspoing length
    day_norm = (30.4167 * int(month) + int(day)) / 365
    encoding = []
    for freq in range(1, num_freqs + 1):
        # Hour encoding
        encoding.append(math.sin(2 * math.pi * freq * hour_norm))
        encoding.append(math.cos(2 * math.pi * freq * hour_norm))
        
        # Day encoding  
        encoding.append(math.sin(2 * math.pi * freq * day_norm))
        encoding.append(math.cos(2 * math.pi * freq * day_norm))
        
        # Year encoding
        encoding.append(math.sin(2 * math.pi * freq * year_norm))
        encoding.append(math.cos(2 * math.pi * freq * year_norm))
    return np.array(encoding)

    
def prepare_for_spatiotemporal_encoder(split='train', data=None, temporal_encoder=encode_time):
    if data is None:
        data = get_class_labels()
    if split == 'train':
        data = split_train_and_val(data)[0]
    elif split == 'val':
        data = split_train_and_val(data)[1]
    elif split == 'all':
        print("using all data without splitting into train and validation")
    else:
        raise ValueError("Invalid split selected, choose 'train' or 'val'.")

    min_year, max_year = 2005, 2025
    locs_and_temporal_encodings = []
    classes = []
    valid_indices = []
    invalid_times = 0

    for idx, sample in data.iterrows():
        try:
            # temporal encoding
            if "T" in sample["eventDate"]:
                date = sample["eventDate"][:sample["eventDate"].index("T")]
                time = sample["eventDate"][sample["eventDate"].index("T") + 1:sample["eventDate"].index("T") + 6]
            else:
                date = sample["eventDate"]
                time = "09:30"
            temporal = temporal_encoder(time, date, min_year, max_year)
            row = np.concatenate([sample[['decimalLongitude', 'decimalLatitude']].to_numpy(), temporal]).astype(np.float32)
            
            locs_and_temporal_encodings.append(row)
            classes.append(sample['perch_label'])
            valid_indices.append(idx)
        except Exception:
            invalid_times += 1
            print(time, date)
            continue

    print(f"Filtered out {invalid_times} invalid time/date samples.")
    # Filter the dataframe to only include valid rows
    valid_data = data.loc[valid_indices]

    # Now everything stays aligned
    locs_and_temporal_encodings = np.array(locs_and_temporal_encodings)
    classes = np.array(classes)
    dates = valid_data['eventDate'].to_numpy()
    users = valid_data['gbifID'].to_numpy()
    inds = np.ones(len(valid_data))
    audio_files = valid_data['occurrenceID'].to_numpy()
    assert len(classes) == len(audio_files)
    print("Finished preparing inat metadata")

    return (
        locs_and_temporal_encodings,
        classes,
        users,
        dates,
        inds,
        audio_files
    )
if __name__ == "__main__":
    data = load_inat_data()
    print(data.head())
    print(data["eventDate"].head())
    for column in data.columns:
        print(column)
    scientific_names = data['species'].unique()
    print(f"Unique scientific names: {len(scientific_names)}")
    print(len(data))
    data = get_class_labels(data)
    print(data['perch_label'].head())
    print(len(data))
    st_context, classes, users, dates, inds, audio_files = prepare_for_spatiotemporal_encoder(data=data)
    print(st_context[:10])
    print(audio_files[:10])
    print(classes[:10])