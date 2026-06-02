import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import tensorflow as tf
import tensorflow_hub as hub
tf.experimental.numpy.experimental_enable_numpy_behavior()
import pickle
import os
import math
import kagglehub
from pathlib import Path
import soundfile as sf
from collections import defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# save path (Set this yourself)
SAVE_DIR = "/path/to/pickled_wabad_embeds/"
MIN_YEAR, MAX_YEAR = 2000, 2025
SAMPLE_RATE = 32000
CHUNK_DURATION = 5  # seconds
CHUNK_SAMPLES = CHUNK_DURATION * SAMPLE_RATE

def load_perch_gpu_model(url='https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2/2'):
    return hub.load(url)

def load_perch_labels(label_path="metadata\perch_v2_label_mapping.json"):
    """Load Perch 2.0 labels and create mapping from species names to indices"""
    df = pd.read_csv(label_path)
    species_list = df.iloc[:, 0].tolist()
    species_to_idx = {species: idx for idx, species in enumerate(species_list)}
    return species_list, species_to_idx

def load_wabad_metadata(base_path="/scratch/e1583377/wabad/"):
    """Load annotations and updated metadata with coordinates - Fixed for new format"""
    
    # Load annotations - FORCE COMMA SEPARATOR since we know the format
    annotations_path = base_path + "Pooled annotations.csv"
    
    print("Loading Pooled annotations.csv...")
    
    # Try reading with different encodings but ALWAYS with comma separator
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    annotations_df = None
    
    for encoding in encodings:
        try:
            # Read CSV with comma separator - no quoting issues
            annotations_df = pd.read_csv(annotations_path, sep=',', encoding=encoding)
            print(f"? Successfully loaded annotations with encoding: {encoding}")
            
            # Check if we have multiple columns (should be at least 7 columns)
            if len(annotations_df.columns) >= 7:
                print(f"  Found {len(annotations_df.columns)} columns: {annotations_df.columns.tolist()}")
                break
            else:
                print(f"  WARNING: Only {len(annotations_df.columns)} columns found, expected at least 7")
                annotations_df = None
                
        except Exception as e:
            print(f"  Failed with encoding {encoding}: {e}")
            continue
    
    if annotations_df is None:
        # Last resort: try reading with Python engine
        try:
            annotations_df = pd.read_csv(annotations_path, encoding='latin-1', 
                                        engine='python', on_bad_lines='skip')
            print("? Loaded annotations with fallback method")
        except Exception as e:
            raise ValueError(f"Could not load annotations file: {e}")
    
    # Print first few rows to verify
    print(f"\nFirst 3 rows of annotations:")
    print(annotations_df.head(3))
    print(f"\nAnnotations shape: {annotations_df.shape}")
    
    # The columns should be: Species, Site, Recording, Begin_Time_(s), End_Time_(s), Low_Freq_(Hz), High_Freq_(Hz)
    # Check if we need to rename columns
    expected_columns = ['Species', 'Site', 'Recording', 'Begin_Time_(s)', 'End_Time_(s)', 
                       'Low_Freq_(Hz)', 'High_Freq_(Hz)']
    
    # If columns don't match expected, try to infer them
    if len(annotations_df.columns) == 7:
        # Check if the columns are already in the expected format
        if annotations_df.columns.tolist() != expected_columns:
            print(f"\nRenaming columns to standard names...")
            # Assume the columns are in the expected order
            annotations_df.columns = expected_columns
            print(f"New columns: {annotations_df.columns.tolist()}")
    
    # Create friendly column names with spaces for compatibility with existing code
    if 'Begin_Time_(s)' in annotations_df.columns:
        annotations_df['Begin Time (s)'] = annotations_df['Begin_Time_(s)']
    if 'End_Time_(s)' in annotations_df.columns:
        annotations_df['End Time (s)'] = annotations_df['End_Time_(s)']
    if 'Low_Freq_(Hz)' in annotations_df.columns:
        annotations_df['Low Freq (Hz)'] = annotations_df['Low_Freq_(Hz)']
    if 'High_Freq_(Hz)' in annotations_df.columns:
        annotations_df['High Freq (Hz)'] = annotations_df['High_Freq_(Hz)']
    
    # Ensure we have the required columns
    required_columns = ['Species', 'Site', 'Recording', 'Begin Time (s)', 'End Time (s)']
    missing_columns = [col for col in required_columns if col not in annotations_df.columns]
    if missing_columns:
        print(f"ERROR: Missing required columns: {missing_columns}")
        print(f"Available columns: {annotations_df.columns.tolist()}")
        raise ValueError(f"Cannot proceed without columns: {missing_columns}")
    
    # Fix numeric columns (replace comma with decimal if needed)
    for col in ['Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)']:
        if col in annotations_df.columns:
            try:
                # Check if the column contains strings with commas
                if annotations_df[col].dtype == 'object':
                    annotations_df[col] = annotations_df[col].astype(str).str.replace(',', '.').astype(float)
                else:
                    annotations_df[col] = annotations_df[col].astype(float)
                print(f"? Converted {col} to float")
            except Exception as e:
                print(f"Warning: Could not convert column {col} to float: {e}")
    
    # Load metadata with coordinates
    metadata_path = base_path + "Metadata.csv"
    metadata_df = None
    
    print("\nLoading Metadata.csv...")
    
    # Try reading metadata with comma separator
    for encoding in encodings:
        try:
            metadata_df = pd.read_csv(metadata_path, sep=',', encoding=encoding)
            print(f"? Successfully loaded metadata with encoding: {encoding}")
            break
        except Exception as e:
            print(f"  Failed with encoding {encoding}: {e}")
            continue
    
    if metadata_df is None:
        # Try reading with Python engine
        try:
            metadata_df = pd.read_csv(metadata_path, encoding='latin-1', 
                                     engine='python', on_bad_lines='skip')
            print("? Loaded metadata with fallback method")
        except Exception as e:
            print(f"Error parsing metadata: {e}")
            raise
    
    if metadata_df is None:
        raise ValueError("Could not load metadata file")
    
    # Clean column names (remove quotes and extra spaces)
    metadata_df.columns = metadata_df.columns.str.strip().str.replace('"', '')
    
    print(f"\nMetadata columns: {metadata_df.columns.tolist()}")
    print(f"First 3 rows of metadata:\n{metadata_df.head(3)}")
    
    # Find site and coordinate columns
    site_col = None
    lat_col = None
    lon_col = None
    
    # Look for site column
    for col in metadata_df.columns:
        col_lower = col.lower()
        if 'site' in col_lower:
            site_col = col
            break
    
    # If not found, try the first column
    if site_col is None and len(metadata_df.columns) > 0:
        site_col = metadata_df.columns[0]
        print(f"Using first column as site ID: {site_col}")
    
    # Look for latitude and longitude columns
    for col in metadata_df.columns:
        col_lower = col.lower()
        if 'latitude' in col_lower or col_lower == 'lat':
            lat_col = col
        elif 'longitude' in col_lower or col_lower == 'lon' or 'long' in col_lower:
            lon_col = col
    
    # If still not found, try to find numeric columns
    if lat_col is None or lon_col is None:
        numeric_cols = []
        for col in metadata_df.columns:
            try:
                # Check if column is numeric
                if pd.api.types.is_numeric_dtype(metadata_df[col]):
                    numeric_cols.append(col)
            except:
                continue
        
        if len(numeric_cols) >= 2:
            if lat_col is None:
                lat_col = numeric_cols[0]
            if lon_col is None:
                lon_col = numeric_cols[1]
            print(f"Using numeric columns as coordinates: {lat_col}, {lon_col}")
    
    print(f"\nUsing columns - Site: {site_col}, Latitude: {lat_col}, Longitude: {lon_col}")
    
    # Create mapping from Site ID to coordinates
    site_to_coords = {}
    
    for idx, row in metadata_df.iterrows():
        try:
            # Get site ID
            if site_col:
                site_id = str(row[site_col]).strip()
            else:
                continue
            
            # Skip if site_id is NaN or empty
            if pd.isna(site_id) or site_id == 'nan' or site_id == '':
                continue
            
            # Get coordinates if available
            if lat_col and lon_col:
                lat = row[lat_col]
                lon = row[lon_col]
                
                # Convert to float if possible
                if not pd.isna(lat) and not pd.isna(lon):
                    try:
                        # Handle string values
                        if isinstance(lat, str):
                            lat = lat.strip().strip('"')
                        if isinstance(lon, str):
                            lon = lon.strip().strip('"')
                        
                        lat = float(lat)
                        lon = float(lon)
                        site_to_coords[site_id] = (lat, lon)
                        
                        # Also store cleaned version (remove asterisks)
                        clean_id = site_id.replace('*', '').strip()
                        if clean_id != site_id:
                            site_to_coords[clean_id] = (lat, lon)
                        
                        # Store uppercase version for matching
                        site_to_coords[site_id.upper()] = (lat, lon)
                        
                    except Exception as e:
                        print(f"Warning: Could not convert coordinates to float for site {site_id}: lat={lat}, lon={lon}, error={e}")
            else:
                # Use dummy coordinates if none available
                site_to_coords[site_id] = (0.0, 0.0)
                    
        except Exception as e:
            print(f"Warning: Could not parse row {idx}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Loaded coordinates for {len(site_to_coords)} site variants")
    if len(site_to_coords) > 0:
        print(f"\nSite ID to coordinate mapping (first 10):")
        for site_id, (lat, lon) in list(site_to_coords.items())[:10]:
            print(f"  {site_id} -> ({lat}, {lon})")
    
    # Get unique sites from annotations
    unique_annotation_sites = annotations_df['Site'].unique()
    print(f"\n{'='*60}")
    print(f"Unique recording sites in annotations ({len(unique_annotation_sites)} total):")
    print(f"First 20 sites: {unique_annotation_sites[:20]}")
    
    # Verify that all annotation sites have coordinates
    print(f"\n{'='*60}")
    print("Verifying coordinates for all annotation sites...")
    missing_sites = []
    matched_sites = []
    
    for site in unique_annotation_sites:
        if site in site_to_coords:
            matched_sites.append(site)
        else:
            # Try cleaning the site ID (remove asterisks, spaces)
            clean_site = site.replace('*', '').strip()
            if clean_site in site_to_coords:
                matched_sites.append(site)
                site_to_coords[site] = site_to_coords[clean_site]
            else:
                # Try uppercase
                upper_site = site.upper()
                if upper_site in site_to_coords:
                    matched_sites.append(site)
                    site_to_coords[site] = site_to_coords[upper_site]
                else:
                    missing_sites.append(site)
    
    if missing_sites:
        print(f"\nWARNING: {len(missing_sites)} sites from annotations missing in metadata!")
        print(f"First 20 missing sites: {missing_sites[:20]}")
        print("\nAvailable metadata site IDs (first 20):")
        available_sites = [s for s in site_to_coords.keys() if len(s) <= 20]
        print(f"{list(set(available_sites))[:20]}")
        print("\nUsing dummy coordinates for missing sites...")
        for site in missing_sites:
            site_to_coords[site] = (0.0, 0.0)
        print(f"Added dummy coordinates for {len(missing_sites)} missing sites")
    else:
        print(f"? All {len(unique_annotation_sites)} annotation sites have matching coordinates!")
        if matched_sites:
            print(f"  Matched sites: {matched_sites[:10]}...")
    
    print(f"\n{'='*60}")
    print("FINAL CHECK:")
    print(f"Annotations shape: {annotations_df.shape}")
    print(f"Annotations columns: {annotations_df.columns.tolist()}")
    print(f"Has 'Recording' column: {'Recording' in annotations_df.columns}")
    print(f"Sample site: {annotations_df['Site'].iloc[0]}")
    print(f"Sample recording: {annotations_df['Recording'].iloc[0]}")
    print(f"Sample species: {annotations_df['Species'].iloc[0]}")
    print(f"{'='*60}")
    
    return annotations_df, metadata_df, site_to_coords

def get_spatial_context(site_id, site_to_coords, annotations_df):
    """Get spatial context (coordinates) for a given site"""
    # Try exact match first
    if site_id in site_to_coords:
        lat, lon = site_to_coords[site_id]
        return np.array([lon, lat])
    
    # Try removing asterisks or special characters
    clean_id = site_id.replace('*', '').replace('?', '').strip()
    if clean_id in site_to_coords:
        lat, lon = site_to_coords[clean_id]
        return np.array([lon, lat])
    
    # Try uppercase
    upper_id = site_id.upper()
    if upper_id in site_to_coords:
        lat, lon = site_to_coords[upper_id]
        return np.array([lon, lat])
    
    # Try to find if this site ID appears in metadata
    for metadata_site in site_to_coords.keys():
        if site_id in metadata_site or metadata_site in site_id:
            lat, lon = site_to_coords[metadata_site]
            return np.array([lon, lat])
    
    # If still not found, use dummy coordinates
    print(f"Warning: No coordinates found for site '{site_id}', using (0,0)")
    return np.array([0, 0])

def extract_timestamp_from_filename(filename):
    """
    Extract datetime from filename format: {subset}_{YYYYMMDD}_{HHMMSS}.wav
    Example: ARD_20211027_072000.wav -> 2021-10-27 07:20:00
    """
    try:
        basename = Path(filename).stem
        parts = basename.split('_')
        
        if len(parts) >= 3:
            date_str = parts[-2]
            time_str = parts[-1]
            
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            hour = int(time_str[:2])
            minute = int(time_str[2:4])
            second = int(time_str[4:6])
            
            return datetime(year, month, day, hour, minute, second)
    except Exception as e:
        pass
    
    return None

def encode_time_from_datetime(dt, min_year=MIN_YEAR, max_year=MAX_YEAR):
    """Create cyclic encodings for hour and day, plus normalized year"""
    if dt is None:
        return np.array([0, 0, 0, 0, 0.5])
    
    hour_float = dt.hour + dt.minute/60 + dt.second/3600
    hour_sin = math.sin(2 * math.pi * hour_float / 24)
    hour_cos = math.cos(2 * math.pi * hour_float / 24)
    
    day_of_year = dt.timetuple().tm_yday
    day_sin = math.sin(2 * math.pi * day_of_year / 365)
    day_cos = math.cos(2 * math.pi * day_of_year / 365)
    
    year_norm = (dt.year - min_year) / (max_year - min_year)
    
    return np.array([hour_sin, hour_cos, day_sin, day_cos, year_norm])

def chunk_audio_and_get_labels(audio_array, audio_path, annotations_for_file, 
                               species_to_idx, sr=SAMPLE_RATE):
    """Split audio into 5-second chunks and get labels for each chunk"""
    audio_length = len(audio_array)
    chunks = []
    labels_list = []
    chunk_timestamps = []
    
    filename = Path(audio_path).name
    base_timestamp = extract_timestamp_from_filename(filename)
    
    num_chunks = math.ceil(audio_length / CHUNK_SAMPLES)
    
    for i in range(num_chunks):
        start_sample = i * CHUNK_SAMPLES
        end_sample = min(start_sample + CHUNK_SAMPLES, audio_length)
        
        chunk = audio_array[start_sample:end_sample]
        
        if len(chunk) < CHUNK_SAMPLES:
            padded_chunk = np.zeros(CHUNK_SAMPLES)
            start_pad = (CHUNK_SAMPLES - len(chunk)) // 2
            padded_chunk[start_pad:start_pad + len(chunk)] = chunk
            chunk = padded_chunk
        
        start_time = start_sample / sr
        end_time = end_sample / sr
        
        labels = np.zeros(len(species_to_idx))
        for _, ann in annotations_for_file.iterrows():
            # Get the time columns (with spaces)
            ann_start = ann['Begin Time (s)']
            ann_end = ann['End Time (s)']
            species = ann['Species'].strip()
            
            overlap_start = max(start_time, ann_start)
            overlap_end = min(end_time, ann_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            if overlap_duration >= 0.1 and species in species_to_idx:
                labels[species_to_idx[species]] = 1
        
        if labels.sum() > 0:
            chunks.append(chunk)
            labels_list.append(labels)
            
            if base_timestamp:
                chunk_timestamp = base_timestamp + pd.Timedelta(seconds=start_time)
                chunk_timestamps.append(encode_time_from_datetime(chunk_timestamp))
            else:
                chunk_timestamps.append(np.array([0, 0, 0, 0, 0.5]))
    
    return chunks, labels_list, chunk_timestamps

def process_wabad_subset(base_path, species_to_idx, site_to_coords, 
                         annotations_df, subset_name="ARD"):
    """Process a specific subset (ARD, etc.) of WABAD"""
    subset_annotations = annotations_df[annotations_df['Site'] == subset_name]
    
    if len(subset_annotations) == 0:
        print(f"No annotations found for subset {subset_name}")
        return [], [], []
    
    # Group by recording file
    if 'Recording' not in subset_annotations.columns:
        print(f"ERROR: 'Recording' column not found. Available: {subset_annotations.columns.tolist()}")
        return [], [], []
    
    grouped_annotations = subset_annotations.groupby('Recording')
    
    all_chunks = []
    all_labels = []
    all_spatiotemporal = []
    
    audio_base = os.path.join(base_path, subset_name, "Recordings")
    
    if not os.path.exists(audio_base):
        print(f"Warning: Audio directory not found: {audio_base}")
        return [], [], []
    
    spatial_context = get_spatial_context(subset_name, site_to_coords, annotations_df)
    
    for recording_file, annotations in grouped_annotations:
        audio_path = os.path.join(audio_base, recording_file)
        
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found: {audio_path}")
            continue
        
        try:
            audio_array, sr = sf.read(audio_path)
            
            if sr != SAMPLE_RATE:
                from scipy import signal
                new_length = int(len(audio_array) * SAMPLE_RATE / sr)
                audio_array = signal.resample(audio_array, new_length)
                sr = SAMPLE_RATE
            
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            chunks, labels, timestamps = chunk_audio_and_get_labels(
                audio_array, audio_path, annotations, species_to_idx, sr
            )
            
            for timestamp in timestamps:
                spatiotemporal = np.concatenate([spatial_context, timestamp])
                all_spatiotemporal.append(spatiotemporal)
            
            all_chunks.extend(chunks)
            all_labels.extend(labels)
            
            print(f"Processed {recording_file}: {len(chunks)} labeled chunks")
            
        except Exception as e:
            print(f"Error processing {recording_file}: {e}")
            continue
    
    return all_chunks, all_labels, all_spatiotemporal

class WabadTorchDataset(torch.utils.data.Dataset):
    def __init__(self, chunks, labels, spatiotemporal_context):
        self.chunks = chunks
        self.labels = labels
        self.spatiotemporal_context = spatiotemporal_context
        
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        audio = self.chunks[idx].astype(np.float32)
        max_abs = np.max(np.abs(audio))
        if max_abs > 0:
            audio = audio / max_abs * 0.25
        
        return audio, self.spatiotemporal_context[idx], self.labels[idx]

def custom_collate(batch):
    audio_batch = np.array([item[0] for item in batch])
    st_batch = np.array([item[1] for item in batch])
    label_batch = np.array([item[2] for item in batch])
    return audio_batch, st_batch, label_batch

if __name__ == '__main__':
    base_path = "/scratch/e1583377/wabad/"
    
    print("Loading Perch labels...")
    species_list, species_to_idx = load_perch_labels()
    print(f"Loaded {len(species_list)} species")
    
    print("\nLoading WABAD data...")
    annotations_df, metadata_df, site_to_coords = load_wabad_metadata(base_path)
    
    print(f"\nTotal annotations: {len(annotations_df)}")
    print(f"Unique recording sites: {annotations_df['Site'].unique()[:20]}...")
    
    print("\nLoading Perch model...")
    perch_model = load_perch_gpu_model()
    
    subsets = annotations_df['Site'].unique()
    print(f"\nProcessing {len(subsets)} subsets...")
    
    for subset in subsets:
        print(f"\n{'='*50}")
        print(f"Processing {subset} subset...")
        print(f"{'='*50}")
        
        chunks, labels, spatiotemporal = process_wabad_subset(
            base_path, species_to_idx, site_to_coords, annotations_df, subset
        )
        
        if len(chunks) == 0:
            print(f"No labeled chunks processed for {subset}, skipping...")
            continue
        
        print(f"\nTotal labeled chunks: {len(chunks)}")
        print(f"Average labels per chunk: {np.array([l.sum() for l in labels]).mean():.2f}")
        print(f"Max labels in a chunk: {np.array([l.sum() for l in labels]).max()}")
        
        dataset = WabadTorchDataset(chunks, labels, spatiotemporal)
        dataloader = DataLoader(dataset, batch_size=128, 
                               collate_fn=custom_collate, shuffle=False)
        
        all_embeddings = []
        all_labels = []
        all_spatiotemporal = []
        
        print("\nExtracting Perch embeddings...")
        for batch_idx, (audio_array, st_array, label_array) in enumerate(dataloader):
            audio_array = audio_array.astype(np.float32)
            perch_output = perch_model.signatures['serving_default'](
                inputs=audio_array)['embedding'].numpy()
            all_embeddings.append(perch_output)
            all_spatiotemporal.append(st_array)
            all_labels.append(label_array)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches")
        
        os.makedirs(SAVE_DIR, exist_ok=True)
        output_path = os.path.join(SAVE_DIR, f"{subset}_wabad.pkl")
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        st_context = np.concatenate(all_spatiotemporal, axis=0)
        labels_array = np.concatenate(all_labels, axis=0)
        
        with open(output_path, "wb") as f:
            pickle.dump({
                'embeddings': embeddings,
                'labels': labels_array,
                'st_context': st_context,
                'subset': subset,
                'num_samples': len(embeddings)
            }, f)
        
        print(f"\nSaved {len(embeddings)} embeddings to {output_path}")
    
    print("\nAll subsets processed successfully!")