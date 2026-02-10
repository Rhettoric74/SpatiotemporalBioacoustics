import os
import numpy as np
import torch
import pickle
import json
import random
import pandas as pd

EMBEDDINGS_DIR = "/scratch/e1583377/pickled_audio_embeds_peak_select_finally_corrected/"
MIXUP_EMBEDDINGS_DIR = "/scratch/e1583377/pickled_audio_embeds_mixed/"
LABEL_MAPPING_PATH = "/home/svu/e1583377/Spatial_Perch_Transfer_Learning/assets/perch_v2_label_mapping.json"
TEXT_DESCRIPTIONS_PATH = "/home/svu/e1583377/Spatial_Perch_Transfer_Learning/assets/species_descriptions.json"
SPECIES_NAMES_PATH = "/home/svu/e1583377/Spatial_Perch_Transfer_Learning/assets/perch_v2_labels.csv"

with open(LABEL_MAPPING_PATH) as f:
    label_mapping = json.load(f)
    
with open(TEXT_DESCRIPTIONS_PATH) as ft:
    species_descriptions_mapping = json.load(ft)
with open(SPECIES_NAMES_PATH) as fs:
    species_names = pd.read_csv(fs)['inat2024_fsd50k']
    species_names = [species_name.replace("'", "").replace('"', '') for species_name in species_names]
def generate_description_from_ebird(species_name, prelude = "Animalia Chordata Aves "):
    description_dict = species_descriptions_mapping[species_name]
    full_sci_name = prelude + random.choice(description_dict['order']) + " " + random.choice(description_dict['family']) + " " + random.choice(description_dict['scientific_name'])
    if description_dict['common_name'] != description_dict['scientific_name']:
        full_sci_name += " (" + random.choice(description_dict['common_name']) + ")"
    return full_sci_name
def generate_text_descriptions(labels):
    descriptions = []
    for label in labels:
        ebird_code = label_mapping[str(label.item())]
        species_name = species_names[label.item()]
        try:
            description = generate_description_from_ebird(species_name)
        except:
            print(species_name, "has no corresponding description")
            description = "unknown"
        descriptions.append(description)
    return descriptions
def load_all_embeddings(embeddings_dir = EMBEDDINGS_DIR):
    batches_dict = {}
    for embedding_file in os.listdir(embeddings_dir):
        with open(embeddings_dir + embedding_file, "rb") as f:
            batch_data = pickle.load(f)
            if 'spatiotemporal_contexts' in batch_data:
                if "ids" in batch_data:
                    non_nan_indices = ~torch.isnan(batch_data['audio_embeddings']).any(dim=1)
                    batch_data['audio_embeddings'] = batch_data['audio_embeddings'][non_nan_indices]
                    batch_data['spatiotemporal_embeddings'] = batch_data['spatiotemporal_embeddings'][non_nan_indices]
                    batch_data['labels'] = batch_data['labels'][non_nan_indices]
                    # note: ids don't quite match because I forgot to slice valid indices when generating.
                    # can recompute them later if needed, if so I can uncomment this line
                    #batch_data['ids'] = batch_data['ids'][non_nan_indices]
                    batch_data['spatiotemporal_contexts'] = batch_data['spatiotemporal_contexts'][non_nan_indices.cpu().numpy()]
                    # add text descriptions
                    batch_data['text_descriptions'] = generate_text_descriptions(batch_data['labels'])
                    
                    batches_dict[embedding_file] = batch_data
    return batches_dict

def load_st_audio_data(embeddings_dir = EMBEDDINGS_DIR):
    data_dict = load_all_embeddings(embeddings_dir)
    data_list = list(data_dict.values())
    audio_stacked = torch.cat([item['audio_embeddings'] for item in data_list], dim=0)
    context_stacked = np.concatenate([item['spatiotemporal_contexts'] for item in data_list], axis=0)
    labels_list = []
    for item in data_list:
        labels_list += item['labels']
    return audio_stacked, context_stacked, torch.tensor(labels_list, dtype=torch.long)
    


def reconstruct_component_contexts(batch):
    """Reconstruct component_contexts from optimized storage format."""
    if 'component_contexts' in batch:
        # Already reconstructed
        return batch['component_contexts']
    
    if 'component_contexts_flat' not in batch:
        raise ValueError("No component contexts data found")
    
    # Reconstruct from flattened format
    flat_contexts = batch['component_contexts_flat']
    lengths = batch['component_contexts_lengths']
    shapes = batch['component_contexts_shapes']
    
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
    
    return component_contexts


def load_mixup_embeddings(embeddings_dir = MIXUP_EMBEDDINGS_DIR):
    """
    Simplified loader for mixup data.
    Assumes all files follow the format from process_mixing_batch_with_full_context()
    """
    all_audio_embeds = []
    all_st_embeds = []
    all_component_contexts = []
    all_component_weights = []
    all_component_indices = []
    all_labels = []
    
    # Get all pickle files
    pkl_files = [f for f in os.listdir(embeddings_dir) if f.endswith('.pkl')]
    #print(f"Found {len(pkl_files)} pickle files")
    
    for filename in pkl_files:
        filepath = os.path.join(embeddings_dir, filename)
        
        try:
            with open(filepath, 'rb') as f:
                batch = pickle.load(f)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
        
        # Check if it's mixup data (has component_contexts or flattened version)
        has_contexts = ('component_contexts' in batch or 
                       'component_contexts_flat' in batch or
                       'labels_sparse' in batch)
        
        if not has_contexts:
            print(f"Skipping {filename}: not mixup data")
            continue
        
        #print(f"Loading {filename}: {len(batch.get('audio_embeddings', []))} samples")
        
        # Reconstruct component_contexts if needed
        if 'component_contexts_flat' in batch:
            batch['component_contexts'] = reconstruct_component_contexts(batch)
        
        # Convert to tensors if needed
        audio_embeds = batch['audio_embeddings']
        st_embeds = batch['spatiotemporal_embeddings']
        
        if isinstance(audio_embeds, np.ndarray):
            audio_embeds = torch.from_numpy(audio_embeds)
        if isinstance(st_embeds, np.ndarray):
            st_embeds = torch.from_numpy(st_embeds)
        
        # Remove NaN values
        nan_mask = ~torch.isnan(audio_embeds).any(dim=1)
        if not nan_mask.all():
            audio_embeds = audio_embeds[nan_mask]
            st_embeds = st_embeds[nan_mask]
            
            # Get indices of kept samples
            keep_indices = torch.where(nan_mask)[0].tolist()
            
            # Filter lists using list comprehension
            labels_sparse = [batch['labels_sparse'][i] for i in keep_indices]
            component_contexts = [batch['component_contexts'][i] for i in keep_indices]
            component_weights = [batch['component_weights'][i] for i in keep_indices]
            component_indices = [batch['component_indices'][i] for i in keep_indices]
        else:
            labels_sparse = batch['labels_sparse']
            component_contexts = batch['component_contexts']
            component_weights = batch['component_weights']
            component_indices = batch['component_indices']
        
        # Append to lists
        all_audio_embeds.append(audio_embeds)
        all_st_embeds.append(st_embeds)
        all_component_contexts.extend(component_contexts)
        all_component_weights.extend(component_weights)
        all_component_indices.extend(component_indices)
        all_labels.extend(labels_sparse)
    
    # Combine all batches
    if all_audio_embeds:
        combined_data = {
            'audio_embeddings': torch.cat(all_audio_embeds, dim=0),
            'spatiotemporal_embeddings': torch.cat(all_st_embeds, dim=0),
            'labels_sparse': all_labels,
            'component_contexts': all_component_contexts,
            'component_weights': all_component_weights,
            'component_indices': all_component_indices
        }
        
        #print(f"\nLoaded {len(combined_data['audio_embeddings'])} total mixup samples")
        #print(f"Each sample has {np.mean([len(ctx) for ctx in combined_data['component_contexts']]):.2f} component contexts on average")
        
        return combined_data
    else:
        print("No mixup data found")
        return {}






# Test the loading
if __name__ == "__main__":
    DATA_DIR = "/scratch/e1583377/pickled_audio_embeds_mixed/"
    
    # Load mixup data
    mixup_data = load_mixup_embeddings(DATA_DIR)
    
    if mixup_data:
        print(f"\nLoaded mixup data keys: {mixup_data.keys()}")
        print(f"Audio embeddings shape: {mixup_data['audio_embeddings'].shape}")
        print(f"Number of samples: {len(mixup_data['labels_sparse'])}")
        
        # Test reconstruction
        print(f"\nFirst sample info:")
        print(f"  Number of contexts: {len(mixup_data['component_contexts'][0])}")
        print(f"  Number of weights: {len(mixup_data['component_weights'][0])}")
        print(f"  Labels: {mixup_data['labels_sparse'][0]}")
        print(f"  Context shape: {mixup_data['component_contexts'][0][0].shape if mixup_data['component_contexts'][0] else 'No contexts'}")





    