import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Your remaining imports
import numpy as np
import kagglehub
from load_bioclip import load_bioclip
from SpatialRelationEncoder import SphereMixScaleSpatialRelationEncoder
from load_pickled_embeddings import load_st_audio_data
from typing import Dict, List
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import AvgMeter
from early_fusion_classifier_head import BalancedMoE_ST_EmbeddingClassifierHead, MoE_ST_EmbeddingClassifierHead, SpatiotemporalEncoder

# Sample data preparation
import pandas as pd
import numpy as np
class Coords_ST_AudioDataset(Dataset):
    def __init__(self, audio_embeddings: torch.Tensor, 
                 spatiotemporal_context: torch.Tensor, spatiotemporal_encodings, labels):
        self.audio_embeddings = audio_embeddings
        self.spatiotemporal_context = spatiotemporal_context
        # Keep only rows without NaNs
        self.spatiotemporal_encodings = spatiotemporal_encodings
        st_mask = ~torch.isnan(spatiotemporal_encodings).any(axis=1).bool()
        audio_mask = ~torch.isnan(audio_embeddings).any(dim=1).bool()
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        label_mask = ~torch.isnan(labels_tensor).bool()
        mask = st_mask & audio_mask & label_mask
        self.valid_indices = torch.where(mask)[0].cpu().numpy()
        self.labels = labels
        
        assert len(audio_embeddings) == len(spatiotemporal_context)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        idx = self.valid_indices[idx]
        return (self.audio_embeddings[idx], self.spatiotemporal_context[idx], self.spatiotemporal_encodings[idx], self.labels[idx])
def simple_collate_fn(batch):
    audio_embeddings = torch.stack([item[0] for item in batch])
    spatiotemporal_context = torch.stack([torch.from_numpy(item[1]) for item in batch])
    spatiotemporal_encodings = torch.stack([item[2] for item in batch])
    labels = torch.stack([item[3] for item in batch])
    
    return audio_embeddings, spatiotemporal_context, spatiotemporal_encodings, labels
# Mock data - replace with your actual data
data = {
    'lat': np.random.uniform(-90, 90, 1000),
    'lon': np.random.uniform(-180, 180, 1000),
    'expert_assignments': np.random.choice([0, 1, 2, 3], 1000),  # Your gating outputs
    'species': np.random.choice(['sparrow', 'robin', 'eagle'], 1000)
}
df = pd.DataFrame(data)
if __name__ == '__main__':
    print("Starting script")
    num_epochs = 3
    num_experts = 10
    classifier_head = BalancedMoE_ST_EmbeddingClassifierHead(st_embedding_dim = 165, st_hidden_dim = 512, num_experts = num_experts)
    geo_aware_classifier_dict = torch.load("/scratch/e1583377/models/geospatial_mixup_ce_" + str(num_experts) + "_experts_epoch_" + str(num_epochs) + ".pth")
    classifier_head.load_state_dict(geo_aware_classifier_dict)
    DATA_DIR = "/scratch/e1583377/pickled_audio_embeds_3_random_windows_corrected/"
    audio_embeddings, spatiotemporal_context, labels = load_st_audio_data(DATA_DIR)
    print(audio_embeddings.shape, spatiotemporal_context.shape, len(labels))
    print("num_labels", labels.unique().numel())
    # Create dataset
    location_encoder = SphereMixScaleSpatialRelationEncoder(160, frequency_num = 32)
    st_enc = SpatiotemporalEncoder(location_encoder, loc_dim = 2, device = "cuda", temporal_encoding_dim = 5)
    spatiotemporal_encodings = st_enc(spatiotemporal_context).squeeze(1).cpu()
    print(spatiotemporal_encodings.shape)
    device = "cuda"
    classifier_head = classifier_head.to(device)
    
    dataset = Coords_ST_AudioDataset(
        audio_embeddings=audio_embeddings,
        spatiotemporal_context=spatiotemporal_context,
        spatiotemporal_encodings=spatiotemporal_encodings,
        labels = labels.cpu()
    )
    batch_size = 8192
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True,  # Faster GPU transfer
        collate_fn=simple_collate_fn
    )

    # Plot with Plotly (easiest)
    import plotly.express as px
    for audio_embeddings, spatiotemporal_context, spatiotemporal_encodings, labels in dataloader:
        chosen_experts = classifier_head.choose_experts(audio_embeddings.to(device), spatiotemporal_encodings.to(device))
        spatiotemporal_context = spatiotemporal_context.squeeze(1)
        df = pd.DataFrame({
            'lat': spatiotemporal_context[:,1].cpu().numpy(),
            'lon': spatiotemporal_context[:,0].cpu().numpy(),
            'expert_assignments': chosen_experts.cpu().numpy(),
            'species': labels.cpu().numpy() #can map these to strings if I want later
        })
        print("created data frame, plotting")
        fig = px.scatter_geo(df, lat='lat', lon='lon', color='expert_assignments',
                     hover_data=['species'], title='MoE Expert Regionality')
        fig.write_html("expert_regionality_st.html")
        break