

import torch
import torch.nn as nn
import torch.optim as optim

# Your remaining imports
import numpy as np

class SpatiotemporalEncoder(nn.Module):
    # Simply concatenates temporal encodings to location encodings from a location encoder (e.g. SphereMixSpatialRelationEncoder
    def __init__(self, spa_enc, loc_dim = 2, device = 'cuda', temporal_encoding_dim = 5):
        super(SpatiotemporalEncoder, self).__init__()
        self.spa_enc = spa_enc
        self.loc_dim = loc_dim
        self.device = device
        self.temporal_encoding_dim = temporal_encoding_dim
    def forward(self, locations_and_temporal_encodings):
        # Split spatial vs temporal inputs
        # Encode spatial features
        if locations_and_temporal_encodings.shape[2] == 1:
            #print(loc_inputs.shape)
            locations_and_temporal_encodings = locations_and_temporal_encodings.squeeze(1)
        loc_inputs = locations_and_temporal_encodings[:, :, :self.loc_dim]
        temporal_inputs = locations_and_temporal_encodings[:, :, self.loc_dim:]

        loc_embeds = self.spa_enc(loc_inputs).to(self.device)

        # Make sure temporal features are on the same device
        temporal_embeds = torch.from_numpy(temporal_inputs).to(self.device)

        # Concatenate along feature dimension
        return torch.cat((loc_embeds, temporal_embeds), dim=-1)