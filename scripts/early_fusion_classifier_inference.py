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
from spatiotemporal_encoder import SpatiotemporalEncoder
import librosa
import json
SAVE_DIR = "/scratch/e1583377/pickled_birdset_embeds/"
if __name__ == '__main__':
    subset = "HSN"
    num_epochs = 1
    no_st_epochs = 4
    num_experts = 4
    torch_device = "cuda"
    prediction_save_path = "/home/svu/e1583377/ST-Geo-Perch/predictions/geospatial_mixup_last_balanced_mixture_of_" + str(num_experts) + "_experts_early_fusion_classifier_lr_0006_epoch_" + str(num_epochs + 1) + "_" + subset + ".pth"
    base_classifier = BalancedMoE_ST_EmbeddingClassifierHead(st_embedding_dim = 0, st_hidden_dim = 0, num_experts = num_experts)
    geo_aware_classifier = BalancedMoE_ST_EmbeddingClassifierHead(st_embedding_dim = 165, st_hidden_dim = 512, num_experts = num_experts)
    
    base_classifier_dict = torch.load("/scratch/e1583377/models/geospatial_mixup_ce_" + str(num_experts) + "_experts_epoch_" + str(no_st_epochs) + "_no_st.pth")
    geo_aware_classifier_dict = torch.load("/scratch/e1583377/models/geospatial_mixup_last_ce_" + str(num_experts) + "_experts_epoch_" + str(num_epochs) + ".pth")
    base_classifier.load_state_dict(base_classifier_dict)
    geo_aware_classifier.load_state_dict(geo_aware_classifier_dict)
    base_classifier = base_classifier.to(torch_device)
    geo_aware_classifier = geo_aware_classifier.to(torch_device)
    CLASS_LABELS_FILEPATH = "/home/svu/e1583377/Spatial_Perch_Transfer_Learning/assets/perch_v2_label_mapping.json"
    with open(CLASS_LABELS_FILEPATH) as f:
        perch_label_mapping = json.load(f)
    location_encoder = SphereMixScaleSpatialRelationEncoder(160, frequency_num = 32)
    st_enc = SpatiotemporalEncoder(location_encoder, loc_dim = 2, device = torch_device, temporal_encoding_dim = 5)
    print(list(perch_label_mapping.keys())[:10])
    predicted_ebird_codes = []
    geo_aware_predicted_ebird_codes = []
    ground_truth = []
    base_classifier.eval()
    geo_aware_classifier.eval()
    with open(SAVE_DIR + subset + ".pkl", "rb") as f:
        birdset_data = pickle.load(f)
    with torch.no_grad():
        for audio_embeds, st_context, labels in zip(birdset_data["embeddings"], birdset_data['st_context'], birdset_data["labels"]):
            st_encodings = st_enc(np.expand_dims(st_context.astype(np.float32), axis=1))
            st_encodings = st_encodings.squeeze(1)
            pred_logits = base_classifier(torch.from_numpy(audio_embeds).to(torch_device), st_encodings.to(torch_device))
            if type(pred_logits) == tuple:
                pred_logits = pred_logits[0]
            pred_indices = torch.argmax(pred_logits, dim = 1)
            predicted_ebird_codes += [perch_label_mapping[str(index.item())] for index in pred_indices]
            geo_aware_pred_logits = geo_aware_classifier(torch.from_numpy(audio_embeds).to(torch_device), st_encodings.to(torch_device))
            if type(geo_aware_pred_logits) == tuple:
                geo_aware_pred_logits = geo_aware_pred_logits[0]
            geo_pred_indices = torch.argmax(geo_aware_pred_logits, dim = 1)
            geo_aware_predicted_ebird_codes += [perch_label_mapping[str(index.item())] for index in geo_pred_indices]
            #ground_truth += [[perch_label_mapping[str(label)] for label in multilabel] for multilabel in labels]
    with open(prediction_save_path, "wb") as fw:
        pickle.dump({"perch":predicted_ebird_codes, "geo_aware":geo_aware_predicted_ebird_codes}, fw)
    print(predicted_ebird_codes[:10])
    print(ground_truth[:10])