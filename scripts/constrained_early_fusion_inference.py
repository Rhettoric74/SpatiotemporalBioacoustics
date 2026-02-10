import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
import numpy as np
tf.experimental.numpy.experimental_enable_numpy_behavior()
print("TF GPUs:", tf.config.list_physical_devices('GPU'))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
print("Torch devices:", torch.cuda.device_count(), torch.cuda.get_device_name(0))
torch_device = torch.device("cuda:0")
from SpatialRelationEncoder import SphereMixScaleSpatialRelationEncoder
from load_pickled_embeddings import load_st_audio_data, generate_text_descriptions
from typing import Dict, List
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from early_fusion_classifier_head import *
from spatiotemporal_encoder import SpatiotemporalEncoder
import librosa
import json
from birdset_preparation import load_birdset_data
SAVE_DIR = "/scratch/e1583377/pickled_birdset_embeds/"
if __name__ == '__main__':
    subset = "SSW"
    num_epochs = 1
    no_st_epochs = 4
    num_experts = 4
    torch_device = "cuda"
    prediction_save_path = "/home/svu/e1583377/ST-Geo-Perch/predictions/constrained_geospatial_mixup_balanced_mixture_of_" + str(num_experts) + "_experts_early_fusion_classifier_lr_0006_epoch_" + str(num_epochs + 1) + "_" + subset + ".pth"
    base_classifier = BalancedMoE_ST_EmbeddingClassifierHead(st_embedding_dim = 0, st_hidden_dim = 0, num_experts = num_experts)
    geo_aware_classifier = BalancedMoE_ST_EmbeddingClassifierHead(st_embedding_dim = 165, st_hidden_dim = 512, num_experts = num_experts)
    
    base_classifier_dict = torch.load(MODEL_SAVE_DIR + "/geospatial_mixup_ce_" + str(num_experts) + "_experts_epoch_" + str(no_st_epochs) + "_no_st.pth")
    geo_aware_classifier_dict = torch.load(MODEL_SAVE_DIR + "/geospatial_mixup_ce_" + str(num_experts) + "_experts_epoch_" + str(num_epochs) + ".pth")
    base_classifier.load_state_dict(base_classifier_dict)
    geo_aware_classifier.load_state_dict(geo_aware_classifier_dict)
    base_classifier = base_classifier.to(torch_device)
    geo_aware_classifier = geo_aware_classifier.to(torch_device)
    birdset_test = load_birdset_data(subset)
    subset_label_mapping = birdset_test.features['ebird_code']._int2str  # or whatever the feature name is
    dataset_ebird_codes = [subset_label_mapping[i] for i in range(len(subset_label_mapping))]
    print(dataset_ebird_codes[:10])

    # Your model's eBird codes (in the order of model outputs)
    CLASS_LABELS_FILEPATH = "metadata/perch_v2_label_mapping.json"
    with open(CLASS_LABELS_FILEPATH) as f:
        perch_label_mapping = json.load(f)
    model_ebird_codes = [perch_label_mapping[str(i)] for i in range(len(perch_label_mapping))]
    CLASS_LABELS_FILEPATH = "metadata/perch_v2_label_mapping.json"
    with open(CLASS_LABELS_FILEPATH) as f:
        perch_label_mapping = json.load(f)
    # Create mapping from dataset eBird codes to model indices
    dataset_to_model_indices = []
    dataset_label_mapping = {}
    for idx, code in enumerate(dataset_ebird_codes):
        if code in model_ebird_codes:
            label = model_ebird_codes.index(code)
            dataset_to_model_indices.append(label)
            dataset_label_mapping[str(idx)] = code
        else:
            # Handle case where dataset code isn't in model (if possible)
            print(f"Warning: {code} not found in model outputs")
            dataset_to_model_indices.append(-1)  # or use a default index
    valid_mask = torch.tensor([idx != -1 for idx in dataset_to_model_indices], device=torch_device)
    valid_model_indices = torch.tensor([idx for idx in dataset_to_model_indices if idx != -1], device=torch_device)

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
            # no spatiotemporal predictions
            all_logits = base_classifier(torch.from_numpy(audio_embeds).to(torch_device), st_encodings.to(torch_device))
            if type(all_logits) == tuple:
                all_logits = all_logits[0]
            dataset_logits = all_logits[:, valid_model_indices]
            dataset_pred_indices = torch.argmax(dataset_logits, dim=1)
            pred_indices = valid_model_indices[dataset_pred_indices]
            predicted_ebird_codes += [perch_label_mapping[str(index.item())] for index in pred_indices]
            # spatiotemporally aware predictions
            geo_all_logits = geo_aware_classifier(torch.from_numpy(audio_embeds).to(torch_device), st_encodings.to(torch_device))
            if type(geo_all_logits) == tuple:
                geo_all_logits = geo_all_logits[0]
            geo_dataset_logits = geo_all_logits[:, valid_model_indices]
            geo_dataset_pred_indices = torch.argmax(geo_dataset_logits, dim=1)
            geo_pred_indices = valid_model_indices[geo_dataset_pred_indices]
            geo_aware_predicted_ebird_codes += [perch_label_mapping[str(index.item())] for index in geo_pred_indices]
            #ground_truth += [[perch_label_mapping[str(label)] for label in multilabel] for multilabel in labels]
    with open(prediction_save_path, "wb") as fw:
        pickle.dump({"perch":predicted_ebird_codes, "geo_aware":geo_aware_predicted_ebird_codes}, fw)
    print(predicted_ebird_codes[:10])
    print(ground_truth[:10])