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
from load_bioclip import load_bioclip
from SpatialRelationEncoder import SphereMixScaleSpatialRelationEncoder
from load_pickled_embeddings import load_clasp_data, generate_text_descriptions
from typing import Dict, List
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from clasp import CLASPModel, ProjectionHead, SimpleProjectionHead, load_perch_model, SpatiotemporalEncoder
from birdset_preparation import load_birdset_data
import librosa
import json
SAVE_DIR = "/scratch/e1583377/pickled_birdset_embeds/"
if __name__ == '__main__':
    subset = "POW"
    batch_size = 16384
    num_epochs = 100
    embedding_dim = 2048
    use_st = True
    use_st_string = "_no_st"
    if use_st:
        use_st_string = ""
    learning_rate = 1e-3
    str_deep = "deep"
    text_proj_path = 'models/peak_select_text_' + str_deep + '_projector_' + str(num_epochs) +'_epoch_lr_'+ str(learning_rate) + use_st_string + '_d_' + str(embedding_dim) + '.pth'
    audio_proj_path = 'models/peak_select_audio_' + str_deep + '_projector_' + str(num_epochs) +'_epoch_lr_'+ str(learning_rate) + use_st_string + '_d_' + str(embedding_dim) + '.pth'
    st_proj_path = 'models/peak_select_st_' + str_deep + '_projector_' + str(num_epochs) +'_epoch_lr_'+ str(learning_rate) + use_st_string + '_d_' + str(embedding_dim) + '.pth'
    prediction_save_path = "constrained_simple_projection_head_" +str(num_epochs) + "_epoch_lr_" + str(learning_rate) + use_st_string + "_d_" + str(embedding_dim) + "_clasp_" + subset + "_mult_predictions.pkl"
    print(prediction_save_path)
    if str_deep == "deep":
        audio_proj = ProjectionHead(1536, 1024, embedding_dim)
        st_proj = ProjectionHead(165, 256, embedding_dim)
        text_proj = ProjectionHead(512, 1024, embedding_dim)
    else:
        audio_proj = SimpleProjectionHead(1536, 1024, embedding_dim)
        st_proj = SimpleProjectionHead(165, 256, embedding_dim)
        text_proj = SimpleProjectionHead(512, 1024, embedding_dim)
    torch_device = torch.device("cuda:0")
    print(audio_proj)
    # load checkpoints
    """audio_proj_path = 'models/peak_select_simple_audio_projector_200_epoch_lr_0001_no_st_d_512.pth'
    text_proj_path = 'models/peak_select_simple_text_projector_200_epoch_lr_0001_no_st_d_512.pth'
    st_proj_path = 'models/peak_select_st_projector_200_epoch_lr_0001_no_st_d_512.pth'
    """
    audio_state_dict = torch.load(audio_proj_path)
    # Load the state dictionary into the model
    audio_proj.load_state_dict(audio_state_dict)
    #audio_proj = audio_proj.to('cuda')
    text_proj_state_dict = torch.load(text_proj_path)
    text_proj.load_state_dict(text_proj_state_dict)
    st_state_dict = torch.load(st_proj_path)
    st_proj.load_state_dict(st_state_dict)
    text_proj = text_proj.to(torch_device)
    audio_proj = audio_proj.to(torch_device)
    st_proj = st_proj.to(torch_device)
    
    bioclip_model, tokenizer = load_bioclip()
    bioclip_model = bioclip_model.to(torch_device)
    perch_model = load_perch_model()
    location_encoder = SphereMixScaleSpatialRelationEncoder(160, frequency_num = 32)
    st_enc = SpatiotemporalEncoder(location_encoder, loc_dim = 2, device = 'cpu', temporal_encoding_dim = 5)
    clasp_model = CLASPModel(bioclip_model, tokenizer, perch_model, st_enc, text_proj, audio_proj, st_proj, device = torch_device)
    # get text list for specific species
    birdset_test = load_birdset_data(subset)
    subset_label_mapping = birdset_test.features['ebird_code']._int2str  # or whatever the feature name is
    dataset_ebird_codes = [subset_label_mapping[i] for i in range(len(subset_label_mapping))]
    print(dataset_ebird_codes[:10])

    # Your model's eBird codes (in the order of model outputs)
    CLASS_LABELS_FILEPATH = "/home/svu/e1583377/Spatial_Perch_Transfer_Learning/assets/perch_v2_label_mapping.json"
    with open(CLASS_LABELS_FILEPATH) as f:
        perch_label_mapping = json.load(f)
    model_ebird_codes = [perch_label_mapping[str(i)] for i in range(len(perch_label_mapping))]

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
    print(dataset_label_mapping)
    valid_indices = torch.tensor([idx for idx in dataset_to_model_indices if idx != -1])
    text_list = generate_text_descriptions(valid_indices)
    print(text_list[:10])
    clasp_model.load_target_text_embeds(text_list, batch_size = 32)
    print(clasp_model.target_text_embeds.shape)
    minneapolis_st_context = np.expand_dims(np.array([[-93.27, 44.98, 0.99, 0.01, 0.99, 0.01, 0.5]], dtype=np.float32), axis = 1)
    mlps_species = clasp_model.classify_from_spatiotemporal(minneapolis_st_context)
    print([[clasp_model.target_text_list[idx.item()] for idx in row] for row in mlps_species])
    print(len(clasp_model.target_text_list))
    with open(SAVE_DIR + subset + ".pkl", "rb") as f:
        birdset_data = pickle.load(f)
    clasp_model.eval()
    print(clasp_model.target_text_list[:10])
    print(list(perch_label_mapping.keys())[:10])
    predicted_ebird_codes = []
    geo_aware_predicted_ebird_codes = []
    predicted_species = []
    ground_truth = []
    with torch.no_grad():
        for audio_embeds, st_context, labels in zip(birdset_data["embeddings"], birdset_data['st_context'], birdset_data["labels"]):
            """
            print(audio_embeds.shape)
            print(np.max(audio_embeds), np.min(audio_embeds))
            # Test with identical embeddings - should give high self-similarity
            test_audio = torch.from_numpy(audio_embeds[0:1]).to(torch_device)  # One sample
            test_duplicate = test_audio.clone()

            proj_test = clasp_model.audio_projector(test_audio)
            proj_dup = clasp_model.audio_projector(test_duplicate)

            proj_test_norm = nn.functional.normalize(proj_test, dim=-1)
            proj_dup_norm = nn.functional.normalize(proj_dup, dim=-1)
            similarity = proj_test_norm @ proj_dup_norm.T
            print(f"Self-similarity of identical inputs: {similarity.item():.4f}")  # Should be ~1.0
            """
            pred_indices = clasp_model.classify_from_audio_embeds(torch.from_numpy(audio_embeds).to(torch_device))
            predicted_ebird_codes += [subset_label_mapping[index.item()] for index in pred_indices]
            predicted_species += [clasp_model.target_text_list[idx.item()] for idx in pred_indices]
            geo_pred_indices = clasp_model.classify_from_spatiotemporal_and_audio_embeds(
                  torch.from_numpy(audio_embeds).to(torch_device), 
                  np.expand_dims(st_context.astype(np.float32), axis=1)
            )
            # map back to ebird codes
            geo_aware_predicted_ebird_codes += [subset_label_mapping[index.item()] for index in geo_pred_indices]
            ground_truth += [[subset_label_mapping[label] for label in multilabel] for multilabel in labels]
    with open(prediction_save_path, "wb") as fw:
        pickle.dump({"perch":predicted_ebird_codes, "geo_aware":geo_aware_predicted_ebird_codes}, fw)
    print(predicted_species[:10])
    print(predicted_ebird_codes[:10])
    print(ground_truth[:10])
