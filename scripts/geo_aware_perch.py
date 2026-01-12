import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from SpatialRelationEncoder import SphereMixScaleSpatialRelationEncoder, LocationEncoder
from torch.utils.data import DataLoader, TensorDataset
import tensorflow_hub as hub
import tensorflow as tf
import pickle
tf.experimental.numpy.experimental_enable_numpy_behavior()
from spatiotemporal_encoder import SpatiotemporalEncoder
import json
import librosa
CLASS_LABELS_FILEPATH = "/home/svu/e1583377/Spatial_Perch_Transfer_Learning/assets/perch_v2_label_mapping.json"
# Get Perch 2.0 from Kaggle
def load_perch_model(url = 'https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2/2'):
    #TODO: Currently using perch 1.0, need update this to Perch 2.0
    # and update class label mappings and spatiotemporal encoders
    model = hub.load(url)
    return model
def load_location_encoder(saved_model_path = 'xenocanto_location_encoder_dropout.pth', model_device = 'cuda', output_dim = 10932):
    print("loading location encoder to", model_device)
    pe = SphereMixScaleSpatialRelationEncoder(160, frequency_num = 32, device = model_device)
    location_embedding = nn.Sequential(pe, nn.Linear(160, 512), nn.ReLU(inplace=True), nn.Dropout(0.5)) 
    model = LocationEncoder(location_embedding, 160, output_dim, 512, num_users=1)
    state_dict = torch.load(saved_model_path, map_location=torch.device(model_device))
    model.load_state_dict(state_dict)
    model.to(model_device)
    return model
def load_st_encoder(saved_model_path = 'xc_spatiotemporal_encoder_lr_0001.pth', temporal_encoding_dim = 5, model_device = 'cuda', output_dim = 14795):
    print("loading spatiotemporal encoder to", model_device)
    pe = SphereMixScaleSpatialRelationEncoder(160, frequency_num = 32, device = model_device)
    st_encoder = SpatiotemporalEncoder(pe, device=model_device)
    location_embedding = nn.Sequential(st_encoder, nn.Linear(160 + temporal_encoding_dim, 512), nn.ReLU(inplace=True), nn.Dropout(0.5)) 
    model = LocationEncoder(location_embedding, 160 + temporal_encoding_dim, output_dim, 512, num_users=1)
    state_dict = torch.load(saved_model_path, map_location=torch.device(model_device))
    model.load_state_dict(state_dict)
    model.to(model_device)
    return model
class GeoAwarePerch(nn.Module):
    def __init__(self, location_encoder, perch_model, device = 'cuda', output_dim = 14795, output_activation = "softmax"):
        super().__init__()
        if location_encoder == None:
            location_encoder = load_location_encoder(model_device = device, output_dim=output_dim)
        self.location_encoder = location_encoder
        self.perch_model = perch_model
        self.output_dim = output_dim
        self.device = device
        self.output_activation = output_activation
    def forward(self, audio_data, location_data):
        # note that it is not straightforward to backpropagate to both models,
        # because the location_encoder is a pytorch model and perch is a tensorflow model.
        # This behaviour is reasonable because the sphere2vec paper only trains the two models separately,
        # and combines them only during inference
        location_output = self.location_encoder(location_data)
        # get perch logits for the audio
        # convert audio into tensorflow tensor
        audio_data = tf.convert_to_tensor(audio_data, dtype=tf.float32)
        perch_output = self.perch_model.signatures['serving_default'](inputs=audio_data)['label']
        perch_output_torch = torch.from_numpy(perch_output.numpy())
        # load perch output to the same device as the location encoder output
        perch_output_torch = perch_output_torch.to(self.device)
        # need to softmax normalize perch outputs to get probability predictions,
        # as perch 2.0 outputs are raw and unnormalized.
        #perch_output_torch = perch_output_torch.unsqueeze(0)
        if self.output_activation == "softmax":
            perch_probs = torch.softmax(perch_output_torch, dim=1)
        elif self.output_activation == "sigmoid":
            perch_probs = torch.sigmoid(perch_output_torch)
        else:
            raise Exception("Invalid output activation for GeoAware perch. Please use 'softmax' or 'sigmoid'.")
        posterior_logits = location_output * perch_probs
        return posterior_logits
    def embed_audio_and_spatiotemporal(self, audio_data, location_data, device = "cpu"):
        """Compute audio and spatiotemporal embeddings, which can be stored and used for other models."""
        location_output = self.location_encoder.spa_enc(location_data).to(device)
        # get perch logits for the audio
        # convert audio into tensorflow tensor
        audio_data = tf.convert_to_tensor(audio_data, dtype=tf.float32)
        perch_output = self.perch_model.signatures['serving_default'](inputs=audio_data)['embedding']
        perch_output_torch = torch.from_numpy(perch_output.numpy())
        # load perch output to the same device as the location encoder output
        perch_output_torch = perch_output_torch.to(device)
        # need to softmax normalize perch outputs to get probability predictions,
        # as perch 2.0 outputs are raw and unnormalized.
        #perch_output_torch = perch_output_torch.unsqueeze(0)
        return perch_output_torch, location_output

if __name__ == '__main__':

    # Load audio file
    file_path = 'asian_fairy_bluebird.wav'  # Replace with your file path
    y, sr = librosa.load(file_path, sr=32000)  # sr=None preserves original sampling rate

    # Define segment parameters
    start_time = 1  # Start at 10 seconds (adjust as needed)
    end_time = start_time + 5  # End 5 seconds later
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # Extract 5-second segment
    y_segment = y[start_sample:end_sample]
    # Load perch and get prediction
    perch_model = load_perch_model()
    perch_output = perch_model.infer_tf(y_segment[np.newaxis, :])
    perch_class_prediction = np.argmax(perch_output['label'])
    with open(CLASS_LABELS_FILEPATH) as f:
        label_mapping = json.load(f)
    perch_class_prediction = label_mapping[str(np.argmax(perch_output['label']))]
    print("Perch 2.0 prediction", perch_class_prediction)
    print(type(perch_model))
    # load geoaware perch and get prediction
    geo_aware_perch = GeoAwarePerch(load_location_encoder(model_device = 'cuda'), perch_model)
    # singapore coordinates
    location_tensor = torch.tensor([[103.819, 1.352]])
    geo_aware_perch_output = geo_aware_perch(y_segment[np.newaxis, :], location_tensor)
    geo_aware_perch_class_prediction = label_mapping[str(torch.argmax(geo_aware_perch_output).item())]
    print("Geo-Aware Perch 2.0 prediction", geo_aware_perch_class_prediction)
    # adjust with ground truth label according to class_names_dict.json
    # for input audio
    correct_class_label = 326
    print("Correct class probability from perch", torch.softmax(torch.from_numpy(perch_output['label'].numpy()), dim=-1)[0, correct_class_label].item())
    print("Correct class probability from geo-aware perch", geo_aware_perch_output[0][correct_class_label])
    
    
