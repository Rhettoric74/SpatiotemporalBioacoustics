import os

# TensorFlow gets GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import tensorflow_hub as hub
tf.experimental.numpy.experimental_enable_numpy_behavior()

# Optional: allow memory growth on TF GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Now switch visibility for PyTorch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.optim as optim

# Your remaining imports
import numpy as np
import kagglehub
from load_bioclip import load_bioclip
from SpatialRelationEncoder import SphereMixScaleSpatialRelationEncoder
from spatiotemporal_encoder import SpatiotemporalEncoder
from load_pickled_embeddings import load_clasp_data
from typing import Dict, List
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import AvgMeter


def load_perch_model(url = 'google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu'):
    #TODO: Currently using perch 1.0, need update this to Perch 2.0
    # and update class label mappings and spatiotemporal encoders
    path = kagglehub.model_download(url)
    model = tf.saved_model.load(path)
    return model

class ClaspDataset(Dataset):
    def __init__(self, audio_embeddings: torch.Tensor, 
                 spatiotemporal_context: torch.Tensor, 
                 text_prompts: List[str]):
        self.audio_embeddings = audio_embeddings
        self.spatiotemporal_context = spatiotemporal_context
        self.text_prompts = text_prompts
        
        assert len(audio_embeddings) == len(spatiotemporal_context) == len(text_prompts)
    
    def __len__(self):
        return len(self.audio_embeddings)
    
    def __getitem__(self, idx):
        return (self.audio_embeddings[idx],
                self.spatiotemporal_context[idx],
                self.text_prompts[idx])

def simple_collate_fn(batch):
    audio_embeddings = torch.stack([item[0] for item in batch])
    spatiotemporal_context = torch.stack([item[1] for item in batch])
    text_prompts = torch.stack([item[2] for item in batch])
    
    return audio_embeddings, spatiotemporal_context, text_prompts
class Identity(nn.Module):
    """Identity transformation that returns input as output"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

def contrastive_loss(text_feats, audio_feats, spatio_feats, logit_scale):
    # Simple pairwise cosine similarity
    scale = logit_scale.exp().clamp(min=0.05, max=20)  # prevent overflow
    sim_text_audio = text_feats @ audio_feats.T * scale
    sim_text_spatio = text_feats @ spatio_feats.T * scale
    labels = torch.arange(text_feats.size(0), device=text_feats.device)
    
    
    loss_audio = nn.CrossEntropyLoss()(sim_text_audio, labels)
    loss_spatio = nn.CrossEntropyLoss()(sim_text_spatio, labels)
    loss_audio_i2t = nn.CrossEntropyLoss()(sim_text_audio.T, labels)
    loss_spatio_i2t = nn.CrossEntropyLoss()(sim_text_spatio.T, labels)
    audio_loss = (loss_audio + loss_audio_i2t) / 2
    spatio_loss = (loss_spatio + loss_spatio_i2t) / 2
    return audio_loss + spatio_loss, audio_loss, spatio_loss
'''
class ProjectionHead(nn.Module):
    """Taken from Moein Shariatnia's medium article, Simple Implementation of OpenAI CLIP model: A Tutorial"""
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout=0.3
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
        self._initialize_weights()
    def _initialize_weights(self):
        """Proper weight initialization for stability"""
        # Xavier/Glorot initialization for linear layers
        for module in [self.projection, self.fc]:
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        # Initialize LayerNorm properly
        torch.nn.init.ones_(self.layer_norm.weight)
        torch.nn.init.zeros_(self.layer_norm.bias)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
'''
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout)
            )
            for _ in range(depth)
        ])
        self.output_proj = nn.Linear(hidden_dim, out_dim)
        self.final_norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = x + block(x)  # residual
        x = self.output_proj(x)
        x = self.final_norm(x)
        return x
class SimpleProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # or LayerNorm
            nn.ReLU(),  # or GELU
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)
class CLASPModel(nn.Module):
    def __init__(self, text_encoder, text_tokenizer, audio_encoder, st_encoder, text_projector = None, audio_projector = None, st_projector = None, device = "cuda"):
        super().__init__()
        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer
        self.audio_encoder = audio_encoder
        self.st_encoder = st_encoder
        self.audio_projector = audio_projector
        self.text_projector = text_projector
        self.st_projector = st_projector
        self.target_text_embeds = None
        self.target_text_list = None
        self.device = device
    def embed_audio(self, audio):
        audio_embeddings = self.audio_encoder.signatures['serving_default'](inputs=audio)['embedding']
        audio_embeddings = torch.from_numpy(audio_embeddings.numpy()).to(self.device)
        proj_audio = self.audio_projector(audio_embeddings)
        return proj_audio
    def embed_text(self, text):
        tokenized = self.text_tokenizer(text).to(self.device)
        text_embeddings = self.text_encoder.encode_text(tokenized)
        proj_text = self.text_projector(text_embeddings)
        return proj_text
    def embed_spatiotemporal(self, st_context):
        st_embeddings = self.st_encoder(st_context).to(self.device)
        proj_st_context = self.st_projector(st_embeddings)
        return proj_st_context.squeeze(1)
    def forward(self, text, audio, st_context):
        proj_text = self.embed_text(text)
        proj_audio = self.embed_audio(audio)
        proj_st_context = self.embed_spatiotemporal(st_context)
        text_features = nn.functional.normalize(proj_text, dim=-1)
        audio_features = nn.functional.normalize(proj_audio, dim=-1)
        spatio_features = nn.functional.normalize(proj_st_contexts, dim=-1)
        return audio_features @ text_features.T, audio_features @ spatio_features.T
    def load_target_text_embeds(self, text_list, batch_size = 1024):
        all_embeddings = []
        self.target_text_list = text_list
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i:i+batch_size]
        
            with torch.no_grad():
                batch_emb = self.embed_text(batch_texts)
                batch_emb = nn.functional.normalize(batch_emb, dim=-1)
                all_embeddings.append(batch_emb)
        text_feats = torch.cat(all_embeddings, dim=0)
        self.target_text_embeds = text_feats
    def classify_from_audio(self, audio, k = 1):
        if self.target_text_embeds == None:
            print("Call load_target_text_embeds first to load target embeddings.")
            raise Exception("Need to initialize target text embeddings first!")
        audio_features = self.embed_audio(audio)
        audio_features = nn.functional.normalize(audio_features, dim=-1)
        similarity_matrix = audio_features @ self.target_text_embeds.T
        values, indices = torch.topk(similarity_matrix, k, dim = 1)
        return indices
    def classify_from_audio_embeds(self, audio_embeds, k = 1):
        if self.target_text_embeds == None:
            print("Call load_target_text_embeds first to load target embeddings.")
            raise Exception("Need to initialize target text embeddings first!")
        proj_audio = self.audio_projector(audio_embeds)
        audio_features = nn.functional.normalize(proj_audio, dim=-1)
        similarity_matrix = audio_features @ self.target_text_embeds.T
        values, indices = torch.topk(similarity_matrix, k, dim = 1)
        return indices
    def classify_from_spatiotemporal(self, st_context, k = 10):
        if self.target_text_embeds == None:
            print("Call load_target_text_embeds first to load target embeddings.")
            raise Exception("Need to initialize target text embeddings first!")
        st_features = self.embed_spatiotemporal(st_context)
        st_features = nn.functional.normalize(st_features, dim=-1)
        similarity_matrix = st_features @ self.target_text_embeds.T
        values, indices = torch.topk(similarity_matrix, k, dim = 1)
        return indices
    def classify_from_spatiotemporal_and_audio(self, audio, st_context, k = 1):
        if self.target_text_embeds == None:
            print("Call load_target_text_embeds first to load target embeddings.")
            raise Exception("Need to initialize target text embeddings first!")
        st_features = self.embed_spatiotemporal(st_context)
        st_features = nn.functional.normalize(st_features, dim=-1)
        audio_features = self.embed_audio(audio)
        audio_features = nn.functional.normalize(audio_features, dim=-1)
        similarity_matrix = audio_features @ self.target_text_embeds.T + st_features @ self.target_text_embeds.T
        values, indices = torch.topk(similarity_matrix, k, dim = 1)
        return indices
    def classify_from_spatiotemporal_and_audio_embeds(self, audio_embeds, st_context, k = 1):
        if self.target_text_embeds == None:
            print("Call load_target_text_embeds first to load target embeddings.")
            raise Exception("Need to initialize target text embeddings first!")
        st_features = self.embed_spatiotemporal(st_context)
        st_features = nn.functional.normalize(st_features, dim=-1)
        proj_audio = self.audio_projector(audio_embeds)
        audio_features = nn.functional.normalize(proj_audio, dim=-1)
        similarity_matrix = (audio_features @ self.target_text_embeds.T) * (st_features @ self.target_text_embeds.T)
        values, indices = torch.topk(similarity_matrix, k, dim = 1)
        return indices
        
        
def train_epoch(audio_projector, st_enc, st_projector, text_enc, text_proj, train_loader, optimizer, scaler, use_st = True, device = "cuda"):
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    audio_loss_meter = AvgMeter()
    spatio_loss_meter = AvgMeter()
    skipped_batches = 0
    for audio_embeds, st_encoding, text_embedding in tqdm_object:
        #print("audio_embeds", audio_embeds.cpu())
        #print("st_context", st_context)
        #print("text_descriptions", text_descriptions)
        optimizer.zero_grad()
        #torch.nn.utils.clip_grad_norm_(bioclip_model.parameters(), max_norm=0.5)
        audio_feats = audio_projector(audio_embeds.to(device))
        #st_encoding = st_enc(st_context)
        st_feats = st_projector(st_encoding.to(device))
        #print("spatiotemporal feats", st_feats.cpu())
        #print("audio feats", audio_feats.cpu())
        #print("st_dim", st_feats.shape)
        st_feats = st_feats.squeeze(1) 
        # project text encoding
        text_feats = text_proj(text_embedding.to(device))
        #print("text feats", text_feats.cpu())
        #text_feats = text_feats.squeeze().reshape(len(text_descriptions), 512)
        #print(text_feats.shape)
        # Normalize features to prevent large values
        text_feats = torch.nn.functional.normalize(text_feats, p=2, dim=1)
        audio_feats = torch.nn.functional.normalize(audio_feats, p=2, dim=1) 
        st_feats = torch.nn.functional.normalize(st_feats, p=2, dim=1)
        #print((text_feats @ audio_feats.T).cpu())
        #raise Exception("terminating early for debugging purposes")
        loss, audio_loss, spatio_loss = contrastive_loss(text_feats, audio_feats, st_feats, scaler)
        # trying audio-text loss only, without considering spatiotemporal loss
        if use_st:
            loss.backward()
        else:
            # only use audio loss
            audio_loss.backward()
        optimizer.step()
        count = audio_embeds.size(0)
        audio_loss_meter.update(audio_loss.item(), count)
        spatio_loss_meter.update(spatio_loss.item(), count)
        tqdm_object.set_postfix({'audio_loss':audio_loss_meter.avg, 'st_loss': spatio_loss_meter.avg})
    return audio_loss_meter, spatio_loss_meter
def encode_texts_in_batches(texts, bioclip_model, tokenizer, batch_size=1024):
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        with torch.no_grad():
            tokens = tokenizer(batch_texts).to("cuda")
            batch_emb = bioclip_model.encode_text(tokens)
            all_embeddings.append(batch_emb.cpu())
    
    return torch.cat(all_embeddings, dim=0)
if __name__ == '__main__':
    batch_size = 8192
    num_epochs = 100
    embedding_dim = 512
    use_st = True
    use_st_string = "_no_st"
    if use_st:
        use_st_string = ""
    learning_rate = 1e-3
    str_deep = "st_and_audio_deep_no_text"
    selection_strategy = "random"
    text_proj_path = 'models/' + selection_strategy + '_text_' + str_deep + '_projector_' + str(num_epochs) +'_epoch_lr_'+ str(learning_rate) + use_st_string + '_d_' + str(embedding_dim) + '.pth'
    audio_proj_path = 'models/' + selection_strategy + '_audio_' + str_deep + '_projector_' + str(num_epochs) +'_epoch_lr_'+ str(learning_rate) + use_st_string + '_d_' + str(embedding_dim) + '.pth'
    st_proj_path = 'models/' + selection_strategy + '_st_' + str_deep + '_projector_' + str(num_epochs) +'_epoch_lr_'+ str(learning_rate) + use_st_string + '_d_' + str(embedding_dim) + '.pth'
    DATA_DIR = "/scratch/e1583377/pickled_audio_embeds_" + selection_strategy + "_finally_corrected/"
    audio_embeddings, spatiotemporal_context, text_prompts = load_clasp_data(DATA_DIR)
    bioclip_model, tokenizer = load_bioclip()
    bioclip_model.eval()
    bioclip_model = bioclip_model.to('cuda')
    print(audio_embeddings.shape, spatiotemporal_context.shape, len(text_prompts))
    # Create dataset
    location_encoder = SphereMixScaleSpatialRelationEncoder(160, frequency_num = 32)
    st_enc = SpatiotemporalEncoder(location_encoder, loc_dim = 2, device = 'cpu', temporal_encoding_dim = 5)
    spatiotemporal_encodings = st_enc(spatiotemporal_context)
    
    print(spatiotemporal_encodings.shape)
    print(type(spatiotemporal_encodings))
    text_emb = encode_texts_in_batches(text_prompts, bioclip_model, tokenizer)
    dataset = ClaspDataset(
        audio_embeddings=audio_embeddings,
        spatiotemporal_context=spatiotemporal_encodings,
        text_prompts=text_emb
    )
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True,  # Faster GPU transfer
        collate_fn=simple_collate_fn
    )
    if str_deep == "deep":
        audio_proj = ProjectionHead(1536, 1024, embedding_dim).to('cuda')
        st_proj = ProjectionHead(165, 256, embedding_dim).to('cuda')
        text_proj = ProjectionHead(512, 1024, embedding_dim).to('cuda')
    elif str_deep == "audio_deep":
        audio_proj = ProjectionHead(1536, 1024, embedding_dim, depth = 1).to('cuda')
        st_proj = SimpleProjectionHead(165, 256, embedding_dim).to('cuda')
        text_proj = SimpleProjectionHead(512, 1024, embedding_dim).to('cuda')
    elif str_deep == "audio_deep_no_text":
        audio_proj = ProjectionHead(1536, 1024, embedding_dim, depth = 1).to('cuda')
        st_proj = SimpleProjectionHead(165, 256, embedding_dim).to('cuda')
        text_proj = Identity()
    elif str_deep == "st_and_audio_deep_no_text":
        audio_proj = ProjectionHead(1536, 1024, embedding_dim, depth = 5).to('cuda')
        st_proj = ProjectionHead(165, 256, embedding_dim, depth = 1).to('cuda')
        text_proj = Identity()
    else: 
        audio_proj = SimpleProjectionHead(1536, 1024, embedding_dim).to('cuda')
        st_proj = SimpleProjectionHead(165, 256, embedding_dim).to('cuda')
        text_proj = SimpleProjectionHead(512, 1024, embedding_dim).to('cuda')
    """
    # load checkpoints
    
    audio_state_dict = torch.load('models/peak_select_audio_projector_mixed_lr_0001.pth')
    # Load the state dictionary into the model
    audio_proj.load_state_dict(audio_state_dict)
    #audio_proj = audio_proj.to('cuda')
    text_proj_state_dict = torch.load('models/peak_select_text_projector_mixed_lr_00001.pth')
    text_proj.load_state_dict(text_proj_state_dict)
    st_state_dict = torch.load('models/peak_select_projector_mixed_lr_00001.pth')
    st_proj.load_state_dict(st_state_dict)
    """
    print(bioclip_model)
    # make bioclip text encoder trainable
    
    """Make only the last N transformer layers trainable"""
    text_encoder = bioclip_model.transformer
    
    # Freeze all parameters first
    for param in text_encoder.parameters():
        param.requires_grad = False
    """
    # Unfreeze last N layers
    num_trainable_layers = 4
    total_layers = len(text_encoder.resblocks)
    start_layer = total_layers - num_trainable_layers
    
    print(f"Training last {num_trainable_layers} of {total_layers} transformer layers")
    
    for i in range(start_layer, total_layers):
        for param in text_encoder.resblocks[i].parameters():
            param.requires_grad = True"""
    
    # Also make these trainable for better adaptation:
    # Final layer norm
    
    logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07)).to("cuda"))
    for param in bioclip_model.ln_final.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam([
        {'params': audio_proj.parameters(), 'lr': learning_rate},
        {'params': st_proj.parameters(), 'lr': learning_rate},
        {'params': text_proj.parameters(), 'lr': learning_rate},
        {'params':[logit_scale], 'lr':learning_rate}
    ], lr=learning_rate)
    for i in range(num_epochs):
        train_epoch(audio_proj, st_enc, st_proj, bioclip_model, text_proj, dataloader, optimizer, logit_scale, use_st)
        # save checkpoints
        torch.save(text_proj.state_dict(), text_proj_path)
        torch.save(audio_proj.state_dict(), audio_proj_path)
        torch.save(st_proj.state_dict(), st_proj_path)
    
    

        
        