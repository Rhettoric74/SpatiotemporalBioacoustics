import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Your remaining imports
import numpy as np
import kagglehub
from SpatialRelationEncoder import SphereMixScaleSpatialRelationEncoder
from load_pickled_embeddings import load_st_audio_data, load_mixup_embeddings
from typing import Dict, List
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from utils import AvgMeter
import random
from config import *


class MixupAudioDataset(Dataset):
    def __init__(self, 
                 audio_embeddings: torch.Tensor, 
                 labels_sparse: list,
                 component_contexts: list,
                 component_weights: list,
                 st_encoder,  # Your SpatiotemporalEncoder
                 num_classes: int = 14795,
                 context_sampling: str = 'weighted'):
        """
        Precomputes ST embeddings for all component contexts.
        """
        self.audio_embeddings = audio_embeddings
        self.labels_sparse = labels_sparse
        self.component_contexts = component_contexts
        self.component_weights = component_weights
        self.num_classes = num_classes
        self.context_sampling = context_sampling
        
        # Precompute ST embeddings for ALL component contexts
        print("Precomputing spatiotemporal embeddings for all components...")
        self.component_st_embeddings = self._precompute_st_embeddings(st_encoder)
        
        print(f"Mixup dataset: {len(audio_embeddings)} samples, "
              f"{np.mean([len(ctx) for ctx in component_contexts]):.1f} avg components")
    
    def _precompute_st_embeddings(self, st_encoder):
        """Precompute ST embeddings for all component contexts."""
        st_encoder.eval()
        all_st_embeds = []
    
        with torch.no_grad():
            for contexts in self.component_contexts:
                if not contexts:
                    all_st_embeds.append([])
                    continue
            
                # Prepare numpy array: [n_components, 1, 7]
                contexts_list = []
                for ctx in contexts:
                    # Ensure shape [1, 7]
                    if ctx.ndim == 1:
                        ctx = ctx.reshape(1, -1)
                    if ctx.shape[1] != 7:
                        # Force to 7 features
                        ctx = ctx[:, :7] if ctx.shape[1] > 7 else np.pad(ctx, ((0, 0), (0, 7 - ctx.shape[1])))
                    contexts_list.append(ctx)
            
                contexts_array = np.stack(contexts_list, axis=0)
            
                embeds_tensor = st_encoder(contexts_array).squeeze(1)
            
                # Convert to list of tensors
                embeds_list = [embeds_tensor[i] for i in range(len(embeds_tensor))]
                all_st_embeds.append(embeds_list)
    
        return all_st_embeds
    
    def __len__(self):
        return len(self.audio_embeddings)
    
    def __getitem__(self, idx):
        audio_embed = self.audio_embeddings[idx]
        labels = self.labels_sparse[idx]
        weights = self.component_weights[idx]
        st_embeds = self.component_st_embeddings[idx]
        
        # Sample component based on weights
        if self.context_sampling == 'weighted' and weights and len(weights) > 0:
            probs = np.array(weights) / np.sum(weights)
            comp_idx = np.random.choice(len(weights), p=probs)
        else:
            comp_idx = 0 if weights else 0
        
        # Get selected ST embedding
        if st_embeds and len(st_embeds) > comp_idx:
            st_embed = st_embeds[comp_idx]
        else:
            st_embed = torch.zeros_like(audio_embed)
        
        # Get selected label
        if isinstance(labels, list) and comp_idx < len(labels):
            selected_label = labels[comp_idx]
        else:
            selected_label = labels[0] if isinstance(labels, list) and labels else labels
        
        # Create target vector
        unique_labels = list(set(labels)) if isinstance(labels, list) else [labels]
        k = len(unique_labels)
        # old strategy, weighting the one with provided spatiotemporal context higher
        """
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        if k > 0:
            base_weight = 1.0 / (k + 1)
            for label in unique_labels:
                if 0 <= label < self.num_classes:
                    target[label] = base_weight
            
            if 0 <= selected_label < self.num_classes:
                target[selected_label] = 2.0 / (k + 1)
            
            # Normalize
            target = target / target.sum()
        """
        # weight all the target labels equally
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for label in unique_labels:
            if 0 <= label < self.num_classes:
                    target[label] = (1 / k)
        return audio_embed, st_embed, target


def create_mixup_dataloader(data_dir, st_encoder, num_classes=14795, batch_size=8192):
    """Create dataloader with precomputed ST embeddings."""
    # Load mixup data
    data = load_mixup_embeddings(data_dir)
    if not data:
        raise ValueError("No mixup data loaded")
    
    # Create dataset
    dataset = MixupAudioDataset(
        audio_embeddings=data['audio_embeddings'],
        labels_sparse=data['labels_sparse'],
        component_contexts=data['component_contexts'],
        component_weights=data['component_weights'],
        st_encoder=st_encoder,
        num_classes=num_classes,
        context_sampling='weighted'
    )
    
    # Simple collate function
    def collate_fn(batch):
        audio, st, target = zip(*batch)
        return {
            'audio': torch.stack(audio),
            'st': torch.stack(st),
            'target': torch.stack(target)
        }
    
    # Create dataloader
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )



class ST_AudioDataset(Dataset):
    def __init__(self, audio_embeddings: torch.Tensor, 
                 spatiotemporal_context: torch.Tensor, 
                 labels):
        self.audio_embeddings = audio_embeddings
        self.spatiotemporal_context = spatiotemporal_context
        # Keep only rows without NaNs
        st_mask = ~torch.isnan(spatiotemporal_context).any(axis=1).bool()
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
        return (self.audio_embeddings[idx], self.spatiotemporal_context[idx], self.labels[idx])

def simple_collate_fn(batch):
    audio_embeddings = torch.stack([item[0] for item in batch])
    spatiotemporal_context = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    
    return audio_embeddings, spatiotemporal_context, labels
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
class LargeClassifierHead(nn.Module):
    def __init__(self, audio_embedding_dim = 1536, st_embedding_dim = 165, num_classes=14795, hidden_dim=4096, dropout = 0.1, device = "cuda"):
        super().__init__()
        self.audio_embedding_dim = audio_embedding_dim
        self.st_embedding_dim = st_embedding_dim
        self.device = device
        # Layer 1: 1536 -> 4096
        self.fc1 = nn.Linear(audio_embedding_dim + st_embedding_dim, hidden_dim)  # 1536 * 4096 = 6.3M
        # Layer 2: 4096 -> 4096  
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)     # 4096 * 4096 = 16.8M
        # Layer 3: 4096 -> 4096
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)     # 4096 * 4096 = 16.8M
        self.fc4 = nn.Linear(hidden_dim, num_classes)    # 4096 * 15000 = 61.4M
        self.dropout = nn.Dropout(p=dropout)
        
        # Total: 6.3M + 16.8M + 16.8M + 16.8M + 61.4M = 118.1M
        # (Slightly over, but they might have smaller intermediate dimensions)
    
    def forward(self, audio_embeds, st_embeds):
        if self.st_embedding_dim == 0:
            st_embeds = torch.tensor([]).to(self.device)
        x = torch.cat([audio_embeds, st_embeds], dim = 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x
class SimpleClassifierHead(nn.Module):
    def __init__(self, audio_embedding_dim = 1536, st_embedding_dim = 165, num_classes=14795, hidden_dim=5600, dropout = 0.1, device = "cuda"):
        super().__init__()
        self.audio_embedding_dim = audio_embedding_dim
        self.st_embedding_dim = st_embedding_dim
        self.device = device
        self.fc1 = nn.Linear(audio_embedding_dim + st_embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, audio_embeds, st_embeds):
        if self.st_embedding_dim == 0:
            st_embeds = torch.tensor([]).to(self.device)
        x = torch.cat([audio_embeds, st_embeds], dim = 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class FullyLinearClassifierHead(nn.Module):
    def __init__(self, audio_embedding_dim = 1536, st_embedding_dim = 165, num_classes=14795, device = "cuda"):
        super().__init__()
        self.audio_embedding_dim = audio_embedding_dim
        self.st_embedding_dim = st_embedding_dim
        self.device = device
        self.fc1 = nn.Linear(audio_embedding_dim + st_embedding_dim, num_classes)
    
    def forward(self, audio_embeds, st_embeds):
        if self.st_embedding_dim == 0:
            st_embeds = torch.tensor([]).to(self.device)
        x = torch.cat([audio_embeds, st_embeds], dim = 1)
        x = self.fc1(x)
        return x
        
class MaxoutClassifierHead(nn.Module):
    def __init__(self, audio_embedding_dim=1536, st_embedding_dim=165, 
                 num_classes=14795):
        super().__init__()
        self.audio_embedding_dim = audio_embedding_dim
        self.st_embedding_dim = st_embedding_dim
        input_dim = audio_embedding_dim + st_embedding_dim
        
        # Four parallel linear classifiers (this gives the 4x parameter count)
        self.linear1 = nn.Linear(input_dim, num_classes)
        self.linear2 = nn.Linear(input_dim, num_classes)
        self.linear3 = nn.Linear(input_dim, num_classes)
        self.linear4 = nn.Linear(input_dim, num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"ExplicitMaxout parameters: {total_params:,}")
        
    def forward(self, audio_embeds, st_embeds):
        if self.st_embedding_dim == 0:
            x = audio_embeds
        else:
            x = torch.cat([audio_embeds, st_embeds], dim=1)
        
        # Compute all four outputs and take max
        out1 = self.linear1(x)
        out2 = self.linear2(x)
        out3 = self.linear3(x) 
        out4 = self.linear4(x)
        
        # Stack and take max
        outputs = torch.stack([out1, out2, out3, out4], dim=1)  # [batch, 4, num_classes]
        output = outputs.max(dim=1)[0]  # [batch, num_classes]
        
        return output
class MaxoutSTEmbeddingClassifierHead(nn.Module):
    def __init__(self, audio_embedding_dim=1536, st_embedding_dim=165, st_hidden_dim = 512, num_classes=14795, dropout = 0.5):
        super().__init__()
        self.audio_embedding_dim = audio_embedding_dim
        self.st_embedding_dim = st_embedding_dim
        input_dim = audio_embedding_dim + st_hidden_dim
        
        # Four parallel linear classifiers (this gives the 4x parameter count)
        self.st_hidden = nn.Linear(st_embedding_dim, st_hidden_dim)
        self.linear1 = nn.Linear(input_dim, num_classes)
        self.linear2 = nn.Linear(input_dim, num_classes)
        self.linear3 = nn.Linear(input_dim, num_classes)
        self.linear4 = nn.Linear(input_dim, num_classes)
        self.dropout = nn.Dropout(p=dropout)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"ExplicitMaxout parameters: {total_params:,}")
        
    def forward(self, audio_embeds, st_enc):
        if self.st_embedding_dim == 0:
            x = audio_embeds
        else:
            st_embeds = self.dropout(F.relu(self.st_hidden(st_enc)))
            x = torch.cat([audio_embeds, st_embeds], dim=1)
        
        # Compute all four outputs and take max
        out1 = self.linear1(x)
        out2 = self.linear2(x)
        out3 = self.linear3(x) 
        out4 = self.linear4(x)
        
        # Stack and take max
        outputs = torch.stack([out1, out2, out3, out4], dim=1)  # [batch, 4, num_classes]
        output = outputs.max(dim=1)[0]  # [batch, num_classes]
        
        return output

class MoE_ST_EmbeddingClassifierHead(nn.Module):
    def __init__(self, audio_embedding_dim=1536, st_embedding_dim=165, st_hidden_dim = 512, num_classes=14795, num_experts = 4, st_dropout = 0.5, audio_dropout = 0):
        super().__init__()
        self.audio_embedding_dim = audio_embedding_dim
        self.st_embedding_dim = st_embedding_dim
        input_dim = audio_embedding_dim + st_hidden_dim
        
        # Four parallel linear classifiers (this gives the 4x parameter count)
        self.st_hidden = nn.Linear(st_embedding_dim, st_hidden_dim)
        self.experts = nn.ModuleList([nn.Linear(input_dim, num_classes) for i in range(num_experts)])
        self.st_dropout = nn.Dropout(p=st_dropout)
        self.audio_dropout = nn.Dropout(p=audio_dropout)
        self.gate = nn.Linear(input_dim, num_experts)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"ExplicitMaxout parameters: {total_params:,}")
        
    def forward(self, audio_embeds, st_enc):
        if self.st_embedding_dim == 0:
            x = self.audio_dropout(audio_embeds)
        else:
            st_embeds = self.st_dropout(F.relu(self.st_hidden(st_enc)))
            x = torch.cat([self.audio_dropout(audio_embeds), st_embeds], dim=1)
        
        # Compute all four outputs and take max
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        # Stack and take max
        gate_weights = F.softmax(self.gate(x), dim=1)
        output = (expert_outputs * gate_weights.unsqueeze(2)).sum(dim=1)
        
        return output
    def choose_experts(self, audio_embeds, st_enc):
        if self.st_embedding_dim == 0:
            x = audio_embeds
        else:
            st_embeds = self.dropout(F.relu(self.st_hidden(st_enc)))
            x = torch.cat([audio_embeds, st_embeds], dim=1)
        gate_weights = F.softmax(self.gate(x), dim=1)
        return torch.argmax(gate_weights, dim = 1)
class BalancedMoE_ST_EmbeddingClassifierHead(nn.Module):
    def __init__(self, audio_embedding_dim=1536, st_embedding_dim=165, st_hidden_dim=512, 
                 num_classes=14795, num_experts=4, dropout=0.5, load_balance_weight=1):
        super().__init__()
        self.audio_embedding_dim = audio_embedding_dim
        self.st_embedding_dim = st_embedding_dim
        input_dim = audio_embedding_dim + st_hidden_dim
        
        # Four parallel linear classifiers (this gives the 4x parameter count)
        self.st_hidden = nn.Linear(st_embedding_dim, st_hidden_dim)
        self.experts = nn.ModuleList([nn.Linear(input_dim, num_classes) for i in range(num_experts)])
        self.dropout = nn.Dropout(p=dropout)
        self.gate = nn.Linear(input_dim, num_experts)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"ExplicitMaxout parameters: {total_params:,}")
        self.load_balance_weight = load_balance_weight
        
    def forward(self, audio_embeds, st_enc):
        if self.st_embedding_dim == 0:
            x = audio_embeds
        else:
            st_embeds = self.dropout(F.relu(self.st_hidden(st_enc)))
            x = torch.cat([audio_embeds, st_embeds], dim=1)
        
        # Get gate weights and expert outputs
        gate_weights = F.softmax(self.gate(x), dim=1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = (expert_outputs * gate_weights.unsqueeze(2)).sum(dim=1)
        
        return output, gate_weights  # Return both output and gate_weights
    
    # In your training loop:
    def compute_load_balance_loss(self, gate_weights):
        expert_utilization = gate_weights.mean(dim=0)
        mean_util = expert_utilization.mean()
        std_util = expert_utilization.std()
        cv_loss = (std_util / (mean_util + 1e-8)) ** 2
        return cv_loss * self.load_balance_weight
    def choose_experts(self, audio_embeds, st_enc):
        if self.st_embedding_dim == 0:
            x = audio_embeds
        else:
            st_embeds = self.dropout(F.relu(self.st_hidden(st_enc)))
            x = torch.cat([audio_embeds, st_embeds], dim=1)
        gate_weights = F.softmax(self.gate(x), dim=1)
        return torch.argmax(gate_weights, dim = 1)
def train_classifier(
    classifier_head,
    train_loader,
    num_epochs=50,
    learning_rate=5e-3,
    device='cuda'
):
    classifier_head = classifier_head.to(device)
    
    optimizer = optim.AdamW(
        classifier_head.parameters(), 
        lr=learning_rate,
        weight_decay=0.001
    )
    
    # Loss function (add label smoothing for 15k classes)
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'learning_rates': []
    }
    
    patience = 10
    patience_counter = 0
    
    print(f"Training classifier with {sum(p.numel() for p in classifier_head.parameters()):,} parameters")
    
    for epoch in range(num_epochs):
        # Training phase
        classifier_head.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        load_balance_total = 0.0
        for batch_idx, (audio_data, st_data, labels) in enumerate(train_pbar):
            audio_data, st_data, labels = audio_data.to(device), st_data.to(device), labels.to(device)
            
            # Forward pass
            
            output = classifier_head(audio_data, st_data)
            # handle both single logits and logits with gate weights
            if type(output) == tuple:
                logits, gate_weights = output
                load_balance_loss = classifier_head.compute_load_balance_loss(gate_weights)
                loss = criterion(logits, labels) + load_balance_loss
            else:
                logits = output
                loss = criterion(logits, labels)
                load_balance_loss = 0
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            #torch.nn.utils.clip_grad_norm_(classifier_head.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            load_balance_total += load_balance_loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Load Balance loss': f'{load_balance_loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Calculate epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        
        # Update scheduler
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print("saving model...")
        num_experts = str(len(classifier_head.experts))
        save_path = MODEL_SAVE_DIR + "/balanced_mixture_of_" + num_experts + "_experts_early_fusion_classifier_lr_003_epoch_" + str(epoch) + "_no_st.pth"
        torch.save(classifier_head.state_dict(), save_path)
        

    
    return classifier_head, history
    
def train_mixup_classifier(
    classifier_head,
    mixup_loader,
    non_mixup_loader=None,
    num_epochs=50,
    learning_rate=5e-3,
    device='cuda'
):
    classifier_head = classifier_head.to(device)
    optimizer = optim.AdamW(classifier_head.parameters(), lr=learning_rate, weight_decay=0.001)
    
    history = {'train_loss': [], 'train_acc': []}
    
    for epoch in range(num_epochs):
        classifier_head.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_count = 0
        
        # Process non-mixup loader if provided
        if non_mixup_loader is not None:
            regular_pbar = tqdm(non_mixup_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Regular]')
            for batch_idx, (audio_data, st_data, labels) in enumerate(regular_pbar):
                audio_data, st_data, labels = audio_data.to(device), st_data.to(device), labels.to(device)
                
                # Forward - use audio_data and st_data, not audio and st
                output = classifier_head(audio_data, st_data)
                
                if type(output) == tuple:
                    logits, gate_weights = output
                    load_balance = classifier_head.compute_load_balance_loss(gate_weights)
                    # Use labels, not targets
                    ce_loss = F.cross_entropy(logits, labels)
                    loss = ce_loss + load_balance
                else:
                    logits = output
                    # Use labels, not targets
                    loss = F.cross_entropy(logits, labels)
                    load_balance = 0
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier_head.parameters(), 1.0)
                optimizer.step()
                
                # Stats
                train_loss += loss.item()
                batch_count += 1
                
                # Accuracy for regular data
                _, predicted = torch.max(logits, 1)
                
                # Use labels, not targets
                batch_size = labels.size(0)
                train_total += batch_size
                train_correct += (predicted == labels).sum().item()
                
                regular_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*train_correct/train_total:.1f}%'
                })
        
        # Process mixup loader
        mixup_pbar = tqdm(mixup_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Mixup]')
        for batch in mixup_pbar:
            audio = batch['audio'].to(device)
            st = batch['st'].to(device)
            targets = batch['target'].to(device)  # Probability distributions
            
            # Forward
            output = classifier_head(audio, st)
            
            if type(output) == tuple:
                logits, gate_weights = output
                load_balance = classifier_head.compute_load_balance_loss(gate_weights)
                ce_loss = F.cross_entropy(logits, targets)
                loss = ce_loss + load_balance
            else:
                logits = output
                loss = F.cross_entropy(logits, targets)
                load_balance = 0
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier_head.parameters(), 1.0)
            optimizer.step()
            
            # Stats
            train_loss += loss.item()
            batch_count += 1
            
            positive_mask = targets > 0
            _, predicted = torch.max(logits, 1)    
            # For each sample, check if predicted class has positive probability
            correct_mask = torch.zeros(predicted.size(0), dtype=torch.bool, device=device)
            for i in range(predicted.size(0)):
                # Check if the predicted class index has positive probability
                if predicted[i] < targets.shape[1]:  # Ensure index is within bounds
                    correct_mask[i] = positive_mask[i, predicted[i]]
                
            batch_correct = correct_mask.sum().item()
            batch_size = targets.size(0)
            train_total += batch_size
            train_correct += batch_correct
            
            mixup_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.1f}%'
            })
        
        
        # Calculate epoch metrics
        epoch_loss = train_loss / batch_count
        epoch_acc = 100. * train_correct / train_total
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.1f}%')
        
        # Save model
        num_experts = str(len(classifier_head.experts))
        save_path = f"{MODEL_SAVE_DIR}/geospatial_mixup_last_ce_{num_experts}_experts_epoch_{epoch}.pth"
        torch.save(classifier_head.state_dict(), save_path)
    
    return classifier_head, history
"""
def train_classifier(
    classifier_head,
    train_loader,
    val_loader=None,  # Add validation loader
    num_epochs=50,
    learning_rate=5e-3,
    device='cuda',
    patience=2  # Early stopping patience
):
    classifier_head = classifier_head.to(device)
    st_postfix = ""
    try:
        if classifier_head.st_embedding_dim == 0:
            st_postfix = "_no_st"
    except:
        print("Model has no st_embedding_dim attribute")
    
    optimizer = optim.AdamW(
        classifier_head.parameters(), 
        lr=learning_rate,
        weight_decay=0.001
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Enhanced history with validation metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"Training classifier with {sum(p.numel() for p in classifier_head.parameters()):,} parameters")
    
    for epoch in range(num_epochs):
        # Training phase
        classifier_head.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        load_balance_total = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (audio_data, st_data, labels) in enumerate(train_pbar):
            audio_data, st_data, labels = audio_data.to(device), st_data.to(device), labels.to(device)
            
            # Forward pass
            output = classifier_head(audio_data, st_data)
            if type(output) == tuple:
                logits, gate_weights = output
                load_balance_loss = classifier_head.compute_load_balance_loss(gate_weights)
                loss = criterion(logits, labels) + load_balance_loss
            else:
                logits = output
                loss = criterion(logits, labels)
                load_balance_loss = 0
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier_head.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            load_balance_total += load_balance_loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Load Balance': f'{load_balance_loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        if val_loader is not None:
            val_loss, val_acc = validate_classifier(classifier_head, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
        else:
            # If no validation, use training loss for early stopping (not ideal)
            val_loss = avg_train_loss
            val_acc = train_acc
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = classifier_head.state_dict().copy()
            print(f'  ? New best validation loss: {val_loss:.4f}')
        else:
            patience_counter += 1
            print(f'  ? No improvement: {patience_counter}/{patience}')
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        if val_loader is not None:
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        
        # Early stopping
        if patience_counter >= patience:
            print(f'?? Early stopping triggered after {epoch+1} epochs!')
            # Load best model before returning
            if best_model_state is not None:
                classifier_head.load_state_dict(best_model_state)
                # Save checkpoint (optional)
            break
    
    # Load the best model found during training
    if best_model_state is not None:
        classifier_head.load_state_dict(best_model_state)
        print("Loaded best model from early stopping.")
    save_path = fMODEL_SAVE_DIR + "/balanced_mixture_of_4_experts_early_fusion_classifier_lr_0006_epoch_{epoch}" + st_suffix + "early_stopping.pth"
    torch.save(classifier_head.state_dict(), save_path)
    print(f"  Model saved: {save_path}")
    
    return classifier_head, history

def validate_classifier(classifier_head, val_loader, criterion, device):
    classifier_head.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc='[Val]')
        for audio_data, st_data, labels in val_pbar:
            audio_data, st_data, labels = audio_data.to(device), st_data.to(device), labels.to(device)
            
            output = classifier_head(audio_data, st_data)
            if type(output) == tuple:
                logits, _ = output
            else:
                logits = output
            
            loss = criterion(logits, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            val_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*val_correct/val_total:.2f}%'
            })
    
    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100. * val_correct / val_total
    return avg_val_loss, val_acc
"""
# Simple test
def create_mixup_loader(embeddings_dir = MIXUP_TRAIN_SAVE_DIR):
    print("Testing mixup loader with precomputed ST embeddings...")
    
    # Load your ST encoder
    location_encoder = SphereMixScaleSpatialRelationEncoder(160, frequency_num=32)
    st_encoder = SpatiotemporalEncoder(location_encoder, loc_dim=2, device='cpu', temporal_encoding_dim=5)
    
    # Create dataloader
    train_loader = create_mixup_dataloader(
        data_dir=embeddings_dir,
        st_encoder=st_encoder,
        num_classes=14795,
        batch_size=8192
    )
    
    # Test one batch
    batch = next(iter(train_loader))
    print(f"? Batch shapes:")
    print(f"   Audio: {batch['audio'].shape}")
    print(f"   ST: {batch['st'].shape}")
    print(f"   Target: {batch['target'].shape}")
    print(f"   Target sum: {batch['target'].sum(dim=1).mean():.3f}")
    
    return train_loader
    


if __name__ == "__main__":
    mixup_loader = create_mixup_loader(MIXUP_TRAIN_SAVE_DIR)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    #classifier_head = SimpleClassifierHead(st_embedding_dim = 0, dropout = 0.3)
    #classifier_head = FullyLinearClassifierHead(st_embedding_dim = 0)
    #classifier_head = MaxoutClassifierHead(st_embedding_dim = 0)
    #classifier_head = MaxoutSTEmbeddingClassifierHead(st_embedding_dim = 165)
    classifier_head = BalancedMoE_ST_EmbeddingClassifierHead(st_embedding_dim = 165, st_hidden_dim = 512, num_experts = 4)
    # old code for non-mixup data
    
    audio_embeddings, spatiotemporal_context, labels = load_st_audio_data(TRAIN_SAVE_DIR)
    print(audio_embeddings.shape, spatiotemporal_context.shape, len(labels))
    print("num_labels", labels.unique().numel())
    # Create dataset
    location_encoder = SphereMixScaleSpatialRelationEncoder(160, frequency_num = 32)
    st_enc = SpatiotemporalEncoder(location_encoder, loc_dim = 2, device = device, temporal_encoding_dim = 5)
    spatiotemporal_encodings = st_enc(spatiotemporal_context).squeeze(1).cpu()
    
    
    
    dataset = ST_AudioDataset(
        audio_embeddings=audio_embeddings,
        spatiotemporal_context=spatiotemporal_encodings,
        labels = labels.cpu()
    )
    # Create DataLoader
    batch_size = 8192

    no_mixup_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=simple_collate_fn)
    
    #val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=simple_collate_fn)
    
    # Train
    trained_classifier, history = train_mixup_classifier(
        classifier_head=classifier_head,
        mixup_loader=mixup_loader,
        non_mixup_loader=no_mixup_loader,
        num_epochs=10,
        learning_rate=6.41e-4,
        device=device
    )
    print(history)
    
    