import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn.utils.rnn as rnn_utils
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.preprocessing import LabelEncoder

class SlideLevelDataset(Dataset):
    def __init__(self, data_csv, label_txt):
        self.df = pd.read_csv(data_csv)
        self.df['slide_id'] = self.df.iloc[:, 0].apply(lambda x: x.split('_')[0])
        
        label_df = pd.read_csv(label_txt, sep='\t', header=None,
                             names=['tumor_type', 'sample'])
        self.le = LabelEncoder()
        self.tumor_types = self.le.fit_transform(label_df['tumor_type'])  
        
        self.label_dict = dict(zip(label_df['sample'], self.tumor_types))
        self.slide_groups = defaultdict(list)
        for idx, row in self.df.iterrows():
            slide_id = row['slide_id']
            features = row[1:-1].values.astype(np.float32)  
            self.slide_groups[slide_id].append(features)
            
        self.slide_ids = [sid for sid in self.slide_groups.keys() if sid in self.label_dict]
        self.labels = [self.label_dict[sid] for sid in self.slide_ids]
        
        all_features = np.vstack([np.vstack(feats) for feats in self.slide_groups.values()])
        self.scaler = StandardScaler()
        self.scaler.fit(all_features)
        
    def __len__(self):
        return len(self.slide_ids)
    
    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        features = np.vstack(self.slide_groups[slide_id])
        features = self.scaler.transform(features)
        label = self.labels[idx]
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'slide_id': slide_id
        }

class TumorPatchDataset(Dataset):
    def __init__(self, csv_path, scaler=None):
        self.df = pd.read_csv(csv_path)
        
        self.df['slide_id'] = self.df.iloc[:, 0].apply(lambda x: x.split('_')[0])
        self.df['coords'] = self.df.iloc[:, 0].apply(self._parse_coords)
        
        numeric_cols = [col for col in self.df.columns if col not in [self.df.columns[0], 'slide_id', 'coords']]
        self.features = self.df[numeric_cols].values.astype(np.float32)
        
        if scaler:
            self.features = scaler.transform(self.features)
        else:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
            
        self.slide_groups = defaultdict(list)
        self.slide_coords = defaultdict(list)
        for idx, row in self.df.iterrows():
            slide_id = row['slide_id']
            self.slide_groups[slide_id].append(self.features[idx])
            self.slide_coords[slide_id].append(row['coords'])
            
        self.slide_ids = list(self.slide_groups.keys())
        
    def _parse_coords(self, name):
        """slideid_x_y.png"""
        try:
            parts = name.split('_')
            if len(parts) >= 3:
                x = int(parts[-2])
                y = int(parts[-1].split('.')[0])
                return (x, y)
        except:
            return (0, 0)
        return (0, 0)
    
    def __len__(self):
        return len(self.slide_ids)
    
    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        features = np.array(self.slide_groups[slide_id])
        coords = np.array(self.slide_coords[slide_id])
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'coords': torch.tensor(coords, dtype=torch.int32),
            'slide_id': slide_id
        }


def collate_fn(batch):
    """patch collate """
    features = [item['features'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    slide_ids = [item['slide_id'] for item in batch]
    
    features_padded = rnn_utils.pad_sequence(features, batch_first=True, padding_value=0)
    
    masks = torch.ones_like(features_padded[:, :, 0])  
    
    for i, feat in enumerate(features):
        masks[i, len(feat):] = 0
    
    return {
        'features': features_padded,
        'masks': masks,
        'labels': labels,
        'slide_ids': slide_ids
    }

class EnhancedAttentionMIL(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.feature_extractor = nn.ModuleDict({
            'pathway1': nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.GELU(),
                nn.LayerNorm(1024),
                nn.Dropout(0.3)
            ),
            'pathway2': nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.GELU(),
                nn.LayerNorm(512),
                nn.Dropout(0.3),
                nn.Linear(512, 1024),
                nn.GELU(),
                nn.LayerNorm(1024),
                nn.Dropout(0.3)
            )
        })
        self.cross_attention = nn.MultiheadAttention(embed_dim=1024, num_heads=8, dropout=0.2)
        self.fusion_gate = nn.Sequential(
            nn.Linear(1024*2, 512),
            nn.GELU(),
            nn.Linear(512, 2),
            nn.Softmax(dim=-1)
        )
        self.hierarchical_attention = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        self.aux_classifier = nn.Linear(1024, num_classes)

    def forward(self, x, mask=None):
        h1 = self.feature_extractor['pathway1'](x)  # [B, N, 1024]
        h2 = self.feature_extractor['pathway2'](x)  # [B, N, 1024]
        
        h1_original = h1.clone()
        
        h1 = h1.transpose(0, 1)  # [N, B, 1024]
        h2 = h2.transpose(0, 1)  # [N, B, 1024]
        attn_output, _ = self.cross_attention(h1, h2, h2)  # [N, B, 1024]
        attn_output = attn_output.transpose(0, 1)  # [B, N, 1024]
        
        gate_weights = self.fusion_gate(torch.cat([h1_original, attn_output], dim=-1))  # [B, N, 2]
        fused_features = gate_weights[:,:,0:1] * h1_original + gate_weights[:,:,1:2] * attn_output
        
        a = self.hierarchical_attention(fused_features)  # [B, N, 1]
        if mask is not None:
            a = a.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        a = F.softmax(a, dim=1)
        weighted_features = (a * fused_features).sum(dim=1)  # [B, 1024]
        
        logits = self.classifier(weighted_features)
        aux_logits = self.aux_classifier(fused_features.mean(dim=1))  
        
        return F.softmax(logits, dim=-1), a.squeeze(-1), F.softmax(aux_logits, dim=-1)
def train_model(data_csv, label_txt, num_classes=24, epochs=50, patience=5):
    dataset = SlideLevelDataset(data_csv, label_txt)
    
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(dataset.le, f)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(dataset.scaler, f)
    
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=400, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=400, shuffle=False, collate_fn=collate_fn)
    
    input_dim = dataset[0]['features'].shape[1]
#    model = AttentionMIL(input_dim, num_classes).to(device)
    model = EnhancedAttentionMIL(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
#    criterion = nn.CrossEntropyLoss()
    def criterion(logits, aux_logits, labels, alpha=0.3):
        main_loss = F.cross_entropy(logits, labels)
        aux_loss = F.cross_entropy(aux_logits, labels)
        return (1-alpha)*main_loss + alpha*aux_loss
    
    best_val_loss = float('inf')
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            features = batch['features'].to(device)
            masks = batch['masks'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            probs, _,aux_probs = model(features, masks)
            loss = criterion(probs,aux_probs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = probs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                masks = batch['masks'].to(device)
                labels = batch['labels'].to(device)
                
                probs, _ ,aux_probs= model(features, masks)
                loss = criterion(probs,aux_probs, labels)
                
                val_loss += loss.item()
                _, predicted = probs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100. * correct / total
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(torch.load('best_model.pth'))
    return model, dataset.scaler

def visualize_attention(model, dataset, output_dir="attention_maps"):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    def single_collate_fn(batch):
        return {
            'features': batch[0]['features'].unsqueeze(0),  
            'coords': batch[0]['coords'].unsqueeze(0),
            'slide_id': [batch[0]['slide_id']]
        }
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=single_collate_fn)
    
    for batch in tqdm(loader, desc="Generating attention maps"):
        features = batch['features'].to(device)
        coords = batch['coords'][0].cpu().numpy()  
        slide_id = batch['slide_id'][0]
        
        with torch.no_grad():
            probs, attention,_ = model(features)
        
        
        pred_class = probs.argmax(dim=1).item()
        pred_prob = probs[0, pred_class].item()
        
        attention_weights = attention.squeeze().cpu().numpy()
        
        plt.figure(figsize=(15, 10))
        ax = plt.gca()

        ax.invert_yaxis()

        sc = plt.scatter(coords[:, 0], coords[:, 1], c=attention_weights, cmap='turbo', s=50, alpha=0.7)
        plt.colorbar(sc, label='Attention Weight')
        
        k = max(1, int(len(attention_weights)*0.1))
        top_indices = np.argpartition(attention_weights, -k)[-k:]
#        plt.scatter(coords[top_indices, 0], coords[top_indices, 1], edgecolors='red', facecolors='none', s=100, linewidths=1.5)
        
        plt.title(f"Slide: {slide_id}\nPredicted Class: {pred_class} (Prob: {pred_prob:.2f})")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(output_dir, f"{slide_id}_attention.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved attention map for {slide_id} to {output_path}")

def predict_slides(model, feature_csv, scaler, label_encoder, output_dir="results"):
    # 
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Data
    dataset = TumorPatchDataset(feature_csv, scaler)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    results = []
    
    for batch in tqdm(loader, desc="Processing slides"):
        slide_id = batch['slide_id'][0]
        features = batch['features'].to(device)
        coords = batch['coords'][0].cpu().numpy()
        
        with torch.no_grad():
            # prediction and attention
            probs, attention, _ = model(features)  # [1, num_classes], [1, num_patches]
            
            # tumor type
            pred_class_idx = probs.argmax(dim=1).item()
            pred_class = label_encoder.classes_[pred_class_idx]
            pred_prob = probs[0, pred_class_idx].item()
            
            # save results
            results.append({
                'slide_id': slide_id,
                'pred_class': pred_class,
                'pred_prob': pred_prob,
                'all_probs': probs.cpu().numpy()[0]
            })
            
            # tumor type and attention heatmap
            plot_attention_map(
                coords=coords,
                attention=attention[0].cpu().numpy(),  # 取第一个batch
                slide_id=slide_id,
                pred_class=pred_class,
                pred_prob=pred_prob,
                output_dir=output_dir
            )
    
    # save results
    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(output_dir, "slide_predictions.csv"), index=False)
    return result_df

def plot_attention_map(coords, attention, slide_id, pred_class, pred_prob, output_dir):
    plt.figure(figsize=(15, 10))
    ax = plt.gca()
    ax.invert_yaxis()
    sc = plt.scatter(coords[:, 0], coords[:, 1], 
                    c=attention, cmap='viridis', 
                    s=50, alpha=0.7)
    plt.colorbar(sc, label='Attention Weight')
    
    # high attention region（Top 10%）
    k = max(1, int(len(attention)*0.1))
    top_indices = np.argpartition(attention, -k)[-k:]
    #plt.scatter(coords[top_indices, 0], coords[top_indices, 1],
    #           edgecolors='red', facecolors='none',
    #           s=100, linewidths=1.5)
    
    plt.title(f"Slide: {slide_id}\nPredicted: {pred_class} (Probability: {pred_prob:.2f})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, alpha=0.3)
    
    #  
    output_path = os.path.join(output_dir, f"{slide_id}_{pred_class}_attention.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
if __name__ == "__main__":
    print("=== Training Phase ===")
    model, scaler = train_model(
        data_csv="0",  
        #data_csv="data.csv",  #format: slideid_...,feat1,feat2,...  
        label_txt="sampletype.txt",  #format: tumor/normal<tab>slideid
        num_classes=24,  
        epochs=10,
        patience=10
    )
    
    print("\n=== Prediction Phase ===")
    test_dataset = TumorPatchDataset(
        'feature.csv',  #format: slideid_x_y.png,feat1,feat2,...
        scaler=scaler
    )
    visualize_attention(model, test_dataset)
    
    print("\nAll done! Attention maps saved in 'attention_maps' folder.")

    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load pretrained scaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # For new data 
    predictions = predict_slides(
        model=model,
        feature_csv="feature.csv",
        scaler=scaler,
        label_encoder=label_encoder,
        output_dir="new_results"
    )
    
    # print results
    print(predictions[['slide_id', 'pred_class', 'pred_prob']].head())
