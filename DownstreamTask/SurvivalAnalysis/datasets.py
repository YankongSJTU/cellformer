import torch
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
class TumorDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.samples = df['samplename'].values
        self.features = df.drop(columns=['samplename', 'time', 'status']).values
        self.times = df['time'].values
        self.events = df['status'].values
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.features[idx]),
            self.times[idx],
            self.events[idx],
            self.samples[idx]  #
        )
