# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:51:42 2025

@author: adywi
"""

import os
import numpy as np
import librosa
import parselmouth
import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2BertModel, Wav2Vec2BertProcessor
from scipy.stats import pearsonr

def extract_prosodic_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    
    # Pitch (F0)
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    pitch_obj = snd.to_pitch()
    pitch_values = pitch_obj.selected_array['frequency']
    pitch_mean = np.nanmean(pitch_values[pitch_values > 0])
    
    # Energy
    energy = np.mean(librosa.feature.rms(y=y))
    
    # Duration
    duration = librosa.get_duration(y=y, sr=sr)
    
    return pitch_mean, energy

def extract_wav2vec2_layers(file_path, model, processor):
    y, sr = torchaudio.load(file_path)
    y = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(y)
    inputs = processor(y.squeeze(), sampling_rate=16000, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    return [layer.squeeze().mean(dim=0).numpy() for layer in outputs.hidden_states]

def compute_correlations(prosodic_features, wav2vec2_layers):
    correlations = {}

    # Convert lists to numpy arrays for easier operations
    prosodic_features = np.array(prosodic_features)  # Shape: (n_files, 3)
    wav2vec2_layers = np.array(wav2vec2_layers)  # Shape: (n_files, n_layers, feature_dim)

    num_layers = wav2vec2_layers.shape[1]
    
    for layer_idx in range(num_layers):
        layer_values = wav2vec2_layers[:, layer_idx, :].mean(axis=1)  # Average across feature dimensions
        corrs = []
        
        for i in range(prosodic_features.shape[1]):  # Iterate over pitch, energy, duration
            if np.std(prosodic_features[:, i]) > 0 and np.std(layer_values) > 0:
                corr, _ = pearsonr(prosodic_features[:, i], layer_values)
            else:
                corr = 0  # Set to 0 if one of them is constant
            
            corrs.append(corr)

        correlations[f'Layer_{layer_idx}'] = corrs
    
    return correlations

folder_path = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\audio"
# model = Wav2Vec2BertModel.from_pretrained("hf-audio/wav2vec2-bert-CV16-en")
# processor = Wav2Vec2BertProcessor.from_pretrained("hf-audio/wav2vec2-bert-CV16-en")

model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")


all_prosodic = []
all_wav2vec2 = []

for file in os.listdir(folder_path):
    if file.endswith(".wav") and file.startswith("S"):
        file_path = os.path.join(folder_path, file)
        prosodic_features = extract_prosodic_features(file_path)
        wav2vec2_layers = extract_wav2vec2_layers(file_path, model, processor)
        
        all_prosodic.append(prosodic_features)
        all_wav2vec2.append(wav2vec2_layers)

all_prosodic = np.array(all_prosodic)
all_wav2vec2 = np.array(all_wav2vec2)  # Shape: (n_files, n_layers, feature_dim)

correlations = compute_correlations(all_prosodic, all_wav2vec2)

