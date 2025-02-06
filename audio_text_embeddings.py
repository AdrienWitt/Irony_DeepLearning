import os
os.chdir(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning")

import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2Processor, AutoTokenizer, AutoModel, Wav2Vec2BertModel
import torchaudio
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from pydub import AudioSegment
import re
import unidecode
import whisper
from collections import defaultdict
import processing_helpers
import torch
import librosa

output_dir = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\embeddings"

def create_text_cls(text_path, model_text, tokenizer_text, output_dir):
    emplacement = "\\".join(text_path.split("\\")[-2:])
    output_dir = os.path.join(output_dir, emplacement)
    for text_file in os.listdir(text_path):
        file_path = os.path.join(text_path, text_file)
        with torch.no_grad():
            outputs, tokens = processing_helpers.text_embeddings(file_path, tokenizer_text, model_text)
        CLS = outputs[0].detach().numpy()
        CLS_reshaped = CLS.reshape(1,768)
        np.save(os.path.join(output_dir, text_file.replace('.txt', '_CLS.npy')), CLS_reshaped)
        
        
#Create for statements
model_text = AutoModel.from_pretrained("almanach/camembertav2-base")
tokenizer_text = AutoTokenizer.from_pretrained("almanach/camembertav2-base")
statements_path = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\text\statements"
create_text_cls(statements_path, model_text, tokenizer_text, output_dir)
        
# Create audio embeddings

audio_data = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\audio"
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
output_dir = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\embeddings\audio"


def create_audio_embeddings(audio_path, model, processor, output_dir):
    for filename in os.listdir(audio_path):
        file_path = os.path.join(audio_path, filename)
        y, sr = torchaudio.load(file_path)
        y = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(y)
        inputs = processor(y.squeeze(), sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        avg_embedding = (hidden_states[5] + hidden_states[6]) / 2
        avg_embedding = avg_embedding.squeeze().mean(dim=0).numpy()
        avg_embedding_reshaped = avg_embedding.reshape(1,1024)
        np.save(os.path.join(output_dir, filename.replace('.wav', '_layers5-6.npy')), avg_embedding_reshaped)
    
create_audio_embeddings(audio_data, model, processor, output_dir)    
    

           
                
                
                
 
            

        

        