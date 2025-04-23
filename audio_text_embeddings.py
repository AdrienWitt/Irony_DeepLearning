import os
os.chdir(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning")

import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2Processor, AutoTokenizer, AutoModel, Wav2Vec2BertModel
import torchaudio
import numpy as np
import pandas as pd
from pydub import AudioSegment
import re
import whisper
from collections import defaultdict
import processing_helpers
import torch
import librosa
from sentence_transformers import SentenceTransformer

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

context_path = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\text\contexts"
create_text_cls(context_path, model_text, tokenizer_text, output_dir)

###################################################################################################################################        
# Create audio embeddings

audio_path = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\audio"
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
    
create_audio_embeddings(audio_path, model, processor, output_dir)    
    
#####################################################################################################################################
import os
import numpy as np
from transformers import CamembertModel, CamembertTokenizer
import torch

def context_sentence_embeddings(contexts_path, statements_path, output_dir='./'):
    combined_output_dir = os.path.join(output_dir, "text_combined")
    os.makedirs(combined_output_dir, exist_ok=True)
    
    text_model = AutoModel.from_pretrained("almanach/camembertav2-base")
    tokenizer = AutoTokenizer.from_pretrained("almanach/camembertav2-base")
    
    # Load contexts and statements
    contexts, statements = {}, {}
    for context_file in os.listdir(contexts_path):
        if context_file.endswith('.txt'):
            scenario = context_file.split('_')[-1].split('.')[0]
            with open(os.path.join(contexts_path, context_file), "r") as f:
                contexts.setdefault(scenario, []).append((context_file, f.read().strip()))
    for statement_file in os.listdir(statements_path):
        if statement_file.endswith('.txt'):
            scenario = statement_file.split('_')[-1].split('.')[0]
            with open(os.path.join(statements_path, statement_file), "r") as f:
                statements.setdefault(scenario, []).append((statement_file, f.read().strip()))
    
    pairs_count = sum(len(contexts[s]) * len(statements[s]) for s in contexts.keys() if s in statements)
    print(f"Found {len(contexts)} scenarios with contexts")
    print(f"Found {len(statements)} scenarios with statements")
    print(f"Will generate {pairs_count} text embeddings")
    
    # Process pairs
    pair_count = 0
    for scenario, scenario_contexts in contexts.items():
        if scenario not in statements:
            print(f"Warning: No statements found for scenario {scenario}")
            continue
        for context_file, context_text in scenario_contexts:
            inputs_A = tokenizer(context_text, return_tensors="pt")
            with torch.no_grad():
                outputs_A = text_model(**inputs_A)
                emb_A = outputs_A.last_hidden_state.mean(dim=1)  # (1, 768)
            
            for statement_file, statement_text in statements[scenario]:
                inputs_B = tokenizer(statement_text, return_tensors="pt")
                with torch.no_grad():
                    outputs_B = text_model(**inputs_B)
                    emb_B_tokens = outputs_B.last_hidden_state  # (1, seq_len, 768)
                    attn_scores = torch.softmax(torch.matmul(emb_B_tokens, emb_A.transpose(-1, -2)), dim=1)
                    emb_B_weighted = (emb_B_tokens * attn_scores).mean(dim=1)  # (1, 768)
                    diff_embedding = emb_B_weighted - emb_A  # (1, 768)
                    text_embedding = torch.cat([emb_B_weighted, diff_embedding], dim=1).numpy()  # (1, 1536)
                
                # Save embedding
                combined_filename = f"{os.path.splitext(context_file)[0]}_{os.path.splitext(statement_file)[0]}_weighted.npy"
                np.save(os.path.join(combined_output_dir, combined_filename), text_embedding)
                
                pair_count += 1
                if pair_count % 10 == 0:
                    print(f"Processed {pair_count}/{pairs_count} pairs")
    
    print(f"Generated {pair_count} text embeddings in {combined_output_dir}")

# Example usage


contexts_path = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\text\contexts"
statements_path = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\text\statements"
output_dir = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\embeddings\text"
context_sentence_embeddings(contexts_path, statements_path, output_dir)

###################################################################################################################

audio_path = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\audio"
output_dir = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\embeddings\audio_opensmile"

import opensmile
# Initialize OpenSMILE with eGeMAPS feature set
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)


def create_audio_embeddings(audio_path, model, processor, output_dir):
    for filename in os.listdir(audio_path):
        file_path = os.path.join(audio_path, filename)
        y, sr = torchaudio.load(file_path)
        features = smile.process_signal(signal=y.squeeze().numpy(), sampling_rate=sr)
        np.save(os.path.join(output_dir, filename.replace('.wav', '_opensmile.npy')), features)
    
create_audio_embeddings(audio_path, model, processor, output_dir) 

