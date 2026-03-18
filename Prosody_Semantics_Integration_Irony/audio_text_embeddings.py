
import os
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torchaudio
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


contexts_path = r"your context path here with text files"
statements_path = r"your statements path here with text files"
output_dir = r"your output directory for text here"
context_sentence_embeddings(contexts_path, statements_path, output_dir)

###################################################################################################################

audio_path = r"your audio files path for statements here"
output_dir = r"output directory for audio embeddings here"

import opensmile
# Initialize OpenSMILE with eGeMAPS feature set
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)


def create_audio_embeddings(audio_path, output_dir):
    for filename in os.listdir(audio_path):
        file_path = os.path.join(audio_path, filename)
        y, sr = torchaudio.load(file_path)
        features = smile.process_signal(signal=y.squeeze().numpy(), sampling_rate=sr)
        np.save(os.path.join(output_dir, filename.replace('.wav', '_opensmile.npy')), features)
    
create_audio_embeddings(audio_path, output_dir) 

