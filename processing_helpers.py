# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:58:33 2024

@author: adywi
"""

from transformers import GPT2Tokenizer, GPT2Model, Wav2Vec2Model, AutoTokenizer, AutoModel, AutoModelForMaskedLM
from sklearn.decomposition import PCA
import numpy as np
import os
import pandas as pd
from pydub import AudioSegment
import re
import unidecode
import whisper
from collections import defaultdict
import librosa

def load_dataframe(participant_path):
    file_list = [f for f in os.listdir(participant_path) if f.startswith('Resultfile')]
    dfs = {}
    for file_name in file_list:
        full_path = os.path.join(participant_path, file_name)
        df = pd.read_csv(full_path, sep='\t') 
        key = file_name[-8:-4]
        dfs[key] = df
    return dfs


def clean_text(text):
    text = text.lower()
    #text = unidecode.unidecode(text)
    #text = re.sub(r"[^\w\s']", '', text)
    return text


def text_embeddings(text_file, tokenizer_text, model_text):
    with open(text_file, "r") as file:
                    text = file.read()  
    text_cleaned = clean_text(text)
    inputs = tokenizer_text(text_cleaned, return_tensors="pt")
    tokens = tokenizer_text.convert_ids_to_tokens(inputs["input_ids"][0])
    tokens = tokens 
    outputs = model_text(**inputs).last_hidden_state.squeeze(0)
    return outputs, tokens

def audio_embeddings(audio_fie, processor, model_audio):
    audio, sampling_rate = librosa.load(audio_path, sr=16000)

def pad_back_forward(length, desired_length):
    """Calculates the padding needed for back and forward dimensions."""
    pad_back = (desired_length - length) // 2
    pad_forward = desired_length - length - pad_back
    return pad_back, pad_forward


def words_to_tokens_mapping(text_file, audio_file, model_speech, tokenizer_text, model_text):
    #word_to_token_mapping = processing_helpers.words_to_tokens_mapping(text_path, audio_path, model_speech, tokenizer, model_text)
    
    result_large = model_speech.transcribe(audio_file, word_timestamps=True)
    word_time_stamp = []
    for segment in result_large["segments"]:
        for word_data in segment["words"]:
            word = word_data["word"].strip()
            start_time = word_data["start"]              
            word_time_stamp.append({"word": clean_text(word), "start_time": start_time})
                      
            
    outputs, tokens = text_embeddings(text_file, tokenizer_text, model_text)                
    tokens = tokens[1:-1] 

    CLS = outputs[0] 
    SEP  = outputs[-1]            
    embeddings = outputs[1:-1]  
           

    word_to_token_mapping = []
    current_word_index = 0        
    i = 0  # Token index
    
    while i < len(tokens):
        if current_word_index >= len(word_time_stamp):
            print(f"No more words in word_time_stamp, but tokens remain starting from index {i}")
            break

        token = tokens[i]
        frame_index = int(word_time_stamp[current_word_index]["start_time"] // 0.65) if current_word_index < len(word_time_stamp) else -1
        print({word_time_stamp[current_word_index]["start_time"]}, {word_time_stamp[current_word_index]["start_time"] // 0.65})
        
        # === Step 1: Direct match ===
        if token == word_time_stamp[current_word_index]["word"]:
            word_to_token_mapping.append({
                "word": token,
                "token": token,
                "frame_index": frame_index,
            })
            print(f"Direct match: '{token}' to '{word_time_stamp[current_word_index]['word']}'")
            current_word_index += 1
            i += 1
            continue
        
        # === Step 2: Sliding-window token combination ===
        combined_token = token  # Start with the current token
        
        for j in range(i + 1, len(tokens)):
            sub_token = tokens[j][2:] if tokens[j].startswith("##") else tokens[j]
            combined_token += sub_token
            
            if current_word_index < len(word_time_stamp) and combined_token == word_time_stamp[current_word_index]["word"]:
                for k in range(i, j + 1):
                    word_to_token_mapping.append({
                        "word": combined_token,
                        "token": tokens[k],
                        "frame_index": frame_index,
                    })
                print(f"Match: '{combined_token}' to '{word_time_stamp[current_word_index]['word']}' using tokens {tokens[i:j+1]}")
                current_word_index += 1
                i = j + 1
                break
        else:
            # === Step 3: Handle special case for "l'" being split into "l" and "'envers" ===
            if token.endswith("’") and current_word_index + 1 < len(word_time_stamp):
                base = token[:-1]  # Remove the apostrophe from the token
                if base == word_time_stamp[current_word_index]["word"]:
                    # Map "l" from "l'" to "l"
                    word_to_token_mapping.append({
                        "word": base,
                        "token": token,
                        "frame_index": frame_index,
                    })
                    print(f"Special split match: '{base}' to '{word_time_stamp[current_word_index]['word']}' using token '{token}'")
                    current_word_index += 1
                    i += 1  # Move to the next token

                    # Now match the second part "'envers"
                    if current_word_index < len(word_time_stamp) and word_time_stamp[current_word_index]["word"] == "'" + tokens[i]:
                        word_to_token_mapping.append({
                            "word": word_time_stamp[current_word_index]["word"],
                            "token": tokens[i],
                            "frame_index": frame_index,
                        })
                        print(f"Match apostrophe word: '{word_time_stamp[current_word_index]['word']}' to token '{tokens[i]}'")
                        current_word_index += 1
                        i += 1
                        continue
                    
                    if current_word_index < len(word_time_stamp) and word_time_stamp[current_word_index]["word"] != "'" + tokens[i]:
                        combined_token = "'" + tokens[i]
                        
                        for j in range(i + 1, len(tokens)):
                            sub_token = tokens[j][2:] if tokens[j].startswith("##") else tokens[j]
                            combined_token += sub_token
                            
                            if current_word_index < len(word_time_stamp) and combined_token == word_time_stamp[current_word_index]["word"]:
                                for k in range(i, j + 1):
                                    word_to_token_mapping.append({
                                        "word": combined_token,
                                        "token": tokens[k],
                                        "frame_index": frame_index,
                                    })
                                print(f"Match: '{combined_token}' to '{word_time_stamp[current_word_index]['word']}' using tokens {tokens[i:j+1]}")
                                current_word_index += 1
                                i = j + 1
                                break            
                    
                    if current_word_index < len(word_time_stamp) and token == ".":
                        if word_time_stamp[current_word_index]["word"] == "!":
                            # Special case handling where token is "." but word_time_stamp is "!"
                            word_to_token_mapping.append({
                                "word": "!",  # Add the "!" to the mapping
                                "token": token,  # Add the token which is "."
                                "frame_index": frame_index,
                            })
                            print(f"Special case match: '{token}' (period) to '{word_time_stamp[current_word_index]['word']}' (exclamation mark)")
                            current_word_index += 1
                            i += 1  # Move to the next token
                            continue  # Skip to the next token
    return word_to_token_mapping

