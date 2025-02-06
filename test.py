import dataset
import nibabel as nib
import processing_helpers
from collections import defaultdict
import whisper
import unidecode
import re
from pydub import AudioSegment
import pandas as pd
from transformers import GPT2Tokenizer, GPT2Model, Wav2Vec2Model, AutoTokenizer, AutoModel, AutoModelForMaskedLM, Wav2Vec2BertModel
from sklearn.decomposition import PCA
import numpy as np
import os
os.chdir(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning")
import matplotlib.pyplot as plt  # For visualization



camembertav2 = AutoModel.from_pretrained("almanach/camembertav2-base")
tokenizer = AutoTokenizer.from_pretrained("almanach/camembertav2-base")
processor = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
processor = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

root_folder = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\fMRI_study\data"
text_data = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\text\conversations"
fmri_data = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\fmri"
audio_data = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\audio"
model_speech = whisper.load_model("large")



import numpy as np
import dataset

img_size = (100, 100, 100)

voxel = list(np.ndindex(img_size))[56799]
participant_list = list(range(1, 4))


data_path = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\fMRI_study\data"
fmri_data_path = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\fmri"
embeddings_path = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\embeddings\text\statements"


database = dataset.BaseDataset(
    participant_list=participant_list,
    data_path=data_path,
    voxel = voxel,
    fmri_data_path=fmri_data_path,
    img_size=img_size,
    embeddings_text_path = embeddings_path# Ensure img_size is a tuple like (100, 100, 10
)

df = database.get_voxel_values(voxel)




for col in df.columns[:3]:  # Loop through the first 3 columns
    df[f'{col}'] = df[col].astype('category').cat.codes

# Initialize the dataset (preprocesses data once)
dataset = dataset.BaseDataset(participant_list, data_path, fmri_data_path, img_size=img_size)

# Iterate through all voxels
voxels = list(np.ndindex(img_size))
for voxel in voxels:
    # Get the entire dataset for this voxel
    dataset_with_voxel = dataset.get_dataset_with_voxel(voxel)
    
    
embeddings_path = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\embeddings\text\statements"
embeddings_text = np.load(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\embeddings\text\statements\SN_1_CLS.npy")

