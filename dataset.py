# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 13:49:24 2024

@author: adywi
"""

from torch.utils.data import Dataset
import os
import pandas as pd
import processing_helpers 
import numpy as np

import nibabel as nib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class BaseDataset(Dataset):
    def __init__(self, participant_list, data_path, fmri_data_path, img_size=(75, 92, 77), mode='base_features', **kwargs):
        super().__init__()
        self.data_path = data_path
        self.fmri_data_path = fmri_data_path
        self.img_size = img_size
        self.participant_list = participant_list
        self.mode = mode
        self.register_args(**kwargs)
        self.scaler = StandardScaler()
        self.n_components_text=22
        self.n_components_audio=217
        self.base_data = self.create_base_data()
        self.data = self.set_data()

    def register_args(self, **kwargs):
        """Registers additional arguments into the class instance."""
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.kwargs = kwargs


    def pad_to_max(self, img):
        """Pads an image to the maximum desired size."""
        y = img
        y_x, y_y, y_z = y.shape[0], y.shape[1], y.shape[2]

        # Calculate padding for each dimension
        x_back, x_for = processing_helpers.pad_back_forward(y_x, self.img_size[0])
        y_back, y_for = processing_helpers.pad_back_forward(y_y, self.img_size[1])
        z_back, z_for = processing_helpers.pad_back_forward(y_z, self.img_size[2])

        # Get background value (first voxel in flattened array)
        #background_value = y.flat[0]

        # Apply padding using NumPy's pad
        padding = [
            (x_back, x_for),  # Padding for X dimension
            (y_back, y_for),  # Padding for Y dimension
            (z_back, z_for)   # Padding for Z dimension
        ]

        y = np.pad(y, padding, mode='constant', constant_values=0)
        return y

    def load_and_pad(self, image_path):
        """Loads an image and pads it to the specified size."""
        img = nib.load(image_path).get_fdata()
        img_padded = self.pad_to_max(img)
        return img_padded
    
    def apply_pca(self, embeddings_list, n_components, prefix):
        """Applies PCA transformation to embeddings."""
        embeddings_df = pd.DataFrame(np.vstack(embeddings_list))
        pca = PCA(n_components=n_components)
        embeddings_pca = pca.fit_transform(embeddings_df)
        return pd.DataFrame(embeddings_pca, columns=[f"{prefix}_{i+1}" for i in range(n_components)])

    def create_base_data(self):
        """Sets up the data by loading and processing fMRI data."""
        base_data = []
        for participant in self.participant_list:
            participant_data_path = os.path.join(self.data_path, participant)
            dfs = processing_helpers.load_dataframe(participant_data_path)
            
            for df in dfs.values():
                df = df.rename(columns=lambda x: x.strip())
                for index, row in df.iterrows():
                    task = row["task"]
                    if task not in ["irony", "sarcasm"]:
                        continue
                    context = row["Context"]
                    statement = row["Statement"]
                    situation = row["Situation"]
                    evaluation = row["Evaluation_Score"]
                    fmri_file = f"{participant}_{task}_{index}_{statement[:-4]}.nii.gz"
                    fmri_path = os.path.join(self.fmri_data_path, participant, fmri_file)
                    
                    # Extract context and statement conditions
                    parts = row["Condition_name"].split("_")
                    context_cond = parts[0]
                    statement_cond = parts[1]                    
                    img_pad = self.load_and_pad(fmri_path)
               
                    # Append the processed data
                    base_data.append({
                        "participant": participant,
                        "task": task,
                        "context": context,
                        "statement": statement,
                        "situation": situation,
                        "fmri_data": img_pad,
                        "context_condition": context_cond,
                        "statement_condition": statement_cond,
                        "evaluation": evaluation
                    })
        return base_data
    

    def set_data(self):
        if self.mode in ['base_features', 'text', 'audio', 'text_audio']:
            final_data = []
            embeddings_text_list = []
            embeddings_audio_list = []
    
            for item in self.base_data:
                fmri_value = None
                context = item["context_condition"]
                semantic = item["statement_condition"][:2]  # First two characters
                prosody = item["statement_condition"][-3:]
                task = item["task"]
                evaluation = item["evaluation"]
    
                if self.use_base_features:
                    final_data.append({
                        "context": context, "semantic": semantic, "prosody": prosody, 
                        "task": task, "evaluation": evaluation, "fmri_value": fmri_value
                    })  
                else:
                    final_data.append({"fmri_value": fmri_value})
    
                embeddings_text = np.load(os.path.join(self.embeddings_text_path, f"{semantic}_{item['situation']}_CLS.npy"))
                embeddings_text_list.append(embeddings_text)
    
                embeddings_audio = np.load(os.path.join(self.embeddings_audio_path, f"{item['statement'].replace('.wav', '_layers5-6.npy')}"))
                embeddings_audio_list.append(embeddings_audio)
    
            if self.use_base_features:
                df = pd.DataFrame(final_data)
                df.reset_index(drop=True, inplace=True)
                categorical_cols = df.columns[:4]  # First 4 columns are categorical
                df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
                df["evaluation"] = df["evaluation"].fillna(df["evaluation"].median())
            else:
                df = pd.DataFrame(final_data)
    
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
    
        # PCA for text embeddings
        if self.mode in ['text', 'text_audio']:
            df_pca_text = self.apply_pca(embeddings_text_list, self.n_components_text, prefix="pc_text")
            df = pd.concat([df, df_pca_text], axis=1)
            embedding_cols = [col for col in df.columns if col.startswith("pc_text_")]
            df[embedding_cols] = self.scaler.fit_transform(df[embedding_cols])
    
        # PCA for audio embeddings
        if self.mode in ['audio', 'text_audio']:
            df_pca_audio = self.apply_pca(embeddings_audio_list, self.n_components_audio, prefix="pc_audio")
            df = pd.concat([df, df_pca_audio], axis=1)
            embedding_cols = [col for col in df.columns if col.startswith("pc_audio_")]
            df[embedding_cols] = self.scaler.fit_transform(df[embedding_cols])
    
        return df

    def get_voxel_values(self, voxel):
        voxel_values = []        
        for item in self.base_data:
            voxel_values.append(item["fmri_data"][voxel])
        self.data["fmri_value"] = voxel_values
        return self.data

    def get_max_image_size(self):
        """Determines the maximum size for x, y, and z dimensions across all images."""
        max_x, max_y, max_z = 0, 0, 0
        
        for participant in self.participant_list:
            participant_data_path = os.path.join(self.data_path, participant)
            dfs = processing_helpers.load_dataframe(participant_data_path)
    
            for df in dfs.values():
                df = df.rename(columns=lambda x: x.strip())
                for index, row in df.iterrows():
                    task = row["task"]
                    statement = row["Statement"]
                    fmri_file = f"{participant}_{task}_{index}_{statement[:-4]}.nii.gz"
                    fmri_path = os.path.join(self.fmri_data_path, participant, fmri_file)
                    
                    if os.path.exists(fmri_path):  # Ensure the file exists before loading
                        img = nib.load(fmri_path).get_fdata()
                        x, y, z = img.shape
                        
                        # Update max dimensions
                        max_x = max(max_x, x)
                        max_y = max(max_y, y)
                        max_z = max(max_z, z)
    
        return max_x, max_y, max_z       
            
    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
        

   
        

            
        
        
        
        
        
        
        

        
    
    
    



    