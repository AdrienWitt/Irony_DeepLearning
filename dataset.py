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
    def __init__(self, participant_list, data_path, fmri_data_path, img_size=(75, 92, 77), pca_threshold = 0.95, **kwargs):
        super().__init__()
        self.data_path = data_path
        self.fmri_data_path = fmri_data_path
        self.img_size = img_size
        self.participant_list = participant_list
        self.register_args(**kwargs)
        self.scaler = StandardScaler()
        self.pca_threshold = pca_threshold
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
                    age = row["age"]
                    genre = row["genre"]
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
                        "evaluation": evaluation,
                        "age": age,
                        "gender": genre
                    })
        return base_data
    
    def apply_pca(self, embeddings_df, prefix):
        pca = PCA(n_components=self.pca_threshold)  
        embeddings_pca = pca.fit_transform(embeddings_df)
        return pd.DataFrame(embeddings_pca, columns=[f"{prefix}_{i+1}" for i in range(embeddings_pca.shape[1])])
    

    def set_data(self):
        final_data = []
        embeddings_text_list = []
        embeddings_audio_list = []
        embeddings_context_list = []
        
        for item in self.base_data:
            fmri_value = None
            context = item["context_condition"]
            semantic = item["statement_condition"][:2]
            prosody = item["statement_condition"][-3:]
            task = item["task"]
            evaluation = item["evaluation"]
            participant = item["participant"]
            age = item["age"]
            gender = item["gender"]

            if self.use_base_features:
                final_data.append({
                    "context": context, "semantic": semantic, "prosody": prosody, 
                    "task": task, "evaluation": evaluation, 
                    "age": age, "gender": gender, "participant": participant,
                    "fmri_value": fmri_value,
                })  
            else:
                final_data.append({"fmri_value": fmri_value})

            embeddings_text = np.load(os.path.join(self.embeddings_text_path, "statements", f"{semantic}_{item['situation']}_CLS.npy"))
            embeddings_text_list.append(embeddings_text)

            embeddings_audio = np.load(os.path.join(self.embeddings_audio_path, f"{item['statement'].replace('.wav', '_layers5-6.npy')}"))
            embeddings_audio_list.append(embeddings_audio)
            
            embeddings_context = np.load(os.path.join(self.embeddings_text_path, "contexts", f"{context}_{item['situation']}_CLS.npy"))
            embeddings_context_list.append(embeddings_context)

        if self.use_base_features:
            df = pd.DataFrame(final_data)
            df.reset_index(drop=True, inplace=True)
            
            # Define categorical columns (including participant)
            categorical_cols = ['context', 'semantic', 'prosody', 'task', 'gender', 'participant']
            

            # One-hot encode categorical variables
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
            
            # Handle evaluation as ordinal - normalize to [0,1] range
            df['evaluation'] = df['evaluation'].fillna(df['evaluation'].median())
            df['evaluation'] = (df['evaluation'] - df['evaluation'].min()) / (df['evaluation'].max() - df['evaluation'].min())
            
            # Scale age (continuous variable)
            df['age'] = self.scaler.fit_transform(df[['age']])

        else:
            df = pd.DataFrame(final_data)

        if self.use_text:
            embeddings_df = pd.DataFrame(np.vstack(embeddings_text_list))
            embeddings_df.columns = [f"emb_text_{i}" for i in range(embeddings_df.shape[1])]  # Rename columns
            if self.use_pca:
                df_pca_text = self.apply_pca(embeddings_df, prefix="pc_text")
                df = pd.concat([df, df_pca_text], axis=1)
                embedding_cols = [col for col in df.columns if col.startswith("pc_text_")]
                df[embedding_cols] = self.scaler.fit_transform(df[embedding_cols])
            else:
                df = pd.concat([df, embeddings_df], axis=1)
                embedding_cols = [col for col in df.columns if col.startswith("emb_text_")]
                df[embedding_cols] = self.scaler.fit_transform(df[embedding_cols])
                    
        if self.use_audio:
            embeddings_df = pd.DataFrame(np.vstack(embeddings_audio_list))
            embeddings_df.columns = [f"emb_audio_{i}" for i in range(embeddings_df.shape[1])]  # Rename columns
            if self.use_pca:
                df_pca_audio = self.apply_pca(embeddings_df, prefix="pc_audio")
                df = pd.concat([df, df_pca_audio], axis=1)
                embedding_cols = [col for col in df.columns if col.startswith("pc_audio_")]
                df[embedding_cols] = self.scaler.fit_transform(df[embedding_cols])
            else:
                df = pd.concat([df, embeddings_df], axis=1)
                embedding_cols = [col for col in df.columns if col.startswith("emb_audio_")]
                df[embedding_cols] = self.scaler.fit_transform(df[embedding_cols])
                
        if self.use_context:
            embeddings_df = pd.DataFrame(np.vstack(embeddings_context_list))
            embeddings_df.columns = [f"emb_context_{i}" for i in range(embeddings_df.shape[1])]  # Rename columns
            if self.use_pca:
                df_pca_audio = self.apply_pca(embeddings_df, prefix="pc_context")
                df = pd.concat([df, df_pca_audio], axis=1)
                embedding_cols = [col for col in df.columns if col.startswith("pc_context_")]
                df[embedding_cols] = self.scaler.fit_transform(df[embedding_cols])
            else:
                df = pd.concat([df, embeddings_df], axis=1)
                embedding_cols = [col for col in df.columns if col.startswith("emb_context_")]
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
        

   
        

            
        
        
        
        
        
        
        

        
    
    
    



    