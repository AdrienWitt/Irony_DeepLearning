from torch.utils.data import Dataset
import os
import pandas as pd
import processing_helpers 
import numpy as np
import nibabel as nib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

class BaseDataset(Dataset):
    def __init__(self, participant_list, data_path, fmri_data_path, 
                 pca_threshold=0.50, umap_n_neighbors=15, umap_min_dist=0.1, umap_n_components_text=5, umap_n_components_audio = 18,
                 included_tasks=None, **kwargs):
        super().__init__()
        self.data_path = data_path
        self.fmri_data_path = fmri_data_path
        self.participant_list = participant_list
        self.pca_threshold = pca_threshold
        self.umap_n_neighbors = umap_n_neighbors  # UMAP parameter: local vs global structure
        self.umap_min_dist = umap_min_dist  # UMAP parameter: minimum distance in low-dim space
        self.umap_n_components_text = umap_n_components_text  # UMAP parameter: output dimensions
        self.umap_n_components_audio = umap_n_components_audio  # UMAP parameter: output dimensions
        
        # Set included tasks, with default if not provided
        self.included_tasks = included_tasks or ["sarcasm", "irony", "prosody", "semantic", "tom"]
        
        self.scaler = StandardScaler()
        self.register_args(**kwargs)
        self.base_data = self.create_base_data()
        self.data = self.set_data()

    def register_args(self, **kwargs):
        """Registers additional arguments into the class instance."""
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.kwargs = kwargs
    
    def create_base_data(self):
        """Sets up the data by loading and processing fMRI data."""
        base_data = []
        task_counts = {task: 0 for task in self.included_tasks}
        
        for participant in self.participant_list:
            participant_data_path = os.path.join(self.data_path, participant)
            dfs = processing_helpers.load_dataframe(participant_data_path)
            
            for df in dfs.values():
                df = df.rename(columns=lambda x: x.strip())
                for index, row in df.iterrows():
                    task = row["task"]
                    if task not in self.included_tasks:
                        continue
                    
                    # Increment task counter for statistics
                    task_counts[task] += 1
                    
                    context = row["Context"]
                    statement = row["Statement"]
                    situation = row["Situation"]
                    evaluation = row["Evaluation_Score"]
                    age = row["age"]
                    genre = row["genre"]
                    fmri_file = f'{participant}_{task}_{context[:-4]}_{statement[:-4]}_statement_weighted.nii.gz'
                    fmri_path = os.path.join(self.fmri_data_path, participant, fmri_file)
                    
                    # Extract context and statement conditions
                    parts = row["Condition_name"].split("_")
                    context_cond = parts[0]
                    statement_cond = parts[1]                    
                    img = nib.load(fmri_path).get_fdata(dtype=np.float64)
               
                    # Append the processed data
                    base_data.append({
                        "participant": participant,
                        "task": task,
                        "context": context,
                        "statement": statement,
                        "situation": situation,
                        "fmri_data": img,
                        "context_condition": context_cond,
                        "statement_condition": statement_cond,
                        "evaluation": evaluation,
                        "age": age,
                        "gender": genre
                    })
                    
        # Print statistics on loaded data
        print(f"Loaded {len(base_data)} total samples")
        for task, count in task_counts.items():
            print(f"  - {task}: {count} samples")
            
        return base_data
    
    def apply_pca(self, embeddings_df, prefix):
        """Apply PCA to embeddings."""
        # Scale before PCA
        embeddings_scaled = self.scaler.fit_transform(embeddings_df)
        pca = PCA(n_components=self.pca_threshold)  
        embeddings_pca = pca.fit_transform(embeddings_scaled)
        return pd.DataFrame(embeddings_pca, columns=[f"{prefix}_{i+1}" for i in range(embeddings_pca.shape[1])])

    def apply_umap(self, embeddings_df, umap_n_components, prefix):
        """Apply UMAP to embeddings."""
        # Scale before UMAP
        embeddings_scaled = self.scaler.fit_transform(embeddings_df)
        umap_model = umap.UMAP(
            n_neighbors=self.umap_n_neighbors,
            min_dist=self.umap_min_dist,
            n_components=umap_n_components,
            random_state=42  # For reproducibility
        )
        embeddings_umap = umap_model.fit_transform(embeddings_scaled)
        return pd.DataFrame(embeddings_umap, columns=[f"{prefix}_{i+1}" for i in range(embeddings_umap.shape[1])])

    def set_data(self):
        final_data = []
        embeddings_text_list = []
        embeddings_audio_list = []
        embeddings_weighted_list = []
        
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
                    "age": age, "gender": gender,
                    "fmri_value": fmri_value, "participant": participant
                })  
            else:
                final_data.append({"fmri_value": fmri_value})

            embeddings_text = np.load(os.path.join(self.embeddings_text_path, "statements", f"{semantic}_{item['situation']}_CLS.npy"))
            embeddings_text_list.append(embeddings_text)

            embeddings_audio = np.load(os.path.join(self.embeddings_audio_path, f"{item['statement'].replace('.wav', '_layers5-6.npy')}"))
            embeddings_audio_list.append(embeddings_audio)
            
            embeddings_text_weighted = np.load(os.path.join(self.embeddings_text_path, "text_weighted", f"{context}_{item['situation']}_{semantic}_{item['situation']}_weighted.npy"))
            embeddings_weighted_list.append(embeddings_text_weighted)

        if self.use_base_features:
            df = pd.DataFrame(final_data)
            df.reset_index(drop=True, inplace=True)
            categorical_cols = ['context', 'semantic', 'prosody', 'task', 'gender', 'participant']
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
            df['evaluation'] = df['evaluation'].fillna(df['evaluation'].median())
            df['evaluation'] = (df['evaluation'] - df['evaluation'].min()) / (df['evaluation'].max() - df['evaluation'].min())
            df['age'] = self.scaler.fit_transform(df[['age']])
        else:
            df = pd.DataFrame(final_data)

        # Process text embeddings
        if self.use_text:
            embeddings_df = pd.DataFrame(np.vstack(embeddings_text_list))
            embeddings_df.columns = [f"emb_text_{i}" for i in range(embeddings_df.shape[1])]
            if self.use_umap:
                df_umap_text = self.apply_umap(embeddings_df, self.umap_n_components_text, prefix="umap_text")
                df = pd.concat([df, df_umap_text], axis=1)
            elif self.use_pca:
                df_pca_text = self.apply_pca(embeddings_df, prefix="pc_text")
                df = pd.concat([df, df_pca_text], axis=1)
            else:
                # Scale raw embeddings if no dimensionality reduction
                embeddings_scaled = self.scaler.fit_transform(embeddings_df)
                df_scaled = pd.DataFrame(embeddings_scaled, columns=embeddings_df.columns)
                df = pd.concat([df, df_scaled], axis=1)

        # Process audio embeddings
        if self.use_audio:
            embeddings_df = pd.DataFrame(np.vstack(embeddings_audio_list))
            embeddings_df.columns = [f"emb_audio_{i}" for i in range(embeddings_df.shape[1])]
            if self.use_umap:
                df_umap_audio = self.apply_umap(embeddings_df, self.umap_n_components_audio, prefix="umap_audio")
                df = pd.concat([df, df_umap_audio], axis=1)
            elif self.use_pca:
                df_pca_audio = self.apply_pca(embeddings_df, prefix="pc_audio")
                df = pd.concat([df, df_pca_audio], axis=1)
            else:
                # Scale raw embeddings if no dimensionality reduction
                embeddings_scaled = self.scaler.fit_transform(embeddings_df)
                df_scaled = pd.DataFrame(embeddings_scaled, columns=embeddings_df.columns)
                df = pd.concat([df, df_scaled], axis=1)

        # Process context embeddings
        if self.use_text_weighted:
            embeddings_df = pd.DataFrame(np.vstack(embeddings_weighted_list))
            embeddings_df.columns = [f"emb_weighted_{i}" for i in range(embeddings_df.shape[1])]
            
            if self.use_umap:
                df_umap_weighted = self.apply_umap(embeddings_df, prefix="umap_context")
                df = pd.concat([df, df_umap_weighted], axis=1)
            elif self.use_pca:
                df_pca_weighted = self.apply_pca(embeddings_df, prefix="pc_weighted")
                df = pd.concat([df, df_pca_weighted], axis=1)
            else:
                # Scale raw embeddings if no dimensionality reduction
                embeddings_scaled = self.scaler.fit_transform(embeddings_df)
                df_scaled = pd.DataFrame(embeddings_scaled, columns=embeddings_df.columns)
                df = pd.concat([df, df_scaled], axis=1)

        return df

    def get_voxel_values(self, voxel):
        voxel_values = []
        for item in self.base_data:
            voxel_values.append(item["fmri_data"][voxel])
        self.data["fmri_value"] = voxel_values
        return self.data
    
    def __getitem__(self, index):
        return self.base_data[index]

    def __len__(self):
        return len(self.data)