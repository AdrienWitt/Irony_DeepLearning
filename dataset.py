from torch.utils.data import Dataset
import os
import pandas as pd
import processing_helpers 
import numpy as np
import nibabel as nib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
from concurrent.futures import ThreadPoolExecutor


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
                    fmri_file = f'{participant}_{task}_{context[:-4]}_{statement[:-4]}_statement_masked.nii.gz'
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
        embeddings_audio_opensmile_list = []
        
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
            
            embeddings_audio_opensmile = np.load(os.path.join(self.embeddings_audio_opensmile_path, f"{item['statement'].replace('.wav', '_opensmile.npy')}"))
            embeddings_audio_opensmile_list.append(embeddings_audio_opensmile)

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
                
        # Process openSMILE audio features
        if self.use_audio_opensmile:
            embeddings_df = pd.DataFrame(np.vstack(embeddings_audio_opensmile_list))
            embeddings_df.columns = [f"emb_audio_opensmile_{i}" for i in range(embeddings_df.shape[1])]
            if self.use_umap:
                df_umap_opensmile = self.apply_umap(embeddings_df, min(10, embeddings_df.shape[1]), prefix="umap_audio_opensmile")
                df = pd.concat([df, df_umap_opensmile], axis=1)
            elif self.use_pca:
                df_pca_opensmile = self.apply_pca(embeddings_df, prefix="pc_audio_opensmile")
                df = pd.concat([df, df_pca_opensmile], axis=1)
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
    
class WholeBrainDataset(Dataset):
    def __init__(self, participant_list, data_path, fmri_data_path, mask,
                 included_tasks=None, use_base_features=True, 
                 use_text=True, use_text_weighted=True, 
                 use_audio=True, use_audio_opensmile=True, 
                 embeddings_text_path=None, embeddings_text_weighted_path=None,
                 embeddings_audio_path=None, embeddings_audio_opensmile_path=None,
                 pca_threshold=0.50, use_pca=False, use_umap=False, **kwargs):
        super().__init__()
        self.data_path = data_path
        self.fmri_data_path = fmri_data_path
        self.participant_list = participant_list
        self.included_tasks = included_tasks or ["sarcasm", "irony", "prosody", "semantic", "tom"]
        self.use_base_features = use_base_features
        self.use_text = use_text
        self.use_text_weighted = use_text_weighted
        self.use_audio = use_audio
        self.use_audio_opensmile = use_audio_opensmile
        self.embeddings_text_path = embeddings_text_path
        self.embeddings_text_weighted_path = embeddings_text_weighted_path
        self.embeddings_audio_path = embeddings_audio_path
        self.embeddings_audio_opensmile_path = embeddings_audio_opensmile_path
        self.pca_threshold = pca_threshold
        self.use_pca = use_pca
        self.use_umap = use_umap
        self.scaler = StandardScaler()
        self.register_args(**kwargs)
        self.mask = mask
        self.fmri_cache = self.preload_fmri()
        self.data, self.fmri_data, self.ids_list = self.create_data()

    def register_args(self, **kwargs):
        """Registers additional arguments into the class instance."""
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.kwargs = kwargs
    
    def preload_fmri(self):
        """Preload all fMRI .npy files into a cache."""
        fmri_cache = {}
        for participant in self.participant_list:
            participant_fmri_path = os.path.join(self.fmri_data_path, participant)
            if not os.path.exists(participant_fmri_path):
                continue
            for fmri_file in os.listdir(participant_fmri_path):
                if fmri_file.endswith('.npy'):
                    fmri_cache[f"{participant}/{fmri_file}"] = np.load(os.path.join(participant_fmri_path, fmri_file), mmap_mode='r')
        return fmri_cache

    def apply_pca(self, embeddings_df, prefix):
        """Apply PCA to embeddings."""
        embeddings_scaled = self.scaler.fit_transform(embeddings_df)
        pca = PCA(n_components=self.pca_threshold)
        embeddings_pca = pca.fit_transform(embeddings_scaled)
        return pd.DataFrame(embeddings_pca, columns=[f"{prefix}_{i+1}" for i in range(embeddings_pca.shape[1])])

    def process_participant(self, participant):
        """Process data for a single participant in parallel."""
        participant_data_path = os.path.join(self.data_path, participant)
        dfs = processing_helpers.load_dataframe(participant_data_path)
        final_data, fmri_data_list, ids_list = [], [], []
        embeddings_text_list, embeddings_text_weighted_list = [], []
        embeddings_audio_list, embeddings_audio_opensmile_list = [], []
        task_counts = {task: 0 for task in self.included_tasks}
        sample_count = 0
        
        voxel_indices = np.where(self.mask.get_fdata().reshape(-1) > 0)[0]
        
        for df in dfs.values():
            df = df.rename(columns=lambda x: x.strip())
            for index, row in df.iterrows():
                task = row["task"]
                if task not in self.included_tasks:
                    continue
                
                task_counts[task] += 1
                sample_count += 1
                
                context = row["Context"]
                statement = row["Statement"]
                situation = row["Situation"]
                evaluation = row["Evaluation_Score"]
                age = row["age"]
                gender = row["genre"]
                fmri_file = f'{participant}_{task}_{context[:-4]}_{statement[:-4]}_statement.npy'
                fmri_path = f"{participant}/{fmri_file}"
                
                parts = row["Condition_name"].split("_")
                context_cond = parts[0]
                statement_cond = parts[1]
                
                fmri = self.fmri_cache.get(fmri_path)
                if fmri is None:
                    continue
                
                fmri_masked = fmri[:, voxel_indices]
                fmri_data_list.append(fmri_masked)
                
                if self.use_base_features:
                    final_data.append({
                        "context": context_cond,
                        "semantic": statement_cond[:2],
                        "prosody": statement_cond[-3:],
                        "task": task,
                        "evaluation": evaluation,
                        "age": age,
                        "gender": gender,
                        "participant": participant,
                        "situation": situation,
                    })
                
                ids_list.append(int(participant[1:]))
                
                # Load embeddings
                if self.use_text and self.embeddings_text_path:
                    embeddings_text = np.load(os.path.join(self.embeddings_text_path, "statements", f"{statement_cond[:2]}_{situation}_CLS.npy"))
                    embeddings_text_list.append(embeddings_text)
                
                if self.use_text_weighted and self.embeddings_text_path:
                    embeddings_text_weighted = np.load(os.path.join(self.embeddings_text_path, "text_weighted", f"{context_cond}_{situation}_{statement_cond[:2]}_{situation}_weighted.npy"))
                    
                    embeddings_text_weighted_list.append(embeddings_text_weighted)
                    
                if self.use_audio and self.embeddings_audio_path:
                    embeddings_audio = np.load(os.path.join(self.embeddings_audio_path, f"{statement.replace('.wav', '_layers5-6.npy')}"))
                    embeddings_audio_list.append(embeddings_audio)
                
                if self.use_audio_opensmile and self.embeddings_audio_opensmile_path:
                    embeddings_audio_opensmile = np.load(os.path.join(self.embeddings_audio_opensmile_path, f"{statement.replace('.wav', '_opensmile.npy')}"))
                    embeddings_audio_opensmile_list.append(embeddings_audio_opensmile)
        
        return final_data, fmri_data_list, ids_list, embeddings_text_list, embeddings_text_weighted_list, embeddings_audio_list, embeddings_audio_opensmile_list, task_counts, sample_count

    def create_data(self):
        """Loads and processes data into feature DataFrame and fMRI DataFrame using parallel processing."""
        final_data, fmri_data_list, ids_list = [], [], []
        embeddings_text_list, embeddings_text_weighted_list = [], []
        embeddings_audio_list, embeddings_audio_opensmile_list = [], []
        task_counts = {task: 0 for task in self.included_tasks}
        total_samples = 0
        
        # Parallel processing of participants
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.process_participant, self.participant_list))
        
        # Aggregate results
        for result in results:
            final_data.extend(result[0])
            fmri_data_list.extend(result[1])
            ids_list.extend(result[2])
            embeddings_text_list.extend(result[3])
            embeddings_text_weighted_list.extend(result[4])
            embeddings_audio_list.extend(result[5])
            embeddings_audio_opensmile_list.extend(result[6])
            for task, count in result[7].items():
                task_counts[task] += count
            total_samples += result[8]
        
        print(f"Loaded {total_samples} total samples")
        for task, count in task_counts.items():
            print(f"  - {task}: {count} samples")
            
        # Create base DataFrame
        df = pd.DataFrame(final_data) if self.use_base_features else pd.DataFrame(index=range(total_samples))
        if self.use_base_features:
            df.reset_index(drop=True, inplace=True)
            categorical_cols = ['context', 'semantic', 'prosody', 'task', 'gender', 'participant', 'situation']
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
            df['evaluation'] = df['evaluation'].fillna(df['evaluation'].median())
            df['evaluation'] = (df['evaluation'] - df['evaluation'].min()) / (df['evaluation'].max() - df['evaluation'].min())
            df['age'] = self.scaler.fit_transform(df[['age']])
        
        # Process text embeddings
        if self.use_text and embeddings_text_list:
            embeddings_df = pd.DataFrame(np.vstack(embeddings_text_list))
            embeddings_df.columns = [f"emb_text_{i}" for i in range(embeddings_df.shape[1])]
            if self.use_pca:
                df_pca_text = self.apply_pca(embeddings_df, prefix="pc_text")
                df = pd.concat([df, df_pca_text], axis=1)
            else:
                embeddings_scaled = self.scaler.fit_transform(embeddings_df)
                df_scaled = pd.DataFrame(embeddings_scaled, columns=embeddings_df.columns)
                df = pd.concat([df, df_scaled], axis=1)
        
        # Process weighted text embeddings
        if self.use_text_weighted and embeddings_text_weighted_list:
            embeddings_df = pd.DataFrame(np.vstack(embeddings_text_weighted_list))
            embeddings_df.columns = [f"emb_weighted_{i}" for i in range(embeddings_df.shape[1])]
            if self.use_pca:
                df_pca_weighted = self.apply_pca(embeddings_df, prefix="pc_weighted")
                df = pd.concat([df, df_pca_weighted], axis=1)
            else:
                embeddings_scaled = self.scaler.fit_transform(embeddings_df)
                df_scaled = pd.DataFrame(embeddings_scaled, columns=embeddings_df.columns)
                df = pd.concat([df, df_scaled], axis=1)
        
        # Process audio embeddings
        if self.use_audio and embeddings_audio_list:
            embeddings_df = pd.DataFrame(np.vstack(embeddings_audio_list))
            embeddings_df.columns = [f"emb_audio_{i}" for i in range(embeddings_df.shape[1])]
            if self.use_pca:
                df_pca_audio = self.apply_pca(embeddings_df, prefix="pc_audio")
                df = pd.concat([df, df_pca_audio], axis=1)
            else:
                embeddings_scaled = self.scaler.fit_transform(embeddings_df)
                df_scaled = pd.DataFrame(embeddings_scaled, columns=embeddings_df.columns)
                df = pd.concat([df, df_scaled], axis=1)
        
        # Process openSMILE audio embeddings
        if self.use_audio_opensmile and embeddings_audio_opensmile_list:
            embeddings_df = pd.DataFrame(np.vstack(embeddings_audio_opensmile_list))
            embeddings_df.columns = [f"emb_audio_opensmile_{i}" for i in range(embeddings_df.shape[1])]
            if self.use_pca:
                df_pca_opensmile = self.apply_pca(embeddings_df, prefix="pc_audio_opensmile")
                df = pd.concat([df, df_pca_opensmile], axis=1)
            else:
                embeddings_scaled = self.scaler.fit_transform(embeddings_df)
                df_scaled = pd.DataFrame(embeddings_scaled, columns=embeddings_df.columns)
                df = pd.concat([df, df_scaled], axis=1)
        
        # Convert to numpy for faster access
        fmri_data = np.vstack(fmri_data_list)
        ids_list = np.array(ids_list, dtype=np.int32)
        
        return df, fmri_data, ids_list
    
    def __getitem__(self, index):
        """Returns a single sample: one row from feature array and fMRI array."""
        return {
            "features": self.data[index],
            "fmri_data": self.fmri_data[index]
        }
    
    def __len__(self):
        """Returns the total number of samples."""
        return len(self.data)