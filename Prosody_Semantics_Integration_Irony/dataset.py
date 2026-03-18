import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset
import analysis_helpers


class WholeBrainDataset(Dataset):
    def __init__(self, participant_list, data_path, fmri_data_path, mask,
                 included_tasks=None, use_base_features=True,
                 use_text=True, use_audio=True,
                 embeddings_text_path=None, embeddings_audio_path=None,
                 pca_threshold=0.50, use_pca=False, **kwargs):
        super().__init__()
        self.data_path = data_path
        self.fmri_data_path = fmri_data_path
        self.participant_list = participant_list
        self.included_tasks = included_tasks or ["sarcasm", "irony", "prosody", "semantic", "tom"]
        self.use_base_features = use_base_features
        self.use_text = use_text          
        self.use_audio = use_audio        
        self.embeddings_text_path = embeddings_text_path      
        self.embeddings_audio_path = embeddings_audio_path   
        self.pca_threshold = pca_threshold
        self.use_pca = use_pca
        self.scaler = StandardScaler()

        for name, value in kwargs.items():
            setattr(self, name, value)

        self.mask = mask
        self.fmri_cache = self.preload_fmri()
        self.data, self.fmri_data, self.ids_list = self.create_data()

    def preload_fmri(self):
        """Preload all fMRI .npy files into a cache."""
        fmri_cache = {}
        for participant in self.participant_list:
            participant_fmri_path = os.path.join(self.fmri_data_path, participant)
            if not os.path.exists(participant_fmri_path):
                continue
            for fmri_file in os.listdir(participant_fmri_path):
                if fmri_file.endswith('.npy'):
                    key = f"{participant}/{fmri_file}"
                    fmri_cache[key] = np.load(os.path.join(participant_fmri_path, fmri_file), mmap_mode='r')
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
        dfs = analysis_helpers.load_dataframe(participant_data_path)

        final_data = []
        fmri_data_list = []
        ids_list = []
        embeddings_text_list = []   
        embeddings_audio_list = []  
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
                    })
                
                ids_list.append(int(participant[1:]))

                if self.use_text and self.embeddings_text_path:
                    text_file = f"{context_cond}_{situation}_{statement_cond[:2]}_{situation}_weighted.npy"
                    embeddings_text = np.load(os.path.join(self.embeddings_text_path, text_file))
                    embeddings_text_list.append(embeddings_text)

                if self.use_audio and self.embeddings_audio_path:
                    audio_file = statement.replace('.wav', '_opensmile.npy')
                    embeddings_audio = np.load(os.path.join(self.embeddings_audio_path, audio_file))
                    embeddings_audio_list.append(embeddings_audio)

        return (final_data, fmri_data_list, ids_list,
                embeddings_text_list, embeddings_audio_list,
                task_counts, sample_count)

    def create_data(self):
        """Loads and processes data into feature DataFrame and fMRI DataFrame using parallel processing."""
        final_data = []
        fmri_data_list = []
        ids_list = []
        embeddings_text_list = []
        embeddings_audio_list = []
        task_counts = {task: 0 for task in self.included_tasks}
        total_samples = 0

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.process_participant, self.participant_list))

        for (part_final, part_fmri, part_ids,
             part_text, part_audio,
             part_task_counts, part_count) in results:

            final_data.extend(part_final)
            fmri_data_list.extend(part_fmri)
            ids_list.extend(part_ids)
            embeddings_text_list.extend(part_text)
            embeddings_audio_list.extend(part_audio)
            for task, count in part_task_counts.items():
                task_counts[task] += count
            total_samples += part_count

        print(f"Loaded {total_samples} total samples")
        for task, count in task_counts.items():
            print(f" - {task}: {count} samples")

        # Create base DataFrame
        if self.use_base_features and final_data:
            df = pd.DataFrame(final_data)
            df.reset_index(drop=True, inplace=True)
            categorical_cols = ['context', 'semantic', 'prosody', 'task', 'gender', 'participant']
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
            df['evaluation'] = df['evaluation'].fillna(df['evaluation'].median())
            df['evaluation'] = (df['evaluation'] - df['evaluation'].min()) / (df['evaluation'].max() - df['evaluation'].min())
            df['age'] = self.scaler.fit_transform(df[['age']])
        else:
            df = pd.DataFrame(index=range(total_samples))

        # Process text embeddings (weighted)
        if self.use_text and embeddings_text_list:
            emb_df = pd.DataFrame(np.vstack(embeddings_text_list))
            emb_df.columns = [f"emb_text_{i}" for i in range(emb_df.shape[1])]
            if self.use_pca:
                pca_df = self.apply_pca(emb_df, "pc_text")
                df = pd.concat([df, pca_df], axis=1)
            else:
                scaled = self.scaler.fit_transform(emb_df)
                df = pd.concat([df, pd.DataFrame(scaled, columns=emb_df.columns)], axis=1)

        # Process audio embeddings (openSMILE)
        if self.use_audio and embeddings_audio_list:
            emb_df = pd.DataFrame(np.vstack(embeddings_audio_list))
            emb_df.columns = [f"emb_audio_{i}" for i in range(emb_df.shape[1])]
            if self.use_pca:
                pca_df = self.apply_pca(emb_df, "pc_audio")
                df = pd.concat([df, pca_df], axis=1)
            else:
                scaled = self.scaler.fit_transform(emb_df)
                df = pd.concat([df, pd.DataFrame(scaled, columns=emb_df.columns)], axis=1)

        if not fmri_data_list:
            raise ValueError("No fMRI data was loaded. Check paths and file naming.")
        fmri_data = np.vstack(fmri_data_list)
        ids_array = np.array(ids_list, dtype=np.int32)

        return df, fmri_data, ids_array

    def __getitem__(self, index):
        """Returns a single sample."""
        features = self.data.iloc[index].values.astype(np.float32)
        fmri = self.fmri_data[index].astype(np.float32)
        return {
            "features": features,
            "fmri_data": fmri
        }

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.data)