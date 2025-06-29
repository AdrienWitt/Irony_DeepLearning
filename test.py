import os

os.chdir(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning")

import analysis_helpers
import dataset

paths = analysis_helpers.get_paths()
participant_list = os.listdir(paths["data_path"])

# Define args as a class-like object
class Args:
    pca_threshold = 0.50
    use_base_features = True
    use_text = False
    use_audio = True
    use_text_combined = True
    use_pca = True
    use_umap = False

args = Args()

voxel = (20,20,20)

database_train = analysis_helpers.load_dataset(args, paths, participant_list)
data = database_train.get_voxel_values(voxel)

import numpy as np
emb = np.load(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\embeddings\text\statements\SN_1_CLS.npy")

import nibabel as nib
 
path1 = paths["fmri_data_path"]
path2 = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\fmri\mean"

for foldername, subfolders, filenames in os.walk(path1):
    for filename in filenames:
        file_path = os.path.join(foldername, filename)
        nii = nib.load(file_path).get_fdata()
        print(nii.dtype)
        print(nii[50,50,50])
        
      
a = np.load(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\results\correlation_map_folds_text_audio_base.npy")


a = [0, 0, 0, 0]
np.std(a) 

c = np.load(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\embeddings\text\text_combined\CN_1_SN_1_weighted.npy")

###################################################################################################################################
import nibabel as nib
from nilearn import image, masking, plotting
import numpy as np 

input_path = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\fmri\weighted\p26\p26_irony_CNf2_12_SPnegf1_12_statement_weighted.nii.gz"

# Load fMRI image
img = nib.load(input_path)
# Convert image data to numpy array
data = img.get_fdata()

# Find coordinates where values are zero
zero_coords = np.column_stack(np.where(data == 0))

print(f"Number of zero voxels: {len(zero_coords)}")
print("Sample coordinates of zero voxels (x, y, z):")
print(zero_coords[:10])  # Print first 10 coordinates

flattened_data = data.flatten()

from collections import Counter
most_common_value, count = Counter(flattened_data).most_common(1)[0]


# Plot using Nilearn
plotting.plot_epi(img, title="Original Image", cut_coords=(0, 0, 0), display_mode='ortho')


plotting.show()

# Example usage
preprocess_fmri_with_nilearn_plotting("input.nii.gz", "output.nii.gz")

import nibabel as nib
nii = nib.load(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\fmri\weighted\p01\p01_irony_CNf4_8_SPnegh1_8_statement_weighted.nii")


value = nii.get_fdata()

display = plotting.plot_epi(nii)


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
        self.data, self.fmri_data, self.ids_list = self.create_data()

    def register_args(self, **kwargs):
        """Registers additional arguments into the class instance."""
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.kwargs = kwargs
    
    def apply_pca(self, embeddings_df, prefix):
        """Apply PCA to embeddings."""
        embeddings_scaled = self.scaler.fit_transform(embeddings_df)
        pca = PCA(n_components=self.pca_threshold)
        embeddings_pca = pca.fit_transform(embeddings_scaled)
        return pd.DataFrame(embeddings_pca, columns=[f"{prefix}_{i+1}" for i in range(embeddings_pca.shape[1])])

    def create_data(self):
        """Loads and processes data into feature DataFrame and fMRI DataFrame."""
        final_data = []
        embeddings_text_list = []
        embeddings_text_weighted_list = []
        embeddings_audio_list = []
        embeddings_audio_opensmile_list = []
        fmri_data_list = []
        ids_list = []
        task_counts = {task: 0 for task in self.included_tasks}
        sample_count = 0
        
        mask_data = self.mask.get_fdata()
        flattened_mask = mask_data.reshape(-1)
        voxel_indices = np.where(flattened_mask > 0)[0]
        
        for participant in self.participant_list:
            participant_data_path = os.path.join(self.data_path, participant)
            dfs = processing_helpers.load_dataframe(participant_data_path)
            
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
                    fmri_path = os.path.join(self.fmri_data_path, participant, fmri_file)
                    
                    parts = row["Condition_name"].split("_")
                    context_cond = parts[0]
                    statement_cond = parts[1]
                    
                    fmri = np.load(fmri_path)
 
                    fmri_masked = fmri[:, voxel_indices]
                    fmri_data_list.append(fmri_masked)
                                        
                    # Prepare base features
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
                            "situation" : situation,
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
        
        print(f"Loaded {sample_count} total samples")
        for task, count in task_counts.items():
            print(f"  - {task}: {count} samples")
            
        # Create base DataFrame
        df = pd.DataFrame(final_data) if self.use_base_features else pd.DataFrame(index=range(sample_count))
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
        
        # Create fMRI DataFrame
        fmri_data = np.vstack(fmri_data_list)
        ids_list = np.array(ids_list)
        

        
        return df, fmri_data, ids_list
    
    def __getitem__(self, index):
        """Returns a single sample: one row from feature DataFrame and fMRI DataFrame."""
        return {
            "features": self.data.iloc[index].values,
            "fmri_data": self.fmri_data.iloc[index].values
        }
    
    def __len__(self):
        """Returns the total number of samples."""
        return len(self.data)
    