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
