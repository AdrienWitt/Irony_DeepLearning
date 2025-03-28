import os

os.chdir(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning")

import analysis_helpers
import dataset

paths = analysis_helpers.get_paths()
participant_list = os.listdir(paths["data_path"])

# Define args as a class-like object
class Args:
    img_size = (75, 92, 77)
    pca_threshold = 0.50
    use_base_features = True
    use_text = True
    use_audio = True
    use_context = False
    use_pca = False
    use_umap = True

args = Args()

voxel = (0,4,5)

database_train = analysis_helpers.load_dataset(args, paths, participant_list)
data = database_train.get_voxel_values(voxel)



import nibabel as nib
 
path1 = paths["fmri_data_path"]
path2 = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\fmri\mean"

for foldername, subfolders, filenames in os.walk(path1):
    for filename in filenames:
        file_path = os.path.join(foldername, filename)
        nii = nib.load(file_path).get_fdata()
        print(nii.dtype)
        print(nii[50,50,50])
        
        
import numpy as np 
a = np.load(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\results\correlation_map_folds_text_audio_base.npy")

