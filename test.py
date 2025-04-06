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
    use_text = True
    use_audio = True
    use_context = False
    use_pca = Tru
    use_umap = False

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
###################################################################################################################################
import nibabel as nib
from nilearn import image, masking, plotting

# Load fMRI image
img = nib.load(input_path)

# Compute brain mask and apply skull stripping
brain_mask = masking.compute_brain_mask(img)
brain_img = image.math_img("img * mask", img=img, mask=brain_mask)

# Crop image to remove excess background
cropped_img = image.crop_img(brain_img)

# Resample to a fixed shape
final_img = image.resample_img(cropped_img, target_affine=cropped_img.affine, target_shape=target_shape)

# Save the processed image
nib.save(final_img, output_path)

# Plot using Nilearn
plotting.plot_epi(img, title="Original Image", cut_coords=(0, 0, 0), display_mode='ortho')
plotting.plot_epi(brain_img, title="Skull-Stripped", cut_coords=(0, 0, 0), display_mode='ortho')
plotting.plot_epi(cropped_img, title="Cropped", cut_coords=(0, 0, 0), display_mode='ortho')
plotting.plot_epi(final_img, title="Resampled (Fixed Shape)", cut_coords=(0, 0, 0), display_mode='ortho')

plotting.show()

# Example usage
preprocess_fmri_with_nilearn_plotting("input.nii.gz", "output.nii.gz")

import nibabel as nib
nii = nib.load(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\fmri\weighted\p01\p01_irony_CNf4_8_SPnegh1_8_statement_weighted.nii")


value = nii.get_fdata()

display = plotting.plot_epi(nii)
