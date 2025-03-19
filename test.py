import os

os.chdir(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning")

import dataset
import analysis_helpers

paths = analysis_helpers.get_paths()
participant_list = os.listdir(paths["data_path"])

# Define args as a class-like object
class Args:
    img_size = (75, 92, 77)
    pca_threshold = 0.90
    use_base_features = True
    use_text = True
    use_audio = True
    use_context = False
    use_pca = True

args = Args()

voxel = (0,4,5)

database_train = analysis_helpers.load_dataset(args, paths, participant_list)
data = database_train.get_voxel_values(voxel)


