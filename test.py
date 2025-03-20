import os

os.chdir(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning")

import dataset_umap
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
    use_pca = False
    use_umap = True

args = Args()

voxel = (0,4,5)

def load_dataset(args, paths, participant_list):
    """Loads the dataset using parsed arguments."""


    dataset_args = {
        "data_path": paths["data_path"],
        "fmri_data_path": paths["fmri_data_path"],
        "img_size": tuple(args.img_size),
        "embeddings_text_path": paths["embeddings_text_path"],
        "embeddings_audio_path": paths["embeddings_audio_path"],
        "use_base_features": args.use_base_features,
        "use_text": args.use_text,
        "use_audio": args.use_audio,
        "use_context": args.use_context,
        "pca_threshold": args.pca_threshold,
        "use_pca" : args.use_pca
    }

    database = dataset_umap.BaseDataset(participant_list=participant_list, **dataset_args)
    
    return database


database_train = load_dataset(args, paths, participant_list)
data = database_train.get_voxel_values(voxel)


