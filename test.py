import os


os.chdir(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning")

import dataset
import analysis_helpers

paths = analysis_helpers.get_paths()
participant_list = os.listdir(paths["data_path"])[0:10]
args = {
    "pca_threshold": 0.50,
    "use_base_features": True,
    "use_text": True,
    "use_audio": True,
    "use_context": True,
    "use_pca": True,
    "embeddings_text_path": paths["embeddings_text_path"],
    "embeddings_audio_path": paths["embeddings_audio_path"]
}

database_train = analysis_helpers.load_dataset(args, paths, participant_list)


