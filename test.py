import os
import time
import argparse
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from joblib import Parallel, delayed
import pandas as pd 
os.chdir(r"C:\Users\wittmann\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning")
import dataset  

args = argparse.Namespace(
    img_size=[75, 92, 77],
    use_context = False,
    use_text = False,
    use_audio = False,
    pca_threshold = .50,
    use_base_features=True,
    alpha=0.5,
    num_jobs=-1,
    use_pca=True
)



# Function to perform voxel-wise Ridge regression
def voxel_analysis(voxel, database_train, database_test, alpha):
    """Train Ridge regression and compute correlation for a given voxel."""
    df_train = database_train.get_voxel_values(voxel)
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    X_train = df_train.drop(columns=["fmri_value"]).values
    y_train = df_train["fmri_value"].values  

    df_test = database_test.get_voxel_values(voxel)
    X_test = df_test.drop(columns=["fmri_value"]).values
    y_test = df_test["fmri_value"].values  

    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)

    correlation = pearsonr(y_pred, y_test)[0] if np.std(y_test) > 0 else 0
    return voxel, correlation

import analysis_helpers
paths = analysis_helpers.get_paths()

for file in os.listdir(os.path.join(paths["embeddings_text_path"], "statements")):
    path = os.path.join(paths["embeddings_text_path"], "statements", file)
    print(path)
    emb = np.load(path)


database_train, database_test = load_dataset(args, paths)
voxel = (38, 77, 50)
df = database_train.get_voxel_values(voxel)

text_cols = [col for col in df.columns if col.startswith("pc_text_")]
audio_cols = [col for col in df.columns if col.startswith("pc_audio_")]


# Generate voxel list dynamically
voxel_list = list(np.ndindex(tuple(args.img_size)))

data = database_train.get_voxel_values((38, 78, 34))

# Initialize correlation map
correlation_map = np.zeros(tuple(args.img_size))

# Parallel processing for all voxels
results = Parallel(n_jobs=args.num_jobs)(
    delayed(voxel_analysis)(voxel, database_train, database_test, args.alpha) for voxel in voxel_list
)

# Store correlations and update the fMRI-sized array
correlations = []  # List to store correlation values
for voxel, corr in results:
    correlation_map[voxel] = corr
    correlations.append(corr)  # Collect correlations for mean calculation

# Compute the mean correlation
mean_correlation = np.mean(correlations)
print(f"Mean correlation: {mean_correlation:.4f}")

# Save the correlation map
result_file = os.path.join(paths["results_path"], f"correlation_map_{args.mode}.npy")
np.save(result_file, correlation_map)

end_time = time.time()  # Stop timing
elapsed_time = end_time - start_time
print(f"Correlation map saved as '{result_file}'")
print(f"Total execution time: {elapsed_time:.2f} seconds")

