"""
Helper utilities: path management and a small wrapper `load_dataset` used by
`encoding.py` to construct the `WholeBrainDataset` instance and return the
features DataFrame, fMRI array and participant ids.

Note: update `get_paths()` to reflect your environment before running.
"""

import os
from pathlib import Path


def get_paths():
    """Return project-relative paths used by the pipeline.

    Edit these entries to match your local filesystem layout before running.
    """
    base = Path(__file__).resolve().parent.parent

    # By default expect fMRI files under `data/fmri` and behavioral under
    # `data/behavioral`. Results are written to `results_wholebrain_irosar`.
    paths = {
        'data_path': str(base / 'data' / 'behavioral'),
        'fmri_data_path': str(base / 'data' / 'fmri'),
        'embeddings_text_path': str(base / 'embeddings' / 'text'),
        'embeddings_audio_path': str(base / 'embeddings' / 'audio'),
        'results_path': str(base / 'results')
    }

    # ensure results directories exist
    os.makedirs(paths['results_path'], exist_ok=True)

    return paths


def load_dataset(args, paths, participant_list, mask):
    """Construct and return (features_df, fmri_array, ids_array).

    This wrapper centralizes the dataset construction used by `encoding.py`.
    """
    from dataset import WholeBrainDataset

    ds = WholeBrainDataset(
        participant_list=participant_list,
        data_path=paths['data_path'],
        fmri_data_path=paths['fmri_data_path'],
        mask=mask,
        embeddings_text_path=paths['embeddings_text_path'],
        embeddings_audio_path=paths['embeddings_audio_path'],
        included_tasks=args.include_tasks,
        use_base_features=args.use_base_features,
        use_text=args.use_text,
        use_audio=args.use_audio,
        pca_threshold=args.pca_threshold,
        use_pca=args.use_pca,
        use_umap=args.use_umap
    )

    features_df, fmri_array, ids_array = ds.create_data()
    return features_df, fmri_array, ids_array
