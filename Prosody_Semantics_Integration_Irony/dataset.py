"""
Dataset loader for encoding pipeline.
Keeps only `text` (statement-weighted) and `audio` (openSMILE) embeddings
and returns a features DataFrame, fMRI array and participant ids.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from audio_text_embeddings import load_text_weighted, load_audio_opensmile


def _apply_pca(arr, n_components=0.95):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(arr)


class WholeBrainDataset:
    def __init__(self, participant_list, data_path, fmri_data_path, mask,
                 included_tasks=None, use_base_features=True,
                 use_text=True, use_audio=True,
                 embeddings_text_path=None, embeddings_audio_path=None,
                 pca_threshold=0.95, use_pca=False, use_umap=False):
        self.participant_list = participant_list
        self.data_path = data_path
        self.fmri_data_path = fmri_data_path
        self.mask = mask
        self.included_tasks = included_tasks or ['irony', 'sarcasm']
        self.use_base_features = use_base_features
        self.use_text = use_text
        self.use_audio = use_audio
        self.embeddings_text_path = embeddings_text_path
        self.embeddings_audio_path = embeddings_audio_path
        self.pca_threshold = pca_threshold
        self.use_pca = use_pca
        self.scaler = StandardScaler()

    def create_base_data(self):
        base = []
        for participant in self.participant_list:
            pdir = os.path.join(self.data_path, participant)
            if not os.path.isdir(pdir):
                continue

            for fname in os.listdir(pdir):
                if fname.endswith('.csv'):
                    df = pd.read_csv(os.path.join(pdir, fname))
                elif fname.endswith('.txt'):
                    # behavioral exports are space-delimited text files
                    df = pd.read_csv(os.path.join(pdir, fname), sep='\s+')
                else:
                    continue

                for _, row in df.iterrows():
                    task = row.get('task') or row.get('Task')
                    if task not in self.included_tasks:
                        continue

                    # Prefer explicit filename column if available, otherwise build
                    # filename from Context/Statement fields present in the behavioral files.
                    fname_field = row.get('filename')
                    if fname_field:
                        fmri_fname = f"{participant}_{fname_field}.npy"
                    else:
                        context_fn = row.get('Context') or row.get('context') or ''
                        statement_fn = row.get('Statement') or row.get('statement') or ''
                        ctx_base = os.path.splitext(str(context_fn))[0]
                        stmt_base = os.path.splitext(str(statement_fn))[0]
                        fmri_fname = f"{participant}_{task}_{ctx_base}_{stmt_base}_statement.npy"

                    fmri_path = os.path.join(self.fmri_data_path, participant, fmri_fname)
                    if not os.path.exists(fmri_path):
                        continue

                    fmri = np.load(fmri_path)
                    base.append({
                        'participant': participant,
                        'task': task,
                        'context_condition': row.get('Context') or row.get('context_condition') or row.get('context'),
                        'statement_condition': row.get('Condition_name') or row.get('statement_condition'),
                        'situation': row.get('Situation') or row.get('situation'),
                        'statement': row.get('Statement') or row.get('statement') or statement_fn,
                        'fmri': fmri,
                        'age': row.get('age'),
                        'evaluation': row.get('Evaluation_Score') or row.get('evaluation')
                    })

        return base

    def create_data(self):
        base = self.create_base_data()
        features = []
        fmri_list = []
        ids = []

        text_embs = []
        audio_embs = []
        text_idx = []
        audio_idx = []

        for i, s in enumerate(base):
            row = {}
            if self.use_base_features:
                row['context'] = s.get('context_condition')
                row['semantic'] = s.get('statement_condition')[:2] if s.get('statement_condition') else None
                row['prosody'] = s.get('statement_condition')[-3:] if s.get('statement_condition') else None
                row['task'] = s.get('task')
                row['participant'] = s.get('participant')
                row['situation'] = s.get('situation')
                row['age'] = s.get('age')
                row['evaluation'] = s.get('evaluation')

            # load embeddings
            if self.use_text and self.embeddings_text_path:
                emb = load_text_weighted(self.embeddings_text_path, row.get('context'), s.get('situation'), row.get('semantic'))
                if emb is not None:
                    text_idx.append(i)
                    text_embs.append(emb)

            if self.use_audio and self.embeddings_audio_path:
                aemb = load_audio_opensmile(self.embeddings_audio_path, s.get('statement'))
                if aemb is not None:
                    audio_idx.append(i)
                    audio_embs.append(aemb)

            features.append(row)
            fmri_list.append(s.get('fmri'))
            ids.append(int(s.get('participant').lstrip('p')) if str(s.get('participant')).startswith('p') else s.get('participant'))

        df = pd.DataFrame(features)

        # attach text embeddings
        if text_embs:
            text_arr = np.vstack(text_embs)
            if self.use_pca:
                text_arr = _apply_pca(text_arr, self.pca_threshold)
            for j in range(text_arr.shape[1]):
                col = f'text_{j}'
                df.loc[text_idx, col] = text_arr[:, j]

        if audio_embs:
            audio_arr = np.vstack(audio_embs)
            if self.use_pca:
                audio_arr = _apply_pca(audio_arr, self.pca_threshold)
            for j in range(audio_arr.shape[1]):
                col = f'audio_{j}'
                df.loc[audio_idx, col] = audio_arr[:, j]

        # numeric conversions and scaling
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(df['age'].median())
            df['age'] = (df['age'] - df['age'].mean()) / (df['age'].std() + 1e-10)

        if 'evaluation' in df.columns:
            df['evaluation'] = pd.to_numeric(df['evaluation'], errors='coerce').fillna(df['evaluation'].median())
            df['evaluation'] = (df['evaluation'] - df['evaluation'].min()) / (df['evaluation'].max() - df['evaluation'].min())

        # dummy encode categorical variables
        cat_cols = ['context', 'semantic', 'prosody', 'task', 'participant']
        present = [c for c in cat_cols if c in df.columns]
        if present:
            df = pd.get_dummies(df, columns=present, drop_first=True, dtype=int)

        fmri_arr = np.vstack(fmri_list)
        ids_arr = np.array(ids)

        return df, fmri_arr, ids_arr


def get_original_variables_dataframe(dataset):
    """Return a DataFrame with original variables (no dummy coding).

    If a `WholeBrainDataset` instance is provided, the function rebuilds the
    base records and returns the original categorical variables. If a features
    DataFrame is provided, it attempts to return the subset of original columns.
    """
    if isinstance(dataset, WholeBrainDataset):
        base = dataset.create_base_data()
        rows = []
        for r in base:
            rows.append({
                'context': r.get('context_condition'),
                'semantic': r.get('statement_condition')[:2] if r.get('statement_condition') else None,
                'prosody': r.get('statement_condition')[-3:] if r.get('statement_condition') else None,
                'task': r.get('task'),
                'participant': r.get('participant'),
                'situation': r.get('situation'),
                'age': r.get('age'),
                'evaluation': r.get('evaluation')
            })
        return pd.DataFrame(rows)
    elif isinstance(dataset, pd.DataFrame):
        want = [c for c in ['context', 'semantic', 'prosody', 'task', 'participant', 'situation', 'age', 'evaluation'] if c in dataset.columns]
        return dataset[want].copy()
    else:
        raise ValueError('Unsupported dataset type')
