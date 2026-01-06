"""
Load or generate embeddings for text (statement-weighted) and audio (openSMILE).

This module provides two small helpers used by `dataset.py`.
"""

import os
import numpy as np


def load_text_weighted(embeddings_base, context, situation, semantic_code):
    """Load the precomputed statement-weighted text embedding.

    Expected filename: `<context>_<situation>_<semantic>_<situation>_weighted.npy`
    Returns None if file is missing.
    """
    fname = f"{context}_{situation}_{semantic_code}_{situation}_weighted.npy"
    p = os.path.join(embeddings_base, 'text_weighted', fname)
    if not os.path.exists(p):
        return None
    return np.load(p)


def load_audio_opensmile(embeddings_base, statement_filename):
    """Load precomputed openSMILE features.

    Expected filename: `<statement_basename>_opensmile.npy`
    Returns None if file is missing.
    """
    fname = statement_filename.replace('.wav', '_opensmile.npy')
    p = os.path.join(embeddings_base, fname)
    if not os.path.exists(p):
        return None
    return np.load(p)
