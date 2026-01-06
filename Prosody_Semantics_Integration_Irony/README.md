# Prosody_Semantics_Integration_Irony

Cleaned, shareable project for fMRI encoding of prosody & semantics (irony task).
This repository contains a minimal, documented set of scripts to reproduce the
encoding pipeline used in the project.

Files
- `analysis_helpers.py` — path helpers and dataset loader wrapper
- `dataset.py` — dataset loader that reads behavioral files, fMRI arrays and embeddings
- `audio_text_embeddings.py` — generation of text-weighted and openSMILE audio embeddings
- `ridge_cv.py` — ridge regression + GroupKFold CV (adapted from )
- `encoding.py` — main runner script
- `requirements.txt` — Python deps

Quick usage
1. Adjust paths in `analysis_helpers.get_paths()` to match your layout.
2. Run:
```bash
python encoding.py --use_text --use_audio
```

