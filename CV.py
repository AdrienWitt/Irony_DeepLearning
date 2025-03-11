import os
import time
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
import analysis_helpers


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run fMRI Ridge Regression analysis.")

    # **Analysis step selection**
    step_group = parser.add_argument_group("Analysis Step Selection")
    step_group.add_argument("--step", type=int, choices=[1, 2, 3], required=True,
                          help="Analysis step to run: 1=base features comparison, 2=PCA threshold optimization, 3=alpha optimization")
    
    # **Dataset-related arguments**
    dataset_group = parser.add_argument_group("Dataset Arguments")
    dataset_group.add_argument("--img_size", type=int, nargs=3, default=[75, 92, 77],
                             help="Size of fMRI images as three integers (default: 75 92 77).")
    dataset_group.add_argument("--use_base_features", action="store_true", 
                             help="Include base features in dataset (default: False).")
    dataset_group.add_argument("--use_text", action="store_true", 
                             help="Include text in dataset (default: False).")
    dataset_group.add_argument("--use_audio", action="store_true", 
                             help="Include audio in dataset (default: False).")
    dataset_group.add_argument("--use_context", action="store_true", 
                             help="Include context in dataset (default: False).")
    dataset_group.add_argument("--use_pca", action="store_true",
                             help="Use PCA for embeddings (default: False)")

    # **Analysis-related arguments**
    analysis_group = parser.add_argument_group("Analysis Arguments")
    analysis_group.add_argument("--num_jobs", type=int, default=20,
                              help="Number of parallel jobs for voxel processing (default: 20).")
    
    # Step-specific arguments
    step1_group = parser.add_argument_group("Step 1: Base Features Comparison")
    step1_group.add_argument("--fixed_alpha", type=float, default=100.0,
                           help="Fixed alpha value for base features comparison (default: 100.0)")
    
    step2_group = parser.add_argument_group("Step 2: PCA Threshold Optimization")
    step2_group.add_argument("--pca_threshold", type=float, nargs='+', default=[0.50, 0.60, 0.70, 0.80],
                           help="List of explained variance thresholds for PCA (default: [0.50, 0.60, 0.70, 0.80])")
    step2_group.add_argument("--step2_use_best_base", action="store_true",
                           help="Use the best base features configuration from step 1 (default: False)")
    
    step3_group = parser.add_argument_group("Step 3: Alpha Optimization")
    step3_group.add_argument("--alpha_values", type=float, nargs='+', default=[10, 100, 1000, 5000],
                           help="List of alpha values for Ridge regression (default: [10, 100, 1000, 5000])")
    step3_group.add_argument("--step3_use_best_pca", action="store_true",
                           help="Use the best PCA threshold from step 2 (default: False)")
    step3_group.add_argument("--step3_use_best_base", action="store_true",
                           help="Use the best base features configuration from step 1 (default: False)")

    return parser.parse_args()


def cv(df_train, voxel, alpha_values, pca_thresholds, step, fixed_alpha=None, use_best_base=False, use_best_pca=False):
    paths = analysis_helpers.get_paths()
        
    # Create a unique random seed from voxel coordinates
    random_seed = int(voxel[0] * 10000 + voxel[1] * 100 + voxel[2])
    
    df_train = df_train.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    
    # Identify embedding columns
    text_cols = [col for col in df_train.columns if col.startswith(('emb_text_', 'pc_text_'))]
    audio_cols = [col for col in df_train.columns if col.startswith(('emb_audio_', 'pc_audio_'))]
    embedding_cols = text_cols + audio_cols    
    
    # Prepare features and target
    X_train = df_train.drop(columns=["fmri_value"])
    y_train = df_train["fmri_value"]  # Keep as pandas Series like in working code

    if step == 1:  # Base features comparison with fixed alpha
        # Create two different feature sets
        X_with_base = X_train
        X_without_base = X_train[embedding_cols]  # Only keep embedding columns
        
        # Define pipelines for both configurations without PCA
        pipeline_with_base = Pipeline([
            ('ridge', Ridge(alpha=fixed_alpha, random_state=random_seed))
        ])
        
        pipeline_without_base = Pipeline([
            ('ridge', Ridge(alpha=fixed_alpha, random_state=random_seed))
        ])
        
        # Fit both pipelines
        pipeline_with_base.fit(X_with_base, y_train)
        pipeline_without_base.fit(X_without_base, y_train)
        
        # Get scores
        with_base_score = pipeline_with_base.score(X_with_base, y_train)
        without_base_score = pipeline_without_base.score(X_without_base, y_train)
        
        return {
            'with_base': {'score': with_base_score},
            'without_base': {'score': without_base_score}
        }
    
    elif step == 2:  # PCA threshold optimization
        # Determine whether to use base features based on previous results
        if use_best_base:
            # Load previous results to determine best configuration
            prev_results_path = os.path.join(paths["results_path"], "step1_results.csv")
            prev_results = pd.read_csv(prev_results_path)
            use_base = prev_results['with_base_score'].mean() > prev_results['without_base_score'].mean()
        else:
            use_base = True  # Default to using base features
        
        # Select appropriate feature set
        X = X_train if use_base else X_train[embedding_cols]
        
        # Create a custom transformer that applies PCA with same threshold to both text and audio
        class SharedPCA:
            def __init__(self, n_components=0.5):
                self.n_components = n_components
                self.text_pca = PCA(n_components=n_components)
                self.audio_pca = PCA(n_components=n_components)
                print(f"\nTesting PCA threshold: {n_components} for both text and audio")
                
            def fit(self, X, y=None):
                # Split features
                text_features = X[text_cols]
                audio_features = X[audio_cols]
                other_features = X.drop(columns=text_cols + audio_cols)
                
                # Fit PCAs
                self.text_pca.fit(text_features)
                self.audio_pca.fit(audio_features)
                
                # Store column names
                self.other_cols = other_features.columns
                return self
                
            def transform(self, X):
                # Transform text and audio features
                text_transformed = self.text_pca.transform(X[text_cols])
                audio_transformed = self.audio_pca.transform(X[audio_cols])
                
                # Get other features
                other_features = X[self.other_cols]
                
                # Combine all features
                result = np.hstack([text_transformed, audio_transformed, other_features])
                return result
                
            def get_params(self, deep=True):
                return {"n_components": self.n_components}
                
            def set_params(self, **params):
                self.n_components = params["n_components"]
                self.text_pca = PCA(n_components=self.n_components)
                self.audio_pca = PCA(n_components=self.n_components)
                print(f"\nTesting PCA threshold: {self.n_components} for both text and audio")
                return self
        
        pipeline = Pipeline([
            ('preprocessor', SharedPCA()),
            ('ridge', Ridge(alpha=fixed_alpha, random_state=random_seed))
        ])
        
        # Now we only need to search over one parameter
        param_grid = {
            'preprocessor__n_components': pca_thresholds
        }
        
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
        grid_search.fit(X, y_train)
        
        best_threshold = grid_search.best_params_['preprocessor__n_components']
        return {
            'best_pca_text': best_threshold,
            'best_pca_audio': best_threshold,
            'score': grid_search.best_score_
        }
    
    else:  # step 3: Alpha optimization
        # Determine whether to use base features and best PCA threshold
        if use_best_base:
            prev_results_path = os.path.join(paths["results_path"], "step1_results.csv")
            prev_results = pd.read_csv(prev_results_path)
            use_base = prev_results['with_base_score'].mean() > prev_results['without_base_score'].mean()
        else:
            use_base = True
        
        if use_best_pca:
            prev_results_path = os.path.join(paths["results_path"], "step2_results.csv")
            prev_results = pd.read_csv(prev_results_path)
            best_pca_text = prev_results['best_pca_text'].mode()[0]
            best_pca_audio = prev_results['best_pca_audio'].mode()[0]
        else:
            best_pca_text = pca_thresholds[0]
            best_pca_audio = pca_thresholds[0]
        
        # Select appropriate feature set
        X = X_train if use_base else X_train[embedding_cols]
        
        # Define pipeline with fixed PCA components
        pipeline = Pipeline([
            ('preprocessor', ColumnTransformer(
                transformers=[
                    ('text', PCA(n_components=best_pca_text), text_cols),
                    ('audio', PCA(n_components=best_pca_audio), audio_cols),
                ],
                remainder='passthrough'
            )),
            ('ridge', Ridge(random_state=random_seed))
        ])
        
        param_grid = {'ridge__alpha': alpha_values}
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
        grid_search.fit(X, y_train)
        
        return {
            'best_alpha': grid_search.best_params_['ridge__alpha'],
            'score': grid_search.best_score_
        }


def process_voxel(args_tuple):
    # Unpack arguments
    voxel, df_train, args = args_tuple
    
    return cv(
        df_train,
        voxel,
        args.alpha_values if args.step == 3 else None,
        args.pca_threshold if args.step == 2 else None,
        args.step,
        args.fixed_alpha if args.step in [1, 2] else None,  # Use fixed_alpha for both step 1 and 2
        args.step2_use_best_base if args.step == 2 else args.step3_use_best_base if args.step == 3 else False,
        args.step3_use_best_pca if args.step == 3 else False
    )


def main():
    start_time = time.time()
    args = parse_arguments()
    
    # Print configuration
    print(f"\nRunning Step {args.step} with configuration:")
    print(f"- Use base features: {args.use_base_features}")
    print(f"- Use text: {args.use_text}")
    print(f"- Use audio: {args.use_audio}")
    print(f"- Use context: {args.use_context}")
    print(f"- Use PCA: {args.use_pca}")
    print(f"- Image size: {args.img_size}")
    print(f"- Number of parallel jobs: {args.num_jobs}")
    
    if args.step == 1:
        print(f"- Fixed alpha: {args.fixed_alpha}")
    elif args.step == 2:
        print(f"- PCA thresholds: {args.pca_threshold}")
        print(f"- Use best base features config: {args.step2_use_best_base}")
    else:  # step 3
        print(f"- Alpha values: {args.alpha_values}")
        print(f"- Use best base features config: {args.step3_use_best_base}")
        print(f"- Use best PCA threshold: {args.step3_use_best_pca}")

    paths = analysis_helpers.get_paths()
    
    # Get list of voxels to process
    voxel_list = list(np.ndindex(tuple(args.img_size)))
    
    # Load dataset once
    participant_list = os.listdir(paths["data_path"])
    database_train = analysis_helpers.load_dataset(args, paths, participant_list)
    
    # Get top voxels
    top_voxels_path = os.path.join(paths["results_path"], "top10_voxels.csv")
    top_voxels = analysis_helpers.get_top_voxels(database_train, tuple(args.img_size), voxel_list, top_voxels_path)
    print(f"\nUsing {len(top_voxels)} top voxels for analysis.")
    
    # Method 2: Using multiprocessing Pool
    from multiprocessing import Pool

    # Prepare arguments for each voxel - get voxel values beforehand
    process_args = [
        (voxel, database_train.get_voxel_values(voxel), args)
        for voxel in top_voxels
    ]

    # Create a pool of workers and map the work
    with Pool(processes=args.num_jobs) as pool:
        results = pool.map(process_voxel, process_args)

    # Process and save results
    if args.step == 1:
        # Calculate average scores
        with_base_scores = [result['with_base']['score'] for result in results]
        without_base_scores = [result['without_base']['score'] for result in results]
        
        avg_with_base = np.mean(with_base_scores)
        avg_without_base = np.mean(without_base_scores)
        
        print("\nBase Features Comparison Results:")
        print(f"Average R² score with base features: {avg_with_base:.4f}")
        print(f"Average R² score without base features: {avg_without_base:.4f}")
        print(f"Difference: {avg_with_base - avg_without_base:.4f}")
        
        # Save results
        df_results = pd.DataFrame([
            {
                'voxel': voxel,
                'with_base_score': result['with_base']['score'],
                'without_base_score': result['without_base']['score']
            }
            for voxel, result in zip(top_voxels, results)
        ])
        results_path = os.path.join(paths["results_path"], "step1_results.csv")
    
    elif args.step == 2:
        # Calculate average scores for each PCA threshold
        best_pca_texts = [result['best_pca_text'] for result in results]
        best_pca_audios = [result['best_pca_audio'] for result in results]
        scores = [result['score'] for result in results]
        
        print("\nPCA Threshold Optimization Results:")
        print(f"Average R² score: {np.mean(scores):.4f}")
        print(f"Most common best PCA threshold for text: {Counter(best_pca_texts).most_common(1)[0][0]}")
        print(f"Most common best PCA threshold for audio: {Counter(best_pca_audios).most_common(1)[0][0]}")
        
        # Save results
        df_results = pd.DataFrame([
            {
                'voxel': voxel,
                'best_pca_text': result['best_pca_text'],
                'best_pca_audio': result['best_pca_audio'],
                'score': result['score']
            }
            for voxel, result in zip(top_voxels, results)
        ])
        results_path = os.path.join(paths["results_path"], "step2_results.csv")
    
    else:  # step 3
        # Calculate average scores for each alpha
        best_alphas = [result['best_alpha'] for result in results]
        scores = [result['score'] for result in results]
        
        print("\nAlpha Optimization Results:")
        print(f"Average R² score: {np.mean(scores):.4f}")
        print(f"Most common best alpha: {Counter(best_alphas).most_common(1)[0][0]}")
        
        # Save results
        df_results = pd.DataFrame([
            {
                'voxel': voxel,
                'best_alpha': result['best_alpha'],
                'score': result['score']
            }
            for voxel, result in zip(top_voxels, results)
        ])
        results_path = os.path.join(paths["results_path"], "step3_results.csv")
    
    df_results.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}!")

    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()