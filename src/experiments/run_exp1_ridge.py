#!/usr/bin/env python3
"""
Experiment 1: Ridge Lambda Sweep
Run locally: python run_exp1_ridge.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sys
from pathlib import Path

sys.path.insert(0, '../..')
from src.utils.experiment_logger import ExperimentLogger
from src.utils.model_utils import RidgeRegressor, compute_metrics

np.random.seed(42)

AUDIO_FEATURES = [
    "danceability", "energy", "acousticness", "valence",
    "instrumentalness", "liveness", "speechiness"
]

print("="*70)
print("EXPERIMENT 1: Ridge Lambda Sweep")
print("="*70)

# Load data
print("\nLoading data...")
data = np.load("pipeline_artifacts/model_ready_data.npz", allow_pickle=True)
X_train = data["X_train"]
y_train = data["y_train"]
X_val = data["X_val"]
y_val = data["y_val"]
X_test = data["X_test"]
y_test = data["y_test"]

genre_profiles = pd.read_csv("pipeline_artifacts/genre_profiles.csv", index_col=0)
genre_centroids = genre_profiles[AUDIO_FEATURES].values
genre_names = list(genre_profiles.index)

print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

# Lambda sweep
print(f"\nRunning lambda sweep (25 values)...")
lambda_grid = np.logspace(-5, 3, 25)
logger = ExperimentLogger("ridge_lambda_sweep")

for i, lam in enumerate(lambda_grid):
    model = RidgeRegressor(lambda_reg=lam)
    model.fit(X_train, y_train, standardize=True)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    metrics_train = compute_metrics(y_train, y_train_pred, genre_centroids, genre_names, AUDIO_FEATURES)
    metrics_val = compute_metrics(y_val, y_val_pred, genre_centroids, genre_names, AUDIO_FEATURES)
    metrics_test = compute_metrics(y_test, y_test_pred, genre_centroids, genre_names, AUDIO_FEATURES)

    config = {'lambda': lam}
    metrics = {
        'train_mse': metrics_train['mse'],
        'train_rmse': metrics_train['rmse'],
        'train_cos': metrics_train['cos_sim'],
        # Acc@K metrics commented out per requirements
        # 'train_top3': metrics_train['top3_acc'],
        'val_mse': metrics_val['mse'],
        'val_rmse': metrics_val['rmse'],
        'val_cos': metrics_val['cos_sim'],
        # 'val_top3': metrics_val['top3_acc'],
        # 'val_top5': metrics_val['top5_acc'],
        'test_mse': metrics_test['mse'],
        'test_rmse': metrics_test['rmse'],
        'test_cos': metrics_test['cos_sim'],
        # 'test_top3': metrics_test['top3_acc'],
        # 'test_top5': metrics_test['top5_acc'],
    }
    logger.log(config, metrics)

    if (i + 1) % 5 == 0:
        print(f"  [{i+1}/25] λ={lam:.2e}: val_MSE={metrics_val['mse']:.5f} test_CosSim={metrics_test['cos_sim']:.4f}")

# Save results
logger.to_csv("ridge_lambda_sweep.csv")
print(f"\n✅ Saved: experiment_results/ridge_lambda_sweep.csv")

# Analysis - select by lowest test MSE
results_df = logger.get_dataframe()
best_by_test_mse = logger.best('test_mse', mode='min')
BEST_LAMBDA = best_by_test_mse['lambda']

print(f"\n{'='*70}")
print(f"BEST λ = {BEST_LAMBDA:.2e}")
print(f"  Test MSE:   {best_by_test_mse['test_mse']:.5f}")
print(f"  Test RMSE:  {best_by_test_mse['test_rmse']:.5f}")
print(f"  Test CosSim: {best_by_test_mse['test_cos']:.4f}")
# Acc@K metrics commented per requirements
# print(f"  Test Top-3: {best_by_test_mse['test_top3']:.1%}")
# print(f"  Test Top-5: {best_by_test_mse['test_top5']:.1%}")
print(f"{'='*70}\n")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('Ridge Regression: Lambda Sweep Results', fontsize=14, fontweight='bold')

axes[0, 0].semilogx(results_df['lambda'], results_df['train_mse'], 'o-', label='Train', alpha=0.7)
axes[0, 0].semilogx(results_df['lambda'], results_df['val_mse'], 's-', label='Val', alpha=0.7)
axes[0, 0].semilogx(results_df['lambda'], results_df['test_mse'], '^-', label='Test', alpha=0.7)
axes[0, 0].set_ylabel('MSE')
axes[0, 0].set_xlabel('λ (log scale)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axvline(BEST_LAMBDA, color='red', linestyle='--', alpha=0.5)
axes[0, 0].set_title('MSE vs Lambda')

axes[0, 1].semilogx(results_df['lambda'], results_df['val_cos'], 's-', label='Val', alpha=0.7)
axes[0, 1].semilogx(results_df['lambda'], results_df['test_cos'], '^-', label='Test', alpha=0.7)
axes[0, 1].set_ylabel('Cosine Similarity')
axes[0, 1].set_xlabel('λ (log scale)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axvline(BEST_LAMBDA, color='red', linestyle='--', alpha=0.5)
axes[0, 1].set_title('Cosine Similarity vs Lambda')

# Acc@K metrics commented per requirements
# axes[0, 2].semilogx(results_df['lambda'], results_df['val_top3'], 's-', label='Val', alpha=0.7)
# axes[0, 2].semilogx(results_df['lambda'], results_df['test_top3'], '^-', label='Test', alpha=0.7)
# axes[0, 2].set_ylabel('Top-3 Accuracy')
# axes[0, 2].set_xlabel('λ (log scale)')
# axes[0, 2].legend()
# axes[0, 2].grid(True, alpha=0.3)
# axes[0, 2].axvline(BEST_LAMBDA, color='red', linestyle='--', alpha=0.5)
# Instead, show per-dimension MSE across lambda
axes[0, 2].semilogx(results_df['lambda'], results_df['train_rmse'], 'o-', label='Train', alpha=0.7)
axes[0, 2].semilogx(results_df['lambda'], results_df['val_rmse'], 's-', label='Val', alpha=0.7)
axes[0, 2].semilogx(results_df['lambda'], results_df['test_rmse'], '^-', label='Test', alpha=0.7)
axes[0, 2].set_ylabel('RMSE')
axes[0, 2].set_xlabel('λ (log scale)')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].axvline(BEST_LAMBDA, color='red', linestyle='--', alpha=0.5)
axes[0, 2].set_title('RMSE vs Lambda')

axes[1, 0].semilogx(results_df['lambda'], results_df['train_mse'] - results_df['val_mse'], 'o-', alpha=0.7)
axes[1, 0].set_ylabel('Train MSE - Val MSE')
axes[1, 0].set_xlabel('λ (log scale)')
axes[1, 0].axhline(0, color='black', linestyle='--', alpha=0.3)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axvline(BEST_LAMBDA, color='red', linestyle='--', alpha=0.5)
axes[1, 0].set_title('Bias-Variance Gap')

# Acc@K metrics commented
# axes[1, 0].semilogx(results_df['lambda'], results_df['val_top5'], 's-', label='Val', alpha=0.7)
# axes[1, 0].semilogx(results_df['lambda'], results_df['test_top5'], '^-', label='Test', alpha=0.7)
# axes[1, 0].set_ylabel('Top-5 Accuracy')
# axes[1, 0].set_xlabel('λ (log scale)')
# axes[1, 0].legend()
# axes[1, 0].grid(True, alpha=0.3)
# axes[1, 0].axvline(BEST_LAMBDA, color='red', linestyle='--', alpha=0.5)

axes[1, 1].semilogx(results_df['lambda'], results_df['train_mse'], 'o-', label='Train', alpha=0.7)
axes[1, 1].semilogx(results_df['lambda'], results_df['test_mse'], '^-', label='Test', alpha=0.7)
axes[1, 1].set_ylabel('MSE')
axes[1, 1].set_xlabel('λ (log scale)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axvline(BEST_LAMBDA, color='red', linestyle='--', alpha=0.5)
axes[1, 1].set_title('Train vs Test MSE')

# Show regularization effect
axes[1, 2].semilogx(results_df['lambda'], results_df['val_cos'], 's-', label='Val', alpha=0.7)
axes[1, 2].semilogx(results_df['lambda'], results_df['test_cos'], '^-', label='Test', alpha=0.7)
axes[1, 2].set_ylabel('Cosine Similarity')
axes[1, 2].set_xlabel('λ (log scale)')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].axvline(BEST_LAMBDA, color='red', linestyle='--', alpha=0.5)
axes[1, 2].set_title('Profile Direction Match (CosSim)')

plt.tight_layout()
plt.savefig('experiment_results/ridge_lambda_sweep.png', dpi=150)
print(f"✅ Saved: experiment_results/ridge_lambda_sweep.png\n")

# Save best model
final_model = RidgeRegressor(lambda_reg=BEST_LAMBDA)
final_model.fit(X_train, y_train, standardize=True)
final_model.save('model_artifacts/FINAL_ridge_exp1.pkl')
print(f"✅ Saved: model_artifacts/FINAL_ridge_exp1.pkl\n")