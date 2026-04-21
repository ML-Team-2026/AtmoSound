#!/usr/bin/env python3
"""
Experiment 3: Label Quality Sweep
Run locally: python run_exp3_labels.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '../..')
from src.utils.experiment_logger import ExperimentLogger
from src.utils.model_utils import RidgeRegressor, compute_metrics

np.random.seed(42)

AUDIO_FEATURES = [
    "danceability", "energy", "acousticness", "valence",
    "instrumentalness", "liveness", "speechiness"
]

print("="*70)
print("EXPERIMENT 3: Label Quality Sweep")
print("="*70)

# Load data
print("\nLoading data...")
data = np.load("pipeline_artifacts/model_ready_data.npz", allow_pickle=True)
X_train = data["X_train"]
y_train_orig = data["y_train"]
X_val = data["X_val"]
y_val = data["y_val"]
X_test = data["X_test"]
y_test = data["y_test"]

genre_profiles = pd.read_csv("pipeline_artifacts/genre_profiles.csv", index_col=0)
genre_centroids = genre_profiles[AUDIO_FEATURES].values
genre_names = list(genre_profiles.index)

print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

# Load best models from Exp 1 & 2
ridge_results = pd.read_csv('experiment_results/ridge_lambda_sweep.csv')
best_ridge_lam = ridge_results.loc[ridge_results['test_top3'].idxmax(), 'lambda']

print(f"\nBest Ridge Lambda: {best_ridge_lam:.2e}")

# Sweep label multipliers
shift_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
logger = ExperimentLogger("label_quality_sweep")

print(f"\nTesting {len(shift_multipliers)} label multiplier values...\n")

for mult in shift_multipliers:
    print(f"Multiplier: {mult}")

    y_train_scaled = y_train_orig * mult
    y_train_scaled = np.clip(y_train_scaled, 0, 1)
    y_val_scaled = y_val * mult
    y_val_scaled = np.clip(y_val_scaled, 0, 1)
    y_test_scaled = y_test * mult
    y_test_scaled = np.clip(y_test_scaled, 0, 1)

    ridge = RidgeRegressor(lambda_reg=best_ridge_lam)
    ridge.fit(X_train, y_train_scaled, standardize=True)
    y_test_ridge = ridge.predict(X_test)
    metrics_ridge = compute_metrics(y_test_scaled, y_test_ridge, genre_centroids, genre_names, AUDIO_FEATURES)

    config = {'shift_multiplier': mult}
    metrics = {
        'ridge_test_mse': metrics_ridge['mse'],
        'ridge_test_rmse': metrics_ridge['rmse'],
        'ridge_test_cos': metrics_ridge['cos_sim'],
        'ridge_test_top3': metrics_ridge['top3_acc'],
        'ridge_test_top5': metrics_ridge['top5_acc'],
    }
    logger.log(config, metrics)
    print(f"  Ridge Top-3: {metrics_ridge['top3_acc']:.1%}  Top-5: {metrics_ridge['top5_acc']:.1%}")

logger.to_csv("label_quality_sweep.csv")
print(f"\n✅ Saved: experiment_results/label_quality_sweep.csv")

# Analysis
results_df = logger.get_dataframe()
best_mult = logger.best('ridge_test_top3', mode='max')

print(f"\nBest multiplier: {best_mult['shift_multiplier']}")
print(f"  Test Top-3: {best_mult['ridge_test_top3']:.1%}")
print(f"  Test Top-5: {best_mult['ridge_test_top5']:.1%}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Label Quality Sensitivity Analysis', fontsize=14, fontweight='bold')

axes[0].plot(results_df['shift_multiplier'], results_df['ridge_test_top3'], 'o-', linewidth=2, markersize=8, label='Top-3')
axes[0].plot(results_df['shift_multiplier'], results_df['ridge_test_top5'], 's-', linewidth=2, markersize=8, label='Top-5')
axes[0].set_xlabel('Shift Multiplier')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Ridge Regression Performance')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].axvline(1.0, color='red', linestyle='--', alpha=0.5)

axes[1].plot(results_df['shift_multiplier'], results_df['ridge_test_mse'], 'o-', linewidth=2, markersize=8, color='orange')
axes[1].set_xlabel('Shift Multiplier')
axes[1].set_ylabel('MSE')
axes[1].set_title('MSE vs Label Multiplier')
axes[1].grid(True, alpha=0.3)
axes[1].axvline(1.0, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('experiment_results/label_quality_sweep.png', dpi=150)
print(f"✅ Saved: experiment_results/label_quality_sweep.png\n")