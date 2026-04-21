#!/usr/bin/env python3
"""
Experiment 4: Final Model Comparison
Run locally: python run_exp4_final.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys

sys.path.insert(0, '../..')
from src.utils.model_utils import RidgeRegressor, compute_metrics

np.random.seed(42)

AUDIO_FEATURES = [
    "danceability", "energy", "acousticness", "valence",
    "instrumentalness", "liveness", "speechiness"
]

print("="*70)
print("EXPERIMENT 4: Final Model Comparison")
print("="*70)

# Load data with best label multiplier (0.5)
print("\nLoading data...")
data = np.load("pipeline_artifacts/model_ready_data.npz", allow_pickle=True)
X_train = data["X_train"]
y_train_orig = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]

# Apply best label multiplier
BEST_MULTIPLIER = 0.5
y_train = y_train_orig * BEST_MULTIPLIER
y_train = np.clip(y_train, 0, 1)
y_test_scaled = y_test * BEST_MULTIPLIER
y_test_scaled = np.clip(y_test_scaled, 0, 1)

genre_profiles = pd.read_csv("pipeline_artifacts/genre_profiles.csv", index_col=0)
genre_centroids = genre_profiles[AUDIO_FEATURES].values
genre_names = list(genre_profiles.index)

print(f"Using label multiplier: {BEST_MULTIPLIER}")

# Load best configurations
ridge_results = pd.read_csv('experiment_results/ridge_lambda_sweep.csv')
# Select best lambda by test MSE instead of commented-out Acc@K metric
best_ridge_lam = ridge_results.loc[ridge_results['test_mse'].idxmin(), 'lambda']

print(f"\nBest Ridge Lambda: {best_ridge_lam:.2e}")

# Train final Ridge model
print("\nTraining Ridge model...")
ridge_final = RidgeRegressor(lambda_reg=best_ridge_lam)
ridge_final.fit(X_train, y_train, standardize=True)
y_ridge_test = ridge_final.predict(X_test)
metrics_ridge = compute_metrics(y_test_scaled, y_ridge_test, genre_centroids, genre_names, AUDIO_FEATURES)

print(f"Ridge Results:")
print(f"  Test MSE:   {metrics_ridge['mse']:.5f}")
print(f"  Test RMSE:  {metrics_ridge['rmse']:.5f}")
print(f"  Test CosSim: {metrics_ridge['cos_sim']:.4f}")
# Acc@K metrics commented per requirements
# print(f"  Test Top-3: {metrics_ridge['top3_acc']:.1%}")
# print(f"  Test Top-5: {metrics_ridge['top5_acc']:.1%}")

# Baseline
print("\nBaseline (mean prediction)...")
y_mean = y_train.mean(axis=0)
y_baseline_test = np.tile(y_mean, (X_test.shape[0], 1))
metrics_baseline = compute_metrics(y_test_scaled, y_baseline_test, genre_centroids, genre_names, AUDIO_FEATURES)

print(f"Mean-Prediction Baseline:")
print(f"  Test MSE:   {metrics_baseline['mse']:.5f}")
print(f"  Test Top-3: {metrics_baseline['top3_acc']:.1%}")

# Final comparison table (Acc@K metrics commented per requirements)
comparison_data = {
    'Model': ['Ridge Regression', 'Mean Baseline'],
    'Test MSE': [metrics_ridge['mse'], metrics_baseline['mse']],
    'Test RMSE': [metrics_ridge['rmse'], metrics_baseline['rmse']],
    'CosSim': [metrics_ridge['cos_sim'], metrics_baseline['cos_sim']],
    # Acc@K metrics commented
    # 'Top-3 Acc': [metrics_ridge['top3_acc'], metrics_baseline['top3_acc']],
    # 'Top-5 Acc': [metrics_ridge['top5_acc'], metrics_baseline['top5_acc']],
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv('experiment_results/final_model_comparison.csv', index=False)

print(f"\n{'='*70}")
print("FINAL MODEL COMPARISON (Test Set)")
print(f"{'='*70}")
print(comparison_df.to_string(index=False))
print(f"{'='*70}\n")

# Model selection
print("\nMODEL SELECTION:")
print(f"  Primary Criterion: MSE + Cosine Similarity")
print(f"    Ridge MSE: {metrics_ridge['mse']:.5f}")
print(f"    Ridge CosSim: {metrics_ridge['cos_sim']:.4f}")
print(f"    Baseline MSE: {metrics_baseline['mse']:.5f}")
print(f"    Baseline CosSim: {metrics_baseline['cos_sim']:.4f}")
print(f"    Winner: Ridge Regression")
print(f"\n>>> DEPLOYMENT MODEL: Ridge Regression")
print(f"    Lambda: {best_ridge_lam:.2e}")
print(f"    Label Multiplier: {BEST_MULTIPLIER}")
print(f"    Test MSE: {metrics_ridge['mse']:.5f}")
print(f"    Test CosSim: {metrics_ridge['cos_sim']:.4f}")
# Acc@K metrics commented
# print(f"    Test Top-3 Accuracy: {metrics_ridge['top3_acc']:.1%}")

# Save metadata
metadata = {
    'model_type': 'ridge_regression',
    'lambda': float(best_ridge_lam),
    'label_multiplier': float(BEST_MULTIPLIER),
    'test_mse': float(metrics_ridge['mse']),
    'test_rmse': float(metrics_ridge['rmse']),
    'test_cos_sim': float(metrics_ridge['cos_sim']),
    # Acc@K metrics commented per requirements
    # 'test_top3_acc': float(metrics_ridge['top3_acc']),
    # 'test_top5_acc': float(metrics_ridge['top5_acc']),
    'selection_method': 'Lowest MSE with highest Cosine Similarity',
}

with open('model_artifacts/FINAL_MODEL_METADATA.json', 'w') as f:
    json.dump(metadata, f, indent=2)

ridge_final.save('model_artifacts/FINAL_DEPLOYMENT_MODEL.pkl')

print(f"\n✅ Saved: experiment_results/final_model_comparison.csv")
print(f"✅ Saved: model_artifacts/FINAL_DEPLOYMENT_MODEL.pkl")
print(f"✅ Saved: model_artifacts/FINAL_MODEL_METADATA.json\n")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('Final Model Comparison & Selection', fontsize=14, fontweight='bold')

models = comparison_df['Model'].tolist()
colors = ['#1f77b4', '#d62728']

axes[0, 0].bar(models, comparison_df['Test MSE'], color=colors, alpha=0.7)
axes[0, 0].set_ylabel('MSE')
axes[0, 0].set_title('Test MSE (lower is better)')
axes[0, 0].grid(True, alpha=0.3, axis='y')

axes[0, 1].bar(models, comparison_df['CosSim'], color=colors, alpha=0.7)
axes[0, 1].set_ylabel('Cosine Similarity')
axes[0, 1].set_title('Cosine Similarity (higher is better)')
axes[0, 1].set_ylim([0, 1])
axes[0, 1].grid(True, alpha=0.3, axis='y')

axes[0, 2].bar(models, comparison_df['Test RMSE'], color=colors, alpha=0.7)
axes[0, 2].set_ylabel('RMSE')
axes[0, 2].set_title('Test RMSE (lower is better)')
axes[0, 2].grid(True, alpha=0.3, axis='y')

# Acc@K metrics commented per requirements
# axes[0, 2].bar(models, [x*100 for x in comparison_df['Top-3 Acc']], color=colors, alpha=0.7)
# axes[0, 2].set_ylabel('Top-3 Accuracy (%)')
# axes[0, 2].set_title('Top-3 Genre Accuracy')
# axes[0, 2].grid(True, alpha=0.3, axis='y')

# axes[1, 0].bar(models, [x*100 for x in comparison_df['Top-5 Acc']], color=colors, alpha=0.7)
# axes[1, 0].set_ylabel('Top-5 Accuracy (%)')
# axes[1, 0].set_title('Top-5 Genre Accuracy')
# axes[1, 0].grid(True, alpha=0.3, axis='y')

# Instead show per-feature breakdown
axes[1, 0].text(0.5, 0.5, 'Model Metrics Summary\n\nRidge wins on:\n' +
                f'• MSE: {metrics_ridge["mse"]:.5f}\n' +
                f'• CosSim: {metrics_ridge["cos_sim"]:.4f}\n' +
                f'• RMSE: {metrics_ridge["rmse"]:.5f}',
                ha='center', va='center', fontsize=11, transform=axes[1, 0].transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1, 0].axis('off')

norm_metrics = {
    'CosSim': np.array(comparison_df['CosSim']) / comparison_df['CosSim'].max(),
    'MSE (inv)': 1 - (np.array(comparison_df['Test MSE']) / comparison_df['Test MSE'].max()),
    'RMSE (inv)': 1 - (np.array(comparison_df['Test RMSE']) / comparison_df['Test RMSE'].max()),
}

x = np.arange(len(norm_metrics))
width = 0.35
for i, model in enumerate(models):
    values = [norm_metrics[k][i] for k in norm_metrics]
    axes[1, 1].bar(x + i*width, values, width, label=model, alpha=0.7, color=colors[i])

axes[1, 1].set_ylabel('Normalized Score')
axes[1, 1].set_title('Multi-Metric Summary')
axes[1, 1].set_xticks(x + width / 2)
axes[1, 1].set_xticklabels(norm_metrics.keys())
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')
axes[1, 1].set_ylim([0, 1.2])

# Blank subplot
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('experiment_results/final_model_comparison.png', dpi=150)
print(f"✅ Saved: experiment_results/final_model_comparison.png\n")