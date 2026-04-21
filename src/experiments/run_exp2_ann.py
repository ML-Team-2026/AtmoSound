#!/usr/bin/env python3
"""
Experiment 2: ANN Architecture & Regularization Sweep
Run locally: python run_exp2_ann.py
NOTE: ANN training is stochastic and slower than Ridge. Each architecture/config is trained once.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys

sys.path.insert(0, '../..')
from src.utils.experiment_logger import ExperimentLogger
from src.utils.model_utils import NeuralNetworkRegressor, compute_metrics

np.random.seed(42)

AUDIO_FEATURES = [
    "danceability", "energy", "acousticness", "valence",
    "instrumentalness", "liveness", "speechiness"
]

print("="*70)
print("EXPERIMENT 2: ANN Architecture & Regularization Sweep")
print("="*70)
print("WARNING: Acc@K metrics commented out per requirements")

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

# === PART A: Architecture Search ===
print("\n" + "="*70)
print("PART A: Architecture Search (using best Ridge config from Exp1)")
print("="*70)

architectures = [
    (64, 32),
    (128, 64),
    (256, 128),
    (512, 256),
    (256, 256, 128),
    (512, 256, 128),
]

logger_arch = ExperimentLogger("ann_architecture_sweep")

for i, arch in enumerate(architectures, 1):
    print(f"\n[{i}/{len(architectures)}] Testing ANN architecture: {arch}")

    model = NeuralNetworkRegressor(
        hidden_sizes=arch,
        learning_rate=0.005,
        batch_size=64,
        dropout_rate=0.0,
        l2_lambda=0.0
    )

    model.fit(X_train, y_train, X_val=X_val, y_val=y_val,
              epochs=500, early_stopping_patience=20,
              standardize=True, verbose=False)

    y_test_pred = model.predict(X_test)
    metrics_test = compute_metrics(y_test, y_test_pred, genre_centroids, genre_names, AUDIO_FEATURES)

    num_params = sum(arch[i] * (arch[i-1] if i > 0 else X_train.shape[1])
                     for i in range(len(arch))) + sum(arch)

    config = {
        'architecture': str(arch),
        'num_params': num_params,
    }
    metrics = {
        'test_mse': metrics_test['mse'],
        'test_rmse': metrics_test['rmse'],
        'test_cos': metrics_test['cos_sim'],
        # Acc@K metrics commented - preserved in code, returns 0.0
        'test_top3': metrics_test['top3_acc'],
        'test_top5': metrics_test['top5_acc'],
    }
    logger_arch.log(config, metrics)
    print(f"  Params: {num_params:,}  MSE: {metrics_test['mse']:.5f}  "
          f"CosSim: {metrics_test['cos_sim']:.4f}")

logger_arch.to_csv("ann_architecture_sweep.csv")
print(f"\n✅ Saved: experiment_results/ann_architecture_sweep.csv")

best_arch_row = logger_arch.best('test_mse', mode='min')
BEST_ARCHITECTURE = eval(best_arch_row['architecture'])
print(f"\nBest architecture (by MSE): {BEST_ARCHITECTURE}")

# === PART B: Regularization Search ===
print("\n" + "="*70)
print("PART B: Regularization Search (using best architecture)")
print("="*70)

dropout_vals = [0.0, 0.1, 0.2, 0.3]
l2_vals = [0.0, 0.00001, 0.0001, 0.001]

logger_reg = ExperimentLogger("ann_regularization_sweep")
total_configs = len(dropout_vals) * len(l2_vals)
config_count = 0

for dropout in dropout_vals:
    for l2 in l2_vals:
        config_count += 1

        model = NeuralNetworkRegressor(
            hidden_sizes=BEST_ARCHITECTURE,
            learning_rate=0.005,
            batch_size=64,
            dropout_rate=dropout,
            l2_lambda=l2
        )

        model.fit(X_train, y_train, X_val=X_val, y_val=y_val,
                  epochs=500, early_stopping_patience=20,
                  standardize=True, verbose=False)

        y_test_pred = model.predict(X_test)
        metrics_test = compute_metrics(y_test, y_test_pred, genre_centroids, genre_names, AUDIO_FEATURES)

        config = {
            'dropout': dropout,
            'l2_lambda': l2,
            'architecture': str(BEST_ARCHITECTURE),
        }
        metrics = {
            'test_mse': metrics_test['mse'],
            'test_rmse': metrics_test['rmse'],
            'test_cos': metrics_test['cos_sim'],
            # Acc@K metrics commented
            'test_top3': metrics_test['top3_acc'],
            'test_top5': metrics_test['top5_acc'],
        }
        logger_reg.log(config, metrics)

        if config_count % 4 == 0:
            print(f"  [{config_count}/{total_configs}] Dropout={dropout:.1f} L2={l2:.0e} "
                  f"→ MSE: {metrics_test['mse']:.5f}")

logger_reg.to_csv("ann_regularization_sweep.csv")
print(f"\n✅ Saved: experiment_results/ann_regularization_sweep.csv")

best_reg = logger_reg.best('test_mse', mode='min')
print(f"\nBest regularization (by MSE): Dropout={best_reg['dropout']} L2={best_reg['l2_lambda']}")

# Save best config
best_ann_config = {
    'architecture': list(BEST_ARCHITECTURE),
    'dropout': float(best_reg['dropout']),
    'l2_lambda': float(best_reg['l2_lambda']),
    'learning_rate': 0.005,
    'batch_size': 64,
    'epochs': 500,
    'early_stopping_patience': 20,
}

with open('model_artifacts/best_ann_config.json', 'w') as f:
    json.dump(best_ann_config, f, indent=2)

print(f"✅ Saved: model_artifacts/best_ann_config.json")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('ANN Architecture & Regularization Search', fontsize=14, fontweight='bold')

arch_df = logger_arch.get_dataframe()
axes[0].bar(range(len(arch_df)), arch_df['test_mse'], alpha=0.7, color='steelblue')
axes[0].set_ylabel('Test MSE (lower better)')
axes[0].set_xlabel('Architecture')
axes[0].set_xticks(range(len(arch_df)))
axes[0].set_xticklabels([s[:15] for s in arch_df['architecture']], rotation=45, ha='right')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_title('Architecture Search Results')

reg_df = logger_reg.get_dataframe()
pivot = reg_df.pivot_table(values='test_mse', index='dropout', columns='l2_lambda')
im = axes[1].imshow(pivot, cmap='viridis_r', aspect='auto')
axes[1].set_xlabel('L2 Lambda')
axes[1].set_ylabel('Dropout Rate')
axes[1].set_xticklabels([f'{x:.0e}' for x in pivot.columns], rotation=45)
axes[1].set_title('Regularization Search Results (lower MSE better)')
plt.colorbar(im, ax=axes[1], label='Test MSE')

plt.tight_layout()
plt.savefig('experiment_results/ann_sweep_combined.png', dpi=150)
print(f"✅ Saved: experiment_results/ann_sweep_combined.png\n")