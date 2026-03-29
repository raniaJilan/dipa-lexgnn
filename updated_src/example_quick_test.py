"""
Quick Example: Running LEX-GNN with Different Loss Functions
This script demonstrates how to quickly test different loss functions
"""

import torch
from model import LEXGNN
from data_handler import load_processed_data
from model_trainer_enhanced import train, test, get_loss_function

# Configuration
DATA_NAME = 'yelp'  # or 'amazon'
DEVICE = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
SEED = 42

# Model hyperparameters (from paper)
BATCH_SIZE = 1024
N_LAYER = 2
N_HEAD = 4
N_HIDDEN = 64
DROPOUT = 0.0

# Training hyperparameters
EPOCHS = 50  # Reduced for quick testing
VALID_EPOCHS = 3
EARLY_STOP = 20
LR = 0.005
WEIGHT_DECAY = 0.0001
BETA = 0.5  # Weight for auxiliary loss

print("="*80)
print("LEX-GNN Loss Function Comparison - Quick Example")
print("="*80)

# Load data
print(f"\nLoading {DATA_NAME} dataset...")
n_input, train_loader, valid_loader, test_loader = load_processed_data(
    DATA_NAME, BATCH_SIZE, N_LAYER
)
print(f"Input dimension: {n_input}")

# Test different loss functions
loss_configs = [
    ('Cross-Entropy (Baseline)', 'ce', {}),
    ('Focal Loss', 'focal', {'alpha': 0.25, 'gamma': 2.0}),
    ('Weighted CE', 'weighted_ce', {'pos_weight': 5.0}),
]

results = {}

for loss_desc, loss_name, loss_params in loss_configs:
    print("\n" + "="*80)
    print(f"Testing: {loss_desc}")
    print("="*80)
    
    # Initialize model
    model = LEXGNN(n_input, 2, N_HIDDEN, N_LAYER, N_HEAD, DROPOUT)
    
    # Train
    model_best, best_epoch, train_time = train(
        model, train_loader, valid_loader,
        epochs=EPOCHS,
        valid_epochs=VALID_EPOCHS,
        beta=BETA,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        early_stop=EARLY_STOP,
        seed=SEED,
        device=DEVICE,
        loss_name=loss_name,
        loss_params=loss_params
    )
    
    # Test
    auc, f1, gm, ap, auc1, prec, rec = test(model_best, test_loader, DEVICE)
    
    results[loss_desc] = {
        'AUC': auc,
        'F1': f1,
        'Precision': prec,
        'Recall': rec,
        'G-mean': gm,
        'AP': ap,
        'Best Epoch': best_epoch,
        'Time (s)': train_time
    }
    
    print(f"\n{loss_desc} Results:")
    print(f"  AUC: {auc:.4f}")
    print(f"  F1-score: {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  Best epoch: {best_epoch}")

# Print summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"{'Loss Function':<30} {'AUC':<10} {'F1':<10} {'Precision':<12} {'Recall':<10}")
print("-"*80)

for loss_desc, metrics in results.items():
    print(f"{loss_desc:<30} "
          f"{metrics['AUC']:<10.4f} "
          f"{metrics['F1']:<10.4f} "
          f"{metrics['Precision']:<12.4f} "
          f"{metrics['Recall']:<10.4f}")

print("="*80)

# Find best performing loss
best_loss = max(results.items(), key=lambda x: x[1]['AUC'])
print(f"\nBest performing loss function: {best_loss[0]}")
print(f"AUC: {best_loss[1]['AUC']:.4f}")
print("\nNote: This is a quick test with reduced epochs.")
print("For full results, run with EPOCHS=300 as in the paper.")
