import os
import argparse
import torch
import random
import numpy as np
from model import LEXGNN
from data_handler import load_processed_data
from model_trainer_enhanced import *
import warnings
warnings.filterwarnings("ignore")

# Limit CPU usage to prevent high CPU utilization
torch.set_num_threads(20)
# torch.set_num_interop_threads(20)


def set_random_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed) 


def run_single_experiment(args):
    """Run experiment with a single loss function"""
    print('='*80)
    print('SINGLE LOSS FUNCTION EXPERIMENT')
    print('='*80)
    print(f'Dataset: {args.data_name.upper()}')
    print(f'Loss Function: {args.loss_function.upper()}')
    print('='*80)

    # Set seed
    set_random_seed(args.seed)

    # GPU
    device = torch.device(args.cuda_id)
    torch.cuda.set_device(device)

    # Load data
    n_input, train_loader, valid_loader, test_loader = load_processed_data(
        args.data_name, 
        args.batch_size, 
        args.n_layer
    )

    # Define model
    model = LEXGNN(n_input, 2, args.n_hidden, args.n_layer, args.n_head, args.dropout).to(device)

    # Prepare loss function parameters
    loss_params = {}
    if args.loss_function == 'weighted_ce':
        loss_params['pos_weight'] = args.pos_weight
    elif args.loss_function == 'focal':
        loss_params['alpha'] = args.focal_alpha
        loss_params['gamma'] = args.focal_gamma
    elif args.loss_function == 'margin':
        loss_params['margin'] = args.margin
    elif args.loss_function == 'huber':
        loss_params['delta'] = args.huber_delta
    elif args.loss_function == 'contrastive':
        loss_params['margin'] = args.contrastive_margin
        loss_params['temperature'] = args.contrastive_temp
    elif args.loss_function == 'dice':
        loss_params['smooth'] = args.dice_smooth

    # Train 
    model_best, ep, et = train(
        model, train_loader, valid_loader, 
        args.epochs, args.valid_epochs, 
        args.beta, args.lr, args.wd, 
        args.early_stop, args.seed, device,
        loss_name=args.loss_function,
        loss_params=loss_params
    )

    # Test 
    auc, f1, gm, ap, auc1, prec, rec = test(model_best, test_loader, device)
    
    print('\n' + '='*80)
    print('FINAL TEST RESULTS')
    print('='*80)
    print(f'Loss Function: {args.loss_function.upper()}')
    print(f'AUC (cls): {auc:.4f} | AUC (pre): {auc1:.4f}')
    print(f'F1-macro: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}')
    print(f'G-mean: {gm:.4f} | AP: {ap:.4f}')
    print(f'Best Epoch: {ep} | Training Time: {et:.2f}s')
    print('='*80)


def run_comparative_study(args):
    """Run comparative study with multiple loss functions"""
    print('='*80)
    print('COMPARATIVE STUDY: MULTIPLE LOSS FUNCTIONS')
    print('='*80)
    print(f'Dataset: {args.data_name.upper()}')
    print('='*80)

    # Set seed
    set_random_seed(args.seed)

    # GPU
    device = torch.device(args.cuda_id)
    torch.cuda.set_device(device)

    # Load data
    n_input, train_loader, valid_loader, test_loader = load_processed_data(
        args.data_name, 
        args.batch_size, 
        args.n_layer
    )

    # Model parameters
    model_params = {
        'in_dim': n_input,
        'n_class': 2,
        'hidden_dim': args.n_hidden,
        'n_layer': args.n_layer,
        'num_heads': args.n_head,
        'dropout': args.dropout
    }

    # Define loss function configurations
    loss_configs = [
        # 1. Cross-Entropy (Baseline from paper)
        {
            'name': 'ce',
            'params': {}
        },
        # 2. Weighted Cross-Entropy
        {
            'name': 'weighted_ce',
            'params': {'pos_weight': args.pos_weight}
        },
        # 3. Focal Loss
        {
            'name': 'focal',
            'params': {'alpha': args.focal_alpha, 'gamma': args.focal_gamma}
        },
        # 4. Margin Loss
        {
            'name': 'margin',
            'params': {'margin': args.margin}
        },
        # 5. Huber Loss
        {
            'name': 'huber',
            'params': {'delta': args.huber_delta}
        },
        # 6. Contrastive Loss
        {
            'name': 'contrastive',
            'params': {'margin': args.contrastive_margin, 'temperature': args.contrastive_temp}
        },
        # 7. Dice Loss
        {
            'name': 'dice',
            'params': {'smooth': args.dice_smooth}
        }
    ]

    # Run comparative study
    results = train_with_multiple_losses(
        LEXGNN, model_params,
        train_loader, valid_loader, test_loader,
        args.epochs, args.valid_epochs,
        args.beta, args.lr, args.wd,
        args.early_stop, args.seed, device,
        loss_configs
    )

    # Print summary
    print('\n' + '='*80)
    print('COMPARATIVE STUDY SUMMARY')
    print('='*80)
    print(f"{'Loss Function':<20} {'AUC':<10} {'F1':<10} {'Precision':<12} {'Recall':<10} {'AP':<10}")
    print('-'*80)
    
    for loss_name, metrics in results.items():
        print(f"{loss_name.upper():<20} "
              f"{metrics['auc']:<10.4f} "
              f"{metrics['f1']:<10.4f} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<10.4f} "
              f"{metrics['ap']:<10.4f}")
    print('='*80)

    # Save results
    df = save_results_to_csv(results, f'{args.data_name}_loss_comparison.csv')
    print(f"\nDetailed results saved to {args.data_name}_loss_comparison.csv")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='LEX-GNN with Multiple Loss Functions')
    
    # Data parameters
    parser.add_argument('--data_name', type=str, default='yelp', 
                       help='Dataset to use: amazon or yelp')
    parser.add_argument('--batch_size', type=int, default=1024)
    
    # Model parameters
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)
    
    # Training parameters
    parser.add_argument('--seed', type=int, default=2, help='Random seed')
    parser.add_argument('--epochs', type=int, default=300, help='Maximum number of training epochs')
    parser.add_argument('--valid_epochs', type=int, default=3, help='Validate every N epochs')
    parser.add_argument('--early_stop', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--beta', type=float, default=0.5, 
                       help='Weight for auxiliary loss (label exploration)')
    parser.add_argument('--cuda_id', type=int, default=0)
    
    # Experiment mode
    parser.add_argument('--mode', type=str, default='single', 
                       choices=['single', 'comparative'],
                       help='Run single experiment or comparative study')
    parser.add_argument('--loss_function', type=str, default='ce',
                       choices=['ce', 'weighted_ce', 'focal', 'margin', 'huber', 'contrastive', 'dice'],
                       help='Loss function to use in single mode')
    
    # Loss function specific parameters
    # Weighted CE
    parser.add_argument('--pos_weight', type=float, default=5.0,
                       help='Weight for positive class in weighted CE')
    
    # Focal Loss
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                       help='Alpha parameter for focal loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Gamma parameter for focal loss')
    
    # Margin Loss
    parser.add_argument('--margin', type=float, default=1.0,
                       help='Margin for margin loss')
    
    # Huber Loss
    parser.add_argument('--huber_delta', type=float, default=1.0,
                       help='Delta parameter for Huber loss')
    
    # Contrastive Loss
    parser.add_argument('--contrastive_margin', type=float, default=2.0,
                       help='Margin for contrastive loss')
    parser.add_argument('--contrastive_temp', type=float, default=0.5,
                       help='Temperature for contrastive loss')
    
    # Dice Loss
    parser.add_argument('--dice_smooth', type=float, default=1.0,
                       help='Smoothing parameter for Dice loss')
    
    args = parser.parse_args()
    
    # Run experiment based on mode
    if args.mode == 'single':
        run_single_experiment(args)
    else:  # comparative
        run_comparative_study(args)


if __name__ == '__main__':
    main()