"""
Multi-Seed Training Runner for LEX-GNN Loss Function Comparison

Runs each loss function configuration across multiple random seeds,
logs all individual results, and produces statistical significance
reports with Friedman + Nemenyi tests and CD diagrams.

Usage:
    # Run full 7 losses × 10 seeds on yelp
    python run_multi_seed.py --data_name yelp --n_seeds 10

    # Run on amazon with 5 seeds and custom GPU
    python run_multi_seed.py --data_name amazon --n_seeds 5 --cuda_id 1

    # Run a single loss with 10 seeds (for debugging)
    python run_multi_seed.py --data_name yelp --n_seeds 10 --losses ce focal

Paper-faithful defaults:
    batch_size=1024, n_layer=2, n_head=4, n_hidden=64, dropout=0.0,
    lr=0.005, wd=0.0001, beta=0.5, epochs=300, early_stop=100
"""

import os
import sys
import argparse
import json
import time
import random
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist

from model import LEXGNN
from data_handler import load_processed_data
from model_trainer_enhanced import (
    train, test, get_loss_function, reset_model_parameters
)
from stats_analysis import generate_full_report


# =========================================================================
# Constants
# =========================================================================

# 10 fixed seeds for reproducibility across all experiments.
# Using well-separated values to avoid correlated initialisations.
DEFAULT_SEEDS = [0, 1, 2, 3, 4, 42, 123, 256, 512, 1024]

# Paper-faithful hyperparameters (Table 2 in LEX-GNN paper)
PAPER_DEFAULTS = {
    'batch_size': 1024,
    'n_layer': 2,
    'n_head': 4,
    'n_hidden': 64,
    'dropout': 0.0,
    'lr': 0.005,
    'wd': 0.0001,
    'beta': 0.5,
    'epochs': 300,
    'valid_epochs': 3,
    'early_stop': 100,
}

# Metrics to track (including AUC-PR = average_precision)
METRICS = ['auc', 'f1', 'precision', 'recall', 'gmean', 'ap', 'auc_pre']

# All loss function configurations for the comparative study
ALL_LOSS_CONFIGS = {
    'ce': {},
    'weighted_ce': {'pos_weight': 5.0},
    'focal': {'alpha': 0.25, 'gamma': 2.0},
    'margin': {'margin': 1.0},
    'huber': {'delta': 1.0},
    'contrastive': {'margin': 2.0, 'temperature': 0.5},
    'dice': {'smooth': 1.0},
}


# =========================================================================
# Seed Management
# =========================================================================

def set_random_seed(seed: int):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# =========================================================================
# Result Logging
# =========================================================================

def log_single_result(filepath: str, record: dict):
    """Append a single experiment result as a line to a text file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'a') as f:
        f.write(json.dumps(record) + "\n")


def write_seed_summary(filepath: str, loss_name: str, seed: int,
                        metrics_dict: dict, best_epoch: int,
                        train_time: float):
    """Write a human-readable per-seed summary line."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'a') as f:
        line = (
            f"loss={loss_name:<16} seed={seed:<6} "
            f"AUC={metrics_dict['auc']:.4f}  "
            f"F1={metrics_dict['f1']:.4f}  "
            f"P={metrics_dict['precision']:.4f}  "
            f"R={metrics_dict['recall']:.4f}  "
            f"GM={metrics_dict['gmean']:.4f}  "
            f"AP={metrics_dict['ap']:.4f}  "
            f"AUC_pre={metrics_dict['auc_pre']:.4f}  "
            f"epoch={best_epoch}  time={train_time:.1f}s\n"
        )
        f.write(line)


def write_aggregate_summary(filepath: str, all_results: dict, seeds: list):
    """
    Write a formatted aggregate summary table showing mean ± std
    for every loss function across all seeds.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    loss_names = sorted(all_results.keys())

    with open(filepath, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("AGGREGATE RESULTS (mean ± std across seeds)\n")
        f.write(f"Seeds: {seeds}\n")
        f.write("=" * 100 + "\n\n")

        header = (
            f"{'Loss':<16} "
            f"{'AUC':<18} {'F1':<18} {'Precision':<18} "
            f"{'Recall':<18} {'G-Mean':<18} {'AP':<18}\n"
        )
        f.write(header)
        f.write("-" * 100 + "\n")

        for loss_name in loss_names:
            seed_metrics = all_results[loss_name]
            row_parts = [f"{loss_name:<16}"]
            for metric in ['auc', 'f1', 'precision', 'recall', 'gmean', 'ap']:
                values = np.array([seed_metrics[s][metric] for s in seeds])
                mean = values.mean()
                std = values.std(ddof=1) if len(values) > 1 else 0.0
                row_parts.append(f"{mean:.4f}±{std:.4f}")
            f.write(" ".join(f"{p:<18}" for p in row_parts) + "\n")

        f.write("=" * 100 + "\n")


# =========================================================================
# Single Experiment Runner
# =========================================================================

def run_single_seed(model_class, n_input: int, train_loader, valid_loader,
                     test_loader, loss_name: str, loss_params: dict,
                     seed: int, device, hparams: dict) -> dict:
    """
    Run one complete train→test cycle for a given loss and seed.

    Returns a dict with all metric values, best_epoch, and train_time.
    """
    set_random_seed(seed)

    # Initialise a fresh model for this seed
    model = model_class(
        n_input, 2,
        hparams['n_hidden'],
        hparams['n_layer'],
        hparams['n_head'],
        hparams['dropout']
    ).to(device)

    model_best, best_epoch, train_time = train(
        model, train_loader, valid_loader,
        epochs=hparams['epochs'],
        valid_epochs=hparams['valid_epochs'],
        beta=hparams['beta'],
        lr=hparams['lr'],
        weight_decay=hparams['wd'],
        early_stop=hparams['early_stop'],
        seed=seed,
        device=device,
        loss_name=loss_name,
        loss_params=loss_params,
    )

    auc, f1, gmean, ap, auc_pre, prec, rec = test(model_best, test_loader, device)

    return {
        'auc': auc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'gmean': gmean,
        'ap': ap,         # This is AUC-PR (average precision)
        'auc_pre': auc_pre,
        'best_epoch': best_epoch,
        'train_time': train_time,
    }


# =========================================================================
# Full Multi-Seed Runner
# =========================================================================

def run_multi_seed_experiment(args):
    """
    Main entry point: runs all requested losses × all seeds,
    logs every result, and produces statistical reports.
    """
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    _rank = dist.get_rank() if dist.is_initialized() else 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join("results", f"{args.data_name}_{timestamp}")
    if _rank == 0:
        os.makedirs(result_dir, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    seeds = DEFAULT_SEEDS[:args.n_seeds]
    loss_names = args.losses if args.losses else list(ALL_LOSS_CONFIGS.keys())

    hparams = {k: getattr(args, k, PAPER_DEFAULTS[k]) for k in PAPER_DEFAULTS}

    if _rank == 0:
        # Save experiment configuration
        config = {
            'data_name': args.data_name,
            'seeds': seeds,
            'loss_functions': loss_names,
            'hparams': hparams,
            'timestamp': timestamp,
            'cuda_id': args.cuda_id,
        }
        config_path = os.path.join(result_dir, "experiment_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    # File paths for logging (only used by rank 0)
    raw_log_path = os.path.join(result_dir, "raw_results.jsonl")
    seed_summary_path = os.path.join(result_dir, "per_seed_results.txt")
    aggregate_path = os.path.join(result_dir, "aggregate_results.txt")

    if _rank == 0:
        with open(seed_summary_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write(f"LEX-GNN Multi-Seed Results | Dataset: {args.data_name.upper()}\n")
            f.write(f"Date: {timestamp} | Seeds: {seeds}\n")
            f.write(f"Hyperparams: {hparams}\n")
            f.write("=" * 100 + "\n\n")

    # ------------------------------------------------------------------
    # GPU setup
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device(args.cuda_id)
        torch.cuda.set_device(device)
        if _rank == 0:
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        if _rank == 0:
            print("WARNING: CUDA not available, running on CPU (will be slow).")

    # ------------------------------------------------------------------
    # Load data once (shared across all seeds and losses)
    # ------------------------------------------------------------------
    if _rank == 0:
        print(f"\nLoading {args.data_name.upper()} dataset...")
    n_input, train_loader, valid_loader, test_loader = load_processed_data(
        args.data_name, hparams['batch_size'], hparams['n_layer'], num_workers=4,
    )
    if _rank == 0:
        print(f"Input feature dimension: {n_input}")

    # ------------------------------------------------------------------
    # Run experiments
    # ------------------------------------------------------------------
    # all_results[loss_name][seed] = {metric: value, ...}
    all_results = {}
    total_runs = len(loss_names) * len(seeds)
    run_idx = 0

    experiment_start = time.time()

    for loss_name in loss_names:
        loss_params = ALL_LOSS_CONFIGS.get(loss_name, {})
        all_results[loss_name] = {}

        if _rank == 0:
            print(f"\n{'='*80}")
            print(f"LOSS FUNCTION: {loss_name.upper()}")
            print(f"Parameters: {loss_params}")
            print(f"{'='*80}")

        for seed in seeds:
            run_idx += 1
            if _rank == 0:
                print(f"\n--- [{run_idx}/{total_runs}] "
                      f"loss={loss_name}, seed={seed} ---")

            result = run_single_seed(
                LEXGNN, n_input,
                train_loader, valid_loader, test_loader,
                loss_name, loss_params,
                seed, device, hparams,
            )

            if _rank == 0:
                # Store result
                all_results[loss_name][seed] = result

                # Log raw JSON line
                record = {
                    'loss': loss_name,
                    'seed': seed,
                    'dataset': args.data_name,
                    **result,
                }
                log_single_result(raw_log_path, record)

                # Log human-readable line
                write_seed_summary(seed_summary_path, loss_name, seed,
                                    result, result['best_epoch'],
                                    result['train_time'])

                print(f"    AUC={result['auc']:.4f}  F1={result['f1']:.4f}  "
                      f"AP={result['ap']:.4f}  epoch={result['best_epoch']}")

    if _rank == 0:
        total_time = time.time() - experiment_start
        print(f"\nAll {total_runs} runs completed in {total_time:.1f}s "
              f"({total_time/60:.1f} min)")

        # ------------------------------------------------------------------
        # Aggregate summary
        # ------------------------------------------------------------------
        write_aggregate_summary(aggregate_path, all_results, seeds)
        print(f"Aggregate results saved to: {aggregate_path}")

        # ------------------------------------------------------------------
        # Statistical significance analysis
        # ------------------------------------------------------------------
        if len(seeds) >= 3:
            analysis_metrics = ['auc', 'f1', 'precision', 'recall', 'gmean', 'ap']
            report_path = generate_full_report(
                all_results, analysis_metrics,
                output_dir=result_dir,
                dataset_name=args.data_name,
            )
            print(f"Statistical report saved to: {report_path}")
        else:
            print("WARNING: Need at least 3 seeds for statistical tests. "
                  "Skipping significance analysis.")

        # ------------------------------------------------------------------
        # Final console summary
        # ------------------------------------------------------------------
        print(f"\n{'='*80}")
        print("EXPERIMENT COMPLETE")
        print(f"{'='*80}")
        print(f"All output files in: {result_dir}/")
        print(f"  - experiment_config.json   (reproducibility config)")
        print(f"  - raw_results.jsonl        (machine-readable per-seed results)")
        print(f"  - per_seed_results.txt     (human-readable per-seed results)")
        print(f"  - aggregate_results.txt    (mean ± std table)")
        if len(seeds) >= 3:
            print(f"  - {args.data_name}_statistical_report.txt  "
                  f"(Friedman, Nemenyi, Wilcoxon)")
        print(f"{'='*80}")


# =========================================================================
# CLI
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="LEX-GNN Multi-Seed Training with Statistical Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core experiment settings
    parser.add_argument('--data_name', type=str, default='yelp',
                        choices=['yelp', 'amazon'],
                        help='Dataset name')
    parser.add_argument('--n_seeds', type=int, default=10,
                        help='Number of random seeds to run (uses first N '
                             'from the fixed seed list)')
    parser.add_argument('--losses', nargs='+', type=str, default=None,
                        choices=list(ALL_LOSS_CONFIGS.keys()),
                        help='Subset of loss functions to run. '
                             'Default: all 7 losses.')
    parser.add_argument('--cuda_id', type=int, default=0,
                        help='CUDA device ID')
    parser.add_argument('--ddp', action='store_true',
                        help='Enable DistributedDataParallel (launch with torchrun)')

    # Model hyperparameters (paper defaults)
    parser.add_argument('--batch_size', type=int,
                        default=PAPER_DEFAULTS['batch_size'])
    parser.add_argument('--n_layer', type=int,
                        default=PAPER_DEFAULTS['n_layer'])
    parser.add_argument('--n_head', type=int,
                        default=PAPER_DEFAULTS['n_head'])
    parser.add_argument('--n_hidden', type=int,
                        default=PAPER_DEFAULTS['n_hidden'])
    parser.add_argument('--dropout', type=float,
                        default=PAPER_DEFAULTS['dropout'])

    # Training hyperparameters (paper defaults)
    parser.add_argument('--epochs', type=int,
                        default=PAPER_DEFAULTS['epochs'])
    parser.add_argument('--valid_epochs', type=int,
                        default=PAPER_DEFAULTS['valid_epochs'])
    parser.add_argument('--early_stop', type=int,
                        default=PAPER_DEFAULTS['early_stop'])
    parser.add_argument('--lr', type=float,
                        default=PAPER_DEFAULTS['lr'])
    parser.add_argument('--wd', type=float,
                        default=PAPER_DEFAULTS['wd'])
    parser.add_argument('--beta', type=float,
                        default=PAPER_DEFAULTS['beta'])

    # CPU thread limiting
    parser.add_argument('--cpu_threads', type=int, default=20,
                        help='Limit CPU threads for PyTorch')

    return parser.parse_args()


def main():
    args = parse_args()

    # Apply CPU thread limit
    torch.set_num_threads(args.cpu_threads)
    os.environ['OMP_NUM_THREADS'] = str(args.cpu_threads)
    os.environ['MKL_NUM_THREADS'] = str(args.cpu_threads)

    if args.ddp:
        dist.init_process_group('nccl')
        args.cuda_id = int(os.environ.get('LOCAL_RANK', 0))

    try:
        run_multi_seed_experiment(args)
    finally:
        if args.ddp:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()