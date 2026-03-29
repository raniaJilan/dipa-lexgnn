"""
Visualization Script for Loss Function Comparison Results
Generates plots comparing different loss functions
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_loss_comparison(csv_file, output_prefix='loss_comparison'):
    """
    Create visualizations from loss comparison results
    
    Args:
        csv_file: Path to CSV file with results
        output_prefix: Prefix for output image files
    """
    # Read results
    df = pd.read_csv(csv_file, index_col=0)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Loss Function Comparison for Fraud Detection', fontsize=16, fontweight='bold')
    
    metrics = ['auc', 'f1', 'precision', 'recall', 'gmean', 'ap']
    titles = ['AUC-ROC', 'F1-Score (Macro)', 'Precision (Macro)', 
              'Recall (Macro)', 'G-Mean', 'Average Precision']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        ax = axes[idx // 3, idx % 3]
        
        # Sort by metric value
        sorted_df = df.sort_values(by=metric, ascending=False)
        
        # Create bar plot
        bars = ax.barh(sorted_df.index.str.upper(), sorted_df[metric], color=color, alpha=0.7)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, sorted_df[metric])):
            ax.text(value + 0.01, i, f'{value:.4f}', 
                   va='center', fontweight='bold', fontsize=9)
        
        # Highlight baseline (CE)
        if 'ce' in sorted_df.index or 'cross_entropy' in sorted_df.index:
            baseline_idx = None
            if 'ce' in sorted_df.index:
                baseline_idx = list(sorted_df.index).index('ce')
            elif 'cross_entropy' in sorted_df.index:
                baseline_idx = list(sorted_df.index).index('cross_entropy')
            
            if baseline_idx is not None:
                bars[baseline_idx].set_edgecolor('black')
                bars[baseline_idx].set_linewidth(2)
                bars[baseline_idx].set_alpha(1.0)
        
        ax.set_xlabel(title, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.3)
        
        # Add baseline line if CE exists
        if 'ce' in df.index:
            baseline_value = df.loc['ce', metric]
            ax.axvline(baseline_value, color='red', linestyle='--', 
                      linewidth=1, alpha=0.5, label='Baseline (CE)')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_metrics.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_prefix}_metrics.png")
    plt.close()
    
    # Create training time comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_df = df.sort_values(by='train_time', ascending=True)
    bars = ax.barh(sorted_df.index.str.upper(), sorted_df['train_time'], 
                   color='#34495e', alpha=0.7)
    
    for i, (bar, value) in enumerate(zip(bars, sorted_df['train_time'])):
        ax.text(value + max(sorted_df['train_time'])*0.02, i, 
               f'{value:.2f}s', va='center', fontweight='bold')
    
    ax.set_xlabel('Training Time (seconds)', fontweight='bold')
    ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_time.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_prefix}_time.png")
    plt.close()
    
    # Create radar chart for top 3 loss functions
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Select metrics for radar chart
    radar_metrics = ['auc', 'f1', 'precision', 'recall', 'gmean', 'ap']
    
    # Get top 3 loss functions by AUC
    top_losses = df.nlargest(3, 'auc')
    
    # Number of variables
    num_vars = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.upper() for m in radar_metrics], fontsize=10)
    
    colors_radar = ['#3498db', '#e74c3c', '#2ecc71']
    
    for idx, (loss_name, row) in enumerate(top_losses.iterrows()):
        values = row[radar_metrics].tolist()
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, 
               label=loss_name.upper(), color=colors_radar[idx])
        ax.fill(angles, values, alpha=0.15, color=colors_radar[idx])
    
    ax.set_ylim(0, 1)
    ax.set_title('Top 3 Loss Functions - Performance Radar', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_radar.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_prefix}_radar.png")
    plt.close()
    
    # Create heatmap of all metrics
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize metrics to [0, 1] for better visualization
    heatmap_data = df[metrics].T
    
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlGn', 
               cbar_kws={'label': 'Score'}, ax=ax, vmin=0, vmax=1,
               linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel('Loss Function', fontweight='bold')
    ax.set_ylabel('Metric', fontweight='bold')
    ax.set_title('Loss Function Performance Heatmap', 
                fontsize=14, fontweight='bold')
    
    # Capitalize column names
    ax.set_xticklabels([col.upper() for col in heatmap_data.columns], rotation=45)
    ax.set_yticklabels([row.upper() for row in heatmap_data.index], rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_prefix}_heatmap.png")
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for metric in metrics:
        best_loss = df[metric].idxmax()
        best_value = df[metric].max()
        print(f"\nBest {metric.upper()}: {best_loss.upper()} ({best_value:.4f})")
        
        if 'ce' in df.index:
            baseline = df.loc['ce', metric]
            improvement = ((best_value - baseline) / baseline) * 100
            print(f"  Improvement over baseline: {improvement:+.2f}%")
    
    print("\n" + "="*80)


def create_comparison_table(csv_file, output_file='comparison_table.txt'):
    """
    Create a formatted comparison table
    
    Args:
        csv_file: Path to CSV file with results
        output_file: Output text file for the table
    """
    df = pd.read_csv(csv_file, index_col=0)
    
    # Sort by AUC
    df = df.sort_values(by='auc', ascending=False)
    
    # Create table
    with open(output_file, 'w') as f:
        f.write("="*120 + "\n")
        f.write("LOSS FUNCTION COMPARISON - DETAILED RESULTS\n")
        f.write("="*120 + "\n\n")
        
        # Header
        f.write(f"{'Loss Function':<20} {'AUC':<10} {'F1':<10} {'Prec':<10} "
               f"{'Recall':<10} {'G-Mean':<10} {'AP':<10} {'Epoch':<8} {'Time(s)':<10}\n")
        f.write("-"*120 + "\n")
        
        # Data
        for loss_name, row in df.iterrows():
            f.write(f"{loss_name.upper():<20} "
                   f"{row['auc']:<10.4f} "
                   f"{row['f1']:<10.4f} "
                   f"{row['precision']:<10.4f} "
                   f"{row['recall']:<10.4f} "
                   f"{row['gmean']:<10.4f} "
                   f"{row['ap']:<10.4f} "
                   f"{int(row['best_epoch']):<8} "
                   f"{row['train_time']:<10.2f}\n")
        
        f.write("="*120 + "\n")
        
        # Add baseline comparison if CE exists
        if 'ce' in df.index:
            f.write("\nIMPROVEMENT OVER BASELINE (Cross-Entropy):\n")
            f.write("-"*120 + "\n")
            
            baseline = df.loc['ce']
            for loss_name, row in df.iterrows():
                if loss_name != 'ce':
                    auc_imp = ((row['auc'] - baseline['auc']) / baseline['auc']) * 100
                    f1_imp = ((row['f1'] - baseline['f1']) / baseline['f1']) * 100
                    f.write(f"{loss_name.upper():<20} AUC: {auc_imp:+.2f}%  F1: {f1_imp:+.2f}%\n")
    
    print(f"\nSaved: {output_file}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_results.py <csv_file> [output_prefix]")
        print("Example: python visualize_results.py yelp_loss_comparison.csv yelp")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else 'loss_comparison'
    
    print(f"Processing results from: {csv_file}")
    print(f"Output prefix: {output_prefix}")
    print()
    
    # Generate visualizations
    plot_loss_comparison(csv_file, output_prefix)
    
    # Create text table
    create_comparison_table(csv_file, f'{output_prefix}_table.txt')
    
    print("\nVisualization complete!")
    print(f"Generated files:")
    print(f"  - {output_prefix}_metrics.png")
    print(f"  - {output_prefix}_time.png")
    print(f"  - {output_prefix}_radar.png")
    print(f"  - {output_prefix}_heatmap.png")
    print(f"  - {output_prefix}_table.txt")
