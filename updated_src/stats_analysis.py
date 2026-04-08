"""
Statistical Significance Testing for Loss Function Comparison

Implements:
  - Friedman test (non-parametric test for comparing multiple classifiers
    across multiple datasets/seeds)
  - Nemenyi post-hoc test (pairwise comparison when Friedman is significant)
  - Critical Difference (CD) diagram generation
  - Wilcoxon signed-rank tests for pairwise comparisons
  - Summary statistics with confidence intervals

References:
  Demšar, J. (2006). Statistical comparisons of classifiers over multiple
  data sets. JMLR, 7, 1-30.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
from typing import Dict, List, Tuple, Optional


# =========================================================================
# Core Statistical Tests
# =========================================================================

def friedman_test(score_matrix: np.ndarray) -> Tuple[float, float]:
    """
    Friedman test for comparing k classifiers across N experiments.

    The Friedman test ranks the classifiers for each experiment (seed)
    independently, then tests whether the average ranks differ
    significantly from the expected mean rank under H0 (all classifiers
    perform equally).

    Args:
        score_matrix: Shape (N, k) where N = number of seeds/experiments,
                      k = number of classifiers (loss functions).
                      Higher score = better.

    Returns:
        (chi2_statistic, p_value)
    """
    n_seeds, n_classifiers = score_matrix.shape

    # Rank classifiers within each seed (higher score → rank 1)
    ranks = np.zeros_like(score_matrix)
    for i in range(n_seeds):
        ranks[i] = stats.rankdata(-score_matrix[i])  # negative for descending

    avg_ranks = ranks.mean(axis=0)

    # Friedman chi-squared statistic
    chi2 = (12 * n_seeds / (n_classifiers * (n_classifiers + 1))) * \
           (np.sum(avg_ranks ** 2) - (n_classifiers * (n_classifiers + 1) ** 2) / 4)

    # Degrees of freedom = k - 1
    df = n_classifiers - 1
    p_value = 1 - stats.chi2.cdf(chi2, df)

    return chi2, p_value


def nemenyi_critical_difference(n_classifiers: int, n_seeds: int,
                                  alpha: float = 0.05) -> float:
    """
    Compute the Nemenyi critical difference (CD) threshold.

    If the difference in average ranks between two classifiers exceeds
    this CD value, the difference is statistically significant at level alpha.

    Uses the Studentized range distribution (q_alpha) table values from
    Demšar (2006), Table 5.

    Args:
        n_classifiers: Number of classifiers being compared (k)
        n_seeds: Number of experiments (N)
        alpha: Significance level (0.05 or 0.10)

    Returns:
        Critical difference value
    """
    # q_alpha values for the Studentized range statistic
    # Rows indexed by k (number of classifiers), columns by alpha
    # Source: Demšar (2006), extracted from statistical tables
    q_alpha_table = {
        0.05: {
            2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728,
            6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
        },
        0.10: {
            2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459,
            6: 2.589, 7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920
        }
    }

    if alpha not in q_alpha_table:
        raise ValueError(f"alpha must be 0.05 or 0.10, got {alpha}")
    if n_classifiers not in q_alpha_table[alpha]:
        raise ValueError(f"n_classifiers must be 2-10, got {n_classifiers}")

    q_alpha = q_alpha_table[alpha][n_classifiers]
    cd = q_alpha * np.sqrt(n_classifiers * (n_classifiers + 1) / (6 * n_seeds))
    return cd


def compute_average_ranks(score_matrix: np.ndarray) -> np.ndarray:
    """
    Compute average ranks across all seeds. Rank 1 = best.

    Args:
        score_matrix: Shape (N, k), higher is better.

    Returns:
        Array of shape (k,) with average ranks.
    """
    n_seeds = score_matrix.shape[0]
    ranks = np.zeros_like(score_matrix)
    for i in range(n_seeds):
        ranks[i] = stats.rankdata(-score_matrix[i])
    return ranks.mean(axis=0)


def wilcoxon_pairwise(score_matrix: np.ndarray,
                       classifier_names: List[str],
                       alpha: float = 0.05) -> pd.DataFrame:
    """
    Wilcoxon signed-rank test for all pairwise classifier comparisons.

    This is a stronger pairwise test than Nemenyi when you only care
    about specific comparisons (e.g., each loss vs baseline CE).

    Args:
        score_matrix: Shape (N, k), higher is better.
        classifier_names: List of k classifier names.
        alpha: Significance level.

    Returns:
        DataFrame with pairwise test results.
    """
    n_classifiers = score_matrix.shape[1]
    results = []

    for i, j in combinations(range(n_classifiers), 2):
        scores_i = score_matrix[:, i]
        scores_j = score_matrix[:, j]
        diff = scores_i - scores_j

        # Skip if all differences are zero (Wilcoxon can't handle this)
        if np.all(diff == 0):
            stat, p_val = 0.0, 1.0
        else:
            stat, p_val = stats.wilcoxon(scores_i, scores_j,
                                          alternative='two-sided')

        results.append({
            'classifier_a': classifier_names[i],
            'classifier_b': classifier_names[j],
            'mean_a': scores_i.mean(),
            'mean_b': scores_j.mean(),
            'statistic': stat,
            'p_value': p_val,
            'significant': p_val < alpha,
            'winner': classifier_names[i] if scores_i.mean() > scores_j.mean()
                      else classifier_names[j]
        })

    return pd.DataFrame(results)


# =========================================================================
# CD Diagram (text-based, suitable for .txt output)
# =========================================================================

def generate_cd_diagram_text(avg_ranks: np.ndarray,
                              classifier_names: List[str],
                              cd: float,
                              title: str = "Critical Difference Diagram"
                              ) -> str:
    """
    Generate a text-based Critical Difference diagram.

    Classifiers connected by a horizontal bar are NOT significantly
    different from each other (their rank difference < CD).

    Args:
        avg_ranks: Average rank for each classifier.
        classifier_names: Names of classifiers.
        cd: Critical difference value.
        title: Diagram title.

    Returns:
        Multi-line string representation of the CD diagram.
    """
    # Sort classifiers by average rank (best = rank 1 at top)
    sorted_indices = np.argsort(avg_ranks)
    sorted_names = [classifier_names[i] for i in sorted_indices]
    sorted_ranks = avg_ranks[sorted_indices]

    lines = []
    lines.append(title)
    lines.append("=" * 70)
    lines.append(f"Critical Difference (CD) = {cd:.4f} (alpha=0.05)")
    lines.append(f"Classifiers connected by a bracket are NOT significantly different.")
    lines.append("")

    # Print ranked list
    lines.append(f"{'Rank':<8} {'Classifier':<20} {'Avg Rank':<12} {'Groups'}")
    lines.append("-" * 70)

    # Identify groups of classifiers that are NOT significantly different
    n = len(sorted_ranks)
    groups = []
    for i in range(n):
        group = [i]
        for j in range(i + 1, n):
            if sorted_ranks[j] - sorted_ranks[i] < cd:
                group.append(j)
        if len(group) > 1:
            groups.append(group)

    # Remove subgroups (keep only maximal groups)
    maximal_groups = []
    for g in groups:
        is_subset = False
        for other in groups:
            if g != other and set(g).issubset(set(other)):
                is_subset = True
                break
        if not is_subset:
            maximal_groups.append(g)

    # Assign group labels
    group_labels = [''] * n
    group_chars = 'abcdefghijklmnopqrstuvwxyz'
    for gi, group in enumerate(maximal_groups):
        char = group_chars[gi] if gi < len(group_chars) else f'g{gi}'
        for idx in group:
            if group_labels[idx]:
                group_labels[idx] += f', {char}'
            else:
                group_labels[idx] = char

    for i, (name, rank) in enumerate(zip(sorted_names, sorted_ranks)):
        marker = " <-- best" if i == 0 else ""
        gl = group_labels[i] if group_labels[i] else "-"
        lines.append(f"{i+1:<8} {name:<20} {rank:<12.4f} {gl}{marker}")

    lines.append("")
    lines.append("Group legend (classifiers sharing a group letter are NOT significantly different):")
    for gi, group in enumerate(maximal_groups):
        char = group_chars[gi] if gi < len(group_chars) else f'g{gi}'
        members = [sorted_names[idx] for idx in group]
        lines.append(f"  Group {char}: {', '.join(members)}")

    return "\n".join(lines)


# =========================================================================
# Summary Report Generation
# =========================================================================

def compute_confidence_interval(data: np.ndarray,
                                  confidence: float = 0.95
                                  ) -> Tuple[float, float, float]:
    """
    Compute mean and confidence interval for a 1-D array.

    Args:
        data: Array of values (one per seed).
        confidence: Confidence level.

    Returns:
        (mean, ci_lower, ci_upper)
    """
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    if n < 2 or se == 0:
        return mean, mean, mean

    t_val = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    ci_lower = mean - t_val * se
    ci_upper = mean + t_val * se
    return mean, ci_lower, ci_upper


def generate_full_report(all_results: Dict,
                          metrics: List[str],
                          output_dir: str = "results",
                          dataset_name: str = "dataset") -> str:
    """
    Generate a complete statistical analysis report.

    Args:
        all_results: Nested dict structure:
            {loss_name: {seed: {metric_name: value, ...}, ...}, ...}
        metrics: List of metric names to analyze.
        output_dir: Directory to save reports.
        dataset_name: Name of the dataset for the report title.

    Returns:
        Path to the generated report file.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"{dataset_name}_statistical_report.txt")

    loss_names = sorted(all_results.keys())
    seeds = sorted(next(iter(all_results.values())).keys())
    n_seeds = len(seeds)
    n_classifiers = len(loss_names)

    lines = []
    lines.append("=" * 80)
    lines.append(f"STATISTICAL ANALYSIS REPORT")
    lines.append(f"Dataset: {dataset_name.upper()}")
    lines.append(f"Number of loss functions: {n_classifiers}")
    lines.append(f"Number of random seeds: {n_seeds}")
    lines.append(f"Seeds used: {seeds}")
    lines.append("=" * 80)

    # ---- Section 1: Summary Statistics with Confidence Intervals ----
    lines.append("\n" + "=" * 80)
    lines.append("SECTION 1: SUMMARY STATISTICS (mean ± 95% CI)")
    lines.append("=" * 80)

    for metric in metrics:
        lines.append(f"\n--- {metric.upper()} ---")
        header = f"{'Loss Function':<20} {'Mean':<10} {'Std':<10} {'95% CI':<24} {'Min':<10} {'Max':<10}"
        lines.append(header)
        lines.append("-" * 80)

        for loss_name in loss_names:
            values = np.array([all_results[loss_name][s][metric] for s in seeds])
            mean, ci_lo, ci_hi = compute_confidence_interval(values)
            std = np.std(values, ddof=1) if len(values) > 1 else 0.0
            lines.append(
                f"{loss_name:<20} {mean:<10.4f} {std:<10.4f} "
                f"[{ci_lo:.4f}, {ci_hi:.4f}]   {values.min():<10.4f} {values.max():<10.4f}"
            )

    # ---- Section 2: Friedman Test ----
    lines.append("\n" + "=" * 80)
    lines.append("SECTION 2: FRIEDMAN TEST")
    lines.append("=" * 80)
    lines.append("H0: All loss functions perform equally.")
    lines.append("H1: At least one loss function performs differently.\n")

    for metric in metrics:
        # Build score matrix: rows=seeds, cols=loss functions
        score_matrix = np.zeros((n_seeds, n_classifiers))
        for j, loss_name in enumerate(loss_names):
            for i, seed in enumerate(seeds):
                score_matrix[i, j] = all_results[loss_name][seed][metric]

        chi2, p_val = friedman_test(score_matrix)
        sig = "YES" if p_val < 0.05 else "NO"

        lines.append(f"Metric: {metric.upper()}")
        lines.append(f"  Friedman chi2 = {chi2:.4f}, p-value = {p_val:.6f}")
        lines.append(f"  Significant at alpha=0.05? {sig}")

        avg_ranks = compute_average_ranks(score_matrix)
        lines.append(f"  Average ranks: " +
                      ", ".join(f"{loss_names[k]}={avg_ranks[k]:.2f}"
                                for k in np.argsort(avg_ranks)))

        if p_val < 0.05 and n_classifiers >= 2:
            cd = nemenyi_critical_difference(n_classifiers, n_seeds, alpha=0.05)
            lines.append(f"  Nemenyi CD (alpha=0.05) = {cd:.4f}")
            lines.append("")

            cd_text = generate_cd_diagram_text(avg_ranks, loss_names, cd,
                                                title=f"CD Diagram for {metric.upper()}")
            lines.append(cd_text)
        lines.append("")

    # ---- Section 3: Pairwise Wilcoxon Tests ----
    lines.append("\n" + "=" * 80)
    lines.append("SECTION 3: PAIRWISE WILCOXON SIGNED-RANK TESTS")
    lines.append("=" * 80)
    lines.append("Tests whether each pair of loss functions differs significantly.\n")

    for metric in metrics:
        score_matrix = np.zeros((n_seeds, n_classifiers))
        for j, loss_name in enumerate(loss_names):
            for i, seed in enumerate(seeds):
                score_matrix[i, j] = all_results[loss_name][seed][metric]

        df_wilcoxon = wilcoxon_pairwise(score_matrix, loss_names)
        lines.append(f"--- {metric.upper()} ---")
        header = (f"{'Classifier A':<16} {'Classifier B':<16} "
                  f"{'Mean A':<10} {'Mean B':<10} {'p-value':<12} {'Sig?':<6} {'Winner'}")
        lines.append(header)
        lines.append("-" * 90)

        for _, row in df_wilcoxon.iterrows():
            sig_marker = "***" if row['p_value'] < 0.01 else \
                         "**" if row['p_value'] < 0.05 else \
                         "*" if row['p_value'] < 0.10 else ""
            lines.append(
                f"{row['classifier_a']:<16} {row['classifier_b']:<16} "
                f"{row['mean_a']:<10.4f} {row['mean_b']:<10.4f} "
                f"{row['p_value']:<12.6f} {sig_marker:<6} {row['winner']}"
            )
        lines.append("")

    # ---- Section 4: Baseline Comparison ----
    if 'ce' in loss_names:
        lines.append("\n" + "=" * 80)
        lines.append("SECTION 4: IMPROVEMENT OVER BASELINE (Cross-Entropy)")
        lines.append("=" * 80)

        for metric in metrics:
            lines.append(f"\n--- {metric.upper()} ---")
            ce_values = np.array([all_results['ce'][s][metric] for s in seeds])
            ce_mean = ce_values.mean()

            for loss_name in loss_names:
                if loss_name == 'ce':
                    continue
                other_values = np.array([all_results[loss_name][s][metric]
                                          for s in seeds])
                other_mean = other_values.mean()
                pct_change = ((other_mean - ce_mean) / ce_mean) * 100

                # Wilcoxon test against baseline
                diff = other_values - ce_values
                if np.all(diff == 0):
                    p_val = 1.0
                else:
                    _, p_val = stats.wilcoxon(other_values, ce_values)

                sig_marker = "***" if p_val < 0.01 else \
                             "**" if p_val < 0.05 else \
                             "*" if p_val < 0.10 else ""

                lines.append(
                    f"  {loss_name:<16} mean={other_mean:.4f}  "
                    f"change={pct_change:+.2f}%  p={p_val:.6f} {sig_marker}"
                )

    report_text = "\n".join(lines)

    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"\nStatistical report saved to: {report_path}")
    return report_path