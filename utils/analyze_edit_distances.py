#!/usr/bin/env python3
"""
Polars-based edit distance analysis with streaming for large datasets.

Usage:
    # Analyze valid reads only
    python analyze_edit_distances.py <valid_parquet> [--ont_error_rate 0.05] [--sample_size 10000]

    # Compare valid and invalid reads
    python analyze_edit_distances.py <valid_parquet> --invalid_parquet <invalid_parquet> [options]
"""

import argparse
import sys
from pathlib import Path
from math import ceil

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sns.set_style("whitegrid")

# Helpers

def is_concatenated_read(lf):
    """
    Detect concatenated reads by checking for comma-separated values in Starts/Ends fields.
    Returns a boolean expression identifying concatenated reads.
    """
    # Check if any segment has comma-separated starts (indicating multiple instances)
    schema = lf.collect_schema()
    concat_checks = []

    for col in schema:
        if col.endswith("_Starts") or col.endswith("_Ends"):
            concat_checks.append(
                pl.col(col).cast(pl.Utf8).str.contains(",")
            )

    if concat_checks:
        # True if ANY segment field contains commas
        return pl.any_horizontal(concat_checks).fill_null(False)
    else:
        return pl.lit(False)


def add_segment_scores(lf, segments, segment_lengths):
    """
    Add inverted normalized edit distance scores for each segment.

    Score = 1.0 - (edit_distance / segment_length)
    - 1.0 = perfect match (0 edit distance)
    - 0.0 = worst possible (edit distance >= segment length)
    - NMF and concatenated reads get null scores

    Returns LazyFrame with added score columns.
    """
    # Identify concatenated reads
    is_concat = is_concatenated_read(lf)

    score_exprs = []

    for seg in segments:
        ed_col = f"{seg}_edit_distance"
        length = segment_lengths.get(seg)

        if length is None or length == 0:
            continue

        # Inverted normalized score: 1.0 = perfect, 0.0 = worst
        score_expr = (
            pl.when(is_concat)
            .then(None)  # Skip concatenated reads
            .when(pl.col(ed_col) == "NMF")
            .then(None)  # Skip undetected segments
            .otherwise(
                (1.0 - (
                    pl.col(ed_col)
                    .cast(pl.Float32, strict=False)
                    / pl.lit(float(length))
                )).clip(0.0, 1.0)  # Clamp to [0, 1]
            )
            .alias(f"{seg}_score")
        )
        score_exprs.append(score_expr)

    # Add all score columns
    if score_exprs:
        lf = lf.with_columns(score_exprs)

        # Add composite read-level quality score (mean across all segment scores)
        score_cols = [f"{seg}_score" for seg in segments if segment_lengths.get(seg)]
        lf = lf.with_columns([
            pl.mean_horizontal(score_cols).alias("read_quality_score")
        ])

    return lf


def calculate_ont_threshold(segment_length, error_rate):
    """Calculate acceptable edit distance based on ONT error rate and segment length."""
    if segment_length is None or segment_length == 0:
        return None
    return ceil(segment_length * error_rate)

# Identify edit distance columns

def identify_edit_distance_columns(schema):
    """Identify edit distance columns and categorize segments."""
    edit_cols = [c for c in schema if c.endswith("_edit_distance")]

    barcode_segments = []
    fixed_segments = []

    for col in edit_cols:
        segment = col.replace("_edit_distance", "")
        if segment in ("CBC", "i7", "i5"):
            barcode_segments.append(segment)
        else:
            fixed_segments.append(segment)

    return edit_cols, barcode_segments, fixed_segments

# Estimate segment lengths efficiently by sampling

def compute_segment_lengths(lf, segments, sample_size=10_000):
    """
    Estimate median segment lengths using streaming-friendly sampling.

    Args:
        lf: LazyFrame with annotation data
        segments: List of segment names
        sample_size: Number of rows to sample

    Returns:
        Dictionary mapping segment names to median lengths
    """
    print(f"  Sampling {sample_size:,} reads for length estimation...")

    # Get total rows for sampling calculation
    total_rows = lf.select(pl.len()).collect().item()

    if total_rows > sample_size:
        # Sample evenly across file
        skip = max(1, total_rows // sample_size)
        df_sample = (
            lf.with_row_index("_idx")
            .filter(pl.col("_idx") % skip == 0)
            .head(sample_size)
            .collect(engine="streaming")
        )
        print(f"  (from {total_rows:,} total reads)")
    else:
        df_sample = lf.collect(engine="streaming")

    lengths = {}

    for seg in segments:
        start_col = f"{seg}_Starts"
        end_col = f"{seg}_Ends"
        seq_col = f"{seg}_Sequences"

        length = None

        # Try Start/End positions first
        if start_col in df_sample.columns and end_col in df_sample.columns:
            # Handle comma-separated values (e.g., "100, 200") by taking first
            starts = (
                df_sample[start_col]
                .cast(pl.Utf8)
                .str.split(",")
                .list.first()
                .str.strip_chars()
                .cast(pl.Int64, strict=False)
            )
            ends = (
                df_sample[end_col]
                .cast(pl.Utf8)
                .str.split(",")
                .list.first()
                .str.strip_chars()
                .cast(pl.Int64, strict=False)
            )

            # Calculate lengths (filter out nulls and negatives)
            diffs = (ends - starts).filter(
                starts.is_not_null() &
                ends.is_not_null() &
                ((ends - starts) > 0)
            )

            if len(diffs) > 0:
                length = diffs.median()

        # Fallback: estimate from sequence column
        if length is None and seq_col in df_sample.columns:
            seq_lengths = df_sample[seq_col].str.len_bytes().drop_nulls()
            if len(seq_lengths) > 0:
                length = seq_lengths.median()

        lengths[seg] = length

    return lengths


def add_segment_length_columns(lf, segments):
    """
    Add actual segment length columns (not just median) for distribution analysis.

    Returns LazyFrame with {segment}_length columns added.
    """
    length_exprs = []

    for seg in segments:
        start_col = f"{seg}_Starts"
        end_col = f"{seg}_Ends"
        seq_col = f"{seg}_Sequences"

        # Try Start/End positions first
        if start_col in lf.collect_schema() and end_col in lf.collect_schema():
            # Handle comma-separated values by taking first
            length_expr = (
                pl.when(
                    pl.col(start_col).cast(pl.Utf8).str.contains(",").fill_null(False)
                )
                .then(None)  # Skip concatenated reads
                .otherwise(
                    (
                        pl.col(end_col).cast(pl.Utf8).str.split(",").list.first().str.strip_chars().cast(pl.Int64, strict=False)
                        - pl.col(start_col).cast(pl.Utf8).str.split(",").list.first().str.strip_chars().cast(pl.Int64, strict=False)
                    )
                )
                .alias(f"{seg}_length")
            )
            length_exprs.append(length_expr)
        # Fallback: use sequence length
        elif seq_col in lf.collect_schema():
            length_expr = (
                pl.col(seq_col)
                .str.len_bytes()
                .alias(f"{seg}_length")
            )
            length_exprs.append(length_expr)

    if length_exprs:
        lf = lf.with_columns(length_exprs)

    return lf


def compute_segment_length_statistics(lf, segments, median_lengths, anomaly_thresholds=None):
    """
    Compute comprehensive length statistics for each segment.

    Args:
        lf: LazyFrame with segment length columns
        segments: List of segment names
        median_lengths: Dictionary of median segment lengths
        anomaly_thresholds: Dictionary of minimum expected lengths per segment (e.g., {'cDNA': 50, 'polyA': 10})

    Returns DataFrame with length statistics per segment.
    """
    print("\nCalculating segment length statistics...")

    if anomaly_thresholds is None:
        anomaly_thresholds = {}

    all_length_stats = []

    for seg in segments:
        length_col = f"{seg}_length"

        if length_col not in lf.collect_schema():
            continue

        median_length = median_lengths.get(seg)
        min_expected = anomaly_thresholds.get(seg)

        # Compute statistics
        stats_expr = [
            pl.col(length_col).count().alias("reads_with_length"),
            pl.col(length_col).mean().alias("mean_length"),
            pl.col(length_col).median().alias("median_length"),
            pl.col(length_col).std().alias("std_length"),
            pl.col(length_col).min().alias("min_length"),
            pl.col(length_col).max().alias("max_length"),
            pl.col(length_col).quantile(0.01, "nearest").alias("p01_length"),
            pl.col(length_col).quantile(0.05, "nearest").alias("p05_length"),
            pl.col(length_col).quantile(0.25, "nearest").alias("p25_length"),
            pl.col(length_col).quantile(0.75, "nearest").alias("p75_length"),
            pl.col(length_col).quantile(0.95, "nearest").alias("p95_length"),
            pl.col(length_col).quantile(0.99, "nearest").alias("p99_length"),
        ]

        result = lf.select(stats_expr).collect().to_dicts()[0]

        # Calculate outliers (>2 std from median)
        if median_length and result["std_length"]:
            lower_bound = median_length - 2 * result["std_length"]
            upper_bound = median_length + 2 * result["std_length"]

            outlier_count = lf.filter(
                (pl.col(length_col).cast(pl.Int64, strict=False) < lower_bound) |
                (pl.col(length_col).cast(pl.Int64, strict=False) > upper_bound)
            ).select(pl.len()).collect().item()

            outlier_pct = 100 * outlier_count / result["reads_with_length"] if result["reads_with_length"] > 0 else 0
        else:
            outlier_count = None
            outlier_pct = None
            lower_bound = None
            upper_bound = None

        # Detect anomalies (very short segments)
        anomaly_count = None
        anomaly_pct = None
        if min_expected is not None:
            anomaly_count = lf.filter(
                (pl.col(length_col).cast(pl.Int64, strict=False).is_not_null()) &
                (pl.col(length_col).cast(pl.Int64, strict=False) < min_expected)
            ).select(pl.len()).collect().item()

            anomaly_pct = 100 * anomaly_count / result["reads_with_length"] if result["reads_with_length"] > 0 else 0

        stats = {
            "segment": seg,
            "reads_with_length": result["reads_with_length"],
            "mean_length": result["mean_length"],
            "median_length": result["median_length"],
            "std_length": result["std_length"],
            "min_length": result["min_length"],
            "max_length": result["max_length"],
            "p01_length": result["p01_length"],
            "p05_length": result["p05_length"],
            "p25_length": result["p25_length"],
            "p75_length": result["p75_length"],
            "p95_length": result["p95_length"],
            "p99_length": result["p99_length"],
            "outlier_count": outlier_count,
            "outlier_percent": outlier_pct,
            "outlier_lower_bound": lower_bound,
            "outlier_upper_bound": upper_bound,
            "min_expected_length": min_expected,
            "anomaly_count": anomaly_count,
            "anomaly_percent": anomaly_pct,
        }

        all_length_stats.append(stats)

        # Print with anomaly info if available
        if anomaly_count is not None:
            print(f"  {seg}: mean={result['mean_length']:.1f}bp, std={result['std_length']:.1f}bp, "
                  f"anomalies (<{min_expected}bp)={anomaly_pct:.1f}% ({anomaly_count:,} reads)" if result['std_length']
                  else f"  {seg}: mean={result['mean_length']:.1f}bp, anomalies (<{min_expected}bp)={anomaly_pct:.1f}%")
        else:
            print(f"  {seg}: mean={result['mean_length']:.1f}bp, std={result['std_length']:.1f}bp" if result['std_length']
                  else f"  {seg}: mean={result['mean_length']:.1f}bp")

    return pl.DataFrame(all_length_stats)


# Compute statistics per segment (streaming-friendly)

def compute_segment_statistics(lf, segment, segment_length, ont_error_rate):
    """
    Compute comprehensive statistics for one segment using Polars, handling string-float edit distances robustly.

    Args:
        lf: LazyFrame with annotation data
        segment: Segment name
        segment_length: Median segment length
        ont_error_rate: Expected ONT error rate

    Returns:
        Dictionary with statistics
    """
    col = f"{segment}_edit_distance"

    # Filter out "NMF" (No Match Found) and convert to integers
    # Cast string -> float -> round -> int
    ed_col = (
        pl.col(col)
        .cast(pl.Float32, strict=False)
        .round(0)
        .cast(pl.Int32, strict=False)
    )

    # Filter to only valid values (exclude NMF which becomes null after cast)
    ed_valid = ed_col.filter((pl.col(col) != "NMF") & ed_col.is_not_null())

    # ONT-aware thresholds
    t2 = calculate_ont_threshold(segment_length, 0.02)
    t5 = calculate_ont_threshold(segment_length, ont_error_rate)
    t10 = calculate_ont_threshold(segment_length, 0.10)

    # Aggregation expressions
    agg_exprs = [
        pl.len().alias("total_reads"),           # total rows
        ed_valid.count().alias("reads_with_segment"),

        # Basic statistics
        ed_valid.mean().alias("mean"),
        ed_valid.median().alias("median"),
        ed_valid.std().alias("std"),
        ed_valid.min().alias("min"),
        ed_valid.max().alias("max"),

        # Percentiles
        ed_valid.quantile(0.25, "nearest").alias("p25"),
        ed_valid.quantile(0.75, "nearest").alias("p75"),
        ed_valid.quantile(0.95, "nearest").alias("p95"),
        ed_valid.quantile(0.99, "nearest").alias("p99"),

        # Quality counts
        (ed_valid == 0).sum().alias("perfect_matches"),
        (ed_valid <= 1).sum().alias("ed1_or_less"),
        (ed_valid <= 3).sum().alias("ed3_or_less"),
        (ed_valid <= 5).sum().alias("ed5_or_less"),
        (ed_valid <= t2 if t2 is not None else pl.lit(None)).sum().alias("pass_2pct"),
        (ed_valid <= t5 if t5 is not None else pl.lit(None)).sum().alias("pass_5pct"),
        (ed_valid <= t10 if t10 is not None else pl.lit(None)).sum().alias("pass_10pct"),

        # NMF (No Match Found) counts
        (pl.col(col) == "NMF").sum().alias("nmf_count"),
    ]

    # Compute aggregation
    result = lf.select(agg_exprs).collect().to_dicts()[0]

    total = result["total_reads"]
    detected = result["reads_with_segment"]

    stats = {
        "segment": segment,
        "median_length": segment_length,
        "total_reads": total,
        "reads_with_segment": detected,
        "reads_missing_segment": total - detected,
        "percent_detected": 100 * detected / total if total > 0 else None,
        "nmf_count": result["nmf_count"],
        "percent_nmf": 100 * result["nmf_count"] / total if total > 0 else None,

        # Basic stats
        "mean": result["mean"],
        "median": result["median"],
        "std": result["std"],
        "min": result["min"],
        "max": result["max"],

        # Percentiles
        "p25": result["p25"],
        "p75": result["p75"],
        "p95": result["p95"],
        "p99": result["p99"],

        # Quality metrics
        "perfect_matches": result["perfect_matches"],
        "percent_perfect": 100 * result["perfect_matches"] / detected if detected > 0 else None,
        "edit_dist_1_or_less": result["ed1_or_less"],
        "percent_ed1_or_less": 100 * result["ed1_or_less"] / detected if detected > 0 else None,
        "edit_dist_3_or_less": result["ed3_or_less"],
        "percent_ed3_or_less": 100 * result["ed3_or_less"] / detected if detected > 0 else None,
        "edit_dist_5_or_less": result["ed5_or_less"],
        "percent_ed5_or_less": 100 * result["ed5_or_less"] / detected if detected > 0 else None,

        # ONT thresholds
        "ont_threshold_2pct": t2,
        "ont_threshold_5pct": t5,
        "ont_threshold_10pct": t10,
        "percent_passing_2pct": 100 * result["pass_2pct"] / detected if detected > 0 and result["pass_2pct"] is not None else None,
        "percent_passing_5pct": 100 * result["pass_5pct"] / detected if detected > 0 and result["pass_5pct"] is not None else None,
        "percent_passing_10pct": 100 * result["pass_10pct"] / detected if detected > 0 and result["pass_10pct"] is not None else None,
    }

    return stats

# Visualization

def plot_edit_distance_distributions(lf, segments, segment_lengths, ont_error_rate, output_pdf):
    """Generate comprehensive visualizations of edit distance distributions."""
    print(f"\nCreating visualizations in {output_pdf}...")

    # For plotting, we need to collect the data (can't plot lazily)
    # But we only collect the edit distance columns to minimize memory
    edit_cols = [f"{seg}_edit_distance" for seg in segments]
    # Filter "NMF" and cast to Int32
    df_plot = lf.select([
        pl.when(pl.col(col) != "NMF")
        .then(pl.col(col).cast(pl.Float32, strict=False).round(0).cast(pl.Int32))
        .otherwise(None)
        .alias(col)
        for col in edit_cols
    ]).collect()

    with PdfPages(output_pdf) as pdf:
        # 1. Individual histograms for each segment
        for segment in segments:
            edit_dist_col = f"{segment}_edit_distance"
            if edit_dist_col not in df_plot.columns:
                continue

            values = df_plot[edit_dist_col].drop_nulls().to_numpy()
            if len(values) == 0:
                continue

            segment_length = segment_lengths.get(segment)
            t2 = calculate_ont_threshold(segment_length, 0.02)
            t5 = calculate_ont_threshold(segment_length, ont_error_rate)
            t10 = calculate_ont_threshold(segment_length, 0.10)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Histogram
            ax1.hist(values, bins=min(50, int(np.ceil(values.max())) + 1), edgecolor='black', alpha=0.7)
            ax1.axvline(np.median(values), color='red', linestyle='--',
                       label=f'Median: {np.median(values):.1f}')

            if t2 is not None:
                ax1.axvline(t2, color='green', linestyle='--',
                           linewidth=2, label=f'ONT 2% error: {t2}')
            if t5 is not None:
                ax1.axvline(t5, color='orange', linestyle='--',
                           linewidth=2, label=f'ONT {int(ont_error_rate*100)}% error: {t5}')
            if t10 is not None:
                ax1.axvline(t10, color='purple', linestyle='--',
                           linewidth=2, label=f'ONT 10% error: {t10}')

            ax1.set_xlabel('Edit Distance')
            ax1.set_ylabel('Frequency')
            title = f'{segment} Edit Distance Distribution'
            if segment_length:
                title += f'\n(median length: {segment_length:.0f} bp)'
            ax1.set_title(title)
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)

            # Cumulative distribution
            sorted_vals = np.sort(values)
            cumulative = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            ax2.plot(sorted_vals, cumulative, linewidth=2)

            if t2 is not None:
                pct_passing = 100 * (values <= t2).sum() / len(values)
                ax2.axvline(t2, color='green', linestyle='--', linewidth=2, alpha=0.7,
                           label=f'ONT 2%: {t2} ({pct_passing:.1f}% pass)')
            if t5 is not None:
                pct_passing = 100 * (values <= t5).sum() / len(values)
                ax2.axvline(t5, color='orange', linestyle='--', linewidth=2, alpha=0.7,
                           label=f'ONT {int(ont_error_rate*100)}%: {t5} ({pct_passing:.1f}% pass)')
            if t10 is not None:
                pct_passing = 100 * (values <= t10).sum() / len(values)
                ax2.axvline(t10, color='purple', linestyle='--', linewidth=2, alpha=0.7,
                           label=f'ONT 10%: {t10} ({pct_passing:.1f}% pass)')

            ax2.set_xlabel('Edit Distance')
            ax2.set_ylabel('Cumulative Fraction')
            ax2.set_title(f'{segment} Cumulative Distribution')
            ax2.legend(loc='lower right')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # 2. Comparison box plot
        fig, ax = plt.subplots(figsize=(14, 6))

        plot_data = []
        plot_labels = []
        ont_thresholds_2pct = []
        ont_thresholds_5pct = []
        ont_thresholds_10pct = []

        for segment in segments:
            edit_dist_col = f"{segment}_edit_distance"
            if edit_dist_col in df_plot.columns:
                values = df_plot[edit_dist_col].drop_nulls().to_numpy()
                if len(values) > 0:
                    plot_data.append(values)
                    plot_labels.append(segment)

                    segment_length = segment_lengths.get(segment)
                    ont_thresholds_2pct.append(calculate_ont_threshold(segment_length, 0.02))
                    ont_thresholds_5pct.append(calculate_ont_threshold(segment_length, ont_error_rate))
                    ont_thresholds_10pct.append(calculate_ont_threshold(segment_length, 0.10))

        if plot_data:
            bp = ax.boxplot(plot_data, tick_labels=plot_labels, patch_artist=True, showfliers=False)

            # Color code
            for i, (patch, label) in enumerate(zip(bp['boxes'], plot_labels)):
                if label in ['CBC', 'i7', 'i5']:
                    patch.set_facecolor('lightblue')
                else:
                    patch.set_facecolor('lightcoral')

            # Plot thresholds
            x_positions = range(1, len(plot_labels) + 1)
            for x, t2, t5, t10 in zip(x_positions, ont_thresholds_2pct, ont_thresholds_5pct, ont_thresholds_10pct):
                if t2 is not None:
                    ax.plot([x-0.3, x+0.3], [t2, t2], 'g-', linewidth=2, alpha=0.7)
                if t5 is not None:
                    ax.plot([x-0.3, x+0.3], [t5, t5], 'orange', linewidth=2, alpha=0.7)
                if t10 is not None:
                    ax.plot([x-0.3, x+0.3], [t10, t10], 'purple', linewidth=2, alpha=0.7)

            ax.set_ylabel('Edit Distance')
            ax.set_title('Edit Distance Comparison Across Segments')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')

            legend_elements = [
                Patch(facecolor='lightblue', label='Barcode segments'),
                Patch(facecolor='lightcoral', label='Fixed segments'),
                Line2D([0], [0], color='green', linewidth=2, label='ONT 2% threshold'),
                Line2D([0], [0], color='orange', linewidth=2, label=f'ONT {int(ont_error_rate*100)}% threshold'),
                Line2D([0], [0], color='purple', linewidth=2, label='ONT 10% threshold')
            ]
            ax.legend(handles=legend_elements, loc='upper right')

            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # 3. Error rate distribution
        fig, ax = plt.subplots(figsize=(12, 6))

        error_rate_data = []
        error_rate_labels = []

        for segment in segments:
            edit_dist_col = f"{segment}_edit_distance"
            segment_length = segment_lengths.get(segment)

            if edit_dist_col in df_plot.columns and segment_length and segment_length > 0:
                values = df_plot[edit_dist_col].drop_nulls().to_numpy()
                if len(values) > 0:
                    error_rates = (values / segment_length) * 100
                    error_rate_data.append(error_rates)
                    error_rate_labels.append(f'{segment}\n({segment_length:.0f}bp)')

        if error_rate_data:
            bp = ax.boxplot(error_rate_data, tick_labels=error_rate_labels, patch_artist=True, showfliers=False)

            for i, (patch, label) in enumerate(zip(bp['boxes'], error_rate_labels)):
                segment_name = label.split('\n')[0]
                if segment_name in ['CBC', 'i7', 'i5']:
                    patch.set_facecolor('lightblue')
                else:
                    patch.set_facecolor('lightcoral')

            ax.axhline(2, color='green', linestyle='--', linewidth=2, alpha=0.7, label='2% error rate')
            ax.axhline(ont_error_rate * 100, color='orange', linestyle='--', linewidth=2, alpha=0.7,
                      label=f'{int(ont_error_rate*100)}% error rate')
            ax.axhline(10, color='purple', linestyle='--', linewidth=2, alpha=0.7, label='10% error rate')

            ax.set_ylabel('Error Rate (%)')
            ax.set_title('Error Rate Distribution Across Segments')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='upper right')

            plt.tight_layout()
            pdf.savefig()
            plt.close()

# Filtering recommendations

def generate_filtering_recommendations(stats_df, ont_error_rate):
    """Generate filtering code recommendations."""
    print("\n" + "="*80)
    print("FILTERING RECOMMENDATIONS")
    print("="*80)

    for row in stats_df.iter_rows(named=True):
        segment = row['segment']
        print(f"\n{segment} (median length: {row['median_length']:.0f} bp):")

        # Handle cases where statistics might be None
        if row['percent_perfect'] is not None:
            print(f"  Perfect matches: {row['percent_perfect']:.1f}%")
        else:
            print(f"  Perfect matches: N/A (no segments detected)")

        if row['median'] is not None:
            print(f"  Median edit distance: {row['median']:.1f}")
        else:
            print(f"  Median edit distance: N/A (no segments detected)")

        print(f"\n  ONT-based thresholds:")

        if row['ont_threshold_2pct'] is not None:
            print(f"    - Conservative (2%):      <= {row['ont_threshold_2pct']} ({row['percent_passing_2pct']:.1f}% pass)")
        if row['ont_threshold_5pct'] is not None:
            print(f"    - Permissive ({int(ont_error_rate*100)}%):       <= {row['ont_threshold_5pct']} ({row['percent_passing_5pct']:.1f}% pass)")
        if row['ont_threshold_10pct'] is not None:
            print(f"    - Very permissive (10%):  <= {row['ont_threshold_10pct']} ({row['percent_passing_10pct']:.1f}% pass)")

    print("\n" + "="*80)
    print("RECOMMENDED FILTERING CODE")
    print("="*80)
    print("\nimport polars as pl")
    print("lf = pl.scan_parquet('annotations_valid.parquet')")

    # --- Conservative (2% error rate) ---
    print("\n# Conservative (2% error rate)")
    print("filtered_conservative = lf.filter(")
    conservative_filters = []
    for row in stats_df.iter_rows(named=True):
        if row['ont_threshold_2pct'] is not None:
            conservative_filters.append(f"    (pl.col('{row['segment']}_edit_distance') <= {row['ont_threshold_2pct']})")
    print(" &\n".join(conservative_filters))
    print(").collect()")

    # --- Permissive (5% error rate) ---
    print("\n# Permissive (5% error rate)")
    print("filtered_permissive = lf.filter(")
    permissive_filters = []
    for row in stats_df.iter_rows(named=True):
        if row['ont_threshold_5pct'] is not None:
            permissive_filters.append(f"    (pl.col('{row['segment']}_edit_distance') <= {row['ont_threshold_5pct']})")
    print(" &\n".join(permissive_filters))
    print(").collect()")


############################################################
# Valid vs Invalid Comparison
############################################################

def analyze_invalid_by_reason(lf_invalid, segments, segment_lengths, ont_error_rate, max_reasons=20):
    """
    Analyze invalid reads broken down by failure reason.

    Args:
        lf_invalid: LazyFrame with invalid reads
        segments: List of segment names
        segment_lengths: Dictionary of segment lengths
        ont_error_rate: ONT error rate threshold
        max_reasons: Maximum number of top reasons to analyze (default: 20)

    Returns DataFrame with statistics per reason category.
    """
    print("\nAnalyzing invalid reads by failure reason...")

    # Get reason counts first to identify top N
    reason_counts = (
        lf_invalid.group_by("reason")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .collect()
    )

    total_unique = len(reason_counts)
    print(f"  Found {total_unique:,} unique failure reasons")

    # Limit to top N reasons
    top_reasons_df = reason_counts.head(max_reasons)
    top_reasons = top_reasons_df["reason"].to_list()
    top_counts = dict(zip(top_reasons_df["reason"].to_list(), top_reasons_df["count"].to_list()))

    print(f"  Analyzing top {len(top_reasons)} of {total_unique:,} failure reasons")

    all_reason_stats = []

    for reason in top_reasons:
        # Filter to this reason
        lf_reason = lf_invalid.filter(pl.col("reason") == reason)

        # Get count from our pre-computed dictionary
        count = top_counts[reason]

        # Compute stats for each segment
        for segment in segments:
            stats = compute_segment_statistics(
                lf_reason, segment, segment_lengths.get(segment), ont_error_rate
            )
            stats["reason"] = reason
            stats["reason_count"] = count
            all_reason_stats.append(stats)

    return pl.DataFrame(all_reason_stats)


def plot_valid_vs_invalid_comparison(lf_valid, lf_invalid, segments, segment_lengths, output_pdf):
    """
    Create violin plots comparing edit distance distributions between valid and invalid reads.
    """
    print(f"\nCreating valid vs invalid comparison plots in {output_pdf}...")

    with PdfPages(output_pdf) as pdf:
        for segment in segments:
            edit_col = f"{segment}_edit_distance"
            segment_length = segment_lengths.get(segment)

            # Collect data for both datasets
            valid_data = lf_valid.select([
                pl.when(pl.col(edit_col) != "NMF")
                .then(pl.col(edit_col).cast(pl.Float32, strict=False).round(0).cast(pl.Int32))
                .otherwise(None)
                .alias(edit_col)
            ]).collect()

            invalid_data = lf_invalid.select([
                pl.when(pl.col(edit_col) != "NMF")
                .then(pl.col(edit_col).cast(pl.Float32, strict=False).round(0).cast(pl.Int32))
                .otherwise(None)
                .alias(edit_col)
            ]).collect()

            valid_values = valid_data[edit_col].drop_nulls().to_numpy()
            invalid_values = invalid_data[edit_col].drop_nulls().to_numpy()

            if len(valid_values) == 0 and len(invalid_values) == 0:
                continue

            # Create comparison plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Violin plot
            plot_data = []
            labels = []
            if len(valid_values) > 0:
                plot_data.append(valid_values)
                labels.append('Valid')
            if len(invalid_values) > 0:
                plot_data.append(invalid_values)
                labels.append('Invalid')

            if plot_data:
                parts = ax1.violinplot(plot_data, positions=range(len(plot_data)),
                                      showmeans=True, showmedians=True)
                ax1.set_xticks(range(len(labels)))
                ax1.set_xticklabels(labels)
                ax1.set_ylabel('Edit Distance')
                title = f'{segment} - Valid vs Invalid'
                if segment_length:
                    title += f'\n(median length: {segment_length:.0f} bp)'
                ax1.set_title(title)
                ax1.grid(True, alpha=0.3, axis='y')

                # Add sample sizes
                for i, (data, label) in enumerate(zip(plot_data, labels)):
                    ax1.text(i, ax1.get_ylim()[1]*0.95, f'n={len(data):,}',
                            ha='center', va='top', fontsize=9)

            # Cumulative distribution comparison
            if len(valid_values) > 0:
                sorted_valid = np.sort(valid_values)
                cumulative_valid = np.arange(1, len(sorted_valid) + 1) / len(sorted_valid)
                ax2.plot(sorted_valid, cumulative_valid, linewidth=2,
                        label=f'Valid (n={len(valid_values):,})', color='blue')

            if len(invalid_values) > 0:
                sorted_invalid = np.sort(invalid_values)
                cumulative_invalid = np.arange(1, len(sorted_invalid) + 1) / len(sorted_invalid)
                ax2.plot(sorted_invalid, cumulative_invalid, linewidth=2,
                        label=f'Invalid (n={len(invalid_values):,})', color='red', alpha=0.7)

            ax2.set_xlabel('Edit Distance')
            ax2.set_ylabel('Cumulative Fraction')
            ax2.set_title(f'{segment} - Cumulative Distribution')
            ax2.legend(loc='lower right')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            pdf.savefig()
            plt.close()


def plot_invalid_reason_breakdown(reason_stats_df, segments, output_pdf):
    """
    Plot edit distance statistics broken down by invalid reason.
    """
    print(f"\nCreating invalid reason breakdown plots in {output_pdf}...")

    with PdfPages(output_pdf) as pdf:
        # Get top reasons by read count
        reason_counts = (
            reason_stats_df
            .group_by("reason")
            .agg(pl.col("reason_count").first())
            .sort("reason_count", descending=True)
        )

        top_reasons = reason_counts["reason"].to_list()[:10]  # Top 10 reasons

        for segment in segments:
            seg_data = reason_stats_df.filter(pl.col("segment") == segment)

            if len(seg_data) == 0:
                continue

            # Filter to top reasons
            seg_data = seg_data.filter(pl.col("reason").is_in(top_reasons))

            if len(seg_data) == 0:
                continue

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Mean edit distance by reason
            reasons = seg_data["reason"].to_list()
            means = seg_data["mean"].to_list()
            counts = seg_data["reason_count"].to_list()

            # Filter out None values and sort by mean edit distance
            valid_data = [(r, m, c) for r, m, c in zip(reasons, means, counts) if m is not None]
            sorted_data = sorted(valid_data, key=lambda x: x[1])
            reasons_sorted, means_sorted, counts_sorted = zip(*sorted_data) if sorted_data else ([], [], [])

            if reasons_sorted:
                y_pos = np.arange(len(reasons_sorted))
                bars = ax1.barh(y_pos, means_sorted, color='skyblue', edgecolor='black')
                ax1.set_yticks(y_pos)
                ax1.set_yticklabels([f"{r[:40]}..." if len(r) > 40 else r for r in reasons_sorted],
                                   fontsize=8)
                ax1.set_xlabel('Mean Edit Distance')
                ax1.set_title(f'{segment} - Mean Edit Distance by Failure Reason')
                ax1.grid(True, alpha=0.3, axis='x')

                # Add read counts as text
                for i, (bar, count) in enumerate(zip(bars, counts_sorted)):
                    width = bar.get_width()
                    ax1.text(width, bar.get_y() + bar.get_height()/2,
                            f' n={count:,}', va='center', fontsize=7)

            # Perfect match rate by reason
            perfect_pcts = seg_data["percent_perfect"].to_list()
            # Filter out None values and sort by perfect match rate (descending)
            valid_data2 = [(r, p, c) for r, p, c in zip(reasons, perfect_pcts, counts) if p is not None]
            sorted_data2 = sorted(valid_data2, key=lambda x: x[1], reverse=True)
            reasons_sorted2, perfect_sorted, counts_sorted2 = zip(*sorted_data2) if sorted_data2 else ([], [], [])

            if reasons_sorted2:
                y_pos = np.arange(len(reasons_sorted2))
                bars = ax2.barh(y_pos, perfect_sorted, color='lightgreen', edgecolor='black')
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels([f"{r[:40]}..." if len(r) > 40 else r for r in reasons_sorted2],
                                   fontsize=8)
                ax2.set_xlabel('Perfect Match Rate (%)')
                ax2.set_title(f'{segment} - Perfect Matches by Failure Reason')
                ax2.grid(True, alpha=0.3, axis='x')

                # Add read counts
                for i, (bar, count) in enumerate(zip(bars, counts_sorted2)):
                    width = bar.get_width()
                    ax2.text(width, bar.get_y() + bar.get_height()/2,
                            f' n={count:,}', va='center', fontsize=7)

            plt.tight_layout()
            pdf.savefig()
            plt.close()


def compare_detection_rates(stats_valid, stats_invalid):
    """
    Compare segment detection rates (NMF rates) between valid and invalid reads.
    """
    print("\n" + "="*80)
    print("SEGMENT DETECTION RATE COMPARISON")
    print("="*80)

    comparison = []

    for seg_valid in stats_valid:
        segment = seg_valid['segment']

        # Find matching invalid segment
        seg_invalid = next((s for s in stats_invalid if s['segment'] == segment), None)

        if seg_invalid:
            comparison.append({
                'segment': segment,
                'valid_nmf_pct': seg_valid['percent_nmf'],
                'invalid_nmf_pct': seg_invalid['percent_nmf'],
                'nmf_diff': seg_invalid['percent_nmf'] - seg_valid['percent_nmf'],
                'valid_mean_ed': seg_valid['mean'],
                'invalid_mean_ed': seg_invalid['mean'],
                'ed_diff': (seg_invalid['mean'] - seg_valid['mean']) if seg_invalid['mean'] and seg_valid['mean'] else None
            })

    comp_df = pl.DataFrame(comparison)
    print(comp_df)

    return comp_df


############################################################
# Quality Score Analysis
############################################################

def plot_score_distributions(lf, segments, output_pdf):
    """
    Plot quality score distributions for each segment and overall read quality.
    """
    print(f"\nCreating quality score distribution plots in {output_pdf}...")

    # Collect score columns
    score_cols = [f"{seg}_score" for seg in segments] + ["read_quality_score"]
    existing_cols = [c for c in score_cols if c in lf.collect_schema()]

    if not existing_cols:
        print("  No score columns found - skipping score visualization")
        return

    df_scores = lf.select(existing_cols).collect()

    with PdfPages(output_pdf) as pdf:
        # 1. Individual segment score distributions
        for seg in segments:
            score_col = f"{seg}_score"
            if score_col not in df_scores.columns:
                continue

            scores = df_scores[score_col].drop_nulls().to_numpy()
            if len(scores) == 0:
                continue

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Histogram
            ax1.hist(scores, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
            ax1.axvline(np.median(scores), color='red', linestyle='--',
                       linewidth=2, label=f'Median: {np.median(scores):.3f}')
            ax1.axvline(np.mean(scores), color='orange', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(scores):.3f}')
            ax1.set_xlabel('Quality Score (1.0 = perfect)')
            ax1.set_ylabel('Frequency')
            ax1.set_title(f'{seg} - Quality Score Distribution\n(n={len(scores):,} non-concatenated reads)')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)

            # Cumulative distribution
            sorted_scores = np.sort(scores)
            cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
            ax2.plot(sorted_scores, cumulative, linewidth=2, color='skyblue')
            ax2.axvline(np.median(scores), color='red', linestyle='--',
                       linewidth=2, alpha=0.7, label=f'Median: {np.median(scores):.3f}')
            ax2.axhline(0.95, color='green', linestyle='--', linewidth=1, alpha=0.5,
                       label='95th percentile')
            ax2.set_xlabel('Quality Score')
            ax2.set_ylabel('Cumulative Fraction')
            ax2.set_title(f'{seg} - Cumulative Score Distribution')
            ax2.legend(loc='lower right')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # 2. Overall read quality score
        if "read_quality_score" in df_scores.columns:
            read_scores = df_scores["read_quality_score"].drop_nulls().to_numpy()

            if len(read_scores) > 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

                # Histogram
                ax1.hist(read_scores, bins=50, edgecolor='black', alpha=0.7, color='coral')
                ax1.axvline(np.median(read_scores), color='red', linestyle='--',
                           linewidth=2, label=f'Median: {np.median(read_scores):.3f}')
                ax1.axvline(np.mean(read_scores), color='orange', linestyle='--',
                           linewidth=2, label=f'Mean: {np.mean(read_scores):.3f}')
                ax1.set_xlabel('Read Quality Score (1.0 = perfect)')
                ax1.set_ylabel('Frequency')
                ax1.set_title(f'Overall Read Quality Distribution\n(n={len(read_scores):,} reads)')
                ax1.legend(loc='upper left')
                ax1.grid(True, alpha=0.3)

                # Cumulative distribution
                sorted_scores = np.sort(read_scores)
                cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
                ax2.plot(sorted_scores, cumulative, linewidth=2, color='coral')
                ax2.axvline(np.median(read_scores), color='red', linestyle='--',
                           linewidth=2, alpha=0.7, label=f'Median: {np.median(read_scores):.3f}')
                ax2.axhline(0.95, color='green', linestyle='--', linewidth=1, alpha=0.5,
                           label='95th percentile')
                ax2.set_xlabel('Read Quality Score')
                ax2.set_ylabel('Cumulative Fraction')
                ax2.set_title('Cumulative Read Quality Distribution')
                ax2.legend(loc='lower right')
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                pdf.savefig()
                plt.close()


def plot_score_comparison(lf_valid, lf_invalid, segments, output_pdf):
    """
    Compare quality score distributions between valid and invalid reads.
    """
    print(f"\nCreating score comparison plots in {output_pdf}...")

    with PdfPages(output_pdf) as pdf:
        # 1. Compare overall read quality scores
        if "read_quality_score" in lf_valid.collect_schema():
            valid_scores = lf_valid.select("read_quality_score").collect()["read_quality_score"].drop_nulls().to_numpy()
            invalid_scores = lf_invalid.select("read_quality_score").collect()["read_quality_score"].drop_nulls().to_numpy()

            if len(valid_scores) > 0 or len(invalid_scores) > 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

                # Violin plot
                plot_data = []
                labels = []
                if len(valid_scores) > 0:
                    plot_data.append(valid_scores)
                    labels.append('Valid')
                if len(invalid_scores) > 0:
                    plot_data.append(invalid_scores)
                    labels.append('Invalid')

                if plot_data:
                    parts = ax1.violinplot(plot_data, positions=range(len(plot_data)),
                                          showmeans=True, showmedians=True)
                    ax1.set_xticks(range(len(labels)))
                    ax1.set_xticklabels(labels)
                    ax1.set_ylabel('Read Quality Score')
                    ax1.set_title('Read Quality Score - Valid vs Invalid')
                    ax1.grid(True, alpha=0.3, axis='y')

                    # Add sample sizes and medians
                    for i, (data, label) in enumerate(zip(plot_data, labels)):
                        ax1.text(i, ax1.get_ylim()[1]*0.95,
                                f'n={len(data):,}\nmedian={np.median(data):.3f}',
                                ha='center', va='top', fontsize=9)

                # Cumulative distributions
                if len(valid_scores) > 0:
                    sorted_valid = np.sort(valid_scores)
                    cumulative_valid = np.arange(1, len(sorted_valid) + 1) / len(sorted_valid)
                    ax2.plot(sorted_valid, cumulative_valid, linewidth=2,
                            label=f'Valid (n={len(valid_scores):,})', color='blue')

                if len(invalid_scores) > 0:
                    sorted_invalid = np.sort(invalid_scores)
                    cumulative_invalid = np.arange(1, len(sorted_invalid) + 1) / len(sorted_invalid)
                    ax2.plot(sorted_invalid, cumulative_invalid, linewidth=2,
                            label=f'Invalid (n={len(invalid_scores):,})', color='red', alpha=0.7)

                ax2.set_xlabel('Read Quality Score')
                ax2.set_ylabel('Cumulative Fraction')
                ax2.set_title('Cumulative Score Distribution')
                ax2.legend(loc='lower right')
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                pdf.savefig()
                plt.close()

        # 2. Per-segment score comparisons
        for seg in segments:
            score_col = f"{seg}_score"

            if score_col not in lf_valid.collect_schema():
                continue

            valid_seg_scores = lf_valid.select(score_col).collect()[score_col].drop_nulls().to_numpy()
            invalid_seg_scores = lf_invalid.select(score_col).collect()[score_col].drop_nulls().to_numpy()

            if len(valid_seg_scores) == 0 and len(invalid_seg_scores) == 0:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))

            # Cumulative distributions
            if len(valid_seg_scores) > 0:
                sorted_valid = np.sort(valid_seg_scores)
                cumulative_valid = np.arange(1, len(sorted_valid) + 1) / len(sorted_valid)
                ax.plot(sorted_valid, cumulative_valid, linewidth=2,
                       label=f'Valid (n={len(valid_seg_scores):,}, median={np.median(valid_seg_scores):.3f})',
                       color='blue')

            if len(invalid_seg_scores) > 0:
                sorted_invalid = np.sort(invalid_seg_scores)
                cumulative_invalid = np.arange(1, len(sorted_invalid) + 1) / len(sorted_invalid)
                ax.plot(sorted_invalid, cumulative_invalid, linewidth=2,
                       label=f'Invalid (n={len(invalid_seg_scores):,}, median={np.median(invalid_seg_scores):.3f})',
                       color='red', alpha=0.7)

            ax.set_xlabel(f'{seg} Quality Score')
            ax.set_ylabel('Cumulative Fraction')
            ax.set_title(f'{seg} - Score Comparison (Valid vs Invalid)')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            pdf.savefig()
            plt.close()


def export_ranked_reads(lf, output_file, n_top=1000, n_bottom=1000):
    """
    Export top and bottom ranked reads by quality score.

    Args:
        lf: LazyFrame with read_quality_score column
        output_file: Path to output TSV file
        n_top: Number of top-scoring reads to export
        n_bottom: Number of bottom-scoring reads to export
    """
    print(f"\nExporting ranked reads to {output_file}...")

    if "read_quality_score" not in lf.collect_schema():
        print("  No read_quality_score column found - skipping ranked export")
        return

    # Get top and bottom reads
    top_reads = (
        lf.filter(pl.col("read_quality_score").is_not_null())
        .sort("read_quality_score", descending=True)
        .head(n_top)
        .collect()
    )

    bottom_reads = (
        lf.filter(pl.col("read_quality_score").is_not_null())
        .sort("read_quality_score", descending=False)
        .head(n_bottom)
        .collect()
    )

    # Add rank column
    top_reads = top_reads.with_columns([
        pl.lit("top").alias("rank_category"),
        (pl.arange(1, len(top_reads) + 1)).alias("rank")
    ])

    bottom_reads = bottom_reads.with_columns([
        pl.lit("bottom").alias("rank_category"),
        (pl.arange(1, len(bottom_reads) + 1)).alias("rank")
    ])

    # Combine
    ranked_reads = pl.concat([top_reads, bottom_reads])

    # Save
    ranked_reads.write_csv(output_file, separator="\t")
    print(f"  Exported {len(top_reads):,} top reads and {len(bottom_reads):,} bottom reads")


############################################################
# Anomalous Read Detection and Export
############################################################

def export_anomalous_reads(lf, anomaly_thresholds, output_file):
    """
    Export reads with anomalously short segments to TSV file.

    Args:
        lf: LazyFrame with segment length columns
        anomaly_thresholds: Dictionary of minimum expected lengths per segment
        output_file: Path to output TSV file
    """
    if not anomaly_thresholds:
        print("  No anomaly thresholds specified - skipping anomalous read export")
        return

    print(f"\nExporting reads with anomalous segment lengths to {output_file}...")

    # Build filter conditions for any segment below threshold
    conditions = []
    for seg, min_length in anomaly_thresholds.items():
        length_col = f"{seg}_length"
        if length_col in lf.collect_schema():
            # Cast to Int64 to ensure numeric comparison
            conditions.append(
                (pl.col(length_col).cast(pl.Int64, strict=False).is_not_null()) &
                (pl.col(length_col).cast(pl.Int64, strict=False) < min_length)
            )

    if not conditions:
        print("  No matching segment columns found")
        return

    # Combine conditions with OR (anomalous in ANY segment)
    combined_filter = conditions[0]
    for condition in conditions[1:]:
        combined_filter = combined_filter | condition

    # Filter and collect
    anomalous_reads = lf.filter(combined_filter).collect()

    if len(anomalous_reads) == 0:
        print("  No anomalous reads found")
        return

    # Save
    anomalous_reads.write_csv(output_file, separator="\t")
    print(f"  Exported {len(anomalous_reads):,} reads with anomalous segment lengths")

    # Print breakdown by segment
    for seg, min_length in anomaly_thresholds.items():
        length_col = f"{seg}_length"
        if length_col in anomalous_reads.columns:
            count = len(anomalous_reads.filter(
                (pl.col(length_col).cast(pl.Int64, strict=False).is_not_null()) &
                (pl.col(length_col).cast(pl.Int64, strict=False) < min_length)
            ))
            if count > 0:
                print(f"    {seg} <{min_length}bp: {count:,} reads")


def print_anomaly_summary(length_stats_df):
    """
    Print summary of anomalous segment lengths.
    """
    # Filter to segments with anomaly detection enabled
    anomalous_segments = length_stats_df.filter(
        pl.col("min_expected_length").is_not_null()
    )

    if len(anomalous_segments) == 0:
        return

    print("\n" + "="*80)
    print("SEGMENT LENGTH ANOMALY DETECTION")
    print("="*80)
    print(f"{'Segment':<15} {'Min Expected':<15} {'Anomalous Reads':<20} {'Percentage':<15} {'Min Observed':<15}")
    print("-"*80)

    for row in anomalous_segments.iter_rows(named=True):
        segment = row['segment']
        min_exp = row['min_expected_length']
        anom_count = row['anomaly_count']
        anom_pct = row['anomaly_percent']
        min_obs = row['min_length']

        print(f"{segment:<15} {min_exp:<15.0f} {anom_count:<20,} {anom_pct:<14.2f}% {min_obs:<15.0f}")

    print("\nInterpretation:")
    print("  - 'Min Expected' is the threshold below which segments are considered anomalous")
    print("  - 'Anomalous Reads' are reads with segment lengths below the threshold")
    print("  - 'Min Observed' shows the actual minimum length found in the data")
    print("  - High anomaly percentages in 'valid' reads may indicate model errors")


############################################################
# Segment Length Distribution Analysis
############################################################

def plot_segment_length_distributions(lf, segments, median_lengths, output_pdf):
    """
    Plot segment length distributions with outlier boundaries.
    """
    print(f"\nCreating segment length distribution plots in {output_pdf}...")

    # Collect length columns
    length_cols = [f"{seg}_length" for seg in segments]
    existing_cols = [c for c in length_cols if c in lf.collect_schema()]

    if not existing_cols:
        print("  No length columns found - skipping length visualization")
        return

    df_lengths = lf.select(existing_cols).collect()

    with PdfPages(output_pdf) as pdf:
        for seg in segments:
            length_col = f"{seg}_length"
            if length_col not in df_lengths.columns:
                continue

            lengths = df_lengths[length_col].drop_nulls().to_numpy()
            if len(lengths) == 0:
                continue

            median_len = median_lengths.get(seg)
            std_len = np.std(lengths)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Histogram
            ax1.hist(lengths, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
            ax1.axvline(np.median(lengths), color='red', linestyle='--',
                       linewidth=2, label=f'Median: {np.median(lengths):.1f}bp')
            ax1.axvline(np.mean(lengths), color='orange', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(lengths):.1f}bp')

            # Outlier boundaries (±2 std)
            if median_len and std_len:
                lower_bound = median_len - 2 * std_len
                upper_bound = median_len + 2 * std_len
                ax1.axvline(lower_bound, color='purple', linestyle=':', linewidth=2,
                           alpha=0.7, label=f'±2σ bounds')
                ax1.axvline(upper_bound, color='purple', linestyle=':', linewidth=2, alpha=0.7)

            ax1.set_xlabel('Segment Length (bp)')
            ax1.set_ylabel('Frequency')
            ax1.set_title(f'{seg} - Length Distribution\n(n={len(lengths):,} non-concatenated reads)')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)

            # Cumulative distribution
            sorted_lengths = np.sort(lengths)
            cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
            ax2.plot(sorted_lengths, cumulative, linewidth=2, color='steelblue')
            ax2.axvline(np.median(lengths), color='red', linestyle='--',
                       linewidth=2, alpha=0.7, label=f'Median: {np.median(lengths):.1f}bp')
            ax2.axhline(0.95, color='green', linestyle='--', linewidth=1, alpha=0.5,
                       label='95th percentile')

            ax2.set_xlabel('Segment Length (bp)')
            ax2.set_ylabel('Cumulative Fraction')
            ax2.set_title(f'{seg} - Cumulative Length Distribution')
            ax2.legend(loc='lower right')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            pdf.savefig()
            plt.close()


def plot_length_comparison(lf_valid, lf_invalid, segments, output_pdf):
    """
    Compare segment length distributions between valid and invalid reads.
    """
    print(f"\nCreating length comparison plots in {output_pdf}...")

    with PdfPages(output_pdf) as pdf:
        for seg in segments:
            length_col = f"{seg}_length"

            if length_col not in lf_valid.collect_schema():
                continue

            valid_lengths = lf_valid.select(length_col).collect()[length_col].drop_nulls().to_numpy()
            invalid_lengths = lf_invalid.select(length_col).collect()[length_col].drop_nulls().to_numpy()

            if len(valid_lengths) == 0 and len(invalid_lengths) == 0:
                continue

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Violin plot
            plot_data = []
            labels = []
            if len(valid_lengths) > 0:
                plot_data.append(valid_lengths)
                labels.append('Valid')
            if len(invalid_lengths) > 0:
                plot_data.append(invalid_lengths)
                labels.append('Invalid')

            if plot_data:
                parts = ax1.violinplot(plot_data, positions=range(len(plot_data)),
                                      showmeans=True, showmedians=True)
                ax1.set_xticks(range(len(labels)))
                ax1.set_xticklabels(labels)
                ax1.set_ylabel('Segment Length (bp)')
                ax1.set_title(f'{seg} - Length Distribution (Valid vs Invalid)')
                ax1.grid(True, alpha=0.3, axis='y')

                # Add sample sizes and medians
                for i, (data, label) in enumerate(zip(plot_data, labels)):
                    ax1.text(i, ax1.get_ylim()[1]*0.95,
                            f'n={len(data):,}\nmedian={np.median(data):.1f}bp',
                            ha='center', va='top', fontsize=9)

            # Cumulative distributions
            if len(valid_lengths) > 0:
                sorted_valid = np.sort(valid_lengths)
                cumulative_valid = np.arange(1, len(sorted_valid) + 1) / len(sorted_valid)
                ax2.plot(sorted_valid, cumulative_valid, linewidth=2,
                        label=f'Valid (n={len(valid_lengths):,}, median={np.median(valid_lengths):.1f}bp)',
                        color='blue')

            if len(invalid_lengths) > 0:
                sorted_invalid = np.sort(invalid_lengths)
                cumulative_invalid = np.arange(1, len(sorted_invalid) + 1) / len(sorted_invalid)
                ax2.plot(sorted_invalid, cumulative_invalid, linewidth=2,
                        label=f'Invalid (n={len(invalid_lengths):,}, median={np.median(invalid_lengths):.1f}bp)',
                        color='red', alpha=0.7)

            ax2.set_xlabel('Segment Length (bp)')
            ax2.set_ylabel('Cumulative Fraction')
            ax2.set_title(f'{seg} - Cumulative Length Distribution')
            ax2.legend(loc='lower right')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            pdf.savefig()
            plt.close()


############################################################
# Main
############################################################

def main():
    parser = argparse.ArgumentParser(
        description="Polars-based edit distance analysis (streaming, fast)"
    )
    parser.add_argument("parquet_file", help="Path to valid annotations parquet file")
    parser.add_argument("--invalid_parquet", "-i", help="Path to invalid annotations parquet file (optional)")
    parser.add_argument("--output_dir", "-o", help="Output directory", default=None)
    parser.add_argument("--ont_error_rate", "-e", type=float, default=0.05,
                       help="Expected ONT error rate (default: 0.05)")
    parser.add_argument("--sample_size", type=int, default=10_000,
                       help="Sample size for length estimation (default: 10000)")
    parser.add_argument("--analyze_reasons", action="store_true", default=False,
                       help="Analyze invalid reads by failure reason (can be slow with many unique reasons)")
    parser.add_argument("--max_reasons", type=int, default=20,
                       help="Maximum number of top failure reasons to analyze (default: 20)")

    # Anomaly detection arguments
    parser.add_argument("--min_cdna_length", type=int, default=None,
                       help="Minimum expected cDNA length in bp (default: None, no anomaly detection)")
    parser.add_argument("--min_polya_length", type=int, default=None,
                       help="Minimum expected polyA length in bp (default: None, no anomaly detection)")
    parser.add_argument("--min_umi_length", type=int, default=None,
                       help="Minimum expected UMI length in bp (default: None, no anomaly detection)")
    parser.add_argument("--min_cbc_length", type=int, default=None,
                       help="Minimum expected CBC length in bp (default: None, no anomaly detection)")

    args = parser.parse_args()

    # Build anomaly thresholds dictionary from command-line args
    anomaly_thresholds = {}
    if args.min_cdna_length is not None:
        anomaly_thresholds['cDNA'] = args.min_cdna_length
    if args.min_polya_length is not None:
        anomaly_thresholds['polyA'] = args.min_polya_length
    if args.min_umi_length is not None:
        anomaly_thresholds['UMI'] = args.min_umi_length
    if args.min_cbc_length is not None:
        anomaly_thresholds['CBC'] = args.min_cbc_length

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.parquet_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print anomaly detection configuration
    if anomaly_thresholds:
        print("\nAnomaly Detection Thresholds:")
        for seg, min_len in anomaly_thresholds.items():
            print(f"  {seg}: minimum {min_len}bp")
    else:
        print("\nNo anomaly detection thresholds specified (use --min_cdna_length, --min_polya_length, etc.)")

    # Load data lazily
    print(f"Scanning {args.parquet_file}...")
    lf = pl.scan_parquet(args.parquet_file)

    # Identify segments
    schema = lf.collect_schema()
    edit_dist_cols, barcode_segments, fixed_segments = identify_edit_distance_columns(schema)
    all_segments = barcode_segments + fixed_segments

    if not edit_dist_cols:
        print("ERROR: No edit distance columns found!")
        print("Available columns:", list(schema.keys()))
        sys.exit(1)

    print(f"\nFound {len(all_segments)} segments:")
    print(f"  Barcode segments: {', '.join(barcode_segments)}")
    print(f"  Fixed segments: {', '.join(fixed_segments)}")

    # Estimate segment lengths
    print("\nCalculating median segment lengths...")
    segment_lengths = compute_segment_lengths(lf, all_segments, args.sample_size)
    for segment, length in segment_lengths.items():
        if length is not None:
            print(f"  {segment}: {length:.0f} bp")

    # Add segment length columns for distribution analysis
    print("\nAdding segment length columns...")
    lf = add_segment_length_columns(lf, all_segments)

    # Add quality scores (inverted normalized edit distances)
    print("\nComputing quality scores (excluding concatenated reads)...")
    lf = add_segment_scores(lf, all_segments, segment_lengths)

    # Compute segment length statistics
    length_stats_df = compute_segment_length_statistics(lf, all_segments, segment_lengths, anomaly_thresholds)

    # Compute statistics per segment
    print(f"\nCalculating statistics (ONT error rate: {args.ont_error_rate*100:.0f}%)...")
    all_stats = []
    for seg in all_segments:
        print(f"  Processing {seg}...")
        stats = compute_segment_statistics(
            lf, seg, segment_lengths.get(seg), args.ont_error_rate
        )
        all_stats.append(stats)

    # Create stats DataFrame
    stats_df = pl.DataFrame(all_stats)

    # Save statistics
    stats_file = output_dir / "edit_distance_statistics.tsv"
    stats_df.write_csv(stats_file, separator="\t")
    print(f"\nSaved statistics to {stats_file}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    display_cols = [
        'segment', 'median_length', 'reads_with_segment', 'nmf_count', 'percent_nmf',
        'percent_perfect', 'median', 'ont_threshold_2pct', 'percent_passing_2pct',
        'ont_threshold_5pct', 'percent_passing_5pct',
        'ont_threshold_10pct', 'percent_passing_10pct'
    ]
    print(stats_df.select(display_cols))

    # Generate visualizations
    plot_file = output_dir / "edit_distance_distributions.pdf"
    plot_edit_distance_distributions(lf, all_segments, segment_lengths, args.ont_error_rate, plot_file)
    print(f"\nSaved visualizations to {plot_file}")

    # Generate recommendations
    generate_filtering_recommendations(stats_df, args.ont_error_rate)

    # Generate quality score visualizations
    score_plot_file = output_dir / "quality_score_distributions.pdf"
    plot_score_distributions(lf, all_segments, score_plot_file)
    print(f"\nSaved quality score plots to {score_plot_file}")

    # Export ranked reads
    ranked_file = output_dir / "ranked_reads.tsv"
    export_ranked_reads(lf, ranked_file, n_top=1000, n_bottom=1000)

    # Save segment length statistics
    length_stats_file = output_dir / "segment_length_statistics.tsv"
    length_stats_df.write_csv(length_stats_file, separator="\t")
    print(f"\nSaved segment length statistics to {length_stats_file}")

    # Print length summary
    print("\n" + "="*80)
    print("SEGMENT LENGTH STATISTICS")
    print("="*80)
    length_display_cols = ['segment', 'reads_with_length', 'mean_length', 'median_length',
                           'std_length', 'min_length', 'p01_length', 'p05_length', 'max_length', 'outlier_percent']
    print(length_stats_df.select(length_display_cols))

    # Print anomaly summary if thresholds are specified
    print_anomaly_summary(length_stats_df)

    # Export anomalous reads if thresholds are specified
    if anomaly_thresholds:
        anomalous_file = output_dir / "anomalous_reads_valid.tsv"
        export_anomalous_reads(lf, anomaly_thresholds, anomalous_file)

    # Generate segment length visualizations
    length_plot_file = output_dir / "segment_length_distributions.pdf"
    plot_segment_length_distributions(lf, all_segments, segment_lengths, length_plot_file)
    print(f"\nSaved length distribution plots to {length_plot_file}")

    # ========== INVALID READS ANALYSIS (if provided) ==========
    if args.invalid_parquet:
        print("\n" + "="*80)
        print("ANALYZING INVALID READS")
        print("="*80)

        # Load invalid data
        print(f"\nScanning {args.invalid_parquet}...")
        lf_invalid = pl.scan_parquet(args.invalid_parquet)

        # Verify it has the same segments
        schema_invalid = lf_invalid.collect_schema()
        if "reason" not in schema_invalid:
            print("WARNING: Invalid parquet file missing 'reason' column!")

        # Add segment length columns for invalid reads
        print("\nAdding segment length columns for invalid reads...")
        lf_invalid = add_segment_length_columns(lf_invalid, all_segments)

        # Add quality scores for invalid reads
        print("\nComputing quality scores for invalid reads...")
        lf_invalid = add_segment_scores(lf_invalid, all_segments, segment_lengths)

        # Compute segment length statistics for invalid reads
        length_stats_invalid_df = compute_segment_length_statistics(lf_invalid, all_segments, segment_lengths, anomaly_thresholds)

        # Compute statistics for invalid reads
        print("\nCalculating invalid read statistics...")
        all_stats_invalid = []
        for seg in all_segments:
            print(f"  Processing {seg}...")
            stats = compute_segment_statistics(
                lf_invalid, seg, segment_lengths.get(seg), args.ont_error_rate
            )
            all_stats_invalid.append(stats)

        stats_invalid_df = pl.DataFrame(all_stats_invalid)

        # Save invalid statistics
        stats_invalid_file = output_dir / "edit_distance_statistics_invalid.tsv"
        stats_invalid_df.write_csv(stats_invalid_file, separator="\t")
        print(f"\nSaved invalid read statistics to {stats_invalid_file}")

        # Print invalid summary
        print("\n" + "="*80)
        print("INVALID READS SUMMARY STATISTICS")
        print("="*80)
        print(stats_invalid_df.select(display_cols))

        # Compare valid vs invalid
        print("\n" + "="*80)
        print("VALID vs INVALID COMPARISON")
        print("="*80)

        # Detection rate comparison
        comp_df = compare_detection_rates(all_stats, all_stats_invalid)
        comp_file = output_dir / "valid_vs_invalid_comparison.tsv"
        comp_df.write_csv(comp_file, separator="\t")
        print(f"\nSaved comparison to {comp_file}")

        # Generate comparison visualizations
        comparison_plot_file = output_dir / "valid_vs_invalid_comparison.pdf"
        plot_valid_vs_invalid_comparison(lf, lf_invalid, all_segments, segment_lengths, comparison_plot_file)
        print(f"\nSaved comparison plots to {comparison_plot_file}")

        # Generate quality score comparison
        score_comparison_file = output_dir / "quality_score_comparison.pdf"
        plot_score_comparison(lf, lf_invalid, all_segments, score_comparison_file)
        print(f"\nSaved score comparison plots to {score_comparison_file}")

        # Export ranked invalid reads
        ranked_invalid_file = output_dir / "ranked_reads_invalid.tsv"
        export_ranked_reads(lf_invalid, ranked_invalid_file, n_top=1000, n_bottom=1000)

        # Save invalid segment length statistics
        length_stats_invalid_file = output_dir / "segment_length_statistics_invalid.tsv"
        length_stats_invalid_df.write_csv(length_stats_invalid_file, separator="\t")
        print(f"\nSaved invalid segment length statistics to {length_stats_invalid_file}")

        # Print invalid length summary
        print("\n" + "="*80)
        print("INVALID READS - SEGMENT LENGTH STATISTICS")
        print("="*80)
        print(length_stats_invalid_df.select(length_display_cols))

        # Print anomaly summary for invalid reads
        if anomaly_thresholds:
            print("\n" + "="*80)
            print("INVALID READS - ANOMALY DETECTION")
            print("="*80)
            print_anomaly_summary(length_stats_invalid_df)

            # Export anomalous invalid reads
            anomalous_invalid_file = output_dir / "anomalous_reads_invalid.tsv"
            export_anomalous_reads(lf_invalid, anomaly_thresholds, anomalous_invalid_file)

        # Generate segment length comparison
        length_comparison_file = output_dir / "segment_length_comparison.pdf"
        plot_length_comparison(lf, lf_invalid, all_segments, length_comparison_file)
        print(f"\nSaved length comparison plots to {length_comparison_file}")

        # Analyze invalid reads by failure reason (optional - can be slow)
        if "reason" in schema_invalid and args.analyze_reasons:
            reason_stats_df = analyze_invalid_by_reason(
                lf_invalid, all_segments, segment_lengths, args.ont_error_rate, args.max_reasons
            )

            # Save reason breakdown
            reason_file = output_dir / "invalid_by_reason.tsv"
            reason_stats_df.write_csv(reason_file, separator="\t")
            print(f"\nSaved reason breakdown to {reason_file}")

            # Print top reasons summary
            print("\n" + "="*80)
            print("TOP FAILURE REASONS (by read count)")
            print("="*80)

            reason_summary = (
                reason_stats_df
                .group_by("reason")
                .agg(pl.col("reason_count").first().alias("read_count"))
                .sort("read_count", descending=True)
                .head(10)
            )
            print(reason_summary)

            # Generate reason breakdown plots
            reason_plot_file = output_dir / "invalid_reason_breakdown.pdf"
            plot_invalid_reason_breakdown(reason_stats_df, all_segments, reason_plot_file)
            print(f"\nSaved reason breakdown plots to {reason_plot_file}")
        elif "reason" in schema_invalid and not args.analyze_reasons:
            print("\nSkipping failure reason analysis (use --analyze_reasons to enable)")
            print("  Note: With many unique failure reasons, this can be slow")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
