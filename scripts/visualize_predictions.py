#!/usr/bin/env python3
"""
Visualize predictions from a CSV file.

This script loads a predictions CSV file and creates visualizations
showing point predictions, confidence intervals, and horizon comparisons.

Usage:
    python scripts/visualize_predictions.py [--file path/to/predictions.csv] [--output plots/]
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def load_predictions(file_path: str) -> pd.DataFrame:
    """Load predictions from CSV file."""
    df = pd.read_csv(file_path)
    
    # Convert date columns to datetime
    df['prediction_timestamp'] = pd.to_datetime(df['prediction_timestamp'])
    df['prediction_date'] = pd.to_datetime(df['prediction_date'])
    
    return df


def create_time_series_plot(df: pd.DataFrame, output_path: Path):
    """Create time series plot showing predictions over time."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sort by prediction date for proper plotting
    df_sorted = df.sort_values('prediction_date')
    
    # Plot point predictions
    ax.plot(df_sorted['prediction_date'], df_sorted['point_prediction'], 
            'o-', linewidth=2.5, markersize=10, label='Point Prediction',
            color='#2E86AB', alpha=0.8)
    
    # Plot confidence intervals
    ax.fill_between(df_sorted['prediction_date'], 
                    df_sorted['lower_bound'], 
                    df_sorted['upper_bound'],
                    alpha=0.3, color='#F18F01', label='95% Confidence Interval')
    
    # Add horizon labels
    for _, row in df_sorted.iterrows():
        ax.annotate(f'H{int(row["horizon"])}', 
                   xy=(row['prediction_date'], row['point_prediction']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.7)
    
    ax.set_xlabel('Prediction Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
    ax.set_title('Emergency Room Patient Predictions with Confidence Intervals', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved time series plot: {output_path}")
    plt.show()
    plt.close()


def create_horizon_comparison_plot(df: pd.DataFrame, output_path: Path):
    """Create bar plot comparing predictions across horizons."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Predictions Analysis Across Forecast Horizons', 
                fontsize=16, fontweight='bold', y=1.02)
    
    # Sort by horizon
    df_sorted = df.sort_values('horizon')
    
    # Plot 1: Point predictions by horizon
    ax1 = axes[0, 0]
    bars1 = ax1.bar(df_sorted['horizon'], df_sorted['point_prediction'], 
                    color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Forecast Horizon (days)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Point Prediction (patients)', fontsize=11, fontweight='bold')
    ax1.set_title('Point Predictions by Horizon', fontsize=12, fontweight='bold')
    ax1.set_xticks(df_sorted['horizon'])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Confidence interval width by horizon
    ax2 = axes[0, 1]
    ci_width = df_sorted['upper_bound'] - df_sorted['lower_bound']
    bars2 = ax2.bar(df_sorted['horizon'], ci_width, 
                    color='#F18F01', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Forecast Horizon (days)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('CI Width (patients)', fontsize=11, fontweight='bold')
    ax2.set_title('Confidence Interval Width by Horizon', fontsize=12, fontweight='bold')
    ax2.set_xticks(df_sorted['horizon'])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Prediction range (lower to upper) by horizon
    ax3 = axes[1, 0]
    x_pos = df_sorted['horizon']
    y_lower = df_sorted['lower_bound']
    y_upper = df_sorted['upper_bound']
    y_point = df_sorted['point_prediction']
    
    # Plot error bars
    for i, (h, lower, upper, point) in enumerate(zip(x_pos, y_lower, y_upper, y_point)):
        ax3.plot([h, h], [lower, upper], 'k-', linewidth=2, alpha=0.6)
        ax3.plot(h, point, 'o', markersize=12, color='#A23B72', 
                markeredgecolor='black', markeredgewidth=1.5)
    
    ax3.set_xlabel('Forecast Horizon (days)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Number of Patients', fontsize=11, fontweight='bold')
    ax3.set_title('Prediction Range by Horizon', fontsize=12, fontweight='bold')
    ax3.set_xticks(df_sorted['horizon'])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Prediction date vs horizon (heatmap-style or line)
    ax4 = axes[1, 1]
    # Create a scatter plot with horizon on x and predictions on y
    scatter = ax4.scatter(df_sorted['horizon'], df_sorted['point_prediction'],
                         s=150, c=df_sorted['horizon'], cmap='viridis',
                         edgecolors='black', linewidth=1.5, alpha=0.7)
    
    # Add confidence interval lines
    for _, row in df_sorted.iterrows():
        ax4.plot([row['horizon'], row['horizon']], 
                [row['lower_bound'], row['upper_bound']],
                'k-', linewidth=1.5, alpha=0.4)
    
    ax4.set_xlabel('Forecast Horizon (days)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Point Prediction (patients)', fontsize=11, fontweight='bold')
    ax4.set_title('Prediction Distribution by Horizon', fontsize=12, fontweight='bold')
    ax4.set_xticks(df_sorted['horizon'])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved horizon comparison plot: {output_path}")
    plt.show()
    plt.close()


def create_summary_statistics(df: pd.DataFrame):
    """Print summary statistics about the predictions."""
    
    print("\n" + "="*70)
    print("PREDICTION SUMMARY STATISTICS")
    print("="*70)
    
    print(f"\nPrediction Timestamp: {df['prediction_timestamp'].iloc[0]}")
    print(f"Model Version: {df['model_version'].iloc[0]}")
    print(f"Number of Predictions: {len(df)}")
    print(f"Date Range: {df['prediction_date'].min().date()} to {df['prediction_date'].max().date()}")
    
    print("\n" + "-"*70)
    print(f"{'Horizon':<10} {'Date':<12} {'Point Pred':<12} {'Lower':<12} {'Upper':<12} {'CI Width':<12}")
    print("-"*70)
    
    for _, row in df.sort_values('horizon').iterrows():
        ci_width = row['upper_bound'] - row['lower_bound']
        print(f"{int(row['horizon']):<10} {row['prediction_date'].date()} "
              f"{row['point_prediction']:<12.2f} {row['lower_bound']:<12.2f} "
              f"{row['upper_bound']:<12.2f} {ci_width:<12.2f}")
    
    print("\n" + "-"*70)
    print("Overall Statistics:")
    print(f"  Mean Point Prediction: {df['point_prediction'].mean():.2f} patients")
    print(f"  Std Point Prediction:  {df['point_prediction'].std():.2f} patients")
    print(f"  Mean CI Width:         {(df['upper_bound'] - df['lower_bound']).mean():.2f} patients")
    print(f"  Min Prediction:        {df['point_prediction'].min():.2f} patients (H{int(df.loc[df['point_prediction'].idxmin(), 'horizon'])})")
    print(f"  Max Prediction:        {df['point_prediction'].max():.2f} patients (H{int(df.loc[df['point_prediction'].idxmax(), 'horizon'])})")
    print("="*70 + "\n")


def main():
    """Main function to visualize predictions."""
    
    parser = argparse.ArgumentParser(
        description='Visualize predictions from a CSV file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/visualize_predictions.py
  python scripts/visualize_predictions.py --file data/predictions/predictions_20251031_110647.csv
  python scripts/visualize_predictions.py --file data/predictions/predictions_20251031_110647.csv --output custom_plots/
        """
    )
    
    parser.add_argument('--file', type=str, 
                       default='data/predictions/predictions_20251031_110647.csv',
                       help='Path to predictions CSV file (default: data/predictions/predictions_20251031_110647.csv)')
    
    parser.add_argument('--output', type=str, default='plots',
                       help='Output directory for plots (default: plots/)')
    
    args = parser.parse_args()
    
    # Resolve paths relative to project root
    file_path = project_root / args.file
    output_dir = project_root / args.output
    
    # Check if file exists
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("PREDICTIONS VISUALIZATION")
    print("="*70)
    print(f"\nLoading predictions from: {file_path}")
    
    # Load predictions
    df = load_predictions(str(file_path))
    
    # Print summary statistics
    create_summary_statistics(df)
    
    # Generate file name from input file
    input_filename = file_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create visualizations
    print("Creating visualization...")
    
    # Time series plot
    time_series_path = output_dir / f'{input_filename}_time_series.png'
    create_time_series_plot(df, time_series_path)
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\nPlot saved to: {output_dir}/")
    print(f"  - {input_filename}_time_series.png")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

