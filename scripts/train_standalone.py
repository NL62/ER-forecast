#!/usr/bin/env python3
"""
Standalone training script (no Prefect required).

This script runs the complete training pipeline locally without
needing the Prefect server. Perfect for testing and development.

Usage:
    python scripts/train_standalone.py --quick
    python scripts/train_standalone.py --full
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mlflow
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from src.data.preprocessing import (
    load_raw_data,
    remove_duplicates,
    aggregate_to_daily,
    handle_missing_dates,
)
from src.data.weather_integration import fetch_weather_data, merge_weather_data
from src.data.feature_engineering import engineer_features_for_horizon, remove_nan_rows
from src.models.train import train_model_for_horizon
from src.utils.logging_config import setup_logging

console = Console()


class TrainingUI:
    """Rich-based training progress display with live updates."""
    
    def __init__(self, horizons: list, n_trials: int):
        self.horizons = horizons
        self.n_trials = n_trials
        self.current_horizon = None
        self.current_trial = 0
        self.best_mae = float('inf')
        self.last_mae = None
        self.results = {}
        self.start_time = datetime.now()
        self.horizon_start_time = None
        
        # Create progress bars
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            expand=True,
        )
        
        # Overall progress task
        self.overall_task = self.progress.add_task(
            "[cyan]Overall Progress",
            total=len(horizons)
        )
        
        # Trial progress task (with best MAE in description)
        self.trial_task = self.progress.add_task(
            "[yellow]Optuna Trials",
            total=n_trials,
            visible=False
        )
    
    def on_trial_complete(self, trial_num: int, total_trials: int, trial_mae: float, best_mae: float):
        """Callback for each completed Optuna trial."""
        self.current_trial = trial_num
        self.last_mae = trial_mae
        self.best_mae = best_mae
        
        # Update progress bar with current best MAE in description
        self.progress.update(
            self.trial_task, 
            completed=trial_num,
            description=f"[yellow]Horizon {self.current_horizon} [green]Best: {best_mae:.4f}[/green]"
        )
    
    def start_horizon(self, horizon: int):
        """Called when starting a new horizon."""
        self.current_horizon = horizon
        self.current_trial = 0
        self.best_mae = float('inf')
        self.last_mae = None
        self.horizon_start_time = datetime.now()
        
        # Reset trial progress
        self.progress.reset(self.trial_task)
        self.progress.update(
            self.trial_task,
            description=f"[yellow]Horizon {horizon} [dim]searching...[/dim]",
            visible=True,
            total=self.n_trials
        )
    
    def complete_horizon(self, horizon: int, metadata: dict, duration: float):
        """Called when a horizon is complete."""
        self.results[horizon] = {
            'metadata': metadata,
            'duration': duration
        }
        self.progress.update(self.overall_task, advance=1)
        self.progress.update(self.trial_task, visible=False)


def print_header(horizons: list, n_trials: int):
    """Print the training header."""
    header_text = Text()
    header_text.append("ER FORECAST MODEL TRAINING\n", style="bold cyan")
    header_text.append(f"Horizons: {horizons}\n", style="dim")
    header_text.append(f"Optuna trials per horizon: {n_trials}\n", style="dim")
    header_text.append(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim")
    
    console.print(Panel(header_text, title="[bold]Standalone Training", border_style="cyan"))


def print_summary(results: dict, total_time: float):
    """Print the final training summary."""
    console.print()
    
    # Results table
    table = Table(
        title="[bold green]Training Complete!",
        box=box.DOUBLE_EDGE,
        show_header=True,
        header_style="bold white on blue"
    )
    table.add_column("Horizon", justify="center", style="cyan", width=12)
    table.add_column("Test MAE", justify="right", style="green", width=12)
    table.add_column("Test RMSE", justify="right", style="yellow", width=12)
    table.add_column("Coverage", justify="right", style="magenta", width=12)
    table.add_column("Duration", justify="right", style="blue", width=12)
    
    for horizon in sorted(results.keys()):
        r = results[horizon]
        meta = r['metadata']
        coverage = meta.get('test_coverage')
        coverage_str = f"{coverage*100:.1f}%" if coverage else "N/A"
        
        table.add_row(
            f"{horizon} day{'s' if horizon > 1 else ''}",
            f"{meta['test_mae']:.4f}",
            f"{meta['test_rmse']:.4f}",
            coverage_str,
            f"{r['duration_seconds']:.1f}s"
        )
    
    # Add summary row
    avg_mae = sum(r['metadata']['test_mae'] for r in results.values()) / len(results)
    table.add_section()
    table.add_row(
        "[bold]Average",
        f"[bold]{avg_mae:.4f}",
        "",
        "",
        f"[bold]{total_time:.1f}s"
    )
    
    console.print(table)


def main():
    """Run standalone training with rich progress display."""
    
    parser = argparse.ArgumentParser(description='Train ER forecast models')
    parser.add_argument('--quick', action='store_true', help='Quick training (7 horizons, 150 trials)')
    parser.add_argument('--full', action='store_true', help='Full training (7 horizons, 300 trials)')
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 2], help='Horizons to train')
    parser.add_argument('--trials', type=int, default=5, help='Number of Optuna trials')
    parser.add_argument('--quiet', action='store_true', help='Minimal output (no progress bars)')
    
    args = parser.parse_args()
    
    # Set defaults
    if args.quick:
        horizons = [1, 2, 3, 4, 5, 6, 7]
        n_trials = 150
    elif args.full:
        horizons = [1, 2, 3, 4, 5, 6, 7]
        n_trials = 300
    else:
        horizons = args.horizons
        n_trials = args.trials
    
    # Setup logging (to file only, not console - we use rich for console output)
    setup_logging(level='INFO', log_file='logs/training.log', console=False)
    
    # Print header
    print_header(horizons, n_trials)
    
    # =================================================================
    # STEP 1: Load and Process Data
    # =================================================================
    with console.status("[bold cyan]Loading and preprocessing data...", spinner="dots"):
        df = load_raw_data('data/raw/emergency_visits.csv')
        df = remove_duplicates(df)
        df = aggregate_to_daily(df)
        df = handle_missing_dates(df)
        df = df[df['Patients_per_day'] > 0].copy()
    
    console.print(f"  [green]>[/green] Loaded [bold]{len(df)}[/bold] days of patient data")
    
    # Fetch weather
    with console.status("[bold cyan]Fetching weather data...", spinner="dots"):
        start_date = df['Date'].min().strftime('%Y-%m-%d')
        end_date = df['Date'].max().strftime('%Y-%m-%d')
        weather_df = fetch_weather_data(start_date, end_date, 59.6099, 16.5448)
        df = merge_weather_data(df, weather_df)
    
    console.print(f"  [green]>[/green] Fetched [bold]{len(weather_df)}[/bold] days of weather data")
    console.print()
    
    # =================================================================
    # STEP 2: Train Models
    # =================================================================
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("ER_Patient_Forecasting_Standalone")
    
    results = {}
    ui = TrainingUI(horizons, n_trials)
    
    with ui.progress:
        for horizon in horizons:
            ui.start_horizon(horizon)
            start_time = datetime.now()
            
            # Engineer features for this specific horizon
            df_horizon = engineer_features_for_horizon(df.copy(), weather_df, horizon)
            df_horizon = remove_nan_rows(df_horizon)
            
            # Train model with progress callback
            model, metadata = train_model_for_horizon(
                df=df_horizon,
                horizon=horizon,
                n_trials=n_trials,
                progress_callback=ui.on_trial_complete if not args.quiet else None
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Save model
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            model_path = model_dir / f"forecaster_horizon_{horizon}.pkl"
            model.save_model(str(model_path))
            
            results[horizon] = {
                'model': model,
                'metadata': metadata,
                'duration_seconds': duration
            }
            
            ui.complete_horizon(horizon, metadata, duration)
            
            # Print horizon result
            console.print(
                f"  [green]>[/green] Horizon {horizon}: "
                f"MAE=[bold green]{metadata['test_mae']:.4f}[/bold green], "
                f"RMSE=[yellow]{metadata['test_rmse']:.4f}[/yellow] "
                f"[dim]({duration:.1f}s)[/dim]"
            )
    
    # =================================================================
    # STEP 3: Summary
    # =================================================================
    total_time = sum(r['duration_seconds'] for r in results.values())
    print_summary(results, total_time)
    
    return results


if __name__ == "__main__":
    main()
