"""
Model training with Optuna hyperparameter tuning.
"""

import logging
from typing import Tuple, Dict, Any, Optional, Callable
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from src.models.lightgbm_model import LightGBMForecaster
from src.models.evaluation import calculate_mae

# Configure module logger
logger = logging.getLogger(__name__)


class TrialProgressCallback:
    """
    Optuna callback that reports trial progress via a user-provided function.
    
    This allows the standalone script to display rich progress without
    coupling the training module to any specific UI library.
    """
    
    def __init__(self, on_trial_complete: Optional[Callable[[int, int, float, float], None]] = None):
        """
        Args:
            on_trial_complete: Callback function(trial_number, n_trials, trial_mae, best_mae)
        """
        self.on_trial_complete = on_trial_complete
        self.n_trials = 0
    
    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if self.on_trial_complete and trial.state == optuna.trial.TrialState.COMPLETE:
            self.on_trial_complete(
                trial.number + 1,
                self.n_trials,
                trial.value,
                study.best_value
            )


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.60,
    val_ratio: float = 0.10,
    calib_ratio: float = 0.10,
    test_ratio: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split time series data into train/val/calib/test sets (respecting temporal order)."""
    total_ratio = train_ratio + val_ratio + calib_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    n = len(df)
    
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)
    calib_end = val_end + int(calib_ratio * n)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    calib_df = df.iloc[val_end:calib_end].copy()
    test_df = df.iloc[calib_end:].copy()
    
    logger.info(f"Data split: Train={len(train_df)} ({len(train_df)/n*100:.1f}%), "
                f"Val={len(val_df)} ({len(val_df)/n*100:.1f}%), "
                f"Calib={len(calib_df)} ({len(calib_df)/n*100:.1f}%), "
                f"Test={len(test_df)} ({len(test_df)/n*100:.1f}%)")
    
    return train_df, val_df, calib_df, test_df


def create_shifted_target(
    df: pd.DataFrame,
    horizon: int,
    target_col: str = 'Patients_per_day'
) -> pd.DataFrame:
    """Create shifted target for forecast horizon (e.g., horizon=1 means predict tomorrow)."""
    logger.info(f"Creating shifted target for horizon={horizon} days")
    
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame")
    
    if not 1 <= horizon <= 7:
        raise ValueError(f"Horizon must be between 1 and 7, got {horizon}")
    
    df = df.copy()
    
    df['target'] = df[target_col].shift(-horizon)
    
    original_rows = len(df)
    df = df.dropna(subset=['target']).reset_index(drop=True)
    removed_rows = original_rows - len(df)
    
    logger.info(f"Created target for horizon {horizon}: {len(df)} samples "
                f"({removed_rows} rows removed due to shifting)")
    
    return df


def optuna_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    horizon: int = 1
) -> float:
    """Optuna objective function - suggests hyperparameters, trains model, returns validation MAE."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=10),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200, step=10),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100, step=5),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.05),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.05),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0, step=0.1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0, step=0.1),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 10.0, log=True),
    }
    
    params.update({
        'random_state': 42,
        'n_jobs': -1,
    })
    
    forecaster = LightGBMForecaster(horizon=horizon)
    forecaster.train_point_model(X_train, y_train, params)
    
    preds = forecaster.predict(X_val, return_intervals=False)
    y_pred = preds['point_prediction'].values
    
    val_mae = calculate_mae(y_val.values, y_pred)
    
    return val_mae


def train_model_for_horizon(
    df: pd.DataFrame,
    horizon: int,
    n_trials: int = 300,
    target_col: str = 'Patients_per_day',
    optuna_config: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[int, int, float, float], None]] = None
) -> Tuple[LightGBMForecaster, Dict[str, Any]]:
    """
    Train a complete forecasting model for a specific horizon.
    
    This is the main training function that:
    1. Creates shifted target
    2. Splits data
    3. Runs Optuna hyperparameter search
    4. Trains final model with best parameters
    5. Trains quantile models for confidence intervals
    
    Args:
        df: DataFrame with features and target
        horizon: Forecast horizon in days (1-7)
        n_trials: Number of Optuna trials
        target_col: Name of target column
        optuna_config: Optional Optuna configuration dict
    
    Returns:
        Tuple of (trained_model, metadata_dict)
    """
    logger.info(f"Starting training for horizon {horizon} days")
    logger.info(f"Data shape: {df.shape}, Optuna trials: {n_trials}")
    
    # Create shifted target
    df_horizon = create_shifted_target(df, horizon, target_col)
    
    # Split data
    train_df, val_df, calib_df, test_df = split_data(df_horizon)
    
    # Separate features and target
    feature_cols = [col for col in df_horizon.columns if col not in ['target', 'Date', target_col]]
    
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    
    X_val = val_df[feature_cols]
    y_val = val_df['target']
    
    X_calib = calib_df[feature_cols]
    y_calib = calib_df['target']
    
    X_test = test_df[feature_cols]
    y_test = test_df['target']
    
    logger.info(f"Features: {len(feature_cols)}")
    logger.debug(f"First 10 features: {feature_cols[:10]}")
    
    # Run Optuna hyperparameter search
    logger.info(f"Starting Optuna hyperparameter search ({n_trials} trials)")
    
    # Suppress Optuna's verbose logging during optimization
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Create study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # Setup progress callback if provided
    callbacks = []
    if progress_callback:
        trial_callback = TrialProgressCallback(on_trial_complete=progress_callback)
        trial_callback.n_trials = n_trials
        callbacks.append(trial_callback)
    
    # Optimize
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val, horizon),
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=callbacks if callbacks else None,
        catch=(Exception,)  # Continue even if some trials fail
    )
    
    # Get best parameters
    best_params = study.best_params
    best_val_mae = study.best_value
    
    logger.info(f"Optuna search complete!")
    logger.info(f"Best validation MAE: {best_val_mae:.4f}")
    # logger.debug(f"Trial history: {study.trials_dataframe()}")
    logger.info(f"Best parameters: {best_params}")
    
    # Train final model with best parameters
    logger.info("Training final model with best parameters")
    
    forecaster = LightGBMForecaster(horizon=horizon)
    
    # Add fixed parameters
    final_params = best_params.copy()
    final_params.update({
        'random_state': 42,
        'n_jobs': -1,
    })
    
    # Train point model
    forecaster.train_point_model(X_train, y_train, final_params)
    
    # Train quantile models for confidence intervals
    logger.info("Training quantile models for confidence intervals")
    forecaster.train_quantile_models(X_train, y_train, final_params)
    
    # Evaluate on test set
    from src.models.evaluation import evaluate_model
    test_metrics = evaluate_model(forecaster, X_test, y_test)
    
    logger.info(f"Test set evaluation:")
    logger.info(f"  MAE: {test_metrics['mae']:.4f}")
    logger.info(f"  RMSE: {test_metrics['rmse']:.4f}")
    if 'coverage' in test_metrics:
        logger.info(f"  Coverage: {test_metrics['coverage']*100:.2f}%")
        logger.info(f"  Interval Width: {test_metrics['interval_width']:.2f}")
    
    # Compile metadata
    metadata = {
        'horizon': horizon,
        'n_trials': n_trials,
        'best_params': best_params,
        'best_val_mae': best_val_mae,
        'test_mae': test_metrics['mae'],
        'test_rmse': test_metrics['rmse'],
        'test_coverage': test_metrics.get('coverage'),
        'test_interval_width': test_metrics.get('interval_width'),
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_calib': len(X_calib),
        'n_test': len(X_test),
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
    }
    
    logger.info(f"Training complete for horizon {horizon}")
    
    return forecaster, metadata


def calibrate_conformal_prediction(
    model: LightGBMForecaster,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    alpha: float = 0.05
) -> float:
    """
    Calibrate confidence intervals using conformal prediction.
    
    Uses the calibration set to compute a correction factor that
    ensures the confidence intervals have the desired coverage.
    
    Args:
        model: Trained LightGBMForecaster
        X_calib: Calibration features
        y_calib: Calibration target
        alpha: Significance level (0.05 for 95% CI)
    
    Returns:
        Correction factor to widen intervals
    
    Note:
        Currently returns 1.0 (no correction) as quantile regression
        is used directly. This function is a placeholder for future
        conformal prediction implementation.
    """
    logger.info("Calibrating confidence intervals with conformal prediction")
    
    # Generate predictions on calibration set
    preds = model.predict(X_calib, return_intervals=True)
    
    if 'lower_bound' not in preds.columns or 'upper_bound' not in preds.columns:
        logger.warning("Quantile models not available, cannot calibrate")
        return 1.0
    
    y_pred = preds['point_prediction'].values
    y_lower = preds['lower_bound'].values
    y_upper = preds['upper_bound'].values
    y_true = y_calib.values
    
    # Calculate non-conformity scores (residuals)
    residuals = np.abs(y_true - y_pred)
    
    # Calculate quantile of residuals for desired coverage
    quantile_value = np.quantile(residuals, 1 - alpha)
    
    # Calculate current interval widths
    current_widths = y_upper - y_lower
    avg_current_width = current_widths.mean()
    
    # Correction factor (currently not applied, just logged)
    correction_factor = quantile_value / (avg_current_width / 2)
    
    logger.info(f"Conformal calibration:")
    logger.info(f"  Residual quantile ({1-alpha:.2%}): {quantile_value:.2f}")
    logger.info(f"  Current avg interval width: {avg_current_width:.2f}")
    logger.info(f"  Suggested correction factor: {correction_factor:.3f}")
    
    # For now, return 1.0 (no correction)
    # In future, could apply this correction to widen/narrow intervals
    return 1.0


def train_all_horizons(
    df: pd.DataFrame,
    horizons: list = None,
    n_trials: int = 300,
    target_col: str = 'Patients_per_day'
) -> Dict[int, Tuple[LightGBMForecaster, Dict[str, Any]]]:
    """
    Train models for all forecast horizons (1-7 days).
    
    Trains separate models for each horizon in sequence.
    
    Args:
        df: DataFrame with features and target
        horizons: List of horizons to train (default: [1, 2, 3, 4, 5, 6, 7])
        n_trials: Number of Optuna trials per horizon
        target_col: Name of target column
    
    Returns:
        Dictionary mapping horizon -> (model, metadata)
    """
    if horizons is None:
        horizons = [1, 2, 3, 4, 5, 6, 7]
    
    logger.info(f"Training models for {len(horizons)} horizons: {horizons}")
    logger.info(f"Optuna trials per horizon: {n_trials}")
    
    results = {}
    
    for horizon in horizons:
        logger.info(f"\n{'='*70}")
        logger.info(f"TRAINING HORIZON {horizon} DAY{'S' if horizon > 1 else ''} AHEAD")
        logger.info(f"{'='*70}\n")
        
        try:
            model, metadata = train_model_for_horizon(
                df=df,
                horizon=horizon,
                n_trials=n_trials,
                target_col=target_col
            )
            
            results[horizon] = (model, metadata)
            
            logger.info(f"Horizon {horizon} complete: "
                       f"Test MAE={metadata['test_mae']:.4f}, "
                       f"Coverage={metadata.get('test_coverage', 0)*100:.1f}%")
            
        except Exception as e:
            logger.error(f"Failed to train horizon {horizon}: {e}")
            raise
    
    logger.info(f"\n{'='*70}")
    logger.info(f"ALL HORIZONS TRAINING COMPLETE")
    logger.info(f"{'='*70}\n")
    
    # Summary
    logger.info("Summary of results:")
    for h in sorted(results.keys()):
        meta = results[h][1]
        logger.info(f"  Horizon {h}: MAE={meta['test_mae']:.4f}, "
                   f"Coverage={meta.get('test_coverage', 0)*100:.1f}%")
    
    return results
