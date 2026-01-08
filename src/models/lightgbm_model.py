"""
LightGBM model wrapper for ER patient forecasting.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMRegressor

# Configure module logger
logger = logging.getLogger(__name__)


class LightGBMForecaster:
    """
    Wrapper class for LightGBM regression with quantile regression support.
    
    This class trains separate models for:
    - Point prediction (objective='regression')
    - Lower quantile (alpha=0.025 for 95% CI)
    - Upper quantile (alpha=0.975 for 95% CI)
    
    Attributes:
        point_model: Main regression model for point predictions
        lower_model: Model for lower confidence bound (2.5th percentile)
        upper_model: Model for upper confidence bound (97.5th percentile)
        feature_names: List of feature column names
        horizon: Forecast horizon in days (1-7)
    """
    
    def __init__(self, horizon: int = 1):
        """
        Initialize LightGBM forecaster.
        
        Args:
            horizon: Forecast horizon in days (1-7)
        """
        self.horizon = horizon
        self.point_model: Optional[LGBMRegressor] = None
        self.lower_model: Optional[LGBMRegressor] = None
        self.upper_model: Optional[LGBMRegressor] = None
        self.feature_names: Optional[list] = None
        
        logger.info(f"Initialized LightGBMForecaster for horizon={horizon} days")
    
    def train_point_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Train main regression model for point predictions.
        
        Args:
            X_train: Training features
            y_train: Training target values
            params: LightGBM hyperparameters (if None, uses defaults)
        """
        logger.info(f"Training point prediction model (horizon={self.horizon})")
        
        if params is None:
            params = self._get_default_params()
        
        # Store feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        
        # Create and train point model
        self.point_model = LGBMRegressor(
            objective='regression',
            metric='mae',
            verbosity=-1,
            **params
        )
        
        self.point_model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train)],
            eval_metric='mae',
        )
        
        logger.info(f"Point model trained with {len(X_train)} samples")
        #logger.debug(f"Feature importance: {self.point_model.feature_importances_[:5]}")
        logger.debug(f"Best iteration: {self.point_model.best_iteration_}")
    
    def train_quantile_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: Optional[Dict[str, Any]] = None,
        quantiles: Tuple[float, float] = (0.025, 0.975)
    ) -> None:
        """
        Train quantile regression models for confidence intervals.
        
        Args:
            X_train: Training features
            y_train: Training target values
            params: LightGBM hyperparameters (if None, uses defaults)
            quantiles: Tuple of (lower, upper) quantiles
        """
        logger.info(f"Training quantile regression models for confidence intervals")
        logger.info(f"Quantiles: {quantiles[0]:.3f}, {quantiles[1]:.3f} (95% CI)")
        
        if params is None:
            params = self._get_default_params()
        
        lower_quantile, upper_quantile = quantiles
        
        # Train lower quantile model
        logger.debug(f"Training lower quantile model (alpha={lower_quantile})")
        self.lower_model = LGBMRegressor(
            objective='quantile',
            alpha=lower_quantile,
            metric='quantile',
            verbosity=-1,
            **params
        )
        self.lower_model.fit(X_train, y_train)
        
        # Train upper quantile model
        logger.debug(f"Training upper quantile model (alpha={upper_quantile})")
        self.upper_model = LGBMRegressor(
            objective='quantile',
            alpha=upper_quantile,
            metric='quantile',
            verbosity=-1,
            **params
        )
        self.upper_model.fit(X_train, y_train)
        
        logger.info("Quantile models trained successfully")
    
    def predict(
        self,
        X: pd.DataFrame,
        return_intervals: bool = False
    ) -> pd.DataFrame:
        """
        Generate predictions with optional confidence intervals.
        
        Args:
            X: Features for prediction
            return_intervals: If True, include confidence intervals
        
        Returns:
            DataFrame with point_prediction, and optionally lower_bound/upper_bound
        """
        if self.point_model is None:
            raise ValueError("Point model not trained. Call train_point_model() first.")
        
        logger.debug(f"Generating predictions for {len(X)} samples")
        
        # Generate point predictions
        point_preds = self.point_model.predict(X)
        
        results = pd.DataFrame({
            'point_prediction': point_preds
        })
        
        # Add confidence intervals if requested
        if return_intervals:
            if self.lower_model is None or self.upper_model is None:
                logger.warning("Quantile models not trained. Cannot return confidence intervals.")
                logger.warning("Call train_quantile_models() first. Returning point predictions only.")
            else:
                lower_preds = self.lower_model.predict(X)
                upper_preds = self.upper_model.predict(X)
                
                # Enforce ordering: lower <= point <= upper
                # Quantile crossing can occur with independently trained models
                for i in range(len(point_preds)):
                    if lower_preds[i] > point_preds[i] or upper_preds[i] < point_preds[i]:
                        logger.warning(f"Quantile crossing detected at index {i}: "
                                      f"lower={lower_preds[i]:.1f}, point={point_preds[i]:.1f}, "
                                      f"upper={upper_preds[i]:.1f}. Clamping bounds to point.")
                        # Clamp bounds to point prediction
                        if lower_preds[i] > point_preds[i]:
                            lower_preds[i] = point_preds[i]
                        if upper_preds[i] < point_preds[i]:
                            upper_preds[i] = point_preds[i]
                
                results['lower_bound'] = lower_preds
                results['upper_bound'] = upper_preds
                
                # Log interval statistics
                interval_widths = upper_preds - lower_preds
                logger.debug(f"Mean interval width: {interval_widths.mean():.2f}")
                logger.debug(f"Interval width range: [{interval_widths.min():.2f}, {interval_widths.max():.2f}]")
        
        return results
    
    def save_model(self, filepath: str) -> None:
        """Save trained models to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to: {filepath}")
        
        model_data = {
            'point_model': self.point_model,
            'lower_model': self.lower_model,
            'upper_model': self.upper_model,
            'feature_names': self.feature_names,
            'horizon': self.horizon,
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved successfully ({filepath.stat().st_size / 1024:.1f} KB)")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'LightGBMForecaster':
        """Load trained models from disk."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        logger.info(f"Loading model from: {filepath}")
        
        model_data = joblib.load(filepath)
        
        # Create new instance
        forecaster = cls(horizon=model_data['horizon'])
        forecaster.point_model = model_data['point_model']
        forecaster.lower_model = model_data.get('lower_model')
        forecaster.upper_model = model_data.get('upper_model')
        forecaster.feature_names = model_data.get('feature_names')
        
        logger.info(f"Model loaded successfully (horizon={forecaster.horizon})")
        
        return forecaster
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from point model."""
        if self.point_model is None:
            raise ValueError("Point model not trained")
        
        if self.feature_names is None:
            raise ValueError("Feature names not available")
        
        importance_scores = self.point_model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df = importance_df.head(top_n)
        
        logger.debug(f"Top feature: {importance_df.iloc[0]['feature']} "
                    f"(importance={importance_df.iloc[0]['importance']:.1f})")
        
        return importance_df
    
    def _get_default_params(self) -> Dict[str, Any]:
        """
        Get default LightGBM hyperparameters.
        
        Returns:
            Dictionary of default parameters
        """
        return {
            'n_estimators': 100,
            'max_depth': 6,
            'num_leaves': 31,
            'learning_rate': 0.1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
        }
    
    def __repr__(self) -> str:
        """String representation of the forecaster."""
        models_trained = []
        if self.point_model is not None:
            models_trained.append('point')
        if self.lower_model is not None and self.upper_model is not None:
            models_trained.append('quantile')
        
        models_str = ', '.join(models_trained) if models_trained else 'none'
        
        return f"LightGBMForecaster(horizon={self.horizon}, models=[{models_str}])"
