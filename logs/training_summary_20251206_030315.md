# ER Patient Forecast - Training Summary

**Training Date**: 2025-12-06 03:03:15
**Optuna Trials per Horizon**: 10
**MAE Promotion Threshold**: 15.00 patients

## Results by Horizon

| Horizon | Test MAE | Test RMSE | Coverage | Interval Width | Promoted | Version |
|---------|----------|-----------|----------|----------------|----------|---------|
| 1 day | 7.3343 | 9.8251 | 96.1% | 43.15 | Yes | 8 |
| 2 days | 7.6412 | 10.2673 | 96.1% | 43.20 | Yes | 8 |
| 3 days | 6.9033 | 9.5130 | 96.4% | 38.99 | Yes | 8 |
| 4 days | 6.8962 | 9.4599 | 95.7% | 36.25 | Yes | 8 |
| 5 days | 6.9182 | 9.5189 | 96.4% | 38.57 | Yes | 8 |
| 6 days | 6.7389 | 9.3664 | 96.5% | 38.46 | Yes | 8 |
| 7 days | 6.8300 | 9.3177 | 96.1% | 36.10 | Yes | 8 |

## Summary Statistics

- **Models Trained**: 7
- **Models Promoted**: 7
- **Average Test MAE**: 7.0374

## Next Steps

- Models are now available in MLflow Model Registry
- Promoted models are in Production stage
- Daily prediction flow will use production models
