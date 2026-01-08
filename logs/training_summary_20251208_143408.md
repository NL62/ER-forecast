# ER Patient Forecast - Training Summary

**Training Date**: 2025-12-08 14:34:08
**Optuna Trials per Horizon**: 10
**MAE Promotion Threshold**: 15.00 patients

## Results by Horizon

| Horizon | Test MAE | Test RMSE | Coverage | Interval Width | Promoted | Version |
|---------|----------|-----------|----------|----------------|----------|---------|
| 1 day | 7.4023 | 9.9028 | 96.1% | 43.10 | Yes | 9 |
| 2 days | 7.0214 | 9.6340 | 95.3% | 35.21 | Yes | 9 |
| 3 days | 6.9711 | 9.6061 | 96.5% | 38.99 | Yes | 9 |
| 4 days | 6.9572 | 9.5465 | 96.5% | 36.38 | Yes | 9 |
| 5 days | 6.9631 | 9.5788 | 96.4% | 38.74 | Yes | 9 |
| 6 days | 6.8138 | 9.4792 | 96.4% | 38.65 | Yes | 9 |
| 7 days | 6.7832 | 9.0465 | 96.4% | 35.78 | Yes | 9 |

## Summary Statistics

- **Models Trained**: 7
- **Models Promoted**: 7
- **Average Test MAE**: 6.9874

## Next Steps

- Models are now available in MLflow Model Registry
- Promoted models are in Production stage
- Daily prediction flow will use production models
