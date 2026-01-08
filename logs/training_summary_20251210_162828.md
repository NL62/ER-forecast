# ER Patient Forecast - Training Summary

**Training Date**: 2025-12-10 16:28:28
**Optuna Trials per Horizon**: 10
**MAE Promotion Threshold**: 15.00 patients

## Results by Horizon

| Horizon | Test MAE | Test RMSE | Coverage | Interval Width | Promoted | Version |
|---------|----------|-----------|----------|----------------|----------|---------|
| 1 day | 7.1188 | 9.5393 | 96.9% | 39.75 | Yes | 12 |
| 2 days | 6.9791 | 9.5413 | 95.3% | 35.37 | Yes | 12 |
| 3 days | 6.9126 | 9.5154 | 96.5% | 38.89 | Yes | 12 |
| 4 days | 6.7997 | 9.1516 | 96.5% | 36.36 | Yes | 12 |
| 5 days | 6.8373 | 9.2277 | 96.9% | 38.71 | Yes | 12 |
| 6 days | 6.6328 | 8.9996 | 96.9% | 38.64 | Yes | 12 |
| 7 days | 6.9527 | 9.2642 | 96.4% | 40.14 | Yes | 12 |

## Summary Statistics

- **Models Trained**: 7
- **Models Promoted**: 7
- **Average Test MAE**: 6.8904

## Next Steps

- Models are now available in MLflow Model Registry
- Promoted models are in Production stage
- Daily prediction flow will use production models
