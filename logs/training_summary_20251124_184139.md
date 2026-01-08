# ER Patient Forecast - Training Summary

**Training Date**: 2025-11-24 18:41:39
**Optuna Trials per Horizon**: 10
**MAE Promotion Threshold**: 15.00 patients

## Results by Horizon

| Horizon | Test MAE | Test RMSE | Coverage | Interval Width | Promoted | Version |
|---------|----------|-----------|----------|----------------|----------|---------|
| 1 day | 7.3300 | 10.5093 | 95.0% | 42.49 | Yes | 4 |
| 2 days | 8.6297 | 11.7028 | 89.6% | 31.93 | Yes | 4 |
| 3 days | 7.5451 | 10.8747 | 93.7% | 41.11 | Yes | 4 |
| 4 days | 7.6000 | 11.0869 | 95.4% | 39.08 | Yes | 4 |
| 5 days | 8.1623 | 11.3135 | 94.1% | 35.60 | Yes | 4 |
| 6 days | 7.5158 | 10.8541 | 93.7% | 43.69 | Yes | 4 |
| 7 days | 7.6257 | 10.9303 | 94.5% | 38.48 | Yes | 4 |

## Summary Statistics

- **Models Trained**: 7
- **Models Promoted**: 7
- **Average Test MAE**: 7.7726

## Next Steps

- Models are now available in MLflow Model Registry
- Promoted models are in Production stage
- Daily prediction flow will use production models
