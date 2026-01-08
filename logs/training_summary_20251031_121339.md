# ER Patient Forecast - Training Summary

**Training Date**: 2025-10-31 12:13:39
**Optuna Trials per Horizon**: 10
**MAE Promotion Threshold**: 15.00 patients

## Results by Horizon

| Horizon | Test MAE | Test RMSE | Coverage | Interval Width | Promoted | Version |
|---------|----------|-----------|----------|----------------|----------|---------|
| 1 day | 7.3299 | 10.5092 | 94.6% | 42.13 | Yes | 37 |
| 2 days | 8.1384 | 11.1541 | 95.4% | 45.60 | Yes | 35 |
| 3 days | 7.5412 | 10.8721 | 92.9% | 41.11 | Yes | 34 |
| 4 days | 7.5652 | 10.9858 | 95.4% | 41.35 | Yes | 34 |
| 5 days | 8.1448 | 11.2650 | 94.1% | 35.60 | Yes | 34 |
| 6 days | 7.5044 | 10.8447 | 93.7% | 43.69 | Yes | 34 |
| 7 days | 7.6119 | 10.9159 | 94.1% | 38.73 | Yes | 34 |

## Summary Statistics

- **Models Trained**: 7
- **Models Promoted**: 7
- **Average Test MAE**: 7.6908

## Next Steps

- Models are now available in MLflow Model Registry
- Promoted models are in Production stage
- Daily prediction flow will use production models
