# ER Patient Forecast - Training Summary

**Training Date**: 2025-12-06 02:46:26
**Optuna Trials per Horizon**: 10
**MAE Promotion Threshold**: 15.00 patients

## Results by Horizon

| Horizon | Test MAE | Test RMSE | Coverage | Interval Width | Promoted | Version |
|---------|----------|-----------|----------|----------------|----------|---------|
| 1 day | 6.7695 | 8.6624 | 97.1% | 39.69 | Yes | 6 |
| 2 days | 7.1829 | 9.2304 | 97.1% | 43.88 | Yes | 6 |
| 3 days | 6.6129 | 8.6077 | 96.7% | 38.63 | Yes | 6 |
| 4 days | 6.6571 | 8.6655 | 96.2% | 37.93 | Yes | 6 |
| 5 days | 6.8010 | 8.7049 | 96.6% | 39.34 | Yes | 6 |
| 6 days | 6.6898 | 8.7133 | 95.8% | 37.50 | Yes | 6 |
| 7 days | 6.8808 | 8.8424 | 95.0% | 35.52 | Yes | 6 |

## Summary Statistics

- **Models Trained**: 7
- **Models Promoted**: 7
- **Average Test MAE**: 6.7991

## Next Steps

- Models are now available in MLflow Model Registry
- Promoted models are in Production stage
- Daily prediction flow will use production models
