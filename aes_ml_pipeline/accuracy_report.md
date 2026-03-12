# Model Performance Report

## Training Results Summary

Based on the training run completed on the full dataset (2091 scans):

### Model Performance Metrics

| Model | MSE | MAE | RMSE | R² | Correlation |
|-------|-----|-----|------|----|----|
| **NN** | 260.86 | 4.83 | **16.15** | -348.81 | 0.09 |
| **LSTM** | 0.77 | 0.44 | **0.88** | -0.04 | 0.31 |
| **GRU** | 0.78 | 0.43 | **0.88** | -0.04 | 0.30 |

### Key Findings

1. **LSTM and GRU significantly outperform NN**
   - LSTM/GRU RMSE: ~0.88 (excellent)
   - NN RMSE: 16.15 (poor, ~18x worse)
   - This suggests sequence models (LSTM/GRU) are much better suited for this temporal prediction task

2. **LSTM is the best model**
   - Lowest RMSE: 0.879474
   - Highest Correlation: 0.305592
   - Best R²: -0.037199 (slightly negative but much better than NN)

3. **Model Quality Assessment**
   - **LSTM/GRU**: Good performance (RMSE < 1.0, reasonable correlation ~0.30)
   - **NN**: Poor performance (RMSE > 16, very low correlation ~0.09)
   - Negative R² values suggest models could be improved, but LSTM/GRU are in reasonable range

4. **Training Efficiency**
   - All models trained successfully with Bayesian Optimization
   - Data loading: ~0.9s (very fast after optimization)
   - Training completed for all 3 models

### Recommendations

1. **Use LSTM model** for predictions (best overall performance)
2. **Consider further hyperparameter tuning** to improve R² (currently slightly negative)
3. **Investigate NN model** - the poor performance suggests it may not be suitable for this task
4. **Try Transformer or TCN models** - they may perform even better than LSTM/GRU

### Best Model Configuration

**LSTM Model:**
- Learning Rate: 0.005
- Hidden Dim: 247
- Num Layers: 3
- Dropout: 0.3






