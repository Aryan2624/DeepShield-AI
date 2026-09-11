# DeepShield AI Saved Models

Trained model binaries are not stored directly in this GitHub repository because they may be very large.

## Malicious URL Detector

Current candidate model:

- Algorithm: Random Forest
- Validation Macro F1: 0.9197
- Final Test Accuracy: 0.9342
- Final Test Macro F1: 0.9186
- Final Test ROC-AUC: 0.9899
- Features: 30 engineered lexical URL features

Local filename:

```text
url_detector_candidate.joblib