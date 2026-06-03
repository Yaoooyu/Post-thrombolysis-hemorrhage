| Method | Type | Combined AUC | Combined PR-AUC | Brier | Accuracy | Sensitivity | Specificity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MLP without adaptation | MLP branch | 0.518 | 0.443 | 0.415 | 0.412 | 1.000 | 0.000 |
| Mean-Std TTA | MLP branch | 0.604 | 0.499 | 0.301 | 0.555 | 0.755 | 0.414 |
| Mean-Std + Logit Prior | MLP branch | 0.658 | 0.558 | 0.247 | 0.655 | 0.571 | 0.714 |
| Modality reliability weighting | MLP-TTA branch | 0.661 | 0.561 | 0.244 | 0.664 | 0.592 | 0.714 |
| ExtraTrees EHR-anchor only | EHR-anchor / ensemble | 0.724 | 0.671 | 0.224 | 0.697 | 0.531 | 0.814 |
| ExtraTrees + MLP-TTA, fixed w=0.5 | EHR-anchor / ensemble | 0.697 | 0.637 | 0.219 | 0.622 | 0.531 | 0.686 |
| Single pooled center-weighted fusion | Reference | 0.754 | 0.702 | 0.220 | 0.723 | 0.612 | 0.800 |
| Hospital-specific optimized fusion (final) | Final | 0.734 | 0.668 | 0.214 | 0.739 | 0.633 | 0.814 |
