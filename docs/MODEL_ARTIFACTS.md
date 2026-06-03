# Model Artifacts

This GitHub package keeps only the final selected main model and excludes all other model weights.

## Included model

- Final multimodal MLP fusion model: `models/final_main_model/task_04_multimodal_mlp_fusion.pt`

## Excluded model artifacts

The following server-side model weights are intentionally not included. The repository keeps the corresponding scripts, metrics, prediction tables, leaderboards, and configuration/runtime notes so users can reproduce them by retraining.

- Task 1 AutoGluon EHR model: `/root/autodl-fs/lyy/frozen_models/task_01_ehr_AutoGluonModels`
- Task 2 AutoGluon EHR + CT report model: `/root/autodl-fs/lyy/frozen_models/task_02_ehr_cttext_AutoGluonModels`
- Task 3 pure CT ResNet18 model: `/root/autodl-fs/lyy/frozen_models/task_03_pure_ct_resnet18.pth`
- Task4 exploratory/end-to-end model weights, including FiLM experiments, are excluded unless they are the final selected main model.

## Included reproduction materials

- Training and evaluation scripts: `src/`
- Task-level metrics and predictions: `results/`
- ROC/PR and summary figures: `figures/`
- Data split and ID mapping metadata: `data/`
