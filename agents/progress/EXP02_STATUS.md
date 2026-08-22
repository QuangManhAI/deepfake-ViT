# EXP-02 STATUS — Accuracy Improvement Plan

- **Title:** Kế hoạch cải thiện accuracy & khắc phục detect yếu
- **Date created:** 2026-08-22
- **Last updated:** 2026-08-22
- **Description:** Tracking progress cho EXP-02 accuracy improvement experiments.
- **Status:** Planning Complete → Awaiting Execution
- **Experiment doc:** [../experiments/EXP_02_ACCURACY_IMPROVEMENT_PLAN.md](../experiments/EXP_02_ACCURACY_IMPROVEMENT_PLAN.md)

## Log

- 2026-08-22: Planning document EXP-02 created with comprehensive analysis of current weaknesses:
  - Overall accuracy: 93.93% (target: >97.5%)
  - Top weakness: MidJourney (31.82% detection), whichfaceisreal (49.02% detection)
  - EFS domain accuracy: 81.50% (target: >95%)
  - 5 phases identified with 15 experiments
  - Root cause analysis: training data lacks EFS methods (RC-1) accounts for 75% of errors

## Current Baseline

| Metric | Value |
|:---|:---:|
| Overall Accuracy | 93.93% |
| EFS Domain Acc | 81.50% |
| MidJourney Det | 31.82% |
| whichfaceisreal Det | 49.02% |
| ROC-AUC | 98.33% |

## Blockers

- None currently

## Next Steps

1. Phase 1A: Threshold Optimization (quick win, 2h)
2. Phase 1B: TTA Implementation (quick win, 4h)
3. Phase 2A: Add EFS data to training set (highest impact, 8h)

## Links

- Experiment plan: [../experiments/EXP_02_ACCURACY_IMPROVEMENT_PLAN.md](../experiments/EXP_02_ACCURACY_IMPROVEMENT_PLAN.md)
- Predecessor: [../experiments/EXP_01_ACCURACY_OPTIMIZATION_PLAN.md](../experiments/EXP_01_ACCURACY_OPTIMIZATION_PLAN.md)
- Error analysis: [../../notebooks/02_error_analysis.ipynb](../../notebooks/02_error_analysis.ipynb)
- Training notebook: [../../notebooks/02_training_balanced_dataset.ipynb](../../notebooks/02_training_balanced_dataset.ipynb)
