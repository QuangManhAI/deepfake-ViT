# EVAL_STATUS.md — Evaluation

- **Title:** Evaluation & ViT-vs-CNN Comparison
- **Date created:** 2026-08-18
- **Last updated:** 2026-08-22
- **Description:** Status of evaluation, comparison, and attention visualization.
- **Status:** In Progress
- **Phase doc:** [../phases/EVAL.md](../phases/EVAL.md)

## Log

- 2026-08-18: Eval scripts secured (`weights_only=True`, no `allow_pickle`).
- 2026-08-22: 40-method evaluation report published
  (`experiments/results/eval/report_40_methods_v3.md`): DINOv3 ViT-S/16 vs
  ConvNeXt-Tiny on `test_data_v3` (identity-disjoint, linear probe).
- 2026-08-22: Data leakage analysis published
  (`experiments/results/eval/test_data_v3-build-and-leakage.md`): no identity
  leakage; protocol measures same-method generalization, not zero-shot unseen
  methods.
- 2026-08-22: Consolidated eval JSON reports and `lora_probs.npz` moved from
  `outputs/eval/` to the canonical `experiments/results/eval/` tree.

## Blockers (if any)

- Running eval requires the GPU env + downloaded test data/checkpoints.

## Decisions

- Report with 5W1H per [rules/RESULTS_REPORTING.md](../rules/RESULTS_REPORTING.md).

## Next step

- Produce the consolidated ViT-vs-CNN comparison report + attention figures
  (EXP-01 execution).

## Links

- Phase doc: [../phases/EVAL.md](../phases/EVAL.md)
- Eval artifacts: [../../experiments/results/eval/](../../experiments/results/eval/)
- Overview: [../OVERVIEW.md](../OVERVIEW.md)
