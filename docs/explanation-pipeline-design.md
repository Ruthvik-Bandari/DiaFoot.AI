# Why DiaFoot.AI is a cascaded multi-task pipeline

This document explains the design decisions behind DiaFoot.AI v2: why it classifies before it
segments, why it defers to clinicians, and why the honest results look the way they do. If you
want commands, see the how-to guides; this is the "why."

## The problem v1 could not solve

DiaFoot.AI v1 was a single U-Net++ segmenter trained only on ulcer images. On paper it looked
excellent — 91.7% Dice, 84.9% IoU. In practice it was clinically useless, for one reason:

> It had never seen a healthy foot. Every training image contained an ulcer, so the model
> learned "there is always a wound; find it." Show it healthy skin and it confidently segments
> an ulcer that isn't there.

A model that cannot say "this is not a wound" cannot triage. Its high Dice was measuring how
well it outlined wounds *in a set where every image had a wound* — a benchmark that does not
exist in a clinic waiting room.

## The approach: classify, then segment, then measure

v2 restructures the system as a cascade of three stages, each with a single job:

![Inference pipeline](diagrams/inference-pipeline.svg)

```
          ┌─────────────────────┐
 image ──▶│ 1. Triage classifier│──▶ Healthy / Non-DFU / DFU  (+ confidence)
          └─────────┬───────────┘
                    │ wound suspected
                    ▼
          ┌─────────────────────┐
          │ 2. Wound segmenter  │──▶ binary wound mask
          └─────────┬───────────┘
                    ▼
          ┌─────────────────────┐
          │ 3. Area measurement │──▶ wound_area_mm²
          └─────────────────────┘
```

Putting a classifier first gives the system the vocabulary v1 lacked: it can now abstain on
healthy skin instead of hallucinating a wound. The segmenter only runs when a wound is
plausible, so its Dice is measured on the cases where segmentation is actually the question.

### Multi-task training

The models are trained with multiple heads (classification, segmentation, and Wagner staging)
sharing an encoder, with task weights and a curriculum that warms up the classifier first, then
unfreezes segmentation, then staging. Shared representation learning across related tasks is the
motivation; the trade-off is a more delicate training recipe (task-weight balancing via GradNorm,
staged unfreezing) than a single-task model needs.

## The defer-to-clinician gate

A medical triage tool that is confidently wrong is worse than one that knows when to ask for
help. DiaFoot.AI has two abstention mechanisms:

1. **Quality gate.** Before inference, the API scores brightness, blur, and size. Images that
   fail return `defer_to_clinician: true` with a reason — no prediction is attempted on an image
   too dark or blurry to judge.
2. **Confidence defer.** Calibrated classifier confidence below a tuned threshold defers the case.
   At a 0.95 threshold, the system keeps 93.5% of cases and is **99.7% accurate on the ones it
   keeps**, sending the uncertain 6.5% to a human.

This is a deliberate trade: coverage for reliability. The tuned threshold comes from a coverage
sweep, not a guess, and is stored so the API loads it at startup.

### Why calibration matters here

Raw neural-network confidence is usually overconfident, which makes a confidence-based defer gate
meaningless. Temperature scaling (T=0.4) recalibrates it: expected calibration error drops from
0.039 to 0.007. Only after calibration does "defer below 0.95" mean what it says.

## The leakage story — why the honest numbers are lower

During v2 evaluation, a **data-leakage audit** found ~96,829 near-duplicate image pairs spanning
the train and test sets (same wound photographed under slight variation, augmented copies, and
re-encodes). A model tested partly on images it effectively trained on will report inflated
scores. The audit removed every leaked pair and rebuilt the splits from scratch.

The consequence: honest test metrics are lower than the pre-fix numbers, and that is the point.

| Metric | Pre-fix (leaked / cherry-picked) | Honest (clean splits) |
|---|---|---|
| Segmentation Dice (headline) | 0.98 (leaked) / 0.859 (DFU-only subgroup) | 0.89 DFU-only, 0.93 median mixed |
| Classification accuracy | inflated | 0.984 |

The re-check on the rebuilt splits reports **zero** overlap on all four axes (path, canonical id,
content hash, perceptual near-duplicate). The reproducible procedure is in
[HPC_HONEST_RERUN_RUNBOOK.md](HPC_HONEST_RERUN_RUNBOOK.md).

### Mean vs median Dice: reading segmentation honestly

On the full mixed test set, mean Dice (~0.65–0.72) is far below median Dice (0.93). This is not
noise — it is the correct behavior of the metric on a mixed dataset:

- Healthy and non-DFU images have **empty** ground-truth masks.
- Dice on an empty mask is 0 if the model predicts even one false-positive pixel, and 1 only if
  it predicts nothing.
- A handful of such near-zero scores collapse the *mean* while leaving the *median* (dominated by
  real wounds) high.

So judge wound-outlining quality from the DFU-only mean (0.89) and the median (0.93); read the
mixed-set mean as "how often does it draw a wound where there is none," which the classifier
stage is there to suppress.

## Trade-offs, stated plainly

- **Cascade vs single model.** The cascade adds a failure mode (a classifier error stops the
  segmenter from running) in exchange for the ability to abstain on healthy skin. For a triage
  tool, abstention is worth more than end-to-end simplicity.
- **Coverage vs reliability.** The defer gate throws away ~6.5% of cases to buy 99.7% accuracy
  on the rest. Tunable per deployment.
- **Honest vs impressive.** The clean-split numbers are lower than what the leaked splits showed.
  We publish the lower numbers because the higher ones were measuring the wrong thing.

## Alternatives considered

- **Single end-to-end segmenter (v1).** Rejected: cannot triage, hallucinates wounds on healthy
  skin.
- **DFU-only headline metric.** Rejected as the *headline*: it flatters the model by hiding the
  empty-mask cases. Kept as a *subgroup* metric where it answers a real question ("how well does
  it outline actual wounds?").
- **Multiple segmentation backbones** (U-Net++, FUSegNet, nnU-Net, DINOv2+UPerNet). DINOv2+UPerNet
  is deployed; the others remain for ablation so architecture choices are evidence-based, not
  assumed. Comparison numbers live in `results/all_models_comparison.json`.

## Related

- [reference-architecture.md](reference-architecture.md) — the module map behind these stages
- [howto-train.md](howto-train.md) — how to train the models described here
- [../README.md](../README.md) — the honest results tables in full
