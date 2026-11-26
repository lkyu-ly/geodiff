# GeoDiff QM9 and Drugs Implementation Plan

> **For agentic workers:** Use test-driven-development and the existing isolated `feat/geodiff` worktree. Do not commit, push, upload or open a PR in PaddleMaterials without the user's explicit authorization.

**Goal:** Support both released author checkpoints and the corresponding complete conformer split packages, verified from actual local archives. For Drugs, run loading and short training/sampling smoke only, not cross-framework accuracy alignment.

**Architecture:** Retain one `GeoDiff` model class; use separate self-contained QM9 and Drugs YAML plus four-name weight packages, like existing PaddleMaterials variants. Both sets use the same GEOM conformer data loader and collator. Use the real PaddleMaterials download and package helpers with local ZIP sources in disposable verification, not a fake URL in the production registry. Preserve existing model math and native state layout.

**Spec:** User-approved dual default pretrained resources; [original README](../../geodiff/README.md#L24-L64), [original Drugs configuration](https://github.com/MinkaiXu/GeoDiff/blob/pretrain/configs/drugs_default.yml), [dual-preset research](../../docs/GeoDiff_QM9与Drugs双预设调研.md), and [original integration spec](../geodiff-ppmat/spec.md) for stable training and sampling interfaces. The separate QM9 property-evaluation assets are outside the conformer splits unless separately requested.

## Constraints

- Keep the approved model core, optimizers, graph and sampling formulas unchanged. No ongoing Torch/PyG dependency; author conversion/export only in the existing offline environment.
- Use local archives before freeze; no real URL needed. No permanent loopback/placeholder registry entries. Preserve previous validated package snapshots when rebuilding container layout.
- `GeoDiff` + `GeomConformerDataset` + `GeomConformerCollator` shared by both presets. Named recipes `geodiff_qm9_default` and `geodiff_drugs_default` retain separate data directories and author defaults. Drugs: `smooth_conv: true`, batch size 32, joint clip 30000, seed 2021, 5000 diffusion steps and a 10,000,000-step training cap; QM9 defaults remain unchanged.

## Tasks

1. Verify author references, actual original/converted tensor payload and full QM9 member counts. Convert `checkpoints/drugs_default.pt` with `tools/convert_released_weights.py` to `runs/paddle_drugs/drugs_default.pdparams`; require the same 198-entry conversion validation as QM9.
2. Add a Drugs case to the existing real sampling CLI test. Observe the missing recipe red. Rename the shared reader/collator and all config callers in one cutover; add the Drugs recipe alongside QM9. Verify updated graph collation and existing QM9 plus new Drugs SDF behavior in the narrow suite tests.
3. Pilot the actual layout: create local weight bundle and an explicitly smoke-only one-graph-per-split Drugs data archive. Include the top-level directory entry before file members; reproduce the prior returned-root problem and verify standard extraction on corrected local copies. Run one training update/validation and two sampling steps on actual Drugs data and native weights. No numerical precision comparison.
4. Export the full author Drugs train/val/test once after the pilot. With the same existing complete QM9 NPZ bytes, rebuild the enclosing archives as needed for root-directory lookup and shared reader config. Verify local downloads/checksum, package-root lookup, and actual loaded train/sampling data on the final containers at small batch size, without running full validation or sampling. Keep prior QM9 ZIP copies under ignored `runs/`; keep all four verified archives outside Git. Formal link registration stays deferred.
