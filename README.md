# University research project · Machine Learning Security 

**Few-shot detection of steganographic modifications in neural-network weights: replication, out-of-distribution evaluation, and matched controls.**

**Project contributors:** 
Riccardo D'Annibale and Fabrizio Giuseppe Mazzilli 


> **Main finding.** On a controlled zoo of 1,000 small CNN checkpoints, a Model X-Ray Siamese detector trained on X-LSB detects the known family well (**centroid AUROC ≈ 0.904**), but transfers poorly to our *synthetic 16-bit spread-spectrum* embedding (**AUROC 0.478 on Ubuntu; 0.475 on Mac M4**). This is an **internal attack-family-shift experiment**, not evidence that real malware is universally undetectable.

![E2: centroid AUROC under known and unseen embedding families](results/e2_final/plots/e2_test_id_vs_ood_centroid.png)

## Research question

> Can a few-shot detector of hidden information in neural-network weights reproduce measurable in-distribution performance, and how does it behave when confronted with an unseen embedding family, under controlled checkpoint-level splits and paired checks?

Our objective is **evaluation**, not to build or execute malware. All payloads are synthetic and inert. The CNNs are original models from the GHRP/STL10 model zoo; their numerical weights are used as the analysis unit.

## At a glance

| Experiment | Design | Key result | Evidence |
|---|---|---|---|
| **E1 — ID replication** | Siamese trained/tested on X-LSB, disjoint source checkpoints; X ∈ {1, 8, 16, 23}. | For **X=23**, centroid test AUROC: **0.901863 Ubuntu**, **0.904378 Mac M4**. | mettere sezione report finale|
| **E2 — unseen embedding family** | Train/calibrate using **X-LSB X=23**; evaluate the *same detector per fit* on ID X-LSB versus OOD synthetic spread-spectrum α ∈ {0.24, 0.16}. | At **α=0.24**, centroid test AUROC: **0.477620 Ubuntu**, **0.474500 Mac**; paired ΔAUROC (OOD−ID): **−0.426653 / −0.429800**. | mettere sezione report finale |
| **E3 — retrospective controls** | Verify 1:1 benign/stego checkpoint pairing, balance of recorded identities, unchanged benign scores/FPR, and TPR/FNR across families. | At α=0.24, centroid **TPR 0.1815 Ubuntu / 0.1785 Mac** at frozen thresholds; benign FPR unchanged across conditions on each platform. | mettere sezione report finale |

All AUROC entries above are **means over 10 fits per platform**, not estimates from 10 independent test datasets. Both platforms reuse the same **200 distinct test checkpoint identities**; the repetition axis changes fit seeds/few-shot training.

## What we actually tested

**Models and data.** 1,000 original small CNN checkpoints, **2,864 float32 parameters each**, from a GHRP/STL10 zoo. Frozen splits: **600 source checkpoint identities for the training pool, 200 for validation, 200 for test**. Each fit trains from only **three benign/stego pairs (six examples)**. The same source checkpoint never crosses these splits.

**Detector.** Model X-Ray's few-shot Siamese pipeline; weights are transformed into 100×100 GF representations. A learned embedding is scored by **centroid** (primary) or **1-NN** (secondary) class-distance margin. Youden thresholds are calibrated **only on X-LSB validation** and transferred unchanged to the test conditions.

**Attacks.** E1 uses canonical **X-LSB** modifications. E2 introduces a **new, custom, keyed 16-bit spread-spectrum-inspired embedding**, evaluated only as OOD. It is **inspired by principles discussed in MaleficNet**, but is **not the original MaleficNet implementation**. No active malware or executable payloads are involved.

**Platform replication.** Ubuntu 24.04 in VirtualBox (x86_64, CPU) and macOS on Apple M4 (ARM64, CPU). Hardware differs; **CNN architecture, source zoo, experiment logic, and checkpoint identities do not**.

## Important limitations

- **Not a causal family-only comparison:** X-LSB X=23 and spread α=0.24/0.16 differ in perturbation magnitude **and** payload capacity. The pilot found mean relative L2 perturbation ≈0.31 for X-LSB versus 0.24/0.16 for spread.
- **Not original MaleficNet and not real-malware detection:** only inert bytes and a custom 16-bit synthetic embedding were studied.
- **Not an external untouched holdout:** checkpoint identities are disjoint within the zoo, but the same zoo was explored earlier; E3 reuses E2 test predictions retrospectively.
- **Not a universal negative result:** the original Model X-Ray paper also discusses successful generalization to other spread-spectrum settings. The protocols, models, and attacks are not necessarily comparable.
- **No bitwise TF reproducibility guarantee:** Ubuntu/Mac results agree closely numerically, not necessarily bit for bit.
- **Published summaries are not the complete raw archive:** full prediction CSVs and model-zoo checkpoint files are retained locally, subject to dataset licensing and permissions.

For a detailed discussion, see the [integrated technical report](exam/PROJECT_REPORT.md).

## Repository map

```text
.
├── README.md                        # Project overview and results
├── REPRODUCING.md                   # How to inspect / reproduce experiments
├── external/ModelXRay/              # Upstream Git submodule; its own terms apply
├── manifests/e1_uniform_v1/         # Frozen checkpoint split and protocol
├── scripts/                         # Provenance, original runners and E2/E3 audit tools
├── tests/                           # Integrity / regression checks
├── configs/                         # Environment snapshots where available
├── results/
│   ├── figures/e1/                 # E1 figures
│   ├── e2_final/                   # E2 tables, Mac replica, figures
│   └── e3_final/                   # E3 controls, platform summaries, figure
├── docs/                            # Method, frozen protocols, audit notes, licensing
└── exam/
    ├── PROJECT_REPORT.md           # Full E1+E2+E3 technical report
    └── PROJECT_REPORT.docx         # Editable report
```

## Reproducing and verifying results

**Fast, no-training check (Python standard library only):**

```bash
python3 scripts/verify_published_results.py
```

This cross-checks E2 paired AUROC rows against the published summaries and E3 summary structure / cross-family FPR controls; it **does not recompute AUROC from raw predictions**. Its existence after applying the accompanying portfolio package is a convenience layer on top of the frozen audits.

**Read-only repository inventory:**

```bash
python3 scripts/audit_repository.py
```

**Full experimental replication** requires the original checkpoint zoo, fixed environment, [upstream Model X-Ray](https://github.com/danigil/ModelXRay) at pinned commit, dataset hashes and the full E1/E2 runner commands. Start with **[REPRODUCING.md](REPRODUCING.md)** rather than running test-phase scripts blindly. **E2 test execution is approval-gated and should not be re-run to select favorable outcomes.**

### Dataset identity and checksums

| Object | SHA-256 |
|---|---|
| `weights.npy` (1000 × 2864 float32) | `5a03ff6349010405fd05c7c27aee746143259fdf0a000acb4c9e05cbc8dd60ad` |
| `manifests/e1_uniform_v1/splits.csv` | `3f4236bf2b3e7a91e11dd67037c5dc3262382fd8c77ce333b49f30d6e4b93a6f` |
| Model X-Ray upstream commit | `a880e332f556fc71faf8d3753931005d5d019339` |
| E2 detector runner `scripts/run_e2_paired_v0.py` | `4719e6d947c5c5f7acba09f464ec0d2e5abf5182c642af8a063789ca217849ac` |

The zoo is described in the [dataset/provenance notes](docs/E1_REPRODUCIBILITY.md). Original large checkpoints are not bundled with this project.


