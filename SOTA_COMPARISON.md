# Common-Protocol SOTA Comparison

## Protocol

- BOSSbase thresholded at 128.
- 100 training images and 100 disjoint test images.
- Identical messages at 64, 128, 256, and 512 bits for every method.
- Exact restoration, serialized auxiliary bits, DRD, runtime, and availability.
- Dong searches `l=1,...,10` and retains the exact minimum-DRD result.
- Statistical tests use paired images and Wilcoxon with Holm correction.

Huynh--Nguyen uses the published `T=5` and reproduces the eight published
reference capacities. PPOCP and Dong use conservative serialized
synchronization data where the papers do not define an independent wire format.

## Matched-Image Curves

| Payload | Matched | ABM net / DRD | Dong net / DRD | Huynh net / DRD | PPOCP net / DRD |
|---:|---:|---:|---:|---:|---:|
| 64 | 86 | **-57 / 0.041** | -224 / 0.056 | -2200 / 0.035 | -997 / **0.023** |
| 128 | 83 | **-7 / 0.080** | -176 / 0.108 | -2096 / 0.070 | -1653 / **0.041** |
| 256 | 69 | **93 / 0.135** | -81 / 0.200 | -1812 / 0.117 | -2856 / **0.064** |
| 512 | 46 | **309 / 0.176** | 96 / 0.332 | -1191 / 0.178 | -4822 / **0.097** |

Values are mean net bits and median DRD. ABM has the only positive signed
trapezoidal net-bits/DRD area (`9.41`); the area is descriptive because methods
span different DRD ranges.

ABM's paired net advantage is significant against every competitor at every
payload after Holm correction (`p < 2.2e-12` at 256 bits). PPOCP and
Huynh--Nguyen retain significantly lower DRD than ABM. ABM has significantly
lower DRD than Dong at all four payloads.

## Binarization

An image is classed as suitable when threshold 128 agrees with Otsu on at least
95% of pixels and the foreground fraction lies in `[0.05, 0.95]`. This selects
22/100 images.

Suitability improves availability: ABM is available on all 22 suitable images
through 256 bits, versus 60/78 unsuitable images at 256 bits. It does not
explain away the net result: ABM remains net positive in both strata at 256 and
512 bits.

## Runtime and Memory

At 256 bits on ten protocol images:

| Method | Median time | Median incremental peak RSS |
|---|---:|---:|
| PPOCP | **0.16 s** | 0.70 MiB |
| Huynh--Nguyen | 1.51 s | **0.02 MiB** |
| ABM | 3.56 s | 2.56 MiB |
| Dong (`l=1,...,10`) | 16.97 s | 0.05 MiB |

RSS is the sampled process peak minus the pre-call RSS, not total memory.

## Steganalysis

Grouped five-fold validation keeps each cover/stego pair in one fold. Features
are foreground and transition rates plus 2x2 and 3x3 pattern histograms.

| Payload | ABM AUC | Dong AUC | Huynh AUC | PPOCP AUC |
|---:|---:|---:|---:|---:|
| 64 | 0.706 | 0.656 | **0.599** | 0.653 |
| 128 | 0.826 | 0.759 | 0.688 | **0.654** |
| 256 | 0.900 | 0.784 | 0.774 | **0.691** |
| 512 | 0.957 | 0.896 | 0.909 | **0.690** |

These are logistic-detector AUC values. Random Forest was also evaluated; for
ABM its AUC is 0.623, 0.691, 0.769, and 0.851 respectively. It is therefore not
excluded, but the linear detector is stronger on this feature space.

## Positioning

ABM is the strongest evaluated method for **net payload under exact
reversibility**. It is the only method with positive mean net payload at
256 bits and remains substantially ahead at 512 bits.

This is not a universal SOTA result. PPOCP remains best in median DRD and
steganalysis resistance. ABM's current practical bottleneck is detectability,
not capacity, reversibility, runtime, or memory. A global SOTA claim would
require reducing the statistical footprint while retaining the positive-net
advantage.

Raw results and analysis are in `sota_multiload/`.
