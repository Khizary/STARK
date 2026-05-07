# STARK: Scalable Top-k Adaptive Reduction Kernel

STARK is a federated embedding compression research project built on top of an OSCAR-style CLIP conditioning pipeline. The core research question it addresses is: can CLIP text embeddings be compressed mid-pipeline, before federated transmission, without degrading the downstream accuracy that depends on them?

The project's answer is yes, provided compression is defined over the right geometric quantity. STARK introduces **SCOUT** (Scalable Cosine-threshold OUtput Trimmer), an adaptive top-k selector that operates on individual token vectors within a CLIP embedding and retains the minimum number of dimensions needed to preserve the embedding's direction, rather than its magnitude, to within a user-specified angular tolerance θ. Because Stable Diffusion conditions image generation on CLIP embedding direction via cross-attention, cosine similarity is the correct objective. Energy retention, the conventional proxy, is indirect and suboptimal for this use case.

---

## Background and Motivation

### Federated learning and embedding bottlenecks

In federated learning, data does not leave its source node. Nodes instead share model gradients, synthetic representations, or in OSCAR-derived pipelines, CLIP embeddings of locally generated captions. OSCAR-style one-shot federated learning (as in OSCAR and its derivatives) uses BLIP captioning to generate descriptions of local data, encodes those descriptions through a CLIP text encoder, and transmits the resulting embeddings to a central server for Stable Diffusion synthesis of a shared synthetic dataset.

The CLIP encoder used here is `clip-vit-large-patch14`, which produces hidden states of shape `[N_captions, 77, 768]`, 77 token positions, each represented as a 768-dimensional vector. Transmitting these raw embeddings is a non-trivial cost in federated setups: 77 × 768 = 59, 136 floats per caption. STARK compresses this representation before transmission by zeroing out low-magnitude dimensions, reducing the non-zero payload while preserving the directional fidelity that Stable Diffusion actually depends on.

### Why cosine similarity, not energy

Standard compression methods for spectral or frequency data use energy thresholding: retain the top-k components that account for τ fraction of total signal energy. This is geometrically meaningful when the application depends on reconstruction fidelity in L2 norm. But Stable Diffusion's cross-attention mechanism operates on the direction of the CLIP embedding, not its magnitude, a scaled version of the same vector produces identical attention outputs. Angular distortion is therefore the correct metric.

Cosine similarity directly measures angular distortion: `cos_sim(original, compressed) = cos(θ)`, where θ is the angle between the two vectors in R^768. A cosine similarity of 0.99 corresponds to an angular deviation of approximately 8.1°, 0.999 to approximately 2.6°, and 0.95 to approximately 18.2°. SCOUT's threshold parameter is this cosine similarity bound, not an energy fraction. The energy curve is computed and visualised as supporting evidence, to confirm that CLIP embeddings are non-uniform in energy distribution, which is a prerequisite for compression being worthwhile at all, but it does not drive the selection criterion.

---

## The SCOUT Algorithm

SCOUT operates token-by-token. For each 768-dimensional token vector in a CLIP hidden state, it greedily adds dimensions in descending order of absolute magnitude until the cosine similarity between the current compressed vector and the original meets or exceeds the threshold θ. The result is the minimum k for that token.

Formally: given a token vector **v** ∈ R^768, let σ be the permutation that sorts dimensions by |v_i| descending. SCOUT finds

```
k* = min { k : cos_sim(v, v_k) ≥ θ }
```

where v_k is the sparse vector that retains only the top-k dimensions under σ and zeros the rest. This is computed greedily in a single pass and terminates as soon as the threshold is met. Zero-norm tokens (padding positions in the 77-token sequence) are handled as a special case: they are returned at full dimensionality k=768 without compression, since cosine similarity is undefined for zero vectors.

The key implementation detail is that selection is by absolute magnitude of the raw (non-squared) values. This gives a slightly different selection than energy-based sorting (which sorts by v_i²), but the greedy magnitude-first order tends to converge to the threshold in fewer steps because large-magnitude dimensions contribute disproportionately to the dot product in the numerator of the cosine similarity.

---

## Project Structure

The project is split into three notebooks with distinct responsibilities. They are designed to be run sequentially on Kaggle, with each notebook consuming outputs from the previous one.

### `Part-a.ipynb`, Baseline pipeline and embedding generation

This is the heavy pipeline notebook. It runs the full OSCAR-style preprocessing sequence: BLIP captioning of NICO++ class images, CLIP text encoding of the generated captions, and Stable Diffusion synthesis of a class-conditional synthetic dataset for downstream training. The CLIP embeddings are saved as `.npy` files per class to a persistent output directory for consumption by the Master notebook.

The notebook is intentionally pinned to a stable Stable Diffusion pipeline environment to ensure reproducibility of the embedding generation and synthesis steps. These steps are sensitive to diffusers version and tokenizer behaviour, and re-running them on a different library version can produce subtly different embeddings.

Expected runtime is approximately 5–6 hours on Kaggle T4 × 2 for the full pipeline. The primary outputs that matter for downstream analysis are the per-class embedding files saved to `embeddings/nico_unique/`, each of shape `[N_captions, 77, 768]`.

### `Part-b.ipynb`, Downstream ResNet training and evaluation

This notebook trains a ResNet classifier on the synthetic NICO++ dataset generated by Part-a and evaluates it at multiple fixed top-k compression levels. The fixed-k accuracy results it produces are the empirical ground truth that the Master notebook's SCOUT analysis is validated against.

The notebook is designed for the latest Kaggle training environment and fast iteration, no heavy generation pipeline is required at this stage. Expected runtime is under 1 hour on the same hardware.

The fixed-k accuracy results from this notebook feed directly into the Master notebook's config block as `FIXED_K_RESULTS`. The experimentally established values are:

| Top-k (dims retained per token) | Test accuracy (%) |
|---|---|
| 768 (full, uncompressed) | 73.0 |
| 512 | 71.0 |
| 128 | 71.0 |
| 64 | 66.3 |
| 32 | 52.1 |

The plateau between k=128 and k=512 is the critical finding from this notebook. Accuracy is essentially flat across a 4× range in embedding size (128 to 512 dimensions), which means there is a large compression window that does not sacrifice task performance. SCOUT's job is to land inside this window automatically.

### `Master.ipynb` (SCOUT analysis), Compression analysis and reporting

This notebook is the primary research artifact. It loads the saved CLIP embeddings from Part-a, confirms the non-uniformity of CLIP energy distribution, runs SCOUT at multiple θ values, records the resulting k distributions, verifies cosine similarity fidelity, and produces all figures and summary artefacts for the paper.

It does not re-run any synthesis or training. Total runtime is approximately 6–7 minutes on CPU-only Kaggle, including all plotting.

---

## Dataset

The analysis notebook uses NICO++ embeddings extracted from 12 object classes: bear, cat, chair, dog, flower, hat, kangaroo, lizard, motorcycle, scooter, shrimp, and train. Each class contributes 20 captions, yielding 240 total CLIP embeddings of shape `[77, 768]`, or 240 × 77 = 18, 480 individual token vectors used in the compression analysis.

A secondary analysis is included for a DomainNet subset, allowing the SCOUT threshold to be validated for cross-dataset transferability. The energy curve comparison between NICO++ and DomainNet shows near-identical shape, indicating that a threshold calibrated on NICO++ is likely to generalise.

---

## Experimental Results

### Energy curve

The cumulative energy curve averaged across all 18, 480 token vectors is strongly concave: a small number of top-magnitude dimensions account for a large fraction of total L2 energy. This confirms that the embedding space is non-uniform and that aggressive compression is geometrically feasible. The 10th–90th percentile band is narrow, indicating this property is consistent across token positions and caption types, not an artifact of averaging.

### SCOUT k distributions (θ sweep)

The following results are from the SCOUT sweep on the `bear` class embedding (20 captions), used as the representative sample. The dynamic k is the mean selected across all 77 token positions per caption.

| θ | Angular tolerance | Mean k selected | Compression | Achieved cos sim |
|---|---|---|---|---|
| 0.999 | ≤ 2.6° | 500.9 | 34.8% | 0.9991 |
| 0.990 | ≤ 8.1° | 437.8 | 43.0% | 0.9901 |
| 0.950 | ≤ 18.2° | 305.8 | 60.2% | 0.9507 |
| 0.900 | ≤ 25.8° | 220.6 | 71.3% | 0.9015 |

For comparison, the fixed-k baselines achieve:

| Fixed k | Achieved cos sim | Compression |
|---|---|---|
| 512 | 1.0000 | 33.3% |
| 256 | 0.9213 | 66.7% |
| 128 | 0.7977 | 83.3% |
| 64 | 0.6678 | 91.7% |
| 32 | 0.5498 | 95.8% |

### Key finding

At θ=0.99, SCOUT selects a mean k of approximately 438, which falls inside the accuracy plateau [128, 512] with confirmed cosine similarity of 0.9901, a maximum angular deviation of 8.1°. This means a single threshold value automatically produces compression that is directionally faithful enough to preserve SD conditioning fidelity, without requiring any knowledge of the downstream accuracy curve.

The practical implication for federated learning is that nodes can compress their CLIP embeddings to roughly 57% of their original dimensionality (438/768) before transmission, at an angular cost below 8.1° per token, while staying within the region of the k-accuracy curve that gives plateau-level accuracy.

---

## Output Artefacts

The Master notebook saves the following files to `OUTPUT_DIR` (`/kaggle/working/scout_analysis/`):

| File | Contents |
|---|---|
| `energy_curve.png` | Cumulative energy curve averaged over all NICO++ tokens, with percentile band and cosine threshold horizontal reference lines |
| `k_distribution_cosine.png` | Histogram of k values selected by SCOUT at each θ, showing per-token variation around the mean |
| `scout_main_result.png` | Fixed-k accuracy curve from Part-b overlaid with vertical lines showing where each θ places its mean k, including ± 1 std band and the 70–73% plateau shading |
| `scout_summary.csv` | Per-θ summary table: mean k, median k, std, min, max, compression %, plateau status |
| `scout_final_summary.json` | Machine-readable report including the plateau range definition, full per-θ summary, and the recommended θ=0.99 operating point |

---

## How to Run

1. Run `Part-a.ipynb` on Kaggle (T4 × 2, ~5–6 hours) to generate BLIP captions, extract CLIP embeddings, and produce the synthetic NICO++ training dataset. Note the output path for the `embeddings/nico_unique/` directory.
2. Run `Part-b.ipynb` to train the ResNet downstream classifier on the synthetic data and record fixed-k accuracy results.
3. Update the `EMB_DIR` and `FIXED_K_RESULTS` config block in `Master.ipynb` to point to your Part-a embedding output path and your Part-b accuracy numbers.
4. Run `Master.ipynb` for SCOUT analysis and full report generation.

---

## How STARK Differs from OSCAR

The original OSCAR paper is about object-aware image-text grounding and pre-training: it uses object tags as anchors during pre-training to align visual and textual representations. This project uses OSCAR conceptually as a captioning-and-conditioning pipeline, it generates image captions via BLIP and encodes them through CLIP to produce conditioning embeddings for Stable Diffusion synthesis, but the novel contribution is entirely post-encoding.

STARK/SCOUT intervenes at the embedding level, after OSCAR-style preprocessing has produced CLIP hidden states, and before those embeddings are transmitted or used for synthesis. The contribution is: (1) the cosine-threshold framing for compression, replacing energy-based thresholding; (2) the demonstration that a single θ value can reliably select k inside the accuracy plateau without per-dataset tuning; and (3) the federated interpretation, where compression ratio directly maps to a reduction in per-node communication cost.

---

## Notes for Reviewers

The three-notebook structure is deliberate. Separating the heavy synthesis step (Part-a) from the downstream training (Part-b) and the compression analysis (Master) makes the pipeline reproducible at each stage independently. A reviewer can rerun only the Master notebook in a few minutes to verify all compression claims without re-triggering the 5-hour SD pipeline.

The recommended operating point θ=0.99 is conservative relative to what the plateau data would permit, k=128 also achieves plateau accuracy, implying even θ=0.90 (mean k≈221) would likely be fine in practice. The 0.99 recommendation reflects a design choice to stay well inside the plateau and maintain a visually convincing cosine similarity value for the paper.

The DomainNet secondary analysis is included to show that the energy curve shape, and therefore the threshold calibration, is not dataset-specific. The near-identical curve profiles for NICO++ and DomainNet support the claim that SCOUT thresholds transfer across domains without retuning.
