# Visualizations and math notes

This directory logs a set of TensorBoard figures and a few optional panels for analysis.
Most of them are implemented in `aux.py` under `ImageViz` and are called from the
`tb_*` helpers.

## Per-MLP analytics

- Tag: `per_mlp/accuracy_heatmap`
- What it shows: a heatmap of per-MLP EMA accuracy over time snapshots.
- Math:
  - For class k, the per-micro-step accuracy is `acc_k = mean(pred == k)`.
  - Exponential moving average (EMA): `ema_k <- decay * ema_k + (1 - decay) * acc_k`.
  - The heatmap stacks recent EMA snapshots (rows = k, columns = snapshot index).

- Tag: `classifier/confusion_matrix`
- What it shows: a row-normalized confusion matrix of predicted class vs true class.
- Math:
  - Confusion counts are accumulated as `conf[true_k, pred_k] += 1`.
  - Visualization uses row normalization: `p(pred | true) = conf / row_sum`.

## Pairwise distance + OT panel

- Tag: `pairwise/dist_ot_panel`
- What it shows (2x2):
  - Heatmap of average pairwise L2 distances between class point clouds.
  - Heatmap of Wasserstein-2 distances (OT).
  - Mean W1 over time.
  - Mean W2 over time.
- Math:
  - For each class pair (i, j), average distance is `mean(||x_i - x_j||_2)`.
  - W1 uses OT cost with L2 ground metric.
  - W2 uses OT cost with squared L2 ground metric, then `W2 = sqrt(cost)`.
  - W1/W2 are computed via Sinkhorn or EMD (POT).

## KL free-energy panel (available, optional)

- Function: `ImageViz.plot_pairwise_kl_free_energy_panel(...)`
- What it shows (2x2):
  - Heatmap of KL_eps between entropically smoothed point clouds.
  - Heatmap of ratio `KL_eps / W2_eps`.
  - Mean KL_eps over time.
  - Mean ratio over time.
- Math:
  - Inputs are expected to be computed outside (KL and W2 at the same epsilon).
  - Ratio uses `KL_eps / max(W2_eps, eps)` for stability.

## Information + confusion panel

- Tag: `info/info_matrix_panel`
- What it shows (2x2):
  - Heatmap of information matrix Lambda.
  - Heatmap of row-normalized confusion `p(pred | true)`.
  - Q(Lambda) and Q(Σ^T Σ) over time.
  - Effective dimension (normalized) over time.
- Math:
  - Per-class step vectors are normalized: `u_{b,k} = delta_{b,k} / ||delta_{b,k}||`.
  - Information matrix: `Lambda_ij = E_b[(u_{b,i} dot u_{b,j})^2]`.
  - Confusion uses reconstructor predictions with the same uv transform as training.
  - Stochasticity:
    - Confusion Σ is row-stochastic by construction (rows sum to 1 when counts exist).
    - Σ is generally not column-stochastic, so it is not doubly stochastic.
    - Λ is symmetric and nonnegative but not stochastic.
  - Invariants:
    - `Q(A) = det(A)^2 / det(A .* A)` with small diagonal regularization.
    - `d_eff(A) = (tr A)^2 / tr(A @ A)` and then normalized by K.
    - For a comparable, symmetric PSD confusion statistic, we use `G = Σ^T Σ` in the
      bottom plots (Q and d_eff) rather than Σ directly.

## Latent path projections

- Tag: `paths/spectral_projection`
- What it shows: trajectories projected to the top-2 PCA directions.
- Math:
  - Build trajectories `z(t, k)` from a single starting point via model rollout.
  - Subtract the starting point, stack all points, run SVD/PCA, project onto
    the first two components.

- Tag: `paths/sector_projection`
- What it shows: trajectories projected into K angular sectors.
- Math:
  - Each class k has a base angle `theta_k = 2*pi*k/K`.
  - The radial coordinate is projection onto a "principal gradient" direction g
    (mean final displacement).
  - The tangential offset within a sector is projection onto a dominant variance
    direction v that is orthogonal to g.

## Manifold projections (final points)

- Tag: `manifold/final_points_embeddings`
- What it shows: a 1x3 panel of t-SNE, Isomap, and SpectralEmbedding on final points.
- Math/behavior:
  - Points are the final latent positions per class (B per class).
  - t-SNE uses PCA init on each run and is then aligned to the previous embedding.
  - If t-SNE becomes degenerate (near-constant embedding), it falls back to PCA.
  - A fixed batch can be cached for this panel so each step embeds the same
    initial points (set `embedding_use_fixed_batch` in training).
  - All methods are aligned to the previous embedding using a Procrustes
    transform (rotation + scale) for visual comparability across steps.
  - Isomap/Spectral use an adaptive kNN size to keep the graph connected; if the
    graph is still disconnected, the plot falls back to a PCA projection.

## Image grids (triplet + diffs)

- Tag: `images/triplet`
- What it shows: a grid of reference, step1, and step2 images.
- Tag: `images/diff_triplet_abs`
- What it shows: absolute differences (step1 - ref) and (step2 - step1).
- Math:
  - These are direct pixel-space differences for quick qualitative debugging.

## Histograms

- Tag: `train/logits`
- What it shows: histogram of classifier logits per step.

- Tag: `potential_distribution/k`
- What it shows: per-class histogram of potential predictions.
