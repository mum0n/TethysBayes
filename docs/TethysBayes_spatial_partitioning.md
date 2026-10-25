---
title: "TethysBayes Spatial and Spatiotemporal Partitioning: Methodologies, Mathematics, and Practical Guide"
format: html
---

# TethysBayes Spatial and Spatiotemporal Partitioning: Methodologies, Mathematics, and Practical Guide

## 1. Introduction & Theoretical Motivation

In Bayesian spatiotemporal modeling, spatial processes are typically categorized into two primary paradigms:
1. **Continuous Spatial Fields (Geostatistics / Point Processes)**: Modeled via Gaussian Processes (GPs), Stochastic Partial Differential Equations (SPDEs), or Random Fourier Features (RFF).
2. **Areal / Discrete Unit Models (Lattice / Disease Mapping)**: Modeled via Intrinsic Autoregressive (ICAR), Besag-York-Mollié (BYM2), Leroux, or Simultaneous Autoregressive (SAR) specifications.

While real-world observational data often arrives as continuous geographic coordinates $(s_x, s_y)$ or point observations (e.g., GPS telemetry, sensor networks, epidemiological case reports, ecological surveys), fitting full continuous Gaussian Processes scales with cubic complexity $\mathcal{O}(N^3)$ in the number of observations $N$. Areal GMRF models (like BYM2 and ICAR), in contrast, scale efficiently as $\mathcal{O}(S^{1.5})$ to $\mathcal{O}(S)$ where $S$ is the number of discrete spatial units.

The **`TethysBayes` Partitioning Subsystem** provides an extensible suite of algorithms to discretize continuous 2D spatial domains into discrete areal units $\Omega = \bigcup_{i=1}^S A_i$, construct topological neighborhood adjacency matrices $W \in \mathbb{R}^{S \times S}$, enforce graph connectivity, scale precision matrices for BYM2 models, discretize temporal dimensions, and perform spatial cross-validation.

```
                               ┌────────────────────────────────────────┐
                               │  Continuous Observations (s_x, s_y, t)  │
                               └───────────────────┬────────────────────┘
                                                   │
                   ┌───────────────────────────────┴───────────────────────────────┐
                   ▼                                                               ▼
     ┌────────────────────────────┐                                  ┌────────────────────────────┐
     │   Spatial Partitioning     │                                  │   Temporal Discretization  │
     │  (:avt, :cvt, :hex, etc.)  │                                  │  (Quantile, Jenks, KMeans) │
     └─────────────┬──────────────┘                                  └─────────────┬──────────────┘
                   │                                                               │
                   ▼                                                               ▼
     ┌────────────────────────────┐                                  ┌────────────────────────────┐
     │  Areal Units & Graph (W)   │                                  │     Time Units (t_idx)     │
     │  Centroids & Polygons      │                                  │     Breaks & Midpoints     │
     └─────────────┬──────────────┘                                  └─────────────┬──────────────┘
                   │                                                               │
                   └───────────────────────────────┬───────────────────────────────┘
                                                   │
                                                   ▼
                                 ┌───────────────────────────────────┐
                                 │  Spatiotemporal Indexing (st_idx) │
                                 │  BYM2 Spectral Scaling Factor (s) │
                                 │  Spatial Block Cross-Validation   │
                                 └───────────────────────────────────┘
```

---

## 2. Mathematical Formulations & Tessellation Algorithms

### 2.1. Voronoi Tessellation Fundamentals

Given a bounded 2D domain $\Omega \subset \mathbb{R}^2$ and a set of $S$ distinct generator seeds $C = \{c_1, c_2, \dots, c_S\} \subset \Omega$, the Voronoi cell $V_i$ corresponding to generator $c_i$ is defined as:

$$
V_i = \left\{ x \in \Omega \;\middle|\; \|x - c_i\|_2 \le \|x - c_j\|_2, \quad \forall j \ne i \right\}
$$

The collection $\mathcal{V}(C) = \{V_1, V_2, \dots, V_S\}$ forms a convex polygonal partition of $\Omega$. The Delaunay triangulation $\mathcal{D}(C)$ is the planar geometric dual of $\mathcal{V}(C)$: two seeds $c_i$ and $c_j$ share an edge in $\mathcal{D}(C)$ if and only if their Voronoi cells share a boundary segment $\partial V_i \cap \partial V_j \ne \emptyset$.

In `TethysBayes`, all Voronoi polygons are computed via Delaunay triangulation (`DelaunayTriangulation.jl`) and clipped against the domain's convex or buffered hull geometry using `LibGEOS.jl`.

---

### 2.2. Centroidal Voronoi Tessellation (`:cvt`)

A Centroidal Voronoi Tessellation (CVT) is a Voronoi diagram where each generator seed $c_i$ coincides exactly with the geometric center of mass (centroid) $c_i^*$ of its associated Voronoi region $V_i$:

$$
c_i^* = \frac{\int_{V_i} x \, \rho(x) \, dx}{\int_{V_i} \rho(x) \, dx}
$$

When the density function $\rho(x) \equiv 1$ (uniform density), $c_i^*$ is the geometric centroid. CVTs minimize the spatial quantization energy:

$$
\mathcal{E}(C, \mathcal{V}) = \sum_{i=1}^S \int_{V_i} \|x - c_i\|^2 \, dx
$$

#### Algorithm (Lloyd's Relaxation)
1. **Initialize**: Sample $S$ initial seeds $C^{(0)} = \{c_1^{(0)}, \dots, c_S^{(0)}\}$ from the data extent using density-weighted KD-Tree sampling.
2. **Voronoi Construction**: Construct $\mathcal{V}(C^{(t)})$ clipped to $\Omega$.
3. **Centroid Update**: Compute geometric centroids $c_i^{(t+1)} = \text{centroid}(V_i^{(t)})$.
4. **Convergence Check**: Compute displacement $\delta = \frac{1}{S} \sum_{i=1}^S \|c_i^{(t+1)} - c_i^{(t)}\|$. Terminate if $\delta < \text{tolerance}$ or upon reaching `max_iter`.

*Statistical Properties*: Generates uniform, isotropic, hexagonal-like cells that minimize boundary irregularity across the domain.

---

### 2.3. K-Means Density-Balanced Voronoi Tessellation (`:kvt`)

Unlike CVT which moves centroids toward the geometric center of the polygon, K-Means Voronoi Tessellation (KVT) moves seeds toward the empirical mean of the observational data points residing within each Voronoi cell:

$$
c_i^{(t+1)} = (1 - \gamma) c_i^{(t)} + \gamma \left( \frac{1}{|N_i|} \sum_{x_j \in N_i} x_j \right)
$$

where:
- $N_i = \{x_j \in \text{Data} \mid \arg\min_k \|x_j - c_k\| = i\}$ is the set of points assigned to cell $i$.
- $\gamma \in (0, 1]$ is a relaxation damping factor (default: $\gamma_0 = 0.7$, decaying by $0.99$ per iteration).

*Statistical Properties*: Clusters smaller, high-resolution polygons in regions with dense observations while expanding larger polygons in sparse regions, keeping point counts per cell balanced.

---

### 2.4. Quadtree Voronoi Tessellation (`:qvt`)

QVT is a top-down hierarchical space-partitioning method. Starting with the entire dataset in a single region $\Omega$, it recursively subdivides regions into four quadrants based on bivariate medians:

$$
\tilde{x} = \text{median}(\{x_j \mid x_j \in R\}), \quad \tilde{y} = \text{median}(\{y_j \mid y_j \in R\})
$$

Regions are split into:

$$
R_{1} = \{p \in R \mid p_x \le \tilde{x}, p_y \le \tilde{y}\}, \quad R_{2} = \{p \in R \mid p_x > \tilde{x}, p_y \le \tilde{y}\}
$$

$$
R_{3} = \{p \in R \mid p_x \le \tilde{x}, p_y > \tilde{y}\}, \quad R_{4} = \{p \in R \mid p_x > \tilde{x}, p_y > \tilde{y}\}
$$

*Splitting Criteria*: Subdivisions continue on regions exceeding `min_points` until the target unit count `target_units` or coefficient of variation (CV) threshold is achieved.

---

### 2.5. Binary Voronoi Tessellation (`:bvt`)

Binary Voronoi Tessellation recursively bisects regions along their principal axis of maximum variance:

$$
d^* = \arg\max_{d \in \{x, y\}} \operatorname{Var}(p_d \mid p \in R)
$$

The region is bisected along the median of dimension $d^*$:

$$
R_{\text{left}} = \{p \in R \mid p_{d^*} \le \text{median}(p_{d^*})\}, \quad R_{\text{right}} = \{p \in R \mid p_{d^*} > \text{median}(p_{d^*})\}
$$

*Statistical Properties*: Fast, robust to anisotropic coordinate scaling, and naturally adapts to elongated spatial tracks or linear features (e.g., river networks, coastlines).

---

### 2.6. Hierarchical Voronoi Tessellation (`:hvt`)

HVT combines $k$-means spatial initialization with iterative Lloyd refinement and adaptive cell splitting. If a Voronoi cell contains more than `max_points` observations after centroid stabilization, the cell is split into two child seeds placed along its local dispersion axis:

$$
c_{\text{new}, 1} = \bar{x}_{\text{cluster}} \times 0.99, \quad c_{\text{new}, 2} = \bar{x}_{\text{cluster}} \times 1.01
$$

---

### 2.7. Agglomerative Voronoi Tessellation (`:avt`)

AVT is a bottom-up hierarchical clustering method. It begins in an over-partitioned state ($S_{\text{init}} = 2 \cdot \text{target\_units}$) and iteratively merges the smallest, underpopulated, or constraint-violating polygon into its nearest neighbor:

$$
c_{\text{merged}} = \frac{n_i c_i + n_j c_j}{n_i + n_j}
$$

where $j = \arg\min_{k \ne i} \|c_i - c_k\|^2$. Merging continues until all remaining units satisfy `min_points`, `min_time_slices`, and `min_area` constraints.

---

### 2.8. Hexagonal Grid Tessellation (`:hexagonal` / `:hexbin`)

Hexagonal tessellations partition 2D space into regular honeycombs. Hexagons have optimal geometric properties:
- **Maximum Area-to-Perimeter Ratio**: Minimizes boundary edge distortion.
- **Isotropic Neighborhood**: Every interior hexagon shares equidistant boundaries with exactly 6 neighbors, eliminating the diagonal vs. orthogonal distance asymmetry of square grids.

```
           (x, y + R)
          /          \
  (x - dx/2, y + R/2) (x + dx/2, y + R/2)
        |              |
        |   (x, y)     |  Radius: R
        |              |  Width:  dx = √3 · R
  (x - dx/2, y - R/2) (x + dx/2, y - R/2)
          \          /    Height: dy = 1.5 · R
           (x, y - R)
```

#### Geometry & Formulas:
For a regular hexagon with radius $R$ (center to vertex):
- Area: $A_{\text{hex}} = \frac{3\sqrt{3}}{2} R^2 \approx 2.598 R^2$
- Target Radius from Target Area: $R = \sqrt{\frac{2 A_{\text{target}}}{3\sqrt{3}}}$
- Horizontal column spacing: $\Delta x = \sqrt{3} R$
- Vertical row spacing: $\Delta y = \frac{3}{2} R$
- Row Offset: Odd rows are shifted horizontally by $\frac{\sqrt{3}}{2} R$.

Vertices for cell centered at $(x_c, y_c)$:

$$
v_k = \left( x_c + R \cos\left(\frac{\pi}{6} + \frac{k\pi}{3}\right), \; y_c + R \sin\left(\frac{\pi}{6} + \frac{k\pi}{3}\right) \right), \quad k = 0, \dots, 5
$$

---

### 2.9. Regular Lattice Grid (`:lattice`)

Square or rectangular raster grid partitioning. Given bounding box $[x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}]$:
- Cell width $L_x = \frac{x_{\max} - x_{\min}}{\text{cols}}$
- Cell height $L_y = L_x \times \text{aspect\_ratio}$

`TethysBayes` implements analytical $\mathcal{O}(N)$ stencil generation for grid adjacency matrices without LibGEOS geometric intersection overhead.

---

### 2.10. Force-Directed Inferred Spatial Layout (`assign_spatial_units_inferred`)

When observational data contains an adjacency matrix $W$ but lacks geographic coordinates $(s_x, s_y)$ (such as standard regional benchmark datasets like the Scottish Lip Cancer data), `assign_spatial_units_inferred` reconstructs continuous coordinates via a force-directed spring layout:

$$
c_i^{(t+1)} = c_i^{(t)} + \eta \left( \frac{1}{|N(i)|} \sum_{j \in N(i)} c_j^{(t)} - c_i^{(t)} \right)
$$

where $\eta$ is the learning rate. Once coordinates stabilize, Voronoi boundaries and a hull geometry are constructed.

---

## 3. Spatial Topology, Adjacency Graphs & BYM2 Scaling

### 3.1. Contiguity Types: Queen vs. Rook

Spatial adjacency matrices $W \in \{0, 1\}^{S \times S}$ define the neighborhood topology:
- **Queen Contiguity**: Areas $A_i$ and $A_j$ are neighbors ($W_{ij} = 1$) if they share at least one boundary point or edge ($\partial A_i \cap \partial A_j \ne \emptyset$).
- **Rook Contiguity**: Areas $A_i$ and $A_j$ are neighbors if they share a boundary linear edge of non-zero length ($\operatorname{length}(\partial A_i \cap \partial A_j) > 0$).

---

### 3.2. Disconnected Components & Island Bridging (`ensure_connected!`)

Spatial models (such as ICAR or BYM2) require a connected spatial graph $\mathcal{G}$ to ensure an identifiable singular precision matrix with a single rank-deficiency of 1. If physical geography (e.g., islands, separated peninsulas) creates $K > 1$ disjoint components:

1. `ensure_connected!` calculates the geometric center of each connected component.
2. A 2D KD-Tree searches for the closest pair of components.
3. The closest node pair $(u, v)$ across the two components is connected with an edge:

$$
(u^*, v^*) = \arg\min_{u \in C_a, v \in C_b} \|c_u - c_v\|_2^2
$$

This guarantees a single connected component with minimal perturbation to the graph topology.

---

### 3.3. BYM2 Spectral Scaling Factor ($\operatorname{scalefactor}$)

The BYM2 model (Riebler et al., 2016) decomposes spatial random effects into a structured intrinsic GMRF ($u$) and an unstructured white-noise effect ($v$):

$$
\theta_s = \sigma_s \left( \sqrt{1 - \rho} \cdot v_s + \sqrt{\rho / s_{\text{scale}}} \cdot u_s \right)
$$

where $u \sim \operatorname{ICAR}(W)$ with singular precision matrix $Q = \operatorname{diag}(W \mathbf{1}) - W$. To ensure that the structured component has an empirical marginal variance of approximately 1 (making $\rho \in [0, 1]$ interpretable as the proportion of variance explained by spatial structure), the scaling factor $s_{\text{scale}}$ is computed from the non-zero eigenvalues $\lambda_1, \dots, \lambda_{S-1}$ of $Q$:

$$
s_{\text{scale}} = \exp\left( -\frac{1}{S-1} \sum_{i=1}^{S-1} \log \lambda_i \right)
$$

In `TethysBayes`, this is evaluated via `scaling_factor_bym2(W)`:

```julia
scalefactor = scaling_factor_bym2(W)
```

---

### 3.4. Spatial Weighting Transformations (`spatial_weights_matrix`)

`TethysBayes` provides standardized spatial econometric weighting schemes via `spatial_weights_matrix(W; style=...)`:

| Style | Mathematical Definition | Properties & Modeling Implications |
| :--- | :--- | :--- |
| `:binary` | $W_{ij} \in \{0, 1\}$ | Standard symmetric, unweighted adjacency for ICAR and BYM2 priors. |
| `:row_standardized` / `:row_norm` | $W_{ij}^* = \frac{W_{ij}}{\sum_{k} W_{ik}}$ | Row sums equal 1; represents spatial moving averages in SAR/SEM models ($W y$). |
| `:variance_stabilized` | $W_{ij}^* = \frac{W_{ij}}{\sqrt{\sum_{k} W_{ik}}}$ | Stabilizes variance across units with highly unequal neighbor counts. |

---

## 4. Comprehensive Options & Their Modeling Implications

`assign_spatial_units` provides granular controls over polygon dimensions, unit counts, and convergence thresholds:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Granular Sizing & Count Controls                         │
├──────────────────────────┬────────────────────────────┬─────────────────────────┤
│ 1. Count Enforcement     │ 2. Area Controls           │ 3. Pruning & Merging    │
│  - target_units=15       │  - target_area=50.0        │  - min_area=5.0         │
│  - exact_units=true      │  - min_area / max_area     │  - merge_small_polygons │
│  - grid_resolution=(6,8) │  - radius / length_scale    │  - prune_empty=true     │
└──────────────────────────┴────────────────────────────┴─────────────────────────┘
```

### 4.1. Exhaustive Options Reference Table

| Option | Type | Default | Relevant Methods | Mathematical & Statistical Implications |
| :--- | :--- | :--- | :--- | :--- |
| `area_method` | `Symbol` | `:avt` | All | Selects the spatial partitioning algorithm (`:avt`, `:cvt`, `:kvt`, `:qvt`, `:bvt`, `:hvt`, `:hexagonal`, `:lattice`). Determines whether partitioning is geometric, density-adaptive, or regular grid. |
| `target_units` | `Integer` | `10` | All | Desired number of spatial partitions $S$. Controls model resolution: larger $S$ captures fine-scale heterogeneity but increases parameter count and computation. |
| `exact_units` | `Bool` | `false` | All | When `true`, strictly guarantees *exactly* `target_units` final polygons via deterministic post-clipping bisection (if $K < S$) or adjacent-pair union merging (if $K > S$). |
| `target_area` | `Real` | `nothing` | All | Desired mean area per polygon in coordinate units. Automatically computes `target_units = max(1, round(total_area / target_area))` and scales hexagon radius $R = \sqrt{\frac{2 A}{3\sqrt{3}}}$ or lattice side length $L = \sqrt{A}$. |
| `min_area` | `Real` | `0.0` | All | Minimum allowed polygon surface area. Prevents tiny boundary fragments/slivers generated during polygon clipping from skewing spatial statistics. |
| `max_area` | `Real` | `Inf` | All | Maximum allowed polygon surface area. In hierarchical methods, forces subdivision of large rural/unpopulated cells. |
| `min_points` | `Integer` | `1` | `:avt`, `:kvt`, `:qvt`, `:bvt`, `:hvt` | Minimum observations per polygon. Setting `min_points > 1` prevents unobserved or sparse units that produce poorly identified random intercepts. |
| `max_points` | `Integer` | `length(s_x)` | `:qvt`, `:bvt`, `:hvt` | Maximum observations per polygon. In tree methods, triggers child splitting when a cluster exceeds this threshold. |
| `min_time_slices` | `Integer` | `1` | `:avt`, `:kvt`, `:bvt` | Minimum unique time points required in each polygon. Essential for spatiotemporal models so each spatial unit has time series depth. |
| `t_idx` | `Vector{Int}` | `ones(...)` | `:avt`, `:kvt`, `:bvt` | Vector of observation time stamps passed to spatial partitioning to evaluate `min_time_slices`. |
| `min_total_arealunits` | `Integer` | `3` | Voronoi methods | Minimum lower bound threshold before triggering fallback single-unit aggregation. |
| `max_total_arealunits` | `Integer` | `target_units * 2` | Voronoi methods | Upper bound cap on partitions generated during tree splitting. |
| `target_cv` | `Real` | `1.0` | `:qvt`, `:bvt`, `:hvt` | Target coefficient of variation of point counts across units. Serves as early convergence stopping criterion. |
| `tolerance` | `Real` | `0.1` | `:cvt`, `:kvt`, `:avt`, `:hvt` | Centroid displacement displacement tolerance $\delta$. Smaller values produce stricter geometric convergence at the cost of additional iterations. |
| `buffer_dist` | `Real` | `0.5` | Voronoi methods | Distance to expand the convex hull boundary. Ensures peripheral data points do not lie outside clipped polygon boundaries. |
| `merge_small_polygons` | `Bool` | `false` | All | Automatically merges polygons with $\text{area} < \text{min\_area}$ into their neighbor sharing the longest boundary edge using `LibGEOS.union`. |
| `prune_empty` | `Bool` | `false` | All | Drops spatial units containing 0 data points and updates the graph and $W$ matrix. Reduces model dimension to active units only. |
| `radius` | `Real` | `nothing` | `:hexagonal` | Explicit hexagon radius $R$. Direct control over honeycomb cell resolution. |
| `length_scale` | `Real` | `nothing` | `:lattice` | Explicit square cell side length $L$. Direct control over raster cell resolution. |
| `grid_resolution` | `Int` or `(Int, Int)` | `nothing` | `:lattice` | Explicit $(rows, cols)$ or $N \times N$ tiling for regular grids. |
| `aspect_ratio` | `Real` | `1.0` | `:lattice` | Cell aspect ratio $L_y / L_x$. Used to accommodate non-square geographic bounding boxes without cell distortion. |
| `input_polygons` | `Vector` | `nothing` | Custom | User-supplied custom geometries (e.g. Shapefile, GeoJSON). Bypasses automatic tessellation while constructing centroids, boundaries, and graph topology. |
| `geom_hull` | `LibGEOS.Geometry` | `nothing` | All | Custom boundary clipping polygon (e.g. administrative boundary, coastline). |

---

### 4.2. Return Value Specification

The output of `assign_spatial_units` is a unified `NamedTuple` containing:

```julia
(
    centroids = final_centroids,        # Vector{Tuple{Float64, Float64}} of unit centroids
    polygons = polys_coords,            # Vector{Vector{Tuple{Float64, Float64}}} of closed polygon vertices
    adjacency_edges = v_edges,          # Vector{Tuple{Tuple, Tuple}} of centroid pairs sharing an edge
    graph = g,                          # Graphs.SimpleGraph spatial topology object
    W = W,                              # S x S Float64 adjacency matrix
    hull_coords = hull_coords,          # Bounding envelope / clipping hull coordinates
    s_idx = new_assigns,                # Vector{Int} (length N) mapping observations to unit index 1:S
    s_x = s_x,                          # Original x-coordinates
    s_y = s_y,                          # Original y-coordinates
    s_vals = collect(1:n_units),        # Vector of unique spatial unit labels
    areas = areas,                      # Vector{Float64} of individual polygon surface areas
    point_counts = point_counts,        # Vector{Int} of observation counts per polygon
    n_units = n_units,                  # Total number of spatial units S
    metrics = metrics,                  # NamedTuple of area & density summary statistics
    termination_reason = reason         # Diagnostic string indicating algorithm stopping condition
)
```

#### Diagnostic Metrics Structure (`metrics`)
- `mean_density`, `sd_density`, `cv_density`, `min_density`, `max_density`: Observation count statistics across units.
- `mean_area`, `sd_area`, `cv_area`, `min_area`, `max_area`, `total_area`: Surface area distribution metrics.

---

## 5. Temporal & Spatiotemporal Discretization

### 5.1. Discretization Methods (`discretize_data`)

The `discretize_data` function partitions continuous 1D variables (such as time $t$, continuous covariates $x$, or spatial coordinates) into $K$ categorical intervals:

| Method | Mathematical Definition | Use Case & Modeling Implications |
| :--- | :--- | :--- |
| `"quantile"` | Bins defined by empirical quantiles: $q_k = F_X^{-1}(k/K)$. | Guarantees balanced observation counts per bin; robust against heavy-tailed data. |
| `"regular"` | Equal-width bins: $\Delta x = \frac{\max(X) - \min(X)}{K}$. | Standard uniform time intervals (e.g., calendar years, fixed time steps). |
| `"quantile_regular"` | Equal-width bins between quantiles $[\alpha, 1-\alpha]$ (default: $[0.025, 0.975]$). | Prevents extreme temporal outliers from over-extending outer bin widths. |
| `"kmeans"` | 1D $k$-means clustering minimizing within-bin variance $\sum_k \sum_{x \in B_k} (x - \mu_k)^2$. | Natural clustering for irregularly sampled temporal data. |
| `"jenks"` | Jenks Natural Breaks optimization maximizing Goodness of Variance Fit (GVF). | Optimal classification for unimodal/multimodal distributions with distinct regimes. |
| `"provided"` | Explicit user-supplied break points `brks`. | Predefined regulatory, administrative, or domain-specific thresholds. |

---

### 5.2. Unified Spatiotemporal Indexing (`assign_spatiotemporal_units`)

For spatiotemporal models with spatiotemporal interaction terms (e.g. Type I–IV interaction models, Knorr-Held 2000), `assign_spatiotemporal_units` synchronizes spatial areal unit indices $s \in \{1, \dots, S\}$ and temporal intervals $t \in \{1, \dots, T\}$ into a single joint index $st \in \{1, \dots, S \cdot T\}$:

$$
st = (t - 1) \cdot S + s
$$

```julia
st_res = assign_spatiotemporal_units(df; 
    space_x = :lon, 
    space_y = :lat, 
    time_var = :year, 
    area_method = :hexagonal, 
    target_units = 20, 
    time_method = "unique"
)
```

Returns:
- `au_spatial`: Spatial areal units object.
- `au_temporal`: Temporal units object.
- `s_idx`, `t_idx`, `st_idx`: Vectors of assignments for each observation.
- `S`, `T`, `ST`: Dimensional scalars ($S$, $T$, $S \times T$).
- `W_spatial`: Spatial adjacency matrix.
- `scaling_factor_spatial`: BYM2 precision scaling factor.

---

### 5.3. Spatial Block Cross-Validation (`spatial_block_cv`)

Standard random $k$-fold cross-validation results in overoptimistic predictive performance on spatial data due to spatial autocorrelation between adjacent training and test points (Roberts et al., 2017). `spatial_block_cv` partitions the domain into $K$ spatially disjoint blocks:
- **`:kmeans`**: Partitions coordinate space via $k$-means clustering into $K$ compact geographical folds.
- **`:grid`**: Partitions the bounding box into a regular grid and assigns checkerboard fold IDs.

```julia
folds = spatial_block_cv(df.lon, df.lat; n_folds=5, method=:kmeans)
```

---

## 6. Graph Construction & Topology Utilities

### 6.1. Coordinate-to-Graph Generators

When polygon boundaries are not required but graph topology is needed for CAR/SAR modeling:

- **`spatial_knn_graph(coords, k)`**: Constructs a $k$-nearest neighbors directed/undirected graph and binary adjacency matrix.
- **`spatial_radius_graph(coords, radius)`**: Constructs a distance-threshold graph connecting all points within distance `radius`.

### 6.2. Conversion Utilities

- **`adjacency_matrix_to_nb(W)`**: Converts matrix $W$ to neighbor list `Vector{Vector{Int}}`.
- **`nb_to_adjacency_matrix(nb)`**: Converts neighbor list to dense adjacency matrix.
- **`nodes(adj)`**: Extracts undirected edge pairs `(node1, node2)` and computes the BYM2 scaling factor.
- **`libgeos_lattice_adjacency_matrix(rows, cols; contiguity=:queen)`**: Fast algebraic $\mathcal{O}(N)$ lattice adjacency constructor.

---

## 7. Runnable Julia Examples

### Example 1: Point Observation Partitioning with Exact Hexagonal Units

```julia
using TethysBayes, DataFrames, Random

# 1. Generate synthetic coordinates
rng = MersenneTwister(42)
N = 300
df = DataFrame(
    lon = rand(rng, N) .* 100.0,
    lat = rand(rng, N) .* 50.0,
    year = rand(rng, 2020:2024, N),
    y = rand(rng, 0:15, N)
)

# 2. Partition space into exactly 16 regular hexagons
au = assign_spatial_units(df;
    x = :lon,
    y = :lat,
    area_method = :hexagonal,
    target_units = 16,
    exact_units = true,
    min_area = 10.0,
    merge_small_polygons = true
)

println("Generated $(au.n_units) polygons.")
println("Mean polygon area: $(au.metrics.mean_area)")
println("BYM2 Scaling Factor: $(scaling_factor_bym2(au.W))")

# 3. Visualize spatial graph
p = spatial_graph_plot(au=au, title="16 Hexagonal Spatial Units")

```

---

### Example 2: Centroidal Voronoi Tessellation (`:cvt`) with Target Area

```julia
# Partition domain aiming for ~150 sq units per polygon
au_cvt = assign_spatial_units(df.lon, df.lat;
    area_method = :cvt,
    target_area = 150.0,
    exact_units = false,
    prune_empty = true
)

# Row-standardized spatial weights matrix
W_std = spatial_weights_matrix(au_cvt.W; style=:row_standardized)
```

---

### Example 3: End-to-End Bayesian BYM2 Model with Partitioned Data

```julia
# 1. Assign spatial and temporal units
st_data = assign_spatiotemporal_units(df;
    space_x = :lon,
    space_y = :lat,
    time_var = :year,
    area_method = :hexagonal,
    target_units = 12,
    exact_units = true
)

# 2. Attach unit indices to DataFrame
df.s_idx = st_data.s_idx
df.year_idx = st_data.t_idx

# 3. Fit hierarchical BYM2 + AR1 spatiotemporal model
m = @tbm(
    likelihood(y, family=poisson) ~
        intercept() +
        random(s_idx, model=bym2) +
        random(year_idx, model=ar1),
    df,
    W = st_data.W_spatial,
    verbose = false
)

# 4. Sample posterior
chn = sample(m, NUTS(), 200; progress=false)

# 5. Extract comprehensive results and plot spatial effects
res = model_results_comprehensive(m, chn)
choropleth(st_data.au_spatial.polygons, res.effects.s_idx.structured.mean; title="Posterior Structured Spatial Effect")
```

---

## 8. Cross-Mesh Topological Resharding & Spatial Transfer Operators

In multi-tiered ecological and oceanographic workflows, data sources at different physical layers often possess disparate spatial densities and resolutions. For example, substrate grab stations may form an areal graph $G_{\text{src}} = (V_{\text{src}}, W_{\text{src}})$ with $N_{\text{src}} = 25$ units, whereas final population assessments require predictions on a fine-scale management mesh $G_{\text{dest}} = (V_{\text{dest}}, W_{\text{dest}})$ with $N_{\text{dest}} = 80$ units.

The `TethysBayes` partitioning module provides native linear transfer operators $P \in \mathbb{R}^{N_{\text{dest}} \times N_{\text{src}}}$ to map spatial fields seamlessly between arbitrary irregular polygonal tessellations.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        CROSS-MESH GEOMETRIC TRANSFER OPERATOR                          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  Source Mesh A (A_1, ..., A_m) ──► Latent Field u_src or Samples U_src [N_src × S]     │
│                                           │                                            │
│                 Transfer Matrix P: P_ji = Area(B_j ∩ A_i) / Area(B_j)                  │
│                                           │                                            │
│  Destination Mesh B (B_1, ..., B_n) ◄─────┴── u_dest = P × u_src                       │
│                                               U_dest = P × U_src                       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 8.1. Mathematical Formulations

1. **Area-Weighted Polygon Intersection (`method = :area_weighted`)**:
   For destination polygon $B_j$ and source polygon $A_i$, the transfer weight represents the fraction of $B_j$'s area overlapping $A_i$:

   $$
   P_{j, i} = \frac{\text{Area}(B_j \cap A_i)}{\text{Area}(B_j)}, \quad \sum_{i=1}^{N_{\text{src}}} P_{j, i} = 1
   $$

   Computed geometrically via `LibGEOS.intersection` across all intersecting polygon pairs.

2. **Gaussian Kernel Spatial Decay (`method = :gaussian_kernel`)**:
   Transfer weights decay smoothly as a Gaussian function of Euclidean distance between unit centroids:

   $$
   \tilde{P}_{j, i} = \exp\left(-\frac{\|\mathbf{c}_{\text{dest}, j} - \mathbf{c}_{\text{src}, i}\|^2}{2 h^2}\right), \quad P_{j, i} = \frac{\tilde{P}_{j, i}}{\sum_{k=1}^{N_{\text{src}}} \tilde{P}_{j, k}}
   $$

   where $h$ is the spatial bandwidth parameter (defaults to mean nearest-neighbor distance).

3. **Inverse Distance Weighting (`method = :inverse_distance`)**:

   $$
   \tilde{P}_{j, i} = \frac{1}{(\|\mathbf{c}_{\text{dest}, j} - \mathbf{c}_{\text{src}, i}\| + \epsilon)^p}, \quad P_{j, i} = \frac{\tilde{P}_{j, i}}{\sum_{k=1}^{N_{\text{src}}} \tilde{P}_{j, k}}
   $$

### 8.2. Full Monte Carlo Sample Matrix vs. Moment Resharding

- **Full Monte Carlo Mode (`mode = :samples`)**:
  When downstream likelihoods are non-Gaussian (e.g., Gamma, Negative Binomial, Hurdle), resharding the raw $N_{\text{src}} \times S$ posterior sample matrix preserves higher-order moments, skewness, and multi-modal distributions:

  $$
  \mathbf{U}_{\text{dest}} = \mathbf{P} \mathbf{U}_{\text{src}} \in \mathbb{R}^{N_{\text{dest}} \times S}
  $$

- **Moments Mode (`mode = :moments`)**:
  Reshards the posterior mean vector and variance vector directly:

  $$
  \mathbf{u}_{\text{dest}} = \mathbf{P} \mathbf{u}_{\text{src}}, \quad \boldsymbol{\sigma}^2_{\text{dest}} = \mathbf{P} \boldsymbol{\sigma}^2_{\text{src}}
  $$

### 8.3. Practical Usage

```julia
# 1. Generate two distinct spatial partitionings
au_coarse = assign_spatial_units(df_sed.s_x, df_sed.s_y; area_method=:cvt, target_units=20)
au_fine   = assign_spatial_units(df_all.s_x, df_all.s_y; area_method=:cvt, target_units=60)

# 2. Compute the geometric transfer operator matrix
P = compute_network_transfer_matrix(au_coarse, au_fine; method=:area_weighted)

# 3. Reshard full posterior sample matrix (60 units × 500 draws)
coarse_draws = randn(20, 500) # [N_src × S]
fine_results = reshard_spatial_field(coarse_draws, au_coarse, au_fine; mode=:samples)

# 4. Extract empirical summary statistics
println("Fine-scale resharded summary:")
display(fine_results.summary)
```

---

## 9. Assumptions, Limitations & Best Practices

1. **Projected Coordinate Reference Systems (CRS)**:
   All geometric calculations (Euclidean distance, Shoelace polygon area) assume planar Cartesian coordinates (e.g., UTM or local projected meters/kilometers). Longitude/latitude degrees should be projected to a suitable planar CRS prior to partitioning to avoid high-latitude distortion.
2. **Modifiable Areal Unit Problem (MAUP)**:
   Aggregating point data into discrete polygons involves scale and zoning effects (Openshaw, 1984). Sensitivity analysis should be performed by fitting models across multiple resolutions (`target_units=10`, `target_units=25`, `target_units=50`) or comparing `:hexagonal` vs. `:cvt`.
3. **Graph Connectivity**:
   BYM2 and ICAR spatial random effects require connected spatial graphs. `ensure_connected!` automatically bridges disconnected islands to prevent precision matrix singularity.
4. **Computational Complexity**:
   Point assignment and Lloyd relaxation use `NearestNeighbors.KDTree`, scaling at $\mathcal{O}(N \log S)$. Tessellation of up to $10^6$ points across hundreds of areal units executes in seconds.

---

## 9. References

1. **Besag, J.** (1974). Spatial interaction and the statistical analysis of lattice systems. *Journal of the Royal Statistical Society: Series B (Methodological)*, 36(2), 192–225.
2. **Du, Q., Faber, V., & Gunzburger, M.** (1999). Centroidal Voronoi tessellations: Applications and algorithms. *SIAM Review*, 41(4), 637–676.
3. **Fotheringham, A. S., & Wong, D. W.** (1991). The modifiable areal unit problem in multivariate statistical analysis. *Environment and Planning A*, 23(7), 1025–1044.
4. **Jenks, G. F.** (1967). The data model concept in statistical mapping. *International Yearbook of Cartography*, 7, 186–190.
5. **Knorr-Held, L.** (2000). Bayesian modelling of inseparable space-time variation in disease rates. *Statistics in Medicine*, 19(17-18), 2555–2567.
6. **Lloyd, S.** (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129–137.
7. **Okabe, A., Boots, B., Sugihara, K., & Chiu, S. N.** (2000). *Spatial Tessellations: Concepts and Applications of Voronoi Diagrams* (2nd ed.). John Wiley & Sons.
8. **Openshaw, S.** (1984). *The Modifiable Areal Unit Problem*. Geo Books, Norwich.
9. **Riebler, A., Sørbye, S. H., Simpson, D., & Rue, H.** (2016). An intuitive Bayesian spatial model for disease mapping that accounts for scaling. *Statistical Methods in Medical Research*, 25(4), 1145–1165.
10. **Roberts, D. R., et al.** (2017). Cross-validation strategies for data with temporal, spatial, hierarchical or phylogenetic structure. *Ecography*, 40(8), 913–929.
