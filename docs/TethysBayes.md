---
title: "The TethysBayes Framework: A Comprehensive Architectural & Methodological Overview"
format: html
---

# The TethysBayes Framework: A Comprehensive Architectural & Methodological Overview

## 1. Introduction

The **Bayesian Spatio-Temporal Modeling (`TethysBayes`)** framework provides a composable, formula-driven probabilistic programming interface for complex hierarchical, spatial, temporal, and spatiotemporal models in Julia. Built on top of [Turing.jl](https://github.com/TuringLang/Turing.jl) and the Julia scientific computing ecosystem, `TethysBayes` bridges the gap between the intuitive, high-level modeling syntax of tools like R's `brms` / `INLA` and the computational speed and flexibility of modern probabilistic programming languages.

### Core Architectural Principles:

1. **Decoupled Observation & Latent Processes**:
   The observation likelihood model (Left-Hand Side) is cleanly separated from the latent spatiotemporal field specifications (Right-Hand Side), enabling modular combinations of likelihood distributions and latent processes.
2. **Algebra of Model Components**:
   Model formulas support algebraic operators (`+`, `⊗`, `|>`, `∘`) to compose high-dimensional Kronecker space-time interactions, spatially-varying coefficients, and multi-fidelity hierarchies.
3. **Automatic Differentiability (AD-First)**:
   All component constructors and latent transformations are designed for Automatic Differentiation compatibility, eliminating numerical bottlenecks in gradient-based samplers like NUTS.
4. **End-to-End Workflow Integration**:
   `TethysBayes` centralizes the entire modeling lifecycle: continuous spatial discretization $\to$ topological graph extraction $\to$ prior predictive checking $\to$ adaptive block-sampling $\to$ posterior parameter reconstruction $\to$ analytical SQL querying $\to$ spatial block cross-validation $\to$ Bayesian DAG pipeline $\to$ cross-mesh resharding $\to$ hierarchical cut-posterior simulation $\to$ continuous field inference with flexible projection kernels $\to$ visualization.

```
                               ┌────────────────────────────────────────────────────────┐
                               │  Continuous Point / Tabular Data (s_x, s_y, t, y, x)   │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                               ┌───────────────────────────┴────────────────────────────┐
                               ▼                                                        ▼
                 ┌───────────────────────────┐                            ┌───────────────────────────┐
                 │    Spatial Partitioning   │                            │   Temporal Discretization │
                 │  (:hexagonal, :cvt, etc.) │                            │ (Quantile, Jenks, KMeans) │
                 └─────────────┬─────────────┘                            └─────────────┬─────────────┘
                               │                                                        │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │  Declarative Formula Specification: @tbm(...)         │
                               │  likelihood(y, family=poisson) ~ intercept() +         │
                               │  random(s_idx, model=bym2) ⊗ random(year, model=ar1)   │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │  Code Assembly & Parameter Registry: TethysBayes_config       │
                               │  (Precision matrices, bases, scaling factors)          │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │  Inference: get_optimal_sampler / sample(m, NUTS(), N) │
                               │  (Automatic Gibbs Block Partitioning & NUTS)           │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │  Diagnostics & Post-Processing:                        │
                               │  model_results_comprehensive / TethysBayes_plots / choropleth │
                               └────────────────────────────────────────────────────────┘
```

---

## 2. The Formula Interface

The `@tbm` macro provides a concise, unquoted formula DSL that parses mathematical relationships into an Abstract Syntax Tree (AST).

### 2.1. Basic Syntax

```julia
m = @tbm(
    likelihood(outcome, family=..., modifiers...) ~
        intercept() +
        fixed(covariate, ...) +
        random(grouping_covariate, model=..., ...) +
        other_modules(...),
    data_frame,
    W = spatial_adjacency_matrix,
    au = areal_units_object,
    verbose = false,
    keyword_arguments...
)
```

- **`verbose=false`**: Suppresses internal AST debug dumps, dynamic model compilation code, and automatic prior predictive check messages. Default is `false`.
- **`W`**: Adjacency matrix ($S \times S$) required for discrete spatial GMRFs (`:bym2`, `:icar`, `:besag`, `:leroux`, `:sar`).
- **`au`**: Areal units NamedTuple from `assign_spatial_units`, containing polygons, centroids, and metrics.

---

### 2.2. Observation Likelihood Specification (`likelihood()`)

The Left-Hand Side (LHS) `likelihood()` specifies the conditional observation distribution $p(y \mid \eta)$ and observation-level modifiers:

| Parameter | Example Usage | Description & Mathematical Role |
| :--- | :--- | :--- |
| `family` | `family=:poisson` | Observation likelihood distribution family (19 families supported). |
| `log_offsets` | `log_offsets=log_pop` | Additive offset on the linear predictor link scale ($\eta' = \eta + \text{offset}$). Essential for epidemiological rate models ($\log(\text{Expected})$). |
| `weights` | `weights=sample_w` | Observation-level log-likelihood weighting: $\ell_i(\theta) = w_i \cdot \log p(y_i \mid \eta_i)$. |
| `trials` | `trials=n_total` | Number of binomial trials $N_i$ for `:binomial` models. |
| `zero_inflated` | `zero_inflated=true` | Adds a structural zero-inflation mixture probability $\pi \sim \operatorname{Beta}(1, 1)$. |
| `hurdle` | `hurdle=0` | Truncates likelihood below the hurdle threshold and models positive outcomes conditionally. |
| `volatility` | `volatility=true` | Enables spatiotemporally varying residual variance $\sigma_{y, i}$. |
| `censor_lower` | `censor_lower=limit_L` | Left-censoring threshold ($y_i \le c_{\text{lower}}$). |
| `censor_upper` | `censor_upper=limit_U` | Right-censoring threshold ($y_i \ge c_{\text{upper}}$). |

#### Supported Likelihood Families

| Family | Symbol (`family=...`) | Link Function | Output Support | Core Parameters & Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Gaussian** | `:gaussian` | $\mu = \eta$ | $y \in \mathbb{R}$ | Residual standard deviation $\sigma_y \sim \operatorname{Exponential}(1.0)$. |
| **Poisson** | `:poisson` | $\lambda = \exp(\eta)$ | $y \in \{0, 1, 2, \dots\}$ | Rate parameter $\lambda_i = \exp(\eta_i + \text{offset}_i)$. Equidispersed count processes. |
| **Negative Binomial** | `:negbin` | $\mu = \exp(\eta)$ | $y \in \{0, 1, 2, \dots\}$ | Dispersion $r \sim \operatorname{Gamma}(2, 0.5)$, $\operatorname{Var}(y) = \mu + \mu^2/r$. Overdispersed counts. |
| **Bernoulli** | `:bernoulli` | $p = \operatorname{logistic}(\eta)$ | $y \in \{0, 1\}$ | Binary classification and presence/absence. |
| **Binomial** | `:binomial` | $p = \operatorname{logistic}(\eta)$ | $y \in \{0, \dots, N_i\}$ | Aggregated binary trials given `trials=N`. |
| **Beta** | `:beta` | $\mu = \operatorname{logistic}(\eta)$ | $y \in (0, 1)$ | Precision parameter $\kappa \sim \operatorname{Exponential}(1.0)$. Fractional proportions. |
| **Gamma** | `:gamma` | $\mu = \exp(\eta)$ | $y > 0$ | Shape $\alpha$, scale $\theta = \mu / \alpha$. Positive right-skewed measurements. |
| **Log-Normal** | `:lognormal` | $\mu_{\log} = \eta$ | $y > 0$ | Scale $\sigma_{\log} \sim \operatorname{Exponential}(1.0)$. Multiplicative growth. |
| **Student's T** | `:studentt` | $\mu = \eta$ | $y \in \mathbb{R}$ | Heavy-tailed robust regression with degrees of freedom $\nu$. |
| **Exponential** | `:exponential` | $\lambda = \exp(-\eta)$ | $y > 0$ | Memoryless survival time and event duration. |
| **Weibull** | `:weibull` | $\lambda = \exp(\eta)$ | $y > 0$ | Non-constant hazard rates in survival analysis. |
| **GEV** | `:gev` | $\mu = \eta$ | $y \in \mathbb{R}$ | Generalized Extreme Value for block maxima and extreme events. |
| **Zero-Inflated Poisson** | `:zipoisson` | $\lambda = \exp(\eta)$ | $y \in \{0, 1, \dots\}$ | Count data with structural zero inflation $\pi$. |
| **Zero-Inflated NegBin** | `:zinegbin` | $\mu = \exp(\eta)$ | $y \in \{0, 1, \dots\}$ | Overdispersed counts with structural zero inflation $\pi$. |
| **Ordered Logistic** | `:ordered_logistic` | Cutpoints $c_k$ | $y \in \{1, \dots, K\}$ | Ordered rating categories and Likert scales. |
| **Ordered Probit** | `:ordered_probit` | Normal CDF $\Phi(\cdot)$ | $y \in \{1, \dots, K\}$ | Latent Gaussian threshold crossing models. |
| **Categorical** | `:categorical` | Softmax $\eta_k$ | $y \in \{1, \dots, K\}$ | Unordered discrete choice modeling. |
| **Multinomial** | `:multinomial` | Softmax $\eta_k$ | Count Vector | Compositional counts across competing classes. |
| **Dirichlet** | `:dirichlet` | Softmax $\eta_k$ | Simplex $\Delta^{K-1}$ | Continuous compositional proportions summing to 1. |

---

### 2.3. Right-Hand Side (RHS) Modules

The RHS formula combines linear fixed effects, structured random fields, and process dynamics:

| Module | Purpose | Key Parameters | Example Usage |
| :--- | :--- | :--- | :--- |
| `intercept()` | Controls global intercept prior. | `prior` | `intercept(prior=Normal(0, 5))` |
| `fixed()` | Fixed-effect regression coefficients. | `prior`, `contrast` | `fixed(elevation, prior=Normal(0, 1))` |
| `random()` | Structured & unstructured random fields. | `model`, `sigma`, `rho`, etc. | `random(s_idx, model=bym2)` |
| `mixed()` | Correlated random slopes and intercepts. | `model`, `method` | `mixed(1 + poverty \| region)` |
| `dynamics()` | Mechanistic state-space differential equations. | `model`, `r`, `K`, `velocity` | `dynamics(time, model=:logistic, r=Normal(0.5, 0.1))` |
| `eigen()` | Bayesian PCA factor analysis. | `n_factors`, `pca_sd` | `eigen(pollutant1, pollutant2, n_factors=1)` |
| `nested()`, `transfer()`, `fidelity()` | Multi-fidelity supervised proxy models & transfer learning. | `formula`, `data_source`, `mapping`, `prior`, `fixed` | `transfer(:proxy)` or `nested(proxy, formula="...", data_source=df)` |
| `sciml()` | Scientific Machine Learning ODE/PDE integration. | `model_func`, `solver` | `sciml(t, model_func=my_ode)` |
| `custom()` | User-injected raw Turing code fragments. | `code_fragment` | `custom(code_fragment="...")` |

---

## 3. The Algebra of Components: Composition, Interactions & Varying Curves

The formula parser evaluates algebraic operators to create sophisticated spatiotemporal structures:

```
                            ┌───────────────────────────────────────────────┐
                            │              Formula Operators                │
                            ├───────────────────────┬───────────────────────┤
                            │  Kronecker Product ⊗ │  Pipe Operator |>     │
                            │  Space-Time GMRF      │  Varying Coefficients │
                            └───────────────────────┴───────────────────────┘
```

### 3.1. Kronecker Product (`⊗`) for Space-Time Interactions

Constructs inseparable Knorr-Held (2000) Type I–IV spatiotemporal interactions via Kronecker precision algebra:

$$
Q_{st} = Q_t \otimes Q_s
$$

```julia
# Knorr-Held Type IV interaction: Structured space (ICAR) evolving over smooth time (AR1)
@tbm(
    likelihood(cases, family=poisson) ~ 
        intercept() + 
        random(s_idx, model=icar) ⊗ random(year, model=ar1),
    df, W=W
)
```

### 3.2. Pipe Operator (`|>`) for Varying Coefficients (SVC / TVC)

The pipe operator routes covariates through latent fields to create spatially or temporally varying effects:

```julia
# Spatially Varying Coefficient (SVC): The impact of poverty varies across regions
@tbm(
    likelihood(y) ~ intercept() + (poverty |> random(s_idx, model=icar)),
    df, W=W
)

# Spatially Varying Temporal Curve: A seasonal P-spline curve whose shape varies across space
@tbm(
    likelihood(y) ~ intercept() + (random(s_idx, model=icar) |> random(month, model=pspline)),
    df, W=W
)
```

### 3.3. Declarative Multi-Equation Coupling (`:primary => ..., :proxy => ...`)

High-fidelity models often integrate auxiliary, low-fidelity proxies (such as satellite retrievals or coarse numerical simulations). Rather than nesting formula strings, `TethysBayes` supports declarative multi-equation pair syntax:

$$
\boldsymbol{\eta}_{\text{primary}} = \mathbf{X}_{\text{hi}} \boldsymbol{\beta}_{\text{hi}} + \sum_k f_k(\mathbf{s}, t) + \rho \cdot \boldsymbol{\eta}_{\text{proxy}}[\mathcal{M}]
$$

```julia
# Declarative multi-equation system with unquoted Julia formulas and observation mapping
model = @tbm(
    :primary => (
        formula = likelihood(pm25_ground) ~ 1 + fixed(elev) + transfer(:satellite),
        data    = df_ground
    ),
    :satellite => (
        formula = likelihood(aod) ~ 1 + fixed(humidity) + random(s_idx, model=bym2),
        data    = df_satellite,
        mapping = :satellite_cell_id,   # Aligns ground points to satellite pixels
        prior   = Normal(1.0, 0.25)     # Custom coupling weight prior rho ~ Normal
    )
)
```

---

## 4. Comprehensive Component Catalog

`TethysBayes` organizes over 45 statistical and physical components across five structural families:

### 4.1. Discrete Spatial GMRF Models

Designed for aggregated regional data relying on an adjacency matrix $W$:

| Model (`model=...`) | Mathematical Summary | Key Assumptions & Utility |
| :--- | :--- | :--- |
| **`:bym2`** | $\phi_s = \sigma (\sqrt{1-\rho} v_s + \sqrt{\rho / s_{\text{scale}}} u_s)$ | **Standard Disease Mapping**: Decomposes spatial variance into structured ICAR clustering and IID noise. Directly interpretable mixing parameter $\rho \in [0, 1]$. |
| **`:icar` / `:besag`** | $u \sim \mathcal{N}(0, (\tau Q)^{-1}), \; Q = \operatorname{diag}(W\mathbf{1}) - W$ | **Intrinsic CAR**: Pure local neighbor smoothing. Enforces sum-to-zero constraint $\sum u_i = 0$. |
| **`:leroux`** | $Q = (1 - \rho)I + \rho Q_{\text{ICAR}}$ | **Proper CAR**: Full-rank precision matrix for all $\rho \in [0, 1)$. Avoids boundary singularities. |
| **`:sar`** | $\phi = (I - \rho W_{\text{std}})^{-1} \epsilon$ | **Simultaneous Autoregressive**: Models spatial lag spillover feedback in spatial econometrics. |
| **`:localadaptive`**| $\phi \sim \mathcal{N}(\mu_{\text{cluster}(s)}, (\sigma^2 Q)^{-1})$ | **Regime Shifts**: Combines Leroux smoothing with cluster-specific mean shifts across borders. |
| **`:bcgn`** | Bipartite Graph Convolutional Network | **Bipartite Networks**: Spatial smoothing across multi-scale bipartite graphs. |
| **`:dag`** | Directed Acyclic Graph spatial model | **Causal Topology**: Non-reciprocal spatial flow relationships. |

---

### 4.2. Continuous Geostatistical & Spectral Models

Designed for continuous 2D/3D point-referenced coordinates:

| Model (`model=...`) | Mathematical Summary | Key Assumptions & Computational Complexity |
| :--- | :--- | :--- |
| **`:gp`** | $f \sim \mathcal{GP}(0, k(x, x'))$ | **Exact Kriging**: Matérn/SE covariance. Exact dense $\mathcal{O}(N^3)$ Gaussian process. |
| **`:sparsegp` / `:fitc`**| $K_{ff} \approx K_{fu} K_{uu}^{-1} K_{uf} + \operatorname{diag}(\cdot)$ | **Scalable Pseudo-Inputs**: Low-rank FITC approximation scaling as $\mathcal{O}(NM^2)$ for $N > 10^5$. |
| **`:rff`** | $z(x) = \sqrt{2/D}\cos(Wx + b)$ | **Random Fourier Features**: Bochner spectral features providing linear $\mathcal{O}(ND)$ time scaling. |
| **`:spde`** | $(\kappa^2 - \Delta)^{\alpha/2} u = \mathcal{W}$ | **Finite Element Mesh**: Triangulated mesh representation of continuous Matérn fields ($\mathcal{O}(N^{1.5})$). |
| **`:nystrom`** | Nyström low-rank projection | **Kernel Approximation**: Efficient low-rank eigendecomposition on landmark points. |
| **`:waveletgp`** | Wavelet Multiresolution GP | **Multi-Scale**: Wavelet decomposition across localized spatial frequency bands. |

---

### 4.3. Temporal & Longitudinal Models

| Model (`model=...`) | Mathematical Summary | Key Assumptions & Utility |
| :--- | :--- | :--- |
| **`:ar1` / `:ar2`** | $x_t = \sum_{k=1}^p \rho_k x_{t-k} + \sigma \epsilon_t$ | **Autoregressive**: Stationary serial correlation and short-term forecasting. |
| **`:rw1` / `:rw2`** | $\Delta x_t \sim \mathcal{N}(0, \sigma^2)$ or $\Delta^2 x_t \sim \mathcal{N}(0, \sigma^2)$ | **Random Walks**: Nonparametric stochastic level and smooth curvature trends. |
| **`:harmonic` / `:cyclic`**| $f(t) = \sum (\beta_c \cos(\omega t) + \beta_s \sin(\omega t))$ | **Periodicity**: Deterministic or stochastic Fourier seasonality (annual, diurnal). |
| **`:tar`** | Threshold Autoregressive | **Nonlinear Regimes**: Asymmetric regime switching at tipping points. |

---

### 4.4. Nonparametric Smooths & Splines

| Model (`model=...`) | Mathematical Summary | Key Assumptions & Utility |
| :--- | :--- | :--- |
| **`:pspline`** | Penalized B-splines with difference penalty | **Flexible 1D Curves**: Nonparametric nonlinear covariate effects ($f(x)$). |
| **`:bspline`** | Unpenalized basis splines | **Polynomial Splines**: Fixed-knot flexible curve fitting. |
| **`:tps`** | Thin Plate Regression Splines | **Multidimensional Smooths**: Isotropic 2D/3D surface smoothing. |
| **`:adaptivesmooth`**| Spatially adaptive P-splines | **Heterogeneous Smoothness**: Varying penalty parameter $\lambda(x)$ for sharp local transitions. |

---

## 5. Spatial & Spatiotemporal Partitioning Subsystem

Continuous point coordinates $(s_x, s_y)$ can be discretized into discrete areal units using `assign_spatial_units`, generating neighborhood graphs $W$ and BYM2 spectral scaling factors.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    TethysBayes Spatial Partitioning Methodologies                      │
├──────────────────────────┬────────────────────────────┬─────────────────────────┤
│ 1. Geometric Voronoi     │ 2. Density-Adaptive        │ 3. Regular Grids        │
│  - :cvt (Lloyd's energy) │  - :kvt (K-means balanced) │  - :hexagonal (Uniform) │
│  - :avt (Bottom-up merge)│  - :qvt (Quadtree splits)  │  - :lattice (Raster)    │
│  - :bvt (Variance tree)  │  - :hvt (Hierarchical)     │  - Inferred spring map  │
└──────────────────────────┴────────────────────────────┴─────────────────────────┘
```

### 5.1. Sizing, Count & Geometry Controls

- **`exact_units=true`**: Dynamically guarantees *exactly* $S$ output polygons via post-clipping bisection and union merging.
- **`target_area=A`**: Scales hexagon radius $R = \sqrt{\frac{2 A}{3\sqrt{3}}}$ or lattice length scale $L = \sqrt{A}$.
- **`min_area` & `merge_small_polygons=true`**: Eliminates boundary slivers by merging small fragments into their longest-sharing neighbor using `LibGEOS.union`.
- **`prune_empty=true`**: Removes unobserved polygons containing 0 data points.
- **`ensure_connected!`**: Automatically detects disconnected spatial islands and adds minimal bridging edges to guarantee a fully connected graph for GMRF identifiability.

### 5.2. Joint Spatiotemporal Discretization (`assign_spatiotemporal_units`)

```julia
st_res = assign_spatiotemporal_units(df;
    space_x = :lon,
    space_y = :lat,
    time_var = :year,
    area_method = :hexagonal,
    target_units = 16,
    exact_units = true,
    time_method = "unique"
)
```

Synchronizes spatial areal units ($s \in 1\dots S$) and temporal intervals ($t \in 1\dots T$) into a joint composite index $st = (t - 1) \cdot S + s$, returning $W$, $S$, $T$, $ST$, and the BYM2 spectral scaling factor.

---

## 6. Prior Specification Architecture, Penalized Complexity (`:pcpriors`) & Distributional Choices

`TethysBayes` implements a principled, three-tier hierarchical prior resolution engine:
1. **Local Parameter In-line Overrides** (Highest Precedence): e.g. `random(s_idx, model=bym2, sigma=(1.0, 0.01), rho=Beta(2, 2))`.
2. **Global Overrides Dictionary**: Passed via `hyperpriors=Dict(:sigma => Exponential(1.0), :bym2_rho => Beta(1, 1))`.
3. **Automated Prior Schemes (`prior_scheme`)**: Pre-calibrated scheme presets:
   - `:pcpriors` (Default): Penalized Complexity priors (Simpson et al., 2017).
   - `:informative`: Moderately regularizing priors for small-sample/data-scarce regimes.
   - `:uninformative`: Wide/flat reference priors.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    TethysBayes Prior Hierarchy Resolution                              │
├──────────────────────────┬────────────────────────────┬─────────────────────────┤
│ 1. Local Formula Call    │ 2. Global Hyperpriors Dict │ 3. Prior Scheme Presets │
│  - sigma = (1.0, 0.01)   │  - :icar_sigma => ...      │  - :pcpriors (Default)  │
│  - rho = Beta(2, 2)      │  - :sigma => ...           │  - :informative         │
│  - prior = Normal(0, 1)  │  - :length_scale => ...     │  - :uninformative       │
└──────────────────────────┴────────────────────────────┴─────────────────────────┘
```

---

### 6.1. Penalized Complexity (PC) Priors (`:pcpriors`)

Penalized Complexity (PC) priors (Simpson et al., 2017) provide an axiomatic framework for constructing priors that prevent overfitting and enforce Occam's razor.

#### Core Principles of PC Priors
1. **Parsimony / Base Model**: Every complex component (e.g. spatial field, autocorrelation, non-linear spline) is viewed as an extension of a simpler "base model" (e.g., $\sigma = 0$ corresponds to no spatial field; $\rho = 0$ corresponds to independence; $\ell = \infty$ corresponds to a flat constant).
2. **Information-Theoretic Distance**: Model divergence from the base model $f_0$ to the flexible model $f$ is measured using Kullback-Leibler Divergence (KLD):

   $$
   d(f \parallel f_0) = \sqrt{2 \operatorname{KLD}(f \parallel f_0)}
   $$

3. **Constant Rate Penalization**: Placing an Exponential prior on the distance $d$ yields an invariant prior that penalizes deviation from simplicity at a constant rate $\lambda$:

   $$
   \pi(d) = \lambda \exp(-\lambda d)
   $$

4. **Intuitive Quantile Constraints**: Users parameterize the prior via an interpretable tail probability constraint $(U, \alpha)$:

   $$
   P(\text{parameter} > U) = \alpha
   $$

   meaning: *"The prior probability that the effect scale exceeds $U$ is only $\alpha$ (e.g., 1% or 5%)."*

#### Mathematical Quantile Formulations in `TethysBayes`

* **Standard Deviation & Scale ($\sigma, \kappa$)**:
  - *Base Model*: $\sigma = 0$ (no latent variation).
  - *Constraint*: $P(\sigma > U) = \alpha \implies \lambda = -\frac{\log(\alpha)}{U} \implies \sigma \sim \operatorname{Exponential}(\lambda)$.
  - *Example*: `random(s_idx, model=icar, sigma=(1.0, 0.01))` sets $P(\sigma > 1.0) = 0.01 \implies \lambda = -\log(0.01)/1.0 \approx 4.605$.

* **Autoregressive / Spatial Correlation ($\rho \in [0, 1)$ or $(-1, 1)$)**:
  - *Base Model*: $\rho = 0$ (independent noise).
  - *Transform*: $\theta = -\log(1 - \rho) \sim \operatorname{Exponential}(\lambda)$.
  - *Constraint*: $P(\rho > U) = \alpha \implies \lambda = \frac{\log(\alpha)}{\log(1 - U)}$.
  - *Example*: `random(year, model=ar1, rho=(0.5, 0.05))` sets $P(\rho > 0.5) = 0.05 \implies \lambda = \log(0.05)/\log(0.5) \approx 4.322$.

* **Gaussian Process / Spline Lengthscale ($\ell$)**:
  - *Base Model*: $\ell = \infty$ (infinitely smooth / constant flat function).
  - *Transform*: $\theta = 1/\ell \sim \operatorname{Exponential}(\lambda)$.
  - *Constraint*: $P(\ell < U) = \alpha \implies \lambda = -U \log(\alpha)$.
  - *Example*: `random(x, model=gp, length_scale=(0.1, 0.01))` sets $P(\ell < 0.1) = 0.01$ (penalizing high-frequency micro-scale fluctuations).

* **Fixed Effect & Regression Coefficients ($\beta$)**:
  - *Base Model*: $\beta = 0$ (no covariate effect).
  - *Constraint*: $P(|\beta| > U) = \alpha \implies \beta \sim \operatorname{Normal}(0, \sigma_{\beta})$ where $\sigma_{\beta} = \frac{-U}{\Phi^{-1}(\alpha/2)}$.
  - *Example*: `fixed(elevation, prior=(2.0, 0.05))` sets $P(|\beta| > 2.0) = 0.05 \implies \beta \sim \operatorname{Normal}(0, 1.02)$.

---

### 6.2. Alternative Prior Distributions & Practical Considerations

| Prior Distribution | Parameter Domain | Typical Application | Key Properties & Practical Considerations |
| :--- | :--- | :--- | :--- |
| **`Exponential(λ)`** | $\sigma > 0$ | GMRF, BYM2, Spline scales | **PC-optimal**: Penalizes variance away from 0 at constant rate. Non-zero density at 0 prevents artificial variance inflation. |
| **`Beta(α, β)`** | $\rho \in [0, 1]$ | BYM2 mixing $\rho$, Leroux $\rho$ | `Beta(1, 1)` = Uniform; `Beta(2, 2)` = Bell-shaped around 0.5; `Beta(1, 2)` = Shrinks toward IID noise. |
| **`truncated(Cauchy(0, s), 0, Inf)`** | $\sigma > 0$ | Robust variance estimation | Heavy $1/x^2$ tails allow large outlier variances without over-shrinkage; may cause wide energy basins in HMC. |
| **`truncated(Normal(0, s), 0, Inf)`** | $\sigma > 0$ | Moderately regularized scales | Gaussian tail decay; stronger regularization than Cauchy against extreme variances. |
| **`InverseGamma(α, β)`** | $\sigma^2 > 0$, $\ell > 0$ | GP lengthscales, conjugate scales | **Pathology Warning (Gelman 2006)**: Vague $\operatorname{IG}(\epsilon, \epsilon)$ has an unintended spike near 0 that creates severe funnel singularities in HMC. Use only when calibrated (e.g. $\operatorname{IG}(3, 3)$). |
| **`LogNormal(μ, σ)`** | $x > 0$ | Physical / mechanistic rates | Strictly positive, right-skewed; ideal for velocity $v$, diffusion $D$, and carrying capacity $K$ in movement models. |
| **`LKJCholesky(K, η)`** | Correlation Cholesky $L$ | MCAR spatial, multivariate random slopes | $\eta = 1$ = Uniform over correlation matrices; $\eta > 1$ (e.g. $\eta = 2$) shrinks off-diagonal correlations toward 0. |

---

## 7. Sampling, Inference & Sampler Optimization

`TethysBayes` provides a multi-paradigm inference engine combining composite Gibbs partitioning, gradient-based HMC/NUTS, gradient-free slice sampling, and variational approximations.
 
Though MCMC sampling is our gold-standard, we also have other
optimization-based options that can be worth considering. All of these
methods are boosted by Automatic Differentiation, some require smooth
differentiable likelihood surfaces, while others are robust and can be
range bound. See the [whole list
here](https://docs.sciml.ai/Optimization/stable/optimization_packages/optim/).

- Maximum likelihood (ML) estimation can be much faster than MCMC as
pure optimization of a point mass is considerably simpler as priors
are ignored and there is no need to carry posterior samples. Many
specialized optimization algorithms exist that have been tried and
tested over many years.

- Maximum a-posteriori (MAP) estimation is the same as ML except that
prior information is used as well and so a bit closer to MCMC in
spirit, though the focus is still upon the point estimates.

- Variational Inference is also an optimization method. However, it
approaches the problem using the ELBO estimator (log-likelihood aka
"evidence" of the observed data, \(\log p(x)\)). In complex models,
calculating the true posterior distribution \(p(z|x)\) is impossible
(intractable). Instead, we choose a simpler, flexible distribution
\(q(z)\) to approximate it. Maximizing the ELBO is equivalent to
minimizing the difference (Kullback-Leibler divergence) between
\(q(z)\) and the true posterior. The ELBO allows us to use
gradient-based optimization methods (like Stochastic Gradient Descent)
to solve Bayesian inference problems. In principle it is closest to
MCMC and able to describe the posterior distribution reasonably well.

It is, similar to INLA in that optimization is also used but INLA uses
a Laplace Approximation which comes with the assumption/constraint
that the posterior marginals can be accurately approximated by a
Taylor-series expansion. VI has no limits to distributional
constraints (INLA internally assumes Gaussian), and requires that the
likelihood is smooth enough for optimization. Note that optimizers for
non-differentiable surfaces exist too...

All three methods are accessible with the same Turing/Julia model (but
not Laplace Approximation, to my knowledge, though some Julia projects
seem to have used LA). As a bonus you also get Automatic
Differentiation for free and can use the same Turing model.

The following code snippet shows how to run them. One can even use the
solution from one method as the starting point of another (though that
runs the risk of starting from a pathologiucal suboptimimum).



```{julia}
#| Optimization approaches

using Optim, AdvancedVI, Turing

# ML 
res_opt = maximum_likelihood(m, LBFGS() )  # many optimizers available
res_opt.optim_result.retcode
res_opt.params

res = model_results_comprehensive(m, res_opt, inp_scot, au_scot, n_samples=100);
 
# MAP  
res_map = maximum_a_posteriori( m, LBFGS() )
res_map.optim_result.retcode   #
res_map.params
res = model_results_comprehensive(m, res_map, inp_scot, au_scot);


# Variational Inference 
samples_per_step = 10
max_iters = 1000
n_samples = 1000

# q_init = q_locationscale
# q_init = q_meanfield_gaussian
# q_init = q_fullrank_gaussian
q_init = q_fullrank_gaussian

chn_vi = vi(m, q_init, max_iters, adtype=AutoForwardDiff(),
show_progress=true; optimizer=LBFGS() ) #;,
optimizer=Flux.ADAM(1e-1));

# Convert to reconstruct-compatible format
# chn = convert_advi_to_reconstruct_format(chn_vi, m, n_samples)

# Reconstruct and Visualize: not great
res = model_results_comprehensive(m, chn_vi, inp_scot, au_scot);

pairs(res.metrics)

 
```

### 7.1. Automatic Composite Gibbs Partitioning (`get_optimal_sampler`)

`get_optimal_sampler` inspects parameter supports and structure via `ParamRegistry` and builds a partitioned composite Gibbs sampler tailored to the model geometry:
- **Discrete Parameters**: `PG` (Particle Gibbs).
- **Gaussian Latent Vectors**: `ESS` (Elliptical Slice Sampling) for high-dimensional GMRFs.
- **Bounded Scalars ($\sigma > 0, \rho \in [0, 1]$)**: `Slice` or unconstrained `NUTS`.
- **Differentiable Continuous Parameter Blocks**: `NUTS(adaptation_steps, target_acceptance; init_ϵ=init_ϵ, max_depth=max_depth, adtype=adtype)`.

```julia
# 1. Optimal Composite Gibbs Sampler with Pre-conditioned Proposals
os = get_optimal_sampler(m; 
    init_ϵ = :auto,       # Dimensional scaling ϵ ~ 0.5 * D^(-1/4)
    max_depth = 10,       # Spectral condition number bound
    min_ϵ = 1e-4,         # Lower bound to prevent step collapse
    max_ϵ = 1.0,          # Upper bound to prevent divergence
    adtype = AutoForwardDiff()
)
chn = sample(m, os, 500; progress=false)

# 2. Standard Pure NUTS
chn = sample(m, NUTS(), 500; progress=false)

# 3. Variational Inference (ADVI)
vi_res = vi(m, ADVI(10, 1000))

# 4. Maximum A-Posteriori (MAP)
map_res = optimize(m, MAP())
```

---

### 7.2. Theoretical Foundations & Considerations for NUTS Parameter Tuning

In high-dimensional spatial and spatiotemporal Bayesian models, default unconstrained sampler heuristics often encounter severe pathologies during warmup. `TethysBayes` addresses these using principles from Hamiltonian dynamics, symplectic geometry, and spectral graph theory.

#### A. Initial Step Size ($\epsilon$) Pre-Conditioning & Bounds
* **The Failure Mode of Standard Search**: Turing's default `find_good_stepsize` initializes $\epsilon$ by doubling/halving until a single leapfrog acceptance probability is $\approx 0.5$. In high-dimensional spatial fields (e.g., $S = 500$ areas), steep local gradients at initial values frequently cause $\epsilon$ to collapse to $10^{-6} - 10^{-8}$, triggering maximum tree-depth stalls (1024 leapfrog steps per sample), or explode to $10^2$, causing immediate divergence.
* **Dimensional Curvature Scaling**: Following Roberts & Rosenthal (2001), the optimal step size for a block of dimension $D$ scales as $\mathcal{O}(D^{-1/4})$. Standardized latent innovations (`ure ~ MvNormal(0, I)`) have Hessian curvature $\approx I$, yielding a robust initial proposal:

  $$
  \epsilon_{\text{init}} = \operatorname{clamp}\left(0.5 \cdot D^{-1/4},\, \epsilon_{\min},\, \epsilon_{\max}\right)
  $$

  - Global scalars ($D = 1-5$): $\epsilon \approx 0.35 - 0.50$.
  - Spatial fields ($D = 50-5000$): $\epsilon \approx 0.05 - 0.18$.
* **Safe Envelopes (`min_ϵ` & `max_ϵ`)**: Clamping $\epsilon \in [10^{-4}, 1.0]$ guarantees the dual averaging controller begins inside a numerically stable region.

#### B. Target Acceptance Rate ($\delta$) Selection
* **Asymptotic Efficiency ($\delta^* = 0.651$)**: For fixed-trajectory HMC in isotropic Gaussian targets, $\delta = 0.65$ maximizes Expected Squared Jumping Distance per gradient evaluation ($\text{ESJD} / N_{\text{grad}}$).
* **Curvature Suppression in Hierarchies ($\delta = 0.80 - 0.95$)**: In NUTS, variable-length trajectories must traverse regions of varying curvature (e.g. hierarchical spatial variances $\sigma \to 0$, producing Neal's funnel). The symplectic leapfrog integrator accumulates $\mathcal{O}(\epsilon^2)$ local energy error. Setting a higher target $\delta \in [0.80, 0.95]$ forces the dual averaging adaptation to shrink $\epsilon$, preventing the trajectory from overshooting the typical set and producing **divergent transitions**.

#### C. Maximum Tree Depth (`max_depth`) & Spectral Condition Numbers
* **Harmonic Oscillator Stopping Time**: In Hamiltonian dynamics, a Gaussian target mode with variance $\sigma^2$ oscillates with period $T = 2\pi \sigma$. The optimal trajectory length before a U-turn is a half-period: $\tau^* = \pi \sigma_{\max}$.
* **Spectral Condition Number Bound**: For GMRFs with precision eigenvalues $\lambda_1 \le \dots \le \lambda_n$, the condition number is $\kappa = \lambda_{\max} / \lambda_{\min}$. The steps required to traverse the slowest mode is $L^* \approx \pi \sqrt{\kappa}$, yielding the theoretical minimum tree depth:

  $$
  \text{depth}_{\max}^* = \left\lceil \log_2\left(\pi \sqrt{\kappa}\right) \right\rceil = \left\lceil \log_2(\pi) + \frac{1}{2}\log_2(\kappa) \right\rceil
  $$

| Model Structure | Condition Number $\kappa$ | Recommended `max_depth` | Recommended Target $\delta$ |
| :--- | :--- | :--- | :--- |
| **Spectral / Non-Centered / Marginalized** | $\kappa \le 100$ | **$4 - 6$** (16–64 steps) | **$0.65$** (Fastest exploration) |
| **Standard ICAR / BYM2 / Leroux / AR(1)** | $100 < \kappa \le 10^4$ | **$6 - 8$** (64–256 steps) | **$0.80$** (Balanced stability) |
| **Unregularized GPs / Stiff Dynamics / SciML** | $\kappa > 10^4$ | **$8 - 10$** (256–1024 steps)| **$0.90 - 0.95$** (Suppresses divergence) |

#### D. Principled Warmup & Adaptation Length Calibration ($N_{\text{adapt}}$)

* **Why Fixed Ratios (e.g., 25% or 50%) Fail**:
  - **Short/Debug Runs ($N_{\text{samples}} = 100$)**: A fixed $25\%$ yields only $25$ steps. Dual averaging fails to converge ($\mathcal{O}(1/\sqrt{t})$ rate), mass matrix estimates remain singular, and subsequent sampling suffers massive divergence and autocorrelation.
  - **Large Production Runs ($N_{\text{samples}} = 10,000$)**: A fixed $25\%$ forces $2,500$ warmup steps, wasting hours of compute when the metric and step size already reached numerical equilibrium after $\approx 300 - 500$ steps.
* **The Three Distinct Adaptation Tasks**:
  1. **Locating the Typical Set (Initial Stage)**: Moving from arbitrary initial parameter values to the high-probability manifold ($\approx 75 - 100$ steps).
  2. **Dual Averaging Step-Size Stability (Continuous)**: Step size $\epsilon_t$ adaptation (Nesterov, 2009; Hoffman & Gelman, 2014) requires $N \ge 150 - 300$ steps to achieve $\pm 5\%$ asymptotic stability around $\epsilon^*$.
  3. **Metric / Mass Matrix ($M$) Estimation**:
     - *Diagonal Metric*: Estimating coordinate marginal variances $\sigma_i^2$ scales as $\mathcal{O}(\sqrt{D})$.
     - *Dense Metric*: Estimating the $D(D+1)/2$ full covariance entries requires at least $3D - 5D$ draws for well-conditioned empirical covariance inversion.
* **The `TethysBayes` Principled Formula (`adaptation_steps = :auto`)**:

  $$
  N_{\text{adapt}}^{\text{diag}}(D) = \operatorname{clamp}\left(150 + 25 \sqrt{D}, \; 100, \; 1000\right)
  $$

  $$
  N_{\text{adapt}}^{\text{dense}}(D) = \operatorname{clamp}\left(150 + 4 D, \; 150, \; 1500\right)
  $$

  If the spectral condition number $\kappa > 1000$, a $1.3\times$ multiplier is applied to account for slow modes. If total $N_{\text{samples}}$ is provided, adaptation is capped at $\min(N_{\text{adapt}}, \lceil 0.5 \cdot N_{\text{samples}} \rceil)$.

#### E. Diagnostics: Tree-Depth Saturation & E-BFMI
* **Tree-Depth Saturation ($f_{\text{sat}}$)**: The fraction of transitions hitting `max_depth`. If $f_{\text{sat}} > 0.05$ with zero divergences, increase `max_depth` by 1–2. If divergences exist, increase $\delta$ or use spectral reparameterization.
* **Energy-Bayesian Fraction of Missing Information (E-BFMI)**:

  $$
  \text{E-BFMI} = \frac{\sum_{i=1}^N (E_i - E_{i-1})^2}{\sum_{i=1}^N (E_i - \bar{E})^2}
  $$

  $\text{E-BFMI} \ge 0.3$ confirms the momentum distribution efficiently explores the energy spectrum.

---

### 7.3. Pre-inspecting Proposal Configurations (`precompute_step_sizes`)

You can inspect the exact analytical condition numbers, dimensional step sizes, principled adaptation lengths, and recommended tree depths before running MCMC:

```julia
props = precompute_step_sizes(m; min_ϵ=1e-4, max_ϵ=1.0)
# Returns a Dict with block dimensions, initial step sizes, condition numbers, max depths, and adaptation steps
```

---

## 8. Diagnostics, Posterior Reconstruction & Plotting

### 8.1. Comprehensive Results (`model_results_comprehensive`)

`model_results_comprehensive` reconstructs posterior fields from the MCMC chain:

```julia
res = model_results_comprehensive(m, chn)

# Access parameter summary table and metrics
display(res.parameters) # R-hat, ESS, posterior means
println("Model RMSE: ", res.metrics.rmse)
println("WAIC: ", res.metrics.waic)
```

### 8.2. Plotting (`src/plotting.jl`)

```julia
# 1. Generate full diagnostic and spatial effect plots
plots_res = TethysBayes_plots(res; au=st_res.au_spatial, save_dir="output/plots")
display(plots_res.plots[:spatial])

# 2. Spatial Adjacency Graph & Polygon Tessellation
spatial_graph_plot(au=st_res.au_spatial, title="Hexagonal Spatial Grid Topology")

# 3. Credible Interval Timeseries Ribbon
timeseries_ci(1:T, res.pstats.time_mean, res.pstats.time_lower, res.pstats.time_upper; title="Temporal Trend")

# 4. Display All Automated Diagnostic Plots
model_results_plots(res)
```

### 8.3. Model State & Analytical Database Persistence and Sampling Updating (`src/input_output.jl`)

`TethysBayes` uses a two-tier decoupled architecture:
- **JLD2 (`.jld2`)**: Full serialization of live Turing model states (`m`), data, and MCMC chains (`chn`).
- **DuckDB (`.duckdb`)**: Embedded relational analytical database storing normalized performance metrics, parameter stats, predictions, and plot datasets for zero-copy SQL analytics.

```julia
# 1. Save unified bundle (.jld2 and .duckdb)
save_TethysBayes_bundle("runs/model_bym2", m, chn, res; au=st_res.au_spatial)

# 2. Query results with SQL via DuckDB
df_high_risk = query_duckdb("runs/model_bym2.duckdb", 
    "SELECT * FROM plot_data_latent_field_spatial WHERE latent_field_mean > 1.2 ORDER BY latent_field_mean DESC")

# 3. Reload model and extend sampling with more iterations
bundle = load_TethysBayes_bundle("runs/model_bym2")
chn_extended = extend_sampling(bundle.model, bundle.chain, 500; progress=false)
```

### 8.4. Modular DAG Pipeline & Cross-Mesh Resharding (`src/pipeline.jl`)

The `TethysBayes_pipeline` orchestrator coordinates multi-tier cut-posterior models, transferring
spatial fields across mismatched irregular polygonal tessellations via geometric linear
transfer operators ($P \in \mathbb{R}^{N_{\text{dest}} \times N_{\text{src}}}$):

```julia
pipe_res = TethysBayes_pipeline(
    :depth     => (formula = "likelihood(depth) ~ intercept() + random(s_x, s_y, model=rff)", data = df_bathy, derivatives = [:slope, :curvature, :bpi]),
    :substrate => (formula = "likelihood(grain) ~ intercept() + fixed(depth_mu, error_sd=:depth_sd) + random(s_idx, model=bym2)", data = df_sub, au = au_sub),
    :biology   => (formula = "likelihood(catch, family=gamma) ~ intercept() + fixed(grain_mu, error_sd=:grain_sd) + random(s_idx, model=bym2)", data = df_bio, au = au_master);
    master_au = au_master,
    duckdb_path = "project_db/pipeline.duckdb"
)
```

### 8.5. Continuous Surface Derivatives & Differential Geometry (`src/derivatives.jl`)

Calculates exact continuous surface derivatives ($\nabla z, \nabla^2 z$), slopes, compass
aspects, profile/planform curvatures, and circular Bessel Bathymetric Position Indices
($\text{BPI}_r$) directly across `RFF`, `SpectralGP`, `WaveletGP`, `SPDE`, `PSpline`, and `TPS`:

```julia
derivs = TethysBayes_surface_derivatives(m_depth, chn_depth, query_coords; metrics=[:slope, :curvature, :bpi], radii=[10.0, 25.0])
```

### 8.6. Mechanistic Movement & Mark-Recapture Telemetry (`src/movement.jl`)

Models Advection-Diffusion-Reaction population dynamics, integrating Lagrangian mark-recapture
tracking encounters ($u_{\text{rel}} \to u_{\text{rec}}$) with optional Eulerian population abundance surveys.
The advective drift velocity $\mathbf{v} \propto \nabla \text{HSI}$ is driven by Habitat Suitability Index
surfaces with functional relationships (`:exponential`, `:linear`, or `:logistic`):

```julia
m_mov = @tbm(
    likelihood(density, family=poisson) ~ 
        intercept() + 
        movement(
            unit_id, year,
            habitat = :habitat_p,
            relationship = :exponential,
            mark_recapture_data = df_telemetry,
            velocity = Truncated(Normal(1.0, 0.5), 0.0, Inf),
            diffusion = LogNormal(-0.5, 0.5)
        ),
    df_survey, W = au.W
)
```

For comprehensive guides, see:
- [**Integrated Hierarchical Workflows & ADR Telemetry** (`docs/hierarchical_workflow/hierarchical_workflow.md`)](hierarchical_workflow/hierarchical_workflow.md)
- [**Spatial & Spatiotemporal Partitioning Guide** (`docs/TethysBayes_spatial_partitioning.md`)](TethysBayes_spatial_partitioning.md)
- [**Input / Output & Persistence Guide** (`docs/TethysBayes_input_output.md`)](TethysBayes_input_output.md)
- [**Custom Components & Spatial SEIR Modeling Guide** (`docs/TethysBayes_custom.md`)](TethysBayes_custom.md)
- [**Technical API Reference** (`docs/TethysBayes_api.md`)](TethysBayes_api.md)

---

## 9. Complete Runnable Example: End-to-End Spatiotemporal Modeling

```julia
using TethysBayes, DataFrames, Random, Plots

# 1. Simulate point-referenced epidemiological data
rng = MersenneTwister(42)
N = 400
df = DataFrame(
    lon = rand(rng, N) .* 100.0,
    lat = rand(rng, N) .* 50.0,
    year = rand(rng, 2020:2024, N),
    elevation = randn(rng, N),
    y = rand(rng, 0:20, N)
)

# 2. Partition space-time into 16 regular hexagons across 5 years
st_data = assign_spatiotemporal_units(df;
    space_x = :lon,
    space_y = :lat,
    time_var = :year,
    area_method = :hexagonal,
    target_units = 16,
    exact_units = true,
    merge_small_polygons = true
)

df.s_idx = st_data.s_idx
df.year_idx = st_data.t_idx

# 3. Define and fit hierarchical BYM2 + AR1 + Fixed Effects model
m = @tbm(
    likelihood(y, family=poisson) ~
        intercept() +
        fixed(elevation) +
        random(s_idx, model=bym2) +
        random(year_idx, model=ar1),
    df,
    W = st_data.W_spatial,
    verbose = false
)

# 4. Sample posterior with NUTS
chn = sample(m, NUTS(), 300; progress=false)

# 5. Extract results and plot choropleth map
res = model_results_comprehensive(m, chn)
p_map = choropleth(st_data.au_spatial.polygons, res.effects.s_idx.structured.mean; title="BYM2 Spatial Random Effect")
p_graph = spatial_graph_plot(au=st_data.au_spatial; title="Hexagonal Grid Topology")
plot(p_map, p_graph, layout=(1, 2), size=(1000, 450))
```

---

## 10. References

1. **Besag, J.** (1974). Spatial interaction and the statistical analysis of lattice systems. *Journal of the Royal Statistical Society: Series B*, 36(2), 192–225.
2. **Du, Q., Faber, V., & Gunzburger, M.** (1999). Centroidal Voronoi tessellations: Applications and algorithms. *SIAM Review*, 41(4), 637–676.
3. **Gelfand, A. E., et al.** (2003). Spatial modeling with spatially varying coefficient processes. *Journal of the American Statistical Association*, 98(462), 387–396.
4. **Knorr-Held, L.** (2000). Bayesian modelling of inseparable space-time variation in disease risk. *Statistical Methods in Medical Research*, 9(3), 205–220.
5. **Leroux, B. G., Lei, X., & Breslow, N.** (2000). Estimation of disease rates in small areas: A new mixed model for spatial dependence. In *Statistical Models in Epidemiology, the Environment, and Clinical Trials* (pp. 179–191). Springer.
6. **Lindgren, F., Rue, H., & Lindström, J.** (2011). An explicit link between Gaussian fields and Gaussian Markov random fields: The SPDE approach. *Journal of the Royal Statistical Society: Series B*, 73(4), 423–498.
7. **Lloyd, S.** (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129–137.
8. **Rasmussen, C. E., & Williams, C. K. I.** (2006). *Gaussian Processes for Machine Learning*. MIT Press.
9. **Riebler, A., Sørbye, S. H., Simpson, D., & Rue, H.** (2016). An intuitive Bayesian spatial model for disease mapping that accounts for scaling. *Statistical Methods in Medical Research*, 25(4), 1145–1165.
10. **Roberts, D. R., et al.** (2017). Cross-validation strategies for data with temporal, spatial, hierarchical or phylogenetic structure. *Ecography*, 40(8), 913–929.
11. **Simpson, D., Rue, H., Riebler, A., Martins, T. G., & Sørbye, S. H.** (2017). Penalising model component complexity: A principled, practical approach to constructing priors. *Statistical Science*, 32(1), 1–28.