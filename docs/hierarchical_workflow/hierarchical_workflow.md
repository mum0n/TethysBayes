---
title: "Unified Hierarchical Bayesian Workflows: Multi-Tier DAGs, Surface Derivatives, EIV Priors, Habitat Suitability & ADR Telemetry in TethysBayes"
subtitle: "Standard & Advanced Modes — Tempered Power Posteriors, Full Spatial Covariance EIV, Hybrid Quadrature Resharding, Continuous Physiological HSI, and Coupled Hydrodynamic-Active Advection"
author: "TethysBayes Development Group"
date: "2026-08-24"
format:
  html:
    toc: true
    toc-depth: 4
    number-sections: true
    code-fold: show
    code-summary: "Show Julia Code"
    theme: cosmo
    highlight-style: github
  pdf:
    toc: true
    number-sections: true
    latex-engine: lualatex
bibliography: ../references.bib
csl: ../chicago-author-date.csl
---

# Unified Hierarchical Bayesian Workflows: Multi-Tier DAGs, Surface Derivatives, EIV Priors, Habitat Suitability & ADR Telemetry

## 1. Summary

Complex ecological and environmental systems are inherently **multi-scale, multi-fidelity,
and multi-tiered**. Rather than fitting all physical, oceanographic, community, and population
processes inside a single monolithic joint likelihood, modern Bayesian computation relies on
**Modular Bayesian Inference (Cut-Posterior DAG Pipelines)** [@Plummer_2015; @Jacob_2017;
@Hooten_Hefley_2019].

The **Bayesian Spatio-Temporal Models (`TethysBayes`)** ecosystem establishes a unified, principled
framework connecting:

1. **Continuous Surface Modeling & Differential Geometry**: Analytical derivation of slopes,
   aspects, profile/planform curvatures, and Bessel circular Bathymetric Position Indices
   ($\text{BPI}_r$) across 6 model families (RFF, SpectralGP, WaveletGP, SPDE, PSpline, TPS).
2. **Errors-in-Variables (EIV) Uncertainty Propagation**: Carrying upstream posterior
   uncertainties forward as latent Gaussian measurement error priors without feedback
   contamination, ranging from diagonal marginals to full spatial covariance matrices.
3. **Multi-Scale Cross-Mesh Topological Transfer Operators**: Linear geometric transfer
   matrices ($\mathbf{P} \in \mathbb{R}^{N_{\text{dest}} \times N_{\text{src}}}$) and Full Monte
   Carlo Matrix Resharding ($\mathbf{U}_{\text{dest}} = \mathbf{P} \mathbf{U}_{\text{src}}$)
   transferring posterior fields across mismatched irregular tessellations.
4. **Second-Last Tier Habitat Suitability Index (HSI) Determination**: Spatial suitability
   probabilities $\pi_{\text{HSI}}(\mathbf{s}) \in [0, 1]$, via quantile Bernoulli, continuous
   soft-sigmoid, or two-part hurdle formulations.
5. **Mechanistic Movement & Lagrangian Mark-Recapture Telemetry**: ADR population dynamics
   fusing Eulerian abundance surveys with Lagrangian individual tag tracking
   ($u_{\text{rel}} \to u_{\text{rec}}$), with optional hydrodynamic ocean current coupling.
6. **Two-Tier Input-Output Persistence**: Out-of-core DuckDB relational analytics alongside
   JLD2 binary checkpointing and RFC 7946 GeoJSON GIS export.
7. **Configurable Standard & Advanced Modes**: A single unified engine (`hierarchical_workflow.jl`)
   supports both baseline cut-posterior formulations and advanced methodological innovations via
   `PipelineOptions` flags.
8. **Individual Size-Sex-Maturity Composition & Poststratification**: A linked trio of sub-models
   (ALR-GMRF size composition, Bernoulli sex-ratio, parametric maturity ogive) fit on individual
   biological sub-sample records, combined with Tier 5 abundance via poststratification to
   reconstruct $\hat{N}(s, t, \ell, g, m)$ at arbitrary locations and times as input to a
   size-structured biological model.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               The Unified 6-Tier Hierarchical Marine Ecological DAG                    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   [ Tier 1: Continuous Bathymetry (RFF GP Surface) ]                                   │
│       │   Standard: mesh evaluation                                                    │
│       │   Advanced: Hybrid sub-polygon quadrature (Issue 5.3)                          │
│         │                                                                              │
│         ├──────────────────────────────┬──────────────────────────────┐                │
│         ▼ (Depth & Slope)              ▼ (Depth)                      ▼ (Depth)        │
│   [ Tier 2: Substrate ]         [ Tier 3: Oceanography ]       [ Tier 4: Community ]   │
│   Standard: diagonal EIV        Standard: cut-posterior         Standard: Bernoulli HSI │
│   Advanced: full Σ_x EIV (5.2)  Advanced: tempered λ (5.1)      Advanced: soft-sigmoid  │
│         │                              │                              │                │
│         │ (Full MC Matrix P_2->M)      │ (Full MC Matrix P_3->M)      │                │
│         └──────────────────────────────┴──────────────────────────────┤                │
│                                                                       ▼                │
│                                          [ Tier 5: Target Species Biomass on G_master ]│
│                                          - Gamma GMRF with propagated EIV              │
│                                                     │                                 │
│                                                     ▼ (N(s,t))                        │
│                     [ Tier 6: Individual Size-Sex-Maturity Composition ]               │
│                     - 6a: Size composition  ALR-GMRF (BYM2 x AR1)                     │
│                     - 6b: Sex ratio         Bernoulli GMRF (BYM2 x AR1)               │
│                     - 6c: Maturity ogive    Bernoulli + CW covariate + BYM2            │
│                     - 6d: Poststratification  N(s,t,l,g,m) -> tier6_composition DB     │
│                                                                                        │
│   [ Master Harmonized DuckDB Table & GeoJSON Refugia Export ]                          │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Formulations

### 2.1. Multi-Tier Cut-Posterior Theory

For a multi-tiered DAG with observed datasets $\mathbf{Y}_1, \dots, \mathbf{Y}_K$ and latent
parameters $\boldsymbol{\theta}_1, \dots, \boldsymbol{\theta}_K$, classical joint inference
evaluates:

$$
p(\boldsymbol{\theta}_1, \dots, \boldsymbol{\theta}_K \mid \mathbf{Y}_1, \dots, \mathbf{Y}_K) \propto \prod_{k=1}^K p(\mathbf{Y}_k \mid \boldsymbol{\theta}_k, \boldsymbol{\theta}_{\text{pa}(k)}) p(\boldsymbol{\theta}_k)
$$

In practice, misspecifications in downstream tiers (e.g., fisheries catchability) can leak
backward and corrupt physical parameters (e.g., bathymetry or ocean temperature). The
**cut-posterior distribution** [@Plummer_2015] severs this feedback loop:

$$
p_{\text{cut}}(\boldsymbol{\theta}_1, \dots, \boldsymbol{\theta}_K \mid \mathbf{Y}_1, \dots, \mathbf{Y}_K) = p(\boldsymbol{\theta}_1 \mid \mathbf{Y}_1) \prod_{k=2}^K p(\boldsymbol{\theta}_k \mid \mathbf{Y}_k, \boldsymbol{\theta}_{\text{pa}(k)})
$$

#### 2.1.1. Tempered Power Posterior Fractional Feedback ($\lambda \in (0, 1]$)

To balance modular robustness against downstream information loss, the standard strict cut-posterior
can be relaxed via a **tempered power posterior** [@Plummer_2015; @Jacob_2017; @Bissiri_2016],
enabled by setting `tier3_feedback_lambda > 0` in `PipelineOptions`. Let
$\boldsymbol{\theta}_{\text{phys}}$ be physical tier parameters and $\boldsymbol{\theta}_{\text{bio}}$
be biological tier parameters. The tempered joint posterior is:

$$
p_\lambda(\boldsymbol{\theta}_{\text{phys}}, \boldsymbol{\theta}_{\text{bio}} \mid \mathbf{Y}_{\text{phys}}, \mathbf{Y}_{\text{bio}}) \propto p(\boldsymbol{\theta}_{\text{phys}} \mid \mathbf{Y}_{\text{phys}}) \, p(\boldsymbol{\theta}_{\text{bio}} \mid \boldsymbol{\theta}_{\text{phys}}) \, \left[ p(\mathbf{Y}_{\text{bio}} \mid \boldsymbol{\theta}_{\text{bio}}, \boldsymbol{\theta}_{\text{phys}}) \right]^\lambda
$$

- **$\lambda = 0$**: Pure cut-posterior — no backward information flow.
- **$\lambda = 1$**: Full joint posterior — vulnerable to feedback contamination.
- **$\lambda \in (0, 1)$ (default $\lambda = 0.25$)**: Bounded, regularized feedback; allows
  dense biological surveys along unmapped slopes or frontal zones to partially refine physical
  boundaries without destabilizing upstream physical fits.

---

### 2.2. Continuous Surface Derivatives & Topographic Differential Geometry

For any continuous surface model $z(\mathbf{s}) = \mathbf{B}(\mathbf{s}) \mathbf{c}$, `TethysBayes`
computes exact spatial derivatives without numerical finite-difference grid artifacts:

1. **Gradient & Seabed Slope**:

   $$
   \nabla z(\mathbf{s}) = \begin{bmatrix} \frac{\partial z}{\partial x} \\ \frac{\partial z}{\partial y} \end{bmatrix}, \quad \text{Slope}(\mathbf{s}) = \arctan\left( \sqrt{ \left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2 } \right)
   $$

2. **Aspect (Compass Direction of Steepest Descent)**:

   $$
   \text{Aspect}(\mathbf{s}) = \operatorname{mod}\left( 180 - \operatorname{atan2}\left(\frac{\partial z}{\partial y}, \frac{\partial z}{\partial x}\right) \cdot \frac{180}{\pi}, 360 \right)
   $$

3. **Hessian & Curvatures**:

   $$
   \mathbf{H}(\mathbf{s}) = \begin{bmatrix} z_{xx} & z_{xy} \\ z_{xy} & z_{yy} \end{bmatrix}
   $$

   - **Profile Curvature** (parallel to steepest gradient):

     $$
     \kappa_{\text{profile}} = \frac{z_{xx} z_x^2 + 2 z_{xy} z_x z_y + z_{yy} z_y^2}{(z_x^2 + z_y^2) (1 + z_x^2 + z_y^2)^{3/2}}
     $$

   - **Planform Curvature** (perpendicular to gradient):

     $$
     \kappa_{\text{planform}} = \frac{z_{xx} z_y^2 - 2 z_{xy} z_x z_y + z_{yy} z_x^2}{(z_x^2 + z_y^2)^{3/2}}
     $$

4. **Continuous Bessel Circular Bathymetric Position Index ($\text{BPI}_r$)**:

   $$
   \text{BPI}_r(\mathbf{s}) = z(\mathbf{s}) - \bar{z}_{C_r(\mathbf{s})}
   $$

   For Random Fourier Features $z(\mathbf{s}) = \sum_{j=1}^M c_j \cos(\boldsymbol{\omega}_j^\top \mathbf{s} + b_j)$, the exact circular annulus mean evaluates analytically via the order-zero Bessel function of the first kind $J_0$:

   $$
   \bar{z}_{C_r(\mathbf{s})} = \sum_{j=1}^M c_j J_0(\|\boldsymbol{\omega}_j\| r) \cos(\boldsymbol{\omega}_j^\top \mathbf{s} + b_j)
   $$

---

### 2.3. Errors-in-Variables (EIV) Uncertainty Propagation

When downstream models ingest predictions $\hat{x}_i$ with standard errors $\sigma_{x, i}$ from
upstream tiers, naive point substitution induces **attenuation bias** (underestimating effect sizes).
`TethysBayes` implements latent Gaussian measurement error priors:

$$
x_i \sim \mathcal{N}(\mu_{x, i}, \sigma^2_{x, i}), \quad \eta_i = \beta_0 + \beta_1 x_i + \dots
$$

Declared via:

```julia
fixed(depth_mu, error_sd=:depth_sd)
```

#### 2.3.1. Full Spatial Covariance Matrix EIV (Advanced)

The standard diagonal EIV ignores spatial autocorrelation in upstream posterior uncertainty.
The advanced workflow (`tier2_cov_mode=:full`) propagates the **full empirical spatial covariance
matrix** $\boldsymbol{\Sigma}_{\mathbf{x}} \in \mathbb{R}^{S \times S}$ from the resharded
posterior sample ensemble $\mathbf{U}_{\text{dest}} \in \mathbb{R}^{S \times N_{\text{samples}}}$:

$$
\mathbf{U}_{\text{dest}} = \mathbf{P} \, \mathbf{U}_{\text{src}}
$$

$$
\boldsymbol{\Sigma}_{\mathbf{x}} = \frac{1}{N_{\text{samples}} - 1} \sum_{s=1}^{N_{\text{samples}}} \left(\mathbf{u}_{\text{dest}}^{(s)} - \hat{\boldsymbol{\mu}}_{\mathbf{x}}\right) \left(\mathbf{u}_{\text{dest}}^{(s)} - \hat{\boldsymbol{\mu}}_{\mathbf{x}}\right)^\top
$$

In downstream MCMC sampling, the latent covariate vector $\mathbf{x}$ follows the joint prior:

$$
\mathbf{x} \sim \mathcal{N}\left(\hat{\boldsymbol{\mu}}_{\mathbf{x}}, \, \boldsymbol{\Sigma}_{\mathbf{x}} + \delta \mathbf{I}\right)
$$

This preserves spatial autocorrelation in measurement uncertainty, preventing underestimation
of credible intervals on spatial gradients ($\nabla \mathbf{x}$) and boundary contrasts.

---

### 2.4. Cross-Mesh Topological Transfer & Full Monte Carlo Matrix Resharding

When transferring a spatial field from source graph $G_{\text{src}}$ to destination graph
$G_{\text{dest}}$, the geometric transfer operator $\mathbf{P} \in \mathbb{R}^{N_{\text{dest}} \times N_{\text{src}}}$
is constructed via areal overlap intersection:

$$
P_{ij} = \frac{\operatorname{Area}(A_{\text{dest}, i} \cap A_{\text{src}, j})}{\operatorname{Area}(A_{\text{dest}, i})}
$$

```
            Source Mesh G_src                       Destination Mesh G_dest
       ┌───────────┬───────────┐                     ┌───────────────────┐
       │ A_src, 1  │ A_src, 2  │  ── Transfer ──>   │    A_dest, 1      │
       ├───────────┼───────────┤      Matrix P       │  (Overlaps 1,2,3) │
       │ A_src, 3  │ A_src, 4  │                     └───────────────────┘
       └───────────┴───────────┘
```

#### Full Monte Carlo Matrix Resharding (`mode = :samples`)

To preserve complete posterior covariance across non-Gaussian downstream likelihoods,
the entire posterior sample matrix is transformed directly:

$$
\mathbf{U}_{\text{dest}} = \mathbf{P} \mathbf{U}_{\text{src}} \in \mathbb{R}^{N_{\text{dest}} \times S}
$$

Summary statistics are evaluated empirically without assuming normality:

```julia
resharded = reshard_spatial_field(pred_samples, au_src, au_dest; mode=:samples)
```

#### 2.4.1. Hybrid Continuous Basis-Polygon Quadrature Resharding (Advanced)

Standard piecewise-constant areal transfer assumes homogeneity within each source polygon,
smoothing over sharp bathymetric trenches and shelf breaks. The advanced mode
(`tier1_mesh_eval=:continuous`) implements **Hybrid Quadrature Basis Resharding**:

1. For each destination polygon $B_j \subset \Omega$, generate a regular sub-polygon quadrature
   grid $\{ \mathbf{s}_{j, k} \}_{k=1}^{K_q} \subset B_j$.
2. Evaluate exact continuous RFF or SpectralGP partial derivatives at each quadrature point.
3. Compute the areal-integrated spatial summary:

   $$
   \bar{z}_j = \frac{1}{K_q} \sum_{k=1}^{K_q} z(\mathbf{s}_{j, k}), \quad \bar{\nabla z}_j = \frac{1}{K_q} \sum_{k=1}^{K_q} \nabla z(\mathbf{s}_{j, k}), \quad \operatorname{Var}_{\text{sub}}(z_j) = \frac{1}{K_q} \sum_{k=1}^{K_q} (z(\mathbf{s}_{j, k}) - \bar{z}_j)^2
   $$

This retains fine-scale sub-polygon topographic roughness, curvature, and shelf-break geometry
upon transfer to coarse regional graphs.

---

### 2.5. Second-Last Tier Habitat Suitability Index (HSI) Determination

Prior to the final management/biomass tier, the **Habitat Suitability Index ($\text{HSI}$)** is
determined on the haul network $G_4$ and resharded onto the canonical master mesh $G_{\text{master}}$.

#### Standard: Quantile Threshold Bernoulli Model (`tier4_hsi_mode=:binary_quantile`)

Trawl tows with total community density below the empirical 5th percentile are classified as
unsuitable habitat ($z_i = 0$), those above as suitable ($z_i = 1$):

$$
q_p = \operatorname{quantile}(\mathbf{y}, p), \quad z_i = \begin{cases} 1 & \text{if } y_i \ge q_p \\ 0 & \text{if } y_i < q_p \end{cases}
$$

$$
z_i \sim \operatorname{Bernoulli}(\pi_i), \quad \operatorname{logit}(\pi_i) = \beta_0 + \sum_{k} \beta_k x_{k, i} + \text{sre}_{\text{spatial}}(s_i) + \text{tre}_{\text{temporal}}(t_i)
$$

The predicted latent logit field transforms to the probability scale:

$$
\text{HSI}(\mathbf{s}) = \frac{1}{1 + \exp(-\eta(\mathbf{s}))} \in [0, 1]
$$

#### Advanced: Continuous Soft-Sigmoid Physiological HSI (`tier4_hsi_mode=:soft_sigmoid`)

Instead of a step-function threshold, HSI is modeled as a continuous differentiable soft-sigmoid
function scaled by the empirical interquartile range ($\operatorname{IQR}$):

$$
\text{HSI}_i = \frac{1}{1 + \exp\left(-\kappa \cdot \frac{y_i - q_p}{\operatorname{IQR}(y)}\right)}, \quad \kappa = 2.5, \quad q_p = \operatorname{quantile}(\mathbf{y}, 0.05)
$$

On spatial graph $G_4$, the continuous HSI is modeled via a space-time GMRF:

$$
\operatorname{logit}(\text{HSI}_i) = \alpha + \beta_{\text{depth}} z_i + \beta_{\text{grain}} g_i + \beta_{\text{temp}} T_i + \phi_i + \psi_{s(i), t(i)}
$$

#### Alternative Formulations & Comparative Trade-offs

| Formulation | Mathematical Definition | Key Advantages | Best Use Case |
| :--- | :--- | :--- | :--- |
| **Quantile Bernoulli** *(Standard)* | $z_i = \mathbb{I}(y_i \ge q_p) \sim \operatorname{Bernoulli}(\pi_i)$ | Robust to extreme outliers; intuitive probability of viable habitat. | Standard trawl surveys with low-density sampling noise. |
| **Continuous Soft Sigmoid** *(Advanced)* | $\text{HSI}_i = \frac{1}{1 + \exp\left(-\kappa \cdot \frac{y_i - q_p}{\text{IQR}(\mathbf{y})}\right)}$ | Differentiable; preserves continuous variance without sharp step cutoffs. | Continuous gradient modeling across ecotones and transition zones. |
| **Two-Part Hurdle / Delta** | $\mathbb{E}[Y_i] = P(Y_i > 0) \cdot \mathbb{E}[Y_i \mid Y_i > 0]$ | Separates true ecological presence/absence from positive density. | Highly zero-inflated datasets with distinct encounter mechanisms. |
| **Environmental Niche Distance** | $\text{HSI}(\mathbf{s}) = \exp\left(-\frac{1}{2} (\mathbf{x}(\mathbf{s}) - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x}(\mathbf{s}) - \boldsymbol{\mu})\right)$ | Independent of sampling catchability; pure physiological envelope. | Multi-species bioclimatic envelope & thermal tolerance mapping. |

---

### 2.6. Individual Size-Sex-Maturity Composition & Poststratification

Tier 6 fits three linked sub-models on individual biological sub-sample records
(one measured animal per row) collected within survey tows. The output is a
poststratified abundance array $\hat{N}(s, t, \ell, g, m)$ whose marginals over
$(\ell, g, m)$ recover the Tier 5 abundance field.

#### 2.6.1. Data: Individual Biological Sub-Samples

For each positive-catch crab tow $i$, $n_{i,\text{bio}} \sim \text{Uniform}(30, 50)$
individuals are measured. Each individual $j$ yields:
- **Carapace width** $\text{CW}_{ij}$ (mm): log-normal, continuous.
- **Sex** $g_{ij} \in \{0, 1\}$ (0=female, 1=male): Bernoulli.
- **Maturity** $m_{ij} \in \{0, 1\}$ (0=immature, 1=mature): Bernoulli, sex-specific logistic ogive.

Individuals are binned into $L = 5$ carapace width classes:
$$
[0, 40),\ [40, 60),\ [60, 80),\ [80, 100),\ [100, \infty)\ \text{mm CW}
$$
with $L$ and the break vector configurable via `PipelineOptions.tier6_bin_breaks`.

#### 2.6.2. Sub-model 6a: Size Composition via Additive Log-Ratio GMRF

Let $\mathbf{n}_i = (n_{i1}, \ldots, n_{iL}) \in \mathbb{Z}^L_{\ge 0}$ be the
count of individuals in each size bin for tow $i$, with $N_i = \sum_\ell n_{i\ell}$.
Size composition is modelled via $L - 1$ independent GMRF sub-models on the
additive log-ratio (ALR) scale with the $L$-th bin as reference:

$$
\text{alr}_{i\ell} = \log\!\frac{n_{i\ell} + 0.5}{n_{iL} + 0.5}, \quad \ell = 1, \ldots, L-1
$$

$$
\text{alr}_{i\ell} \sim \mathcal{N}(\eta_{i\ell},\, \sigma^2_\ell)
$$

$$
\eta_{i\ell} = \alpha_\ell
  + \beta_\ell^{\text{hsi}} \tilde{\text{HSI}}_i
  + \beta_\ell^{\text{temp}} \tilde{T}_i
  + \beta_\ell^{\text{depth}} \tilde{z}_i
  + \phi_\ell(s_i)
  + \psi_\ell(s_i, t_i)
$$

where $\tilde{\cdot}$ denotes EIV uncertainty-propagated covariates from upstream
tiers, $\phi_\ell$ is a BYM2 spatial random effect, and $\psi_\ell$ is a
BYM2$\otimes$AR1 spatiotemporal random effect. Predicted size proportions on the
master grid are recovered by inverse-ALR (softmax):

$$
\hat{\pi}_\ell(u, t) = \frac{\exp(\hat{\eta}_\ell(u, t))}
  {1 + \sum_{k=1}^{L-1} \exp(\hat{\eta}_k(u, t))}
$$

#### 2.6.3. Sub-model 6b: Sex Ratio (Bernoulli BYM2$\otimes$AR1)

Individual sex is modelled at the observation level with EIV environmental
covariates and a spatiotemporal random effect:

$$
g_{ij} \sim \operatorname{Bernoulli}(\rho_i), \quad
\operatorname{logit}(\rho_i) = \alpha_g
  + \beta_g^{\text{hsi}} \tilde{\text{HSI}}_i
  + \beta_g^{\text{temp}} \tilde{T}_i
  + \phi_g(s_i, t_i)
$$

The fitted $\hat{\rho}(u, t)$ is the predicted probability of male at each
$(u, t)$ node of the master grid.

#### 2.6.4. Sub-model 6c: Maturity Ogive (Bernoulli + CW Covariate + BYM2)

Individual maturity follows a parametric sex-specific logistic ogive with a
spatial random effect capturing local deviations (e.g. growth effects of
temperature, food availability):

$$
m_{ij} \sim \operatorname{Bernoulli}(\mu_{ij}), \quad
\operatorname{logit}(\mu_{ij}) = \alpha_m
  + \beta_m^{\text{cw}} \cdot \text{CW}_{ij}^{\text{std}}
  + \beta_m^{\text{sex}} \cdot g_{ij}
  + \phi_m(s_i)
$$

where $\text{CW}_{ij}^{\text{std}} = (\text{CW}_{ij} - 65) / 20$ is the
standardised carapace width. The synthetic ogive uses L50 = 65 mm (male) and
L50 = 45 mm (female) as the generating inflection points.

The fitted maturity probability at bin midpoint $\bar{\ell}$ for sex $g$ is:

$$
\hat{\mu}(\bar{\ell}, g) = \frac{1}{1 + \exp\!\left(-\left(
  \hat{\alpha}_m + \hat{\beta}_m^{\text{cw}} \cdot \frac{\bar{\ell} - 65}{20}
  + \hat{\beta}_m^{\text{sex}} \cdot g \right)\right)}
$$

#### 2.6.5. Sub-step 6d: Poststratified Abundance Reconstruction

For each posterior draw $s$, spatial unit $u$, year $t$, size bin $\ell$,
sex $g$, and maturity state $m$:

$$
\hat{N}^{(s)}(u, t, \ell, g, m) =
  \hat{N}^{(s)}_{\text{T5}}(u, t)
  \cdot \hat{\pi}^{(s)}_\ell(u, t)
  \cdot \begin{cases}
    \hat{\rho}^{(s)}(u, t) & g = 1 \\
    1 - \hat{\rho}^{(s)}(u, t) & g = 0
  \end{cases}
  \cdot \begin{cases}
    \hat{\mu}^{(s)}(\bar{\ell}, g) & m = 1 \\
    1 - \hat{\mu}^{(s)}(\bar{\ell}, g) & m = 0
  \end{cases}
$$

This structure preserves full posterior uncertainty by operating sample-wise.
Summing over all $\ell, g, m$ recovers $\hat{N}^{(s)}_{\text{T5}}(u, t)$ within
numerical precision (marginal consistency check). Results are stored in the
long-format DuckDB table `tier6_composition`:

```sql
SELECT unit_id, year, size_bin, sex, maturity, n_mean, n_lower, n_upper
FROM tier6_composition
WHERE year = 2023 AND sex = 1  -- mature males, 2023
ORDER BY unit_id, size_bin;
```

This table is the direct input to a size-structured biological model
(e.g., dynamic Leslie matrix, delay-difference, or von Bertalanffy growth
applied spatially).

---

### 2.7. Mechanistic Advection-Diffusion-Reaction (ADR) & Telemetry

On the canonical master mesh $G_{\text{master}}$, the population density state evolves via:

$$
\mathbf{C}_t = \mathbf{C}_{t-1} + \Delta t \left( -v \mathbf{A} \mathbf{C}_{t-1} - \operatorname{diag}(\mathbf{D}) \mathbf{L} \mathbf{C}_{t-1} + r \mathbf{C}_{t-1} \odot \left(1 - \frac{\mathbf{C}_{t-1}}{K}\right) \right) + \sigma \boldsymbol{\epsilon}_t
$$

where $\mathbf{L} = \mathbf{D}_{\text{deg}} - \mathbf{W}$ is the discrete graph Laplacian, and
$\mathbf{A}$ is the directed advection operator driven by the HSI gradient:

$$
\mathbf{A} = \mathbf{D}_{\text{in}} - \mathbf{W}_{\text{dir}}^\top
$$

The directed weight $W_{\text{dir}, ij}$ supports multiple functional relationships (`:exponential`,
`:linear`, `:logistic`). For the default `:exponential` relationship:

$$
W_{\text{dir}, ij} = \begin{cases} W_{ij} \cdot \exp(\gamma \cdot (\text{HSI}_j - \text{HSI}_i)) & \text{if } \text{HSI}_j > \text{HSI}_i \\ 0 & \text{otherwise} \end{cases}
$$

#### Advanced: Coupled Hydrodynamic-Active Advection (`movement_mode=:coupled_hydrodynamic`)

Animal movement is driven by two vector fields:

1. **Passive Hydrodynamic Drift**: Background ocean current velocity $\mathbf{u}_{\text{ocean}}(\mathbf{s}, t)$.
2. **Active Chemotactic Advection**: Gradient ascent $\mathbf{v}_{\text{active}}(\mathbf{s}) = \gamma \nabla \text{HSI}(\mathbf{s})$.

The total advective velocity field is:

$$
\mathbf{v}_{\text{total}}(\mathbf{s}, t) = \mathbf{u}_{\text{ocean}}(\mathbf{s}, t) + \gamma \nabla \text{HSI}(\mathbf{s})
$$

On discrete graph $G = (V, E)$, the directed weight matrix incorporates both hydrodynamic and
habitat gradient terms:

$$
W_{\text{dir}, ij} = W_{ij} \cdot \exp\left(\gamma (\text{HSI}_j - \text{HSI}_i) + \frac{\mathbf{u}_{\text{ocean}}(\mathbf{c}_i) \cdot (\mathbf{c}_j - \mathbf{c}_i)}{\|\mathbf{c}_j - \mathbf{c}_i\|}\right)
$$

The row-stochastic Markov transition kernel $\mathbf{\Gamma} \in \mathbb{R}^{S \times S}$ with
isotropic diffusion fraction $\omega_{\text{diff}} \in [0, 1]$:

$$
\Gamma_{ij} = (1 - \omega_{\text{diff}}) \frac{W_{\text{dir}, ij}}{\sum_{k} W_{\text{dir}, ik}} + \omega_{\text{diff}} \frac{W_{ij}}{\sum_{k} W_{ik}}
$$

For mark-recapture telemetry encounters ($u_{\text{rel}} \to u_{\text{rec}}$ across $k$ time steps),
the transition probability kernel evaluates via matrix exponentiation:

$$
\pi_m(u_{\text{rec}} \mid u_{\text{rel}}, k, z_m) = \frac{\left[\mathbf{\Gamma}^k\right]_{u_{\text{rel}}, u_{\text{rec}}} \cdot \exp(\beta_{\text{het}} z_m)}{\sum_{j=1}^S \left[\mathbf{\Gamma}^k\right]_{u_{\text{rel}}, j} \cdot \exp(\beta_{\text{het}} z_m)}
$$

---

## 3. Segmented Architecture, Update Schedules & Smart Restarts

In operational marine science and ecological monitoring, the data collection frequency varies
across several orders of magnitude:

| Tier / Component | Data Source | Primary Horizon | Update Frequency | Computational Cost |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Bathymetry** | Multibeam Sonar & Lidar | Spatial (Static) | **Decadal / Multi-Year** | High (Continuous RFF GP) |
| **Tier 2: Substrate** | Benthic Grabs & Core Samples | Spatial (Static) | **Multi-Year (5–10 yrs)** | Moderate (BYM2 GMRF) |
| **Tier 3: Temperature** | CTD Casts & Mooring Sensors | Spatiotemporal | **Seasonal / Annual** | High (Kronecker BYM2 $\otimes$ AR1) |
| **Tier 4: Community** | Multi-Species Trawl Hauls | Spatiotemporal | **Annual Ecosystem Survey** | High (HSI + PCA GMRFs) |
| **Tier 5: Target Species** | Directed Trap / Trawl Surveys | Spatiotemporal | **Annual Assessment** | Moderate (Gamma EIV GMRF) |
| **Tier 6: Composition** | Individual Biological Sub-Samples | Spatiotemporal | **Annual Survey** | Moderate (3 linked GMRFs + poststr.) |
| **Step 7: Telemetry ADR** | Acoustic Tag Recoveries | Spatiotemporal | **Continuous / Annual** | Fast (Matrix Exponentiation) |
| **Step 8: Harmonization** | DuckDB Relational Aggregation | Master Mesh | **On-Demand SQL** | Sub-second (Zero-Copy) |

When an annual survey introduces new observations for Tier 5 or Tier 4, re-fitting upstream
physical models (bathymetry, substrate) wastes hours of compute. `hierarchical_workflow.jl`
implements a **Segmented Execution & Smart Restart** engine backed by DuckDB metadata tables.

---

### 3.1. Data Provenance & Trait Fingerprinting

Each dataset is hashed and fingerprinted via `compute_data_traits()` before modeling:

- **Dimensions**: Row count ($N$) and column count ($K$).
- **Spatial Envelope**: Bounding box coordinates $(x_{\min}, x_{\max}, y_{\min}, y_{\max})$.
- **Temporal Horizon**: Survey years $[t_{\min}, t_{\max}]$ and number of survey years.
- **Content Hash**: Deterministic column-summary hash identifying data mutations.

The execution state is tracked in the `pipeline_manifest` DuckDB table:

```sql
SELECT tier_id, tier_name, status, update_frequency, last_run, data_rows, upstream_deps
FROM pipeline_manifest;
```

---

### 3.2. Automated Status Inspection & Restart Recommendation

Before execution, `check_pipeline_status(db_path, datasets)` inspects the manifest, verifies
persisted `.jld2` bundles, checks upstream dependency timestamps, and recommends actionable
restart plans:

```
===================================================================================================
TethysBayes HIERARCHICAL WORKFLOW: SEGMENT STATUS & RESTART RECOMMENDATIONS
===================================================================================================
Tier / Segment               Update Schedule      Status     Last Run (UTC)         Action
---------------------------------------------------------------------------------------------------
Tier 1: Bathymetry           Decadal / Multi-Year CACHED     2026-08-24 07:15:22    [SKIP]
Tier 2: Sediment Substrate   Multi-Year           CACHED     2026-08-24 07:16:10    [SKIP]
Tier 3: Ocean Temperature    Seasonal / Annual    CACHED     2026-08-24 07:18:40    [SKIP]
Tier 4: Community & HSI      Annual Survey        CACHED     2026-08-24 07:22:05    [SKIP]
Tier 5: Target Snow Crab     Annual Survey        STALE      2026-08-23 12:00:00    [RUN] Data Changed
Step 7: Telemetry ADR        Post-Assessment      STALE      -                      [RUN] Initial Run
Step 8: Master Refugia SQL   Final Integration    STALE      -                      [RUN] Initial Run
===================================================================================================
```

---

### 3.3. Unified Configuration Options & Tier Flags (`PipelineOptions`)

The workflow engine is fully parameterizable, allowing selection between baseline standard
formulations and advanced methodological innovations globally or per-tier:

```julia
opts = PipelineOptions(
    mode = :advanced,                      # Global preset: :standard vs. :advanced
    tier1_mesh_eval = :continuous,         # :mesh vs. :continuous (exact basis derivatives)
    tier2_cov_mode = :full,                # :diagonal vs. :full (empirical spatial covariance EIV)
    tier3_feedback_lambda = 0.25,          # 0.0 (strict cut) to 1.0 (joint), 0.25 (tempered)
    tier4_hsi_mode = :soft_sigmoid,        # :binary_quantile vs. :soft_sigmoid
    tier4_hsi_kappa = 3.0,                 # Soft-sigmoid slope parameter (κ)
    movement_mode = :coupled_hydrodynamic  # :gradient_taxis vs. :coupled_hydrodynamic
)
```

| Option | Standard (`:standard`) | Advanced (`:advanced`) |
| :--- | :--- | :--- |
| `tier1_mesh_eval` | `:mesh` | `:continuous` |
| `tier2_cov_mode` | `:diagonal` | `:full` |
| `tier3_feedback_lambda` | `0.0` | `0.25` |
| `tier4_hsi_mode` | `:binary_quantile` | `:soft_sigmoid` |
| `movement_mode` | `:gradient_taxis` | `:coupled_hydrodynamic` |

---

### 3.4. Single-Call Direct Segment Execution & CLI Control

Scientists can run or update any individual segment directly in a single function or CLI
invocation without running the entire upstream pipeline. Missing prerequisites are loaded
automatically from the DuckDB/JLD2 cache:

#### Direct Julia API Calls (`update_segment!`)

```julia
# Update ONLY Tier 5 (Target Snow Crab) when annual survey data is updated:
update_segment!(:tier5; force=true)

# Update ONLY Tier 4 using soft-sigmoid HSI formulation:
update_segment!(:tier4; opts=PipelineOptions(tier4_hsi_mode=:soft_sigmoid))

# Update ONLY Step 7 with coupled hydrodynamic-active advection:
update_segment!(:telemetry; opts=PipelineOptions(movement_mode=:coupled_hydrodynamic))

# Update multiple targeted tiers programmatically:
run_hierarchical_workflow(steps=[:tier4, :tier5], opts=PipelineOptions(mode=:advanced), force=true)
```

#### Targeted Command-Line Execution

```bash
# 1. Inspect status and restart recommendations without running MCMC
julia --project=. docs/hierarchical_workflow/hierarchical_workflow.jl --status

# 2. Run full workflow with standard baseline settings
julia --project=. docs/hierarchical_workflow/hierarchical_workflow.jl --standard

# 3. Run full workflow with advanced methodological innovations
julia --project=. docs/hierarchical_workflow/hierarchical_workflow.jl --mode=advanced
 
# 4. Run/update ONLY Tier 5 Snow Crab
julia --project=. docs/hierarchical_workflow/hierarchical_workflow.jl --tier5

# 5. Run/update ONLY Tier 4 with soft-sigmoid HSI
julia --project=. docs/hierarchical_workflow/hierarchical_workflow.jl --tier4 --hsi=soft

# 6. Run/update multiple specific tiers
julia --project=. docs/hierarchical_workflow/hierarchical_workflow.jl --steps=tier4,tier5

# 7. Force full re-fitting of all tiers regardless of cache
julia --project=. docs/hierarchical_workflow/hierarchical_workflow.jl --force

# 8. Clean DuckDB database and re-initialize from scratch
julia --project=. docs/hierarchical_workflow/hierarchical_workflow.jl --clean
```

---

### 3.5. Step-by-Step Modular Pipeline Breakdown

#### Step 0: Multi-Tier Ecosystem & Telemetry Data Generation

Generates synthetic multi-scale datasets via `TethysBayes_data(type="hierarchical")` and
`TethysBayes_data(type="telemetry")`:

- **Bathymetry**: $N = 1,000$ continuous soundings.
- **Substrate**: $N = 500$ sediment grab stations.
- **Temperature**: $N = 1,000$ CTD casts across 10 years.
- **Community**: $N = 15,000$ records across 30 species and 500 tows.
- **Target Species**: $N = 400$ snow crab survey tows.
- **Telemetry**: $N = 200$ mark-recapture encounters across 100 individuals.

#### Step 1: Multi-Mesh Topological Graph Construction

Constructs independent, topology-preserving spatial tessellations for each tier:

- $G_2$ (Substrate): $S = 25$ units.
- $G_3$ (Oceanography): $S = 40$ units.
- $G_4$ (Community & HSI): $S = 50$ units.
- $G_{\text{master}}$ (Assessment & ADR): $S = 80$ units.

#### Step 2: Tier 1 Bathymetric Surface & Analytical Differential Geometry

Fits a continuous Random Fourier Features (RFF) GP and evaluates analytical derivatives
(`run_tier1_depth`). In advanced mode, evaluates exact continuous basis at destination polygon
quadrature nodes, bypassing piecewise-constant approximation error:

```julia
m_depth = @tbm(likelihood(depth, family=gaussian) ~ intercept() + random(s_x, s_y, model=rff), df_bathy)
chn_depth = sample(m_depth, NUTS(40, 0.65), 80; progress=false)

# Analytical slopes, curvatures, and circular BPI on master network
pred_depth_master = TethysBayes_surface_derivatives(
    m_depth, chn_depth,
    DataFrame(s_x=[c[1] for c in au_master.centroids], s_y=[c[2] for c in au_master.centroids]);
    metrics = [:slope, :curvature, :bpi], radii = [10.0, 25.0]
)
```

#### Step 3: Tier 2 Sediment Substrate Modeling with EIV

Propagates depth and slope uncertainties to network $G_2$ ($S = 25$) via latent Gaussian EIV
priors (`run_tier2_substrate`). Standard: diagonal EIV. Advanced: full empirical spatial
covariance $\boldsymbol{\Sigma}_{\mathbf{x}}$:

```julia
m_sub = @tbm(
    likelihood(grain_size_phi, family=gaussian) ~
        intercept() +
        fixed(depth_mu, error_sd=:depth_sd) +
        fixed(slope_mu, error_sd=:slope_sd) +
        random(s_idx, model=bym2),
    df_sub, W=au_sub.W
)
```

Reshards full posterior sample matrices ($\mathbf{U}_{\text{dest}} = \mathbf{P} \mathbf{U}_{\text{src}}$)
to $G_4$ and $G_{\text{master}}$.

#### Step 4: Tier 3 Oceanographic Temperature with Space-Time Kronecker Interactions

Models bottom water temperature on $G_3$ ($S = 40$) using harmonic seasonality and a separable
space-time GMRF (`run_tier3_temperature`). Standard: strict cut-posterior. Advanced: tempered
power posterior ($\lambda = 0.25$):

```julia
m_temp = @tbm(
    likelihood(bottom_temperature, family=gaussian) ~
        intercept() +
        fixed(depth_mu, error_sd=:depth_sd) +
        random(month, model=harmonic, period=12.0) +
        (random(s_idx, model=bym2) ⊗ random(year_idx, model=ar1)),
    df_temp, W=au_temp.W
)
```

#### Step 5: Tier 4 Community Ordination & HSI Determination

On haul network $G_4$ ($S = 50$), extracts major community axes (PC1, PC2) via Hellinger
ordination and determines habitat suitability (`run_tier4_community`). Standard: quantile
threshold Bernoulli. Advanced: continuous soft-sigmoid Beta GMRF:

```julia
# Standard:
m_hsi = @tbm(
    likelihood(habitat_suitable, family=bernoulli) ~
        intercept() +
        fixed(depth_mu, error_sd=:depth_sd) +
        fixed(grain_mu, error_sd=:grain_sd) +
        fixed(temp_mu, error_sd=:temp_sd) +
        (random(s_idx, model=bym2) ⊗ random(year_idx, model=ar1)),
    haul_meta, W=au_comm.W
)

# Advanced (soft_sigmoid, HSI ∈ (0,1), Beta family):
m_hsi = @tbm(
    likelihood(HSI_target, family=beta) ~
        intercept() +
        fixed(depth_mu, error_sd=:depth_sd) +
        fixed(grain_mu, error_sd=:grain_sd) +
        fixed(temp_mu, error_sd=:temp_sd) +
        (random(s_idx, model=bym2) ⊗ random(year_idx, model=ar1)),
    haul_meta, W=au_comm.W
)
```

#### Step 6: Tier 5 Snow Crab Biomass on Master Network $G_{\text{master}}$

Integrates all upstream resharded environmental predictors, community gradients, and HSI into
the final biomass assessment model (`run_tier5_biomass`):

```julia
m_crab = @tbm(
    likelihood(total_biomass_kg, family=gamma, log_offsets=log_effort) ~
        intercept() +
        fixed(hsi_mu, error_sd=:hsi_sd) +
        fixed(grain_mu, error_sd=:grain_sd) +
        fixed(temp_mu, error_sd=:temp_sd) +
        fixed(pc1_mu, error_sd=:pc1_sd) +
        fixed(pc2_mu, error_sd=:pc2_sd) +
        (random(s_idx, model=bym2) ⊗ random(year_idx, model=ar1)),
    df_crab_pos, W=au_master.W
)
```

#### Step 7: Telemetry Transition Kernel, CRW Dispersal & Visualizations

Computes the Markovian transition probability kernel $\mathbf{\Gamma}(\text{HSI}, W)$, simulates
individual movement paths with directional persistence, and constructs the publication dashboard
(`run_step7_telemetry_movement`). Standard: habitat gradient taxis. Advanced: coupled
hydrodynamic-active advection:

```julia
Gamma_master = compute_suitability_transition_kernel(
    community_pred_df.hsi_mean, au_master.W;
    sensitivity = 1.2, diffusion_weight = 0.3, relationship = :exponential
)

sim_paths = simulate_posterior_trajectories(
    Gamma_master, fill(1, 25), 15, au_master;
    rho_persistence = 1.2
)
```

#### Step 8: Master Harmonization Table & Zero-Copy DuckDB SQL Queries

Saves integrated multi-tier predictions into `master_hierarchical_project.duckdb`, runs
high-speed out-of-core SQL habitat queries, and exports RFC 7946 GeoJSON spatial maps
(`run_step8_master_harmonization`):

```julia
df_critical_habitat = query_duckdb(db_path, """
    SELECT unit_id, s_x, s_y, hsi_mean, biomass_mean, temp_mean, substrate_mean, depth_mean
    FROM master_harmonized_summary
    WHERE hsi_mean >= 0.70
    ORDER BY biomass_mean DESC
""")
```

---

## 4. Comparative Summary: Standard vs. Advanced Mode

| Feature / Domain | Standard (`:standard`) | Advanced (`:advanced`) |
| :--- | :--- | :--- |
| **Inference Modularization** | Strict Cut-Posterior ($p_{\text{cut}}$) | Tempered Power Posterior ($\lambda = 0.25$) |
| **Covariate Uncertainty EIV** | Marginal Diagonal Variances ($\sigma_i^2$) | Full Spatial Covariance Matrix ($\boldsymbol{\Sigma}_{\mathbf{x}}$) |
| **Cross-Mesh Resharding** | Piecewise-Constant Linear Transfer ($\mathbf{P} \mathbf{u}$) | Hybrid Continuous Basis-Polygon Quadrature |
| **Habitat Suitability HSI** | Binary Bernoulli Quantile Threshold ($p = 0.05$) | Continuous Physiological Soft-Sigmoid ($\operatorname{IQR}$-scaled) |
| **Movement & Advection** | Pure Habitat Gradient Ascent ($\mathbf{v} \propto \nabla \text{HSI}$) | Coupled Hydrodynamic-Active ($\mathbf{v} = \mathbf{u}_{\text{ocean}} + \gamma \nabla \text{HSI}$) |
| **Persistence & Analytics** | DuckDB Tables & Out-of-Core Queries | Harmonized Refugia Tables, GIS GeoJSON & Dashboard |

---

## 5. API Reference

| Function | Module | Description |
| :--- | :--- | :--- |
| `TethysBayes_data` | `src/data.jl` | Generates synthetic multi-tier hierarchical marine ecosystem datasets. |
| `compute_data_traits` | `src/hierarchical.jl` | Computes statistical trait fingerprints (dimensions, bounds, hash) for datasets. |
| `init_pipeline_manifest!` | `src/hierarchical.jl` | Creates and initializes the `pipeline_manifest` DuckDB table. |
| `check_pipeline_status` | `src/hierarchical.jl` | Evaluates DAG dependencies and hashes to recommend segment restart actions. |
| `is_tier_up_to_date` | `src/hierarchical.jl` | Validates if a tier's serialized model and DuckDB results are valid and current. |
| `write_tier_table!` / `read_tier_table` | `src/hierarchical.jl` | High-speed relational table reading and writing in DuckDB. |
| `map_to_units` | `src/hierarchical.jl` | Fast Euclidean mapping of point coordinates to discrete polygon unit centroids. |
| `TethysBayes_pipeline` | `src/pipeline.jl` | Declarative multi-tier DAG pipeline orchestrator. |
| `TethysBayes_surface_derivatives` | `src/derivatives.jl` | Computes continuous surface derivatives ($\nabla z, \nabla^2 z$, slope, curvature, BPI). |
| `compute_network_transfer_matrix` | `src/pipeline.jl` | Constructs linear transfer matrix $\mathbf{P}$ between mismatched meshes. |
| `reshard_spatial_field` | `src/pipeline.jl` | Reshards posterior moments or sample matrices ($U_{\text{dest}} = P U_{\text{src}}$). |
| `summarize_sample_matrix` | `src/pipeline.jl` | Evaluates empirical quantiles and credible intervals from sample matrices. |
| `movement(...)` | `src/components/movement.jl` | Declarative ADR movement and mark-recapture telemetry formula module. |
| `compute_suitability_transition_kernel` | `src/movement.jl` | Computes Markov transition kernel $\mathbf{\Gamma}$ from HSI and graph $W$. |
| `simulate_posterior_trajectories` | `src/movement.jl` | Simulates individual paths with directional persistence (Correlated Random Walk). |
| `calculate_regional_connectivity` | `src/movement.jl` | Aggregates fine-scale transitions into macro-regional migration matrices. |
| `save_TethysBayes_bundle` | `src/input_output.jl` | Serializes models (.jld2) and normalized results tables (.duckdb). |
| `query_duckdb` | `src/input_output.jl` | Executes high-performance out-of-core SQL queries directly against DuckDB tables. |
| `export_spatial_results_to_geojson` | `src/input_output.jl` | Exports spatial polygon geometries and predictions to standard RFC 7946 GeoJSON. |

---

## 6. Methodological Criticisms, Limitations & Directions for Improvement

### 6.1. Cut-Posterior Modular Inference vs. Full Joint Bayesian Coherence

1. **Severing the Feedback Loop & The Likelihood Principle**: In modular Bayesian inference, the
   cut-posterior $p_{\text{cut}}(\boldsymbol{\theta}_1, \boldsymbol{\theta}_2 \mid \mathbf{Y}_1, \mathbf{Y}_2) = p(\boldsymbol{\theta}_1 \mid \mathbf{Y}_1) p(\boldsymbol{\theta}_2 \mid \mathbf{Y}_2, \boldsymbol{\theta}_1)$
   intentionally blocks backward information flow from $\mathbf{Y}_2$ to $\boldsymbol{\theta}_1$
   [@Plummer_2015; @Jacob_2017]. While this prevents downstream biological misspecifications from
   corrupting upstream physical models, it formally violates Bayesian coherency and the Likelihood
   Principle. High-density catches along an unmapped submarine trench cannot refine the upstream
   bathymetric depth surface.
2. **Directions for Improvement**: The tempered power posterior ($\lambda \in (0, 1)$, Section 2.1.1)
   is implemented as a configurable option (`tier3_feedback_lambda`). Remaining directions include:
   - Sequential Monte Carlo chains with modular bridging weights.
   - Variational cut-posterior approximations for high-dimensional joint inference.

---

### 6.2. Errors-in-Variables Off-Diagonal Spatial Covariance

1. **Marginal vs. Joint Spatial Measurement Uncertainty**: The standard diagonal EIV
   ($x_i \sim \mathcal{N}(\hat{\mu}_{x, i}, \hat{\sigma}^2_{x, i})$) neglects off-diagonal
   spatial covariance $\boldsymbol{\Sigma}_{\mathbf{x}}$ across adjacent spatial units. This
   can slightly underestimate the joint uncertainty of spatial gradients ($\nabla x$) or contrasts.
2. **Directions for Improvement**: Full spatial covariance EIV (`tier2_cov_mode=:full`) is
   implemented. Further improvements include:
   - Propagating full GMRF precision matrices $\mathbf{Q}_x$ or low-rank Cholesky factors
     $\mathbf{L}_x \mathbf{z}$ rather than dense empirical covariance estimates.
   - Direct Monte Carlo sample indexing during MCMC, drawing upstream posterior samples
     $\mathbf{u}_{\text{src}}^{(s)}$ sequentially without Gaussian approximation.

---

### 6.3. Piecewise-Constant Cross-Mesh Transfer Operators

1. **Areal Averaging across Sharp Environmental Discontinuities**: The linear geometric transfer
   operator $P_{ji} = \frac{\operatorname{Area}(B_j \cap A_i)}{\operatorname{Area}(B_j)}$ assumes
   piecewise-constant fields across source polygons. Coarse source tessellations relative to local
   physical gradients smooth out sub-polygon topographic variation (shelf edges, canyons, frontal
   boundaries).
2. **Directions for Improvement**: Hybrid basis-polygon resharding (`tier1_mesh_eval=:continuous`)
   is implemented. Further improvements include:
   - **Wasserstein Barycentric Resharding**: Use regularized optimal transport to transfer spatial
     probability measures across mismatched graphs, preserving sharp spatial mass boundaries.

---

### 6.4. Habitat Suitability Index Discretization & Advective Dynamics

1. **Threshold Discretization vs. Continuous Ecological Gradients**: The default quantile
   threshold Bernoulli classification imposes an arbitrary binary boundary discarding continuous
   density variation. Pure habitat gradient ascent neglects spatial memory, ontogenetic migrations,
   site fidelity, and passive hydrodynamic advection.
2. **Directions for Improvement**: Continuous soft-sigmoid HSI (`tier4_hsi_mode=:soft_sigmoid`)
   and coupled hydrodynamic advection (`movement_mode=:coupled_hydrodynamic`) are implemented.
   Further improvements include:
   - **Dynamic Energy Budget (DEB) Suitability**: Link HSI directly to metabolic thermal
     performance curves and prey encounter rates rather than empirical catch quantiles.

---

### 6.5. Computational Scalability for High-Dimensional Spatiotemporal Graphs

1. **Propagator Inversion & Kronecker Memory Bottlenecks**: For large-scale territorial meshes
   ($S > 5,000$ spatial units, $T > 50$ temporal steps), computing dense LU factorizations of
   the ADR propagator matrix $\mathbf{M} = \mathbf{I} - \Delta t (v \mathbf{A} + \operatorname{diag}(\mathbf{D}) \mathbf{L})$
   inside Turing MCMC sampling becomes computationally demanding. High-dimensional Kronecker
   products ($\mathbf{Q}_{st} = \mathbf{Q}_t \otimes \mathbf{Q}_s$) challenge CPU memory limits.
2. **Directions for Improvement**:
   - **Matrix-Free Krylov Solvers**: Use Arnoldi/Lanczos iterations or Chebyshev polynomial
     approximations to evaluate $\exp(-\Delta t \mathbf{M}) \mathbf{C}$ without dense factorizations.
   - **GPU Acceleration**: Dispatch sparse graph Laplacian and advection operations to GPU via
     Julia's `KernelAbstractions.jl` and `CUDA.jl`.

---

## 7. References

::: {#refs}
:::
