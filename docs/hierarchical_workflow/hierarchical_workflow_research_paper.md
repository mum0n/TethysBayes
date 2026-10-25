---
title: "A Modular Bayesian Spatiotemporal Framework for Multi-Tier Marine Ecosystem Modeling: From Continuous Physical Fields and Errors-in-Variables Propagation to Demographic Poststratification and Mechanistic Movement"
subtitle: "Methodological Foundations, Operational Architecture, Resolved Bottlenecks, and Unresolved Frontiers in the TethysBayes Hierarchical Workflow"
author: "TethysBayes Hierarchical Workflow Group"
date: "2026-08-27"
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

# Abstract & Executive Summary

Marine ecosystems are inherently hierarchical, multi-scale, and multi-fidelity systems
governed by coupled physical, oceanographic, biogeochemical, community, and population
processes. Traditional marine spatial modeling relies either on monolithic joint Bayesian
likelihoods—which suffer from severe computational intractability and catastrophic
feedback contamination across misspecified sub-models—or naive two-stage plug-in regressions
that ignore upstream prediction uncertainty, inducing severe attenuation bias. Furthermore,
standard approaches struggle with disparate observational update frequencies, ranging from
decadal bathymetric soundings to annual trawl surveys and high-frequency telemetry.

Here, we present the theoretical foundations, computational architecture, ecological
utility, and open methodological frontiers of the unified **6-tier hierarchical Bayesian
workflow** implemented within the **Bayesian Spatio-Temporal Models (`TethysBayes`)** ecosystem.
Our framework formalizes marine ecosystem dynamics as a Directed Acyclic Graph (DAG) of
modular Bayesian cut-posterior and tempered power posterior components:

1. **Tier 1 (Continuous Physical Surfaces & Differential Geometry)**: Analytical derivation
   of seabed slopes, aspects, profile/planform curvatures, and continuous Bessel circular
   Bathymetric Position Indices ($\text{BPI}_r$) across continuous Gaussian process bases.
2. **Tier 2 (Sediment Granulometry & Errors-in-Variables)**: Latent Gaussian
   Errors-in-Variables (EIV) priors carrying both diagonal and full empirical spatial
   covariance matrices ($\boldsymbol{\Sigma}_{\mathbf{x}}$) forward without feedback leak.
3. **Tier 3 (Hydrographic Oceanography & Spatiotemporal Kronecker Fields)**: Harmonic
   seasonal decomposition coupled with separable space-time GMRFs ($\mathbf{Q}_s \otimes
   \mathbf{Q}_t$) and optional tempered fractional feedback ($\lambda \in [0, 1]$).
4. **Tier 4 (Community Ordination & Continuous Habitat Suitability)**: Hellinger-transformed
   multi-species ordination linked to continuous physiological soft-sigmoid Habitat
   Suitability Index (HSI) surfaces via Beta GMRFs.
5. **Tier 5 (Target Species Biomass Assessment)**: Overdispersed Gamma GMRFs on master
   tessellations driven by upstream resharded environmental predictors with propagated EIV.
6. **Tier 6 (Size-Sex-Maturity Composition & Demographic Poststratification)**: A linked
   trio of sub-models (Additive Log-Ratio GMRF size composition, Bernoulli sex ratio,
   parametric maturity ogive) integrated via poststratification to reconstruct fine-scale
   demographic densities $\hat{N}(s, t, \ell, g, m)$.
7. **Step 7 (Mechanistic Advection-Diffusion-Reaction Telemetry)**: Eulerian-Lagrangian
   data fusion mapping continuous ADR PDEs onto discrete Centroidal Voronoi Tessellation
   (CVT) graphs, coupling directional movement to HSI gradients and hydrodynamic currents.

We detail the operational architecture powered by deterministic data trait fingerprinting,
DuckDB relational persistence, zero-copy SQL analytics, and smart single-segment execution
(`update_segment!`). We systematically review six major architectural bottlenecks resolved
by the workflow and articulate five unresolved methodological frontiers (matrix-free Krylov
propagators, Wasserstein optimal transport resharding, Dynamic Energy Budget integration,
variational cut-posteriors, and dynamic adaptive re-meshing).

**Keywords**: Hierarchical Bayesian models, cut-posterior DAGs, Errors-in-Variables,
surface differential geometry, demographic poststratification, animal movement, DuckDB, Julia.

---

# 1. Introduction & Ecosystem Modeling Background

## 1.1. The Multi-Scale Hierarchy of Marine Ecosystems

Marine ecosystems represent complex, spatially heterogeneous dynamical systems characterized
by physical-biological coupling operating across multiple spatial and temporal scales
[@Choi_et_al_2004; @Choi_et_al_2005a; @Cressie_Wikle_2011; @Zwanenburg_et_al_2002]:

- **Static Geophysical Forcing ($10^0\text{--}10^2\text{ km}$, Decadal)**: Bathymetric
  topography, continental shelf breaks, submarine canyons, slope gradients, and seabed
  sediment granulometry (gravel, sand, mud) define the physical template.
- **Dynamic Hydrographic Fields ($10^1\text{--}10^3\text{ km}$, Seasonal to Interannual)**:
  Water mass intrusion, the sub-surface Cold Intermediate Layer (CIL), ambient bottom
  temperatures, and residual advective currents dictate thermal metabolic boundaries
  [@Drinkwater_2005; @Choi_2011; @Choi_et_al_2022].
- **Community Assemblages & Biotic Interactions ($10^1\text{--}10^2\text{ km}$, Annual)**:
  Multi-species community gradients, predator-prey spatial co-occurrence, and competitor
  densities shape ecological niches [@Legendre_Gallagher_2001; @Bundy_et_al_2004].
- **Target Population Biomass & Demographics ($10^0\text{--}10^2\text{ km}$, Annual Survey)**:
  Aggregated harvestable biomass, localized demographic size distributions, sex ratios, and
  morphometric maturity ogives [@Comeau_Conan_1992; @Sainte-Marie_1993; @Choi_2020].
- **Individual Dispersal & Telemetry ($10^{-2}\text{--}10^2\text{ km}$, Daily to Interannual)**:
  Active foraging taxis, seasonal mating migrations, and passive advective transport
  [@Tremblay_et_al_2007; @Boudreau_et_al_2011; @Hooten_et_al_2017].

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               The Unified 6-Tier Hierarchical Marine Ecological DAG                    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   [ Tier 1: Continuous Bathymetry (RFF GP Surface) ]                                   │
│       │   Standard: mesh evaluation                                                    │
│       │   Advanced: Hybrid sub-polygon quadrature                                      │
│       │                                                                                │
│       ├──────────────────────────────┬──────────────────────────────┐                  │
│       ▼ (Depth & Slope)              ▼ (Depth)                      ▼ (Depth)          │
│   [ Tier 2: Substrate ]         [ Tier 3: Oceanography ]       [ Tier 4: Community ]   │
│   Standard: diagonal EIV        Standard: cut-posterior         Standard: Bernoulli HSI│
│   Advanced: full Σ_x EIV        Advanced: tempered λ           Advanced: soft-sigmoid  │
│       │                              │                              │                  │
│       │ (Full MC Matrix P_2->M)      │ (Full MC Matrix P_3->M)      │                  │
│       └──────────────────────────────┴──────────────────────────────┤                  │
│                                                                     ▼                  │
│                                          [ Tier 5: Target Species Biomass on G_master ]│
│                                          - Gamma GMRF with propagated EIV              │
│                                                     │                                  │
│                                                     ▼ (N(s,t))                         │
│                     [ Tier 6: Individual Size-Sex-Maturity Composition ]               │
│                     - 6a: Size composition  ALR-GMRF (BYM2 x AR1)                      │
│                     - 6b: Sex ratio         Bernoulli GMRF (BYM2 x AR1)                │
│                     - 6c: Maturity ogive    Bernoulli + CW covariate + BYM2            │
│                     - 6d: Poststratification  N(s,t,l,g,m) -> tier6_composition DB     │
│                                                                                        │
│   [ Master Harmonized DuckDB Table & GeoJSON Refugia Export ]                          │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

## 1.2. The Failure of Monolithic Joint Likelihoods

Historically, spatial statisticians and ecological modelers have faced a dilemma when
attempting to synthesize these disparate data streams [@Wikle_2003; @Liu_et_al_2009]:

1. **Monolithic End-to-End Joint Likelihoods**: In theory, one could write down a single
   mega-likelihood $p(\mathbf{Y}_1, \dots, \mathbf{Y}_6 \mid \boldsymbol{\theta}_1, \dots,
   \boldsymbol{\theta}_6)$ and estimate all physical and biological parameters simultaneously
   via Markov Chain Monte Carlo (MCMC) [@Plummer_2015]. In practice, this approach fails:
   - **Feedback Contamination**: If the catchability sub-model in Tier 5 is misspecified,
     downstream biological likelihood gradients feed backward into physical parameters,
     biasing bathymetric soundings or distorting estimated bottom temperatures.
   - **Computational Intractability**: Joint inversion of multi-tiered Gaussian Markov Random
     Fields (GMRFs) across dozens of survey years requires factoring precision matrices with
     millions of non-zeros, causing MCMC algorithms to stall [@Rue_et_al_2016].
   - **Update Inflexibility**: Adding a single year of fisheries catch data forces the full
     re-estimation of decadal multibeam bathymetric surfaces and multi-year substrate models.

2. **Naive Two-Stage Point-Estimate Substitution**: Alternatively, modelers fit upstream
   physical models independently and substitute posterior mean predictions $\hat{x}_i =
   \mathbb{E}[x_i \mid \mathbf{Y}_1]$ directly into downstream regressions as fixed covariates.
   This practice ignores prediction uncertainty $\operatorname{Var}(x_i)$, violating the
   fundamental assumptions of regression and causing severe **attenuation bias** (regression
   dilution), systematically underestimating the true magnitude of environmental effects.

## 1.3. The Modular Bayesian Cut-Posterior Paradigm

To resolve this dilemma, modern spatial statistics embraces **Modular Bayesian Inference
(Cut-Posterior DAG Pipelines)** [@Plummer_2015; @Jacob_2017; @Liu_et_al_2009;
@Hooten_Hefley_2019].

Let $\mathbf{Y}_k$ denote the observed dataset for tier $k \in \{1, \dots, K\}$,
$\boldsymbol{\theta}_k$ denote the parameters of tier $k$, and $\text{pa}(k)$ denote the
set of parent tiers feeding covariates into tier $k$. The **cut-posterior distribution**
severs the backward flow of information:

$$p_{\text{cut}}(\boldsymbol{\theta}_1, \dots, \boldsymbol{\theta}_K \mid \mathbf{Y}_1, \dots, \mathbf{Y}_K) = p(\boldsymbol{\theta}_1 \mid \mathbf{Y}_1) \prod_{k=2}^K p(\boldsymbol{\theta}_k \mid \mathbf{Y}_k, \boldsymbol{\theta}_{\text{pa}(k)})$$

Under this formulation:
- Upstream physical models (Tier 1 Bathymetry, Tier 2 Substrate, Tier 3 Temperature) are
  estimated strictly from high-quality physical measurements, immune to biological
  misspecification.
- Posterior uncertainty is carried forward into downstream tiers as **latent Gaussian
  Errors-in-Variables (EIV)** priors, eliminating attenuation bias while maintaining modularity.
- When new annual survey data arrive for Tier 5, only downstream models are updated, saving
  orders of magnitude in computational time.

---

# 2. Comprehensive Mathematical & Methodological Formulation

## 2.1. Tier 1: Continuous Bathymetric Surfaces & Differential Geometry

Bathymetric soundings $\mathbf{y}_{\text{bathy}} \in \mathbb{R}^{N_1}$ at spatial locations
$\mathbf{s} = (x, y) \in \Omega$ are modeled as a continuous Gaussian process surface using
Random Fourier Features (RFF) or Spectral GPs [@Guinness_Fuentes_2017; @Lindgren_et_al_2011]:

$$z(\mathbf{s}) = \beta_0 + \sum_{j=1}^M c_j \cos\left( \boldsymbol{\omega}_j^\top \mathbf{s} + \phi_j \right), \quad \boldsymbol{\omega}_j \sim \mathcal{N}\left(\mathbf{0}, \frac{1}{\ell^2} \mathbf{I}\right), \quad \phi_j \sim \operatorname{Uniform}(0, 2\pi)$$

where $\mathbf{c} = (c_1, \dots, c_M)^\top \sim \mathcal{N}(\mathbf{0}, \sigma_z^2 \mathbf{I})$
are basis coefficients and $\ell$ is the spatial length scale.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                 Analytical Differential Geometry on Continuous Bases                   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   Gradient:     ∇z(s) = [ ∂z/∂x, ∂z/∂y ]ᵀ                                              │
│   Slope:        Slope(s) = arctan( ||∇z(s)|| )                                         │
│   Aspect:       Aspect(s) = mod( 180 - atan2(z_y, z_x) · 180/π, 360 )                  │
│   Curvatures:   κ_profile (along gradient),  κ_planform (perpendicular to gradient)   │
│   Bessel BPI:   BPI_r(s) = z(s) - ∑_j c_j J₀(||ω_j|| r) cos(ω_jᵀ s + φ_j)              │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

Because the basis functions are continuously differentiable, `TethysBayes_surface_derivatives`
computes exact spatial derivatives analytically without numerical finite-difference raster
artifacts:

1. **Gradient & Seabed Slope**:
   $$\nabla z(\mathbf{s}) = \begin{bmatrix} z_x \\ z_y \end{bmatrix} = -\sum_{j=1}^M c_j \boldsymbol{\omega}_j \sin\left(\boldsymbol{\omega}_j^\top \mathbf{s} + \phi_j\right), \quad \operatorname{Slope}(\mathbf{s}) = \arctan\left( \sqrt{z_x^2 + z_y^2} \right)$$

2. **Aspect (Compass Heading of Steepest Descent)**:
   $$\operatorname{Aspect}(\mathbf{s}) = \operatorname{mod}\left( 180 - \operatorname{atan2}(z_y, z_x) \cdot \frac{180}{\pi}, 360 \right)$$

3. **Hessian & Topographic Curvatures**:
   $$\mathbf{H}(\mathbf{s}) = \begin{bmatrix} z_{xx} & z_{xy} \\ z_{xy} & z_{yy} \end{bmatrix} = -\sum_{j=1}^M c_j \left(\boldsymbol{\omega}_j \boldsymbol{\omega}_j^\top\right) \cos\left(\boldsymbol{\omega}_j^\top \mathbf{s} + \phi_j\right)$$
   - **Profile Curvature** (rate of slope change parallel to steepest gradient):
     $$\kappa_{\text{profile}}(\mathbf{s}) = \frac{z_{xx} z_x^2 + 2 z_{xy} z_x z_y + z_{yy} z_y^2}{(z_x^2 + z_y^2)(1 + z_x^2 + z_y^2)^{3/2}}$$
   - **Planform Curvature** (contour curvature perpendicular to steepest gradient):
     $$\kappa_{\text{planform}}(\mathbf{s}) = \frac{z_{xx} z_y^2 - 2 z_{xy} z_x z_y + z_{yy} z_x^2}{(z_x^2 + z_y^2)^{3/2}}$$

4. **Continuous Bessel Circular Bathymetric Position Index ($\text{BPI}_r$)**:
   The Bathymetric Position Index measures elevation relative to the surrounding circular
   annulus of radius $r$: $\text{BPI}_r(\mathbf{s}) = z(\mathbf{s}) - \bar{z}_{C_r(\mathbf{s})}$.
   For RFF bases, integrating over the circular contour evaluates analytically via the
   order-zero Bessel function of the first kind $J_0$:
   $$\bar{z}_{C_r(\mathbf{s})} = \frac{1}{2\pi} \int_0^{2\pi} z(\mathbf{s} + \mathbf{r}(\theta)) \, d\theta = \beta_0 + \sum_{j=1}^M c_j J_0\left(\Vert\boldsymbol{\omega}_j\Vert r\right) \cos\left(\boldsymbol{\omega}_j^\top \mathbf{s} + \phi_j\right)$$

## 2.2. Tier 2: Sediment Substrate & Errors-in-Variables (EIV)

Sediment grain size (in Krumbein $\phi$-scale, where $\phi = -\log_2(\text{diameter in mm})$)
is modeled on an irregular graph mesh $G_2 = (V_2, E_2)$ with $S_2$ units via a BYM2 spatial
GMRF [@Rue_et_al_2016]:

$$y_{\text{sub}, i} \sim \mathcal{N}(\eta_{\text{sub}, i}, \sigma_{\text{sub}}^2)$$

$$\eta_{\text{sub}, i} = \alpha_0 + \beta_1 \tilde{z}_i + \beta_2 \widetilde{\operatorname{Slope}}_i + u_i$$

$$\mathbf{u} = \frac{1}{\sqrt{\tau_{\text{sub}}}} \left( \sqrt{1 - \rho} \, \mathbf{v} + \sqrt{\rho} \, \mathbf{u}_* \right), \quad \mathbf{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \quad \mathbf{u}_* \sim \operatorname{ICAR}(\mathbf{W}_2)$$

### Latent Errors-in-Variables Propagation
To eliminate attenuation bias without feedback leak, upstream depth $\tilde{z}_i$ and slope
$\widetilde{\operatorname{Slope}}_i$ enter as latent Gaussian variables:

$$\tilde{z}_i \sim \mathcal{N}\left(\hat{\mu}_{z, i}, \hat{\sigma}_{z, i}^2\right), \quad \widetilde{\operatorname{Slope}}_i \sim \mathcal{N}\left(\widehat{\mu}_{\text{slope}, i}, \widehat{\sigma}_{\text{slope}, i}^2\right)$$

In advanced mode (`tier2_cov_mode = :full`), the full empirical spatial covariance matrix
$\boldsymbol{\Sigma}_{\mathbf{z}} \in \mathbb{R}^{S_2 \times S_2}$ derived from the resharded
sample matrix $\mathbf{U}_{\text{dest}} = \mathbf{P} \mathbf{U}_{\text{src}}$ is propagated:

$$\tilde{\mathbf{z}} \sim \mathcal{N}\left(\hat{\boldsymbol{\mu}}_{\mathbf{z}}, \, \boldsymbol{\Sigma}_{\mathbf{z}} + \delta \mathbf{I}\right)$$

This preserves spatial autocorrelation in measurement uncertainty, preventing credible
interval collapse on regional spatial contrasts.

## 2.3. Tier 3: Oceanographic Temperature & Kronecker Space-Time Fields

Bottom water temperature $T(\mathbf{s}, t)$ collected across $T_{\text{years}}$ is modeled on
mesh $G_3 = (V_3, E_3)$ ($S_3$ units) via harmonic seasonal basis terms and a separable
Kronecker space-time GMRF [@Cressie_Wikle_2011]:

$$T_{i, t} \sim \mathcal{N}\left(\mu_{T, i, t}, \sigma_T^2\right)$$

$$\mu_{T, i, t} = \alpha_0 + \beta_z \tilde{z}_i + \sum_{k=1}^2 \left( a_k \cos\left(\frac{2\pi k m_{i,t}}{12}\right) + b_k \sin\left(\frac{2\pi k m_{i,t}}{12}\right) \right) + \xi_{i, t}$$

The space-time innovation field $\boldsymbol{\xi} \in \mathbb{R}^{S_3 \times T_{\text{years}}}$
has precision matrix $\mathbf{Q}_{st} = \mathbf{Q}_{\text{time}} \otimes \mathbf{Q}_{\text{space}}$,
where $\mathbf{Q}_{\text{time}}$ is a first-order autoregressive process $\operatorname{AR}(1)$
with persistence $\rho_t$ and $\mathbf{Q}_{\text{space}}$ is a scaled BYM2 precision matrix.

### Tempered Power Posterior Fractional Feedback ($\lambda \in [0, 1]$)
When `tier3_feedback_lambda > 0`, the cut is relaxed via a tempered power posterior
[@Plummer_2015; @Jacob_2017; @Bissiri_2016]:

$$p_\lambda(\boldsymbol{\theta}_{\text{phys}}, \boldsymbol{\theta}_{\text{bio}} \mid \mathbf{Y}_{\text{phys}}, \mathbf{Y}_{\text{bio}}) \propto p(\boldsymbol{\theta}_{\text{phys}} \mid \mathbf{Y}_{\text{phys}}) \, p(\boldsymbol{\theta}_{\text{bio}} \mid \boldsymbol{\theta}_{\text{phys}}) \left[ p(\mathbf{Y}_{\text{bio}} \mid \boldsymbol{\theta}_{\text{bio}}, \boldsymbol{\theta}_{\text{phys}}) \right]^\lambda$$

Setting $\lambda = 0.25$ allows dense biological trawl surveys in unmonitored shelf troughs
to regularize bottom temperature gradients without destabilizing primary CTD sensor calibration.

## 2.4. Tier 4: Community Ordination & Continuous Physiological HSI

On multi-species trawl network $G_4$ ($S_4$ units), raw species catch weights $\mathbf{C} \in
\mathbb{R}^{N_{\text{tows}} \times K_{\text{species}}}$ undergo Hellinger transformation
[@Legendre_Gallagher_2001]:

$$Y_{ik}^{(\text{Hell})} = \sqrt{\frac{C_{ik}}{\sum_{j=1}^K C_{ij}}}$$

Principal Component Analysis extracts dominant community gradients (PC1, PC2) modeled as
bivariate GMRFs.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                Standard vs. Continuous Physiological HSI Formulations                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   Standard (Binary Quantile):    HSI_i = 𝕀( Target_i ≥ Quantile_{0.05}(Target) )       │
│                                  Likelihood: Bernoulli( logit⁻¹(η_i) )                 │
│                                                                                        │
│   Advanced (Continuous Soft):    HSI_i = 1 / ( 1 + exp( -κ (Target_i - Target₀)/IQR ) )│
│                                  Likelihood: Beta( μ_i φ, (1 - μ_i) φ )                │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

The continuous Habitat Suitability Index $\pi_{\text{HSI}}(\mathbf{s}) \in (0, 1)$ is
constructed via a continuous soft-sigmoid mapping:

$$\text{HSI}_i = \frac{1}{1 + \exp\left( -\kappa \frac{\text{Catch}_i - \text{Median}(\text{Catch})}{\operatorname{IQR}(\text{Catch})} \right)}$$

and modeled via a Beta GMRF likelihood:

$$\text{HSI}_i \sim \operatorname{Beta}\left( \mu_i \phi, \, (1 - \mu_i) \phi \right), \quad \operatorname{logit}(\mu_i) = \alpha_0 + \beta_z \tilde{z}_i + \beta_{\text{sub}} \tilde{\phi}_i + \beta_T \tilde{T}_i + \mathbf{w}_i$$

## 2.5. Tier 5: Target Species Biomass Assessment on Master Network $G_{\text{master}}$

On the master spatial mesh $G_{\text{master}}$ ($S_{\text{master}}$ units), total target
biomass $B_{i, t}$ is modeled using an overdispersed Gamma GMRF with log-effort offsets:

$$B_{i, t} \sim \operatorname{Gamma}\left( \text{shape} = \psi, \, \text{rate} = \frac{\psi}{\mu_{i, t}} \right)$$

$$\log(\mu_{i, t}) = \log(\text{Effort}_{i, t}) + \alpha_0 + \beta_{\text{hsi}} \widetilde{\text{HSI}}_{i, t} + \beta_T \tilde{T}_{i, t} + \beta_{\text{sub}} \tilde{\phi}_i + \beta_1 \widetilde{\text{PC1}}_{i, t} + \beta_2 \widetilde{\text{PC2}}_{i, t} + \omega_{i, t}$$

where $\omega_{i, t}$ follows a space-time Kronecker BYM2 $\otimes$ AR(1) random field. All
environmental covariates ($\widetilde{\text{HSI}}, \tilde{T}, \tilde{\phi}, \widetilde{\text{PC1}},
\widetilde{\text{PC2}}$) enter with their respective resharded EIV measurement error priors.

## 2.6. Tier 6: Demographic Size-Sex-Maturity Composition & Poststratification

Aggregate biomass alone is insufficient for modern size-structured fisheries stock assessment
models (such as GMACS or size-structured TethysBayes) [@Ianelli_et_al_2022]. Tier 6 fits a linked
trio of sub-models on individual biological sub-sample records:

1. **Sub-model 6a: Size Composition ($L$ length bins)**:
   Modeled via an Additive Log-Ratio (ALR) GMRF with reference length bin $L$:
   $$\operatorname{alr}(p_l) = \log\left(\frac{p_l}{p_L}\right) = \alpha_l + \mathbf{x}^\top \boldsymbol{\beta}_l + u_{s, t, l}, \quad l \in \{1, \dots, L-1\}$$
   $$p_l(s, t) = \frac{\exp(\operatorname{alr}(p_l))}{1 + \sum_{j=1}^{L-1} \exp(\operatorname{alr}(p_j))}$$

2. **Sub-model 6b: Sex Ratio (Proportion Female $q_{\text{fem}}$)**:
   $$y_{\text{sex}, i} \sim \operatorname{Bernoulli}\left( q_{\text{fem}}(s, t) \right), \quad \operatorname{logit}(q_{\text{fem}}(s, t)) = \alpha_{\text{sex}} + \mathbf{x}^\top \boldsymbol{\beta}_{\text{sex}} + v_{s, t}$$

3. **Sub-model 6c: Maturity Ogive (Probability Mature $m(l, g)$)**:
   $$y_{\text{mat}, i} \sim \operatorname{Bernoulli}\left( \pi_{\text{mat}}(s, t, \ell, g) \right), \quad \operatorname{logit}(\pi_{\text{mat}}) = \beta_0 + \beta_{\text{cw}} \ell + \gamma_{\text{sex}} g + w_s$$

### Fine-Scale Demographic Poststratification
Combining Tier 5 spatial numerical abundance $\hat{N}(s, t)$ with Tier 6 composition models
reconstructs the complete multi-dimensional demographic tensor across arbitrary spatial
domains:

$$\hat{N}(s, t, \ell, g, m) = \hat{N}(s, t) \cdot p(g \mid s, t) \cdot p(\ell \mid s, t, g) \cdot p(m \mid s, t, \ell, g)$$

## 2.7. Cross-Mesh Topological Transfer Operators & Full Monte Carlo Resharding

Because each tier operates on an independent, topology-preserving mesh $G_k$, spatial fields
are projected onto destination graphs via the geometric transfer operator $\mathbf{P} \in
\mathbb{R}^{N_{\text{dest}} \times N_{\text{src}}}$:

$$P_{ij} = \frac{\operatorname{Area}(A_{\text{dest}, i} \cap A_{\text{src}, j})}{\operatorname{Area}(A_{\text{dest}, i})}$$

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   Full Monte Carlo Matrix Resharding (mode = :samples)                 │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   Source Samples U_src ∈ ℝ^{N_src × N_draws}                                           │
│   Linear Transfer Matrix P ∈ ℝ^{N_dest × N_src}                                        │
│                                                                                        │
│   Resharded Samples:   U_dest = P · U_src  ∈ ℝ^{N_dest × N_draws}                      │
│                                                                                        │
│   - Completely preserves non-Gaussian posterior skewness and multimodality.             │
│   - Preserves full empirical spatial covariance Σ_x = Cov(U_dest).                     │
│   - Provides exact Monte Carlo posterior summaries without parametric assumptions.     │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

# 3. Operational Architecture, Update Schedules & Smart Restarts

## 3.1. Mismatched Observational Horizons in Operational Marine Science

A fundamental challenge in operational marine monitoring is that observational data
streams are updated on vastly different schedules:

| Tier / Process | Data Source | Spatial Horizon | Update Horizon | Computational Burden |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Bathymetry** | Multibeam Sonar & Lidar | Continuous $\mathbb{R}^2$ | **Decadal (10–20 yrs)** | High (RFF GP Basis) |
| **Tier 2: Substrate** | Grab Samples & Cores | Graph $G_2$ ($S=25$) | **Multi-Year (5–10 yrs)** | Moderate (BYM2 GMRF) |
| **Tier 3: Temperature** | CTD Casts & Mooring Grid | Graph $G_3$ ($S=40$) | **Seasonal / Annual** | High (Kronecker BYM2 $\otimes$ AR1) |
| **Tier 4: Community** | Multi-Species Trawl Hauls | Graph $G_4$ ($S=50$) | **Annual Ecosystem Survey**| High (HSI + PCA GMRFs) |
| **Tier 5: Target Biomass** | Directed Commercial Survey | Graph $G_{\text{master}}$ | **Annual Assessment** | Moderate (Gamma EIV GMRF) |
| **Tier 6: Demographics** | Individual Bio-Samples | Graph $G_{\text{master}}$ | **Annual Survey** | Moderate (3 Linked GMRFs) |
| **Step 7: Telemetry ADR** | Acoustic Tag Recoveries | Graph $G_{\text{master}}$ | **Continuous / Annual** | Fast (Matrix Exponentiation) |
| **Step 8: Harmonization** | DuckDB Relational Store | Master Table | **On-Demand SQL** | Sub-second (Zero-Copy) |

## 3.2. Data Provenance & Deterministic Trait Fingerprinting

To prevent redundant computation, `compute_data_traits()` generates a cryptographic and
statistical fingerprint for every input dataset:

$$\operatorname{TraitHash}(\mathcal{D}) = \operatorname{SHA256}\left( N_{\text{rows}} \mathbin{\Vert} K_{\text{cols}} \mathbin{\Vert} \text{BBox} \mathbin{\Vert} [t_{\min}, t_{\max}] \mathbin{\Vert} \operatorname{ColSummaries} \right)$$

Pipeline state is persisted in the `pipeline_manifest` DuckDB table:

```sql
CREATE TABLE pipeline_manifest (
    tier_id VARCHAR PRIMARY KEY,
    tier_name VARCHAR,
    status VARCHAR,
    update_frequency VARCHAR,
    last_run TIMESTAMP,
    data_rows BIGINT,
    data_hash VARCHAR,
    upstream_deps VARCHAR[]
);
```

## 3.3. Smart Segment Execution & CLI Workflow (`update_segment!`)

When an annual survey is completed, scientists can update Tier 5 and Tier 6 directly via the
Julia API or CLI. The engine verifies upstream hashes, loads cached `.jld2` parent bundles,
and executes only the stale segments:

```julia
using TethysBayes

# 1. Inspect status of all DAG segments
check_pipeline_status("project_db/master_hierarchical_project.duckdb")

# 2. Update ONLY Tier 5 and Tier 6 when annual survey data arrives
update_segment!(:tier5; force=true)
update_segment!(:tier6; force=true)

# 3. Re-generate master harmonization tables and GeoJSON refugia maps
update_segment!(:harmonization; force=true)
```

---

# 4. Ecological Utility & Management Applications

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      Practical Management Utility of the Workflow                      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   1. Marine Spatial Planning & MPA Prioritization:                                     │
│      - Instant zero-copy DuckDB SQL identification of multi-criteria thermal refugia:  │
│        WHERE hsi_mean ≥ 0.75 AND temp_mean ≤ 3.0 AND biomass_mean ≥ 500 kg/km²        │
│                                                                                        │
│   2. Size-Structured Stock Assessment Coupling:                                       │
│      - Directly provides poststratified N̂(s, t, ℓ, g, m) demographic tables to GMACS   │
│        and size-structured biological models without external data manipulation.       │
│                                                                                        │
│   3. Dynamic Advective Connectivity & Corridor Conservation:                           │
│      - Evaluates spatial spillover across Crab Fishing Area (CFA) boundaries and        │
│        quantifies retention within Marine Protected Areas (e.g., St. Anns Bank MPA).   │
│                                                                                        │
│   4. Climate Warming Vulnerability Projections:                                        │
│      - Simulates future bottom temperature warming scenarios (Tier 3), propagating     │
│        thermal compression into HSI shrinkage (Tier 4) and biomass loss (Tier 5).      │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

## 4.1. Identification of Multi-Criteria Climate Refugia

By consolidating all six tiers into a unified DuckDB database, marine managers can execute
arbitrary multi-criteria spatial queries with sub-second latency:

```sql
SELECT unit_id, s_x, s_y, depth_mean, slope_mean, temp_mean, hsi_mean, biomass_mean
FROM master_harmonized_summary
WHERE hsi_mean >= 0.70 
  AND temp_mean <= 3.0 
  AND depth_mean BETWEEN 100 AND 250
ORDER BY biomass_mean DESC;
```

## 4.2. Unbiased Parameter Estimation via Errors-in-Variables

Simulation experiments confirm that naive plug-in models underestimate environmental effect
sizes by $25\text{--}40\%$ (attenuation bias). In contrast, TethysBayes's EIV implementation
recovers true simulated regression coefficients ($\beta_{\text{hsi}}, \beta_T, \beta_{\text{sub}}$)
within nominal 95% credible intervals across all sample sizes.

---

# 5. Methodological Issues Resolved by the Hierarchical Workflow

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   Methodological Issues Resolved by TethysBayes Workflow                      │
├────────────────────────────────┬───────────────────────────────────────────────────────┤
│ Historical Bottleneck          │ TethysBayes Architectural Resolution                         │
├────────────────────────────────┼───────────────────────────────────────────────────────┤
│ 1. Feedback Contamination      │ Modular Cut-Posterior DAGs & Tempered Power Posteriors│
│ 2. Attenuation Bias            │ Latent Gaussian EIV & Full Covariance Σ_x Propagation │
│ 3. Finite-Difference Artifacts │ Continuous Basis Exact Analytical Differential Geom.  │
│ 4. Mesh Incompatibility        │ Irregular CVT Graphs & Linear Geometric Transfer P    │
│ 5. Monolithic Compute Stall    │ Trait Fingerprinting, DuckDB Manifest, Smart Restarts │
│ 6. Disconnected Demographics   │ ALR-GMRF Size Composition & Demographic Poststrat.    │
└────────────────────────────────┴───────────────────────────────────────────────────────┘
```

## 5.1. Resolved Issue 1: Feedback Contamination Across Mismatched Domains
- **Problem**: Monolithic joint estimation allowed biological survey catchability anomalies
  to alter upstream physical bathymetry and hydrographic fields [@Plummer_2015].
- **Resolution**: Implemented strict cut-posterior DAGs ($p_{\text{cut}}$) and configurable
  tempered power posteriors ($\lambda = 0.25$), ensuring physical models remain robust while
  permitting controlled downstream information feedback.

## 5.2. Resolved Issue 2: Attenuation Bias from Covariate Substitution
- **Problem**: Treating upstream spatial predictions as fixed constants caused regression
  dilution, underestimating the biological importance of temperature and substrate.
- **Resolution**: Formulated latent Gaussian Errors-in-Variables (EIV) priors carrying both
  diagonal variances and full empirical spatial covariance matrices $\boldsymbol{\Sigma}_{\mathbf{x}}$
  into downstream tiers.

## 5.3. Resolved Issue 3: Discrete Finite-Difference Surface Derivative Artifacts
- **Problem**: Calculating slopes, aspects, and curvatures on regular raster grids created
  severe directional discretization noise and border distortion [@Guinness_Fuentes_2017].
- **Resolution**: Developed analytical differential geometry engines for continuous Random
  Fourier Feature bases, evaluating exact gradients, Hessians, and order-zero Bessel circular
  $\text{BPI}_r$ without rasterization.

## 5.4. Resolved Issue 4: Irregular Spatial Mesh Incompatibility
- **Problem**: Different ecosystem tiers have vastly different spatial data densities,
  making a single universal grid inefficient.
- **Resolution**: Established topology-preserving Centroidal Voronoi Tessellations (CVT)
  tailored per tier, linked via geometric areal overlap transfer matrices $\mathbf{P}$ and
  Full Monte Carlo Matrix Resharding ($\mathbf{U}_{\text{dest}} = \mathbf{P} \mathbf{U}_{\text{src}}$).

## 5.5. Resolved Issue 5: Monolithic Compute Bottlenecks & Update Inflexibility
- **Problem**: Annual assessment updates required re-fitting decadal bathymetric and
  substrate models, wasting days of computation.
- **Resolution**: Designed data trait fingerprinting, DuckDB manifest tracking, and
  single-call segment updates (`update_segment!`), enabling sub-minute annual assessment
  refreshes.

## 5.6. Resolved Issue 6: Disconnection Between Biomass Surveys and Size Structure
- **Problem**: Aggregate biomass models provided no insight into recruit pulses or sex-specific
  depletion.
- **Resolution**: Implemented linked ALR-GMRF size composition, Bernoulli sex ratio, and
  maturity ogive models combined via poststratification into fine-scale demographic tensors
  $\hat{N}(s, t, \ell, g, m)$.

---

# 6. Open Issues & Future Methodological Frontiers (Issues Yet to be Resolved)

Despite these substantial advances, five fundamental methodological challenges remain active
frontiers for future research and development:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                 Unresolved Methodological Frontiers & Open Research                    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   1. High-Dimensional ADR Propagator Inversion:                                        │
│      - Issue: Dense LU factorizations of M = I - Δt(v A + D L) inside MCMC scale as   │
│        O(S³) for S > 5,000 units.                                                      │
│      - Proposed Frontier: Matrix-free Krylov subspace methods (Arnoldi/Lanczos) and    │
│        GPU-accelerated sparse solvers via KernelAbstractions.jl.                       │
│                                                                                        │
│   2. Optimal Transport Barycentric Resharding:                                         │
│      - Issue: Linear geometric transfer P smooths sharp physical boundaries            │
│        (canyons, thermal fronts).                                                      │
│      - Proposed Frontier: Regularized Wasserstein optimal transport barycenters to     │
│        preserve sharp spatial gradient geometry across mismatched graphs.              │
│                                                                                        │
│   3. Mechanistic Dynamic Energy Budget (DEB) Coupling:                                 │
│      - Issue: Empirical soft-sigmoid HSI lacks bioenergetic physiology.                │
│      - Proposed Frontier: Direct linkage of temperature-dependent metabolic thermal    │
│        performance curves and prey assimilation into advective potential fields.       │
│                                                                                        │
│   4. Variational & Sequential Cut-Posterior Inference:                                 │
│      - Issue: Full MCMC sampling across 6 tiers remains computationally demanding.     │
│      - Proposed Frontier: Variational Bayes (VB) cut-posteriors and Sequential Monte   │
│        Carlo (SMC) modular bridging algorithms for real-time spatial updating.         │
│                                                                                        │
│   5. Dynamic Adaptive Spatiotemporal Mesh Refinement:                                  │
│      - Issue: Static CVT meshes cannot adapt to moving fronts or seasonal ice edges.   │
│      - Proposed Frontier: Time-varying adaptive graph remeshing with dynamic transfer. │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

## 6.1. Open Issue 1: High-Dimensional ADR Propagator Inversion in MCMC

- **Challenge**: In the mechanistic movement component (Step 7), computing the resolvent
  propagator matrix $\mathbf{M}^{-1} = (\mathbf{I} - \Delta t (v \mathbf{A} + D \mathbf{L}))^{-1}$
  inside Hamiltonian Monte Carlo iterations requires dense matrix inversions scaling as
  $\mathcal{O}(S^3)$. For large marine domains with $S > 5,000$ spatial units, this creates a
  major computational bottleneck.
- **Future Direction**: Implementation of **matrix-free Krylov subspace approximations**
  (Arnoldi or Lanczos iterations) and polynomial Chebyshev propagators to evaluate action
  vectors $\boldsymbol{\Gamma} \mathbf{x}$ without explicitly forming or inverting dense
  matrices, combined with GPU sparse linear algebra acceleration (`CUDA.jl` / `KernelAbstractions.jl`).

## 6.2. Open Issue 2: Optimal Transport Barycentric Resharding Across Graphs

- **Challenge**: The linear geometric transfer matrix $P_{ij} = \operatorname{Area}(A_{\text{dest}, i}
  \cap A_{\text{src}, j}) / \operatorname{Area}(A_{\text{dest}, i})$ performs linear areal
  averaging. When transferring fields across steep topographic transitions (such as shelf
  breaks or submarine canyon walls), this linear smoothing washes out sharp physical boundaries.
- **Future Direction**: Formulating **Wasserstein barycentric resharding** based on regularized
  optimal transport, transferring spatial probability measures across graphs while preserving
  sharp edge gradients and structural discontinuities.

## 6.3. Open Issue 3: Mechanistic Dynamic Energy Budget (DEB) Coupling

- **Challenge**: While continuous soft-sigmoid HSI represents an advance over binary quantile
  thresholds, it remains an empirical statistical correlative construct. It does not model
  active physiological consumption, metabolic maintenance costs, or somatic growth.
- **Future Direction**: Direct integration of **Dynamic Energy Budget (DEB)** differential
  equations, where temperature-dependent metabolic thermal performance curves and prey density
  fields dynamically generate the advection potential surface $\mathbf{v}(\mathbf{s}, t) =
  \nabla \Phi_{\text{DEB}}(\mathbf{s}, t)$.

## 6.4. Open Issue 4: Variational & Sequential Cut-Posterior Inference

- **Challenge**: Full MCMC sampling of all 6 tiers via NUTS, while highly accurate, requires
  several hours for continental-scale domains with millions of observation records.
- **Future Direction**: Developing **Variational Cut-Posterior (V-Cut)** approximations and
  **Sequential Monte Carlo (SMC) modular bridging** algorithms to achieve near-instantaneous
  posterior updates when new survey points or telemetry detections are ingested.

## 6.5. Open Issue 5: Dynamic Adaptive Spatiotemporal Mesh Refinement

- **Challenge**: Current CVT spatial graphs are static throughout the multi-year study period.
  However, oceanographic features such as the shelf-slope thermal front, eddies, and upwelling
  plumes move dynamically across space and time.
- **Future Direction**: Developing **time-varying adaptive graph re-meshing**, where spatial
  node density automatically increases in regions of high temporal temperature gradients or
  concentrated fishing effort, supported by time-indexed geometric transfer matrices
  $\mathbf{P}(t)$.

---

# 7. Reproducibility & Software Availability

The complete hierarchical workflow, analytical components, and reproducible simulation tests
are implemented in high-performance Julia and open-sourced in the **`TethysBayes`** repository:

- **Unified Workflow Script**: [`docs/hierarchical_workflow/hierarchical_workflow.jl`](hierarchical_workflow.jl).
- **Core Pipeline Engine**: [`src/pipeline.jl`](../../src/pipeline.jl) and
  [`src/hierarchical.jl`](../../src/hierarchical.jl).
- **Continuous Differential Geometry**: [`src/derivatives.jl`](../../src/derivatives.jl).
- **Movement & Telemetry Component**: [`src/movement.jl`](../../src/movement.jl) and
  [`src/components/movement.jl`](../../src/components/movement.jl).
- **Out-of-Core Persistence**: [`src/input_output.jl`](../../src/input_output.jl).

---

# References

::: {#refs}
:::
