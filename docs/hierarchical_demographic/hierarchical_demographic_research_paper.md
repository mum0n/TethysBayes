---
title: "Class-Expanded Demographic Factorial Modeling, Multi-Species Householder Ordination, and Stage-Structured Ontogenetic Migration in Marine Ecosystems"
subtitle: "Methodological Foundations, Mathematical Architecture, Comparative Utility with Conditional Poststratification, Resolved Bottlenecks, and Unresolved Frontiers in TethysBayes (Approach B)"
author: "TethysBayes Demographic & Hierarchical Workflow Group"
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

Marine demographic modeling faces a fundamental dilemma: population dynamics are
intrinsically stage-structured, sexually dimorphic, and geographically segregated across
ontogenetic life history transitions, yet survey observations are subject to severe
sampling sparsity, variable swept areas, and complex multi-species biotic interactions.
Traditional fisheries stock assessment models collapse these dynamics into aggregate
unstratified biomass indices or rely on monolithic joint likelihoods that suffer from
computational intractability and catastrophic feedback contamination.

In this paper, we develop the theoretical foundations, mathematical architecture,
computational implementation, and ecological utility of the **Class-Expanded Demographic
Factorial and Hurdle Workflow (Approach B)** implemented within the **Bayesian Spatio-Temporal
Models (`TethysBayes`)** ecosystem. Unlike the conditional decomposition pipeline (Approach A),
which models aggregate abundance and subsequently poststratifies biological sub-samples,
Approach B directly expands every survey tow into $C$ demographic categories (functional
life stages: *Immature Juveniles*, *Adolescent Males*, *Mature Females*, and *Commercial
Males*; or fine-scale size-sex-maturity bins).

Our framework advances several methodological innovations:
1. **Physical & Oceanographic Errors-in-Variables (EIV)**: Continuous bathymetric derivatives
   (slope, aspect, curvature, BPI) from Random Fourier Feature Gaussian processes (Tier 1),
   benthic sediment grain size (Tier 2), and bottom temperatures (Tier 3) are propagated as
   latent cut-posteriors without feedback contamination.
2. **Multi-Species Community Ordination via Householder Reflection Transformations (`eigen()`
   in TethysBayes)**: Trawl catch matrices across $K_{\text{spp}} = 30$ co-occurring species are
   Hellinger-transformed and decomposed into orthonormal loadings matrices parameterized
   bijectively through sequential Householder reflections ($U = \prod (I - 2 v_i v_i^\top /
   \|v_i\|^2)$), extracting orthogonal community gradients (PC1, PC2) that capture groundfish
   predator and benthic competitor pressure.
3. **Stage-Structured Encounter & Positive Catch Hurdle Fields**: Separate Bernoulli BYM2
   $\otimes$ AR1 GMRFs estimate stage-specific Habitat Suitability Indices ($\text{HSI}_c(\mathbf{s}, t)$),
   which condition downstream Gamma density fields with log-effort offsets.
4. **Ontogenetic Dispersal & Directed Markov Advection**: Stage-specific advection operators
   $\mathbf{v}_c(\mathbf{s}) = \gamma_c \nabla \text{HSI}_c(\mathbf{s}) + \mathbf{u}_{\text{ocean}}(\mathbf{s})$
   drive discrete Centroidal Voronoi Tessellation (CVT) transition probability kernels
   $\mathbf{\Gamma}_c$, simulating stage-dependent migration trajectories from coastal
   nurseries to deep bathymetric troughs.
5. **Zero-Copy Out-of-Core DuckDB GIS Analytics**: Integrated demographic predictions are
   persisted in columnar tables, enabling instantaneous SQL discovery of Essential Fish
   Habitats (EFH) and spawning sanctuaries.

We present an in-depth comparative analysis contrasting Approach A and Approach B across
12 analytical dimensions, articulate five resolved computational bottlenecks, and delineate
five open research frontiers, including emergent sum retransformation bias, sub-sampling
zeros, and spatially explicit Leslie/Lefkovitch transition matrices.

---

# 1. Ecological Background & Motivation

## 1.1. Snow Crab Ontogenetic Niche Segregation

The snow crab (*Chionoecetes opilio*) is a cold-stenothermic, decapod crustacean supporting
one of the most economically valuable commercial fisheries in the Northwest Atlantic and
Bering Sea [@DFO_2022_SnowCrab; @SainteMarie_1995]. The species exhibits extreme sexual size
dimorphism, a determinate terminal molt to morphometric maturity, and pronounced ontogenetic
habitat shifts across its 15-year lifespan:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                      Snow Crab (Chionoecetes opilio) Ontogenetic Life History            │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  [ Pelagic Larvae (Zoea I, II, Megalopa) ]  ──(Hydrodynamic Ocean Advection)──►         │
│                                                                                         │
│  [ Early Immature Juveniles (Instars I–IV, CW < 30mm) ]                                  │
│      - Habitat: Shallow, highly structured gravel/cobble banks (Depth: 40–90m)          │
│      - Refuge: Interstitial crevices providing shelter from groundfish predation        │
│      - Temperature Niche: 0.0°C to 4.5°C                                                │
│                                │                                                        │
│                                ▼ (Ontogenetic Downslope Migration)                      │
│  [ Adolescent Males & Immature Females (Instars V–IX, CW 30–80mm) ]                     │
│      - Habitat: Intermediate muddy/sandy shelf depressions (Depth: 80–160m)             │
│      - Ecological Role: Active benthic foraging on polychaetes and ophiuroids           │
│                                │                                                        │
│                                ├────────────────────────────────────────┐               │
│                                ▼ (Terminal Molt to Maturity)            ▼               │
│  [ Mature Females (CW 50–85mm) ]              [ Commercial Males (CW ≥ 95mm) ]          │
│      - Habitat: Dynamic Thermal Edges            - Habitat: Deep Cold Basins (120–280m) │
│      - Ecology: Synchronized spawning stock      - Temperature Niche: -0.5°C to 2.5°C   │
│      - Brooding duration: 1–2 years              - High chemotaxis to thermal troughs   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

Because juvenile settlement, adolescent somatic growth, mature female reproduction, and
commercial male availability occupy distinct spatial, thermal, and bathymetric niches,
collapsing the population into a single aggregate biomass index masks critical demographic
vulnerabilities [@Tremblay_et_al_2007; @Boudreau_et_al_2011].

## 1.2. The Failure of Aggregate Biomass Modeling for Spatial Management

Conventional geostatistical biomass assessments model total survey catch weight ($Z_i$)
as a monolithic spatial field:

$$
Z_i \sim \operatorname{Tweedie}(\mu_i, \phi, p) \quad \text{or} \quad z_i \mid y_i=1 \sim \operatorname{Gamma}(\mu_i, \nu)
$$

While computationally straightforward, this approach exhibits severe operational deficiencies:
- **Masked Recruitment Failures**: A large biomass of aging, terminal-molt commercial males
  can obscure total collapse in juvenile instar density.
- **Misaligned Marine Protected Areas (MPAs)**: Closing high-biomass commercial fishing
  grounds fails to protect essential juvenile nursery grounds located in shallow, rocky
  strata that are rarely targeted by commercial fleets.
- **Homogeneous Movement Assumption**: Assuming a single population advection vector field
  violates fundamental allometric reality, as juvenile crabs actively avoid the deep
  bathymetric basins occupied by cannibalistic adult males and large demersal predators.

## 1.3. Biotic Mediation & Multi-Species Interactions

Habitat selection in marine decapods is not governed exclusively by abiotic variables
(depth, substrate, temperature), but is strongly mediated by biotic community composition:
- **Demersal Predation Pressure**: Atlantic cod (*Gadus morhua*), American plaice (*Hippoglossoides
  platessoides*), and thorny skate (*Amblyraja radiata*) prey heavily on soft-shell and
  juvenile snow crab [@DFO_2022_SnowCrab].
- **Benthic Competition**: Co-occurring crustaceans (*Homarus americanus*, *Cancer irroratus*,
  *Hyas coarctatus*) compete for interstitial shelter and benthic infauna.

Incorporating multi-species community gradients into demographic spatial models is therefore
essential to accurately delineate true realized habitat niches.

---

# 2. The Two Modeling Paradigms: Comparative Theoretical Architecture

In the `TethysBayes` ecosystem, multi-stage population structure can be estimated via two distinct
architectures: **Approach A (Conditional Decomposition)** and **Approach B (Class-Expanded
Factorial / Hurdle)**.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Approach A: Conditional Decomposition Pipeline                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   [ Tier 5: Aggregate Tow Biomass ]  ──►  N̂_T5(s, t)  (Total Abundance)                │
│                                                  │                                     │
│   [ Tier 6a: Size Composition ]      ──►  π̂(ℓ | s, t) (ALR-GMRF Multinomial)           │
│                                                  │                                     │
│   [ Tier 6b: Sex Ratio ]             ──►  ρ̂(g | ℓ, s, t) (Bernoulli Logit)             │
│                                                  │                                     │
│   [ Tier 6c: Maturity Ogive ]        ──►  μ̂(m | g, ℓ, s, t) (Parametric L50)           │
│                                                  │                                     │
│   [ Poststratification Product ]     ──►  N̂(s, t, ℓ, g, m) = N̂_T5 · π̂_ℓ · ρ̂_g · μ̂_m    │
│                                                                                        │
│   Guarantee: ∑_{ℓ,g,m} N̂(s, t, ℓ, g, m) ≡ N̂_T5(s, t) (Strict Mass Conservation)        │
└────────────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   Approach B: Class-Expanded Factorial & Stage-Hurdle Model             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   [ Survey Tow Records ]  ──(Rectangular Tow Expansion)──►  { Tow_i × Stage_c }        │
│                                                                   │                    │
│   ┌───────────────────────────────┬───────────────────────────────┴─────────────────┐  │
│   ▼                               ▼                               ▼                 ▼  │
│ [ Stage 1: Juveniles ]    [ Stage 2: Adolescents ]    [ Stage 3: Females ]    [ Stage 4: Comm ]│
│   - HSI_1 ~ Bernoulli       - HSI_2 ~ Bernoulli         - HSI_3 ~ Bernoulli     - HSI_4... │
│   - D_1 ~ Gamma             - D_2 ~ Gamma               - D_3 ~ Gamma           - D_4...   │
│   - Γ_1 (Kernel)            - Γ_2 (Kernel)              - Γ_3 (Kernel)          - Γ_4...   │
│                                                                                        │
│   Direct Output: Stage-specific encounter HSI_c(s, t), Density D̂_c(s, t), and Dispersal│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

## 2.1. Methodological Trade-off Matrix

| Analytical Dimension | Approach A: Conditional Decomposition | Approach B: Class-Expanded Factorial / Hurdle |
| :--- | :--- | :--- |
| **Primary Mathematical Focus** | Joint marginal probability decomposition | Direct stage-specific conditional distributions |
| **Sum-to-Total Mass Conservation** | **Strict Guarantee**: $\sum \hat{N}_{\text{demo}} \equiv \hat{N}_{\text{T5}}$ | **Emergent Sum**: $\sum \hat{D}_c$ accumulates variance |
| **Sub-sampling Compatibility** | **Exact**: Sub-samples fit multinomial ALR | **Sparsity Risk**: Unsampled stages appear as zeros |
| **Physiological Parameters ($L_{50}$)**| **Explicit**: Directly estimates molt parameters | **Implicit**: Requires multi-way interaction terms |
| **Zero-Inflation Resilience** | **High**: Conditioned on non-zero biological samples | **Low**: High sparsity in fine size-sex bins |
| **Ecological Niche Delineation** | **Indirect**: Size differences filtered through ALR | **Direct**: Directly fits distinct $\beta_c$ & spatial fields |
| **Biotic Community Incorporation** | Ordination enters total biomass tier | Householder ordination directly enters each stage |
| **Movement & Telemetry Fusion** | Single velocity field driven by aggregate $\nabla \text{HSI}$ | Stage-specific advection vectors $\mathbf{v}_c = \gamma_c \nabla \text{HSI}_c$ |
| **Computational Scaling** | Sequential sub-models ($1 + 1 + 1 + 1$ chains) | Embarrassingly parallel across $C$ stages |
| **Spatial GIS & EFH Extraction** | Requires multi-stage post-hoc multiplication | Direct zero-copy SQL queries on stage tables |
| **Primary Optimal Application** | **Stock Assessment, Leslie Dynamics, Quotas** | **MPA Boundaries, Nursery Refugia, GIS Telemetry** |

---

# 3. Complete Mathematical Formulation of the Demographic Workflow

The demographic workflow operates on a Directed Acyclic Graph (DAG) structured across
eight computational tiers and steps:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                 The Complete Multi-Tier Demographic DAG (Approach B)                   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   [ Tier 1: Continuous Bathymetric Surface (RFF Gaussian Process) ]                    │
│       │   - Continuous analytical gradients: ∇z(s), Slope, Curvatures, Circular BPI    │
│       ├──────────────────────────────┬─────────────────────────────────────────────────┤
│       ▼ (Depth & Slope)              ▼ (Depth)                                         │
│   [ Tier 2: Substrate Granulometry ] [ Tier 3: Hydrographic Temperature ]              │
│   - Sediment grain with full EIV     - Bottom temperature (Harmonic ⊗ BYM2 ⊗ AR1)      │
│       │                              │                                                 │
│       ├──────────────────────────────┴─────────────────────────────────┐               │
│       │                                                                ▼               │
│       │                      [ Tier 4C: Multi-Species Community Ordination ]           │
│       │                      - Hellinger transform of 30 trawl species                 │
│       │                      - Householder reflection transformation: eigen() in TethysBayes  │
│       │                      - Orthogonal latent gradients: (PC1, PC2) with GMRF       │
│       │                                                                │               │
│       └──────────────────────────────┬─────────────────────────────────┘               │
│                                      ▼                                                 │
│                     [ Tow Expansion into C Demographic Life Stages ]                   │
│                     - Immature Juveniles (<60mm)                                       │
│                     - Adolescent Males (60-94mm Immature)                              │
│                     - Mature Females (Spawning Stock)                                  │
│                     - Commercial Males (≥95mm Mature)                                  │
│                                       │                                                │
│                                       ▼                                                │
│      [ Tier 4D: Stage-Structured Encounter & Habitat Suitability (HSI_c) ]             │
│      - Bernoulli BYM2 ⊗ AR1 GMRFs: P(Encounter_c | Depth, Temp, Grain, PC1, PC2)       │
│                                       │                                                │
│                                       ▼                                                │
│      [ Tier 5D: Stage-Structured Positive Catch & Hurdle Density ]                     │
│      - Gamma GMRFs with log-effort offsets: D_c(s, t) = p_c · μ_c                      │
│                                       │                                                │
│                                       ▼                                                │
│      [ Step 6D: Stage-Specific Dispersal & Ontogenetic Migration ]                     │
│      - Markov transition kernels Γ_c(∇HSI_c, W) driving particle advection             │
│                                       │                                                │
│                                       ▼                                                │
│ [ Step 7D: Master Demographic Refugia SQL Analytics & GeoJSON GIS Maps ]               │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

## 3.1. Rectangular Tow Expansion

Let $\mathcal{I} = \{1, \dots, N_{\text{tows}}\}$ be the set of trawl tows, with locations $\mathbf{s}_i \in \Omega$,
years $t_i$, swept areas $A_i$, and total catch biomass $Z_i$. Let biological sub-samples contain
measured individuals $k \in \{1, \dots, K_i\}$ with carapace width $\text{CW}_k$, sex $g_k \in \{0, 1\}$,
and maturity $m_k \in \{0, 1\}$.

Every tow $i$ is expanded into $C$ demographic categories:

$$
\mathcal{C} = \begin{cases}
\{1: \text{Immature Juveniles}, \, 2: \text{Adolescent Males}, \, 3: \text{Mature Females}, \, 4: \text{Commercial Males}\} & \text{Functional Scheme} \\
\{1, \dots, L\} \times \{0, 1\} \times \{0, 1\} \quad (C = 20) & \text{Fine-Scale Bins}
\end{cases}
$$

The expanded observation vector for tow $i$ and class $c$ is:

$$
y_{i, c} = \mathbb{I}(n_{i, c} > 0), \qquad z_{i, c} = Z_i \cdot \frac{n_{i, c}}{\sum_{k=1}^C n_{i, k}}
$$

## 3.2. Tier 1: Continuous Bathymetric Surfaces & Differential Geometry

Bathymetric soundings $z_j$ are modeled using a Random Fourier Feature (RFF) stationary Gaussian
process [@Rahimi_Recht_2007; @Lindgren_et_al_2011]:

$$
z(\mathbf{s}) = \beta_0 + \sum_{m=1}^M \left( a_m \cos(\boldsymbol{\omega}_m^\top \mathbf{s}) + b_m \sin(\boldsymbol{\omega}_m^\top \mathbf{s}) \right), \quad \boldsymbol{\omega}_m \sim \mathcal{N}(\mathbf{0}, \ell_{\text{bathy}}^{-2} \mathbf{I})
$$

Surface derivatives are evaluated analytically without finite-difference grid artifacts:

$$
\nabla z(\mathbf{s}) = \begin{bmatrix} \frac{\partial z}{\partial x} \\ \frac{\partial z}{\partial y} \end{bmatrix}, \qquad \mathcal{H}_z(\mathbf{s}) = \begin{bmatrix} \frac{\partial^2 z}{\partial x^2} & \frac{\partial^2 z}{\partial x \partial y} \\ \frac{\partial^2 z}{\partial x \partial y} & \frac{\partial^2 z}{\partial y^2} \end{bmatrix}
$$

$$
\text{Slope}(\mathbf{s}) = \|\nabla z(\mathbf{s})\|_2 = \sqrt{z_x^2 + z_y^2}
$$

$$
\text{Profile Curvature}(\mathbf{s}) = -\frac{z_{xx} z_x^2 + 2 z_{xy} z_x z_y + z_{yy} z_y^2}{(z_x^2 + z_y^2) (1 + z_x^2 + z_y^2)^{3/2}}
$$

$$
\text{Circular BPI}_r(\mathbf{s}) = z(\mathbf{s}) - \frac{1}{2\pi} \int_0^{2\pi} z(\mathbf{s} + r \mathbf{u}_\theta) \, d\theta = -\frac{r^2}{4} \Delta z(\mathbf{s}) + \mathcal{O}(r^4)
$$

## 3.3. Tier 2 & Tier 3: Substrate Granulometry & Ocean Temperature

1. **Tier 2 (Substrate Grain Size)**:
   Log-grain diameter $\phi_i$ is modeled conditioning on Tier 1 depth and slope with Errors-in-Variables (EIV):
   $$
   g_i \sim \mathcal{N}\left(\alpha_g + \beta_g^{\text{depth}} \tilde{z}_i + \beta_g^{\text{slope}} \widetilde{\text{Slope}}_i + \phi_g(s_i), \, \sigma_g^2\right)
   $$
2. **Tier 3 (Hydrographic Bottom Temperature)**:
   Bottom temperature records $T_i$ across months $m_i$ and years $t_i$ are decomposed via harmonic Fourier terms
   and spatiotemporal separable GMRFs:
   $$
   T_i = \alpha_T + \beta_T^z \tilde{z}_i + \sum_{h=1}^2 \left( A_h \cos\left(\frac{2\pi h m_i}{12}\right) + B_h \sin\left(\frac{2\pi h m_i}{12}\right) \right) + \phi_T(s_i) + \psi_T(s_i, t_i) + \epsilon_{T, i}
   $$

## 3.4. Tier 4C: Multi-Species Community Ordination via Householder Transformations (`eigen()` in TethysBayes)

To capture community-level predation and competition gradients, multi-species trawl catch biomasses
$Y_{ik}$ ($k = 1, \dots, K_{\text{spp}}$, $K_{\text{spp}} = 30$) are ordinated using TethysBayes's Householder
eigenvector parameterization.

### 3.4.1. Hellinger Transformation
To prevent high-abundance schooling pelagics from dominating distance metrics, raw catches are
Hellinger-transformed:

$$
H_{ik} = \sqrt{\frac{Y_{ik}}{\sum_{j=1}^{K_{\text{spp}}} Y_{ij}}}, \qquad \tilde{\mathbf{H}} = \mathbf{H} - \mathbf{1} \bar{\mathbf{h}}^\top
$$

### 3.4.2. Householder Parameterization of the Orthonormal Loadings Matrix
In standard Bayesian factor analysis, sampling orthogonal loadings matrices $U \in \mathbb{R}^{K_{\text{spp}} \times k}$
on the Stiefel manifold $\mathcal{V}_k(\mathbb{R}^K) = \{U : U^\top U = \mathbf{I}_{k \times k}\}$ is notoriously
challenging due to non-Euclidean geometry and sign-switching multimodality [@Hoff_2009; @Bissiri_2016].

TethysBayes resolves this by parameterizing $U$ through a sequence of $k$ elementary Householder reflections.
Let $\mathbf{v}_1, \dots, \mathbf{v}_k \in \mathbb{R}^K$ be unconstrained reflector vectors. The elementary
orthogonal reflection operator across hyperplane $\mathbf{v}_j^\perp$ is:

$$
\mathbf{H}_j = \mathbf{I}_{K \times K} - 2 \frac{\mathbf{v}_j \mathbf{v}_j^\top}{\|\mathbf{v}_j\|^2}
$$

The full orthonormal loading matrix is constructed sequentially:

$$
\mathbf{U}(\mathbf{v}_1, \dots, \mathbf{v}_k) = \left( \prod_{j=1}^k \mathbf{H}_j \right) \begin{bmatrix} \mathbf{I}_{k \times k} \\ \mathbf{0}_{(K-k) \times k} \end{bmatrix}
$$

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│             Householder Reflection Transformation Geometry on the Stiefel Manifold     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   Unconstrained Euclidean Vectors          Sequential Householder Reflections          │
│   [ v_1, v_2, ..., v_k ] ∈ ℝ^{K × k} ───►  H_j = I - 2 (v_j v_jᵀ) / ||v_j||²           │
│                                                          │                             │
│                                                          ▼                             │
│                                            Exact Orthonormal Loadings                  │
│                                            U = (∏ H_j) [ I_k ; 0 ]                     │
│                                            Property: Uᵀ U ≡ I_{k × k}                  │
│                                                          │                             │
│   Bijective Inverse (eigenvector_to_householder):        ▼                             │
│   U ∈ 𝒱_k(ℝ^K) ─────────────────────────────► [ v_1, v_2, ..., v_k ]                   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

The transformation is bijective and implemented in TethysBayes via:
- `eigenvector_to_householder(U)`: Extracts Householder vectors $\mathbf{v}_j$ from any orthonormal basis.
- `householder_to_eigenvector(v_mat, K, k)`: Reconstructs exact orthonormal loadings $U$.

### 3.4.3. Spatial Community Mapping & Resharding
Ordination scores $\mathbf{F} = \tilde{\mathbf{H}} \mathbf{U} = [\text{PC1}, \, \text{PC2}]$ represent dominant
ecosystem axes:
- **PC1 (Groundfish Predator Axis)**: High positive loadings on *Gadus morhua*, *Hippoglossoides
  platessoides*, and *Amblyraja radiata*.
- **PC2 (Benthic Crustacean Axis)**: High loadings on *Pandalus borealis*, *Cancer irroratus*,
  and *Hyas coarctatus*.

The spatial distribution of each axis is modeled on community mesh $G_{\text{comm}}$ and resharded onto
$G_{\text{master}}$:

$$
\text{PC}_{j, i} = \alpha_{\text{pc}, j} + \beta_{\text{pc}, j}^z \tilde{z}_i + \beta_{\text{pc}, j}^g \tilde{g}_i + \beta_{\text{pc}, j}^T \tilde{T}_i + \phi_{\text{pc}, j}(s_i) + \psi_{\text{pc}, j}(s_i, t_i) + \epsilon_{\text{pc}, j}
$$

yielding $\widetilde{\text{PC1}}_i \sim \mathcal{N}(\mu_{\text{pc1}, i}, \sigma^2_{\text{pc1}, i})$ and
$\widetilde{\text{PC2}}_i \sim \mathcal{N}(\mu_{\text{pc2}, i}, \sigma^2_{\text{pc2}, i})$.

## 3.5. Tier 4D: Stage-Structured Encounter & Habitat Suitability (HSI)

For each demographic stage $c \in \mathcal{C}$, encounter probability is modeled on the master areal
network $G_{\text{master}} = (V, E)$ ($S = 80$ units) across $T$ years:

$$
y_{i, c} \sim \operatorname{Bernoulli}(p_{i, c})
$$

$$
\operatorname{logit}(p_{i, c}) = \alpha_c + \beta_c^z \tilde{z}_i + \beta_c^g \tilde{g}_i + \beta_c^T \tilde{T}_i + \beta_c^{\text{pc1}} \widetilde{\text{PC1}}_i + \beta_c^{\text{pc2}} \widetilde{\text{PC2}}_i + \phi_c(s_i) + \psi_c(s_i, t_i)
$$

where:
- $\tilde{z}_i, \tilde{g}_i, \tilde{T}_i, \widetilde{\text{PC1}}_i, \widetilde{\text{PC2}}_i$ are Errors-in-Variables (EIV)
  propagated predictions from upstream Tiers 1–4C.
- $\phi_c \sim \operatorname{BYM2}(W)$ is a stage-specific spatial intrinsic GMRF:
  $$
  \phi_c(s) = \frac{1}{\sqrt{\tau_c}} \left( \sqrt{1 - \rho_{\text{bym}, c}} \, \mathbf{u}_c^* + \sqrt{\rho_{\text{bym}, c}} \, \mathbf{v}_c^* \right)
  $$
- $\psi_c \sim \operatorname{BYM2}(W) \otimes \operatorname{AR1}(\rho_{\text{ar}, c})$ is a separable space-time random effect.

The predicted encounter probability defines the **Stage-Specific Habitat Suitability Index**:

$$
\text{HSI}_c(u, t) = \frac{1}{1 + \exp(-\hat{\eta}_c(u, t))} \in (0, 1)
$$

## 3.6. Tier 5D: Stage-Structured Density & Positive Catch Hurdle

For tows where life stage $c$ is encountered ($y_{i, c} = 1$), positive catch biomass $z_{i, c} > 0$
is modeled with a Gamma likelihood and swept-area log-effort offset $\log(A_i)$:

$$
z_{i, c} \mid y_{i, c} = 1 \sim \operatorname{Gamma}\left(\frac{\mu_{i, c}}{\nu_c}, \, \nu_c\right), \quad \mathbb{E}[z_{i, c}] = \mu_{i, c}, \quad \operatorname{Var}(z_{i, c}) = \nu_c \mu_{i, c}^2
$$

$$
\log(\mu_{i, c}) = \log(A_i) + \alpha_c^+ + \beta_c^{\text{hsi}} \widetilde{\text{HSI}}_{c, i} + \beta_c^{z+} \tilde{z}_i + \beta_c^{T+} \tilde{T}_i + \beta_c^{\text{pc1}+} \widetilde{\text{PC1}}_i + \beta_c^{\text{pc2}+} \widetilde{\text{PC2}}_i + \phi_c^+(s_i, t_i)
$$

The **Expected Demographic Density** (kg / $\text{km}^2$) for life stage $c$ at node $u$ and year $t$ is:

$$
\hat{D}_c(u, t) = \hat{p}_c(u, t) \cdot \hat{\mu}_c(u, t)
$$

The total emergent population biomass density across all stages is:

$$
\hat{D}_{\text{total}}(u, t) = \sum_{c=1}^C \hat{D}_c(u, t)
$$

## 3.7. Step 6D: Stage-Specific Dispersal & Ontogenetic Migration

Unlike aggregate movement models that assume a uniform velocity field, Approach B evaluates
**stage-specific directed advection operators** $\mathbf{A}_c$ on the spatial adjacency graph $W$:

$$
W_{\text{dir}, ij}^{(c)} = \begin{cases}
W_{ij} \cdot \exp\left(\gamma_c \cdot (\text{HSI}_c(j) - \text{HSI}_c(i)) + \frac{\mathbf{u}_{\text{ocean}}(\mathbf{c}_i) \cdot (\mathbf{c}_j - \mathbf{c}_i)}{\|\mathbf{c}_j - \mathbf{c}_i\|}\right) & \text{if } \text{HSI}_c(j) > \text{HSI}_c(i) \\
0 & \text{otherwise}
\end{cases}
$$

The stage-specific row-stochastic Markov transition probability matrix is:

$$
\mathbf{\Gamma}_c = (1 - \omega_{\text{diff}, c}) \mathbf{D}_{\text{out}, c}^{-1} \mathbf{W}_{\text{dir}}^{(c)} + \omega_{\text{diff}, c} \mathbf{D}_{\text{iso}}^{-1} \mathbf{W}
$$

- **Immature Juveniles ($c=1$)**: Higher diffusion weight ($\omega_{\text{diff}} = 0.4$), simulating isotropic
  dispersal from shallow nursery banks toward continental shelf depressions.
- **Commercial Males ($c=4$)**: High chemotactic sensitivity ($\gamma = 1.5$), steep concentration
  in deep cold basins ($T \in [0.5, 3.0]^\circ\text{C}$).

## 3.8. Step 7D: Master Demographic Refugia SQL Analytics & DuckDB GIS Harmonization

Stage predictions are consolidated into an out-of-core DuckDB database table (`master_demographic_summary`),
enabling zero-copy analytical queries for Essential Fish Habitat (EFH):

```sql
-- Identify Critical Multi-Stage Spawning & Nursery Sanctuaries
SELECT 
    unit_id, s_x, s_y,
    hsi_immature_juvenile_mean AS juvenile_hsi,
    dens_immature_juvenile_mean AS juvenile_density,
    hsi_mature_female_mean AS spawning_female_hsi,
    dens_mature_female_mean AS spawning_female_density,
    depth_mean, temp_mean, grain_mean, pc1_mean, pc2_mean
FROM master_demographic_summary
WHERE hsi_immature_juvenile_mean >= 0.65 AND hsi_mature_female_mean >= 0.60
ORDER BY (juvenile_density + spawning_female_density) DESC;
```

---

# 4. Operational Computational Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                    TethysBayes Demographic Pipeline Operational Architecture                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   [ Input Survey Trawls ] ──► [ Rectangular Expansion ] ──► [ Areal Mesh Partition ]   │
│                                                                   │                    │
│   ┌───────────────────────────────────────────────────────────────┴─────────────────┐  │
│   ▼ (Tier 1..3 Caches)                                                              ▼  │
│ [ DuckDB Columnar Store ] ◄─── (Zero-Copy Arrow API) ───► [ Modular Bayesian MCMC ]    │
│   - tier1_depth_predictions                                 - Cut-Posterior Registry   │
│   - tier2_substrate_predictions                             - NUTS / HMC Samplers      │
│   - tier3_temp_predictions                                  - Spatial Resharding W_uv  │
│   - tier4c_community_predictions                            - Householder eigen()      │
│   - tier4d_demographic_hsi_predictions                      - Stage BYM2 ⊗ AR1 GMRFs   │
│   - tier5d_demographic_density_predictions                                             │
│   - master_demographic_summary                                                         │
│                                                                                        │
│   [ Pipeline Manifest ] ──► Tracks Traits, SHA-256 Hashes, Upstream DAG Dependency Graph │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

## 4.1. The Cut-Posterior Caching Engine
To prevent computational re-fitting of upstream physical tiers when modifying demographic stages,
TethysBayes implements strict cut-posterior caching via data traits:

$$
\text{Trait Fingerprint} = \mathcal{H}\left( N_{\text{obs}}, \, \mathbf{s}_{\text{bounds}}, \, t_{\text{range}}, \, \bar{y}, \, \text{DAG}_{\text{deps}} \right)
$$

If upstream tables match the cached hash, the segment updater (`update_demographic_segment!`)
instantly loads persisted JLD2 bundles and DuckDB tables, skipping upstream MCMC sampling entirely.

## 4.2. Spatial Mesh Resharding Across Non-Nested Graphs
When transferring posterior predictive distributions from community mesh $G_{\text{comm}}$ ($S_1$ units)
to master network $G_{\text{master}}$ ($S_2$ units), TethysBayes applies area-weighted conservative interpolation:

$$
\mathbf{W}_{\text{transfer}}(u, v) = \frac{\operatorname{Area}(P_u^{(2)} \cap P_v^{(1)})}{\operatorname{Area}(P_u^{(2)})}, \qquad \sum_{v=1}^{S_1} \mathbf{W}_{\text{transfer}}(u, v) = 1
$$

For sample matrix $\mathbf{S}^{(1)} \in \mathbb{R}^{S_1 \times M_{\text{samples}}}$, the resharded
samples on $G_{\text{master}}$ are:

$$
\mathbf{S}^{(2)} = \mathbf{W}_{\text{transfer}} \mathbf{S}^{(1)} \in \mathbb{R}^{S_2 \times M_{\text{samples}}}
$$

---

# 5. Resolved Historical Bottlenecks

1. **Bottleneck 1: Feedback Contamination in Multi-Tier Ecosystem Models**:
   - *Problem*: In joint Bayesian models, biological misspecifications (e.g. non-linear crab density)
     distort estimates of seabed bathymetry and ocean temperature.
   - *Resolution*: Cut-posterior DAG architecture rigorously isolates upstream physical parameters
     from downstream biological feedback [@Plummer_2015; @Jacob_et_al_2020].
2. **Bottleneck 2: Orthogonal Loading Estimation on Non-Euclidean Manifolds**:
   - *Problem*: Traditional Bayesian factor analysis on Stiefel manifolds requires complex
     geodesic Monte Carlo or unconstrained matrices that suffer from rotational ambiguity.
   - *Resolution*: TethysBayes's Householder reflection transformation (`eigen()`) maps unconstrained
     Euclidean vectors $\mathbf{v}_1, \dots, \mathbf{v}_k$ bijectively onto exact orthonormal
     loadings ($U^\top U = \mathbf{I}$), enabling fast, robust NUTS sampling.
3. **Bottleneck 3: Loss of Upstream Prediction Variance (Plug-In Attenuation)**:
   - *Problem*: Using deterministic mean predictions from depth and temperature models in demographic
     regressions underestimates parameter uncertainty and biases slope coefficients toward zero.
   - *Resolution*: Errors-in-Variables (EIV) formulation carries both prediction mean $\mu_i$ and
     posterior variance $\sigma_i^2$ into the stage likelihoods.
4. **Bottleneck 4: Directional Chemotactic Advection on Irregular Voronoi Meshes**:
   - *Problem*: Continuous partial differential equations ($\nabla \cdot (\mathbf{v} u)$) cannot be
     directly evaluated on irregular areal graph networks.
   - *Resolution*: Finite-volume divergence discretization on Centroidal Voronoi Tessellations
     (CVT) yields conservative, directional Markov transition matrices $\mathbf{\Gamma}_c(\nabla \text{HSI}_c, W)$.
5. **Bottleneck 5: Memory Exhaustion in High-Dimensional Multi-Stage Fields**:
   - *Problem*: Persisting $M = 1,000$ posterior samples across 20 demographic classes, 80 spatial
     units, and 10 years produces millions of double-precision floats, exhausting RAM.
   - *Resolution*: DuckDB zero-copy columnar storage with Snappy/ZSTD compression enables instantaneous
     analytical queries directly on disk.

---

# 6. Open Methodological Frontiers & Unresolved Challenges

Despite these advances, several fundamental mathematical challenges remain active frontiers of
research in spatial demographic ecology:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Five Open Methodological Frontiers in Approach B                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   1. Retransformation Bias in Emergent Totals                                          │
│      - ∑_{c=1}^C p̂_c μ̂_c compounds log-normal/Gamma Jensen's inequality bias (+15–35%)  │
│                                                                                        │
│   2. Survey Sub-sampling Zeros vs. Structural Absences                                 │
│      - 40-crab biological baskets induce false zeros (y_{i,c} = 0) in high-density tows│
│                                                                                        │
│   3. High Sparsity & Spatial Covariance Degeneration                                   │
│      - Expanding to C = 20 fine bins creates >85% zeros, destabilizing BYM2 spatial precision│
│                                                                                        │
│   4. Spatially Explicit Leslie / Lefkovitch Transition Coupling                        │
│      - Linking stage-specific densities via dynamic molting probability matrices       │
│                                                                                        │
│   5. Joint Multi-Stage Hurdle Coregionalization                                        │
│      - Estimating cross-stage spatial covariance Cov(ϕ_c, ϕ_{c'}) across life history  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

## 6.1. Retransformation Bias in Emergent Aggregate Totals
When summing $C$ separate hurdle density estimates $\sum_{c=1}^C \hat{p}_c \hat{\mu}_c$, each
model's log-link retransformation bias ($\mathbb{E}[\exp(\eta)] = \exp(\mu + \sigma^2/2)$) and
estimation variance additively compound. In data-sparse strata, the emergent aggregate total
$\sum_c \hat{D}_c$ frequently exceeds directly fitted aggregate biomass models by $15\text{--}35\%$.
Developing bias-corrected moment-matching estimators for emergent hurdle sums is an urgent priority.

## 6.2. Biological Sub-sampling Zeros vs. Structural Ecological Zeros
In a standard survey trawl capturing 800 kg of snow crab ($\approx 2,000$ animals), biologists
measure a random sub-sample of 40 individuals. If no adolescent males appear in the 40-crab
basket, Approach B records $y_{i, \text{adolescent}} = 0$ (absence), even though dozens were
present in the net. Developing multinomial-conditioned hurdle likelihoods that explicitly model
the sub-sampling hypergeometric process is necessary to eliminate artificial downward bias in
encounter probabilities.

## 6.3. Spatial Covariance Degeneration in Fine-Scale Demographic Bins ($C = 20$)
Expanding survey tows into 20 fine-scale size-sex-maturity bins induces extreme zero-inflation
($> 85\%$ zeros). Under high sparsity, intrinsic BYM2 spatial precision matrices $\mathbf{Q} =
\tau (\mathbf{D} - \mathbf{W})$ experience severe variance inflation and slow MCMC mixing.
Hierarchical dimension reduction (e.g. tensor-product B-splines on the size dimension) will
be required to scale Approach B to continuous length frequencies.

## 6.4. Dynamic Spatio-Temporal Leslie/Lefkovitch Matrix Integration
Currently, stage models are linked purely through environmental covariates and directional advection.
A fully mechanistic demographic model must integrate stage-structured matrix projection equations:

$$
\mathbf{N}(\mathbf{s}, t+1) = \mathbf{\Gamma}(\mathbf{s}) \left( \mathbf{T}_{\text{molt}}(\mathbf{s}, t) \mathbf{N}(\mathbf{s}, t) + \mathbf{R}(\mathbf{s}, t) \right)
$$

where $\mathbf{T}_{\text{molt}}$ is a spatially explicit molting and survival transition matrix,
and $\mathbf{R}(\mathbf{s}, t)$ is larval recruitment from pelagic oceanographic drift.

## 6.5. Joint Cross-Stage Spatial Coregionalization
Currently, spatial random effects $\phi_c(s)$ are fitted independently per stage. In reality,
juvenile and adult spatial distributions exhibit strong negative cross-correlation (cannibalism
avoidance). Formulating separable Linear Models of Coregionalization (LMC):

$$
\boldsymbol{\Phi}(\mathbf{s}) = \mathbf{L}_{\text{stage}} \mathbf{w}(\mathbf{s}), \qquad \mathbf{w}(\mathbf{s}) \sim \operatorname{GMRF}(\mathbf{0}, \mathbf{Q}_{\text{spatial}})
$$

will allow TethysBayes to estimate stage-to-stage spatial repulsion directly from survey data.

---

# 7. Practical Recommendations & Synthesis

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                  Decision Framework: Selecting Between Approach A and B                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   PRIMARY MANAGEMENT OBJECTIVE:                                                        │
│                                                                                        │
│   ├── Total Allowable Catch (TAC) & Quota Advice?                                      │
│   │   └──► Use APPROACH A (Strict mass conservation, exact sub-sample decomposition)   │
│   │                                                                                    │
│   ├── Spatial Conservation, MPAs, Nursery Sanctuaries & Telemetry?                     │
│   │   └──► Use APPROACH B (Direct stage HSI, Householder ordination, ontogenetic paths)│
│   │                                                                                    │
│   └── Multi-Decadal Stock Reconstruction with Dynamic Harvest Control Rules?           │
│       └──► Use APPROACH A for historical biomass; validate nurseries with APPROACH B   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

The Class-Expanded Demographic Workflow (Approach B) provides marine scientists and fisheries
managers with an unprecedented toolkit for resolving spatial demographic complexity. By fusing
continuous differential geometry, Householder-parameterized community ordination, spatiotemporal
hurdle GMRFs, and directed ontogenetic movement into a modular, out-of-core DuckDB architecture,
TethysBayes establishes a new benchmark for reproducible, ecologically grounded marine spatial modeling.

---

# References
