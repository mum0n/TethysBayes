---
title: "Class-Expanded Demographic Presence/Absence, Hurdle & Stage-Structured Dispersal Workflows in TethysBayes (Approach B)"
subtitle: "Direct Multi-Stage Encounter Modeling, Ontogenetic Niche Segregation, and Comparative Analysis with Conditional Decomposition"
author: "TethysBayes Development Group"
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

# Class-Expanded Demographic Modeling, Stage-Structured HSI & Ontogenetic Dispersal (Approach B)

## 1. Summary & Overview

In marine fisheries and spatial ecology, population demographics (size structure, sex ratio,
and maturity status) can be modeled using two fundamentally distinct mathematical architectures:

1. **Approach A: Conditional Decomposition Pipeline** (Documented in [`docs/hierarchical_workflow/`](../hierarchical_workflow/)):
   - Total aggregate biomass/abundance is modeled first at the tow level ($\hat{N}_{\text{T5}}(\mathbf{s}, t)$).
   - Biological sub-samples are then decomposed conditionally via linked sub-models: size composition proportions
     $\boldsymbol{\pi}(\ell \mid \mathbf{s}, t)$ (ALR-GMRF), sex ratio $\rho(g \mid \ell, \mathbf{s}, t)$, and
     maturity ogive $\mu(m \mid g, \text{CW}, \mathbf{s}, t)$.
   - Joint abundance is reconstructed via poststratification:
     $$\hat{N}(\mathbf{s}, t, \ell, g, m) = \hat{N}_{\text{T5}}(\mathbf{s}, t) \cdot \pi_\ell \cdot \rho_g \cdot \mu_m$$
2. **Approach B: Class-Expanded Demographic Factorial / Hurdle Model** (This document & [`hierarchical_demographic.jl`](hierarchical_demographic.jl)):
   - Every survey tow is expanded directly into $C$ demographic categories
     (e.g., functional life stages: *Immature Juveniles*, *Adolescent Males*, *Mature Females*, *Commercial Males*;
     or fine-scale bins: $L \times 2 \times 2 = 20$ classes).
   - Each life stage is modeled directly via a two-part spatiotemporal hurdle formulation:
     $$\text{Encounter Probability (Stage HSI)}: \quad y_{i, c} \sim \operatorname{Bernoulli}(p_{i, c})$$
     $$\text{Positive Stage Density}: \quad z_{i, c} \mid y_{i, c}=1 \sim \operatorname{Gamma}(\mu_{i, c}, \nu_c)$$
   - Stage-specific encounter probabilities define distinct, stage-dependent Habitat Suitability Indices
     $\text{HSI}_c(\mathbf{s}, t) = p_{i, c}$, which in turn drive stage-specific advection vector fields
     $\mathbf{v}_c(\mathbf{s}) = \gamma_c \nabla \text{HSI}_c(\mathbf{s}) + \mathbf{u}_{\text{ocean}}(\mathbf{s})$
     in Lagrangian movement models.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│             The Demographic Class-Expanded Marine Ecological DAG (Approach B)          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   [ Tier 1: Continuous Bathymetry (RFF GP Surface) ]                                   │
│       │   - Continuous surface derivatives (Slope, Profile Curvature, Circular BPI)    │
│       ├──────────────────────────────┬─────────────────────────────────────────────────┤
│       ▼ (Depth & Slope)              ▼ (Depth)                                         │
│   [ Tier 2: Substrate ]         [ Tier 3: Oceanography ]                               │
│   - Sediment grain with EIV     - Bottom temperature (Harmonic ⊗ BYM2 ⊗ AR1)           │
│       │                              │                                                 │
│       ├──────────────────────────────┴─────────────────────────────────┐               │
│       │                                                                ▼               │
│       │                      [ Tier 4C: Multi-Species Community Ordination ]           │
│       │                      - Hellinger transform of multispecies trawl catch         │
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
│      - Bernoulli BYM2 ⊗ AR1 GMRFs estimating P(Encounter_c | s, t, Env, PC1, PC2)     │
│                                       │                                                │
│                                       ▼                                                │
│      [ Tier 5D: Stage-Structured Positive Density & Hurdle Catch ]                     │
│      - Gamma GMRFs with log-effort offsets: D_c(s, t) = p_c · μ_c                      │
│                                       │                                                │
│                                       ▼                                                │
│      [ Step 6D: Stage-Specific Dispersal & Ontogenetic Migration ]                     │
│      - Transition kernels Γ_c(∇HSI_c, W) driving stage-specific particle advection     │
│                                       │                                                │
│                                       ▼                                                │
│ [ Step 7D: Master Demographic Refugia SQL Analytics & GeoJSON GIS Maps ]               │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Formulations

### 2.1. Rectangular Tow Expansion

Let $\mathcal{I} = \{1, \dots, N_{\text{tows}}\}$ be the set of survey tows, each associated with
spatial coordinates $\mathbf{s}_i = (x_i, y_i)$, year $t_i$, month $m_i$, and swept area $A_i$ (effort).
Let $\mathcal{C} = \{1, \dots, C\}$ be the demographic classification scheme:

1. **Functional Life Stages ($C = 4$)**:
   - $c = 1$: **Immature Juveniles** ($\text{CW} < 60\text{ mm}$, immature male or female).
   - $c = 2$: **Adolescent Males** ($60 \le \text{CW} < 95\text{ mm}$, immature male).
   - $c = 3$: **Mature Females** (all sizes, mature female spawning stock).
   - $c = 4$: **Commercial Males** ($\text{CW} \ge 95\text{ mm}$, mature hard-shell male).
2. **Fine Bins ($C = L \times 2 \times 2 = 20$)**:
   - Combinations of size bin $\ell \in \{1, \dots, L\}$, sex $g \in \{0, 1\}$, and maturity $m \in \{0, 1\}$.

For each tow $i$ and demographic class $c$, the observation vector contains:
- Encounter binary indicator: $y_{i, c} = \mathbb{I}(n_{i, c} > 0) \in \{0, 1\}$.
- Sub-sample count: $n_{i, c} \in \mathbb{Z}_{\ge 0}$.
- Apportioned biomass catch: $z_{i, c} = Z_i \cdot \frac{n_{i, c}}{\sum_{k=1}^C n_{i, k}} \ge 0$.

---

### 2.2. Tier 4C: Multi-Species Community Ordination via Householder Reflection Transformation (`eigen()` in TethysBayes)

To account for biological interactions (predator/competitor gradients) that influence juvenile settlement
and adult crab foraging, multi-species trawl catches $\mathbf{Y} \in \mathbb{R}^{N_{\text{hauls}} \times K_{\text{spp}}}$
are ordinated using TethysBayes's Householder eigenvector parameterization:

1. **Hellinger Transformation & Centering**:
   $$
   H_{ik} = \sqrt{\frac{Y_{ik}}{\sum_{j=1}^{K_{\text{spp}}} Y_{ij}}}, \qquad \tilde{\mathbf{H}} = \mathbf{H} - \mathbf{1} \bar{\mathbf{h}}^\top
   $$
2. **Householder Reflection Parameterization**:
   Rather than standard unconstrained SVD (which lacks probabilistic priors and has sign ambiguity),
   TethysBayes parameterizes the orthonormal loadings matrix $\mathbf{U} \in \mathbb{R}^{K_{\text{spp}} \times k}$ ($k=2$)
   via a sequence of $k$ elementary Householder reflections:
   $$
   \mathbf{U} = \prod_{i=1}^k \left( \mathbf{I} - 2 \frac{\mathbf{v}_i \mathbf{v}_i^\top}{\Vert\mathbf{v}_i\Vert^2} \right) \begin{bmatrix} \mathbf{I}_{k \times k} \\ \mathbf{0}_{(K_{\text{spp}}-k) \times k} \end{bmatrix}
   $$
   The reflector vectors $\mathbf{v}_i$ are mapped bijectively from orthonormal bases via `eigenvector_to_householder(U)`
   and reconstructed via `householder_to_eigenvector(v_mat, K_spp, k)`.
3. **Latent Gradient Scores**:
   $$
   \mathbf{F} = \tilde{\mathbf{H}} \mathbf{U} = [\text{PC1}, \, \text{PC2}]
   $$
4. **Spatial GMRF Mapping & Resharding**:
   The ordination scores are modeled across the community network $G_{\text{comm}}$:
   $$
   \text{PC}_{j, i} = \alpha_j + \beta_j^{\text{depth}} \tilde{z}_i + \beta_j^{\text{grain}} \tilde{g}_i + \beta_j^{\text{temp}} \tilde{T}_i + \phi_j(s_i) + \psi_j(s_i, t_i) + \epsilon_{j, i}
   $$
   and resharded onto $G_{\text{master}}$ to yield $\widetilde{\text{PC1}}_i \sim \mathcal{N}(\mu_{\text{pc1}, i}, \sigma^2_{\text{pc1}, i})$
   and $\widetilde{\text{PC2}}_i \sim \mathcal{N}(\mu_{\text{pc2}, i}, \sigma^2_{\text{pc2}, i})$.

---

### 2.3. Tier 4D: Stage-Structured Encounter & Habitat Suitability (HSI)

For each demographic stage $c \in \mathcal{C}$, encounter probability is modeled on the master areal
network $G_{\text{master}} = (V, E)$ ($S$ units) across $T$ years:

$$
y_{i, c} \sim \operatorname{Bernoulli}(p_{i, c})
$$

$$
\operatorname{logit}(p_{i, c}) = \alpha_c + \beta_c^{\text{depth}} \tilde{z}_i + \beta_c^{\text{temp}} \tilde{T}_i + \beta_c^{\text{grain}} \tilde{g}_i + \beta_c^{\text{pc1}} \widetilde{\text{PC1}}_i + \beta_c^{\text{pc2}} \widetilde{\text{PC2}}_i + \phi_c(s_i) + \psi_c(s_i, t_i)
$$

where:
- $\tilde{z}_i, \tilde{T}_i, \tilde{g}_i, \widetilde{\text{PC1}}_i, \widetilde{\text{PC2}}_i$ are Errors-in-Variables (EIV)
  propagated predictions from upstream Tiers 1–4C.
- $\phi_c \sim \operatorname{BYM2}(W)$ is a stage-specific spatial intrinsic GMRF.
- $\psi_c \sim \operatorname{BYM2}(W) \otimes \operatorname{AR1}(\rho_c)$ is a separable space-time random effect.

The predicted encounter probability at spatial node $u$ and year $t$ defines the **Stage-Specific Habitat Suitability Index**:

$$
\text{HSI}_c(u, t) = \frac{1}{1 + \exp(-\hat{\eta}_c(u, t))} \in (0, 1)
$$

---

### 2.4. Tier 5D: Stage-Structured Density & Positive Catch Hurdle

For tows where demographic class $c$ is encountered ($y_{i, c} = 1$), the positive catch $z_{i, c} > 0$
is modeled with a Gamma distribution and a log-effort offset:

$$
z_{i, c} \mid y_{i, c} = 1 \sim \operatorname{Gamma}\left(\frac{\mu_{i, c}}{\nu_c}, \, \nu_c\right)
$$

$$
\log(\mu_{i, c}) = \log(A_i) + \alpha_c^+ + \beta_c^{\text{hsi}} \widetilde{\text{HSI}}_{c, i} + \beta_c^{\text{depth}+} \tilde{z}_i + \beta_c^{\text{temp}+} \tilde{T}_i + \beta_c^{\text{pc1}+} \widetilde{\text{PC1}}_i + \beta_c^{\text{pc2}+} \widetilde{\text{PC2}}_i + \phi_c^+(s_i, t_i)
$$

The **Expected Demographic Density** (kg / $\text{km}^2$) for life stage $c$ at node $u$ and year $t$ is:

$$
\hat{D}_c(u, t) = \hat{p}_c(u, t) \cdot \hat{\mu}_c(u, t)
$$

The total emergent population biomass density across all stages is:

$$
\hat{D}_{\text{total}}(u, t) = \sum_{c=1}^C \hat{D}_c(u, t)
$$

---

### 2.5. Step 6D: Stage-Specific Dispersal & Ontogenetic Migration

Unlike aggregate population models where a single velocity field moves all individuals uniformly,
Approach B evaluates **stage-specific directed advection operators** $\mathbf{A}_c$:

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

- **Immature Juveniles ($c=1$)**: High diffusion ($\omega_{\text{diff}} = 0.4$), directed dispersal from shallow nursery banks toward continental shelf troughs.
- **Commercial Males ($c=4$)**: High chemotactic sensitivity ($\gamma = 1.5$), steep concentration in deep cold basins ($T \in [0.5, 3.0]^\circ\text{C}$).

---

## 3. In-Depth Reflection: Approach A vs. Approach B

| Analytical Dimension | Approach A: Conditional Decomposition | Approach B: Class-Expanded Factorial / Hurdle |
| :--- | :--- | :--- |
| **Mathematical Structure** | $\hat{N}_{\text{T5}}(\mathbf{s}, t) \times \boldsymbol{\pi}(\ell) \times \rho(g \mid \ell) \times \mu(m \mid g, \ell)$ | $C$ independent or coupled hurdle models: $\hat{D}_c = p_c \cdot \mu_c$ |
| **Community Covariates** | Tier 4 Ordination feeds total catch $\hat{N}_{\text{T5}}$ | Tier 4C Householder Ordination feeds stage-specific $p_c$ and $\mu_c$ |
| **Sum-to-Total Mass Conservation** | **Strict Guarantee**: $\sum_{\ell, g, m} \hat{N}_{\text{demo}} \equiv \hat{N}_{\text{T5}}$ exactly. | **No Conservation**: $\sum_c \hat{D}_c$ is an emergent sum accumulating $C$ prediction errors. |
| **Survey Sub-sample Compatibility** | **Exact**: Tow total catch fits abundance; 30-crab sub-samples fit compositions. | **Risk of Sampling Zeros**: Unsampled rare classes appear as false absences ($y_{i, c} = 0$). |
| **Allometric Biology & $L_{50}$** | **Explicit Physiological Parameters**: Directly estimates $L_{50}$, molt slopes, and dimorphism. | **Implicit Interactions**: Requires complex multi-way environmental interaction terms. |
| **Sparsity & Zero-Inflation** | **Dense**: Maturity models condition on measured individuals ($N \approx 12,000$). | **High Sparsity**: Expanding to 20 classes induces $80\text{--}90\%$ zeros, hindering GMRF convergence. |
| **Ontogenetic Niche Segregation** | **Indirect**: Spatial size differences mediated through ALR spatial fields. | **Direct & Intuitive**: Directly maps distinct spatial boundaries for nurseries and spawning stock. |
| **Movement & Telemetry** | Single velocity field driven by aggregate $\nabla \text{HSI}$. | Stage-specific advection vectors $\mathbf{v}_c = \gamma_c \nabla \text{HSI}_c$ (ontogenetic migration). |
| **Primary Optimal Use Case** | **Stock Assessment, Dynamic Leslie Matrices, Quotas** | **Marine Protected Areas (MPAs), Nursery Refugia, GIS Maps** |

---

### 3.1. Detailed Caveats of Approach B

#### Caveat 1: Accumulation of Positive Bias in Emergent Totals
When summing $C$ separate hurdle density estimates $\sum_{c=1}^C \hat{p}_c \hat{\mu}_c$, each model's
log-normal or Gamma retransformation bias ($\mathbb{E}[\exp(\eta)] = \exp(\mu + \sigma^2/2)$) and
estimation variance additively compound. In data-sparse strata, the emergent aggregate total
$\sum_c \hat{D}_c$ frequently exceeds the directly fitted total biomass model by $15\text{--}35\%$.

#### Caveat 2: Sampling Zeros vs. Structural Ecological Zeros
In a standard survey trawl capturing 800 kg of snow crab ($\approx 2,000$ animals), biologists
randomly select a basket of 40 individuals for detailed carapace measurement. If no adolescent
males happen to be in the 40-crab basket, Approach B records $y_{i, \text{adolescent}} = 0$ (absence),
even though dozens were present in the net. This induces artificial downward bias on encounter
probabilities in high-density hauls.

---

## 4. Step-by-Step Code Execution & Walkthrough

The complete Approach B pipeline is implemented in [`docs/hierarchical_demographic/hierarchical_demographic.jl`](hierarchical_demographic.jl).

### 4.1. Single-Segment Direct Execution (`update_demographic_segment!`)

Scientists can fit or update any demographic segment in isolation:

```julia
include("docs/hierarchical_demographic/hierarchical_demographic.jl")

# 1. Update ONLY Tier 4C (Community Householder Ordination):
update_demographic_segment!(:tier4c; force=true)

# 2. Update ONLY Tier 4D (Stage-Structured HSI) with functional life stages:
update_demographic_segment!(:tier4d; opts=DemographicPipelineOptions(class_scheme=:functional_stages))

# 3. Update ONLY Tier 5D (Stage Density Hurdle):
update_demographic_segment!(:tier5d; force=true)

# 4. Simulate and visualize stage-specific ontogenetic migrations:
update_demographic_segment!(:step6d; force=true)
```

### 4.2. Command-Line Interface (CLI) Dispatch

```bash
# 1. Check pipeline cache status and traits without sampling:
julia --project=. docs/hierarchical_demographic/hierarchical_demographic.jl --status

# 2. Run full demographic workflow with 4 functional life stages:
julia --project=. docs/hierarchical_demographic/hierarchical_demographic.jl

# 3. Run full demographic workflow with fine-scale (20 bins) resolution:
julia --project=. docs/hierarchical_demographic/hierarchical_demographic.jl --fine-bins

# 4. Force re-fitting of Tier 4C Community Householder Ordination:
julia --project=. docs/hierarchical_demographic/hierarchical_demographic.jl --tier4c --force

# 5. Force re-fitting of Tier 4D Encounter & HSI:
julia --project=. docs/hierarchical_demographic/hierarchical_demographic.jl --tier4d --force
```

---

## 5. DuckDB SQL Analytics & Essential Fish Habitat (EFH)

Integrated stage predictions are stored in `project_db/demographic_hierarchical_project.duckdb`.

### 5.1. SQL Query: Juvenile Nursery Grounds vs. Mature Spawning Refugia

```sql
SELECT 
    unit_id, s_x, s_y,
    hsi_immature_juvenile_mean AS juvenile_hsi,
    dens_immature_juvenile_mean AS juvenile_density,
    hsi_mature_female_mean AS spawning_female_hsi,
    dens_mature_female_mean AS spawning_female_density,
    depth_mean, temp_mean, grain_mean, pc1_mean, pc2_mean
FROM master_demographic_summary
WHERE hsi_immature_juvenile_mean >= 0.65 OR hsi_mature_female_mean >= 0.65
ORDER BY juvenile_density DESC;
```

---

## 6. Summary Recommendations

- **Use Approach A ([`docs/hierarchical_workflow/`](../hierarchical_workflow/))** when the primary objective
  is **biomass assessment, harvest advice, or multi-stage population dynamics** where marginal consistency
  and strict conservation of total numbers are mandatory.
- **Use Approach B ([`docs/hierarchical_demographic/`](./))** when the objective is **spatial conservation planning,
  juvenile nursery identification, or ontogenetic migration modeling** where distinct environmental niches
  and life-stage advection vector fields are the primary focus.
