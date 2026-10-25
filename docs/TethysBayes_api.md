---
title: "TethysBayes Technical API Reference"
format: html
---

# TethysBayes Technical API Reference

This document provides a comprehensive technical reference for the internal architecture, APIs, components, and extension points of the `TethysBayes` framework. It covers the core modeling macro `@tbm`, formula parsing engine, configuration pipeline, `ComponentModel` interface, Parameter Registry system, mathematical component formulations, inference engines, posterior reconstruction, plotting subsystem, spatial partitioning infrastructure, and two-tier persistence subsystem.

---

## 1. Framework Architecture & End-to-End Workflow

The `TethysBayes` framework translates high-level domain formulas into optimized, differentiable Turing.jl probabilistic programs, executes MCMC sampling or variational inference, and reconstructs structured posterior effects:

```
                               ┌────────────────────────────────────────────────────────┐
                               │  User Model Call: @tbm(formula, data, kwargs...)      │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │  Formula Parser: decompose_TethysBayes_formula                │
                               │  (LHS Likelihood & RHS AST Module Categorization)      │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │  Configuration Engine: TethysBayes_config                     │
                               │  (Precomputes, Dimensions, Hyperpriors, Registry)      │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │  Turing Code Generator: TethysBayes_text_assembler            │
                               │  (Dynamic @model generation with AD-safe constructors) │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │  Inference Engine: get_optimal_sampler / TethysBayes_sample   │
                               │  (Automatic Gibbs block-partitioning & NUTS sampling)  │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │  Posterior Reconstruction: model_results_comprehensive│
                               │  (Latent discovery, ParamRegistry lookup, TethysBayes_plots)  │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │  Persistence & SQL Analytics: save_TethysBayes_bundle / DuckDB│
                               │  (JLD2 live models, DuckDB tables, GeoJSON, CSV)       │
                               └────────────────────────────────────────────────────────┘
```

---

## 2. The `@tbm` Macro & Configuration Engine

### 2.1. Macro Signature & Top-Level Options

```julia
m = @tbm(formula, data; W=nothing, au=nothing, verbose=false, kwargs...)
```

| Argument / Keyword | Type | Default | Description & Implications |
| :--- | :--- | :--- | :--- |
| `formula` | `Expr` | *Required* | Model formula specifying observation likelihood on the LHS and latent additive components on the RHS (e.g. `likelihood(y) ~ intercept() + fixed(x) + random(s, model=bym2)`). |
| `data` | `DataFrame` | *Required* | Tabular dataset containing outcome variables, covariates, spatial indices, coordinates, and temporal stamps. |
| `W` | `AbstractMatrix` | `nothing` | Spatial adjacency matrix ($S \times S$) required for discrete GMRF models (`bym2`, `icar`, `besag`, `leroux`, `sar`, etc.). |
| `au` | `NamedTuple` / `Any` | `nothing` | Areal units object from `assign_spatial_units` containing polygon boundaries, centroids, and neighborhood graphs. |
| `verbose` | `Bool` | `false` | When `true`, prints generated Turing model code, compilation metadata, and automatic prior predictive check summaries. |
| `prior_scheme` | `Symbol` | `:pcpriors` | Global prior default scheme: `:pcpriors` (Penalized Complexity), `:informative`, `:uninformative`. |
| `priors` | `Dict{Symbol, Any}` | `Dict()` | Explicit prior overrides mapped by parameter name or component key. |
| `init_params` | `Dict{Symbol, Any}` | `nothing` | Custom initial parameter values for MCMC chains or optimization routines. |
| `sampler_choice` | `Symbol` | `:auto` | Default sampler selection: `:auto` (triggers `get_optimal_sampler`), `:nuts`, `:hmc`, `:mh`, `:slice`, `:variational`. |
| `adtype` | `ADTypes.AbstractADType` | `AutoForwardDiff()` | Automatic differentiation backend passed to Turing (`AutoForwardDiff()`, `AutoReverseDiff()`, `AutoZygote()`). |

---

### 2.2. Left-Hand Side (LHS) Likelihood Options (`likelihood()`)

The LHS `likelihood(outcome, ...)` defines the observation likelihood distribution and observational noise structure:

| Parameter | Type | Default | Description & Mathematical Role |
| :--- | :--- | :--- | :--- |
| `family` | `Symbol` | `:gaussian` | Likelihood distribution family (20 families supported). |
| `reference` | `Symbol` / `String` | `nothing` | Reference category for unordered multi-class models (`:categorical`, `:multinomial`, etc.). Defaults to first category or column. |
| `log_offsets` | `Symbol` / `Vector` | `nothing` | Additive offset on the linear predictor link scale: $\eta' = \eta + \text{offset}$. Essential for modeling rates (e.g., $\log(\text{Expected})$ in Poisson models). |
| `weights` | `Symbol` / `Vector` | `nothing` | Observation-level log-likelihood weighting: $\ell_i(\theta) = w_i \cdot \log p(y_i \mid \eta_i)$. |
| `trials` | `Symbol` / `Vector` | `nothing` | Number of binomial trials $N_i$ for `:binomial` and `:betabinomial` families. |
| `zero_inflated` | `Bool` | `false` | Enables structural zero inflation: $p(y=0) = \pi + (1-\pi)p_0$, $p(y>0) = (1-\pi)p(y)$. |
| `hurdle` | `Real` / `Bool` | `false` | Truncates likelihood below the hurdle threshold and models binary passage independently. |
| `censor_lower` | `Symbol` / `Vector` | `nothing` | Left-censoring bound: observation is known only to satisfy $y_i \le c_{\text{lower}, i}$. |
| `censor_upper` | `Symbol` / `Vector` | `nothing` | Right-censoring bound: observation is known only to satisfy $y_i \ge c_{\text{upper}, i}$. |
| `volatility` | `Bool` | `false` | Enables spatiotemporal stochastic volatility on observation variance $\sigma_{y, i}$. |

#### Supported Likelihood Families

| Family (`family=...`) | Link Function | Output Support | Key Parameters & Priors | Assumptions & Utility |
| :--- | :--- | :--- | :--- | :--- |
| `:gaussian` | $\mu = \eta$ | $y \in \mathbb{R}$ | Residual standard deviation $\sigma_y \sim \operatorname{Exponential}(1.0)$. | Continuous symmetrically distributed residuals with homogeneous or heteroscedastic noise. |
| `:poisson` | $\lambda = \exp(\eta)$ | $y \in \{0, 1, 2, \dots\}$ | Rate parameter $\lambda_i = \exp(\eta_i + \text{offset}_i)$. | Equidispersed count data where $\operatorname{Var}(y) = \mathbb{E}[y]$. Standard for disease rates. |
| `:negbin` | $\mu = \exp(\eta)$ | $y \in \{0, 1, 2, \dots\}$ | Dispersion parameter $r \sim \operatorname{Gamma}(2.0, 0.5)$, $\operatorname{Var}(y) = \mu + \mu^2 / r$. | Overdispersed count data capturing unobserved heterogeneity. |
| `:bernoulli` | $p = \operatorname{logistic}(\eta)$ | $y \in \{0, 1\}$ | Success probability $p_i = 1 / (1 + \exp(-\eta_i))$. | Binary presence/absence and classification outcomes. |
| `:binomial` | $p = \operatorname{logistic}(\eta)$ | $y \in \{0, 1, \dots, N_i\}$ | $y_i \sim \operatorname{Binomial}(N_i, p_i)$. Requires `trials`. | Aggregated binary trials (e.g. positive tests out of total tested). |
| `:beta` | $\mu = \operatorname{logistic}(\eta)$ | $y \in (0, 1)$ | Precision parameter $\kappa \sim \operatorname{Exponential}(1.0)$. | Continuous proportions, percentages, and fractional coverage indices. |
| `:gamma` | $\mu = \exp(\eta)$ | $y > 0$ | Shape $\alpha \sim \operatorname{Exponential}(1.0)$, scale $\theta = \mu / \alpha$. | Right-skewed positive continuous data with constant coefficient of variation. |
| `:lognormal` | $\mu_{\log} = \eta$ | $y > 0$ | Scale $\sigma_{\log} \sim \operatorname{Exponential}(1.0)$. | Multiplicative growth processes and heavy-tailed positive measurements. |
| `:studentt` | $\mu = \eta$ | $y \in \mathbb{R}$ | Degrees of freedom $\nu \sim \operatorname{Gamma}(2.0, 0.1)$, scale $\sigma \sim \operatorname{Exponential}(1.0)$. | Heavy-tailed robust regression resistant to outlier contamination. |
| `:exponential` | $\lambda = \exp(-\eta)$ | $y > 0$ | Rate parameter $\lambda_i = \exp(-\eta_i)$. | Memoryless survival time and inter-arrival event durations. |
| `:weibull` | $\lambda = \exp(\eta)$ | $y > 0$ | Shape parameter $k \sim \operatorname{Exponential}(1.0)$. | Monotonically increasing or decreasing hazard rates in survival analysis. |
| `:gev` | $\mu = \eta$ | $y \in \mathbb{R}$ | Generalized Extreme Value: scale $\sigma > 0$, shape $\xi \in \mathbb{R}$. | Block maxima modeling in environmental hydrology and extreme weather. |
| `:zipoisson` | $\lambda = \exp(\eta)$ | $y \in \{0, 1, \dots\}$ | Zero-inflation probability $\pi \sim \operatorname{Beta}(1, 1)$. | Excess zeros arising from dual generating processes (structural + sampling zeros). |
| `:zinegbin` | $\mu = \exp(\eta)$ | $y \in \{0, 1, \dots\}$ | Zero-inflation $\pi \sim \operatorname{Beta}(1, 1)$, dispersion $r \sim \operatorname{Gamma}(2.0, 0.5)$. | Overdispersed counts with structural zero inflation. |
| `:ordered_logistic` | Cutpoints $c_k$ | $y \in \{1, \dots, K\}$ | Ordered categorical threshold vector $c_1 < c_2 < \dots < c_{K-1}$. | Likert scale survey responses and graded disease severity stages. |
| `:ordered_probit` | Cutpoints $c_k$ | $y \in \{1, \dots, K\}$ | Standard normal CDF link $\Phi(\cdot)$ with ordered cutpoints. | Latent Gaussian threshold crossing models for ordinal ratings. |
| `:categorical` | Softmax $\eta_k$ | $y \in \{1, \dots, K\}$ | Categorical probability vector $p = \operatorname{softmax}(\eta)$ with reference category $\eta_1 \equiv 0$. | Unordered multi-class choice and state classifications. |
| `:multinomial` | Softmax $\eta_k$ | Vector counts | Multinomial count vector across $K$ categories with reference category $\eta_1 \equiv 0$. | Compositional count data across competing categorical outcomes. |
| `:dirichlet_multinomial` | Softmax $\eta_k$ + Dirichlet $\phi$ | Vector counts | Multinomial counts with Dirichlet dispersion $\phi \sim \operatorname{Exponential}(1.0)$. | Overdispersed compositional count data with latent linear predictors. |
| `:dirichlet` | Softmax $\eta_k$ + Dirichlet $\phi$ | Simplex $\Delta^{K-1}$ | Dirichlet concentration vector $\boldsymbol{\alpha} = \phi \cdot \operatorname{softmax}(\boldsymbol{\eta})$. | Continuous compositional proportions summing to 1. |

---

### 2.2.1. Multinomial, Categorical, and Compositional Likelihoods

The `TethysBayes` framework provides native, mathematically identified formulations for unordered
multi-class, multinomial count, and compositional proportion data across $K \ge 2$ categories.

#### Model Specifications & Formula Syntax

1. **Wide Count Format (Multinomial / Dirichlet-Multinomial)**:
   Observations are represented as multiple count columns summing to total trials per row:
   ```julia
   formula = @formula(likelihood(c1 + c2 + c3, family=:multinomial) ~
                      intercept() + fixed(x1) + random(s_idx, model=bym2))
   ```
   Each row vector $\mathbf{y}_i = [y_{i, 1}, \dots, y_{i, K}]^\top$ represents counts
   across $K$ categories with total trials $N_i = \sum_{k=1}^K y_{i, k}$.

2. **Long Categorical Format (Categorical / Multi-Class Logit)**:
   Observations are represented as an integer, string, or symbol categorical column:
   ```julia
   formula = @formula(likelihood(species, family=:categorical, reference="cod") ~
                      intercept() + fixed(temperature) + random(time, model=ar1))
   ```
   The `reference` keyword specifies the baseline category (defaulting to the first level).

3. **Dirichlet and Dirichlet-Multinomial Formulations**:
   - `:dirichlet_multinomial`: Accommodates overdispersed count compositions via a learned
     Dirichlet dispersion parameter $\phi \sim \operatorname{Exponential}(1.0)$.
   - `:dirichlet`: Accommodates continuous simplex observations $\mathbf{y}_i \in \Delta^{K-1}$
     where $\sum_{k=1}^K y_{i, k} = 1$ and $y_{i, k} > 0$.

#### Mathematical Parameterization & Identifiability

In standard multinomial logit models, the softmax transformation is invariant under uniform
translation:
$$\operatorname{softmax}(\boldsymbol{\eta} + c) = \operatorname{softmax}(\boldsymbol{\eta}), \quad \forall c \in \mathbb{R}$$
Estimating $K$ unconstrained linear predictors introduces complete non-identifiability,
inducing MCMC random-walk drift and poor sampling efficiency.

To guarantee mathematical identifiability, `TethysBayes` constrains the linear predictor of the
reference category ($k = 1$) to zero:
$$\eta_{i, 1} \equiv 0$$
For non-reference categories $k \in \{2, \dots, K\}$, latent linear predictors $\eta_{i, k}$
are estimated from the additive latent components:
$$\eta_{i, k} = \alpha_k + \mathbf{x}_i^\top \boldsymbol{\beta}_k + u_{i, k} + \dots$$
The category probability vector $\mathbf{p}_i = [p_{i, 1}, \dots, p_{i, K}]^\top$ on the
simplex $\Delta^{K-1}$ is evaluated via numerically stable softmax:
$$p_{i, k} = \frac{\exp(\eta_{i, k})}{\sum_{j=1}^K \exp(\eta_{i, j})} = \operatorname{softmax}(\boldsymbol{\eta}_i)_k$$

For Dirichlet and Dirichlet-Multinomial families, the Dirichlet concentration vector
$\boldsymbol{\alpha}_i$ is parameterized by precision $\phi$:
$$\boldsymbol{\alpha}_i = \phi \cdot \mathbf{p}_i, \quad \phi \sim \operatorname{Exponential}(1.0)$$
$$\mathbf{y}_i \sim \operatorname{DirichletMultinomial}(N_i, \boldsymbol{\alpha}_i)$$

#### Posterior Reconstruction Outputs

Posterior reconstruction via `model_results_comprehensive` automatically produces:
- **`probabilities`**: Posterior mean and 95% credible intervals for each category probability
  $p_{i, k}$ across all observations $i=1,\dots,N$, strictly satisfying $\sum_{k=1}^K p_{i,k} = 1$.
- **`predicted_category`**: Maximum a posteriori (MAP) predicted dominant category
  $\hat{y}_i = \operatorname{argmax}_k p_{i, k}$ mapped back to categorical factor labels.
- **Joint WAIC / Pointwise Log-Likelihood**: Evaluated across multivariate outcome vectors
  for Bayesian model comparison and leave-one-out cross-validation.

---

### 2.3. Right-Hand Side (RHS) Component Syntax

The RHS formula combines linear fixed effects, structured random fields, and process dynamics:

| Module        | Purpose                                          | Key Parameters                               | Example Usage                                         |          |
| :--------------| :-------------------------------------------------| :---------------------------------------------| :------------------------------------------------------| ----------|
| `intercept()` | Controls global intercept prior.                 | `prior`                                      | `intercept(prior=Normal(0, 5))`                       |          |
| `fixed()`     | Fixed-effect regression coefficients.            | `prior`, `contrast`                          | `fixed(elevation, prior=Normal(0, 1))`                |          |
| `random()`    | Structured & unstructured random fields.         | `model`, `sigma`, `rho`, `length_scale`, etc. | `random(s_idx, model=bym2)`                           |          |
| `mixed()`     | Correlated random slopes and intercepts.         | `model`, `method`                            | `mixed(1 + poverty \                                  | region)` |
| `dynamics()`  | Mechanistic state-space differential equations.  | `model`, `r`, `K`, `velocity`, `diffusion`   | `dynamics(time, model=:logistic, r=Normal(0.5, 0.1))` |          |
| `eigen()`     | Bayesian PCA factor analysis.                    | `n_factors`, `pca_sd`                        | `eigen(pollutant1, pollutant2, n_factors=1)`          |          |
| `nested()`, `transfer()`, `fidelity()` | Multi-fidelity supervised proxy models & transfer learning. | `formula`, `data_source`, `mapping`, `prior`, `fixed` | `transfer(:proxy)` or `nested(proxy, formula="...", data_source=df)` |
| `sciml()`     | Scientific Machine Learning ODE/PDE integration. | `model_func`, `solver`                       | `sciml(t, model_func=my_ode)`                         |          |
| `custom()`    | User-injected raw Turing code fragments.         | `code_fragment`                              | `custom(code_fragment="...")`                         |          |

---

### 2.4. Formula Algebraic Operators

The RHS parser (`decompose_TethysBayes_formula`) supports algebraic composition operators:

- **Addition (`+`)**: Additive combination of linear terms ($\eta = \eta_1 + \eta_2$).
- **Kronecker Product (`⊗`)**: Spatiotemporal or multidimensional interaction ($A \otimes B$).
  - *Example*: `random(s_idx, model=icar) ⊗ random(year, model=ar1)` constructs a Knorr-Held Type IV space-time interaction field ($Q_{st} = Q_t \otimes Q_s$).
- **Pipe (`|>`)**: Spatially-varying or time-varying coefficient models.
  - *Example*: `poverty |> random(s_idx, model=icar)` constructs a spatially-varying slope $\beta(s) \cdot \text{poverty}_i$.
  - *Example*: `random(s_idx, model=icar) |> random(month, model=pspline)` constructs spatially-varying seasonal splines.
- **Composition (`∘`)**: Hierarchical modulation or Log-Gaussian Cox Processes.
  - *Example*: `pointprocess(model=:lgcp) ∘ random(s_idx, model=icar)`.

---

### 2.5. Prior Specification & Penalized Complexity (PC) Priors API

`TethysBayes` provides automated mapping between intuitive quantile constraints and mathematical prior distributions:

```julia
# 1. PC Prior Quantile Constraints in Formula Calls
@tbm(
    likelihood(y, family=poisson) ~
        intercept(prior = Normal(0, 5)) +
        fixed(elevation, prior = (2.0, 0.05)) +       # P(|β| > 2.0) = 0.05 => Normal(0, 1.02)
        random(s_idx, model=bym2, 
            sigma = (1.0, 0.01),                     # P(σ > 1.0) = 0.01   => Exponential(4.605)
            rho = (0.5, 0.05)                        # P(ρ > 0.5) = 0.05   => Exponential on -log(1-ρ)
        ) +
        random(time, model=gp, 
            length_scale = (0.1, 0.01)                # P(ℓ < 0.1) = 0.01   => Exponential on 1/ℓ
        ),
    df, W=W
)
```

| Parameter Type | Quantile Constraint Syntax | Base Model State | Induced Prior Distribution |
| :--- | :--- | :--- | :--- |
| **Standard Deviation (`sigma`, `range`)** | `(U, α)` $\implies P(\sigma > U) = \alpha$ | $\sigma = 0$ (No variation) | $\operatorname{Exponential}(\lambda), \; \lambda = -\log(\alpha)/U$ |
| **Correlation (`rho`)** | `(U, α)` $\implies P(\rho > U) = \alpha$ | $\rho = 0$ (Independent noise) | $\operatorname{Exponential}(\lambda)$ on $\theta = -\log(1-\rho)$ |
| **Length scale (`length_scale`)** | `(U, α)` $\implies P(\ell < U) = \alpha$ | $\ell = \infty$ (Flat constant) | $\operatorname{Exponential}(\lambda)$ on $\theta = 1/\ell$ |
| **Fixed Effects / Slopes (`prior`)** | `(U, α)` $\implies P(\|\beta\| > U) = \alpha$ | $\beta = 0$ (Null effect) | $\operatorname{Normal}(0, \sigma_{\beta}), \; \sigma_{\beta} = \frac{-U}{\Phi^{-1}(\alpha/2)}$ |

---

## 3. The `ComponentModel` Interface & Extension Guide

To add a new latent component, create a struct subtyping `ComponentModel` and implement the four core interface methods.

### 3.1. Interface Lifecycle & Methods

```
                        ┌────────────────────────────────────────────────────────┐
                        │  1. get_precomputes(m, M, mod_data)                    │
                        │     (Data validation, basis/precision precomputes)     │
                        └───────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
                        ┌────────────────────────────────────────────────────────┐
                        │  2. get_priors(m, spec, arch, outcome_idx, M)          │
                        │     (Generates Turing prior code strings)              │
                        └───────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
                        ┌────────────────────────────────────────────────────────┐
                        │  3. get_updates(m, spec, arch, outcome_idx, M)         │
                        │     (Calculates latent effect and updates eta)         │
                        └───────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
                        ┌────────────────────────────────────────────────────────┐
                        │  4. get_effects(m, chain, M, n_samples, ...)           │
                        │     (Extracts posterior samples and computes effects)  │
                        └────────────────────────────────────────────────────────┘
```

#### Method Specifications:

1. **`get_precomputes(m::ComponentModel, M::NamedTuple, mod_data::Dict)::NamedTuple`**
   - Validates required columns in `M.data` and generates data structures (e.g., basis matrices $B$, precision matrix templates $Q$, eigenvalue decompositions $U, \Lambda$). Stored in `spec.hyper`.

2. **`get_priors(m::ComponentModel, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String`**
   - Emits Turing `@model` code declaring prior distributions for hyperparameters (e.g., `sigma`, `rho_unconstrained`) and standard normal innovations `innovations`.

3. **`get_updates(m::ComponentModel, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String`**
   - Emits Turing code computing the realized structured latent field `latent_field` from `innovations` and hyperparameters, and adds the contribution into the linear predictor `eta`.

4. **`get_effects(m::ComponentModel, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple`**
   - Extracts posterior samples from `chain` via `ParamRegistry` and reconstructs posterior trajectories and credible intervals for post-processing and plotting.

---

### 3.2. Canonical Implementation Example: The `IID` Component

Below is a complete implementation of an unstructured group-level random effect ($\phi_g \sim \mathcal{N}(0, \sigma^2)$):

```julia
# 1. Struct Definition
struct IID <: ComponentModel
    sigma::Distribution
    method::Symbol # :noncentered, :centered, :marginalized
end

# 2. Registration in TethysBayes registries
COMPONENT_TYPE_REGISTRY[:iid] = IID
COMPONENT_CONSTRUCTORS[:iid] = (p, params) -> IID(
    get(p, :sigma, Exponential(1.0)),
    get(params, :method, :noncentered)
)

# 3. Precomputations
function get_precomputes(m::IID, M::NamedTuple, mod_data::Dict)::NamedTuple
    n_latent = if mod_data[:structure] == :spatial
        M.s_N
    elseif mod_data[:structure] == :temporal
        M.t_N
    else
        length(unique(M.data[!, mod_data[:var]]))
    end
    return (n_latent = n_latent,)
end

# 4. Turing Priors Generator
function get_priors(m::IID, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent
    
    return """
    $(p_names.sigma) ~ $(_distribution_to_string(m.sigma))
    $(p_names.ure) ~ MvNormal(zeros(T, $(n_latent)), I)
    """
end

# 5. Turing Updates Generator
function get_updates(m::IID, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = spec.structure == :spatial ? "s_idx" : (spec.structure == :temporal ? "t_idx" : "mixed_idx_$(spec.var)")
    
    return """
    $(p_names.sre) = $(p_names.sigma) .* $(p_names.ure)
    $(eta_target) .+= $(p_names.sre)[$(index_var)]
    """
end

# 6. Posterior Reconstruction
function get_effects(m::IID, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    v = generate_full_variable_names(spec, M.model_arch, 1)
    sigma_samples = get_param_samples(chain, M.param_registry, Symbol(v.sigma))
    innovations_samples = get_param_samples(chain, M.param_registry, Symbol(v.innovations))
    
    latent_field = innovations_samples .* reshape(sigma_samples, 1, :)
    index_var = spec.structure == :spatial ? M.s_idx : (spec.structure == :temporal ? M.t_idx : M.data[!, spec.var])
    effect = latent_field[index_var, :]
    
    return (structured=effect, noisy=effect)
end
```

---

### 3.3. Canonical Naming Standard

All parameter symbols generated across components follow the strict sequence:

```
quantity_descriptor_key[_outcome]
```

- **Quantity**: `beta`, `sigma`, `rho`, `length_scale`, `innovations`, `latent_field`, `threshold_unconstrained`, `reflection_vector_unconstrained`, `alpha`, `carrying_capacity`, `intrinsic_growth_rate`.
- **Descriptor**: `unconstrained`, `unscaled`, `inducing`, `diag`, `pic`, `predator`, `cluster`, `st_interaction`, `flat`.
- **Key**: Unique component identifier derived from formula term (`s_idx`, `year`, `space`, etc.).
- **Outcome**: `1`, `2` (for multivariate models).

#### Core Token Definitions:
- **`beta` / `beta_flat`**: Fixed effects regression coefficients.
- **`innovations_<key>`**: i.i.d. standard-normal innovations. These carry no structure of their own; the component scales and correlates them.
- **`latent_field_<key>`**: the reconstructed field, after the component has scaled and correlated the innovations.
- **`rho_unconstrained_<key>`**: Unconstrained transform of correlation parameter $\rho \in (-1, 1)$ or $(0, 1)$.

---

## 4. Parameter Registry Engine (`src/parameters.jl`)

The `ParamRegistry` system provides central parameter discovery, support categorization, and MCMC extraction. Every parameter has exactly one registered name: there is no alias table and no abbreviation fallback, so a name is either registered or it is not.

### 4.1. Core Types

```julia
struct ParamDescriptor
    name::Symbol             # Canonical parameter name (e.g. :sigma_year)
    role::Symbol             # :fixed, :hyper, :latent, :innovation, :precision
    component::Symbol        # Component key (:fixed, :year, :s_idx)
    model_type::Symbol       # :fixed, :ar1, :bym2, :gp, etc.
    support::Symbol          # :real, :positive, :unit_interval, :discrete
    dimension::Int           # 1 for scalar, >1 for vector/matrix
    prior_str::String        # String representation of prior distribution
end

mutable struct ParamRegistry
    names::Vector{String}                                              # Canonical string names
    descriptors::Dict{Symbol, ParamDescriptor}                         # Symbol -> ParamDescriptor
    by_component::Dict{Symbol, Dict{Symbol, Vector{ParamDescriptor}}} # component_key -> role -> [descriptors]
    by_base::Dict{String, Vector{String}}                              # base name -> list of matching full names
    name_to_key::Dict{String, Any}                                     # String/Symbol in chain -> actual indexing key
end
```

### 4.2. Registry Functions

- **`build_param_registry(M::NamedTuple)::ParamRegistry`**: Scans model specification `M` and compiles an initial parameter registry.
- **`calibrate_param_registry(reg::ParamRegistry, vi::VarInfo)::ParamRegistry`**: Introspects DynamicPPL `VarInfo` to calibrate active variable names, dimensions, and empirical supports.
- **`get_param_samples(chain, reg::ParamRegistry, param_sym::Symbol)`**: Fetches posterior samples across `FlexiChain`, `Chains`, `DataFrame`, or `Dict` containers by exact name. There is no alias or abbreviation fallback: every parameter has exactly one registered name, and an unregistered name returns nothing and names the near-misses.
- **`_find_parameter(reg::ParamRegistry, target_name::Symbol)`**: Resolves a parameter name across a chain, accounting for multivariate outcome suffixes (`_1`, `[1]`). It does not guess between candidate names.

---

## 5. Statistical Component Models Taxonomy

### 5.1. Spatial Areal & Graph GMRF Models

#### 1. Intrinsic Conditional Autoregressive (`ICAR`)
- **Mathematical Formulation**:

  $$
  u_i \mid u_{-i} \sim \mathcal{N}\left( \frac{1}{d_i} \sum_{j \in N(i)} u_j, \frac{\sigma^2}{d_i} \right)
  $$

  In matrix notation, $u \sim \mathcal{N}(0, (\tau Q)^{-1})$ where $Q = \operatorname{diag}(W \mathbf{1}) - W$ is the singular graph Laplacian and $\tau = 1/\sigma^2$.
- **Core Assumptions**: First-order intrinsic stationarity; singular precision matrix with sum-to-zero constraint $\sum_{i=1}^S u_i = 0$; spatial Markov property (conditional independence given neighbors).
- **Utility**: Standard spatial smoothing across discrete administrative or geographic regions (Besag, 1974).

#### 2. Besag-York-Mollié (`BYM2`)
- **Mathematical Formulation**:

  $$
  \phi_s = \sigma \left( \sqrt{1 - \rho} \cdot v_s + \sqrt{\rho / s_{\text{scale}}} \cdot u_s \right)
  $$

  where $u \sim \operatorname{ICAR}(W)$, $v \sim \mathcal{N}(0, I_S)$, $\rho \in [0, 1]$ is the spatial mixing parameter, and $s_{\text{scale}} = \exp(-\frac{1}{S-1}\sum_{i=1}^{S-1} \log \lambda_i)$ is the Riebler et al. (2016) spectral scaling factor.
- **Core Assumptions**: Additive orthogonal decomposition into spatially structured clustering ($u$) and unstructured white noise ($v$); graph is fully connected.
- **Utility**: The gold standard in spatial epidemiology and disease mapping. Allows direct interpretation of $\rho$ as the proportion of total variance explained by spatial autocorrelation.

#### 3. Leroux CAR (`Leroux`)
- **Mathematical Formulation**:

  $$
  Q = (1 - \rho) I_S + \rho Q_{\text{ICAR}}, \quad \phi \sim \mathcal{N}\left(0, (\sigma^2 Q)^{-1}\right)
  $$

- **Core Assumptions**: Full-rank proper precision matrix for all $\rho \in [0, 1)$; smooth convex interpolation between pure independence ($\rho=0$) and ICAR ($\rho=1$).
- **Utility**: Prevents boundary edge singularities and avoids rigid sum-to-zero constraints; highly stable for small sample sizes.

#### 4. Simultaneous Autoregressive (`SAR`)
- **Mathematical Formulation**:

  $$
  \phi = \rho W_{\text{std}} \phi + \epsilon, \quad \phi = (I - \rho W_{\text{std}})^{-1} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)
  $$

- **Core Assumptions**: Spatial lag autoregressive process; stationarity requires $\rho \in (1/\lambda_{\min}, 1/\lambda_{\max})$.
- **Utility**: Spatial econometrics, hedonic pricing, and geographic spillover modeling.

#### 5. Local Adaptive GMRF (`LocalAdaptive`)
- **Mathematical Formulation**:

  $$
  \phi \sim \mathcal{N}\left( \mu_{\text{cluster}(s)}, (\sigma^2 Q_{\text{Leroux}})^{-1} \right)
  $$

  where $\mu_g$ is a cluster-specific mean effect estimated across spatial clusters $g \in \{1, \dots, K\}$.
- **Core Assumptions**: Continuous background spatial covariance with piecewise discontinuous mean shifts across geographic regimes.
- **Utility**: Environmental epidemiology across distinct jurisdictions, socioeconomic divides, or natural geographic boundaries.

#### 6. Multivariate Conditional Autoregressive (`MCAR`)
- **Mathematical Formulation**:

  $$
  \operatorname{vec}(\boldsymbol{\Phi}) \sim \mathcal{N}\left(\mathbf{0}, (\mathbf{\Omega}_{\text{cross}} \otimes \mathbf{Q}_{\text{spatial}})^{-1}\right)
  $$

  where $\mathbf{Q}_{\text{spatial}} = (1 - \rho)\mathbf{I}_S + \rho \mathbf{Q}_{\text{ICAR}}$ is the proper spatial precision matrix, and $\mathbf{\Sigma}_{\text{cross}} = \mathbf{\Omega}_{\text{cross}}^{-1} = \operatorname{diag}(\boldsymbol{\sigma})\mathbf{L}_{\text{corr}}\mathbf{L}_{\text{corr}}^T \operatorname{diag}(\boldsymbol{\sigma})$ captures cross-outcome correlations via an LKJ prior.
- **Core Assumptions**: Cross-outcome and spatial dependencies factorize into a Kronecker separable precision structure.
- **Utility**: Joint multi-disease spatial mapping and multi-species ecological co-occurrence across geographic regions (Gelfand & Vounatsou, 2003).

---

### 5.2. Continuous Geostatistical & Point-Referenced Models

#### 1. Gaussian Process (`GP`)
- **Mathematical Formulation**:

  $$
  f(x) \sim \mathcal{GP}(0, k(x, x')), \quad k_{\text{SE}}(d) = \sigma^2 \exp\left( -\frac{d^2}{2\ell^2} \right), \quad k_{\text{Matérn}}(d) = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\sqrt{2\nu}\frac{d}{\ell}\right)^\nu K_\nu\left(\sqrt{2\nu}\frac{d}{\ell}\right)
  $$

- **Core Assumptions**: Second-order stationarity and isotropy (or anisotropic ARD lengthscales $\ell_d$).
- **Utility**: Exact spatial interpolation (Kriging), continuous environmental field mapping, sensor fusion.

#### 2. Nearest Neighbor Gaussian Process (`NNGP`)
- **Mathematical Formulation**:
  Approximates continuous GP joint densities via $m$-nearest neighbor conditioning sets among preceding points in a spatial ordering (Datta et al., 2016):

  $$
  p(w_1, \dots, w_N) \approx p(w_1) \prod_{i=2}^N \mathcal{N}\left(\mathbf{B}_i \mathbf{w}_{N(s_i)}, F_i\right)
  $$

  where $\mathbf{B}_i = C(s_i, N(s_i)) [C(N(s_i), N(s_i))]^{-1}$ and $F_i = C(s_i, s_i) - \mathbf{B}_i C(N(s_i), s_i)$.
- **Core Assumptions**: High-order conditional independence given the $m$ nearest preceding spatial neighbors.
- **Utility**: Highly scalable geostatistics achieving $\mathcal{O}(N m^3)$ runtime and $\mathcal{O}(N m)$ memory for point datasets $N > 10^5$ without pseudo-inputs.

#### 3. Sparse Gaussian Process (`SparseGP` / `FITC` / `PIC`)
- **Mathematical Formulation**:
  Approximates dense covariance $K_{ff}$ via $M \ll N$ pseudo-inputs $u = f(X_u)$ using the Fully Independent Training Conditional (FITC) factorization:

  $$
  K_{ff} \approx Q_{ff} + \operatorname{diag}(K_{ff} - Q_{ff}), \quad Q_{ff} = K_{fu} K_{uu}^{-1} K_{uf}
  $$

- **Core Assumptions**: Latent spatial field covariance is low-rank conditionally independent given inducing points.
- **Utility**: Scales Gaussian Process inference from $\mathcal{O}(N^3)$ to $\mathcal{O}(NM^2)$, making continuous GPs tractable for $N > 10^5$ observations.

#### 4. Random Fourier Features (`RFF`)
- **Mathematical Formulation**:
  By Bochner's theorem, stationary kernel $k(x - x') = \int p(\omega) e^{i \omega^T (x - x')} d\omega \approx z(x)^T z(x')$:

  $$
  z(x) = \sqrt{\frac{2}{D}} \left[ \cos(W x + b) \right], \quad W \sim \mathcal{N}(0, \ell^{-2} I), \quad b \sim \operatorname{Uniform}(0, 2\pi)
  $$

- **Core Assumptions**: Shift-invariant stationary kernel; projection dimension $D$ is sufficiently large.
- **Utility**: Linear $\mathcal{O}(ND)$ time complexity for continuous spatial fields in high-dimensional settings.

#### 5. Stochastic Partial Differential Equations (`SPDE`)
- **Mathematical Formulation**:
  Represents continuous Matérn fields as solutions to the linear fractional SPDE (Lindgren et al., 2011):

  $$
  (\kappa^2 - \Delta)^{\alpha/2} u(s) = \mathcal{W}(s)
  $$

  Discretized on a Constrained Delaunay Triangulation mesh via piecewise linear basis functions $u(s) = \sum_{i=1}^V \psi_i(s) u_i$.
- **Core Assumptions**: Matérn field smoothness $\nu = \alpha - d/2$; finite element discretization accurately captures spatial boundary conditions.
- **Utility**: Bridges continuous Gaussian fields with sparse precision GMRFs ($\mathcal{O}(N^{1.5})$ complexity).

---

### 5.3. Temporal Models

#### 1. Autoregressive Processes (`AR1`, `AR2`)
- **Mathematical Formulation**:
  - $\text{AR}(1)$: $x_t = \rho x_{t-1} + \sigma \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, 1)$
  - $\text{AR}(2)$: $x_t = \rho_1 x_{t-1} + \rho_2 x_{t-2} + \sigma \epsilon_t$
- **Core Assumptions**: Weak stationarity ($|\rho| < 1$ for AR1; triangular stability domain for AR2: $\rho_1 + \rho_2 < 1, \rho_2 - \rho_1 < 1, |\rho_2| < 1$).
- **Utility**: Serial autocorrelation, short-memory momentum, macro-temporal shocks.

#### 2. Random Walks (`RW1`, `RW2`)
- **Mathematical Formulation**:
  - $\text{RW}(1)$: $x_t - x_{t-1} \sim \mathcal{N}(0, \sigma^2)$ (first difference / stochastic level)
  - $\text{RW}(2)$: $x_t - 2x_{t-1} + x_{t-2} \sim \mathcal{N}(0, \sigma^2)$ (second difference / stochastic curvature)
- **Core Assumptions**: Non-stationary stochastic trends; RW2 enforces smooth second-order derivatives.
- **Utility**: Nonparametric time trend estimation without imposing rigid polynomial forms.

#### 3. Harmonic & Periodic Models (`Harmonic`, `Cyclic`)
- **Mathematical Formulation**:

  $$
  f(t) = \sum_{k=1}^K \left( \beta_{\cos, k} \cos\left(\frac{2\pi k t}{P}\right) + \beta_{\sin, k} \sin\left(\frac{2\pi k t}{P}\right) \right)
  $$

- **Core Assumptions**: Strict periodicity with fundamental cycle period $P$ (e.g. 12 months, 24 hours).
- **Utility**: Seasonal epidemiological peaks, environmental annual rhythms, tidal cycles.

#### 4. Threshold Autoregressive (`TAR`)
- **Mathematical Formulation**:

  $$
  x_t = \begin{cases} \rho_1 x_{t-1} + \sigma_1 \epsilon_t & \text{if } x_{t-d} \le c \\ \rho_2 x_{t-1} + \sigma_2 \epsilon_t & \text{if } x_{t-d} > c \end{cases}
  $$

- **Core Assumptions**: Piecewise linear regime switching governed by threshold variable and delay lag $d$.
- **Utility**: Ecological tipping points, predator-prey collapses, financial market regime shifts.

---

### 5.4. Spatiotemporal Interactions (`Composed`, `⊗`)

Knorr-Held (2000) spatiotemporal interactions are constructed via the Kronecker product operator:

$$
\eta_{st} = \alpha + u_s + v_t + \delta_{st}, \quad \delta \sim \mathcal{N}\left(0, \sigma_{\delta}^2 (Q_s \otimes Q_t)^{-1}\right)
$$

| Interaction Type | Spatial Structure ($Q_s$) | Temporal Structure ($Q_t$) | Interpretation |
| :--- | :--- | :--- | :--- |
| **Type I** | $I_S$ (Unstructured) | $I_T$ (Unstructured) | Uncorrelated white-noise space-time shocks. |
| **Type II** | $I_S$ (Unstructured) | $Q_{\text{RW1}}$ or $Q_{\text{AR1}}$ (Structured) | Spatial units follow independent, temporally persistent trajectories. |
| **Type III**| $Q_{\text{ICAR}}$ (Structured) | $I_T$ (Unstructured) | Spatially correlated maps that vary independently from year to year. |
| **Type IV** | $Q_{\text{ICAR}}$ (Structured) | $Q_{\text{RW1}}$ / $Q_{\text{AR1}}$ (Structured) | Spatially correlated patterns that evolve smoothly over continuous time. |

---

### 5.5. Mechanistic & Dynamical Systems (`dynamics()`)

The `dynamics()` module fuses physical and ecological differential equations into the Bayesian state-space framework:

1. **Logistic Population Growth**:

   $$
   N_t = N_{t-1} + r N_{t-1} \left(1 - \frac{N_{t-1}}{K}\right) + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2)
   $$

2. **Advection-Diffusion PDE**:

   $$
   \frac{\partial u}{\partial t} = D \nabla^2 u - v \cdot \nabla u + \epsilon(s, t)
   $$

   Discretized across graph Laplacian $L = D - W$:

   $$
   u_{t} = u_{t-1} - \Delta t \left( D L u_{t-1} + v \cdot \nabla u_{t-1} \right) + \epsilon_t
   $$

- **Core Assumptions**: Known physical/biological differential equations with stochastic environmental noise.
- **Utility**: Physics-Informed Machine Learning (SciML), population viability analysis, pollution plume tracking.

---

### 5.6. Bayesian Factor Analysis (`eigen()`)

Bayesian PCA decomposes $P$ multivariate outcomes into $K \ll P$ orthogonal latent factors:

$$
Y = Z \Lambda^{1/2} U^T + E, \quad E \sim \mathcal{N}(0, \operatorname{diag}(\sigma_{\epsilon, 1}^2, \dots, \sigma_{\epsilon, P}^2))
$$

To enforce strict orthonormality ($U^T U = I_K$) without identification sign-flipping or unconstrained matrix drift, `TethysBayes` parameterizes $U$ using a product of Householder reflections:

$$
H_k = I - 2 \frac{v_k v_k^T}{\|v_k\|^2}, \quad U = \prod_{k=1}^K H_k
$$

- **Core Assumptions**: High-dimensional outcomes are generated by a low-dimensional orthogonal latent subspace.
- **Utility**: Multi-pollutant exposure indices, ecological multi-species co-occurrence, multi-phenotype genetics.

---

### 5.7. Supervised Multi-Fidelity & Transfer Linkage Models (`nested()`, `transfer()`, `fidelity()`)

Multi-fidelity transfer modeling enables joint inference where a primary process (e.g., sparse, high-precision ground monitor observations) is augmented by one or more proxy processes (e.g., dense but biased satellite retrievals, low-fidelity numerical simulations, or auxiliary biological telemetry).

#### 5.7.1. Mathematical Formulation

Let $\mathbf{y}_{\text{hi}} \in \mathbb{R}^{N_{\text{hi}}}$ denote the primary observation vector and $\mathbf{y}_{\text{lo}, m} \in \mathbb{R}^{N_{\text{sub}, m}}$ denote auxiliary/proxy observation vectors for sub-models $m = 1, \dots, M$.

**Primary / High-Fidelity Process**:
$$
\mathbf{y}_{\text{hi}} \sim \operatorname{Likelihood}(\boldsymbol{\eta}_{\text{hi}}, \boldsymbol{\theta}_{\text{hi}})
$$
$$
\boldsymbol{\eta}_{\text{hi}} = \alpha_{\text{hi}} + \mathbf{X}_{\text{hi}} \boldsymbol{\beta}_{\text{hi}} + \sum_{k} f_k(\mathbf{s}, t) + \sum_{m=1}^M \rho_m \cdot \boldsymbol{\eta}_{\text{sub}, m}[\mathcal{M}_m]
$$

**Auxiliary / Sub-Model Process**:
Each sub-model $m$ is estimated jointly as an independent Bayesian structural process:
$$
\mathbf{y}_{\text{lo}, m} \sim \operatorname{Likelihood}(\boldsymbol{\eta}_{\text{sub}, m}, \boldsymbol{\theta}_{\text{sub}, m})
$$
$$
\boldsymbol{\eta}_{\text{sub}, m} = \alpha_{\text{sub}, m} + \mathbf{X}_{\text{lo}, m} \boldsymbol{\beta}_{\text{sub}, m} + \sum_{j} g_{m, j}(\mathbf{s}, t)
$$

**Coupling Weight ($\rho_m$)**:
$$
\rho_m \sim \mathcal{P}_{\text{coupling}} \quad (\text{default: } \operatorname{Normal}(1.0, 0.5))
$$
Optionally, coupling can be fixed ($\rho_m \equiv 1.0$) via `fixed = true`.

**Observation Alignment & Mapping ($\mathcal{M}_m$)**:
When the primary and proxy datasets differ in observation count ($N_{\text{hi}} \ne N_{\text{sub}, m}$), an explicit index mapping vector $\mathcal{M}_m \in \{1, \dots, N_{\text{sub}, m}\}^{N_{\text{hi}}}$ maps each primary observation $i \in \{1, \dots, N_{\text{hi}}\}$ to its corresponding proxy location/cell $\mathcal{M}_m[i]$. Bounds ($1 \le \mathcal{M}_m[i] \le N_{\text{sub}, m}$) and dimensions ($\operatorname{length}(\mathcal{M}_m) = N_{\text{hi}}$) are strictly validated.

#### 5.7.2. Declarative Paired Multi-Equation Syntax

The preferred, modern syntax for defining multi-fidelity systems uses declarative pairs. This avoids escaped nested strings and permits native Julia formula expressions.

##### Using the `@tbm` Macro
```julia
model = @tbm(
    :primary => (
        formula = likelihood(pm25_ground) ~ 1 + fixed(elevation) + transfer(:satellite),
        data    = df_ground
    ),
    :satellite => (
        formula = likelihood(aod_satellite) ~ 1 + fixed(humidity) + random(s_idx, model=bym2),
        data    = df_satellite,
        mapping = :satellite_cell_id,   # Column in df_ground or index vector
        prior   = Normal(1.0, 0.25)     # Custom coupling weight prior
    )
)

# Sample jointly using NUTS or optimal composite Gibbs sampler
chain = sample(model, NUTS(), 1000)
```

##### Using `TethysBayes_config` or `TethysBayes_core`
```julia
cfg = TethysBayes_config(
    :primary => (
        formula = "likelihood(y_hi) ~ 1 + fixed(x) + transfer(:proxy)",
        data    = data_hi
    ),
    :proxy => (
        formula = "likelihood(y_lo) ~ 1 + fixed(x)",
        data    = data_lo,
        mapping = map_indices,          # Vector{Int} of length N_hi
        prior   = Normal(0.0, 1.0),
        fixed   = false
    )
)
```

#### 5.7.3. Classic Formula Supervisor Syntax (`nested()`)

For single-formula workflows, the `nested()` module remains fully supported:
```julia
model = @tbm(
    likelihood(y_hi) ~ 1 + fixed(x) +
        nested(proxy, formula="likelihood(y_lo) ~ 1 + fixed(x)",
               data_source=:proxy_data, mapping=:map_idx, prior=Normal(1.0, 0.5)),
    data_hi,
    proxy_data = data_lo
)
```

#### 5.7.4. Parameter Registry & Namespace Isolation

Sub-model parameters are automatically isolated and prefixed in the `ParamRegistry`:
- Main model: `intercept`, `beta`, `y_sigma`.
- Coupling weight: `rho_nested_<key>` (assigned `role = :nested_weight`, shape `(1,)`).
- Sub-model: `intercept_<key>`, `beta_<key>`, `y_sigma_<key>`, etc.

Parameter queries by role:
```julia
reg = build_param_registry(cfg)
weights = get_descriptors_by_role(reg, :nested_weight)  # Returns ParamDescriptor for rho_nested_*
```

---

## 6. Sampling, Inference & Sampler Optimization

### 6.1. `get_optimal_sampler` and `precompute_step_sizes`

```julia
# Pre-inspect dimension-scaled initial step sizes, tree depths, and principled adaptation steps
proposal_info = precompute_step_sizes(model_obj; min_ϵ=1e-4, max_ϵ=1.0, max_depth=10, adaptation_steps=:auto)

# Build optimal composite Gibbs sampler with pre-conditioned proposal step sizes
os = get_optimal_sampler(
    model_obj;
    sampler_choice = :auto,
    adaptation_steps = :auto,            # Scaled by metric dimension O(sqrt(D)) and condition number
    init_ϵ = :auto,                      # Dimensional scaling ϵ ~ 0.5 * D^(-1/4)
    # init_ϵ = 0.05,                     # Or fixed scalar step size
    # init_ϵ = Dict(:s_idx => 0.03, :year => 0.08), # Or per-component step size
    max_depth = 10,                      # Capped binary tree depth (e.g. 6-10)
    min_ϵ = 1e-4,                        # Lower bound to avoid step collapse
    max_ϵ = 1.0,                         # Upper bound to prevent divergence
    adtype = AutoForwardDiff(),
    n_chains = 3
)
```

`get_optimal_sampler` parses `model_obj` using `ParamRegistry` and builds a block-partitioned composite Gibbs sampler tailored to parameter supports:

1. **Step-Size ($\epsilon$) Pre-Conditioning & Bounds**: Uses optimal Roberts & Rosenthal (2001) dimensional curvature scaling $\epsilon \approx 0.5 \cdot D^{-1/4}$ clamped within `[min_ϵ, max_ϵ]`. This completely bypasses the unstable `find_good_stepsize` search in high-dimensional GMRF/spatial blocks.
2. **Maximum Tree Depth Limiting (`max_depth`)**: Caps the maximum leapfrog trajectory doubling depth to prevent long tree stalls during warmup.
3. **Discrete Variables** ($y \in \mathbb{Z}$): Assigned `PG` (Particle Gibbs).
4. **Gaussian Latent Vectors** ($\mathcal{N}(0, \Sigma)$): Assigned `ESS` (Elliptical Slice Sampling) for gradient-free sampling of high-dimensional latent fields.
5. **Bounded Scalar Parameters** ($\sigma > 0, \rho \in [0, 1]$): Assigned `Slice` sampling or unconstrained `NUTS`.
6. **Unbounded Differentiable Blocks**: Assigned `NUTS(adaptation_steps, target_acceptance; max_depth=max_depth, init_ϵ=init_ϵ, adtype=adtype)`.

---

### 6.2. Sampler Interface Options

```julia
# MCMC Sampling
chn = sample(m, NUTS(), 1000; progress=false)
chn = sample(m, os, 1000; progress=false)
chn = TethysBayes_sample(m, 1000; n_chains=3, progress=false)

# Optimization Inference
map_res = optimize(m, MAP())
mle_res = optimize(m, MLE())
vi_res = vi(m, ADVI(10, 1000))
```

### 6.3. Parameter Registry & Sample Extraction

The `ParamRegistry` tracks parameter roles, dimensions, and priors across the model hierarchy:

```julia
# Inspect or query descriptors by semantic role
reg = build_param_registry(model)
spatial_params = get_descriptors_by_role(reg, :spatial)
nested_weights = get_descriptors_by_role(reg, :nested_weight)

# Direct posterior sample extraction
rho_samples = get_samples(chain, :rho_nested_proxy)  # 1D Vector for scalar
beta_samples = get_samples(chain, :beta)              # 2D Matrix (n_samples, p)
```

---

## 7. Posterior Reconstruction, Prediction & Diagnostics Engine

### 7.1. `model_results_comprehensive`

```julia
res = model_results_comprehensive(model, chain; data=nothing, alpha=0.05)
```

Processes fitted models and MCMC chains into a clean, structured analytical data object (pure data, without graphics rendering):

| Return Field | Type | Description |
| :--- | :--- | :--- |
| `metrics` | `NamedTuple` | Model evaluation & convergence: `rmse`, `r_pearson`, `waic`, `rhat`, `ess`, `time`. |
| `parameters` | `DataFrame` | Parameter-level posterior summary table: `parameters`, `mean`, `std`, `median`, `lower_95`, `upper_95`, `rhat`, `ess`. |
| `effects` | `NamedTuple` | Latent component effect summaries (spatial, temporal, fixed, random, interaction). |
| `predictions` | `NamedTuple` | Observation-level fitted values: `denoised` (`mean`, `median`, `std`, `lower`, `upper`) and `noisy`. |
| `draws` | `NamedTuple` | Raw posterior sample matrices: `predictions_denoised`, `predictions_noisy`, `weights`, `log_likelihood`. |
| `transfer_results` | `Dict{Symbol, Any}` | Reconstructed post-processing results for nested transfer sub-models (or `nothing`). |
| `arch` | `ModelArchitecture` | Model architecture type (`UnivariateArchitecture`, `MultivariateArchitecture`, `MultifidelityArchitecture`). |

### 7.2. Direct Reconstruction (`reconstruct`)

The `reconstruct` function provides lightweight, low-overhead posterior latent field synthesis, point predictions, and credible intervals from an MCMC chain and model configuration (or DynamicPPL model):

```julia
# From model or config NamedTuple
res = reconstruct(chain, cfg; alpha=0.05)
res = reconstruct(model, chain; alpha=0.05)
```

| Return Field | Type | Description |
| :--- | :--- | :--- |
| `predictions_denoised` | `NamedTuple` | Expected value summaries ($g^{-1}(\eta)$): `mean`, `median`, `std`, `lower`, `upper`. |
| `predictions_noisy` | `NamedTuple` | Observation-level predictions incorporating residual dispersion. |
| `raw_predictions_denoised` | `Array` | Raw Monte Carlo draws for expected values. |
| `raw_predictions_noisy` | `Array` | Raw Monte Carlo draws for noisy predictions. |
| `log_likelihood` | `Matrix` | Pointwise observation-level log-likelihood evaluation matrix. |
| `waic` | `Real` / `NamedTuple` | Widely Applicable Information Criterion. |
| `effects` | `NamedTuple` | Summarized posterior effects across all registered model components. |
| `transfer_results` | `Dict{Symbol, Any}` | Reconstructed outputs for each nested/transfer sub-model. |

### 7.3. Diagnostic & Component Effect Plotting (`TethysBayes_plots`)

```julia
plots_res = TethysBayes_plots(res; data=inp_df, au=data_scot.au, save_dir=nothing, save_prefix="", fmt="png", dpi=150)
```

Generates the complete suite of diagnostic plots and underlying tidy datasets from `res`:

| Return Field | Type | Description |
| :--- | :--- | :--- |
| `plots` | `NamedTuple` | Rendered `Plots.Plot` objects (`:posterior_predictive_check`, `:spatial`, `:spatial_s_idx`, `:spatial_observed`, `:spatial_fitted`, `:spatial_residuals`, `:temporal`, `:spacetime_predictions`, `:fixed_effects`, `:conditional_effects`, `:st_interaction_heatmap`, `:svc_*`, `:mixed_effects`). |
| `plots_data` | `NamedTuple` | Underlying tidy dataframes used to construct all diagnostic plots. |

If `save_dir !== nothing`, all generated plots are automatically saved to disk.

### 7.4. Out-of-Sample Prediction (`predict`)

```julia
preds = predict(model, chain, new_dataframe; alpha=0.05)
```

Projects fitted posterior fields onto a new spatial, temporal, or covariate data grid, automatically re-computing basis matrices for smooth spline terms and Gaussian Process cross-covariances.

### 7.5. Model Inspection (`show_model`)

```julia
show_model(model)
```

Prints formatted diagnostic information about the model, including the parsed formula, data schema, active components, generated Turing model code, and parameter registry mappings.

### 7.6. Benchmark Datasets (`TethysBayes_data`)

```julia
dataset = TethysBayes_data("scottish_lip")
```

Loads built-in benchmark spatiotemporal datasets (e.g. Scottish Lip Cancer disease mapping dataset with spatial polygons and adjacency matrix $W$).

---

## 8. Plotting Subsystem API (`src/plotting.jl`)

All visualization methods return standard `Plots.Plot` objects and accept customizable themes (`create_theme`).

### 8.1. Function Reference

| Function | Signature | Description |
| :--- | :--- | :--- |
| `choropleth` | `choropleth(polygons, values; cmap=:viridis, center_zero=false, title="")` | Shaded spatial polygon map with automatic percentile scaling and colorbars. Accepts `(polys, vals)` or `(vals, polys)`. |
| `spatial_graph_plot` | `spatial_graph_plot(centroids, g; polygons=nothing, au=nothing, title="")` | Visualizes spatial partitioning polygons, centroid nodes, graph adjacency edges, and boundary hulls. |
| `plot_kde_simple` | `plot_kde_simple(coords; grid_res=100)` or `plot_kde_simple(df; x=:s_x, y=:s_y)` | 2D Kernel Density Estimation surface heatmap with observation point scatter overlays. |
| `timeseries_ci` | `timeseries_ci(x, mean, lower, upper; ribbon_alpha=0.2, title="")` | Credible interval ribbon timeseries plot for temporal trends and seasonal curves. |
| `render_paths!` | `render_paths!(p, paths; color=:crimson, alpha=0.7)` | Overlays agent/individual movement trajectories onto an existing spatial plot. |
| `map_point_occupancy` | `map_point_occupancy(polygons, centroids, current_unit, previous_unit)` | Highlights occupied spatial units on a background map for discrete tracking. |
| `TethysBayes_plots` | `TethysBayes_plots(model, chain, res, M; au=nothing, data=nothing, outcome=1)` | Comprehensive automated diagnostic plotting suite (PPC, fixed effects, spatial, temporal, SVC, etc.). |
| `model_results_plots` | `model_results_plots(res)` | Iterates and displays all diagnostic plots contained in a `model_results` struct. |
| `save_plot` | `save_plot(p, path; fmt=nothing, dpi=150)` | Exports plot to disk, automatically creating parent directories if needed. |
| `create_theme` | `create_theme(; fontsize=10, fontfamily="sans-serif", size=(900, 600))` | Generates standardized styling dictionaries for publication-ready figures. |

---

## 9. Spatial & Spatiotemporal Partitioning API (`src/partitioning.jl`)

Comprehensive spatial discretization, graph extraction, and spatiotemporal index synchronization.

### 9.1. Function Reference

| Function | Signature | Description |
| :--- | :--- | :--- |
| `assign_spatial_units` | `assign_spatial_units(s_x, s_y; area_method=:avt, target_units=10, exact_units=false, target_area=nothing, min_area=0.0, max_area=Inf, min_points=1, max_points=nothing, length_scale=nothing, radius=nothing, grid_resolution=nothing, aspect_ratio=1.0, prune_empty=false, merge_small_polygons=false, input_polygons=nothing, geom_hull=nothing, kwargs...)` | Primary spatial partitioner. Discretizes 2D coordinates into polygons and builds neighborhood graph $W$. |
| `assign_spatial_units_inferred` | `assign_spatial_units_inferred(W; iterations=50, learning_rate=0.1, buffer_dist=0.5)` | Reconstructs coordinates and Voronoi polygons from an adjacency matrix via force-directed spring layout. |
| `assign_time_units` | `assign_time_units(t_v; time_method="quantile_regular", t_N=10)` | Discretizes continuous or discrete time vectors into categorical temporal indices. |
| `assign_spatiotemporal_units` | `assign_spatiotemporal_units(df; space_x=:s_x, space_y=:s_y, time_var=:t_idx, area_method=:avt, target_units=10, ...)` | Synchronizes space and time partitions into aligned `(s_idx, t_idx, st_idx)` indices and metadata $(S, T, ST)$. |
| `discretize_data` | `discretize_data(X; method="quantile", N_cat=9, brks=nothing, probs=nothing, dx=nothing, minv=nothing, maxv=nothing, quantile_bounds=[0.025, 0.975])` | General 1D variable discretization: `"quantile"`, `"regular"`, `"quantile_regular"`, `"kmeans"`, `"jenks"`, `"provided"`. |
| `spatial_weights_matrix` | `spatial_weights_matrix(W; style=:binary)` | Transforms adjacency $W$ into `:binary`, `:row_standardized` ($D^{-1}W$), or `:variance_stabilized` ($D^{-1/2}W$). |
| `spatial_knn_graph` | `spatial_knn_graph(coords, k)` | Constructs a $k$-nearest neighbors spatial graph and binary adjacency matrix. |
| `spatial_radius_graph` | `spatial_radius_graph(coords, radius)` | Constructs a distance-threshold spatial graph connecting points within distance `radius`. |
| `spatial_block_cv` | `spatial_block_cv(s_x, s_y; n_folds=5, method=:kmeans)` | Generates spatial cross-validation fold IDs using `:kmeans` clustering or `:grid` spatial blocking. |
| `scaling_factor_bym2` | `scaling_factor_bym2(W)` | Computes the BYM2 ICAR precision matrix scaling factor: $s = \exp(-\frac{1}{S-1}\sum \log \lambda_i)$. |
| `ensure_connected!` | `ensure_connected!(g, centroids)` | Connects disjoint spatial graph components by adding minimal bridging edges between closest centroid pairs. |
| `libgeos_lattice_adjacency_matrix` | `libgeos_lattice_adjacency_matrix(rows, cols; contiguity=:queen)` | Fast $\mathcal{O}(N)$ analytical lattice adjacency matrix constructor supporting `:queen` and `:rook` contiguity. |

---

## 10. Cross-Validation Engine (`TethysBayes_cv_orchestrator`)

The `TethysBayes_cv_orchestrator` orchestrates rigorous cross-validation across spatiotemporal datasets:

```julia
cv_res = TethysBayes_cv_orchestrator(
    formula,
    data;
    method = :spatial_block,
    cv_var = :s_idx,
    n_folds = 5,
    sampler = NUTS(500, 0.65),
    n_samples = 500,
    cv_space_vars = [:s_x, :s_y],
    kwargs...
)
```

### Supported CV Methods:
- **`:kfold`**: Standard random $k$-fold cross-validation.
- **`:lolo` (Leave-One-Location-Out)**: Holds out all observations from a specific areal unit or location.
- **`:spatial_block`**: $k$-means spatial block cross-validation to assess geographic generalization without spatial autocorrelation leakage.
- **`:temporal_block`**: Contiguous temporal block holdout for time-series interpolation.
- **`:temporal_forward_chain`**: Rolling-origin forward forecast simulation testing out-of-sample temporal prediction.

---

## 11. Input / Output & Persistence API (`src/input_output.jl`)

The `TethysBayes` framework provides a high-performance, non-redundant serialization and analytical storage subsystem:
- **JLD2 (`.jld2`)**: Full binary serialization of live Turing model states (`m`), formula configurations, data, and posterior MCMC chains (`chn`).
- **DuckDB (`.duckdb`)**: Embedded relational SQL storage for post-processing results (`res`), diagnostic metrics, and posterior samples without memory redundancy.

### 11.1. Function Reference

| Function                               | Signature                                                                                                                  | Description                                                                                                                                                        |
| :---------------------------------------| :---------------------------------------------------------------------------------------------------------------------------| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `save_TethysBayes_model`                      | `save_TethysBayes_model(path, model; chain=nothing, au=nothing, metadata=Dict(), compress=true)`                                  | Serializes the complete Turing `@model` state, configuration `M`, data, and optional MCMC chain to JLD2.                                                           |
| `load_TethysBayes_model`                      | `load_TethysBayes_model(path; calling_module=Main)`                                                                               | Reads JLD2 file and re-instantiates a live, callable `DynamicPPL.Model` ready for sampling or prediction.                                                          |
| `save_TethysBayes_results`                    | `save_TethysBayes_results(duckdb_path, res; model=nothing, chain=nothing, au=nothing, table_prefix="", overwrite=true)`           | Normalizes `model_results_comprehensive` output into relational DuckDB tables (`metrics`, `parameter_stats`, `predictions`, `spatial_geometries`, `plots_data_*`). |
| `load_TethysBayes_results`                    | `load_TethysBayes_results(duckdb_path; table_prefix="")`                                                                          | Reconstructs the `model_results_comprehensive` NamedTuple from DuckDB tables.                                                                                      |
| `query_duckdb`                         | `query_duckdb(duckdb_path, sql_query)`                                                                                     | Executes analytical SQL queries directly against a TethysBayes DuckDB database, returning a DataFrame.                                                                    |
| `export_posterior_samples_to_duckdb`   | `export_posterior_samples_to_duckdb(duckdb_path, chain, model=nothing; table_name="TethysBayes_posterior_samples", format=:tidy)` | Exports MCMC posterior draws in `:tidy` long format `(iteration, chain, parameter, value)` or `:wide` format to DuckDB.                                            |
| `import_posterior_samples_from_duckdb` | `import_posterior_samples_from_duckdb(duckdb_path; table_name="TethysBayes_posterior_samples")`                                   | Reads stored posterior draws from DuckDB into a Julia DataFrame.                                                                                                   |
| `append_posterior_samples`             | `append_posterior_samples(chain1, chain2)`                                                                                 | Concatenates two MCMC chains across sampling iterations for chain extension.                                                                                       |
| `extend_sampling`                      | `extend_sampling(model, prev_chain, n_additional_samples; sampler=NUTS(), kwargs...)`                                      | Warms up from previous chain state, samples additional iterations, and returns merged chains.                                                                      |
| `save_TethysBayes_bundle`                     | `save_TethysBayes_bundle(base_path, model, chain, res; au=nothing, metadata=Dict())`                                              | Unified one-line persistence creating `<base_path>.jld2` (model & chain) and `<base_path>.duckdb` (results).                                                       |
| `load_TethysBayes_bundle`                     | `load_TethysBayes_bundle(base_path; calling_module=Main)`                                                                         | Unified one-line loader recovering `(model=m, chain=chn, results=res, au=au, metadata=meta)`.                                                                      |
| `export_spatial_results_to_geojson`    | `export_spatial_results_to_geojson(geojson_path, res, au; property_keys=nothing)`                                          | Serializes spatial model results and polygon boundaries to standard RFC 7946 GeoJSON.                                                                              |
| `extract_posterior_priors`             | `extract_posterior_priors(source; parameter_names=nothing, prior_family=:normal)`                                          | Extracts posterior parameters and builds fitted prior distributions for sequential Bayesian updating.                                                              |
| `save_model_ensemble`                  | `save_model_ensemble(duckdb_path, ensemble_dict; overwrite=true)`                                                          | Registers a multi-model ensemble in DuckDB and computes $\Delta \text{WAIC}$ and BMA weights.                                                                      |
| `bma_weighted_predictions`             | `bma_weighted_predictions(duckdb_path)`                                                                                    | Computes Bayesian Model Averaged predictions and total variance across all candidate models.                                                                       |
| `save_out_of_sample_predictions`       | `save_out_of_sample_predictions(duckdb_path, pred_df; table_name="out_of_sample_predictions")`                             | Stores out-of-sample prediction DataFrames into DuckDB.                                                                                                            |
| `export_results_to_csv`                | `export_results_to_csv(duckdb_path, table_name, output_csv_path)`                                                          | Exports DuckDB table to CSV.                                                                                                                                       |
| `compact_duckdb`                       | `compact_duckdb(duckdb_path)`                                                                                              | Executes `VACUUM; ANALYZE;` on DuckDB database to reclaim space and optimize query statistics.                                                                     |

---

## 12. Movement & ADR Telemetry API (`src/movement.jl`)

The movement subsystem implements biophysical Advection-Diffusion-Reaction (ADR) population dynamics and integrated mark-recapture telemetry modeling:

### 12.1. Function Reference

| Function | Signature | Description |
| :--- | :--- | :--- |
| `generate_ADR_simulation_bundle` | `generate_ADR_simulation_bundle(domain_size, n_units, n_years, n_marks; area_method=:cvt, rng=Random.GLOBAL_RNG)` | Simulates synthetic joint population density surveys and mark-recapture telemetry encounters on spatial units. |
| `simulate_correlated_density_vector` | `simulate_correlated_density_vector(habitat_prob, rho_target, log_mu, sigma_resid; rng=Random.GLOBAL_RNG)` | Simulates spatial density vectors correlated with underlying habitat suitability index. |
| `compute_velocity_field` | `compute_velocity_field(prob_vec, grid_dim, strength; mode=:exponential)` | Computes spatial advection velocity vectors $\mathbf{v} \propto \nabla \text{HSI}$. |
| `calculate_multistep_transition` | `calculate_multistep_transition(Gamma_base, steps)` | Calculates multi-step dispersal transition probabilities via matrix exponentiation $\mathbf{\Gamma}^k$. |
| `simulate_posterior_trajectories` | `simulate_posterior_trajectories(Gamma_base, start_units, n_steps, au_context; rho_persistence=0.0, rng=Random.GLOBAL_RNG)` | Simulates individual animal trajectories with directional persistence (Correlated Random Walk). |
| `simulate_mechanistic_trajectories` | `simulate_mechanistic_trajectories(Gamma_sequence, start_units, t_start, au_context; rho_persistence=0.0, n_years_sim=1, rng=Random.GLOBAL_RNG)` | Simulates individual movement through dynamic, time-varying transition kernels $\mathbf{\Gamma}_t$. |
| `construct_stochastic_transition_kernel` | `construct_stochastic_transition_kernel(W, hsi_vec; gamma=1.0, residence=0.0, advection=0.0, spatial=false)` | Constructs row-stochastic transition kernel $\mathbf{P} \in [0, 1]^{S \times S}$ (or $\operatorname{Vector}\{\operatorname{Matrix}\}$ for groups). `spatial=false` supports length-$G$ vectors for demographic groups; `spatial=true` supports length-$S$ vectors for spatially varying parameters across units. |
| `compute_directed_adjacency` | `compute_directed_adjacency(W, hsi_vec; gamma=1.0)` | Constructs directed adjacency matrix modulated by habitat gradients: $A_{ij} = W_{ij} \exp(\gamma (h_j - h_i))$. Supports scalar $\gamma$ or spatially varying vector $\gamma \in \mathbb{R}^S$. |
| `predict_path` | `predict_path(P, obs_units, obs_times; group=1)` | Reconstructs the most likely latent path between telemetry fixes using the Viterbi algorithm. Accepts single transition matrix $\mathbf{P}$ or vector of group transition kernels. |
| `predict_corridor` | `predict_corridor(P, start_unit, end_unit, n_steps; group=1)` | Computes the spatial movement corridor and expected occupancy probability field between endpoints over $K$ steps. Accepts single matrix $\mathbf{P}$ or vector of group kernels. |
| `compute_suitability_transition_kernel` | `compute_suitability_transition_kernel(suitability_vec, W; sensitivity=1.0, diffusion_weight=0.1)` | Generates spatial Markov transition kernels biased towards high habitat suitability. |
| `calculate_regional_connectivity` | `calculate_regional_connectivity(Gamma, strata_definition)` | Aggregates fine-scale unit transitions into macro-regional migration probability matrices. |
| `plot_ad_ratio_distribution` | `plot_ad_ratio_distribution(advection_field, diffusion_field)` | Generates diagnostic histogram of local Advection-to-Diffusion (Péclet) ratios. |
| `synthesize_adr_results` | `synthesize_adr_results(chain_or_res, sim_data, vel_vectors; au=sim_data.au)` | Comprehensive post-processing, parameter extraction, propagator reconstruction, and multi-panel visualization. |

---

## 13. Surface Derivatives, Pipeline Orchestrator & Multi-Mesh API

### 13.1. Surface Derivatives & Differential Geometry (`src/derivatives.jl`)

| Function | Signature | Description |
| :--- | :--- | :--- |
| `TethysBayes_surface_derivatives` | `TethysBayes_surface_derivatives(model_obj, chain, coords; metrics=[:slope, :curvature, :bpi], radii=[5.0, 15.0, 30.0], alpha=0.05, return_samples=false, n_samples=nothing)` | Computes exact continuous surface derivatives ($\nabla z, \nabla^2 z$, slopes, aspects, profile/planform curvatures, Bessel BPI) across `RFF`, `SpectralGP`, `WaveletGP`, `SPDE`, `PSpline`, and `TPS`. |
| `compute_topographic_metrics` | `compute_topographic_metrics(z_samples, zx_samples, zy_samples, zxx_samples, zyy_samples, zxy_samples, bpi_samples, radii; alpha=0.05)` | Transforms raw spatial partial derivative sample matrices into geomorphometric summary metrics and empirical credible intervals. |

### 13.2. Modular DAG Pipeline & Topological Resharding (`src/pipeline.jl`)

| Type / Function | Signature | Description |
| :--- | :--- | :--- |
| `PipelineTierSpec` | `PipelineTierSpec(name, formula, data; au=nothing, W=nothing, sampler=NUTS(50, 0.65), n_samples=100, derivatives=Symbol[], radii=[10.0, 25.0])` | Declarative specification for an individual model tier within a multi-scale DAG pipeline. |
| `TethysBayes_pipeline` | `TethysBayes_pipeline(tiers...; master_au=nothing, duckdb_path=":memory:", jld2_path=nothing, geojson_path=nothing, verbose=true)` | Orchestrates sequential multi-tier MCMC execution, dynamic spatial network resharding, DuckDB persistence, and canonical master harmonization. |
| `compute_network_transfer_matrix` | `compute_network_transfer_matrix(au_src, au_dest; method=:area_weighted, bandwidth=nothing)` | Constructs sparse geometric linear transfer matrix $\mathbf{P} \in \mathbb{R}^{N_{\text{dest}} \times N_{\text{src}}}$ via area-weighted intersection, Gaussian kernel, or inverse-distance weighting. |
| `reshard_spatial_field` | `reshard_spatial_field(source, au_src, au_dest; mode=:samples, method=:area_weighted, bandwidth=nothing, alpha=0.05)` | Reshards spatial latent fields across mismatched graphs in Full Monte Carlo sample matrix mode (`mode=:samples`) or moment mode (`mode=:moments`). |
| `summarize_sample_matrix` | `summarize_sample_matrix(samples; alpha=0.05)` | Computes column-wise and row-wise empirical mean, standard deviation, median, and $(1-\alpha)$ credible intervals from an $N \times S$ Monte Carlo sample matrix. |
| `expand_hull` | `expand_hull(s_x, s_y, buffer_dist)` | Computes the 2D convex hull of coordinates and expands it by `buffer_dist` using LibGEOS polygon buffering. |

---

## 14. Population Attributable Risk (PAR / PAF) Engine (`src/par.jl`)

The epidemiological risk attribution engine calculates relative risks, odds ratios, population attributable fractions, and counterfactual intervention scenarios from MCMC posterior chains:

### 14.1. Mathematical Formulations
- **Log-Linear Relative Risk**: With canonical log link ($\eta = \log \mu$), $\text{RR} = \exp(\beta)$.
- **Logistic Models & Baseline Risk**: With logit link ($\eta = \operatorname{logit}(p)$), $\text{OR} = \exp(\beta)$. Conditional on baseline probability $p_0$:
  $$\text{RR} = \frac{\operatorname{logistic}(\operatorname{logit}(p_0) + \beta)}{p_0}$$
- **Levin's Formulation (1953)**: For population exposure prevalence $p_{\text{pop}}$:
  $$\text{PAF} = \frac{p_{\text{pop}} (\text{RR} - 1)}{1 + p_{\text{pop}} (\text{RR} - 1)}$$
- **Miettinen's Formulation (1974)**: For case exposure prevalence $p_{\text{cases}}$:
  $$\text{PAF} = p_{\text{cases}} \frac{\text{RR} - 1}{\text{RR}}$$
- **Prevented Fraction (PF)**: For protective exposures ($\text{RR} < 1$):
  $$\text{PF} = \frac{p_{\text{pop}} (1 - \text{RR})}{p_{\text{pop}} (1 - \text{RR}) + \text{RR}}$$
- **Counterfactual Model-Based PAF**: Full posterior simulation comparing observed vs counterfactual ($E_i = 0$):
  $$\text{PAF}^{(s)} = \frac{\sum_{i=1}^N \mu_i^{(s)}(\mathbf{X}_i) - \sum_{i=1}^N \mu_i^{(s)}(\mathbf{X}_i^*)}{\sum_{i=1}^N \mu_i^{(s)}(\mathbf{X}_i)}$$

### 14.2. Function Reference

| Function | Signature | Description |
| :--- | :--- | :--- |
| `par_from_posterior` | `par_from_posterior(chain, covariate; family="poisson", baseline_eta=nothing, baseline_risk=nothing, exposure_var=nothing, exposure_prevalence=nothing, threshold=nothing, data=nothing, outcome_var=nothing, reference_population="sample", method=:levin, alpha=0.05)` | Computes PAF, PAR, RR, Prevented Fraction, and Attributable Cases across posterior draws. |
| `summarize_par_effects` | `summarize_par_effects(chain; covariates=nothing, families="poisson", baseline_etas=nothing, baseline_risks=nothing, exposure_vars=nothing, exposure_prevalences=nothing, thresholds=nothing, data=nothing, outcome_var=nothing, reference_population="sample", method=:levin, alpha=0.05)` | Batch computes PAR/PAF summaries across multiple fixed effect covariates. |
| `par_counterfactual` | `par_counterfactual(model, chain; exposure_var, counterfactual_value=0.0, data=nothing, alpha=0.05)` | Computes model-based PAF and attributable cases via individual-level counterfactual simulation. |
| `export_par_to_table` | `export_par_to_table(par_results; include_raw_samples=false)` | Exports single or multiple PAR results into a tidy analytical `DataFrame`. |
| `par_credible_interval_plot` | `par_credible_interval_plot(par_result; title="PAF Estimate")` | Generates Plots.jl visual of PAF posterior mean and credible interval. |
| `par_forest_plot` | `par_forest_plot(par_results; title="Forest Plot: PAF by Risk Factor")` | Generates Plots.jl forest plot comparing PAF estimates across risk factors. |

---

## 15. References

1. **Besag, J.** (1974). Spatial interaction and the statistical analysis of lattice systems. *Journal of the Royal Statistical Society: Series B*, 36(2), 192–225.
2. **Besag, J., York, J., & Mollié, A.** (1991). Bayesian image restoration, with applications in spatial statistics. *Annals of the Institute of Statistical Mathematics*, 43(1), 1–59.
3. **Du, Q., Faber, V., & Gunzburger, M.** (1999). Centroidal Voronoi tessellations: Applications and algorithms. *SIAM Review*, 41(4), 637–676.
4. **Gelfand, A. E., et al.** (2003). Spatial modeling with spatially varying coefficient processes. *Journal of the American Statistical Association*, 98(462), 387–396.
5. **Greenland, S., & Drescher, K.** (1993). Maximum likelihood estimation of the attributable fraction from logistic models. *Biometrics*, 49(3), 865–872.
6. **Hooten, M. B., & Hefley, T. J.** (2019). *Bringing Bayesian Models to Life*. CRC Press.
7. **Jenks, G. F.** (1967). The data model concept in statistical mapping. *International Yearbook of Cartography*, 7, 186–190.
8. **Knorr-Held, L.** (2000). Bayesian modelling of inseparable space-time variation in disease risk. *Statistical Methods in Medical Research*, 9(3), 205–220.
9. **Leroux, B. G., Lei, X., & Breslow, N.** (2000). Estimation of disease rates in small areas: A new mixed model for spatial dependence. In *Statistical Models in Epidemiology, the Environment, and Clinical Trials* (pp. 179–191). Springer.
10. **Levin, M. L.** (1953). The occurrence of lung cancer in man. *Acta Unio Int. Cancrum*, 9(3), 531–541.
11. **Lindgren, F., Rue, H., & Lindström, J.** (2011). An explicit link between Gaussian fields and Gaussian Markov random fields: The SPDE approach. *Journal of the Royal Statistical Society: Series B*, 73(4), 423–498.
12. **Lloyd, S.** (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129–137.
13. **Miettinen, O. S.** (1974). Proportion of disease caused or prevented by a given exposure, trait or intervention. *American Journal of Epidemiology*, 99(5), 325–332.
14. **Okubo, A.** (1980). *Diffusion and Ecological Problems: Mathematical Models*. Springer-Verlag.
15. **Rasmussen, C. E., & Williams, C. K. I.** (2006). *Gaussian Processes for Machine Learning*. MIT Press.
16. **Riebler, A., Sørbye, S. H., Simpson, D., & Rue, H.** (2016). An intuitive Bayesian spatial model for disease mapping that accounts for scaling. *Statistical Methods in Medical Research*, 25(4), 1145–1165.
17. **Roberts, D. R., et al.** (2017). Cross-validation strategies for data with temporal, spatial, hierarchical or phylogenetic structure. *Ecography*, 40(8), 913–929.
18. **Rockhill, B., Newman, B., & Weinberg, C.** (1998). Use and misuse of population attributable fractions. *American Journal of Public Health*, 88(1), 15–19.
19. **Turchin, P.** (1998). *Quantitative Analysis of Movement: Measuring and Modeling Population Redistribution in Animals and Plants*. Sinauer Associates.
