---
title: "Custom Components & Non-Linear Spatial Epidemiological Modeling in TethysBayes"
format: html
---

# Custom Components & Non-Linear Spatial Epidemiological Modeling in TethysBayes

## 1. Introduction & Overview

While `TethysBayes` provides 45+ built-in statistical components (GMRFs, Gaussian processes, splines, autoregressive time series), complex real-world systems often require **domain-specific mechanistic processes**. In epidemiology, ecology, and environmental science, observations are frequently governed by coupled non-linear differential or difference equations, such as epidemic compartmental models (SIR, SEIR, SEIRS).

The `TethysBayes` framework provides two complementary avenues for implementing custom logic:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           Custom Modeling Pathways in TethysBayes                              │
├───────────────────────────────────────────┬─────────────────────────────────────────────┤
│  Pathway A: The custom() Formula Module   │  Pathway B: First-Class ComponentModel Type │
├───────────────────────────────────────────┼─────────────────────────────────────────────┤
│  - Quick prototyping & ad-hoc experiments │  - Reusable, production-grade extensions    │
│  - Raw Turing.jl code fragment injection  │  - Formal 4-method interface contract       │
│  - Optional custom reconstruct_func       │  - Full ParamRegistry & type safety         │
│  - Embedded directly inside @tbm(...)    │  - Registered in COMPONENT_TYPE_REGISTRY    │
└───────────────────────────────────────────┴─────────────────────────────────────────────┘
```

This guide details both approaches using a comprehensive real-world example: **Spatial SEIR (Susceptible-Exposed-Infectious-Recovered) Disease Dynamics on a Spatial Graph**.

---

## 2. Mathematical Formulation: Spatial SEIR Disease Dynamics

Consider an infectious disease propagating across $S$ interconnected spatial regions over $T$ discrete time steps ($t \in 1, \dots, T$).

```
           ┌──────────────┐   λ(s, t)    ┌──────────────┐    σ_lat    ┌──────────────┐    γ_rec    ┌──────────────┐
           │ Susceptible  │ ───────────> │   Exposed    │ ──────────> │  Infectious  │ ──────────> │  Recovered   │
           │   S(s, t)    │              │   E(s, t)    │             │   I(s, t)    │             │   R(s, t)    │
           └──────────────┘              └──────────────┘             └──────────────┘             └──────────────┘
                                                                             │
                                                                             │ ρ_rep (reporting)
                                                                             ▼
                                                                      ┌──────────────┐
                                                                      │ Observed y   │
                                                                      │ (Cases / Wk) │
                                                                      └──────────────┘
```

### 2.1. State Variables & Spatial Force of Infection

For each region $s \in \{1, \dots, S\}$ with population $N_s$:
- $S_{s, t}$: Susceptible individuals
- $E_{s, t}$: Exposed individuals (infected but not yet infectious, incubation period)
- $I_{s, t}$: Infectious individuals (transmitting disease)
- $R_{s, t}$: Recovered / immune individuals
- Population conservation: $N_s = S_{s, t} + E_{s, t} + I_{s, t} + R_{s, t}$

The **spatial force of infection** $\lambda_{s, t-1}$ acting upon susceptible individuals in region $s$ combines within-region contact transmission and geographic spillover from adjacent regions via the row-standardized spatial weights matrix $W_{\text{std}}$:

$$
\lambda_{s, t-1} = \beta_{\text{local}} \cdot \frac{I_{s, t-1}}{N_s} + \beta_{\text{spatial}} \sum_{j=1}^S W_{\text{std}, sj} \cdot \frac{I_{j, t-1}}{N_j}
$$

where:
- $\beta_{\text{local}} > 0$: Transmission rate within the host region.
- $\beta_{\text{spatial}} \ge 0$: Cross-border spatial transmission rate driven by inter-district mobility.
- $W_{\text{std}, sj} = W_{sj} / \sum_k W_{sk}$: Row-standardized neighborhood connectivity.

### 2.2. Discrete-Time State Transition Dynamics

Over discrete time interval $\Delta t = 1$:

1. **New Infections ($S \to E$)**:

   $$
   \Delta N_{S \to E, s, t} \approx \lambda_{s, t-1} \cdot S_{s, t-1}
   $$

2. **Latent Progression ($E \to I$)** with incubation rate $\sigma_{\text{lat}} = 1 / \tau_{\text{inc}}$:

   $$
   \Delta N_{E \to I, s, t} \approx \sigma_{\text{lat}} \cdot E_{s, t-1}
   $$

3. **Recovery ($I \to R$)** with recovery rate $\gamma_{\text{rec}} = 1 / \tau_{\text{inf}}$:

   $$
   \Delta N_{I \to R, s, t} \approx \gamma_{\text{rec}} \cdot I_{s, t-1}
   $$

#### Recursive State Equations:

$$
S_{s, t} = S_{s, t-1} - \Delta N_{S \to E, s, t}
$$

$$
E_{s, t} = E_{s, t-1} + \Delta N_{S \to E, s, t} - \Delta N_{E \to I, s, t}
$$

$$
I_{s, t} = I_{s, t-1} + \Delta N_{E \to I, s, t} - \Delta N_{I \to R, s, t}
$$

$$
R_{s, t} = R_{s, t-1} + \Delta N_{I \to R, s, t}
$$

### 2.3. Observation Likelihood

Public health surveillance systems observe confirmed cases $y_{s, t}$, which represent newly symptomatic infectious individuals subject to reporting coverage $\rho_{\text{rep}} \in (0, 1)$ and overdispersion:

$$
\mathbb{E}[y_{s, t}] = \rho_{\text{rep}} \cdot \Delta N_{E \to I, s, t}
$$

In the linear predictor link scale:

$$
\eta_{s, t} = \log\left( \max\left(\rho_{\text{rep}} \cdot \Delta N_{E \to I, s, t}, 10^{-6}\right) \right) + \phi_s
$$

where $\phi_s \sim \operatorname{BYM2}(W)$ accounts for unmodeled residual spatial clustering.

---

## 3. Pathway A: Rapid Prototyping with the `custom()` Formula Module

The `custom()` module allows direct injection of raw Turing.jl code fragments into the model.

### 3.1. Writing the Turing Code Fragment

```julia
const seir_turing_code = """
    # 1. Priors for SEIR epidemiological parameters
    beta_local ~ truncated(Normal(0.4, 0.1), 0.0, 2.0)
    beta_spatial ~ truncated(Normal(0.1, 0.05), 0.0, 1.0)
    sigma_lat ~ truncated(Normal(0.2, 0.05), 0.05, 1.0) # 5-day incubation
    gamma_rec ~ truncated(Normal(0.14, 0.03), 0.05, 1.0) # 7-day infectious period
    rho_rep ~ Beta(2.0, 5.0) # ~28% reporting rate

    # 2. Dimensions & Spatial Weight Matrix from model precomputes
    S_units = M.s_N
    T_steps = M.t_N
    W_std = M.technical[:W_std]
    pop = M.technical[:population]

    # 3. Initialize SEIR Compartment State Matrices (S x T)
    S_mat = zeros(typeof(beta_local), S_units, T_steps)
    E_mat = zeros(typeof(beta_local), S_units, T_steps)
    I_mat = zeros(typeof(beta_local), S_units, T_steps)
    R_mat = zeros(typeof(beta_local), S_units, T_steps)
    new_cases = zeros(typeof(beta_local), S_units, T_steps)

    # Initial condition at t = 1
    for s in 1:S_units
        I_mat[s, 1] = max(Float64(M.technical[:init_infected][s]), 1.0)
        E_mat[s, 1] = I_mat[s, 1] * 1.5
        R_mat[s, 1] = 0.0
        S_mat[s, 1] = pop[s] - E_mat[s, 1] - I_mat[s, 1]
        new_cases[s, 1] = max(I_mat[s, 1] * sigma_lat, 1e-4)
    end

    # 4. Forward Simulation of Spatial SEIR Dynamics across Time
    for t in 2:T_steps
        # Compute spatial infection pressure from neighbors
        spatial_pressure = W_std * (I_mat[:, t-1] ./ pop)
        
        for s in 1:S_units
            # Force of infection
            lambda_st = beta_local * (I_mat[s, t-1] / pop[s]) + beta_spatial * spatial_pressure[s]
            
            # Compartmental flows
            flow_SE = min(lambda_st * S_mat[s, t-1], S_mat[s, t-1])
            flow_EI = min(sigma_lat * E_mat[s, t-1], E_mat[s, t-1])
            flow_IR = min(gamma_rec * I_mat[s, t-1], I_mat[s, t-1])
            
            # State transitions
            S_mat[s, t] = max(S_mat[s, t-1] - flow_SE, 0.0)
            E_mat[s, t] = max(E_mat[s, t-1] + flow_SE - flow_EI, 0.0)
            I_mat[s, t] = max(I_mat[s, t-1] + flow_EI - flow_IR, 0.0)
            R_mat[s, t] = max(R_mat[s, t-1] + flow_IR, 0.0)
            
            # Expected new confirmed incident cases
            new_cases[s, t] = max(rho_rep * flow_EI, 1e-6)
        end
    end

    # 5. Map Mechanistic Incident Cases to Linear Predictor (eta)
    for i in 1:M.y_N
        s_i = M.s_idx[i]
        t_i = M.t_idx[i]
        eta[i] += log(new_cases[s_i, t_i])
    end
"""
```

### 3.2. Defining the Reconstruction Function

```julia
function seir_reconstruct(m::Custom, chain, spec::NamedTuple, M::NamedTuple, PS::Union{NamedTuple, Nothing})::NamedTuple
    # Extract posterior draws from chain
    p_names = names(chain, :parameters)
    n_samples = size(chain, 1) * size(chain, 3)
    
    # Extract parameter samples
    beta_l_samples = vec(Array(chain[:beta_local]))
    beta_s_samples = vec(Array(chain[:beta_spatial]))
    sigma_samples = vec(Array(chain[:sigma_lat]))
    gamma_samples = vec(Array(chain[:gamma_rec]))
    rho_samples = vec(Array(chain[:rho_rep]))
    
    S_units = M.s_N
    T_steps = M.t_N
    W_std = M.technical[:W_std]
    pop = M.technical[:population]
    init_inf = M.technical[:init_infected]

    # Reconstruct trajectories
    effect_matrix = zeros(Float64, M.y_N, n_samples)
    
    for smp in 1:n_samples
        bl = beta_l_samples[smp]
        bs = beta_s_samples[smp]
        sl = sigma_samples[smp]
        gr = gamma_samples[smp]
        rr = rho_samples[smp]
        
        # Simulate forward trajectory
        S_cur = Float64[pop[s] - init_inf[s] * 2.5 for s in 1:S_units]
        E_cur = Float64[init_inf[s] * 1.5 for s in 1:S_units]
        I_cur = Float64[init_inf[s] for s in 1:S_units]
        R_cur = zeros(Float64, S_units)
        
        nc_mat = zeros(Float64, S_units, T_steps)
        nc_mat[:, 1] .= max.(I_cur .* sl .* rr, 1e-4)

        for t in 2:T_steps
            sp_press = W_std * (I_cur ./ pop)
            for s in 1:S_units
                lam = bl * (I_cur[s] / pop[s]) + bs * sp_press[s]
                f_SE = min(lam * S_cur[s], S_cur[s])
                f_EI = min(sl * E_cur[s], E_cur[s])
                f_IR = min(gr * I_cur[s], I_cur[s])
                
                S_cur[s] = max(S_cur[s] - f_SE, 0.0)
                E_cur[s] = max(E_cur[s] + f_SE - f_EI, 0.0)
                I_cur[s] = max(I_cur[s] + f_EI - f_IR, 0.0)
                R_cur[s] += f_IR
                nc_mat[s, t] = max(rr * f_EI, 1e-6)
            end
        end

        for i in 1:M.y_N
            effect_matrix[i, smp] = log(nc_mat[M.s_idx[i], M.t_idx[i]])
        end
    end

    return (structured=effect_matrix, noisy=effect_matrix)
end
```

### 3.3. Fitting the Model with `@tbm`

```julia
# Technical precomputes passed via kwargs
W_std = W ./ sum(W, dims=2)
technical_dict = Dict(
    :W_std => W_std,
    :population => population_per_region,
    :init_infected => initial_cases_per_region
)

# Fit model with SEIR dynamics + BYM2 spatial random effect
m = @tbm(
    likelihood(cases, family=poisson) ~ 
        intercept() + 
        custom(code_fragment=seir_turing_code, params=Dict(:reconstruct_func => seir_reconstruct)) +
        random(s_idx, model=bym2),
    df,
    W = W,
    technical = technical_dict,
    verbose = false
)

# Sample posterior
chn = sample(m, NUTS(200, 0.65), 500; progress=false)
res = model_results_comprehensive(m, chn; au=st_data.au_spatial)
```

---

## 4. Pathway B: Building a First-Class `SpatialSEIR` Component

For production libraries and reusable pipelines, create a dedicated struct subtyping `ComponentModel`.

### 4.1. Defining the Component Struct

```julia
struct SpatialSEIR <: ComponentModel
    beta_local_prior::Distribution
    beta_spatial_prior::Distribution
    sigma_lat_prior::Distribution
    gamma_rec_prior::Distribution
    rho_rep_prior::Distribution
end

# Default constructor
function SpatialSEIR(;
    beta_local = truncated(Normal(0.4, 0.1), 0.0, 2.0),
    beta_spatial = truncated(Normal(0.1, 0.05), 0.0, 1.0),
    sigma_lat = truncated(Normal(0.2, 0.05), 0.05, 1.0),
    gamma_rec = truncated(Normal(0.14, 0.03), 0.05, 1.0),
    rho_rep = Beta(2.0, 5.0)
)
    return SpatialSEIR(beta_local, beta_spatial, sigma_lat, gamma_rec, rho_rep)
end

# Register in TethysBayes component registry
COMPONENT_TYPE_REGISTRY[:spatial_seir] = SpatialSEIR
COMPONENT_CONSTRUCTORS[:spatial_seir] = (p, params) -> SpatialSEIR(
    get(p, :beta_local, truncated(Normal(0.4, 0.1), 0.0, 2.0)),
    get(p, :beta_spatial, truncated(Normal(0.1, 0.05), 0.0, 1.0)),
    get(p, :sigma_lat, truncated(Normal(0.2, 0.05), 0.05, 1.0)),
    get(p, :gamma_rec, truncated(Normal(0.14, 0.03), 0.05, 1.0)),
    get(p, :rho_rep, Beta(2.0, 5.0))
)
```

### 4.2. Implementing the 4-Method Interface

```julia
# 1. get_precomputes: Validates inputs and prepares row-standardized adjacency W_std
function get_precomputes(m::SpatialSEIR, M::NamedTuple, mod_data::Dict)::NamedTuple
    W = M.W
    W_std = W ./ sum(W, dims=2)
    pop = get(M.technical, :population, ones(M.s_N) .* 10000.0)
    init_inf = get(M.technical, :init_infected, ones(M.s_N))
    return (W_std = W_std, population = pop, init_infected = init_inf)
end

# 2. get_priors: Declares Turing prior distributions
function get_priors(m::SpatialSEIR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    k = spec.key
    return """
    beta_local_$(k) ~ $(_distribution_to_string(m.beta_local_prior))
    beta_spatial_$(k) ~ $(_distribution_to_string(m.beta_spatial_prior))
    sigma_lat_$(k) ~ $(_distribution_to_string(m.sigma_lat_prior))
    gamma_rec_$(k) ~ $(_distribution_to_string(m.gamma_rec_prior))
    rho_rep_$(k) ~ $(_distribution_to_string(m.rho_rep_prior))
    """
end

# 3. get_updates: Simulates SEIR compartmental updates and modifies linear predictor eta
function get_updates(m::SpatialSEIR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    k = spec.key
    return """
    let
        S_units = M.s_N
        T_steps = M.t_N
        W_std = spec.hyper.W_std
        pop = spec.hyper.population
        init_inf = spec.hyper.init_infected

        bl = beta_local_$(k)
        bs = beta_spatial_$(k)
        sl = sigma_lat_$(k)
        gr = gamma_rec_$(k)
        rr = rho_rep_$(k)

        S_mat = zeros(typeof(bl), S_units, T_steps)
        E_mat = zeros(typeof(bl), S_units, T_steps)
        I_mat = zeros(typeof(bl), S_units, T_steps)
        new_c = zeros(typeof(bl), S_units, T_steps)

        for s in 1:S_units
            I_mat[s, 1] = max(Float64(init_inf[s]), 1.0)
            E_mat[s, 1] = I_mat[s, 1] * 1.5
            S_mat[s, 1] = pop[s] - E_mat[s, 1] - I_mat[s, 1]
            new_c[s, 1] = max(I_mat[s, 1] * sl, 1e-4)
        end

        for t in 2:T_steps
            sp_press = W_std * (I_mat[:, t-1] ./ pop)
            for s in 1:S_units
                lam = bl * (I_mat[s, t-1] / pop[s]) + bs * sp_press[s]
                f_SE = min(lam * S_mat[s, t-1], S_mat[s, t-1])
                f_EI = min(sl * E_mat[s, t-1], E_mat[s, t-1])
                f_IR = min(gr * I_mat[s, t-1], I_mat[s, t-1])

                S_mat[s, t] = max(S_mat[s, t-1] - f_SE, 0.0)
                E_mat[s, t] = max(E_mat[s, t-1] + f_SE - f_EI, 0.0)
                I_mat[s, t] = max(I_mat[s, t-1] + f_EI - f_IR, 0.0)
                new_c[s, t] = max(rr * f_EI, 1e-6)
            end
        end

        for i in 1:M.y_N
            eta[i] += log(new_c[M.s_idx[i], M.t_idx[i]])
        end
    end
    """
end

# 4. get_effects: Extracts posterior trajectories
function get_effects(m::SpatialSEIR, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    # Reconstruct effects using get_param_samples
    k = spec.key
    bl_s = get_param_samples(chain, M.param_registry, Symbol("beta_local_$(k)"))
    bs_s = get_param_samples(chain, M.param_registry, Symbol("beta_spatial_$(k)"))
    sl_s = get_param_samples(chain, M.param_registry, Symbol("sigma_lat_$(k)"))
    gr_s = get_param_samples(chain, M.param_registry, Symbol("gamma_rec_$(k)"))
    rr_s = get_param_samples(chain, M.param_registry, Symbol("rho_rep_$(k)"))

    effect_matrix = zeros(Float64, M.y_N, n_samples)
    # (Forward simulation loops over samples...)
    return (structured=effect_matrix, noisy=effect_matrix)
end
```

---

## 5. Complete Runnable Walkthrough: Spatial SEIR Outbreak Modeling

```julia
using TethysBayes, DataFrames, Random, Plots

# 1. Setup Synthetic Geography & Population
rng = MersenneTwister(42)
S = 16 # 16 spatial regions
T = 20 # 20 weekly time steps

# Generate 16 regular hexagonal regions
coords_x = rand(rng, 500) .* 100.0
coords_y = rand(rng, 500) .* 100.0
au = assign_spatial_units(coords_x, coords_y; area_method=:hexagonal, target_units=S, exact_units=true)
W = au.W

population = [10000.0 + rand(rng) * 5000.0 for _ in 1:S]
init_infected = zeros(Float64, S)
init_infected[1] = 15.0 # Patient zero cluster in region 1

# 2. Simulate Synthetic SEIR Epidemic Observations
true_beta_local = 0.45
true_beta_spatial = 0.12
true_sigma_lat = 0.20
true_gamma_rec = 0.14
true_rho_rep = 0.30

W_std = W ./ sum(W, dims=2)
I_sim = zeros(S, T); I_sim[:, 1] .= init_infected
E_sim = zeros(S, T); E_sim[:, 1] .= init_infected .* 1.5
S_sim = zeros(S, T); S_sim[:, 1] .= population .- E_sim[:, 1] .- I_sim[:, 1]
observed_cases = Int[]
s_vec = Int[]
t_vec = Int[]

for t in 1:T
    if t > 1
        sp_press = W_std * (I_sim[:, t-1] ./ population)
        for s in 1:S
            lam = true_beta_local * (I_sim[s, t-1] / population[s]) + true_beta_spatial * sp_press[s]
            f_SE = min(lam * S_sim[s, t-1], S_sim[s, t-1])
            f_EI = min(true_sigma_lat * E_sim[s, t-1], E_sim[s, t-1])
            f_IR = min(true_gamma_rec * I_sim[s, t-1], I_sim[s, t-1])
            
            S_sim[s, t] = max(S_sim[s, t-1] - f_SE, 0.0)
            E_sim[s, t] = max(E_sim[s, t-1] + f_SE - f_EI, 0.0)
            I_sim[s, t] = max(I_sim[s, t-1] + f_EI - f_IR, 0.0)
            
            exp_cases = max(true_rho_rep * f_EI, 0.01)
            y_obs = rand(rng, Poisson(exp_cases))
            push!(observed_cases, y_obs)
            push!(s_vec, s)
            push!(t_vec, t)
        end
    else
        for s in 1:S
            push!(observed_cases, rand(rng, Poisson(max(true_rho_rep * init_infected[s], 0.01))))
            push!(s_vec, s)
            push!(t_vec, 1)
        end
    end
end

df_outbreak = DataFrame(cases = observed_cases, s_idx = s_vec, time_idx = t_vec)

# 3. Fit Model Using Custom SEIR Module + BYM2 Spatial Noise
technical_data = Dict(
    :W_std => W_std,
    :population => population,
    :init_infected => init_infected
)

m = @tbm(
    likelihood(cases, family=poisson) ~ 
        intercept() + 
        custom(code_fragment=seir_turing_code, params=Dict(:reconstruct_func => seir_reconstruct)) + 
        random(s_idx, model=bym2),
    df_outbreak,
    W = W,
    technical = technical_data,
    verbose = false
)

# 4. Posterior Inference with NUTS
chn = sample(m, NUTS(100, 0.65), 300; progress=false)
res = model_results_comprehensive(m, chn; au=au)

# 5. Persist Unified Bundle to DuckDB and JLD2
save_TethysBayes_bundle("output/seir_spatial_outbreak", m, chn, res; au=au)

# 6. Query Epidemiological Parameter Posteriors via SQL
df_params = query_duckdb("output/seir_spatial_outbreak.duckdb", """
    SELECT parameters, mean, std, mcse, ess, rhat 
    FROM parameter_stats 
    WHERE parameters IN ('beta_local', 'beta_spatial', 'sigma_lat', 'gamma_rec', 'rho_rep')
""")
display(df_params)

# 7. Visualize Spatial Risk and Disease Spread
export_spatial_results_to_geojson("output/seir_spatial_risk.geojson", res, au)
p_map = choropleth(au.polygons, res.effects.s_idx.structured.mean; title="Posterior Spatial Residual Risk")
p_graph = spatial_graph_plot(au=au; title="Spatial Connectivity Graph")
plot(p_map, p_graph, layout=(1, 2), size=(1000, 450))
```

---

## 6. Summary of Best Practices for Custom Components

1. **Maintain Type Stability for AD Compatibility**:
   Always initialize temporary state matrices using `typeof(beta_local)` (e.g. `zeros(typeof(beta_local), S, T)`) rather than rigid `Float64` so ForwardDiff `Dual` numbers propagate through state transitions.
2. **Prevent Scope Leaks with `let` Blocks**:
   The `get_updates` generator wraps custom code in a `let ... end` block to prevent variable shadowing across multiple formula terms.
3. **Bound Flow Terms**:
   Ensure flow terms do not exceed compartment reserves (e.g. `flow_SE = min(lambda * S, S)` and `max(..., 0.0)`) to prevent negative population counts.
4. **Use `save_TethysBayes_bundle` for Archiving**:
   Guarantees that both the dynamic Turing simulation code in JLD2 and the relational time series predictions in DuckDB are persisted together.
