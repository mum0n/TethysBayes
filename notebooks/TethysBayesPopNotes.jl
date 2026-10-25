### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils

macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 10000000-0000-0000-0000-000000000001
begin
    using Pkg
    Pkg.activate(@__DIR__)
    using TethysBayes
    using Plots
    using CodeTracking
    default(size=(700,400))
    include(normpath(joinpath(@__DIR__, "..", "..", "TethysPop", "src", "delay_difference.jl")))
    "TethysBayes + TethysPop delay_difference loaded."
end

# ╔═╡ 10000000-0000-0000-0000-000000000002
md"""
# Deriso-Schnute delay-difference fishery model

This notebook demonstrates the delay-difference model using both `TethysPop.jl`
(functions included above) and an equivalent implementation built on `TethysBayes`/`Turing`.

> ⚠️ **WARNING: STORE OVERWRITE HAZARD**
>
> Executing MCMC sampling runs manually will trigger intensive estimation and
> overwrite cached model state and workspace store objects.
> By default, `run_model = false` to prevent inadvertent execution upon loading.
> Toggle `run_model = true` only when an explicit manual re-fit and store overwrite is intended.
"""

# ╔═╡ 10000000-0000-0000-0000-000000000020
run_model = false

# ╔═╡ 10000000-0000-0000-0000-000000000003
md"""
## 1. Forward simulation (`TethysPop`)
"""

# ╔═╡ 10000000-0000-0000-0000-000000000004
begin
    rng = MersenneTwister(2026)
    years = collect(1:25)
    catches = 20.0 .+ 5.0 .* rand(rng, 25)
    recruits = 30.0 .+ 10.0 .* rand(rng, 25)
    cfg = DelayDifferenceConfig(0.8, 12.0, 0.3, 2.0)
    res = simulate_delay_difference(years, catches, recruits, cfg; b0=80.0)
    plot(years, res.biomass, label="Biomass B_t", lw=2)
    plot!(years, res.numbers, label="Numbers N_t", lw=2)
    plot!(years, res.catches_biomass, label="Catch Y_t", lw=2)
    plot!(years, res.recruits_biomass, label="Recruitment wk R_t", lw=2)
    xlabel!("Year"); ylabel!("Biomass / numbers")
end

# ╔═╡ 10000000-0000-0000-0000-000000000005
md"""
## 2. Bayesian inference (`TethysPop`)
"""

# ╔═╡ 10000000-0000-0000-0000-000000000006
begin
    obs_survey = res.biomass .* (1.0 .+ 0.15 .* randn(rng, 25))
    model_at = bayesian_delay_difference_model(catches, recruits, obs_survey, cfg;
        obs_cv=0.20, prior_b0_mu=80.0, prior_b0_sd=15.0)
    "TethysPop model constructed (sampling disabled in notebook due to duplicate variable names in source)."
end

# ╔═╡ 10000000-0000-0000-0000-000000000007
md"""
## 3. Equivalent implementation on `TethysBayes`
"""

# ╔═╡ 10000000-0000-0000-0000-000000000008
begin
    model_TethysBayes = bayesian_delay_difference_model(catches, recruits, obs_survey, cfg;
        obs_cv=0.20, prior_b0_mu=80.0, prior_b0_sd=15.0)
    chain_TethysBayes = if run_model
        sample(model_TethysBayes, NUTS(0.65), 200; progress=false, check_model=false)
    else
        "Sampling skipped (run_model = false). Set run_model = true to fit model."
    end
    chain_TethysBayes
end

# ╔═╡ 10000000-0000-0000-0000-000000000009
md"""
## 4. Comparison
"""

# ╔═╡ 40000000-0000-0000-0000-000000000001
md"""
## 5. Hybrid: TethysPop model + TethysBayes sampling/posterior handling
"""

# ╔═╡ 40000000-0000-0000-0000-000000000002
begin
    model_hybrid = bayesian_delay_difference_model(catches, recruits, obs_survey, cfg;
        obs_cv=0.20, prior_b0_mu=80.0, prior_b0_sd=15.0)
    if run_model
        chain_hybrid = sample(model_hybrid, NUTS(0.65), 200; progress=false, check_model=false)
        (mean_s = mean(chain_hybrid[:s]), mean_b0 = mean(chain_hybrid[:b0]))
    else
        (status = "Sampling skipped (run_model = false).",)
    end
end

# ╔═╡ 50000000-0000-0000-0000-000000000001
md"""
## 6. Hybrid `@tbm` model
note: simple linear regression baseline (see section 8 for the dynamics-based version).
"""

# ╔═╡ 50000000-0000-0000-0000-000000000002
begin
    df_hybrid = DataFrame(years=years, catches=catches, y=obs_survey, s_idx=fill(1, length(years)))
    m_hybrid = @tbm(
        likelihood(y, family = gaussian) ~ intercept() + fixed(catches),
        df_hybrid;
        verbose = false
    )
    if run_model
        chain_hybrid_TethysBayes = sample(m_hybrid, NUTS(15, 0.65), 100; progress = false)
        (status = "sampled",)
    else
        (status = "Sampling skipped (run_model = false).",)
    end
end

# ╔═╡ 60000000-0000-0000-0000-000000000001
md"""
## 7. Custom() escape hatch: TethysPop model inside @tbm
"""

# ╔═╡ 60000000-0000-0000-0000-000000000002
md"""
The `extract_model_body()` helper (defined in `TethysBayes/src/model.jl`) strips the
`@model function ... end` wrapper from the `@code_string` output, returning only
the body as a code fragment. The fragment runs inside a `let` block, so `ny`,
`catches`, `recruits`, `obs` and `cfg` must be available in scope.
"""

# ╔═╡ 60000000-0000-0000-0000-000000000003
begin
    ny = length(years)
    obs = obs_survey

    # Extract the raw Turing code fragment from TethysPop' @model function.
    full_text = @code_string TethysPop.bayesian_delay_difference_model(
        catches, recruits, obs_survey, cfg
    )
    code_fragment = extract_model_body(full_text)

    model_custom = @tbm(
        likelihood(y, family = gaussian) ~ intercept() + custom(code_fragment),
        df_hybrid;
        params = Dict(:cfg => cfg),
        verbose = false
    )
    chain_custom = if run_model
        sample(model_custom, NUTS(0.65), 200; progress=false)
    else
        "Sampling skipped (run_model = false)."
    end
    chain_custom
end

# ╔═╡ 60000000-0000-0000-0000-000000000004
md"""
Notes:
- No `@model` stripping tool exists in TethysBayes; authoring the fragment directly is the
  recommended path.
- Parameter names inside the fragment must not collide with other components.
- Add `params = Dict(:cfg => cfg, :reconstruct_func => my_reconstruct_func)` if you
  need custom posterior reconstruction.
"""

# ╔═╡ 70000000-0000-0000-0000-000000000001
md"""
## 8. Biomass dynamics via `dynamics()` component in `@tbm`
"""

# ╔═╡ 70000000-0000-0000-0000-000000000002
md"""
The `dynamics()` component (docs §5.5) supports `model="logistic"` for simple
biomass dynamics: $N_{t+1} = N_t + r N_t (1 - N_t/K) - C_t$.
"""

# ╔═╡ 70000000-0000-0000-0000-000000000003
begin
    df_bio = DataFrame(
        years = years,
        y = obs_survey,
        s_idx = fill(1, length(years)),
        t_idx = years,
    )
    W_bio = Matrix(1.0I, 1, 1)
    m_bio = @tbm(
        likelihood(y, family = gaussian) ~ intercept() +
            dynamics(s_idx, t_idx, model = "logistic",
                     intrinsic_growth_rate = LogNormal(0.0, 1.0),
                     carrying_capacity = LogNormal(log(100.0), 1.0)),
        df_bio;
        W = W_bio,
        verbose = false
    )
    if run_model
        chain_bio = sample(m_bio, NUTS(0.65), 100; progress = false, check_model = false)
        (status = "sampled", keys = collect(keys(chain_bio)))
    else
        (status = "Sampling skipped (run_model = false).",)
    end
end

# ╔═╡ 70000000-0000-0000-0000-000000000004
md"""
## 9. Biomass dynamics via `TethysPop` (bayesian_logistic_discrete)
"""

# ╔═╡ 70000000-0000-0000-0000-000000000005
md"""
`TethysPop.bayesian_logistic_discrete` implements a Schaefer surplus production
model. We wrap it in `@tbm` via `custom()` using `@code_string`.
"""

# ╔═╡ 70000000-0000-0000-0000-000000000006
begin
    removals = catches
    survey_obs = obs_survey

    full_text_bd = @code_string TethysPop.bayesian_logistic_discrete(
        removals, survey_obs
    )
    code_fragment_bd = extract_model_body(full_text_bd)

    df_bio_ct = DataFrame(
        years = years,
        y = obs_survey,
        s_idx = fill(1, length(years)),
    )
    model_bio_ct = @tbm(
        likelihood(y, family = gaussian) ~ intercept() + custom(code_fragment_bd),
        df_bio_ct;
        params = Dict(:removals => removals, :survey_obs => survey_obs),
        verbose = false
    )
    if run_model
        chain_bio_ct = sample(model_bio_ct, NUTS(0.65), 100; progress = false, check_model = false)
        (status = "sampled",)
    else
        (status = "Sampling skipped (run_model = false).",)
    end
end

# ╔═╡ 70000000-0000-0000-0000-000000000007
md"""
## 10. Delay-difference via `dynamics()` component in `@tbm`
"""

# ╔═╡ 70000000-0000-0000-0000-000000000008
md"""
The `dynamics()` component (docs §5.5) supports `model="delay_difference"` directly
inside `@tbm`, using `s_idx`/`t_idx` and a spatial adjacency matrix `W`.
"""

# ╔═╡ 70000000-0000-0000-0000-000000000009
begin
    df_dyn = DataFrame(
        years = years,
        y = obs_survey,
        s_idx = fill(1, length(years)),
        t_idx = years,
    )
    W_dyn = Matrix(1.0I, 1, 1)
    m_dyn = @tbm(
        likelihood(y, family = gaussian) ~ intercept() +
            dynamics(s_idx, t_idx, model = "delay_difference",
                     intrinsic_growth_rate = LogNormal(0.0, 1.0),
                     carrying_capacity = LogNormal(log(100.0), 1.0),
                     natural_mortality_rate = LogNormal(-1.0, 0.5)),
        df_dyn;
        W = W_dyn,
        verbose = false
    )
    if run_model
        chain_dyn = sample(m_dyn, NUTS(0.65), 100; progress = false, check_model = false)
        (status = "sampled", keys = collect(keys(chain_dyn)))
    else
        (status = "Sampling skipped (run_model = false).",)
    end
end

# ╔═╡ 70000000-0000-0000-0000-000000000010
md"""
## 11. Delay-difference via `custom()` + `@code_string` (CodeTracking)
"""

# ╔═╡ 70000000-0000-0000-0000-000000000011
md"""
`CodeTracking.@code_string` extracts the source text of a function at runtime.
We use it to pull the body of `TethysPop.bayesian_delay_difference_model`
and inject it via `custom()`.
"""

# ╔═╡ 70000000-0000-0000-0000-000000000012
begin
    # Extract the full @model source text via CodeTracking.
    full_text = @code_string TethysPop.bayesian_delay_difference_model(
        catches, recruits, obs_survey, cfg
    )

    # Use TethysBayes's extract_model_body to strip the @model wrapper.
    code_fragment_ct = extract_model_body(full_text)

    model_ct = @tbm(
        likelihood(y, family = gaussian) ~ intercept() + custom(code_fragment_ct),
        df_hybrid;
        params = Dict(:cfg => cfg),
        verbose = false
    )
    if run_model
        chain_ct = sample(model_ct, NUTS(0.65), 200; progress = false, check_model = false)
        (status = "sampled",)
    else
        (status = "Sampling skipped (run_model = false).",)
    end
end