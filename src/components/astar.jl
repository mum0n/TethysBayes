"""
    AStar <: ComponentModel

A component model for Bayesian goal-directed migratory pathfinding and least-cost
corridor inference on irregular spatial graphs. It models spatial random effects
parameterized by habitat resistance and travel friction, with native support for
errors-in-variables propagation of habitat suitability index (HSI) uncertainty.

# Version
v1.0.0

# Mathematical Summary
While circuit theory models undirected, exploratory diffusive flux across all possible
pathways, migratory animals frequently exhibit goal-directed navigation toward known
spawning, feeding, or overwintering grounds. The `AStar` component integrates the
heuristic ``A^*`` search algorithm into the Bayesian hierarchical modeling framework.

Let ``\\mathbf{W}`` be the graph adjacency matrix with physical cell centroid
coordinates ``\\mathbf{x}_u \\in \\mathbb{R}^2``. When HSI standard errors
``\\boldsymbol{\\sigma}_H`` are provided (`habitat_se`), latent habitat quality
is modeled with errors-in-variables:
```math
H_u^{\\text{eff}} = H_u^{\\text{obs}} + \\sigma_{H, u} \\cdot u_{H, u}
```
with latent innovations ``u_{H, u} \\sim \\mathcal{N}(0, 1)``.
Between adjacent graph units ``u \\sim v`` with physical distance
``d_{uv} = \\|\\mathbf{x}_u - \\mathbf{x}_v\\|_2``, travel friction is:
```math
r_{uv} = d_{uv} \\cdot \\left(
  \\frac{(1 - H_u^{\\text{eff}})^p + (1 - H_v^{\\text{eff}})^p}{2}
\\right)
```
where ``p \\ge 1.0`` (default ``p = 2.0``) is the friction exponent.

# Spatial Autocorrelation Prior (GMRF)
During MCMC sampling, the spatial random effect ``\\mathbf{s} \\in \\mathbb{R}^S`` is
modeled as a Gaussian Markov Random Field with precision matrix structured by
travel friction:
```math
W_{A, uv} = W_{uv} \\cdot \\exp(-\\beta \\cdot r_{uv})
```
```math
\\mathbf{Q}_A = \\mathbf{D}_A - \\mathbf{W}_A + \\epsilon \\mathbf{I}
```
```math
\\mathbf{s} \\sim \\mathcal{N}\\left( \\mathbf{0},
  \\left( \\frac{1}{\\sigma^2} \\mathbf{Q}_A \\right)^{-1} \\right)
```
Locations connected by low-friction corridors share strong spatial covariance, while
barriers (land, unsuitable thermal regimes) decouple spatial correlation.

# Posterior Inference on Least-Cost Corridors
When tag release (`sources`) and recapture (`sinks`) nodes are provided, the
component propagates the posterior distribution of latent habitat across all MCMC
draws through the ``A^*`` search engine with admissible Euclidean heuristic:
```math
h(u, v) = \\|\\mathbf{x}_u - \\mathbf{x}_v\\|_2
```
This yields the posterior corridor inclusion probability field:
```math
P(s \\in \\text{Corridor} \\mid \\text{data}) =
  \\frac{1}{M} \\sum_{m=1}^M \\mathbb{I}(s \\in \\boldsymbol{\\pi}^{(m)})
```
the consensus medoid trajectory ``\\boldsymbol{\\pi}^*``, and 95% credible intervals
on cumulative migration distance.

# Inputs
- **Required**:
  - A spatial index variable (e.g. `region`, `s_idx`) passed to `random()`.
  - An adjacency matrix `W` passed as a keyword argument to `@tbm(..., W=W)`.
  - Cell centroids passed as `@tbm(..., centroids=centroids)`.
  - `habitat`: A `Symbol` pointing to a column in the data, or a `Vector` of length `s_N`.
- **Optional (in `random()` call)**:
  - `habitat_se`: `Symbol`, `Vector`, or `Real`, standard error on HSI.
  - `beta`: `UnivariateDistribution`, prior for friction sensitivity (default: `Normal(1.0, 1.0)`).
  - `sigma`: `UnivariateDistribution`, prior for spatial scale (default: `Exponential(1.0)`).
  - `sources`: `Vector{Int}` or `Int`, release / origin node indices.
  - `sinks`: `Vector{Int}` or `Int`, recapture / destination node indices.
  - `friction_power`: `Union{Real, UnivariateDistribution, Symbol}`, exponent ``p`` in
    friction formula (default: 2.0). Can be set to a fixed constant (e.g. `2.0`), a
    distribution prior (e.g. `Gamma(2.0, 1.0)`), or `:random` / `:estimate` to treat
    friction power as an inferable Bayesian random variable.
  - `smooth`: `Bool`, whether to apply trajectory smoothing (default: true).
  - `land_mask`: `Vector{Bool}`, indicator vector for land units.
  - `method`: `Symbol`, computational method (`:cholesky`, default).

# Outputs (Parameter Names)
- `beta_<key>`: Friction sensitivity parameter.
- `sigma_<key>`: Marginal standard deviation of the spatial random effect.
- `friction_power_<key>`: Inferred friction power exponent (when modeled as random).
- `innovations_<key>`: Standard normal innovations for the spatial field.
- `innovations_habitat_<key>`: Latent habitat innovation terms (when `habitat_se` is supplied).
- `latent_field_<key>`: Structured spatial effect vector.
- `astar_paths`: `StochasticAStarResult` containing posterior corridor probabilities,
  consensus medoid path, and migration distance credible intervals.

# Key References
- Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic
  determination of minimum cost paths. IEEE Transactions on Systems Science and
  Cybernetics, 4(2), 100-107.
- Hanks, E. M., & Hooten, M. B. (2013). Circuit theory and model-based inference for
  landscape connectivity. Journal of the American Statistical Association, 108(501), 22-33.
"""
struct AStar <: ComponentModel
    beta::UnivariateDistribution
    sigma::Union{UnivariateDistribution, Real}
    friction_power::Union{Float64, UnivariateDistribution}
    smooth::Bool
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:astar] = AStar
COMPONENT_CONSTRUCTORS[:astar] = (p, params) -> begin
    f_pow_raw = if haskey(p, :friction_power)
        p[:friction_power]
    elseif haskey(params, :friction_power)
        params[:friction_power]
    else
        2.0
    end
    f_pow = if f_pow_raw isa UnivariateDistribution
        f_pow_raw
    elseif f_pow_raw in (:random, true, :estimate)
        Gamma(2.0, 1.0)
    elseif f_pow_raw isa Real
        Float64(f_pow_raw)
    else
        f_pow_raw
    end
    AStar(
        get(p, :beta, Normal(1.0, 1.0)),
        get(p, :sigma, Exponential(1.0)),
        f_pow,
        Bool(get(params, :smooth, true)),
        Symbol(get(params, :method, :cholesky))
    )
end

MODEL_TO_STRUCTURE_MAP[:astar] = :spatial

"""
    _resolve_astar_indices(val, M)::Vector{Int}

Internal helper to resolve node indices from vectors, integers, symbols, or expressions.
"""
function _resolve_astar_indices(val, M)::Vector{Int}
    if val isa AbstractVector{<:Integer}
        return collect(Int, val)
    elseif val isa Integer
        return [Int(val)]
    elseif val isa Symbol
        if hasproperty(M, val) && getproperty(M, val) isa AbstractVector{<:Integer}
            return collect(Int, getproperty(M, val))
        elseif hasproperty(M, val) && getproperty(M, val) isa Integer
            return [Int(getproperty(M, val))]
        elseif haskey(M, val) && M[val] isa AbstractVector{<:Integer}
            return collect(Int, M[val])
        elseif haskey(M, val) && M[val] isa Integer
            return [Int(M[val])]
        elseif hasproperty(M.data, val)
            col = M.data[!, val]
            if eltype(col) == Bool
                return findall(col)
            elseif eltype(col) <: Integer
                return unique(filter(x -> x > 0, col))
            end
        elseif isdefined(Main, val)
            main_val = getfield(Main, val)
            if main_val isa AbstractVector{<:Integer}
                return collect(Int, main_val)
            elseif main_val isa Integer
                return [Int(main_val)]
            end
        end
    elseif val isa Expr
        try
            eval_val = Core.eval(Main, val)
            if eval_val isa AbstractVector{<:Integer}
                return collect(Int, eval_val)
            elseif eval_val isa Integer
                return [Int(eval_val)]
            end
        catch
        end
    end
    return Int[]
end

"""
    get_precomputes(m::AStar, M::NamedTuple, mod_data::Dict)::NamedTuple

Precomputes graph connectivity, habitat covariates, node centroids, and errors-in-variables.
"""
function get_precomputes(m::AStar, M::NamedTuple, mod_data::Dict)::NamedTuple
    s_N = M.s_N
    W = M.W
    params = mod_data[:params]
    data = M.data

    if !haskey(params, :habitat)
        error(
            "The `astar` model requires a `habitat` parameter specifying the " *
            "suitability/conductance field."
        )
    end

    habitat_val = params[:habitat]
    local habitat_data::Vector{Float64}

    if habitat_val isa Symbol
        if !hasproperty(data, habitat_val)
            error("Habitat variable ':$habitat_val' not found in data frame.")
        end
        habitat_per_obs = data[!, habitat_val]
        habitat_aggregated = zeros(Float64, s_N)
        counts = zeros(Int, s_N)
        for i in 1:M.y_N
            s_i = M.s_idx[i]
            habitat_aggregated[s_i] += habitat_per_obs[i]
            counts[s_i] += 1
        end
        habitat_data = habitat_aggregated ./ max.(1, counts)
    elseif habitat_val isa AbstractVector
        if length(habitat_val) == s_N
            habitat_data = convert(Vector{Float64}, habitat_val)
        elseif length(habitat_val) == M.y_N
            habitat_aggregated = zeros(Float64, s_N)
            counts = zeros(Int, s_N)
            for i in 1:M.y_N
                s_i = M.s_idx[i]
                habitat_aggregated[s_i] += habitat_val[i]
                counts[s_i] += 1
            end
            habitat_data = habitat_aggregated ./ max.(1, counts)
        else
            error(
                "Provided `habitat` length ($(length(habitat_val))) must match " *
                "s_N ($s_N) or observations ($(M.y_N))."
            )
        end
    else
        error("`habitat` parameter must be a Symbol or Vector of length s_N.")
    end

    # Habitat observation error (errors-in-variables)
    habitat_se_data = zeros(Float64, s_N)
    has_error = false
    if haskey(params, :habitat_se)
        se_val = params[:habitat_se]
        if se_val isa Symbol
            if hasproperty(data, se_val)
                se_per_obs = data[!, se_val]
                se_agg = zeros(Float64, s_N)
                counts = zeros(Int, s_N)
                for i in 1:M.y_N
                    s_i = M.s_idx[i]
                    se_agg[s_i] += se_per_obs[i]
                    counts[s_i] += 1
                end
                habitat_se_data = se_agg ./ max.(1, counts)
                has_error = any(s -> s > 1e-8, habitat_se_data)
            end
        elseif se_val isa AbstractVector
            if length(se_val) == s_N
                habitat_se_data = convert(Vector{Float64}, se_val)
                has_error = any(s -> s > 1e-8, habitat_se_data)
            elseif length(se_val) == M.y_N
                se_agg = zeros(Float64, s_N)
                counts = zeros(Int, s_N)
                for i in 1:M.y_N
                    s_i = M.s_idx[i]
                    se_agg[s_i] += se_val[i]
                    counts[s_i] += 1
                end
                habitat_se_data = se_agg ./ max.(1, counts)
                has_error = any(s -> s > 1e-8, habitat_se_data)
            else
                @warn "habitat_se length mismatch ($(length(se_val)) != $s_N). Ignored."
            end
        elseif se_val isa Real
            habitat_se_data = fill(Float64(se_val), s_N)
            has_error = se_val > 1e-8
        end
    end

    sources_raw = if hasproperty(M, :sources) && !isnothing(M.sources)
        M.sources
    elseif haskey(params, :sources)
        params[:sources]
    else
        Int[]
    end

    sinks_raw = if hasproperty(M, :sinks) && !isnothing(M.sinks)
        M.sinks
    elseif haskey(params, :sinks)
        params[:sinks]
    else
        Int[]
    end

    sources = _resolve_astar_indices(sources_raw, M)
    sinks = _resolve_astar_indices(sinks_raw, M)

    # Node centroids for A* heuristic
    centroids = if hasproperty(M, :centroids) && !isnothing(M.centroids)
        M.centroids
    elseif haskey(params, :centroids)
        params[:centroids]
    else
        nothing
    end

    f_pow_val = if haskey(params, :friction_power)
        p_val = params[:friction_power]
        if p_val in (:random, true, :estimate)
            Gamma(2.0, 1.0)
        elseif p_val isa UnivariateDistribution
            p_val
        elseif p_val isa Real
            Float64(p_val)
        else
            m.friction_power
        end
    else
        m.friction_power
    end
    is_random_power = f_pow_val isa UnivariateDistribution
    smooth = get(params, :smooth, m.smooth)
    land_mask = get(params, :land_mask, nothing)

    I, J, _ = findnz(W)
    return (
        W_I = I,
        W_J = J,
        n_latent = s_N,
        s_N = s_N,
        habitat_data = habitat_data,
        habitat_se_data = habitat_se_data,
        has_error = has_error,
        sources = sources,
        sinks = sinks,
        centroids = centroids,
        friction_power = f_pow_val,
        is_random_power = is_random_power,
        smooth = smooth,
        land_mask = land_mask
    )
end

"""
    get_priors(m::AStar, spec::NamedTuple, arch::String, outcome_idx, M)::String

Defines Bayesian prior distributions for friction sensitivity, spatial scale,
latent habitat innovations under errors-in-variables, and optional random friction power.
"""
function get_priors(
    m::AStar, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    priors = String[]

    push!(priors, "$(_prior_or_constant(p_names.beta, m.beta))")
    push!(priors, "$(_prior_or_constant(p_names.sigma, m.sigma))")

    if spec.hyper.is_random_power
        push!(
            priors,
            "$(p_names.friction_power) ~ " *
            "$(_distribution_to_string(spec.hyper.friction_power))"
        )
    end

    s_N = spec.hyper.s_N
    n_latent = spec.hyper.n_latent

    if spec.hyper.has_error
        push!(priors, "$(p_names.innovations_habitat) ~ MvNormal(zeros(T, $(s_N)), I)")
    end

    push!(priors, "$(p_names.innovations) ~ MvNormal(zeros(T, $(n_latent)), I)")

    return join(priors, "\n    ")
end

"""
    get_updates(m::AStar, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates DynamicPPL code for assembling the travel friction precision matrix and
updating the linear predictor eta.
"""
function get_updates(
    m::AStar, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    f_pow_setup = if spec.hyper.is_random_power
        "f_pow = max(0.05, $(p_names.friction_power))"
    else
        "f_pow = $(spec.hyper.friction_power)"
    end

    code = if spec.hyper.has_error
        """
        # --- AStar Component (Errors-in-Variables): $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            W_I = hyper.W_I
            W_J = hyper.W_J
            s_N = hyper.s_N
            hab_mean = hyper.habitat_data
            hab_se = hyper.habitat_se_data

            hab_latent = hab_mean .+ hab_se .* $(p_names.innovations_habitat)
            hab_clamped = clamp.(hab_latent, 0.001, 0.999)

            # Resistance: r = (1 - h)^p
            $(f_pow_setup)
            res_I = (1.0 .- hab_clamped[W_I]) .^ f_pow
            res_J = (1.0 .- hab_clamped[W_J]) .^ f_pow
            r_edge = (res_I .+ res_J) ./ 2.0

            # Conductance: C = exp(-beta * r)
            C_vals = exp.(-abs($(p_names.beta)) .* r_edge)
            W_A = sparse(W_I, W_J, C_vals, s_N, s_N)
            D_A = Diagonal(vec(sum(W_A, dims=2)))
            Q_A = D_A - W_A

            F = cholesky(Symmetric(Matrix(Q_A) + M.noise * I))
            $(p_names.latent_field) = $(p_names.sigma) .* (F.L' \\ $(p_names.innovations))

            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.s_idx)
        end
        """
    else
        """
        # --- AStar Component (Fixed Resistance): $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            W_I = hyper.W_I
            W_J = hyper.W_J
            s_N = hyper.s_N
            habitat = hyper.habitat_data
            hab_clamped = clamp.(habitat, 0.001, 0.999)

            $(f_pow_setup)
            res_I = (1.0 .- hab_clamped[W_I]) .^ f_pow
            res_J = (1.0 .- hab_clamped[W_J]) .^ f_pow
            r_edge = (res_I .+ res_J) ./ 2.0

            C_vals = exp.(-abs($(p_names.beta)) .* r_edge)
            W_A = sparse(W_I, W_J, C_vals, s_N, s_N)
            D_A = Diagonal(vec(sum(W_A, dims=2)))
            Q_A = D_A - W_A

            F = cholesky(Symmetric(Matrix(Q_A) + M.noise * I))
            $(p_names.latent_field) = $(p_names.sigma) .* (F.L' \\ $(p_names.innovations))

            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.s_idx)
        end
        """
    end

    return code
end

"""
    get_effects(m::AStar, chain, spec::NamedTuple, M::NamedTuple, PS)::NamedTuple

Reconstructs the latent spatial field and evaluates posterior migratory A* path
inference across all MCMC draws.
"""
function get_effects(
    m::AStar, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    n_samples = if occursin("FlexiChain", string(typeof(chain)))
        size(chain, 1) * FlexiChains.nchains(chain)
    else
        size(chain, 1) * size(chain, 3)
    end
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = string.(keys(chain))

    key = spec.key
    hyper = spec.hyper
    W_I = hyper.W_I
    W_J = hyper.W_J
    s_N = hyper.s_N
    habitat_data = hyper.habitat_data
    habitat_se_data = hyper.habitat_se_data
    has_error = hyper.has_error
    noise = M.noise
    f_pow = hyper.friction_power

    s_idx_train = M.s_idx
    s_idx_full = if !isnothing(PS) && hasproperty(PS.data, :s_idx)
        vcat(s_idx_train, PS.data.s_idx)
    else
        s_idx_train
    end
    N_total = length(s_idx_full)

    structured_effects = Vector{Matrix{Float64}}()
    h_latent_matrix = zeros(Float64, s_N, n_samples)

    for k_outcome in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k_outcome)
        beta_name = _find_parameter(
            p_names, string(p_names_k.beta), k_outcome, is_multivariate_model
        )
        sigma_name = _find_parameter(
            p_names, string(p_names_k.sigma), k_outcome, is_multivariate_model
        )
        innovations_name = _find_parameter(
            p_names, string(p_names_k.innovations), k_outcome, is_multivariate_model
        )

        if isempty(beta_name) || isempty(sigma_name) || isempty(innovations_name)
            @warn "Parameters for AStar component $(spec.key) not found."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        beta_samples = get_params_vector(chain, beta_name, 1)[:, 1]
        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        innovations_samples = get_params_matrix(chain, innovations_name, s_N)

        innovations_habitat_samples = if has_error
            hab_name = _find_parameter(
                p_names, string(p_names_k.innovations_habitat), k_outcome, is_multivariate_model
            )
            !isempty(hab_name) ? get_params_matrix(chain, hab_name, s_N) : nothing
        else
            nothing
        end

        f_pow_samples = if hyper.is_random_power
            f_name = _find_parameter(
                p_names, string(p_names_k.friction_power), k_outcome, is_multivariate_model
            )
            if !isempty(f_name)
                get_params_vector(chain, f_name, 1)[:, 1]
            else
                fill(2.0, n_samples)
            end
        else
            fill(Float64(hyper.friction_power isa Real ? hyper.friction_power : 2.0), n_samples)
        end

        reconstructed_effects_k = zeros(Float64, s_N, n_samples)

        for i in 1:n_samples
            hab_i = if has_error && !isnothing(innovations_habitat_samples)
                habitat_data .+ habitat_se_data .* innovations_habitat_samples[i, :]
            else
                habitat_data
            end
            if k_outcome == 1
                h_latent_matrix[:, i] = hab_i
            end

            hab_c = clamp.(hab_i, 0.001, 0.999)
            f_pow_i = f_pow_samples[i]
            res_I = (1.0 .- hab_c[W_I]) .^ f_pow_i
            res_J = (1.0 .- hab_c[W_J]) .^ f_pow_i
            r_edge = (res_I .+ res_J) ./ 2.0

            C_vals = exp.(-abs(beta_samples[i]) .* r_edge)
            W_A_i = sparse(W_I, W_J, C_vals, s_N, s_N)
            D_A_i = Diagonal(vec(sum(W_A_i, dims=2)))
            Q_A_i = D_A_i - W_A_i

            F_i = cholesky(Symmetric(Matrix(Q_A_i) + noise * I))
            innov_i = innovations_samples[i, :]
            reconstructed_effects_k[:, i] = sigma_samples[i] .* (F_i.L' \ innov_i)
        end

        push!(structured_effects, reconstructed_effects_k[s_idx_full, :])
    end

    # If sources, sinks, and centroids are available, compute stochastic A* path inference
    astar_res = if !isempty(hyper.sources) && !isempty(hyper.sinks) &&
                   !isnothing(hyper.centroids)
        rel = hyper.sources[1]
        rec = hyper.sinks[1]
        p_arg = if hyper.is_random_power
            p_k1 = generate_full_variable_names(spec, M.model_arch, 1)
            f_name = _find_parameter(
                p_names, string(p_k1.friction_power), 1, is_multivariate_model
            )
            if !isempty(f_name)
                get_params_vector(chain, f_name, 1)[:, 1]
            else
                hyper.friction_power
            end
        else
            hyper.friction_power
        end

        astar_stochastic_least_cost_path(
            hyper.centroids,
            M.W,
            rel,
            rec;
            hsi_samples = h_latent_matrix,
            friction_power = p_arg,
            land_mask = hyper.land_mask,
            smooth = hyper.smooth
        )
    else
        nothing
    end

    return (
        structured = structured_effects,
        noisy = structured_effects,
        astar_paths = astar_res,
        habitat_latent = h_latent_matrix
    )
end

"""
    get_astar_paths(
        model_result;
        key::Symbol = :astar
    ) -> Union{Nothing, StochasticAStarResult}

    get_astar_paths(
        model_result,
        release::Integer,
        recapture::Integer;
        key::Symbol = :astar,
        friction_power::Union{Nothing, Real, AbstractVector{<:Real}, Distribution} = nothing,
        land_mask = nothing,
        smooth::Bool = true,
        seed::Union{Int, Nothing} = nothing
    ) -> StochasticAStarResult

Extracts or computes stochastic A* least-cost path inference from a fitted TethysBayes model
containing an `AStar` component.
"""
function get_astar_paths(model_result; key::Symbol = :astar)
    eff = if hasproperty(model_result, :effects)
        model_result.effects
    elseif model_result isa Dict
        model_result
    else
        nothing
    end

    if !isnothing(eff)
        if haskey(eff, key) && haskey(eff[key], :astar_paths)
            return eff[key][:astar_paths]
        end
        for (k, v) in pairs(eff)
            if (v isa Dict || v isa NamedTuple) && haskey(v, :astar_paths)
                return v[:astar_paths]
            end
        end
    end
    return nothing
end

function get_astar_paths(
    model_result,
    release::Integer,
    recapture::Integer;
    key::Symbol = :astar,
    friction_power::Union{Nothing, Real, AbstractVector{<:Real}, Distribution} = nothing,
    land_mask = nothing,
    smooth::Bool = true,
    seed::Union{Int, Nothing} = nothing
)::StochasticAStarResult
    M = model_result.M
    spec = nothing
    for comp in M.components
        if comp.key == key || comp.component_obj isa AStar
            spec = comp
            break
        end
    end

    if isnothing(spec)
        error("AStar component not found in model.")
    end

    eff = if hasproperty(model_result, :effects)
        model_result.effects
    elseif model_result isa Dict
        model_result
    else
        nothing
    end

    h_samples = nothing
    if !isnothing(eff)
        if haskey(eff, key) && haskey(eff[key], :habitat_latent)
            h_samples = eff[key][:habitat_latent]
        else
            for (k, v) in pairs(eff)
                if (v isa Dict || v isa NamedTuple) && haskey(v, :habitat_latent)
                    h_samples = v[:habitat_latent]
                    break
                end
            end
        end
    end

    if isnothing(h_samples)
        h_mean = spec.hyper.habitat_data
        h_se = spec.hyper.habitat_se_data
        n_draws = 30
        S = spec.hyper.s_N
        mat = zeros(Float64, S, n_draws)
        for m in 1:n_draws
            mat[:, m] = clamp.(h_mean .+ h_se .* randn(S), 0.001, 0.999)
        end
        h_samples = mat
    end

    centroids = if !isnothing(spec.hyper.centroids)
        spec.hyper.centroids
    elseif hasproperty(M, :centroids) && !isnothing(M.centroids)
        M.centroids
    elseif haskey(M, :centroids)
        M[:centroids]
    else
        error("Node centroids not found in model specification. Required for A*.")
    end

    mask = !isnothing(land_mask) ? land_mask : spec.hyper.land_mask

    f_pow = if !isnothing(friction_power)
        friction_power
    elseif spec.hyper.is_random_power && hasproperty(model_result, :chain)
        p_names = string.(keys(model_result.chain))
        p_k1 = generate_full_variable_names(spec, M.model_arch, 1)
        f_name = _find_parameter(
            p_names, string(p_k1.friction_power), 1, M.model_arch == "multivariate"
        )
        if !isempty(f_name)
            get_params_vector(model_result.chain, f_name, 1)[:, 1]
        else
            spec.hyper.friction_power
        end
    else
        spec.hyper.friction_power
    end

    return astar_stochastic_least_cost_path(
        centroids,
        M.W,
        Int(release),
        Int(recapture);
        hsi_samples = h_samples,
        friction_power = f_pow,
        land_mask = mask,
        smooth = smooth,
        seed = seed
    )
end
