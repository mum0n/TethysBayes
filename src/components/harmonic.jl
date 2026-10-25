# File: c:\home\jae\projects\TethysBayes\src\components\harmonic.jl
"""
    Harmonic <: ComponentModel

A component model for harmonic temporal effects, capturing periodic patterns using
sine and cosine waves. This component can model one or more harmonics, each with its
own amplitude, phase, and potentially its own period.

# Version
v1.0.0

# Mathematical Summary
The component models a function \\(f(t)\\) as a sum of sinusoids. It supports two
parameterizations controlled by the `method` field:

1.  **:twocoefficient (default, AD-friendly)**:
    \$f(t) = \\sum_{k=1}^{N_{harmonics}} \\left( \\beta_{\\cos,k} \\cos\\left(\\frac{2\\pi k
      t}{P_k}\\right) + \\beta_{\\sin,k} \\sin\\left(\\frac{2\\pi k t}{P_k}\\right)
      \\right)\$
    This is the recommended method as it is more efficient for gradient-based MCMC.

2.  **:ampphase (didactic)**:
    \$f(t) = \\sum_{k=1}^{N_{harmonics}} A_k \\cos\\left(\\frac{2\\pi k t}{P_k} + \\phi_k\\right)\$
    where \$A_k\$ is the amplitude, \$P_k\$ is the period, and \$\\phi_k\$ is the phase shift.
    This is retained as a more intuitive, didactic alternative.

# Computational Methods
- `:twocoefficient` (Default, AD-friendly): A two-coefficient (sine and cosine)
  parameterization that is efficient for gradient-based samplers.
- `:ampphase` (Didactic, Not AD-friendly): An amplitude-phase parameterization that
  is more intuitive but can be less efficient for MCMC. Retained for didactic purposes.

# Inputs
- **Required**:
  - A seasonal index variable (e.g., `month`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `nharmonics`: `Int`, the number of harmonic terms to include. Default: `1`.
  - `period`: `Real`, `UnivariateDistribution`, or `Vector`. The period(s) of the
    cycle(s). If a single value, it applies to all harmonics. If a vector, its
    length must match `nharmonics`. Default: `12.0`.
  - `amplitude`: `UnivariateDistribution`, prior for the amplitude (for `:ampphase`).
    Default: `Exponential(1.0)`.
  - `phase`: `UnivariateDistribution`, prior for the phase shift (for `:ampphase`).
    Default: `Beta(1,1)`.
  - `method`: `Symbol`, computational method (`:twocoefficient` or `:ampphase`).
    Default: `:twocoefficient`.

# Outputs (Parameter Names)
- `beta_cos_<key>`: Coefficients for the cosine terms (for `:twocoefficient`).
- `beta_sin_<key>`: Coefficients for the sine terms (for `:twocoefficient`).
- `amplitude_<key>`: Amplitudes of the harmonics (for `:ampphase`).
- `phase_<key>`: Phase shifts of the harmonics (for `:ampphase`).
- `period_<key>`: The period of the cycle(s), if estimated.
- `latent_<key>`: The reconstructed latent harmonic effect.
"""
struct Harmonic <: ComponentModel
    nharmonics::Int
    amplitude::Distribution
    phase::Distribution
    period::Union{Real, UnivariateDistribution, Vector}
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:harmonic] = Harmonic

COMPONENT_CONSTRUCTORS[:harmonic] = (p, params) -> begin
    nharmonics = get(params, :nharmonics, 1)
    period_param = get(params, :period, 12.0)
    method = get(params, :method, :twocoefficient)

    if nharmonics > 1
        if period_param isa Real
            period_param = fill(period_param, nharmonics)
        elseif period_param isa UnivariateDistribution
            period_param = [period_param for _ in 1:nharmonics]
        end
        if period_param isa Vector && length(period_param) != nharmonics
            error(
                "Length of `period` vector ($(length(period_param))) must match " *
                "`nharmonics` ($(nharmonics))."
            )
        end
    else # nharmonics == 1
        if period_param isa Vector
            error(
                "For `nharmonics = 1`, `period` must be a Real or a single " *
                "UnivariateDistribution, not a Vector."
            )
        end
    end
    
    Harmonic(nharmonics, p.amplitude, p.phase, period_param, method)
end

MODEL_TO_STRUCTURE_MAP[:harmonic] = :seasonal

function get_precomputes(
    m::Harmonic, M::NamedTuple, mod_data::Dict
)::NamedTuple
    # Validate that a seasonal index variable is provided or auto-detect it.
    raw_vars = get(mod_data, :variables, Symbol[])
    variables = raw_vars isa AbstractVector ? raw_vars : [raw_vars]
    u_var_sym = if !isempty(variables)
        Symbol(variables[1])
    elseif hasproperty(M, :data)
        _detect_seasonal_column(M.data; allow_nothing=false)
    else
        :u_idx
    end

    # Extract the seasonal index variable from the data.
    raw_u = hasproperty(M, :data) && hasproperty(M.data, u_var_sym) ?
        M.data[!, u_var_sym] : collect(1:get(M, :N_time, 12))
    u_idx = if eltype(raw_u) <: Integer
        Int.(raw_u)
    elseif haskey(M, :u_idx)
        M.u_idx
    else
        collect(1:length(raw_u))
    end
    u_N = haskey(M, :u_N) && M.u_N > 0 ? M.u_N :
        (hasproperty(M, :N_time) ? M.N_time : Int(maximum(u_idx)))
    u_idx_var = u_var_sym

    u_coords = collect(1.0:u_N)
    
    return (
        u_coords=u_coords, 
        u_idx=u_idx,
        u_N=u_N, 
        u_idx_var=u_idx_var,
        model_type=:harmonic,
        period=m.period
    )
end

function get_priors(
    m::Harmonic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    priors = String[]

    if m.method == :twocoefficient
        # Priors for cosine and sine coefficients
        beta_cos_prior = _distribution_to_string(Normal(0, 1))
        beta_sin_prior = _distribution_to_string(Normal(0, 1))
        push!(priors, "$(p_names.beta_cos) ~ filldist($(beta_cos_prior), $(m.nharmonics))")
        push!(priors, "$(p_names.beta_sin) ~ filldist($(beta_sin_prior), $(m.nharmonics))")
    elseif m.method == :ampphase
        # Priors for amplitude and phase
        amplitude_prior_str = _distribution_to_string(m.amplitude)
        phase_prior_str = _distribution_to_string(m.phase)
        push!(priors, "$(p_names.amplitude) ~ filldist($(amplitude_prior_str), $(m.nharmonics))")
        push!(priors, "$(p_names.phase) ~ filldist($(phase_prior_str), $(m.nharmonics))")
    end

    if m.period isa UnivariateDistribution
        push!(priors, "$(_prior_or_constant(p_names.period, m.period))")
    elseif m.period isa Vector{<:UnivariateDistribution}
        period_prior_str = _distribution_to_string(m.period[1])
        push!(priors, "$(p_names.period) ~ filldist($(period_prior_str), $(m.nharmonics))")
    end

    return join(priors, "\n    ")
end

function get_updates(
    m::Harmonic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    u_coords_access = "spec_registry[:$(spec.key)].hyper.u_coords"
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"

    period_access_code = if m.period isa Real
        string(m.period)
    elseif m.period isa UnivariateDistribution
        string(p_names.period)
    elseif m.period isa Vector{<:Real}
        "($(string(m.period)))[k]"
    elseif m.period isa Vector{<:UnivariateDistribution}
        "$(string(p_names.period))[k]"
    else
        error("Unsupported type for m.period in get_updates.")
    end

    loop_body = ""
    if m.method == :twocoefficient
        loop_body = """
            b_cos = $(m.nharmonics > 1 ? "$(p_names.beta_cos)[k]" : p_names.beta_cos)
            b_sin = $(m.nharmonics > 1 ? "$(p_names.beta_sin)[k]" : p_names.beta_sin)
            period_val = $(period_access_code)
            
            angle = (2 * pi * k ./ period_val) .* $(u_coords_access)
            $(p_names.latent_field) .+= b_cos .* cos.(angle) .+ b_sin .* sin.(angle)
        """
    else # :ampphase
        loop_body = """
            amp = $(m.nharmonics > 1 ? "$(p_names.amplitude)[k]" : p_names.amplitude)
            phase = $(m.nharmonics > 1 ? "$(p_names.phase)[k]" : p_names.phase)
            period_val = $(period_access_code)
            
            angle = (2 * pi * k ./ period_val) .* $(u_coords_access)
            $(p_names.latent_field) .+= amp .* cos.(angle .+ (2 * pi * phase))
        """
    end

    init_param = if m.method == :twocoefficient
        p_names.beta_cos
    else
        p_names.amplitude
    end
    
    return """
        # --- Harmonic Component: $(spec.key) ($(m.method)) ---
        let
            T_num = eltype($(init_param))
            u_N_val = spec_registry[:$(spec.key)].hyper.u_N
            u_idx_val = spec_registry[:$(spec.key)].hyper.u_idx
            $(p_names.latent_field) = zeros(T_num, u_N_val)
            for k in 1:$(m.nharmonics)
                $(loop_body)
            end
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), u_idx_val)
        end
    """
end

"""
    get_effects(m::Harmonic, chain, spec::NamedTuple, M::NamedTuple, PS)

Reconstructs the harmonic effect from posterior samples. This version is CPU-only
and uses modern chain accessors.
"""
function get_effects(
    m::Harmonic, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup ---
    n_samples = if occursin("FlexiChain", string(typeof(chain)))
        size(chain, 1) * FlexiChains.nchains(chain)
    else
        size(chain, 1) * size(chain, 3)
    end
    outcomes_N = M.outcomes_N
    p_names = string.(keys(chain))
    
    hyper = spec.hyper
    is_multivariate_model = M.model_arch == "multivariate"
    u_coords = hyper.u_coords
    u_N_hyper = hyper.u_N

    # --- Index Handling ---
    u_idx_train = hyper.u_idx
    u_idx_full = if !isnothing(PS)
        if hasproperty(PS, :u_idx)
            vcat(u_idx_train, PS.u_idx)
        elseif hasproperty(PS, :data) && hasproperty(PS.data, hyper.u_idx_var)
            raw_new = PS.data[!, hyper.u_idx_var]
            if haskey(M, :u_values) && !isnothing(M.u_values)
                u_map = Dict(v => i for (i, v) in enumerate(M.u_values))
                vcat(u_idx_train, [get(u_map, v, 1) for v in raw_new])
            else
                vcat(u_idx_train, Int.(round.(raw_new)))
            end
        else
            u_idx_train
        end
    else
        u_idx_train
    end
    u_idx_int = Int.(round.(u_idx_full))
    N_total = length(u_idx_int)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop ---
    for k_outcome in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k_outcome)

        # --- Get Period Samples ---
        period_name = _find_parameter(p_names, string(p_names_k.period), k_outcome,
            is_multivariate_model)
        period_samples = if m.period isa Real
            fill(m.period, n_samples, m.nharmonics)
        elseif m.period isa Vector{<:Real}
            repeat(m.period', n_samples, 1)
        else
            if isempty(period_name)
                @warn "Period parameter for Harmonic component $(spec.key) not found. Using prior mean."
                mean_period = m.period isa Vector ? mean.(m.period) : [mean(m.period)]
                repeat(mean_period', n_samples, 1)
            else
                get_params_matrix(chain, period_name, m.nharmonics)
            end
        end
        
        # Reshape for broadcasting: [1, n_samples, n_harmonics]
        period_samples_tensor = reshape(period_samples, 1, n_samples, m.nharmonics)

        # --- Calculate Angles ---
        # u_coords: [u_N, 1, 1], k_harmonics: [1, 1, n_harmonics]
        u_coords_tensor = reshape(u_coords, u_N_hyper, 1, 1)
        k_harmonics_tensor = reshape(1:m.nharmonics, 1, 1, m.nharmonics)
        
        # Broadcasting happens here
        angle_tensor = (2 * pi .* k_harmonics_tensor ./ period_samples_tensor) .* u_coords_tensor
        
        cos_angle = cos.(angle_tensor)
        sin_angle = sin.(angle_tensor)

        # --- Get Coefficient Samples and Compute Effect ---
        local reconstructed_effects_k
        if m.method == :twocoefficient
            b_cos_name = _find_parameter(p_names, string(p_names_k.beta_cos), k_outcome,
                is_multivariate_model)
            b_sin_name = _find_parameter(p_names, string(p_names_k.beta_sin), k_outcome,
                is_multivariate_model)
            
            if isempty(b_cos_name) || isempty(b_sin_name)
                @warn "beta_cos or beta_sin for Harmonic $(spec.key) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end

            b_cos_samples = get_params_matrix(chain, b_cos_name, m.nharmonics)
            b_sin_samples = get_params_matrix(chain, b_sin_name, m.nharmonics)

            # Reshape for broadcasting: [1, n_samples, n_harmonics]
            b_cos_tensor = reshape(b_cos_samples, 1, n_samples, m.nharmonics)
            b_sin_tensor = reshape(b_sin_samples, 1, n_samples, m.nharmonics)

            harmonic_effects = b_cos_tensor .* cos_angle .+ b_sin_tensor .* sin_angle
            reconstructed_effects_k = dropdims(sum(harmonic_effects, dims=3),
                dims=3) # Result is [u_N, n_samples]

        else # :ampphase
            amp_name = _find_parameter(p_names, string(p_names_k.amplitude), k_outcome,
                is_multivariate_model)
            phase_name = _find_parameter(p_names, string(p_names_k.phase), k_outcome,
                is_multivariate_model)

            if isempty(amp_name) || isempty(phase_name)
                @warn "amplitude or phase for Harmonic $(spec.key) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end

            amp_samples = get_params_matrix(chain, amp_name, m.nharmonics)
            phase_samples = get_params_matrix(chain, phase_name, m.nharmonics)

            # Reshape for broadcasting: [1, n_samples, n_harmonics]
            amp_tensor = reshape(amp_samples, 1, n_samples, m.nharmonics)
            phase_tensor = reshape(phase_samples, 1, n_samples, m.nharmonics)

            harmonic_effects = amp_tensor .* cos.(angle_tensor .+ (2 * pi .* phase_tensor))
            reconstructed_effects_k = dropdims(sum(harmonic_effects, dims=3),
                dims=3) # Result is [u_N, n_samples]
        end

        # --- Index and Finalize ---
        indexed_effects = reconstructed_effects_k[u_idx_int, :]
        push!(structured_effects, indexed_effects)
    end

    return (structured=structured_effects, noisy=structured_effects)
end
