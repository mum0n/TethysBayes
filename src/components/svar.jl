"""
    SVAR <: ComponentModel

A component for a Spatially Varying Autoregressive (SVAR) process. This component
models a temporal AR(1) process where the autoregressive coefficient `rho` varies
across spatial units. The spatial variation of `rho` is governed by a specified
GMRF model (e.g., ICAR, Leroux).

# Version
v1.0.0

# Mathematical Summary
The SVAR model defines a spatiotemporal process \$\\psi_{it}\$ for spatial unit \$i\$
at time \$t\$ as a first-order autoregressive process with a spatially varying
persistence parameter \$\\rho_i\$:

\$\\psi_{it} = \\rho_i \\psi_{i,t-1} + \\epsilon_{it}, \\quad \\epsilon_{it} \\sim
  \\mathcal{N}(0, \\sigma^2)\$

The spatially varying coefficient \$\\boldsymbol{\\rho} = (\\rho_1, \\dots, \\rho_{s_N})\$
is itself modeled as a latent Gaussian field, typically with a GMRF prior to
encourage spatial smoothness:

\$\\boldsymbol{\\rho}_{field} \\sim \\mathcal{N}(\\mathbf{0}, (\\tau_{\\rho}
  \\mathbf{Q}_{\\rho})^{-1})\$

where \$\\mathbf{Q}_{\\rho}\$ is the precision matrix of a spatial model (e.g., ICAR).
To ensure stationarity of the AR(1) process (\$-1 < \\rho_i < 1\$), the raw field
is transformed using the `tanh` function: \$\\rho_i = \\tanh(\\rho_{field, i})\$.

# Computational Methods (for the `rho` field)
- `:spectral` (Default, AD-friendly): Regularizes coefficients using a spectral
  decomposition of the precision matrix. Recommended for NUTS.
- `:cholesky` (AD-friendly): Uses a dense Cholesky factorization of the precision
  matrix.
- `:cholesky_sparse` (Didactic, Not AD-friendly): Uses sparse Cholesky factorization,
  which is not compatible with most AD backends.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `region`) passed to `random()`.
  - A temporal index variable (e.g., `year`) passed to `random()`.
  - An adjacency matrix `W` passed as a keyword argument to `@tbm`.
- **Optional (in `random()` call)**:
  - `model`: `Symbol`, the GMRF model for the `rho` field (e.g., `:icar`, `:leroux`).
    Default: `:icar`.
  - `precision_field_scale`: `UnivariateDistribution`, prior for the std. dev. of the `rho` field.
    Default: `Exponential(1.0)`.
  - `precision_field_mixing`: `UnivariateDistribution`, prior for the mixing parameter of the `rho` field's
    model (if applicable). Default: `Beta(1,1)`.
  - `sigma`: `UnivariateDistribution`, prior for the std. dev. of the AR(1) innovations.
    Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method for the `rho` field. Default: `:spectral`.

# Outputs (Parameter Names)
- `rho_sigma_<key>`: Standard deviation of the spatial `rho` field.
- `rho_rho_<key>`: Mixing parameter of the spatial `rho` field (if applicable).
- `sigma_<key>`: Standard deviation of the AR(1) innovations.
- `rho_innovations_<key>`: Raw innovations for the spatial `rho` field.
- `innovations_<key>`: Raw innovations for the temporal AR(1) processes.
- `latent_<key>`: The reconstructed spatiotemporal SVAR effect.
"""
struct SVAR <: ComponentModel
    precision_field_model::Symbol
    precision_field_scale::UnivariateDistribution
    precision_field_mixing::Union{UnivariateDistribution, Nothing}
    sigma::Union{UnivariateDistribution, Real}
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:svar] = SVAR

COMPONENT_CONSTRUCTORS[:svar] = (p, params) -> begin
    raw_rho = get(params, :rho_model, get(params, :model, :icar))
    rho_m = raw_rho === :svar ? :icar : raw_rho
    # `precision_field_scale` is `UnivariateDistribution`-only, so it must NOT be fed a pinned
    # `sigma`. The previous fallback used `p.sigma` for it, so pinning `sigma=1.0` produced
    # `MethodError: Cannot convert an object of type Float64 to a value of type
    # UnivariateDistribution` and the component could not be built at all. A pinned sigma
    # populates `sigma` only; this field keeps its own prior.
    p_rho_sigma = hasproperty(p, :precision_field_scale) ? p.precision_field_scale :
        Exponential(0.5)
    p_rho_rho = hasproperty(p, :precision_field_mixing) ? p.precision_field_mixing : nothing
    p_sigma = hasproperty(p, :sigma) ? p.sigma : Exponential(0.5)
    SVAR(
        rho_m,
        p_rho_sigma,
        p_rho_rho,
        p_sigma,
        get(params, :method, :spectral)
    )
end

# `spacetime` was a bare alias for `svar` -- `COMPONENT_CONSTRUCTORS[:spacetime]` was
# assigned the same closure, and both registry entries pointed at the same `SVAR` struct, so
# it was never a distinct space-time model. It is REMOVED. Two reasons:
#
#   * the DSL already expresses a space-time structure the current way, as a Kronecker
#     product of a spatial and a temporal random effect:
#         random(s_idx, model=icar, sigma=1.0) ⊗ random(t_idx, model=ar1, sigma=1.0)
#     which builds a separable `ICAR`-by-`AR1` structure the user chooses and can vary
#     independently, whereas the alias bundled both penalties into one component;
#   * it left `svar` reachable under two names, which is how the two drifted apart in the
#     first place.
#
# A constructor is retained purely to fail with guidance rather than a bare "unknown model".
# It is deliberately NOT in `COMPONENT_TYPE_REGISTRY`, so `spacetime` is no longer a component.
COMPONENT_CONSTRUCTORS[:spacetime] = (p, params) -> throw(ArgumentError(
    "`model=spacetime` has been removed; it was only ever an alias for `model=svar`. Use " *
    "`model=svar` for a single-factor spatial autoregressive field, or express a " *
    "spatiotemporal structure as a Kronecker product of a spatial and a temporal effect, " *
    "e.g. `random(s_idx, model=icar, sigma=1.0) ⊗ random(t_idx, model=ar1, sigma=1.0)`."
))

# NOTE: `:spacetime` survives as a STRUCTURE kind, not a component name. `svar`,
# `tensorproductsmooth` and `dynamics` all map to it, and the assembly, reconstruction and
# plotting code branches on `structure == :spacetime`. Only the component alias is gone.
MODEL_TO_STRUCTURE_MAP[:svar] = :spacetime

function get_precomputes(m::SVAR, M::NamedTuple, mod_data::Dict)::NamedTuple
    # Validation moved from get_datastructures!
    variables = mod_data[:variables]
    if length(variables) < 2 && (isnothing(get(M, :s_N, nothing)) || isnothing(get(M, :t_N, nothing)))
        error("SVAR requires a spatial and a temporal variable, e.g., `random(s, t, model=svar)`.")
    end

    # The processor is now responsible for creating s_idx, t_idx, s_N, t_N.
    # We just need to get them from the main config M.
    s_N = get(M, :s_N, nothing)
    t_N = get(M, :t_N, nothing)
    if isnothing(s_N) || isnothing(t_N)
        error(
            "SVAR component '$(mod_data[:key])' failed: s_N or t_N not found in model " *
            "configuration. This should have been set by the model processor."
        )
    end
    W = get(M, :W, nothing)
    if isnothing(W) && m.precision_field_model != :iid
        @warn "Adjacency matrix `W` not provided for SVAR model's rho field."
    end

    # This returns CPU arrays
    template = build_structure_template(m.precision_field_model, s_N; W=W)

    return (
        Q_rho_template=template.matrix,
        U_rho=template.U,
        L_rho=template.L,
        scaling_factor_rho=template.scaling_factor,
        n_latent_rho=s_N,
        n_latent_svar=s_N * t_N
    )
end


function get_priors(
    m::SVAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    key = spec.key

    priors_acc = String[]
    push!(priors_acc, "$(_prior_or_constant(p_names.precision_field_scale, m.precision_field_scale))")
    if m.precision_field_model in [:leroux, :bym2] && !isnothing(m.precision_field_mixing)
        push!(priors_acc, "$(_prior_or_constant(p_names.precision_field_mixing, m.precision_field_mixing))")
    end
    push!(priors_acc, "$(_prior_or_constant(p_names.sigma, m.sigma))")

    push!(priors_acc,
        "$(p_names.innovations_precision) ~ MvNormal(zeros(T, spec_registry[:$(key)].hyper.n_latent_rho), I)")
    push!(priors_acc,
        "$(p_names.innovations) ~ MvNormal(zeros(T, spec_registry[:$(key)].hyper.n_latent_svar), I)")

    return join(priors_acc, "\n    ")
end

function get_updates(
    m::SVAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key

    local rho_recon_code
    if m.method == :spectral
        rho_recon_code = """
            local hyper_rho = spec_registry[:$(key)].hyper
            local D_rho = $(p_names.precision_field_scale) ./ sqrt.(hyper_rho.L_rho .+ M.noise)
            if "$(string(m.precision_field_model))" in ["icar", "besag"]
                _zero_null_modes!(D_rho, hyper_rho.L_rho)
            end
            local rho_field = hyper_rho.U_rho * (D_rho .* $(p_names.innovations_precision))
        """
    else # :cholesky or :cholesky_sparse
        rho_recon_code = """
            local hyper_rho = spec_registry[:$(key)].hyper
            local Q_rho = hyper_rho.Q_rho_template
            local F_rho = cholesky(Symmetric(Matrix(Q_rho) + M.noise * I))
            local rho_field_unscaled = F_rho.L' \\ $(p_names.innovations_precision)
            if "$(string(m.precision_field_model))" in ["icar", "besag"]
                Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * hyper_rho.n_latent_rho),
                  sum(rho_field_unscaled))
            end
            local rho_field = rho_field_unscaled .* $(p_names.precision_field_scale)
        """
    end

    return """
        # --- SVAR Component: $(key) ($(m.method)) ---
        let
            # 1. Reconstruct the spatially-varying AR(1) coefficient `rho`.
            $(rho_recon_code)
            local rho_s = tanh.(rho_field) # Constrain rho to (-1, 1)

            # 2. Evolve the SVAR state-space.
            local latent_st = zeros(eltype(rho_s), M.s_N, M.t_N)
            local innovations_grid = reshape($(p_names.innovations), M.s_N, M.t_N)
            
            latent_st[:, 1] = ($(p_names.sigma) ./ sqrt.(1 .- rho_s.^2 .+ M.noise)) .*
              innovations_grid[:, 1]
            for t in 2:M.t_N
                latent_st[:, t] = rho_s .* latent_st[:, t-1] .+ $(p_names.sigma) .*
                  innovations_grid[:, t]
            end
            
            # 3. Map the 2D latent field back to the 1D observation vector.
            $(p_names.latent_field) = [latent_st[M.s_idx[i], M.t_idx[i]] for i in 1:M.y_N]

            # 4. Add the effect to the linear predictor.
            $(eta_target) = $(eta_target) .+ $(p_names.latent_field)
        end
    """
end


function get_effects(
    m::SVAR, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions ---
    n_samples = if occursin("FlexiChain", string(typeof(chain)))
        size(chain, 1) * FlexiChains.nchains(chain)
    else
        size(chain, 1) * size(chain, 3)
    end
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = string.(keys(chain))
    
    # --- Get precomputed data ---
    hyper = spec.hyper
    s_N, t_N, noise = M.s_N, M.t_N, M.noise
    n_latent_svar = s_N * t_N
    Q_rho_template_cpu = hyper.Q_rho_template
    U_rho_cpu = hyper.U_rho
    L_rho_cpu = hyper.L_rho

    # --- Index Handling: Combine training and prediction sets on CPU ---
    s_idx_cpu = isnothing(PS) ? M.s_idx : vcat(M.s_idx, get(PS.data, :s_idx, []))
    t_idx_cpu = isnothing(PS) ? M.t_idx : vcat(M.t_idx, get(PS.data, :t_idx, []))
    N_total = length(s_idx_cpu)
    t_N_full = isempty(t_idx_cpu) ? 0 : maximum(t_idx_cpu)
    
    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        v = generate_full_variable_names(spec, M.model_arch, k)
        
        rho_sigma_name = _find_parameter(p_names, string(v.precision_field_scale), k, is_multivariate_model)
        sigma_name = _find_parameter(p_names, string(v.sigma), k, is_multivariate_model)
        innovations_precision_name = _find_parameter(p_names, string(v.innovations_precision), k, is_multivariate_model)
        innovations_name = _find_parameter(p_names, string(v.innovations), k, is_multivariate_model)
        
        if isempty(rho_sigma_name) || isempty(sigma_name) || isempty(innovations_precision_name)||
            isempty(innovations_name)
            @warn "Parameters for SVAR component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        # Extract posterior samples (these are on the CPU)
        rho_sigma_samples_cpu = get_params_vector(chain, rho_sigma_name, 1)[:, 1]
        sigma_samples_cpu = get_params_vector(chain, sigma_name, 1)[:, 1]
        rho_innovations_samples_cpu = get_params_matrix(chain, innovations_precision_name, s_N)
        innovations_samples_cpu = get_params_matrix(chain, innovations_name, n_latent_svar)
        
        # Initialize the output matrix for the full effect on the CPU
        effect_k_cpu = zeros(Float64, N_total, n_samples)

        # --- Sample-wise Reconstruction on the CPU ---
        for s in 1:n_samples
            rho_sigma_s = rho_sigma_samples_cpu[s]
            sigma_s = sigma_samples_cpu[s]
            rho_innovations_s = rho_innovations_samples_cpu[s, :]
            innovations_s = innovations_samples_cpu[s, :]

            local rho_field_s
            if m.method == :spectral
                D_rho_s = rho_sigma_s ./ sqrt.(L_rho_cpu .+ noise)
                if m.precision_field_model in [:icar, :besag]
                    for _ni in _zero_null_modes!(zeros(length(L_rho_cpu)), L_rho_cpu)
                        D_rho_s[_ni] = 0.0
                    end
                end
                rho_field_s = U_rho_cpu * (D_rho_s .* rho_innovations_s)
            else # :cholesky or :cholesky_sparse
                F_rho_cpu = cholesky(Symmetric(Matrix(Q_rho_template_cpu) + noise * I))
                rho_field_unscaled = F_rho_cpu.L' \ rho_innovations_s
                if m.precision_field_model in [:icar, :besag]
                    rho_field_unscaled .-= mean(rho_field_unscaled)
                end
                rho_field_s = rho_field_unscaled .* rho_sigma_s
            end
            rho_s_s = tanh.(rho_field_s)

            latent_st_s = zeros(Float64, s_N, t_N_full)
            innovations_grid_train = reshape(innovations_s, s_N, t_N)
            innovations_grid_full = if t_N_full > t_N
                hcat(innovations_grid_train, randn(Float32, s_N, t_N_full - t_N))
            else
                innovations_grid_train[:, 1:t_N_full]
            end
            
            denom = sqrt.(max.(0, 1 .- rho_s_s.^2))
            latent_st_s[:, 1] = (sigma_s ./ (denom .+ noise)) .* innovations_grid_full[:, 1]
            for t in 2:t_N_full
                latent_st_s[:, t] = rho_s_s .* latent_st_s[:, t-1] .+ sigma_s .* innovations_grid_full[:, t]
            end
            
            linear_indices = s_idx_cpu .+ (t_idx_cpu .- 1) .* s_N
            effect_k_cpu[:, s] = vec(latent_st_s)[linear_indices]
        end
        
        push!(structured_effects, effect_k_cpu)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
 
