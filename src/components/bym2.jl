"""
    BYM2 <: ComponentModel

The Besag-York-Mollié 2 (BYM2) model, which provides an intuitive and well-identified
parameterization for spatial effects by separating them into a structured (ICAR)
and an unstructured (IID) component.

# Version
v1.0.0

# Mathematical Summary
The BYM2 model decomposes a spatial random effect \$\\boldsymbol{\\phi}\$ into two parts:
a spatially structured component \$\\boldsymbol{\\theta}\$ and an unstructured (IID)
  component \$\\boldsymbol{\\epsilon}\$:

\$\\boldsymbol{\\phi} = \\sigma \\left( \\sqrt{\\rho} \\boldsymbol{\\theta}_{scaled} +
  \\sqrt{1 - \\rho} \\boldsymbol{\\epsilon} \\right)\$

where:
- \$\\boldsymbol{\\theta}_{scaled}\$ is a scaled intrinsic CAR (ICAR) process with unit variance.
- \$\\rho \\in [0, 1]\$ is a mixing parameter controlling the proportion of variance
  attributed to the structured spatial effect. It is parameterized on an unconstrained scale
  via `rho_unconstrained`.
- \$\\sigma > 0\$ is the overall marginal standard deviation of the total spatial effect.

# Computational Methods
- `:spectral` (Default, AD-friendly): An efficient, AD-safe method using spectral
  decomposition of the ICAR precision matrix.
- `:cholesky` (AD-friendly): A didactic alternative using dense Cholesky factorization.
- `:cholesky_sparse` (Didactic, Not AD-friendly): A non-AD-safe didactic method using sparse
  Cholesky factorization.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `region`) passed to `random()`.
  - An adjacency matrix `W` passed as a keyword argument to `@tbm`.
- **Optional (in `random()` call)**:
  - `rho_unconstrained`: `UnivariateDistribution`, prior for the unconstrained mixing
    parameter. Default: `Normal(0, 0.5)`.
  - `sigma`: `UnivariateDistribution`, prior for the marginal standard deviation. Default:
    `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:spectral`, `:cholesky`, `:cholesky_sparse`).
    Default: `:spectral`.

# Outputs (Parameter Names)
- `rho_unconstrained_<key>`: The unconstrained mixing parameter.
- `sigma_<key>`: The marginal standard deviation.
- `innovations_<key>`: Standard normal innovations for the spatial field.
- `latent_field_<key>`: Reconstructed structured spatial effect.

# Key References
- Riebler, A., Sørbye, S. H., Simpson, D., & Rue, H. (2016). *An intuitive joint prior for
  variance parameters in hierarchical models*. Statistical Science, 31(1), 114-135.
- Besag, J., York, J., & Mollié, A. (1991). *Bayesian image restoration, with applications
  in spatial statistics*. Annals of the Institute of Statistical Mathematics, 43(1), 1-20.
"""
struct BYM2 <: ComponentModel
    rho_unconstrained::UnivariateDistribution
    sigma::Union{UnivariateDistribution, Real}
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:bym2] = BYM2

COMPONENT_CONSTRUCTORS[:bym2] = (p, params) -> BYM2(
    get(p, :rho_unconstrained, Normal(0, 0.5)),
    p.sigma,
    get(params, :method, :spectral)
)
 
MODEL_TO_STRUCTURE_MAP[:bym2] = :spatial

"""
    get_precomputes(m::BYM2, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-dependent setup for the BYM2 model. This version is CPU-only.
"""
function get_precomputes(m::BYM2, M::NamedTuple, mod_data::Dict)::NamedTuple
    if !hasproperty(M, :W) || !isa(M.W, AbstractMatrix) || isempty(M.W)
        error("BYM2 model requires a valid, non-empty adjacency matrix `W` " *
              "provided via keyword.")
    end

    s_N = size(M.W, 1)

    if !hasproperty(M, :s_idx)
        error("BYM2 component '$(mod_data[:key])' failed: s_idx not found in " *
              "model configuration.")
    end

    template = build_structure_template(:besag, s_N; W=M.W)
    Q_template_scaled = template.matrix
    scaling_factor = template.scaling_factor
    U = template.U
    L_scaled = template.L

    F = cholesky(Symmetric(Matrix(Q_template_scaled) + M.noise * I))

    return (
        Q_template=Q_template_scaled,
        scaling_factor=scaling_factor,
        U=U,
        L=L_scaled,
        n_latent=s_N,
        cholesky_factor=F
    )
end

"""
    _bym2_log_marginal_likelihood(y_residual, s_idx, s_N, U, L_eig, rho, sigma, y_sigma,
      noise=1e-6)

Computes the exact log marginal likelihood for a BYM2 spatial component integrated out
  analytically.
"""
function _bym2_log_marginal_likelihood(
    y_residual::AbstractVector{T},
    s_idx::AbstractVector{Int},
    s_N::Int,
    U::AbstractMatrix,
    L_eig::AbstractVector,
    rho::T,
    sigma::T,
    y_sigma::T,
    noise::Real=1e-6
) where {T}
    N = length(y_residual)
    T_num = promote_type(T, typeof(noise))
    
    # Pre-accumulate observation counts and residual sums per region
    N_s = zeros(T_num, s_N)
    S_s = zeros(T_num, s_N)
    for i in 1:N
        s = s_idx[i]
        if 1 <= s <= s_N
            N_s[s] += one(T_num)
            S_s[s] += y_residual[i]
        end
    end
    
    inv_sigma_y2 = one(T_num) / (y_sigma^2 + T_num(noise))
    scale = sigma^2 + T_num(noise)
    
    # Prior covariance eigenvalues in spectral basis
    max_λ = isempty(L_eig) ? one(T_num) : maximum(abs, L_eig)
    tol = max(T_num(1e-10), T_num(1e-8) * max_λ)
    lambda_rho = Vector{T_num}(undef, s_N)
    for j in 1:s_N
        if L_eig[j] <= tol
            # Null space modes receive unstructured variance
            lambda_rho[j] = (one(T_num) - rho) + T_num(noise)
        else
            lambda_rho[j] = rho / (L_eig[j] + T_num(noise)) +
                            (one(T_num) - rho) + T_num(noise)
        end
    end
    inv_lambda = one(T_num) ./ lambda_rho
    
    # Prior precision matrix Q_prior = U * Diag(inv_lambda) * U'
    Q_prior = Matrix{T_num}(U * Diagonal(inv_lambda) * U')
    
    # Posterior precision matrix Q_post = Q_prior + (scale / y_sigma^2) * Diag(N_s)
    Q_base = copy(Q_prior)
    for s in 1:s_N
        Q_base[s, s] += T_num(noise) + N_s[s] * inv_sigma_y2 * scale
    end
    
    F_post = cholesky(Symmetric(Q_base))
    
    log_det_prior = - sum(log.(lambda_rho)) - s_N * log(scale)
    log_det_diff = log_det_prior - 2 * sum(log.(diag(F_post.U)))
    
    b = S_s .* inv_sigma_y2
    v = F_post.L \ b
    quad_term = scale * dot(v, v)
    
    log_lik = - (N / 2) * log(2 * T_num(pi) * (y_sigma^2 + T_num(noise))) -
              (inv_sigma_y2 / 2) * dot(y_residual, y_residual) +
              (1 / 2) * log_det_diff +
              (1 / 2) * quad_term
              
    return log_lik
end

"""
    get_priors(m::BYM2, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for the BYM2 component's parameters, including the mixing
parameter, overall scale, and innovations for the structured and unstructured parts.
"""
function get_priors(
    m::BYM2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent
    
    if m.method == :marginalized
        return """
        $(_prior_or_constant(p_names.rho_unconstrained, m.rho_unconstrained))
        $(_prior_or_constant(p_names.sigma, m.sigma))
        """
    else
        return """
        $(_prior_or_constant(p_names.rho_unconstrained, m.rho_unconstrained))
        $(_prior_or_constant(p_names.sigma, m.sigma))
        $(p_names.latent_field) ~ MvNormal(zeros(T, $(n_latent)), I)
        $(p_names.innovations) ~ MvNormal(zeros(T, $(n_latent)), I)
        """
    end
end

"""
    get_updates(m::BYM2, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for constructing the BYM2 effect. This version is CPU-only.
"""
function get_updates(
    m::BYM2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, 
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    n_latent = spec.hyper.n_latent

    spectral_code = """
        # --- BYM2 Component (Spectral): $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            rho = logistic($(p_names.rho_unconstrained))
            
            # Construct the diagonal of the spectral transformation matrix D on CPU
            diag_D_cpu = 1.0 ./ sqrt.(hyper.L .+ M.noise)
            # Zero EVERY null direction: a disconnected graph has more than one.
            _zero_null_modes!(diag_D_cpu, hyper.L)
            
            # Apply the spectral transformation: latent = U * D * z
            structured_effect = hyper.U * (diag_D_cpu .* $(p_names.latent_field))
            
            # Combine structured and unstructured components
            local combined_effect = $(p_names.sigma) .* (sqrt(rho) .* structured_effect .+ 
                                sqrt(1.0 - rho) .* $(p_names.innovations))
            
            $(eta_target) = $(eta_target) .+ view(combined_effect, M.s_idx)
        end
    """

    cholesky_code = """
        # --- BYM2 Component (Cholesky, AD-Safe): $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            rho = logistic($(p_names.rho_unconstrained))
            F = hyper.cholesky_factor
            
            latent_field_unscaled = F.L' \\ $(p_names.latent_field)
            Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * $(n_latent)), 
                                       sum(latent_field_unscaled))
            
            local combined_effect = $(p_names.sigma) .* (sqrt(rho) .* latent_field_unscaled .+ 
                                sqrt(1.0 - rho) .* $(p_names.innovations))
            
            $(eta_target) = $(eta_target) .+ view(combined_effect, M.s_idx)
        end
    """

    cholesky_sparse_code = """
        # --- BYM2 Component (Sparse Cholesky, Not AD-Safe): $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            rho = logistic($(p_names.rho_unconstrained))
            Q_penalty = hyper.Q_template
            F = cholesky(Symmetric(Q_penalty + M.noise * I))
            
            latent_field_unscaled = F.L' \\ $(p_names.latent_field)
            Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * $(n_latent)), 
                                       sum(latent_field_unscaled))
            
            local combined_effect = $(p_names.sigma) .* (sqrt(rho) .* latent_field_unscaled .+ 
                                sqrt(1.0 - rho) .* $(p_names.innovations))
            
            $(eta_target) = $(eta_target) .+ view(combined_effect, M.s_idx)
        end
    """

    marginalized_code = """
        # --- BYM2 Component (Marginalized): $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            rho = logistic($(p_names.rho_unconstrained))
            y_residual = M.y_obs .- $(eta_target)
            log_lik_marginalized_$(key) = _bym2_log_marginal_likelihood(
                y_residual,
                M.s_idx,
                hyper.n_latent,
                hyper.U,
                hyper.L,
                rho,
                $(p_names.sigma),
                @isdefined(y_sigma) ? y_sigma : 0.0,
                M.noise
            )
            Turing.@addlogprob! log_lik_marginalized_$(key)
        end
    """

    if m.method == :spectral
        return spectral_code
    elseif m.method == :cholesky
        return cholesky_code
    elseif m.method == :cholesky_sparse
        return cholesky_sparse_code
    elseif m.method == :marginalized
        return marginalized_code
    else; error("Unsupported method '$(m.method)' for BYM2 component. Use :spectral, :cholesky, :cholesky_sparse, or :marginalized."); end
end

"""
    get_effects(m::BYM2, chain, spec::NamedTuple, M::NamedTuple, PS)

Reconstructs the posterior distribution of the BYM2 spatial effect from an MCMC chain.
This version is CPU-only and uses modern chain accessors.
"""
function get_effects(
    m::BYM2, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    n_samples = _get_chain_n_samples(chain)
    outcomes_N = M.outcomes_N
    p_names = string.(keys(chain))
    is_multivariate = outcomes_N > 1
    n_latent = spec.hyper.n_latent
    noise = M.noise

    # Combine spatial indices from training and prediction sets
    s_idx_full = if haskey(M, :s_idx) # Check if spatial index exists in training data
        if !isnothing(PS) && hasproperty(PS.data,
            :s_idx) # If prediction set and it has spatial index
            vcat(M.s_idx, PS.data.s_idx) # Concatenate training and prediction indices
        else
            M.s_idx # Otherwise, use only training indices
        end
    else # If no spatial index in training data, this is an error for BYM2
        error("Spatial index `:s_idx` not found in model configuration for BYM2 component.")
    end
    N_total = length(s_idx_full)

    structured_effects = Vector{Matrix{Float64}}()
    unstructured_effects = Vector{Matrix{Float64}}()
    total_effects = Vector{Matrix{Float64}}()

    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        # Only `sigma` is pinnable: `rho_unconstrained` is `Distribution`-only, so it must
        # come from the chain. A pinned `sigma` is not a chain parameter, so resolve it with
        # the constant fallback rather than reading it by name.
        sigma_samples = _resolve_hyper_samples(
            chain, p_names, p_names_k.sigma, m.sigma, k,
            is_multivariate, n_samples;
            as_matrix = true)
        rho_name = _find_parameter(p_names, string(p_names_k.rho_unconstrained), k,
            is_multivariate)

        if isnothing(sigma_samples) || isempty(rho_name)
            @warn "Parameters for BYM2 component $(spec.key) (outcome $(k)) not resolved, " *
                  "and sigma is not a pinned constant. Returning zero-matrices."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            push!(unstructured_effects, zeros(Float64, N_total, n_samples))
            push!(total_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        rho_samples = logistic.(get_params_vector(chain, rho_name, 1)) # (n_samples, 1)

        structured_latent = zeros(Float64, n_latent, n_samples)
        unstructured_latent = zeros(Float64, n_latent, n_samples)
        
        hyper = spec.hyper

        if m.method == :marginalized
            y_sigma_name = _find_parameter(p_names, "y_sigma", k, is_multivariate)
            y_sigma_samples = if !isempty(y_sigma_name)
                get_params_vector(chain, y_sigma_name, 1)[:, 1]
            else
                fill(1.0, n_samples)
            end
            
            y_vec = M.y_obs isa AbstractMatrix ? M.y_obs[:, k] : M.y_obs
            
            N_s = zeros(Float64, n_latent)
            S_s = zeros(Float64, n_latent)
            for i in 1:length(M.s_idx)
                s = M.s_idx[i]
                if 1 <= s <= n_latent
                    N_s[s] += 1.0
                    S_s[s] += y_vec[i]
                end
            end
            
            U = hyper.U
            L_eig = hyper.L
            
            for i in 1:n_samples # Iterate over each posterior sample
                sig = sigma_samples[i, 1]
                rho_val = rho_samples[i, 1]
                y_sig = y_sigma_samples[i]
                
                scale = sig^2 + noise
                inv_sigma_y2 = 1.0 / (y_sig^2 + noise)
                
                lambda_rho = Vector{Float64}(undef, n_latent)
                lambda_rho[1] = (1.0 - rho_val) + noise
                for j in 2:n_latent
                    lambda_rho[j] = rho_val / (L_eig[j] + noise) + (1.0 - rho_val) + noise
                end
                inv_lambda = 1.0 ./ lambda_rho
                
                Q_prior = Matrix{Float64}(U * Diagonal(inv_lambda) * U')
                Q_base = copy(Q_prior)
                for s in 1:n_latent
                    Q_base[s, s] += noise + N_s[s] * inv_sigma_y2 * scale
                end
                
                F = cholesky(Symmetric(Q_base))
                b = S_s .* inv_sigma_y2
                mu = scale .* (F \ b)
                
                z = randn(n_latent)
                phi = mu .+ sqrt(max(scale, 1e-12)) .* (F.U \ z)
                
                # Decompose into structured and unstructured components via spectral Wiener
                #   filtering
                weights_struct = Vector{Float64}(undef, n_latent)
                weights_struct[1] = 0.0
                for j in 2:n_latent
                    weights_struct[j] = (rho_val / (L_eig[j] + noise)) / lambda_rho[j]
                end
                # Zero every null direction (a disconnected graph has more than one).
                for _ni in _zero_null_modes!(zeros(length(L_eig)), L_eig)
                    weights_struct[_ni] = 0.0
                end
                
                theta = U * (weights_struct .* (U' * phi))
                eps_vec = phi .- theta
                
                structured_latent[:, i] = theta
                unstructured_latent[:, i] = eps_vec
            end
        else
            latent_field_name = _find_parameter(p_names, string(p_names_k.latent_field), k, is_multivariate)
            innovations_field_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate)

            if isempty(latent_field_name) || isempty(innovations_field_name)
                @warn "Innovations (latent_field/innovations) for BYM2 component $(spec.key) (outcome $(k)) not found. Returning zero-matrices."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                push!(unstructured_effects, zeros(Float64, N_total, n_samples))
                push!(total_effects, zeros(Float64, N_total, n_samples))
                continue
            end

            latent_field_innovations_samples = get_params_matrix(chain, latent_field_name, n_latent)
            innovations_samples = get_params_matrix(chain, innovations_field_name, n_latent)

            for i in 1:n_samples # Iterate over each posterior sample
                latent_field_innovations_i = latent_field_innovations_samples[i, :] # Innovations for structured component for current sample
                
                local struct_effect_unscaled # Unscaled structured effect before scaling
                if m.method == :spectral
                    U = hyper.U
                    L = hyper.L
                    diag_D = 1.0 ./ sqrt.(L .+ noise)
                    _zero_null_modes!(diag_D, L)   # zero EVERY null direction; a disconnected graph has more than one
                    struct_effect_unscaled = U * (diag_D .* latent_field_innovations_i)
                else # :cholesky or :cholesky_sparse (use pre-computed dense Cholesky factor)
                    F = hyper.cholesky_factor
                    struct_effect_unscaled = F.L' \ latent_field_innovations_i # Back-solve for unscaled structured effect
                    struct_effect_unscaled .-= mean(struct_effect_unscaled)
                end
                
                structured_latent[:, i] = sigma_samples[i, 1] * sqrt(rho_samples[i,
                    1]) * struct_effect_unscaled
                unstructured_latent[:, i] = sigma_samples[i, 1] * sqrt(1.0 - rho_samples[i,
                    1]) * innovations_samples[i, :]
            end
        end
        
        total_latent = structured_latent .+ unstructured_latent
        
        push!(structured_effects, structured_latent[s_idx_full, :])
        push!(unstructured_effects, unstructured_latent[s_idx_full, :])
        push!(total_effects, total_latent[s_idx_full, :])
    end

    return (structured=structured_effects, unstructured=unstructured_effects, noisy=total_effects)
end
