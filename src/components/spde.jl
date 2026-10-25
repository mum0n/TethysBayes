"""
    SPDE <: ComponentModel

A component model for a spatial field based on the Stochastic Partial Differential
Equation (SPDE) approach. This method provides a direct link between a continuous
Gaussian Process with a Matérn covariance function and a discrete Gaussian Markov
Random Field (GMRF), enabling scalable and principled spatial modeling.

# Version
v1.0.0

# Mathematical Summary
The SPDE approach models a Gaussian Field \$u(s)\$ as the solution to the SPDE:
\$(\\range_value^2 - \\Delta)^{\\alpha/2} u(s) = \\mathcal{W}(s)\$
where:
- \$\\Delta\$ is the Laplacian operator.
- \$\\range_value > 0\$ controls the spatial range of the process.
- \$\\alpha\$ controls the smoothness of the process.
- \$\\mathcal{W}(s)\$ is Gaussian white noise.

For a discrete spatial domain represented by a graph, the Laplacian \$\\Delta\$ is
approximated by the graph Laplacian \$\\mathbf{Q}_{ICAR} = D - W\$. For the common case
where \$\\alpha = 2\$ (which corresponds to a Matérn field with smoothness \$\\nu=1\$),
the precision matrix \$\\mathbf{Q}\$ of the latent field \$\\boldsymbol{\\phi}\$ is given by:
\$\\mathbf{Q} = (\\range_value^2 \\mathbf{I} + \\mathbf{Q}_{ICAR})^T (\\range_value^2 \\mathbf{I} +
  \\mathbf{Q}_{ICAR})\$
The model then samples the latent field from \$\\boldsymbol{\\phi} \\sim \\mathcal{N}(0,
  (\\sigma^2 \\mathbf{Q})^{-1})\$.

# Computational Methods
- `:spectral` (Default, AD-friendly): An efficient, AD-safe method using spectral decomposition.
  Only applicable for isotropic `range` priors.
- `:cholesky` (AD-friendly): An AD-safe didactic alternative using dense Cholesky factorization.
- `:cholesky_sparse` (Didactic, Not AD-friendly): A non-AD-safe didactic method using sparse
  Cholesky
  factorization, suitable for gradient-free samplers.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `region`) passed to `random()`.
  - An adjacency matrix `W` passed as a keyword argument to `@tbm`.
- **Optional (in `random()` call)**:
  - `sigma`: `UnivariateDistribution`, prior for the marginal standard deviation. Default:
    `Exponential(1.0)`.
  - `range`: `UnivariateDistribution` or `Vector{<:UnivariateDistribution}`, prior for the `range`
    parameter(s). Default: `LogNormal(0, 1)`.
  - `method`: `Symbol`, computational method (`:spectral`, `:cholesky`, `:cholesky_sparse`).
    Default: `:spectral`.

# Outputs (Parameter Names)
- `sigma_<key>`: The marginal standard deviation of the SPDE effect.
- `range_<key>`: The spatial range parameter(s).
- `innovations_<key>`: The raw standard normal innovations for the latent field.
- `latent_<key>`: The reconstructed latent SPDE effect.

# Key References
- Lindgren, F., Rue, H., & Lindström, J. (2011). *An explicit link between
  Gaussian fields and Gaussian Markov random fields: The SPDE approach*. Journal
  of the Royal Statistical Society: Series B (Statistical Methodology), 73(4), 423-498.
"""
struct SPDE <: ComponentModel
    sigma::Union{Distribution, Real}
    range::Union{Distribution, Vector{<:Distribution}}
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:spde] = SPDE
COMPONENT_CONSTRUCTORS[:spde] = (p, params) -> SPDE(
    p.sigma, p.range, get(params, :method, :spectral)
)

MODEL_TO_STRUCTURE_MAP[:spde] = :spatial

function get_precomputes(m::SPDE, M::NamedTuple, mod_data::Dict)::NamedTuple
    # Data validation
    if !hasproperty(M, :W) || !isa(M.W, AbstractMatrix) || isempty(M.W)
        error("SPDE model requires a valid, non-empty adjacency matrix `W` provided via keyword.")
    end

    s_N = size(M.W, 1)

    if !hasproperty(M, :s_idx)
        error(
            "SPDE component '$(mod_data[:key])' failed: s_idx not found in model " *
            "configuration. This should have been set by the model processor."
        )
    end

    W = M.W
    W_sym = sparse((W + W') .> 0)
    D = spdiagm(0 => vec(sum(W_sym, dims=2)))
    Q_template_cpu = D - W_sym

    # Perform eigen decomposition on CPU
    eig_decomp = eigen(Symmetric(Matrix(Q_template_cpu)))
    U_cpu = eig_decomp.vectors
    L_cpu = eig_decomp.values
    scaling_factor = _compute_scaling_factor(L_cpu, 1)
    
    Q_template_scaled_cpu = Q_template_cpu ./ scaling_factor
    L_scaled_cpu = L_cpu ./ scaling_factor

    return (
        Q_template=Q_template_scaled_cpu,
        scaling_factor=scaling_factor,
        U=U_cpu,
        L=L_scaled_cpu,
        n_latent=s_N
    )
end


"""
    _spde_log_marginal_likelihood(y_residual, s_idx, s_N, Q_laplacian, L_eig, range, sigma,
      y_sigma, noise=1e-6)

Computes the exact log marginal likelihood for an SPDE spatial component integrated out
  analytically.
"""
function _spde_log_marginal_likelihood(
    y_residual::AbstractVector{T},
    s_idx::AbstractVector{Int},
    s_N::Int,
    Q_laplacian::AbstractMatrix,
    L_eig::AbstractVector,
    range::Real,
    sigma::T,
    y_sigma::T,
    noise::Real=1e-6
) where {T}
    N = length(y_residual)
    T_num = promote_type(T, typeof(noise), typeof(range))
    
    inv_sigma_y2 = one(T_num) / (y_sigma^2 + T_num(noise))
    scale = sigma^2 + T_num(noise)
    
    N_s = zeros(T_num, s_N)
    S_s = zeros(T_num, s_N)
    for i in 1:N
        s = s_idx[i]
        if 1 <= s <= s_N
            N_s[s] += one(T_num)
            S_s[s] += y_residual[i]
        end
    end
    
    L_op = Matrix{T_num}(Q_laplacian) + (T_num(range)^2) * Matrix{T_num}(I, s_N, s_N)
    Q_spde = L_op' * L_op
    
    Q_base = Matrix{T_num}(Q_spde)
    for s in 1:s_N
        Q_base[s, s] += T_num(noise) + N_s[s] * inv_sigma_y2 * scale
    end
    
    F = cholesky(Symmetric(Q_base))
    
    log_det_prior = 2 * sum(log.(T_num(range)^2 .+ L_eig .+ T_num(noise)))
    log_det_diff = - s_N * log(scale) + log_det_prior - 2 * sum(log.(diag(F.U)))
    
    b = S_s .* inv_sigma_y2
    v = F.L \ b
    quad_term = scale * dot(v, v)
    
    log_lik = - (N / 2) * log(2 * T_num(pi) * (y_sigma^2 + T_num(noise))) -
              (inv_sigma_y2 / 2) * dot(y_residual, y_residual) +
              (1 / 2) * log_det_diff +
              (1 / 2) * quad_term
              
    return log_lik
end

function get_priors(
    m::SPDE, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    key = spec.key
    
    priors = String[]
    push!(priors, "$(_prior_or_constant(p_names.sigma, m.sigma))")

    if m.range isa Vector
        range_priors_str = join([_distribution_to_string(p) for p in m.range], ", ")
        push!(priors, "$(p_names.range) ~ Product([$(range_priors_str)])")
    else
        range_prior_str = _distribution_to_string(m.range)
        push!(priors, "$(p_names.range) ~ $(range_prior_str)")
    end
    
    if m.method != :marginalized
        push!(
            priors,
            "$(p_names.innovations) ~ MvNormal(zeros(T, spec_registry[:$(key)].hyper.n_latent), I)"
        )
    end

    return join(priors, "\n    ")
end

function get_updates(
    m::SPDE, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key

    use_spectral = m.method == :spectral && !(m.range isa Vector)

    spectral_code = """
        # --- SPDE Component (Spectral): $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            U = hyper.U
            L = hyper.L
            
            diag_vals = ($(p_names.range)^2 .+ L).^2
            diag_D = $(p_names.sigma) ./ sqrt.(diag_vals .+ M.noise)
            
            $(p_names.latent_field) = U * (diag_D .* $(p_names.innovations))
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.s_idx)
        end
    """

    cholesky_base_code = """
        local hyper = spec_registry[:$(key)].hyper
        local Q_laplacian = hyper.Q_template
        local range_val = $(p_names.range)
        local Q_range_term = if range_val isa AbstractVector
            Diagonal(range_val.^2)
        else
            range_val^2 * I
        end
        local L_operator = Q_range_term + Q_laplacian
        local Q_final = Symmetric(L_operator' * L_operator)
    """

    cholesky_code = """
        # --- SPDE Component (Cholesky, AD-Safe): $(key) ---
        let
            $(cholesky_base_code)
            local F = cholesky(Matrix(Q_final) + M.noise * I)
            $(p_names.latent_field) = $(p_names.sigma) .* (F.L' \\ $(p_names.innovations))
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.s_idx)
        end
    """

    cholesky_sparse_code = """
        # --- SPDE Component (Sparse Cholesky, Not AD-Safe): $(key) ---
        let
            $(cholesky_base_code)
            local F = cholesky(Q_final + M.noise * I)
            $(p_names.latent_field) = $(p_names.sigma) .* (F.L' \\ $(p_names.innovations))
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.s_idx)
        end
    """

    marginalized_code = """
        # --- SPDE Component (Marginalized): $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            y_residual = M.y_obs .- $(eta_target)
            log_lik_marginalized_$(key) = _spde_log_marginal_likelihood(
                y_residual,
                M.s_idx,
                hyper.n_latent,
                hyper.Q_template,
                hyper.L,
                $(p_names.range),
                $(p_names.sigma),
                @isdefined(y_sigma) ? y_sigma : 0.0,
                M.noise
            )
            Turing.@addlogprob! log_lik_marginalized_$(key)
        end
    """

    if m.method == :marginalized
        return marginalized_code
    elseif use_spectral
        return spectral_code
    elseif m.method == :cholesky
        return cholesky_code
    elseif m.method == :cholesky_sparse
        return cholesky_sparse_code
    else
        @warn "SPDE method '$(m.method)' with anisotropic range is not supported by spectral method. Falling back to dense Cholesky."
        return cholesky_code
    end
end

"""
    get_effects(m::SPDE, chain, spec::NamedTuple, M::NamedTuple, PS)

Reconstructs the `SPDE` component's effect from posterior samples. This version is
CPU-only and uses modern chain accessors.
"""
function get_effects(
    m::SPDE, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions ---
    n_samples = _get_chain_n_samples(chain)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = string.(keys(chain))
    
    # --- Get precomputed data (all on CPU) ---
    hyper = spec.hyper
    noise = M.noise
    n_latent = hyper.n_latent
    Q_laplacian_cpu = hyper.Q_template
    U_cpu = hyper.U
    L_cpu = hyper.L

    # --- Index Handling: Combine training and prediction sets on CPU ---
    s_idx_train = M.s_idx
    s_idx_full = if !isnothing(PS) && hasproperty(PS.data, :s_idx)
        vcat(s_idx_train, PS.data.s_idx)
    else
        s_idx_train
    end
    N_total = length(s_idx_full)

    # Determine if spectral method can be used (requires isotropic range)
    use_spectral = m.method == :spectral && !(m.range isa Vector)
    
    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        v = generate_full_variable_names(spec, M.model_arch, k)
        range_name = _find_parameter(p_names, string(v.range), k, is_multivariate_model)

        # Only `sigma` is pinnable here: `range` is `Union{Distribution, Vector{<:Distribution}}`,
        # so a fixed range is not representable and it must come from the chain.
        sigma_samples_cpu = _resolve_hyper_samples(
            chain, p_names, v.sigma, m.sigma, k, is_multivariate_model, n_samples)

        if isnothing(sigma_samples_cpu) || isempty(range_name)
            @warn "Parameters for SPDE component $(spec.key) (outcome $k) not resolved, " *
                  "and sigma is not a pinned constant. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        # Extract posterior samples (CPU)
        range_dim = m.range isa Vector ? length(m.range) : 1
        range_samples_cpu = get_params_matrix(chain, range_name, range_dim)

        # Initialize the output matrix for the full latent field on the CPU
        latent_field_matrix = zeros(Float64, n_latent, n_samples)

        if m.method == :marginalized
            y_sigma_name = _find_parameter(p_names, "y_sigma", k, is_multivariate_model)
            y_sigma_samples = if !isempty(y_sigma_name)
                get_params_vector(chain, y_sigma_name, 1)[:, 1]
            else
                fill(1.0, n_samples)
            end
            
            y_vec = M.y_obs isa AbstractMatrix ? M.y_obs[:, k] : M.y_obs
            
            N_s = zeros(Float64, n_latent)
            S_s = zeros(Float64, n_latent)
            for i in 1:length(s_idx_train)
                s = s_idx_train[i]
                if 1 <= s <= n_latent
                    N_s[s] += 1.0
                    S_s[s] += y_vec[i]
                end
            end
            
            for i in 1:n_samples
                sig = sigma_samples_cpu[i]
                kap = range_samples_cpu[i, 1]
                y_sig = y_sigma_samples[i]
                
                scale = sig^2 + noise
                inv_sigma_y2 = 1.0 / (y_sig^2 + noise)
                
                L_op = Matrix{Float64}(Q_laplacian_cpu) + (kap^2) * Matrix{Float64}(I,
                    n_latent, n_latent)
                Q_spde = L_op' * L_op
                
                Q_base = Matrix{Float64}(Q_spde)
                for s in 1:n_latent
                    Q_base[s, s] += noise + N_s[s] * inv_sigma_y2 * scale
                end
                
                F = cholesky(Symmetric(Q_base))
                b = S_s .* inv_sigma_y2
                mu = scale .* (F \ b)
                
                z = randn(n_latent)
                latent_field_matrix[:, i] = mu .+ sqrt(max(scale, 1e-12)) .* (F.U \ z)
            end
        else
            innovations_name = _find_parameter(p_names, string(v.innovations), k, is_multivariate_model)
            innovations_samples_cpu = get_params_matrix(chain, innovations_name, n_latent)

            for i in 1:n_samples
                current_sigma = sigma_samples_cpu[i]
                current_range = range_samples_cpu[i, :]
                current_innovations = innovations_samples_cpu[i, :]
                
                local latent_field_sample
                if use_spectral
                    range_val = current_range[1]
                    diag_vals = (range_val^2 .+ L_cpu).^2
                    diag_D = current_sigma ./ sqrt.(diag_vals .+ noise)
                    latent_field_sample = U_cpu * (diag_D .* current_innovations)
                else
                    Q_range_term = if m.range isa Vector
                        Diagonal(current_range.^2)
                    else
                        current_range[1]^2 * I
                    end
                    
                    L_operator = Q_range_term + Q_laplacian_cpu
                    Q_final = Symmetric(L_operator' * L_operator)
                    
                    F = cholesky(Matrix(Q_final) + noise * I)
                    latent_field_sample = current_sigma .* (F.L' \ current_innovations)
                end
                latent_field_matrix[:, i] = latent_field_sample
            end
        end
        
        # Index the reconstructed effects for the full observation set
        indexed_effects = latent_field_matrix[s_idx_full, :]
        push!(structured_effects, indexed_effects)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
 
