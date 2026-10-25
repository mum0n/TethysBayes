"""
    SAR <: ComponentModel

A component model for the Simultaneous Autoregressive (SAR) effect, also known as a
proper CAR model. The value at each location is modeled as a linear combination of
its neighbors plus an independent innovation term, leading to a precision matrix of
the form `(I - ρW)'(I - ρW)`.

# Version
v1.0.0

# Mathematical Summary
The Simultaneous Autoregressive (SAR) model defines a spatial random effect
\$\\boldsymbol{\\phi}\$ where the value at each location is a linear combination of its
neighbors plus an independent innovation term. The model is typically expressed as:
\$\\boldsymbol{\\phi} = \\rho \\mathbf{W} \\boldsymbol{\\phi} + \\boldsymbol{\\epsilon}\$
where:
- \$\\rho\$ is the spatial autoregressive parameter.
- \$\\mathbf{W}\$ is a row-standardized adjacency matrix.
- \$\\boldsymbol{\\epsilon} \\sim \\mathcal{N}(\\mathbf{0}, \\sigma^2 \\mathbf{I})\$ are
  independent innovations.

The precision matrix \$\\mathbf{Q}\$ for the SAR model is then given by:
\$\\mathbf{Q} = \\frac{1}{\\sigma^2} (\\mathbf{I} - \\rho \\mathbf{W})^T (\\mathbf{I} -
  \\rho \\mathbf{W})\$

# Computational Methods
- `:cholesky` (Default, AD-friendly): An AD-safe method using dense Cholesky factorization.
- `:cholesky_sparse` (Didactic, Not AD-friendly): A more memory-efficient method using
  sparse Cholesky factorization, suitable for gradient-free samplers.

**Note on Spectral Method**: A direct spectral method (using pre-computed eigenvectors
and eigenvalues) is not provided for the SAR model because its precision matrix
depends on the sampled parameter `rho` in a way that prevents pre-computation of
its spectral decomposition.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `region`) passed to `random()`.
  - An adjacency matrix `W` passed as a keyword argument to `@tbm`.
- **Optional (in `random()` call)**:
  - `rho`: `UnivariateDistribution`, prior for the spatial autoregressive parameter.
    Default: `Normal(0, 0.5)`. Should be constrained to ensure stationarity.
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation of the innovations.
    Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:cholesky` or `:cholesky_sparse`).
    Default: `:cholesky`.

# Outputs (Parameter Names)
- `rho_<key>`: The spatial autoregressive parameter.
- `sigma_<key>`: The standard deviation of the innovations.
- `innovations_<key>`: The raw standard normal innovations for the latent field.
- `latent_<key>`: The reconstructed latent SAR effect.

# Key References
- Cliff, A. D., & Ord, J. K. (1973). *Spatial Autocorrelation*. Pion.
- Wikipedia: Simultaneous autoregressive model
"""
struct SAR <: ComponentModel
    rho::Distribution
    sigma::Union{Distribution, Real}
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:sar] = SAR
COMPONENT_CONSTRUCTORS[:sar] = (p, params) -> SAR(
    p.rho, p.sigma, get(params, :method, :cholesky)
)

MODEL_TO_STRUCTURE_MAP[:sar] = :spatial

function get_precomputes(m::SAR, M::NamedTuple, mod_data::Dict)::NamedTuple
    # Data validation moved from get_datastructures!
    if !hasproperty(M, :W)
        error("SAR model requires an adjacency matrix `W` to be provided via keyword.")
    end

    if !isa(M.W, AbstractMatrix) || isempty(M.W)
        error("Provided `W` for SAR model is not a valid non-empty matrix.")
    end

    s_N = size(M.W, 1)

    # The processor is now responsible for creating s_idx.
    if !hasproperty(M, :s_idx)
        error(
            "SAR component '$(mod_data[:key])' failed: s_idx not found in model " *
            "configuration. This should have been set by the model processor."
        )
    end

    W = sparse(M.W)
    row_sums = sum(W, dims=2)
    non_zero_rows = findall(x -> x > 0, vec(row_sums))
    
    W_std = spzeros(Float64, s_N, s_N)
    if !isempty(non_zero_rows)
        D_inv_vals = 1.0 ./ row_sums[non_zero_rows]
        D_inv = spdiagm(0 => vec(D_inv_vals))
        W_std[non_zero_rows, :] = D_inv * W[non_zero_rows, :]
    end
    return (Q_template=W_std, eigenvalues=eigvals(Matrix(W_std)), n_latent=s_N)
end

"""
    _sar_log_marginal_likelihood(y_residual, s_idx, s_N, W_std, eigenvalues, rho, sigma,
      y_sigma, noise=1e-6)

Computes the exact log marginal likelihood for a SAR spatial process integrated out analytically.
"""
function _sar_log_marginal_likelihood(
    y_residual::AbstractVector{T},
    s_idx::AbstractVector{Int},
    s_N::Int,
    W_std::AbstractMatrix,
    eigenvalues::AbstractVector,
    rho::T,
    sigma::T,
    y_sigma::T,
    noise::Real=1e-6
) where {T}
    N = length(y_residual)
    T_num = promote_type(T, typeof(noise))
    
    # Pre-accumulate observation counts and sums per spatial index
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
    
    I_mat = Matrix{T_num}(I, s_N, s_N)
    B_op = I_mat .- rho .* Matrix{T_num}(W_std)
    Q_prior_unscaled = Matrix{T_num}(B_op' * B_op)
    
    Q_base = copy(Q_prior_unscaled)
    for s in 1:s_N
        Q_base[s, s] += T_num(noise) + N_s[s] * inv_sigma_y2 * scale
    end
    
    F = cholesky(Symmetric(Q_base))
    
    # Determinant term using eigenvalues of W_std
    log_det_prior = 2 * sum(log.(abs.(1.0 .- rho .* real.(eigenvalues)) .+ T_num(noise)))
    log_det_diff = log_det_prior - 2 * sum(log.(diag(F.U)))
    
    # Quadratic term
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
    m::SAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    key = spec.key

    rho_prior_str = _distribution_to_string(m.rho)
    sigma_prior_str = _prior_or_constant(p_names.sigma, m.sigma)
    if m.method == :marginalized
        return """
            $(p_names.rho) ~ $(rho_prior_str)
            $(sigma_prior_str)
        """
    else
        return """
            $(p_names.rho) ~ $(rho_prior_str)
            $(sigma_prior_str)
            $(p_names.innovations) ~ MvNormal(
                zeros(T, spec_registry[:$(key)].hyper.n_latent), I
            )
        """
    end
end

function get_updates(
    m::SAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key

    common_code = """
        local W_std = spec_registry[:$(key)].hyper.Q_template
        local L_op = I - $(p_names.rho) * W_std
    """

    cholesky_code = """
        # --- SAR Component (Direct Autoregressive Solve, AD-Safe): $(key) ---
        let
            $(common_code)
            $(p_names.latent_field) = L_op \\ ($(p_names.sigma) .* $(p_names.innovations))
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.s_idx)
        end
    """

    cholesky_sparse_code = """
        # --- SAR Component (Direct Autoregressive Solve, Sparse): $(key) ---
        let
            $(common_code)
            $(p_names.latent_field) = L_op \\ ($(p_names.sigma) .* $(p_names.innovations))
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.s_idx)
        end
    """

    marginalized_code = """
        # --- SAR Component (Marginalized): $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            y_residual = M.y_obs .- $(eta_target)
            log_lik_marginalized_$(key) = _sar_log_marginal_likelihood(
                y_residual,
                M.s_idx,
                hyper.n_latent,
                hyper.Q_template,
                hyper.eigenvalues,
                $(p_names.rho),
                $(p_names.sigma),
                @isdefined(y_sigma) ? y_sigma : 0.0,
                M.noise
            )
            Turing.@addlogprob! log_lik_marginalized_$(key)
        end
    """

    if m.method == :cholesky
        return cholesky_code
    elseif m.method == :cholesky_sparse
        return cholesky_sparse_code
    elseif m.method == :marginalized
        return marginalized_code
    else
        error("Unsupported method '$(m.method)' for SAR component. Use :cholesky, :cholesky_sparse, or :marginalized.")
    end
end


"""
    get_effects(m::SAR, chain, spec, M, PS)

Reconstructs the SAR effect from posterior samples. 
"""
function get_effects(
    m::SAR, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions ---
    n_samples = _get_chain_n_samples(chain)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = string.(keys(chain))
    
    noise = M.noise
    
    n_latent = spec.hyper.n_latent
    W_dag = spec.hyper.Q_template

    # --- Coordinate/Index Handling: Combine training and prediction sets ---
    s_idx_train = M.s_idx
    s_idx_full = if !isnothing(PS) && hasproperty(PS.data, :s_idx)
        vcat(s_idx_train, PS.data.s_idx)
    else
        s_idx_train
    end
    N_total = length(s_idx_full)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        
        rho_name = _find_parameter(p_names, string(p_names_k.rho), k, is_multivariate_model)
        # Only `sigma` is pinnable here: `rho` is `Distribution`-only, so it must come from
        # the chain. A pinned `sigma` is not a chain parameter, so resolve it with the
        # constant fallback rather than reading it by name.
        sigma_samples = _resolve_hyper_samples(
            chain, p_names, p_names_k.sigma, m.sigma, k, is_multivariate_model, n_samples)

        if isempty(rho_name) || isnothing(sigma_samples)
            @warn "Parameters for SAR component $(spec.key) (outcome $k) not resolved, " *
                  "and sigma is not a pinned constant. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        rho_samples = get_params_vector(chain, rho_name, 1)[:, 1]

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
            
            I_mat = Matrix{Float64}(I, n_latent, n_latent)
            
            for s in 1:n_samples
                rho_val = rho_samples[s]
                sig = sigma_samples[s]
                y_sig = y_sigma_samples[s]
                
                scale = sig^2 + noise
                inv_sigma_y2 = 1.0 / (y_sig^2 + noise)
                
                B_op = I_mat .- rho_val .* Matrix{Float64}(W_dag)
                Q_base = Matrix{Float64}(B_op' * B_op)
                for i in 1:n_latent
                    Q_base[i, i] += noise + N_s[i] * inv_sigma_y2 * scale
                end
                
                F = cholesky(Symmetric(Q_base))
                b = S_s .* inv_sigma_y2
                mu = scale .* (F \ b)
                
                z = randn(n_latent)
                latent_field_matrix[:, s] = mu .+ sqrt(max(scale, 1e-12)) .* (F.U \ z)
            end
        else
            innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)
            innovations_samples = get_params_matrix(chain, innovations_name, n_latent)

            for s in 1:n_samples
                rho = rho_samples[s]
                sigma = sigma_samples[s]
                innovations = innovations_samples[s, :]

                L_op = I - rho * W_dag
                Q_sar = Symmetric(Matrix(L_op' * L_op) / (sigma^2) + noise * I)
                F = cholesky(Q_sar)
                latent_field_matrix[:, s] = F.L' \ innovations
            end
        end
        # Index the reconstructed latent effects to match the observation indices
        indexed_effects = latent_field_matrix[s_idx_full, :]
        push!(structured_effects, indexed_effects)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
 
