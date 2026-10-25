"""
    ICAR <: ComponentModel

A component for an Intrinsic Conditional Autoregressive (ICAR) model, also known as
a Besag model. This is a fundamental model for spatial data on a lattice or graph,
where the value at a location is assumed to be conditionally dependent on the
average of its neighbors.

# Version
v1.0.0

# Mathematical Summary
The ICAR model defines a Gaussian Markov Random Field (GMRF) with a singular
precision matrix (the graph Laplacian), making it an "intrinsic" GMRF. The
conditional distribution of the spatial effect \$\\phi_i\$ at location \$i\$, given all
other locations, is:
\$\\phi_i | \\phi_{j \\ne i} \\sim \\mathcal{N}\\left( \\frac{1}{d_i} \\sum_{j \\sim i}
  \\phi_j, \\frac{\\sigma^2}{d_i} \\right)\$
where \$j \\sim i\$ denotes that \$j\$ is a neighbor of \$i\$, and \$d_i\$ is the number of
neighbors.

The joint precision matrix is the graph Laplacian, \$Q = D - W\$, where \$D\$ is the
diagonal degree matrix and \$W\$ is the adjacency matrix. Because \$Q\$ is
rank-deficient (its rows sum to zero), a sum-to-zero constraint
(\$\\sum_i \\phi_i = 0\$) is imposed on the latent field to ensure identifiability
from the global intercept.

# Computational Methods
- `:spectral` (Default, AD-friendly): Regularizes coefficients using a spectral
  decomposition of the precision matrix. Recommended for gradient-based samplers.
- `:cholesky` (AD-friendly): Uses a pre-computed dense Cholesky factorization.
- `:cholesky_sparse` (Didactic, Not AD-friendly): Uses sparse Cholesky factorization,
  which is not compatible with most AD backends.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `s_idx`) passed to `random()`.
  - An adjacency matrix `W` passed as a keyword argument to `@tbm`.
- **Optional (in `random()` call)**:
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation of the
    ICAR effect. Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:spectral`, `:cholesky`, `:cholesky_sparse`).
    Default: `:spectral`.

# Outputs (Parameter Names)
- `sigma_<key>`: The standard deviation of the ICAR effect.
- `innovations_<key>`: The raw standard normal innovations for the effect.
- `latent_<key>`: The reconstructed latent spatial field.

# Key References
- Besag, J. (1974). Spatial interaction and the statistical analysis of lattice
  systems. *Journal of the Royal Statistical Society: Series B (Methodological)*,
  36(2), 192-225.
- Rue, H., & Held, L. (2005). *Gaussian Markov Random Fields: Theory and
  Applications*. CRC Press.
- Wikipedia: Conditional autoregressive model
"""
struct ICAR <: ComponentModel
    sigma::Union{Distribution, Real}
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:icar] = ICAR
COMPONENT_CONSTRUCTORS[:icar] = (p, params) -> ICAR(
    p.sigma, get(params, :method, :spectral)
)

MODEL_TO_STRUCTURE_MAP[:icar] = :spatial

"""
    get_precomputes(m::ICAR, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes the graph Laplacian (`Q_template`), its Cholesky factorization, and its
spectral decomposition (`U`, `L`) for use by different sampling methods. This is a
CPU-only implementation.
"""
function get_precomputes(m::ICAR, M::NamedTuple, mod_data::Dict)::NamedTuple
    n = (hasproperty(M, :W) && !isempty(M.W) && isa(M.W, AbstractMatrix)) ? size(M.W, 1) : M.s_N
    W = M.W
    
    # build_structure_template returns CPU arrays
    template = build_structure_template(:icar, n; W=W)
    
    # Pre-compute the dense Cholesky factor for the :cholesky method on the CPU.
    F_cpu = cholesky(Symmetric(Matrix(template.matrix) + M.noise * I))
    
    return (
        Q_template=template.matrix, 
        scaling_factor=template.scaling_factor, 
        U=template.U, 
        L=template.L, 
        n_latent=n, 
        cholesky_factor=F_cpu
    )
end

"""
    _icar_log_marginal_likelihood(y_residual, s_idx, s_N, Q_template, L_eig, sigma, y_sigma,
      noise=1e-6)

Computes the exact log marginal likelihood for an ICAR spatial process integrated out analytically.
"""
function _icar_log_marginal_likelihood(
    y_residual::AbstractVector{T},
    s_idx::AbstractVector{Int},
    s_N::Int,
    Q_template::AbstractMatrix,
    L_eig::AbstractVector,
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
    
    Q_base = Matrix{T_num}(Q_template)
    for s in 1:s_N
        Q_base[s, s] += T_num(noise) + N_s[s] * inv_sigma_y2 * scale
    end
    
    F = cholesky(Symmetric(Q_base))
    
    # Generalized log-determinant over non-zero eigenvalues of spatial precision
    max_λ = isempty(L_eig) ? one(T_num) : maximum(abs, L_eig)
    tol = max(T_num(1e-10), T_num(1e-8) * max_λ)
    pos_evals = filter(λ -> λ > tol, L_eig)
    rank_eff = length(pos_evals)
    log_det_prior = isempty(pos_evals) ? zero(T_num) : sum(log.(pos_evals))
    log_det_diff = - rank_eff * log(scale) + log_det_prior - 2 * sum(log.(diag(F.U)))
    
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
    m::ICAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent
    if m.method == :marginalized
        return "$(_prior_or_constant(p_names.sigma, m.sigma))"
    else
        return """ # Priors for sigma and raw innovations
        $(_prior_or_constant(p_names.sigma, m.sigma))
        $(p_names.innovations) ~ MvNormal(zeros(T, $(n_latent)), I)
        """
    end
end

function get_updates(
    m::ICAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    n_latent = spec.hyper.n_latent

    spectral_code = """
        # --- ICAR Component: $(key) (Spectral Method) ---
        let
            hyper = spec_registry[:$(key)].hyper
            U = hyper.U
            L = hyper.L
            diag_D = $(p_names.sigma) ./ sqrt.(L .+ M.noise)
            diag_D[L .<= (1e-10 * maximum(L))] .= 0.0   # zero EVERY null direction (a disconnected graph has more than one)
            $(p_names.latent_field) = U * (diag_D .* $(p_names.innovations))
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.s_idx)
        end
    """

    cholesky_code = """
        # --- ICAR Component: $(key) (Cholesky Method, AD-Safe) ---
        let
            hyper = spec_registry[:$(key)].hyper
            F = hyper.cholesky_factor
            latent_field_unscaled = F.L' \\ $(p_names.innovations)
            
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * $(n_latent)), sum(latent_field_unscaled)
            )
            
            $(p_names.latent_field) = latent_field_unscaled .* $(p_names.sigma)
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.s_idx)
        end
    """

    cholesky_sparse_code = """
        # --- ICAR Component: $(key) (Sparse Cholesky, Not AD-Safe): $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            Q = hyper.Q_template
            F = cholesky(Symmetric(Q + M.noise * I))
            latent_field_unscaled = F.L' \\ $(p_names.innovations)
            
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * $(n_latent)), sum(latent_field_unscaled)
            )
            
            $(p_names.latent_field) = latent_field_unscaled .* $(p_names.sigma)
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.s_idx)
        end
    """

    marginalized_code = """
        # --- ICAR Component: $(key) (Marginalized Method) ---
        let
            hyper = spec_registry[:$(key)].hyper
            y_residual = M.y_obs .- $(eta_target)
            log_lik_marginalized_$(key) = _icar_log_marginal_likelihood(
                y_residual,
                M.s_idx,
                hyper.n_latent,
                hyper.Q_template,
                hyper.L,
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
    else
        error("Unsupported method '$(m.method)' for ICAR component. Use :spectral, :cholesky, :cholesky_sparse, or :marginalized.")
    end
end

"""
    get_effects(m::ICAR, chain, spec::NamedTuple, M::NamedTuple, PS)

Reconstructs the `ICAR` component's effect from posterior samples, applying a
sum-to-zero constraint for identifiability. This function dispatches on the method
used during sampling and is CPU-only.
"""
function get_effects(
    m::ICAR, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    n_samples = _get_chain_n_samples(chain)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = string.(keys(chain))
    
    noise = M.noise
    n_latent = spec.hyper.n_latent

    # --- Coordinate/Index Handling: Combine training and prediction sets on CPU ---
    s_idx_train = M.s_idx
    s_idx_full, N_total = _resolve_effect_indices(M, PS, :s_idx)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        
        # A pinned `sigma` is not a chain parameter, so `_find_parameter` cannot find it.
        # Fall back to the declared constant rather than returning a zero field.
        sigma_samples = _resolve_hyper_samples(
            chain, p_names, p_names_k.sigma, m.sigma, k, is_multivariate_model, n_samples)

        if isnothing(sigma_samples)
            @warn "Parameters for ICAR component $(spec.key) (outcome $k) " *
                  "not found, and sigma is not a pinned constant. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        sigma_samples = reshape(sigma_samples, n_samples, 1)

        # Initialize the output matrix for latent effects
        effect_k_latent = zeros(Float64, n_latent, n_samples)

        # --- Sample-wise Reconstruction ---
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
            
            Q_template = spec.hyper.Q_template
            
            for j in 1:n_samples
                sigma_j = sigma_samples[j, 1]
                y_sig = y_sigma_samples[j]
                
                scale = sigma_j^2 + noise
                inv_sigma_y2 = 1.0 / (y_sig^2 + noise)
                
                Q_base = Matrix{Float64}(Q_template)
                for i in 1:n_latent
                    Q_base[i, i] += noise + N_s[i] * inv_sigma_y2 * scale
                end
                
                F = cholesky(Symmetric(Q_base))
                b = S_s .* inv_sigma_y2
                mu = scale .* (F \ b)
                
                z = randn(n_latent)
                x_train = mu .+ sqrt(max(scale, 1e-12)) .* (F.U \ z)
                x_train .-= mean(x_train)
                effect_k_latent[:, j] = x_train
            end
        else
            innovations_name = _find_parameter(
                p_names, string(p_names_k.innovations), k, is_multivariate_model
            )
            if isempty(innovations_name)
                @warn "innovations for ICAR component $(spec.key) (outcome $k) not found. " *
                      "Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end
            innovations_samples = get_params_matrix(chain, innovations_name, n_latent)

            if m.method == :spectral
                U = spec.hyper.U
                L = spec.hyper.L
                inv_sqrt_L = 1.0 ./ sqrt.(L .+ noise)
                # Zero EVERY null direction (a disconnected graph has more than one).
                _zero_null_modes!(inv_sqrt_L, L)
                # Vectorized Level-3 BLAS across all MCMC draws
                scaled_innov = (inv_sqrt_L .* innovations_samples') .* sigma_samples[:, 1]'
                effect_k_latent = U * scaled_innov
            else # :cholesky or :cholesky_sparse
                F = spec.hyper.cholesky_factor
                # Vectorized Level-3 BLAS triangular solve across all draws
                latent_field_unscaled = F.L' \ innovations_samples'
                latent_field_centered = latent_field_unscaled .- mean(latent_field_unscaled, dims=1)
                effect_k_latent = latent_field_centered .* sigma_samples[:, 1]'
            end
        end

        # Index the reconstructed latent effects to match the observation indices
        indexed_effects = effect_k_latent[s_idx_full, :]
        push!(structured_effects, indexed_effects)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end 
