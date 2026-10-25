"""
    Moran <: ComponentModel

A component model for Moran's I Eigenvector Maps (MEM). This component decomposes
spatial autocorrelation into a set of orthogonal spatial patterns (eigenvectors)
derived from the Moran operator \$(I - 11'/n)W(I - 11'/n)\$. The effect is a linear
combination of these eigenvectors, providing a spectral basis for modeling spatial
processes.

# Version
v1.0.0

# Mathematical Summary
The Moran component models a spatial field \$\\phi\$ as a linear combination of the
eigenvectors of the Moran operator \$\\mathbf{M}\$:
\$\\boldsymbol{\\phi} = \\mathbf{E} \\boldsymbol{\\beta}\$
where:
1.  \$\\mathbf{W}\$ is the spatial adjacency matrix.
2.  \$\\mathbf{H} = \\mathbf{I} - \\frac{1}{n}\\mathbf{1}\\mathbf{1}^T\$ is a centering matrix.
3.  The Moran operator is \$\\mathbf{M} = \\mathbf{HWH}\$.
4.  \$\\mathbf{E}\$ is the matrix whose columns are the eigenvectors of \$\\mathbf{M}\$.
5.  \$\\boldsymbol{\\beta}\$ is a vector of coefficients, which are given a hierarchical
    prior: \$\\beta_k \\sim \\mathcal{N}(0, \\sigma^2)\$.

# Computational Methods
- `:noncentered` (Default, AD-friendly): A non-centered parameterization where coefficients are
  constructed from standard normal innovations. Recommended for gradient-based samplers.
- `:centered` (Didactic, Not AD-friendly): A centered parameterization where coefficients
  are sampled directly
  from `N(0, sigma^2)`. This can be less efficient for MCMC.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `region`) passed to `random()`.
  - An adjacency matrix `W` passed as a keyword argument to `@tbm`.
- **Optional (in `random()` call)**:
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation of the coefficients.
    Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:noncentered` or `:centered`). Default:
    `:noncentered`.

# Outputs (Parameter Names)
- `sigma_<key>`: The standard deviation of the eigenvector coefficients.
- `innovations_<key>`: The raw standard normal innovations for the coefficients (for
  `:noncentered`).
- `latent_<key>`: The latent coefficients (for `:centered`).

# Key References
- Griffith, D. A. (2003). *Spatial autocorrelation and spatial filtering: gaining
  understanding through theory and practice*. Springer Science & Business Media.
"""
struct Moran <: ComponentModel
    sigma::Union{Distribution, Real}
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:moran] = Moran
COMPONENT_CONSTRUCTORS[:moran] = (p, params) -> Moran(
    p.sigma, get(params, :method, :noncentered)
)

MODEL_TO_STRUCTURE_MAP[:moran] = :spatial

"""
    get_precomputes(m::Moran, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-dependent setup for the Moran component. This includes constructing
the Moran operator and computing its eigenvectors, which form the spatial basis.
This is a CPU-only implementation.
"""
function get_precomputes(m::Moran, M::NamedTuple, mod_data::Dict)::NamedTuple
    n = M.s_N
    W = M.W

    # Construct the centering matrix H = I - 11'/n
    H = I - (1/n) * ones(n, n)
    
    # Ensure W is a dense matrix for multiplication with H
    W_mat = Matrix(W)
    
    # Compute the Moran operator M = HWH
    moran_operator = H * W_mat * H
    
    # Perform eigendecomposition on the symmetric Moran operator
    # eigen returns CPU arrays
    eig_result = eigen(Symmetric(moran_operator))
    moran_eigenvectors = eig_result.vectors
    
    # The number of latent dimensions is the number of eigenvectors
    n_latent = size(moran_eigenvectors, 2)

    return (moran_eigenvectors=moran_eigenvectors, n_latent=n_latent)
end

"""
    _moran_log_marginal_likelihood(y_residual, s_idx, s_N, eigenvectors, sigma, y_sigma,
      noise=1e-6)

Computes the exact log marginal likelihood for a Moran eigenvector component integrated out
  analytically.
"""
function _moran_log_marginal_likelihood(
    y_residual::AbstractVector{T},
    s_idx::AbstractVector{Int},
    s_N::Int,
    eigenvectors::AbstractMatrix,
    sigma::T,
    y_sigma::T,
    noise::Real=1e-6
) where {T}
    N = length(y_residual)
    K = size(eigenvectors, 2)
    T_num = promote_type(T, typeof(noise))
    
    # Construct design X = eigenvectors[s_idx, :]
    X = Matrix{T_num}(eigenvectors[s_idx, :])
    
    inv_sigma_y2 = one(T_num) / (y_sigma^2 + T_num(noise))
    scale = sigma^2 + T_num(noise)
    
    XTX = Matrix{T_num}(X' * X)
    XTy = Vector{T_num}(X' * y_residual)
    
    I_mat = Matrix{T_num}(I, K, K)
    Q_base = I_mat .+ (scale * inv_sigma_y2) .* XTX
    for k in 1:K
        Q_base[k, k] += T_num(noise)
    end
    
    F = cholesky(Symmetric(Q_base))
    
    log_det_diff = - K * log(scale) - 2 * sum(log.(diag(F.U)))
    
    b = XTy .* inv_sigma_y2
    v = F.L \ b
    quad_term = scale * dot(v, v)
    
    log_lik = - (N / 2) * log(2 * T_num(pi) * (y_sigma^2 + T_num(noise))) -
              (inv_sigma_y2 / 2) * dot(y_residual, y_residual) +
              (1 / 2) * log_det_diff +
              (1 / 2) * quad_term
              
    return log_lik
end

"""
    get_priors(m::Moran, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for the Moran component's parameters, including the standard
deviation of the coefficients and the raw innovations.
"""
function get_priors(
    m::Moran, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    priors = ["$(_prior_or_constant(p_names.sigma, m.sigma))"]

    if m.method == :noncentered
        push!(
            priors,
            "$(p_names.innovations) ~ MvNormal(zeros(T, $(spec.hyper.n_latent)), I)"
        )
    end
    
    return join(priors, "\n    ")
end

"""
    get_updates(m::Moran, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for constructing the Moran eigenvector effect and adding
it to the linear predictor `eta`. This is a CPU-only implementation.
"""
function get_updates(
    m::Moran, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    n_latent = spec.hyper.n_latent
    
    common_code = """
        moran_eigenvectors = spec_registry[:$(key)].hyper.moran_eigenvectors
    """

    noncentered_code = """
        # --- Moran Eigenvector Component (Non-Centered): $(key) ---
        let
            $(common_code)
            scaled_coeffs = $(p_names.innovations) .* $(p_names.sigma)
            $(p_names.latent_field) = moran_eigenvectors * scaled_coeffs
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.s_idx)
        end
    """

    centered_code = """
        # --- Moran Eigenvector Component (Centered): $(key) ---
        let
            $(common_code)
            $(p_names.latent_field) ~ MvNormal(zeros(T, $(n_latent)), $(p_names.sigma)^2 * I)
            latent_field = moran_eigenvectors * $(p_names.latent_field)
            $(eta_target) = $(eta_target) .+ view(latent_field, M.s_idx)
        end
    """

    marginalized_code = """
        # --- Moran Eigenvector Component (Marginalized): $(key) ---
        let
            $(common_code)
            y_residual = M.y_obs .- $(eta_target)
            log_lik_marginalized_$(key) = _moran_log_marginal_likelihood(
                y_residual,
                M.s_idx,
                size(moran_eigenvectors, 1),
                moran_eigenvectors,
                $(p_names.sigma),
                @isdefined(y_sigma) ? y_sigma : 0.0,
                M.noise
            )
            Turing.@addlogprob! log_lik_marginalized_$(key)
        end
    """

    if m.method == :noncentered
        return noncentered_code
    elseif m.method == :centered
        return centered_code
    elseif m.method == :marginalized
        return marginalized_code
    else
        error("Unsupported method '$(m.method)' for Moran component. Use :noncentered, :centered, or :marginalized.")
    end
end

"""
    get_effects(m::Moran, chain, spec::NamedTuple, M::NamedTuple, PS)

Reconstructs the Moran eigenvector effect from posterior samples. This version is
CPU-only and uses modern chain accessors compatible with `MCMCThreads` output.
"""
function get_effects(
    m::Moran, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    n_samples = _get_chain_n_samples(chain)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = string.(keys(chain))
    noise = get(M, :noise, 1e-6)
    
    structured_effects = Vector{Matrix{Float64}}()
    
    eigenvectors = spec.hyper.moran_eigenvectors
    n_latent = spec.hyper.n_latent

    # --- Index Handling: Combine training and prediction sets on CPU ---
    s_idx_train = M.s_idx
    s_idx_full = if !isnothing(PS) && hasproperty(PS.data, :s_idx)
        vcat(s_idx_train, PS.data.s_idx)
    else
        s_idx_train
    end
    N_total = length(s_idx_full)

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        sigma_samples = _resolve_hyper_samples(
            chain, p_names, p_names_k.sigma, m.sigma, k,
            is_multivariate_model, n_samples;
            as_matrix = true)

        if isnothing(sigma_samples)
            @warn "Parameters for $(spec.key) (outcome $k) not resolved, and " *
                  "sigma is not a pinned constant. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end
        
        local latent_field_matrix
        if m.method == :marginalized
            y_sigma_name = _find_parameter(p_names, "y_sigma", k, is_multivariate_model)
            y_sigma_samples = if !isempty(y_sigma_name)
                get_params_vector(chain, y_sigma_name, 1)[:, 1]
            else
                fill(1.0, n_samples)
            end
            
            y_vec = M.y_obs isa AbstractMatrix ? M.y_obs[:, k] : M.y_obs
            X_train = Matrix{Float64}(eigenvectors[s_idx_train, :])
            XTX = Matrix{Float64}(X_train' * X_train)
            XTy = Vector{Float64}(X_train' * y_vec)
            
            K = n_latent
            I_mat = Matrix{Float64}(I, K, K)
            coeffs_matrix = zeros(Float64, K, n_samples)
            
            for s in 1:n_samples
                sig = sigma_samples[s, 1]
                y_sig = y_sigma_samples[s]
                
                scale = sig^2 + noise
                inv_sigma_y2 = 1.0 / (y_sig^2 + noise)
                
                Q_base = I_mat .+ (scale * inv_sigma_y2) .* XTX
                for j in 1:K
                    Q_base[j, j] += noise
                end
                
                F = cholesky(Symmetric(Q_base))
                b = XTy .* inv_sigma_y2
                mu = scale .* (F \ b)
                
                z = randn(K)
                coeffs_matrix[:, s] = mu .+ sqrt(max(scale, 1e-12)) .* (F.U \ z)
            end
            latent_field_matrix = eigenvectors * coeffs_matrix
        elseif m.method == :noncentered
            innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)
            innovations_samples = get_params_matrix(chain, innovations_name, n_latent) # (n_samples, n_latent)
            
            scaled_coeffs = innovations_samples' .* sigma_samples' # (n_latent, n_samples)
            latent_field_matrix = eigenvectors * scaled_coeffs

        else # :centered
            latent_field_name = _find_parameter(p_names, string(p_names_k.latent_field), k, is_multivariate_model)
            coeffs_samples = get_params_matrix(chain, latent_field_name, n_latent)
            
            latent_field_matrix = eigenvectors * coeffs_samples'
        end
        
        effect_k = latent_field_matrix[s_idx_full, :]
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
