"""
    MCAR <: ComponentModel

The Multivariate Conditional Autoregressive (MCAR) component for joint spatial modeling
of multiple correlated outcomes or diseases across areal units (Gelfand & Vounatsou, 2003).

# Version
v1.0.0

# Mathematical Summary
For \$K\$ multivariate spatial outcomes observed across \$S\$ connected areal units (with
  adjacency matrix \$W\$),
the MCAR model specifies the joint spatial-multivariate precision matrix using a Kronecker
  product decomposition:

\$\\operatorname{vec}(\\boldsymbol{\\Phi}) \\sim \\mathcal{N}\\left(\\mathbf{0},
  (\\mathbf{\\Omega}_{\\text{cross}} \\otimes \\mathbf{Q}_{\\text{spatial}})^{-1}\\right)\$

where:
- \$\\boldsymbol{\\Phi} = [\\boldsymbol{\\phi}_1, \\dots, \\boldsymbol{\\phi}_K] \\in
  \\mathbb{R}^{S \\times K}\$ is the matrix of spatial effects across all \$K\$ outcomes.
- \$\\mathbf{Q}_{\\text{spatial}} = (1 - \\rho) \\mathbf{I}_S + \\rho
  \\mathbf{Q}_{\\text{ICAR}}\$ is the proper Leroux (or BYM2) spatial precision matrix
  parameterized by spatial smoothing parameter \$\\rho \\in [0, 1]\$.
- \$\\mathbf{\\Omega}_{\\text{cross}} = \\mathbf{\\Sigma}_{\\text{cross}}^{-1}\$ is the \$K
  \\times K\$ cross-outcome precision matrix capturing non-spatial correlations between
  outcomes.

Using the spectral decomposition \$\\mathbf{Q}_{\\text{spatial}} = \\mathbf{U}
  \\mathbf{\\Lambda}_{\\text{spatial}} \\mathbf{U}^T\$ and Cholesky decomposition of the
  cross-outcome covariance \$\\mathbf{\\Sigma}_{\\text{cross}} = \\mathbf{L}_{\\text{cross}}
  \\mathbf{L}_{\\text{cross}}^T\$, the latent spatial field matrix is constructed
  constructively from independent standard normal innovations \$\\mathbf{Z} \\in
  \\mathbb{R}^{S \\times K}\$:

\$\\boldsymbol{\\Phi} = \\mathbf{U} \\left( \\mathbf{\\Lambda}_{\\text{spatial}}^{-1/2}
  \\odot \\mathbf{Z} \\right) \\mathbf{L}_{\\text{cross}}^T\$

where \$\\mathbf{L}_{\\text{cross}} = \\operatorname{diag}(\\boldsymbol{\\sigma})
  \\mathbf{L}_{\\text{corr}}\$ with \$\\mathbf{L}_{\\text{corr}} \\sim
  \\operatorname{LKJCholesky}(K, \\eta)\$.

# Computational Methods
- `:spectral` (Default, AD-friendly): Uses spectral decomposition of the spatial graph and
  Cholesky LKJ decomposition of cross-outcome covariance.
- `:leroux` (AD-friendly): Full-rank Leroux spatial precision ensuring proper joint distributions.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `s_idx`) passed to `random()`.
  - An adjacency matrix `W` passed as a keyword argument to `@tbm`.
- **Optional (in `random()` call)**:
  - `rho_unconstrained`: `UnivariateDistribution`, prior for unconstrained spatial mixing
    parameter \$\\rho = \\operatorname{logistic}(\\text{rho\\_unconstrained})\$. Default:
    `Normal(0, 0.5)`.
  - `sigma`: `UnivariateDistribution`, base prior for outcome marginal standard deviations.
    Default: `Exponential(1.0)`.
  - `correlation_lkj`: `Real`, LKJ prior shape parameter for cross-outcome correlation matrix.
    Default: `1.0`.

# Outputs (Parameter Names)
- `rho_unconstrained_<key>`: Spatial autocorrelation parameter.
- `sigma_<key>`: Outcome-specific marginal standard deviations (vector of length \$K\$).
- `L_corr_<key>`: LKJ Cholesky factor for cross-outcome correlations (\$K \\times K\$).
- `innovations_<key>`: Standard normal innovations (\$S \\cdot K\$).
- `latent_field_<key>`: Multivariate spatial realization (\$S \\times K\$).

# Key References
- Gelfand, A. E., & Vounatsou, P. (2003). *Proper multivariate conditional autoregressive
  models for spatial data analysis*. Biostatistics, 4(1), 11-15.
- Jin, X., Carlin, B. P., & Banerjee, S. (2005). *Generalized hierarchical multivariate CAR
  models for areal data*. Biometrics, 61(4), 950-961.
"""
struct MCAR <: ComponentModel
    rho_unconstrained::UnivariateDistribution
    sigma::Union{UnivariateDistribution, Real}
    correlation_lkj::Real
    n_outcomes::Int
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:mcar] = MCAR

COMPONENT_CONSTRUCTORS[:mcar] = (p, params) -> MCAR(
    get(p, :rho_unconstrained, Normal(0.0, 0.5)),
    get(p, :sigma, Exponential(1.0)),
    get(params, :correlation_lkj, 1.0),
    get(params, :n_outcomes, get(params, :K, 2)),
    get(params, :method, :spectral)
)

MODEL_TO_STRUCTURE_MAP[:mcar] = :spatial

function get_precomputes(m::MCAR, M::NamedTuple, mod_data::Dict)::NamedTuple
    if !hasproperty(M, :W) || !isa(M.W, AbstractMatrix) || isempty(M.W)
        error("MCAR model requires a valid, non-empty adjacency matrix `W` provided via keyword to `@tbm`.")
    end

    s_N = size(M.W, 1)
    K = hasproperty(M, :outcomes_N) ? M.outcomes_N : m.n_outcomes

    # Graph Laplacian Q_ICAR = D - W
    deg = vec(sum(M.W, dims=2))
    Q_template = spdiagm(0 => deg) - M.W

    # Spectral decomposition of spatial graph Laplacian
    eig = eigen(Symmetric(Matrix(Q_template)))
    U_spatial = eig.vectors
    lambda_raw = max.(eig.values, 0.0)

    # Spectral scaling
    scaling_factor = _compute_scaling_factor(lambda_raw, 1)
    lambda_scaled = lambda_raw ./ scaling_factor

    return (
        s_N = s_N,
        K = K,
        U_spatial = U_spatial,
        lambda_scaled = lambda_scaled,
        n_latent = s_N * K
    )
end

function get_priors(
    m::MCAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    s_N = spec.hyper.s_N
    K = spec.hyper.K

    # `sigma` is a K-vector (one marginal SD per outcome) and is declared
    # `Union{UnivariateDistribution, Real}`, so a pinned constant is representable. Calling
    # `_distribution_to_string` on it raised
    # `MethodError: no method matching _distribution_to_string(::Float64)`; a scalar prior
    # also cannot express a K-vector, so the pinned case is emitted as a filled constant of
    # the right length rather than via `filldist`.
    sigma_prior = if m.sigma isa Real
        "$(p_names.sigma) = fill($(m.sigma), $(K))"
    else
        "$(p_names.sigma) ~ filldist($(_distribution_to_string(m.sigma)), $(K))"
    end

    return """
    $(_prior_or_constant(p_names.rho_unconstrained, m.rho_unconstrained))
    $(sigma_prior)
    L_corr_$(spec.key) ~ LKJCholesky($(K), $(m.correlation_lkj))
    $(p_names.innovations) ~ MvNormal(zeros(T, $(s_N * K)), I)
    """
end

function get_updates(
    m::MCAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    key = spec.key
    hyper = spec.hyper
    s_N = hyper.s_N
    K = hyper.K

    multivariate_assign = if arch == "multivariate"
        """
        for k in 1:$(K)
            eta_latent[:, k] = eta_latent[:, k] .+ $(p_names.latent_field)[:, k][M.s_idx]
        end
        """
    else
        "eta = eta .+ $(p_names.latent_field)[:, 1][M.s_idx]"
    end

    return """
    # --- Multivariate Conditional Autoregressive (MCAR): $(key) ---
    $(p_names.latent_field) = let
        rho_val = logistic($(p_names.rho_unconstrained))
        U_sp = spec_registry[:$(key)].hyper.U_spatial
        lam_sp = spec_registry[:$(key)].hyper.lambda_scaled
        
        # Spatial Leroux precision eigenvalues: (1 - rho) + rho * lambda
        T_elem = eltype(rho_val)
        inv_sqrt_lam = zeros(T_elem, $(s_N))
        for i in 1:$(s_N)
            q_eig = (1.0 - rho_val) + (rho_val * lam_sp[i])
            inv_sqrt_lam[i] = 1.0 / sqrt(max(q_eig, 1e-6))
        end

        # Unpack standard normal innovations matrix (S x K)
        Z_mat = reshape($(p_names.innovations), $(s_N), $(K))
        
        # Spatial filtering: U * diag(inv_sqrt_lam) * Z
        U_spatial_field = U_sp * (inv_sqrt_lam .* Z_mat)

        # Cross-outcome covariance Cholesky factor: diag(sigma) * correlation_cholesky.L
        L_cholesky = L_corr_$(key).L
        sig_vec = $(p_names.sigma)
        L_cross = Diagonal(sig_vec) * L_cholesky

        # Multivariate spatial realization: U_spatial_field * L_cross' (S x K)
        U_spatial_field * transpose(L_cross)
    end

    $(multivariate_assign)
    """
end

function get_effects(
    m::MCAR, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    v = generate_full_variable_names(spec, M.model_arch, 1)
    key = spec.key
    s_N = spec.hyper.s_N
    K = spec.hyper.K
    U_sp = spec.hyper.U_spatial
    lam_sp = spec.hyper.lambda_scaled

    rho_samples = get_param_samples(chain, M.param_registry, Symbol(v.rho_unconstrained))
    sig_samples = get_param_samples(chain, M.param_registry, Symbol(v.sigma))
    innovations_samples = get_param_samples(chain, M.param_registry, Symbol(v.innovations))

    effect_matrix = zeros(Float64, M.y_N, n_samples)

    for s in 1:n_samples
        rho_val = 1.0 / (1.0 + exp(-rho_samples[s]))
        inv_sqrt_lam = [1.0 / sqrt(max((1.0 - rho_val) + rho_val * lam_sp[i], 1e-6)) for i in 1:s_N]
        
        u_raw = innovations_samples[:, s]
        Z_mat = reshape(u_raw, s_N, K)
        U_sp_field = U_sp * (inv_sqrt_lam .* Z_mat)
        
        sig_k = (sig_samples isa AbstractMatrix) ? sig_samples[:, s] : fill(Float64(sig_samples[s]), K)
        # Reconstruct outcome 1 effect
        effect_1 = U_sp_field[:, 1] .* sig_k[1]
        effect_matrix[:, s] = effect_1[M.s_idx]
    end

    return (structured=effect_matrix, noisy=effect_matrix)
end
