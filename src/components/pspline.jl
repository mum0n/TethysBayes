"""
    PSpline <: ComponentModel

A component model for a P-spline (Penalized B-spline) smoother. This component
creates a basis of B-spline functions and applies a discrete penalty (typically a
random walk) to the coefficients to ensure smoothness and prevent overfitting.

# Version
v1.0.0

# Mathematical Summary
The P-spline models a smooth function \$f(x)\$ as a linear combination of \$K\$ B-spline
basis functions \$B_k(x)\$:
\$f(x) = \\sum_{k=1}^{K} \\beta_k B_k(x)\$
To enforce smoothness, a penalty is applied to the coefficients \$\\boldsymbol{\\beta}\$.
This is achieved by assuming the coefficients follow a Gaussian Markov Random Field
(GMRF) structure. A common choice is a second-order random walk (RW2), which
penalizes deviations from a local linear trend:
\$\\Delta^d \\beta_k = \\sum_{j=0}^d (-1)^j \\binom{d}{j} \\beta_{k-j} \\sim \\mathcal{N}(0,
  \\sigma^{-2})\$
where \$d\$ is the `penalty_order`. The precision matrix \$\\mathbf{Q}\$ for the coefficients
is derived from this random walk structure. The model then samples the coefficients from
\$\\boldsymbol{\\beta} \\sim \\mathcal{N}(\\mathbf{0}, (\\sigma^2 \\mathbf{Q})^{-1})\$.

# Computational Methods
- `:spectral` (Default, AD-friendly): Regularizes coefficients using a spectral
  decomposition of the penalty matrix. Recommended for gradient-based samplers.
- `:cholesky` (AD-friendly): Uses a pre-computed dense Cholesky factorization of the
  penalty matrix.
- `:cholesky_sparse` (Didactic, Not AD-friendly): Uses sparse Cholesky factorization,
  which is not compatible with most AD backends.

# Inputs
- **Required**:
  - One or more coordinate variables (e.g., `x`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `nbins`: `Int`, the number of basis functions. Default: `20`.
  - `degree`: `Int`, the polynomial degree of the B-spline. Default: `3`.
  - `penalty_order`: `Int`, the order of the random walk penalty (1 or 2). Default: `2`.
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation of the
    coefficients. Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:spectral`, `:cholesky`, `:cholesky_sparse`).
    Default: `:spectral`.

# Outputs (Parameter Names)
- `sigma_<key>`: The standard deviation of the B-spline coefficients.
- `innovations_<key>`: The raw standard normal innovations for the coefficients.
- `latent_<key>`: The final smooth effect vector.
"""
struct PSpline <: ComponentModel
    nbins::Int
    degree::Int
    penalty_order::Int
    sigma::Union{Distribution, Real}
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:pspline] = PSpline

COMPONENT_CONSTRUCTORS[:pspline] = (p, params) -> PSpline(
    get(params, :nbins, 20),
    get(params, :degree, 3),
    get(params, :penalty_order, 2),
    p.sigma,
    get(params, :method, :spectral)
)

MODEL_TO_STRUCTURE_MAP[:pspline] = :smooth

function get_precomputes(m::PSpline, M::NamedTuple, mod_data::Dict)::NamedTuple
    variables = mod_data[:variables]

    if isempty(variables)
        error(
            "The PSpline model requires at least one coordinate variable, e.g., " *
            "`random(x, model=:pspline)`."
        )
    end

    for var_sym in variables
        if !hasproperty(M.data, Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for PSpline model not found in data.")
        end
    end

    coords = Matrix{Float64}(M.data[!, Symbol.(variables)])

    if size(coords, 2) > 1
        @warn "PSpline is designed for 1D smooths. For multi-dimensional smoothing, " *
              "consider `tps` or creating tensor products manually."
    end

    B, actual_nbins = TethysBayes_bspline_basis(coords[:, 1], m.nbins, m.degree)
    n_latent = actual_nbins

    penalty_type = m.penalty_order == 1 ? :rw1 : :rw2
    template = build_structure_template(penalty_type, n_latent)
    Q_template = template.matrix

    rank_deficiency = m.penalty_order
    eig_decomp = eigen(Symmetric(Matrix(Q_template)))
    U = eig_decomp.vectors
    L = eig_decomp.values
    scaling_factor = _compute_scaling_factor(L, rank_deficiency)

    Q_template_scaled = Q_template ./ scaling_factor
    L_scaled = L ./ scaling_factor

    F = cholesky(Symmetric(Matrix(Q_template_scaled) + M.noise * I))

    return (
        basis_matrix=B,
        Q_template=Q_template_scaled,
        scaling_factor=scaling_factor,
        U=U,
        L=L_scaled,
        n_latent=n_latent,
        cholesky_factor=F
    )
end

"""
    _pspline_log_marginal_likelihood(y_residual, B_basis, Q_penalty, L_eig, penalty_order,
      sigma, y_sigma, noise=1e-6)

Computes the exact log marginal likelihood for a P-spline component with basis coefficients
  integrated out analytically.
"""
function _pspline_log_marginal_likelihood(
    y_residual::AbstractVector{T},
    B_basis::AbstractMatrix,
    Q_penalty::AbstractMatrix,
    L_eig::AbstractVector,
    penalty_order::Int,
    sigma::T,
    y_sigma::T,
    noise::Real=1e-6
) where {T}
    N = length(y_residual)
    K = size(B_basis, 2)
    T_num = promote_type(T, typeof(noise))
    
    inv_sigma_y2 = one(T_num) / (y_sigma^2 + T_num(noise))
    scale = sigma^2 + T_num(noise)
    
    # B^T B and B^T y_residual
    BTB = Matrix{T_num}(B_basis' * B_basis)
    BTy = Vector{T_num}(B_basis' * y_residual)
    
    # Q_post_scaled = Q_penalty + (sigma^2 / sigma_y^2) * B^T B + noise * I
    Q_base = Matrix{T_num}(Q_penalty) .+ (scale * inv_sigma_y2) .* BTB
    for k in 1:K
        Q_base[k, k] += T_num(noise)
    end
    
    F = cholesky(Symmetric(Q_base))
    
    # Determinant term
    valid_eigs = L_eig[(penalty_order + 1):end]
    log_det_prior = isempty(valid_eigs) ? zero(T_num) : sum(log.(valid_eigs .+ T_num(noise)))
    log_det_diff = - max(K - penalty_order, 1) * log(scale) + log_det_prior - 2 * sum(log.(diag(F.U)))
    
    # Quadratic term
    b = BTy .* inv_sigma_y2
    v = F.L \ b
    quad_term = scale * dot(v, v)
    
    log_lik = - (N / 2) * log(2 * T_num(pi) * (y_sigma^2 + T_num(noise))) -
              (inv_sigma_y2 / 2) * dot(y_residual, y_residual) +
              (1 / 2) * log_det_diff +
              (1 / 2) * quad_term
              
    return log_lik
end

function get_priors(
    m::PSpline, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    sigma_prior_str = _prior_or_constant(p_names.sigma, m.sigma)
    key = spec.key

    if m.method == :marginalized
        return "$(sigma_prior_str)"
    else
        return """
            $(sigma_prior_str)
            $(p_names.innovations) ~ MvNormal(
                zeros(T, spec_registry[:$(key)].hyper.n_latent), I
            )
        """
    end
end

function get_updates(
    m::PSpline, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key

    common_code = """
        local hyper = spec_registry[:$(key)].hyper
        local B_basis = hyper.basis_matrix
    """

    spectral_code = """
        # --- P-Spline Smoother Component (Spectral): $(key) ---
        let
            $(common_code)
            local diag_D = $(p_names.sigma) ./ sqrt.(hyper.L .+ M.noise)
            for i in 1:$(m.penalty_order); diag_D[i] = 0.0; end

            coeffs = hyper.U * (diag_D .* $(p_names.innovations))
            $(p_names.latent_field) = B_basis * coeffs
            $(eta_target) = $(eta_target) .+ $(p_names.latent_field)
        end
    """

    cholesky_code = """
        # --- P-Spline Smoother Component (Cholesky, AD-Safe): $(key) ---
        let
            $(common_code)
            local F = hyper.cholesky_factor
            local coeffs_unscaled = F.L' \\ $(p_names.innovations)

            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * hyper.n_latent), sum(coeffs_unscaled)
            )

            local coeffs = $(p_names.sigma) .* coeffs_unscaled
            $(p_names.latent_field) = B_basis * coeffs
            $(eta_target) = $(eta_target) .+ $(p_names.latent_field)
        end
    """

    cholesky_sparse_code = """
        # --- P-Spline Smoother Component (Sparse Cholesky, Not AD-Safe): $(key) ---
        let
            $(common_code)
            local Q_penalty = hyper.Q_template
            local F = cholesky(Symmetric(Q_penalty + M.noise * I))
            local coeffs_unscaled = F.L' \\ $(p_names.innovations)

            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * hyper.n_latent), sum(coeffs_unscaled)
            )

            local coeffs = $(p_names.sigma) .* coeffs_unscaled
            $(p_names.latent_field) = B_basis * coeffs
            $(eta_target) = $(eta_target) .+ $(p_names.latent_field)
        end
    """

    marginalized_code = """
        # --- P-Spline Smoother Component (Marginalized): $(key) ---
        let
            $(common_code)
            y_residual = M.y_obs .- $(eta_target)
            log_lik_marginalized_$(key) = _pspline_log_marginal_likelihood(
                y_residual,
                B_basis,
                hyper.Q_template,
                hyper.L,
                $(m.penalty_order),
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
        error("Unsupported method '$(m.method)' for PSpline component. Use :spectral, :cholesky, :cholesky_sparse, or :marginalized.")
    end
end

function get_effects(
    m::PSpline, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    n_samples = _get_chain_n_samples(chain)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = string.(keys(chain))
    
    hyper = spec.hyper
    noise = M.noise
    n_latent = hyper.n_latent

    # --- Basis Matrix Handling for Training and Prediction Sets (CPU-only) ---
    B_train = hyper.basis_matrix # This is already a CPU array

    coord_vars = get(spec.params, :positional_args, [])
    coord_var_sym = isempty(coord_vars) ? :none : Symbol(coord_vars[1])

    B_full = if !isnothing(PS) && hasproperty(PS.data, coord_var_sym)
        coords_pred_cpu = PS.data[!, coord_var_sym]
        
        # TethysBayes_bspline_basis is CPU-only
        B_pred, _ = TethysBayes_bspline_basis(coords_pred_cpu, m.nbins, m.degree)
        vcat(B_train, B_pred)
    else
        B_train
    end
    N_total = size(B_full, 1)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        
        # Find parameter names in the MCMC chain
        sigma_samples_cpu = _resolve_hyper_samples(
            chain, p_names, p_names_k.sigma, m.sigma, k,
            is_multivariate_model, n_samples)

        if isnothing(sigma_samples_cpu)
            @warn "Parameters for $(spec.key) (outcome $k) not resolved, and " *
                  "sigma is not a pinned constant. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end


        # Initialize the output matrix for coefficients on the CPU
        coeffs_samples_matrix_cpu = zeros(Float64, n_latent, n_samples)

        # --- Sample-wise Reconstruction on the CPU ---
        if m.method == :marginalized
            y_sigma_name = _find_parameter(p_names, "y_sigma", k, is_multivariate_model)
            y_sigma_samples = if !isempty(y_sigma_name)
                get_params_vector(chain, y_sigma_name, 1)[:, 1]
            else
                fill(1.0, n_samples)
            end
            
            y_vec = M.y_obs isa AbstractMatrix ? M.y_obs[:, k] : M.y_obs
            BTB = Matrix{Float64}(B_train' * B_train)
            BTy = Vector{Float64}(B_train' * y_vec)
            
            for i in 1:n_samples
                sig = sigma_samples_cpu[i]
                y_sig = y_sigma_samples[i]
                
                scale = sig^2 + noise
                inv_sigma_y2 = 1.0 / (y_sig^2 + noise)
                
                Q_base = Matrix{Float64}(hyper.Q_template) .+ (scale * inv_sigma_y2) .* BTB
                for j in 1:n_latent
                    Q_base[j, j] += noise
                end
                
                F = cholesky(Symmetric(Q_base))
                b = BTy .* inv_sigma_y2
                mu = scale .* (F \ b)
                
                z = randn(n_latent)
                coeffs_i = mu .+ sqrt(max(scale, 1e-12)) .* (F.U \ z)
                coeffs_samples_matrix_cpu[:, i] = coeffs_i
            end
        elseif m.method == :spectral
            innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)
            innovations_samples_cpu = get_params_matrix(chain, innovations_name, n_latent)

            U = hyper.U
            L = hyper.L
            
            diag_D_matrix = (sigma_samples_cpu' ./ sqrt.(L .+ noise))
            for i in 1:m.penalty_order
                diag_D_matrix[i, :] .= 0.0
            end
            
            coeffs_samples_matrix_cpu = U * (diag_D_matrix .* innovations_samples_cpu')
        else # :cholesky or :cholesky_sparse
            innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)
            innovations_samples_cpu = get_params_matrix(chain, innovations_name, n_latent)

            F = hyper.cholesky_factor
            for i in 1:n_samples
                innov_i_cpu = innovations_samples_cpu[i, :]
                coeffs_unscaled = F.L' \ innov_i_cpu
                coeffs_centered = coeffs_unscaled .- mean(coeffs_unscaled)
                coeffs_samples_matrix_cpu[:, i] = sigma_samples_cpu[i] .* coeffs_centered
            end
        end
        
        # Perform final matrix multiplication on the CPU
        effect_k_cpu = B_full * coeffs_samples_matrix_cpu
        
        push!(structured_effects, effect_k_cpu)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
