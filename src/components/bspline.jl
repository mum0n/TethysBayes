"""
    BSpline <: ComponentModel

A component model for a B-spline smoother, often referred to as a P-spline
(Penalized B-spline). This component creates a basis of B-spline functions of a
specified degree, centered at knots distributed across the covariate space. The
effect is a linear combination of these basis functions, with coefficients
regularized by a random walk prior to ensure smoothness.

# Version
v1.0.0

# Mathematical Summary
The component models a smooth function \$f(x)\$ as a linear combination of B-spline
basis functions:
\$f(x) = \\sum_{j=1}^{N_{bins}} \\beta_j B_{j,p}(x)\$
where \$B_{j,p}(x)\$ is the j-th B-spline basis function of degree \$p\$, and \$\\beta_j\$
are the coefficients.

To prevent overfitting and ensure smoothness, a penalty is applied to the
coefficients. This is equivalent to placing a Gaussian Markov Random Field (GMRF)
prior on the coefficients. A common choice is a second-order random walk (RW2)
penalty on the differences between adjacent coefficients:
\$\\Delta^2 \\beta_j = \\beta_j - 2\\beta_{j-1} + \\beta_{j-2}\$
This penalizes deviations from a linear trend, encouraging a smooth function.

# Computational Methods
- `:spectral` (Default, AD-friendly): Regularizes coefficients using a spectral
  decomposition of the RW2 penalty matrix. Recommended for gradient-based samplers.
- `:cholesky` (AD-friendly): Uses a dense Cholesky factorization of the penalty
  matrix. Can be less efficient for a large number of basis functions.
- `:cholesky_sparse` (Didactic, Not AD-friendly): Uses a sparse Cholesky
  factorization. Not compatible with AD but retained as a didactic alternative.

# Inputs
- **Required**:
  - One or more coordinate variables (e.g., `x`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `nbins`: `Int`, the number of basis functions. Default: `10`.
  - `degree`: `Int`, the polynomial degree of the B-spline. Default: `3`.
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation of the
    coefficients. Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:spectral`, `:cholesky`, `:cholesky_sparse`).
    Default: `:spectral`.

# Outputs (Parameter Names)
- `sigma_<key>`: The standard deviation of the B-spline coefficients.
- `innovations_<key>`: The raw standard normal innovations for the coefficients.
- `latent_<key>`: The final smooth effect vector.

# Key References
- Eilers, P. H., & Marx, B. D. (1996). *Flexible smoothing with B-splines and
  penalties*. Statistical Science, 11(2), 89-121.
- de Boor, C. (1978). *A Practical Guide to Splines*. Springer-Verlag.
"""
struct BSpline <: ComponentModel
    nbins::Int
    degree::Int
    sigma::Union{Distribution, Real}
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:bspline] = BSpline

COMPONENT_CONSTRUCTORS[:bspline] = (p, params) -> BSpline(
    get(params, :nbins, 10),
    get(params, :degree, 3),
    p.sigma,
    get(params, :method, :spectral)
)

MODEL_TO_STRUCTURE_MAP[:bspline] = :smooth

"""
    get_precomputes(m::BSpline, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes the B-spline basis matrix and the RW2 penalty matrix (and its
spectral/Cholesky decompositions) for the spline coefficients. This is a CPU-only
implementation.
"""
function get_precomputes(m::BSpline, M::NamedTuple, mod_data::Dict)::NamedTuple
    raw_vars = get(mod_data, :variables, [])
    variables = raw_vars isa AbstractVector ? raw_vars : [raw_vars]
    
    n_levels = get(M, :N_levels, m.nbins)
    coords = if hasproperty(M, :data) && !isempty(variables) && hasproperty(M.data,
        Symbol(variables[1]))
        Matrix{Float64}(M.data[!, Symbol.(variables)])
    else
        collect(range(0.0, 1.0, length=n_levels))[:, :]
    end
    
    n_obs, n_dims = size(coords)
    
    # Generate the B-spline basis matrix on the CPU.
    B, actual_nbins = TethysBayes_bspline_basis(coords[:, 1], m.nbins, m.degree)
    n_latent = actual_nbins

    # Build the RW2 penalty matrix template on the CPU.
    template = build_structure_template(:rw2, n_latent)
    Q_template = template.matrix
    
    # Perform eigendecomposition for spectral method on the CPU.
    eig_decomp = eigen(Symmetric(Matrix(Q_template)))
    U = eig_decomp.vectors
    L = eig_decomp.values
    scaling_factor = _compute_scaling_factor(L, 2) # RW2 has rank deficiency of 2
    
    Q_template_scaled = Q_template ./ scaling_factor
    L_scaled = L ./ scaling_factor
    noise_val = get(M, :noise, 1e-6)

    # Pre-compute dense Cholesky factor for the :cholesky method on the CPU.
    F = cholesky(Symmetric(Matrix(Q_template_scaled) + noise_val * I))

    return (
        basis_matrix=B,
        B_matrix=B,
        Q_template=Q_template_scaled,
        scaling_factor=scaling_factor,
        U=U,
        L=L_scaled,
        n_latent=n_latent,
        cholesky_factor=F,
        model_type=:bspline
    )
end

"""
    _bspline_log_marginal_likelihood(y_residual, B_basis, Q_penalty, L_eig, sigma, y_sigma,
      noise=1e-6)

Computes the exact log marginal likelihood for a BSpline component with basis coefficients
  integrated out analytically.
"""
function _bspline_log_marginal_likelihood(
    y_residual::AbstractVector{T},
    B_basis::AbstractMatrix,
    Q_penalty::AbstractMatrix,
    L_eig::AbstractVector,
    sigma::T,
    y_sigma::T,
    noise::Real=1e-6
) where {T}
    N = length(y_residual)
    K = size(B_basis, 2)
    T_num = promote_type(T, typeof(noise))
    
    inv_sigma_y2 = one(T_num) / (y_sigma^2 + T_num(noise))
    scale = sigma^2 + T_num(noise)
    
    BTB = Matrix{T_num}(B_basis' * B_basis)
    BTy = Vector{T_num}(B_basis' * y_residual)
    
    Q_base = Matrix{T_num}(Q_penalty) .+ (scale * inv_sigma_y2) .* BTB
    for k in 1:K
        Q_base[k, k] += T_num(noise)
    end
    
    F = cholesky(Symmetric(Q_base))
    
    # RW2 has rank deficiency of 2
    valid_eigs = L_eig[3:end]
    log_det_prior = isempty(valid_eigs) ? zero(T_num) : sum(log.(valid_eigs .+ T_num(noise)))
    log_det_diff = - max(K - 2, 1) * log(scale) + log_det_prior - 2 * sum(log.(diag(F.U)))
    
    b = BTy .* inv_sigma_y2
    v = F.L \ b
    quad_term = scale * dot(v, v)
    
    log_lik = - (N / 2) * log(2 * T_num(pi) * (y_sigma^2 + T_num(noise))) -
              (inv_sigma_y2 / 2) * dot(y_residual, y_residual) +
              (1 / 2) * log_det_diff +
              (1 / 2) * quad_term
              
    return log_lik
end

"""
    get_priors(m::BSpline, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for the standard deviation `sigma` and the raw innovations for
the B-spline coefficients.
"""
function get_priors(
    m::BSpline, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    sigma_prior_str = _prior_or_constant(p_names.sigma, m.sigma)
    if m.method == :marginalized
        return """
        # Priors for BSpline component: $(spec.key)
        $(sigma_prior_str)
        """
    else
        return """
        # Priors for BSpline component: $(spec.key)
        $(sigma_prior_str)
        $(p_names.innovations) ~ MvNormal(zeros(T, $(spec.hyper.n_latent)), I)
        """
    end
end

"""
    get_updates(m::BSpline, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates Turing code to construct the B-spline smooth effect and add it to the
linear predictor `eta`, dispatching on the chosen computational method.
"""
function get_updates(
    m::BSpline, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    n_latent = spec.hyper.n_latent
    
    common_code = """
        hyper = spec_registry[:$(key)].hyper
        B_basis = hyper.basis_matrix
    """

    spectral_code = """
        # --- B-Spline Smoother Component (Spectral): $(key) ---
        let
            $(common_code)
            
            diag_D = $(p_names.sigma) ./ sqrt.(hyper.L .+ M.noise)
            # Enforce sum-to-zero constraints for RW2 penalty
            # Zero every null direction: the count depends on the spectrum, and a
            # disconnected graph has more than one. See _zero_null_modes!.
            _zero_null_modes!(diag_D, hyper.L)
            
            coeffs = hyper.U * (diag_D .* $(p_names.innovations))
            
            $(p_names.latent_field) = B_basis * coeffs
            
            $(eta_target) = $(eta_target) .+ $(p_names.latent_field)
        end
    """

    cholesky_code = """
        # --- B-Spline Smoother Component (Cholesky, AD-Safe): $(key) ---
        let
            $(common_code)
            
            F = hyper.cholesky_factor
            
            coeffs_unscaled = F.L' \\ $(p_names.innovations)
            
            # Apply soft sum-to-zero constraints for RW2 penalty
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * $(n_latent)), sum(coeffs_unscaled[1:2])
            )
            
            coeffs = $(p_names.sigma) .* coeffs_unscaled
            $(p_names.latent_field) = B_basis * coeffs
            
            $(eta_target) = $(eta_target) .+ $(p_names.latent_field)
        end
    """

    cholesky_sparse_code = """
        # --- B-Spline Smoother Component (Sparse Cholesky, Not AD-Safe): $(key) ---
        let
            $(common_code)
            
            Q_penalty = hyper.Q_template
            F = cholesky(Symmetric(Q_penalty + M.noise * I))
            
            L_sparse = sparse(F.L)
            coeffs_unscaled = L_sparse' \\ $(p_names.innovations)
            
            # Apply soft sum-to-zero constraints for RW2 penalty
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * $(n_latent)), sum(coeffs_unscaled[1:2])
            )
            
            coeffs = $(p_names.sigma) .* coeffs_unscaled
            $(p_names.latent_field) = B_basis * coeffs
            
            $(eta_target) = $(eta_target) .+ $(p_names.latent_field)
        end
    """

    marginalized_code = """
        # --- B-Spline Smoother Component (Marginalized): $(key) ---
        let
            $(common_code)
            y_residual = M.y_obs .- $(eta_target)
            log_lik_marginalized_$(key) = _bspline_log_marginal_likelihood(
                y_residual,
                B_basis,
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
        error("Unsupported method '$(m.method)' for BSpline component. Use :spectral, :cholesky, :cholesky_sparse, or :marginalized.")
    end
end

"""
    get_effects(m::BSpline, chain, spec::NamedTuple, M::NamedTuple, PS)

Reconstructs the posterior distribution of the B-spline smooth effect from the
MCMC chain. This version is CPU-only and uses modern chain accessors.
"""
function get_effects(
    m::BSpline, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    n_samples = _get_chain_n_samples(chain)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = string.(keys(chain))
    
    hyper = spec.hyper
    noise = M.noise
    n_latent = hyper.n_latent

    # --- Basis Matrix Handling for Training and Prediction Sets ---
    B_train = hyper.basis_matrix

    coord_vars = get(spec.params, :positional_args, [])
    coord_var_sym = isempty(coord_vars) ? :none : Symbol(coord_vars[1])

    B_full = if !isnothing(PS) && hasproperty(PS.data, coord_var_sym)
        coords_pred = PS.data[!, coord_var_sym]
        B_pred, _ = TethysBayes_bspline_basis(coords_pred, m.nbins, m.degree)
        vcat(B_train, B_pred)
    else
        B_train
    end
    N_total = size(B_full, 1)

    structured_effects = Vector{Matrix{Float64}}()

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


        # Initialize the output matrix for coefficients
        coeffs_samples_matrix = zeros(Float64, n_latent, n_samples)

        # --- Sample-wise Reconstruction ---
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
                sig = sigma_samples[i, 1]
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
                coeffs_samples_matrix[:, i] = coeffs_i
            end
        elseif m.method == :spectral
            innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)
            innovations_samples = get_params_matrix(chain, innovations_name, n_latent)

            U = hyper.U
            L = hyper.L
            
            diag_D_matrix = (sigma_samples' ./ sqrt.(L .+ noise))
            for i in 1:m.degree+1
                diag_D_matrix[i, :] .= 0.0
            end
            
            coeffs_samples_matrix = U * (diag_D_matrix .* innovations_samples')
        else # :cholesky or :cholesky_sparse
            innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)
            innovations_samples = get_params_matrix(chain, innovations_name, n_latent)

            F = hyper.cholesky_factor
            for i in 1:n_samples
                innov_i = innovations_samples[i, :]
                coeffs_unscaled = F.L' \ innov_i
                coeffs_centered = coeffs_unscaled .- mean(coeffs_unscaled)
                coeffs_samples_matrix[:, i] = sigma_samples[i, 1] .* coeffs_centered
            end
        end
        
        # Perform final matrix multiplication
        effect_k = B_full * coeffs_samples_matrix
        
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end 
