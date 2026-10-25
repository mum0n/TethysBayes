"""
    Leroux <: ComponentModel

A component for a Leroux model, which is a proper Conditional Autoregressive (CAR)
model. It defines spatial correlation as a convex combination of a spatially
structured (ICAR) component and an unstructured (IID) component, controlled by a
single mixing parameter, `rho`.

# Version
v1.0.0

# Mathematical Summary
The Leroux model is a proper CAR model, meaning its precision matrix is always
positive definite. It defines the precision matrix \$\\mathbf{Q}\$ as a convex
combination of an identity matrix \$\\mathbf{I}\$ and a scaled ICAR precision matrix
\$\\mathbf{Q}^*\$ (where \$\\mathbf{Q}^* = D - W\$):
\$\\mathbf{Q} = (1-\\rho)\\mathbf{I} + \\rho\\mathbf{Q}^*\$
This structure allows the model to smoothly interpolate between unstructured random
effects (\$\\rho=0\$) and a fully structured ICAR model (\$\\rho=1\$), providing a
flexible way to model spatial autocorrelation.

# Computational Methods
- `:spectral` (Default, AD-friendly): Regularizes coefficients using a spectral
  decomposition of the ICAR precision matrix. Recommended for gradient-based samplers.
- `:cholesky` (AD-friendly): Uses a dense Cholesky factorization of the
  full Leroux precision matrix, computed on-the-fly.
- `:cholesky_sparse` (Didactic, Not AD-friendly): Uses sparse Cholesky factorization,
  computed on-the-fly.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `region`) passed to `random()`.
  - An adjacency matrix `W` passed as a keyword argument to `@tbm`.
- **Optional (in `random()` call)**:
  - `rho`: A `UnivariateDistribution` for the prior on the mixing parameter. Default: `Beta(1,1)`.
  - `sigma`: A `UnivariateDistribution` for the prior on the overall standard deviation.
    Default: `Exponential(1.0)`.
  - `method`: `Symbol`, specifying the computational method. Default: `:spectral`.

# Outputs (Parameter Names)
- `sigma_<key>`: The overall marginal standard deviation.
- `rho_<key>`: The mixing parameter.
- `innovations_<key>`: The raw standard normal innovations for the effect.
- `latent_<key>`: The reconstructed latent spatial field.

# Key References
- Leroux, B. G., Lei, X., & Breslow, N. (2000). Estimation of disease rates in
  small areas: a new mixed model for spatial dependence. In *Statistical models
  in epidemiology, the environment, and clinical trials* (pp. 179-191). Springer.
- Wikipedia: Conditional autoregressive model
"""
struct Leroux <: ComponentModel
    rho::UnivariateDistribution
    sigma::Union{UnivariateDistribution, Real}
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:leroux] = Leroux

COMPONENT_CONSTRUCTORS[:leroux] = (p, params) -> Leroux(
    p.rho, p.sigma, get(params, :method, :spectral)
)

MODEL_TO_STRUCTURE_MAP[:leroux] = :spatial

"""
    get_precomputes(m::Leroux, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-dependent setup for the Leroux model. This version is CPU-only.
It pre-computes the ICAR precision matrix template and its spectral decomposition.
"""
function get_precomputes(m::Leroux, M::NamedTuple, mod_data::Dict)::NamedTuple
    s_N = get(M, :s_N, 0)
    W = get(M, :W, nothing)
    if s_N == 0 || isnothing(W)
        error(
            "Could not perform pre-computation for Leroux component because " *
            "spatial context (s_N and W) is missing."
        )
    end

    # build_structure_template returns CPU arrays
    template = build_structure_template(:icar, s_N; W=W)
    
    # All precomputed structures remain on the CPU.
    # Do not pre-compute Cholesky factor as it depends on `rho`.
    return (
        Q_template=template.matrix,
        U=template.U,
        L=template.L,
        scaling_factor=template.scaling_factor,
        n_latent=s_N
    )
end

"""
    _leroux_log_marginal_likelihood(y_residual, s_idx, s_N, Q_template, L_eig, rho, sigma,
      y_sigma, noise=1e-6)

Computes the exact log marginal likelihood for a Leroux spatial CAR process integrated out
  analytically.
"""
function _leroux_log_marginal_likelihood(
    y_residual::AbstractVector{T},
    s_idx::AbstractVector{Int},
    s_N::Int,
    Q_template::AbstractMatrix,
    L_eig::AbstractVector,
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
    Q_base = (one(T_num) - rho) .* I_mat .+ rho .* Matrix{T_num}(Q_template)
    for s in 1:s_N
        Q_base[s, s] += T_num(noise) + N_s[s] * inv_sigma_y2 * scale
    end
    
    F = cholesky(Symmetric(Q_base))
    
    # Determinant term
    log_det_prior = sum(log.((one(T_num) - rho) .+ rho .* L_eig .+ T_num(noise)))
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
    m::Leroux, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent
    is_multivariate = (arch == "multivariate")
    is_shared = get(spec.params, :shared, false)
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(_prior_or_constant(p_names.sigma, m.sigma))")
        push!(priors_acc, "$(_prior_or_constant(p_names.rho, m.rho))")
    end
    if m.method != :marginalized
        push!(priors_acc, "$(p_names.innovations) ~ MvNormal(zeros(T, $(n_latent)), I)")
    end
    return join(priors_acc, "\n    ")
end


"""
    get_updates(m::Leroux, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates code to compute the Leroux effect and add it to the linear predictor `eta`.
Supports methods: `:spectral`, `:cholesky`, `:cholesky_sparse`, and `:marginalized`.
"""
function get_updates(
    m::Leroux, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "s_idx"
    key = spec.key

    spectral_code = """
        # --- Leroux Spectral Assembly: $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            diag_D = $(p_names.sigma) ./ sqrt.((1.0 .- $(p_names.rho)) .+ 
                                              $(p_names.rho) .* hyper.L .+ M.noise)
            $(p_names.latent_field) = hyper.U * (diag_D .* $(p_names.innovations))
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.$(index_var))
        end
        """

    cholesky_code = """
        # --- Leroux Cholesky Assembly (Dense, AD-Safe): $(key) ---
        let
            Q_template = spec_registry[:$(key)].hyper.Q_template
            rho_val = $(p_names.rho)
            Q_final = (1.0 - rho_val) .* I(size(Q_template, 1)) .+ rho_val .* Q_template
            F = cholesky(Symmetric(Matrix(Q_final) + M.noise * I))
            $(p_names.latent_field) = $(p_names.sigma) .* (F.U \\ $(p_names.innovations))
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.$(index_var))
        end
        """

    cholesky_sparse_code = """
        # --- Leroux Cholesky Assembly (Sparse, Not AD-Safe): $(key) ---
        let
            Q_template = spec_registry[:$(key)].hyper.Q_template
            rho_val = $(p_names.rho)
            Q_final = (1.0 - rho_val) .* sparse(I, size(Q_template)...) .+ 
                      rho_val .* Q_template
            F = cholesky(Symmetric(Q_final + M.noise * I))
            $(p_names.latent_field) = $(p_names.sigma) .* (F.U \\ $(p_names.innovations))
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.$(index_var))
        end
        """

    marginalized_code = """
        # --- Leroux Marginalized Assembly: $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            y_residual = M.y_obs .- $(eta_target)
            log_lik_marginalized_$(key) = _leroux_log_marginal_likelihood(
                y_residual,
                M.$(index_var),
                hyper.n_latent,
                hyper.Q_template,
                hyper.L,
                $(p_names.rho),
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
        error("Unsupported method '$(m.method)' for Leroux component. Use :spectral, :cholesky, :cholesky_sparse, or :marginalized.")
    end
end

"""
    get_effects(m::Leroux, chain, spec::NamedTuple, M::NamedTuple, PS)

Reconstructs the `Leroux` component's effect from posterior samples. This version
is CPU-only and uses modern chain accessors.
"""
function get_effects(
    m::Leroux, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions ---
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
        sigma_samples = _resolve_hyper_samples(
            chain, p_names, p_names_k.sigma, m.sigma, k,
            is_multivariate_model, n_samples;
            as_matrix = true)
        rho_name = _find_parameter(p_names, string(p_names_k.rho), k, is_multivariate_model)

        if isnothing(sigma_samples) || isempty(rho_name)
            @warn "Parameters for Leroux component $(spec.key) (outcome $k) not resolved, " *
                  "and sigma is not a pinned constant. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        rho_samples = get_params_vector(chain, rho_name, 1) # (n_samples, 1)
        
        # Initialize the output matrix for the full latent field
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
            
            I_mat = Matrix{Float64}(I, n_latent, n_latent)
            Q_template = spec.hyper.Q_template
            
            for s in 1:n_samples
                sigma_s = sigma_samples[s, 1]
                rho_s = rho_samples[s, 1]
                y_sig = y_sigma_samples[s]
                
                scale = sigma_s^2 + noise
                inv_sigma_y2 = 1.0 / (y_sig^2 + noise)
                
                Q_base = (1.0 - rho_s) .* I_mat .+ rho_s .* Matrix{Float64}(Q_template)
                for i in 1:n_latent
                    Q_base[i, i] += noise + N_s[i] * inv_sigma_y2 * scale
                end
                
                F = cholesky(Symmetric(Q_base))
                b = S_s .* inv_sigma_y2
                mu = scale .* (F \ b)
                
                z = randn(n_latent)
                x_train = mu .+ sqrt(max(scale, 1e-12)) .* (F.U \ z)
                effect_k_latent[:, s] = x_train
            end
        else
            innovations_name = _find_parameter(
                p_names, string(p_names_k.innovations), k, is_multivariate_model
            )
            if isempty(innovations_name)
                @warn "innovations for Leroux component $(spec.key) (outcome $k) not found. " *
                      "Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end
            innovations_samples = get_params_matrix(chain, innovations_name, n_latent)

            if m.method == :spectral
                U = spec.hyper.U
                L_eig = spec.hyper.L
                # Vectorized Level-3 BLAS across all MCMC draws
                Z = Matrix{Float64}(undef, n_latent, n_samples)
                for s in 1:n_samples
                    sigma_s = sigma_samples[s, 1]
                    rho_s = rho_samples[s, 1]
                    diag_D_s = sigma_s ./ sqrt.((1.0 - rho_s) .+ rho_s .* L_eig .+ noise)
                    Z[:, s] = diag_D_s .* innovations_samples[s, :]
                end
                effect_k_latent = U * Z
            else # :cholesky or :cholesky_sparse
                Q_template = spec.hyper.Q_template
                I_mat = Matrix{Float64}(I, n_latent, n_latent)
                for s in 1:n_samples
                    sigma_s = sigma_samples[s, 1]
                    rho_s = rho_samples[s, 1]
                    innov_s = innovations_samples[s, :]
                    Q_final = (1.0 - rho_s) .* I_mat .+ rho_s .* Q_template
                    F = cholesky(Symmetric(Q_final + noise * I_mat))
                    effect_k_latent[:, s] = sigma_s .* (F.U \ innov_s)
                end
            end
        end

        # Index the reconstructed effects for the full observation set
        indexed_effects = effect_k_latent[s_idx_full, :]
        push!(structured_effects, indexed_effects)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
