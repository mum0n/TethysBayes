"""
    Besag <: ComponentModel

A component for a Besag model, also known as an Intrinsic Conditional Autoregressive
(ICAR) model. This is a fundamental model for spatial data on a lattice or graph,
where the value at a location is assumed to be conditionally dependent on the
average of its neighbors.

# Version
v1.0.0

# Mathematical Summary
The Besag model defines a Gaussian Markov Random Field (GMRF) with a singular
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

# Assumptions
- The spatial process is locally smooth, with values at neighboring locations being
- The provided adjacency matrix `W` can represent either a single connected graph or
  a disconnected graph with spatial islands. When multiple disconnected components or
  isolated units are detected, `build_structure_template` automatically performs sub-graph
  island normalization and component deflation (Freni-Sterrantino et al., 2018).

# Best Use Case
Modeling structured spatial random effects for areal or lattice data, such as
disease mapping, real estate analysis, or ecological modeling, where there is a
strong prior belief in local spatial smoothing.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `s_idx`).
  - An adjacency matrix `W` passed as a keyword argument to `@tbm`.
- **Optional (in `random()` call)**:
  - `sigma`: `UnivariateDistribution`, prior for the marginal standard deviation. Default:
    `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:spectral`, `:cholesky`, `:cholesky_sparse`).
    Default: `:spectral`.

# Outputs (Parameter Names)
- `sigma_<key>`: The marginal standard deviation of the latent field.
- `innovations_<key>`: The raw standard normal innovations for the latent field.
- `latent_<key>`: The reconstructed latent spatial field.


# Key References
- Besag, J. (1974). Spatial interaction and the statistical analysis of lattice
  systems. *Journal of the Royal Statistical Society: Series B (Methodological)*,
  36(2), 192-225.
- Rue, H., & Held, L. (2005). *Gaussian Markov Random Fields: Theory and
  Applications*. CRC Press.
- Wikipedia: Conditional autoregressive model

# Fields
- `sigma::UnivariateDistribution`: The prior for the marginal standard deviation of
  the conditional spatial effect.
- `method::Symbol`: The computational method. Can be `:spectral` (default, AD-safe),
  `:cholesky` (AD-safe, dense), or `:cholesky_sparse` (didactic, not AD-safe).
"""
struct Besag <: ComponentModel
    sigma::Union{UnivariateDistribution, Real}
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:besag] = Besag
COMPONENT_CONSTRUCTORS[:besag] = (p, params) -> Besag(
    p.sigma, get(params, :method, :spectral)
)

MODEL_TO_STRUCTURE_MAP[:besag] = :spatial

"""
    get_precomputes(m::Besag, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes the graph Laplacian (`Q_template`), its Cholesky factorization, and its
spectral decomposition (`U`, `L`) for use by different sampling methods. This is a
CPU-only implementation.
"""
function get_precomputes(m::Besag, M::NamedTuple, mod_data::Dict)::NamedTuple
    n = (hasproperty(M, :W) && !isempty(M.W) && isa(M.W, AbstractMatrix)) ? size(M.W, 1) : M.s_N
    W = M.W
    
    # build_structure_template returns CPU arrays
    template = build_structure_template(:besag, n; W=W)
    
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
    _besag_log_marginal_likelihood(y_residual, s_idx, s_N, Q_template, L_eig, sigma,
      y_sigma, noise=1e-6)

Computes the exact log marginal likelihood for a Besag (ICAR) spatial component integrated
  out analytically.
"""
function _besag_log_marginal_likelihood(
    y_residual::AbstractVector{T},
    s_idx::AbstractVector{Int},
    s_N::Int,
    Q_template::AbstractMatrix,
    L_eig::AbstractVector,
    sigma::T,
    y_sigma::T,
    noise::Real=1e-6
) where {T}
    return _icar_log_marginal_likelihood(y_residual, s_idx, s_N, Q_template, L_eig, sigma,
        y_sigma, noise)
end

"""
    get_priors(m::Besag, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for the scale parameter `sigma` and the raw innovations `innovations`.
"""
function get_priors(
    m::Besag, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent
    
    if m.method == :marginalized
        return """
        $(_prior_or_constant(p_names.sigma, m.sigma))
        """
    else
        return """
        $(_prior_or_constant(p_names.sigma, m.sigma))
        $(p_names.innovations) ~ MvNormal(zeros(T, $(n_latent)), I)
        """
    end
end

function get_updates(m::Besag, spec::NamedTuple, arch::String, outcome_idx::Union{Int,
    Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    
    spectral_code = """
        # --- Besag Component (Spectral): $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            diag_D = $(p_names.sigma) ./ sqrt.(hyper.L .+ M.noise)
            diag_D[hyper.L .<= (1e-10 * maximum(hyper.L))] .= 0.0   # zero EVERY null direction (a disconnected graph has more than one)
            
            $(p_names.latent_field) = hyper.U * (diag_D .* $(p_names.innovations))
            
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.s_idx)
        end
    """

    cholesky_code = """
        # --- Besag Component (Dense Cholesky): $(key) ---
        let
            F = spec_registry[:$(key)].hyper.cholesky_factor
            latent_field_unscaled = F.L' \\ $(p_names.innovations)
            latent_field_centered = latent_field_unscaled .- mean(latent_field_unscaled)
            
            $(p_names.latent_field) = latent_field_centered .* $(p_names.sigma)
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.s_idx)
        end
    """

    cholesky_sparse_code = """
        # --- Besag Component (Sparse Cholesky): $(key) ---
        let
            Q = spec_registry[:$(key)].hyper.Q_template
            F = cholesky(Symmetric(Q + M.noise * I))
            
            L_sparse = sparse(F.L)
            latent_field_unscaled = L_sparse' \\ $(p_names.innovations)
            
            latent_field_centered = latent_field_unscaled .- mean(latent_field_unscaled)
            
            $(p_names.latent_field) = latent_field_centered .* $(p_names.sigma)
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.s_idx)
        end
    """

    marginalized_code = """
        # --- Besag Component (Marginalized): $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            y_residual = M.y_obs .- $(eta_target)
            log_lik_marginalized_$(key) = _besag_log_marginal_likelihood(
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
        error("Unsupported method '$(m.method)' for Besag component. Use :spectral, :cholesky, :cholesky_sparse, or :marginalized.")
    end
end

"""
    get_effects(m::Besag, chain, spec::NamedTuple, M::NamedTuple, PS)

Reconstructs the `Besag` component's effect from posterior samples, applying a
sum-to-zero constraint for identifiability. This function dispatches on the method
used during sampling and is CPU-only.
"""
function get_effects(
    m::Besag, chain, spec::NamedTuple, M::NamedTuple,
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
    s_idx_train = M.s_idx # Spatial indices for training data
    s_idx_full = if !isnothing(PS) && hasproperty(PS.data, :s_idx) # If prediction set is provided
        vcat(s_idx_train, PS.data.s_idx) # Combine training and prediction indices
    else
        s_idx_train # Otherwise, use only training indices
    end
    N_total = length(s_idx_full) # Total number of observations (training + prediction)

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
                sig = sigma_samples[j, 1]
                y_sig = y_sigma_samples[j]
                
                scale = sig^2 + noise
                inv_sigma_y2 = 1.0 / (y_sig^2 + noise)
                
                Q_base = Matrix{Float64}(Q_template)
                for s in 1:n_latent
                    Q_base[s, s] += noise + N_s[s] * inv_sigma_y2 * scale
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
            innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)
            innovations_samples = get_params_matrix(chain, innovations_name, n_latent)

            if m.method == :spectral
                U = spec.hyper.U
                L = spec.hyper.L
                
                for j in 1:n_samples
                    sigma_j = sigma_samples[j, 1]
                    innov_j = innovations_samples[j, :]
                    
                    diag_D = sigma_j ./ sqrt.(L .+ noise)
                    _zero_null_modes!(diag_D, L)   # zero EVERY null direction; a disconnected graph has more than one
                    effect_k_latent[:, j] = U * (diag_D .* innov_j)
                end
            else # :cholesky or :cholesky_sparse
                F = spec.hyper.cholesky_factor
                
                for j in 1:n_samples
                    sigma_j = sigma_samples[j, 1]
                    innov_j = innovations_samples[j, :]

                    latent_field_unscaled = F.L' \ innov_j
                    latent_field_centered = latent_field_unscaled .- mean(latent_field_unscaled)
                    effect_k_latent[:, j] = latent_field_centered .* sigma_j
                end
            end
        end

        # Index the reconstructed latent effects to match the observation indices
        indexed_effects = effect_k_latent[s_idx_full, :]
        push!(structured_effects, indexed_effects)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end 
