"""
    BCGN <: ComponentModel

A component for a Bipartite Graph Convolutional Network (BCGN), which models a
latent spatial field on one partition of a bipartite graph using a Gaussian
Markov Random Field (GMRF) approach. This component is designed for data
structured on a bipartite graph, where nodes are divided into two disjoint sets,
and edges only connect nodes from different sets.

# Version
v1.0.0

# Mathematical Summary
The BCGN component models a latent spatial field on one partition of a bipartite
graph. Given an adjacency matrix \$W\$ for a unipartite graph, the model first
attempts to find a 2-coloring to partition the graph into two sets, \$V_1\$ and
\$V_2\$. This results in a bipartite adjacency matrix \$B\$ representing connections
between the two sets.

The spatial correlation for nodes in one partition (e.g., \$V_1\$) is then induced
by their shared connections in the other partition (\$V_2\$). This is achieved by
creating a **one-mode projection** of the bipartite graph onto \$V_1\$. The adjacency
matrix for this projected graph is given by \$W_{\\text{proj}} = B B^T\$.

From this projected adjacency matrix, a standard graph Laplacian is constructed:
\$Q = D_{\\text{proj}} - W_{\\text{proj}}\$
where \$D_{\\text{proj}}\$ is the diagonal degree matrix of \$W_{\\text{proj}}\$. The latent field
\$\\phi\$ is then modeled as a GMRF with this precision matrix, scaled by \$\\sigma^2\$:
\$\\phi \\sim \\mathcal{N}(0, (\\sigma^2 Q)^{-1})\$.

To ensure identifiability against a global intercept, a soft sum-to-zero constraint
is applied to the latent field.

# Computational Methods
- `:spectral` (Default, AD-friendly): Uses spectral decomposition of the precision
  matrix for efficient and AD-compatible sampling.
- `:cholesky` (AD-friendly): Uses dense Cholesky factorization of the precision
  matrix. AD-compatible but less efficient than spectral for large graphs.
- `:cholesky_sparse` (Didactic, Not AD-friendly): Uses sparse Cholesky factorization.
  Not AD-compatible for gradient-based samplers but retained as a didactic alternative.

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

# Key References
- Kipf, T. N., & Welling, M. (2016). *Semi-supervised classification with graph
  convolutional networks*. arXiv preprint arXiv:1609.02907.
- Rue, H., & Held, L. (2005). *Gaussian Markov Random Fields: Theory and Applications*. CRC Press.
"""
struct BCGN <: ComponentModel
    sigma::Union{UnivariateDistribution, Real}
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:bcgn] = BCGN
COMPONENT_CONSTRUCTORS[:bcgn] = (p, params) -> BCGN(
    p.sigma, get(params, :method, :spectral)
)
MODEL_TO_STRUCTURE_MAP[:bcgn] = :spatial


"""
    get_precomputes(m::BCGN, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes the precision matrix `Q_template` from the one-mode projection of the
bipartite graph, along with its spectral decomposition, Cholesky factorization,
and a mapping matrix to link latent effects to observations. This is a CPU-only
implementation.
"""
function get_precomputes(m::BCGN, M::NamedTuple, mod_data::Dict)::NamedTuple
    W = get(M, :W, nothing)
    if isnothing(W)
        error("The `bcgn` model requires an adjacency matrix `W`.")
    end

    bipartite_info = adjacency_to_bipartite(W)
    B = bipartite_info.bipartite_adj
    set1_indices = bipartite_info.set1

    if isempty(B) || all(iszero, B)
        error("BCGN component requires a non-empty bipartite adjacency matrix.")
    end

    W_proj = B * B'
    W_proj[diagind(W_proj)] .= 0
    W_proj = dropzeros(W_proj)

    D_proj = spdiagm(0 => vec(sum(W_proj, dims=2)))
    Q_template = D_proj - W_proj
    
    n_latent = size(Q_template, 1)

    set1_map = Dict(original_idx => new_idx for (new_idx, original_idx) in enumerate(set1_indices))
    
    N_obs = M.y_N
    mapping_matrix = spzeros(Float64, N_obs, n_latent)
    for i in 1:N_obs
        original_s_idx = M.s_idx[i]
        if haskey(set1_map, original_s_idx)
            latent_idx = set1_map[original_s_idx]
            mapping_matrix[i, latent_idx] = 1.0
        end
    end

    eig_decomp = eigen(Symmetric(Matrix(Q_template)))
    U = eig_decomp.vectors
    L = eig_decomp.values
    
    F = cholesky(Symmetric(Matrix(Q_template) + M.noise * I))

    return (
        Q_template=Q_template,
        U=U,
        L=L,
        n_latent=n_latent,
        cholesky_factor=F,
        mapping_matrix=mapping_matrix,
        set1_indices=set1_indices
    )
end


"""
    _bcgn_log_marginal_likelihood(y_residual, mapping_matrix, Q_template, L_eig, sigma,
      y_sigma, noise=1e-6)

Computes the exact log marginal likelihood for a BCGN bipartite spatial component integrated
  out analytically.
"""
function _bcgn_log_marginal_likelihood(
    y_residual::AbstractVector{T},
    mapping_matrix::AbstractMatrix,
    Q_template::AbstractMatrix,
    L_eig::AbstractVector,
    sigma::T,
    y_sigma::T,
    noise::Real=1e-6
) where {T}
    N = length(y_residual)
    K = size(mapping_matrix, 2)
    T_num = promote_type(T, typeof(noise))
    
    inv_sigma_y2 = one(T_num) / (y_sigma^2 + T_num(noise))
    scale = sigma^2 + T_num(noise)
    
    ATA = Matrix{T_num}(mapping_matrix' * mapping_matrix)
    ATy = Vector{T_num}(mapping_matrix' * y_residual)
    
    Q_base = Matrix{T_num}(Q_template) .+ (scale * inv_sigma_y2) .* ATA
    for k in 1:K
        Q_base[k, k] += T_num(noise)
    end
    
    F = cholesky(Symmetric(Q_base))
    
    # Laplacian has rank deficiency 1
    valid_eigs = filter(x -> x > 1e-6, L_eig)
    log_det_prior = isempty(valid_eigs) ? zero(T_num) : sum(log.(valid_eigs .+ T_num(noise)))
    log_det_diff = - max(K - 1, 1) * log(scale) + log_det_prior - 2 * sum(log.(diag(F.U)))
    
    b = ATy .* inv_sigma_y2
    v = F.L \ b
    quad_term = scale * dot(v, v)
    
    log_lik = - (N / 2) * log(2 * T_num(pi) * (y_sigma^2 + T_num(noise))) -
              (inv_sigma_y2 / 2) * dot(y_residual, y_residual) +
              (1 / 2) * log_det_diff +
              (1 / 2) * quad_term
              
    return log_lik
end

"""
    get_priors(m::BCGN, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for the innovations and overall scale (`sigma`).
"""
function get_priors(
    m::BCGN, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent
    if m.method == :marginalized
        return """
        # Priors for BCGN component: $(spec.key)
        $(_prior_or_constant(p_names.sigma, m.sigma))
        """
    else
        return """
        # Priors for BCGN component: $(spec.key)
        $(_prior_or_constant(p_names.sigma, m.sigma))
        $(p_names.innovations) ~ MvNormal(zeros(T, $(n_latent)), I)
        """
    end
end

"""
    get_updates(m::BCGN, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates code to compute the BCGN effect and add it to the linear predictor `eta`.
Supports methods: `:spectral`, `:cholesky`, `:cholesky_sparse`, and `:marginalized`.
"""
function get_updates(
    m::BCGN, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    n_latent = spec.hyper.n_latent

    spectral_code = """
        # --- BCGN Component (Spectral): $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            U = hyper.U
            L = hyper.L
            diag_D = $(p_names.sigma) ./ sqrt.(L .+ M.noise)
            diag_D[L .< 1e-6] .= 0.0
            
            latent_field = U * (diag_D .* $(p_names.innovations))
            $(p_names.latent_field) = hyper.mapping_matrix * latent_field
            $(eta_target) = $(eta_target) .+ $(p_names.latent_field)
        end
    """

    cholesky_code = """
        # --- BCGN Component (Cholesky, AD-Safe): $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            F = hyper.cholesky_factor
            latent_field_unscaled = F.L' \\ $(p_names.innovations)
            
            Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * $(n_latent)), sum(latent_field_unscaled))
            
            latent_field = latent_field_unscaled .* $(p_names.sigma)
            $(p_names.latent_field) = hyper.mapping_matrix * latent_field
            $(eta_target) = $(eta_target) .+ $(p_names.latent_field)
        end
    """

    cholesky_sparse_code = """
        # --- BCGN Component (Sparse Cholesky, Not AD-Safe): $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            Q = hyper.Q_template
            F = cholesky(Symmetric(Q + M.noise * I))
            latent_field_unscaled = F.L' \\ $(p_names.innovations)
            
            Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * $(n_latent)), sum(latent_field_unscaled))
            
            latent_field = latent_field_unscaled .* $(p_names.sigma)
            $(p_names.latent_field) = hyper.mapping_matrix * latent_field
            $(eta_target) = $(eta_target) .+ $(p_names.latent_field)
        end
    """

    marginalized_code = """
        # --- BCGN Component (Marginalized): $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            y_residual = M.y_obs .- $(eta_target)
            log_lik_marginalized_$(key) = _bcgn_log_marginal_likelihood(
                y_residual,
                hyper.mapping_matrix,
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
        error("Unsupported method '$(m.method)' for BCGN component. Use `:spectral`, `:cholesky`, `:cholesky_sparse`, or `:marginalized`.")
    end
end

"""
    get_effects(m::BCGN, chain, spec::NamedTuple, M::NamedTuple, PS)

Reconstructs the `BCGN` component's effect from the MCMC chain's posterior samples.
This version is CPU-only and uses modern chain accessors.
"""
function get_effects(
    m::BCGN, chain, spec::NamedTuple, M::NamedTuple,
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
    s_idx_train = Array(M.s_idx)
    
    s_idx_full = if !isnothing(PS) && hasproperty(PS.data, :s_idx)
        vcat(s_idx_train, PS.data.s_idx)
    else
        s_idx_train
    end
    N_total = length(s_idx_full)

    # --- Mapping Matrix Construction on CPU ---
    set1_map = Dict(original_idx => new_idx for (new_idx,
        original_idx) in enumerate(spec.hyper.set1_indices))
    
    mapping_matrix_cpu = spzeros(Float64, N_total, n_latent)
    for i in 1:N_total
        original_s_idx = s_idx_full[i]
        if haskey(set1_map, original_s_idx)
            latent_idx = set1_map[original_s_idx]
            mapping_matrix_cpu[i, latent_idx] = 1.0
        end
    end
    mapping_matrix_full = Matrix(mapping_matrix_cpu)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        sigma_samples = _resolve_hyper_samples(
            chain, p_names, p_names_k.sigma, m.sigma, k,
            is_multivariate_model, n_samples)

        if isnothing(sigma_samples)
            @warn "Parameters for $(spec.key) (outcome $k) not resolved, and " *
                  "sigma is not a pinned constant. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end


        # Initialize the output matrix for latent effects on the CPU
        effect_k_matrix = zeros(Float64, n_latent, n_samples)

        # --- Sample-wise Reconstruction on the CPU ---
        if m.method == :marginalized
            y_sigma_name = _find_parameter(p_names, "y_sigma", k, is_multivariate_model)
            y_sigma_samples = if !isempty(y_sigma_name)
                get_params_vector(chain, y_sigma_name, 1)[:, 1]
            else
                fill(1.0, n_samples)
            end
            
            y_vec = M.y_obs isa AbstractMatrix ? M.y_obs[:, k] : M.y_obs
            A_train = spec.hyper.mapping_matrix
            ATA = Matrix{Float64}(A_train' * A_train)
            ATy = Vector{Float64}(A_train' * y_vec)
            
            for j in 1:n_samples
                sig = sigma_samples[j]
                y_sig = y_sigma_samples[j]
                
                scale = sig^2 + noise
                inv_sigma_y2 = 1.0 / (y_sig^2 + noise)
                
                Q_base = Matrix{Float64}(spec.hyper.Q_template) .+ (scale * inv_sigma_y2) .* ATA
                for i in 1:n_latent
                    Q_base[i, i] += noise
                end
                
                F = cholesky(Symmetric(Q_base))
                b = ATy .* inv_sigma_y2
                mu = scale .* (F \ b)
                
                z = randn(n_latent)
                effect_k_matrix[:, j] = mu .+ sqrt(max(scale, 1e-12)) .* (F.U \ z)
            end
        else
            innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)
            innovations_samples = get_params_vector(chain, innovations_name, n_latent)

            if m.method == :spectral
                U = spec.hyper.U
                L = spec.hyper.L
                innovations_samples_T = innovations_samples'

                for j in 1:n_samples
                    sigma_j = sigma_samples[j]
                    
                    diag_D = sigma_j ./ sqrt.(L .+ noise)
                    diag_D[L .< 1e-6] .= 0.0
                    effect_k_matrix[:, j] = U * (diag_D .* innovations_samples_T[:, j])
                end
            else # :cholesky or :cholesky_sparse
                F = spec.hyper.cholesky_factor
                innovations_samples_T = innovations_samples'

                for j in 1:n_samples
                    sigma_j = sigma_samples[j]

                    latent_field_unscaled = F.L' \ innovations_samples_T[:, j]
                    latent_field_unscaled .-= mean(latent_field_unscaled)
                    effect_k_matrix[:, j] = latent_field_unscaled .* sigma_j
                end
            end
        end
        
        # Apply mapping to get observation-level effects
        indexed_effects = mapping_matrix_full * effect_k_matrix
        push!(structured_effects, indexed_effects)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
