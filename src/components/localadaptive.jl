"""
    LocalAdaptive <: ComponentModel

A component for a Local Adaptive spatial effect. This model combines a global
smoothing structure (based on a Leroux-style precision matrix) with local,
cluster-specific mean effects. This allows the model to capture both smooth spatial
trends and abrupt shifts between distinct spatial regions.

# Version
v1.0.0

# Mathematical Summary
The `LocalAdaptive` component models a latent spatial field \$\\phi\$ as a non-zero
mean Gaussian Markov Random Field (GMRF). The mean of the field, \$\\boldsymbol{\\mu}\$,
is not constant but varies by spatial cluster, while the precision matrix,
\$\\mathbf{Q}\$, captures global spatial correlation.

\$\\boldsymbol{\\phi} \\sim \\mathcal{N}(\\boldsymbol{\\mu}, (\\sigma^2 \\mathbf{Q})^{-1})\$

1.  **Mean Structure (\$\\boldsymbol{\\mu}\$)**: The spatial domain is partitioned into \$k\$
    clusters using k-means on the area centroids. A separate mean effect, \$\\mu_g\$,
    is estimated for each cluster \$g\$. For any spatial unit \$i\$ belonging to
    cluster \$g\$, its mean is \$\\mu_i = \\mu_g\$. A sum-to-zero constraint is applied
    to the cluster means for identifiability.

2.  **Precision Structure (\$\\mathbf{Q}\$)**: The precision matrix is a proper CAR model
    (Leroux-style), defined as a convex combination of an identity matrix
    \$\\mathbf{I}\$ and a scaled ICAR precision matrix \$\\mathbf{Q}_{ICAR}\$:
    \$\\mathbf{Q} = (1-\\rho)\\mathbf{I} + \\rho\\mathbf{Q}_{ICAR}\$
    This allows the model to smoothly interpolate between unstructured random
    effects (\$\\rho=0\$) and a fully structured ICAR model (\$\\rho=1\$).

# Computational Methods
- `:spectral` (Default, AD-friendly): Regularizes coefficients using a spectral
  decomposition of the ICAR precision matrix. Recommended for gradient-based samplers.
- `:cholesky` (AD-friendly): Uses a pre-computed dense Cholesky factorization of the
  full Leroux precision matrix.
- `:cholesky_sparse` (Didactic, Not AD-friendly): Uses sparse Cholesky factorization,
  which is not compatible with most AD backends.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `region`) passed to `random()`.
  - An adjacency matrix `W` passed as a keyword argument to `@tbm`.
  - Spatial coordinates (`s_x`, `s_y`) in the data frame for clustering.
- **Optional (in `random()` call)**:
  - `n_clusters`: `Int`, the number of spatial clusters to identify. Default: `5`.
  - `rho`: A `UnivariateDistribution` for the prior on the mixing parameter. Default: `Beta(1,1)`.
  - `sigma`: A `UnivariateDistribution` for the prior on the overall standard deviation.
    Default: `Exponential(1.0)`.
  - `method`: `Symbol`, specifying the computational method. Default: `:spectral`.

# Outputs (Parameter Names)
- `sigma_<key>`: The overall marginal standard deviation.
- `rho_<key>`: The mixing parameter.
- `innovations_<key>`: The raw standard normal innovations for the centered spatial effect.
- `cluster_innovations_<key>`: The raw standard normal innovations for the cluster means.
- `latent_<key>`: The reconstructed latent spatial field.

# Key References
- Gelfand, A. E., Schmidt, A. M., Banerjee, S., & Sirmans, C. F. (2005).
  *Nonstationary multivariate process modeling through spatially varying
  coregionalization*. Test, 14(2), 263-312.
"""
struct LocalAdaptive <: ComponentModel
    rho::UnivariateDistribution
    sigma::Union{UnivariateDistribution, Real}
    n_clusters::Int
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:localadaptive] = LocalAdaptive

COMPONENT_CONSTRUCTORS[:localadaptive] = (p, params) -> LocalAdaptive(
    p.rho,
    p.sigma,
    get(params, :n_clusters, 5),
    get(params, :method, :spectral)
)

MODEL_TO_STRUCTURE_MAP[:localadaptive] = :spatial

function get_precomputes(m::LocalAdaptive, M::NamedTuple, mod_data::Dict)::NamedTuple
    s_N = get(M, :s_N, 0)
    if s_N == 0
        error("The `localadaptive` model requires a spatial context (`s_N`), but it has not been established. Ensure a spatial variable and adjacency matrix `W` are provided.")
    end
    
    data = M.data

    # The localadaptive model requires centroids for clustering.
    local centroids
    if haskey(M, :centroids)
        centroids = M.centroids
    else
        coord_cols = if haskey(mod_data, :variables) && length(mod_data[:variables]) >= 2
            (Symbol(mod_data[:variables][1]), Symbol(mod_data[:variables][2]))
        else
            try
                _detect_xy_columns(data)
            catch
                nothing
            end
        end

        s_idx_col = if hasproperty(data, :s_idx)
            :s_idx
        elseif haskey(M, :s_idx_var) && hasproperty(data, Symbol(M.s_idx_var))
            Symbol(M.s_idx_var)
        else
            nothing
        end

        if coord_cols !== nothing && s_idx_col !== nothing &&
           hasproperty(data, coord_cols[1]) && hasproperty(data, coord_cols[2])
            cx, cy = coord_cols
            gdf = groupby(data, s_idx_col)
            unique_coords_df = DataFrames.combine(gdf, [cx, cy] .=> first, renamecols=false)
            coord_map = Dict(row[s_idx_col] => (row[cx], row[cy]) for row in eachrow(unique_coords_df))

            if length(coord_map) < s_N
                error("The `localadaptive` model requires coordinates for all $(s_N) spatial units, but only found coordinates for $(length(coord_map)) unique units in the data. Please provide a complete `centroids` vector as a keyword argument.")
            end

            centroids = [(coord_map[i][1], coord_map[i][2]) for i in 1:s_N]
        else
            error("The `localadaptive()` model requires centroids for clustering. Provide them via the `centroids` keyword argument, or ensure spatial coordinates and indices (s_idx) are in the data frame.")
        end
    end
    
    if length(centroids) != s_N
        error("The number of provided centroids ($(length(centroids))) does not match the number of spatial units s_N ($(s_N)).")
    end

    n_clusters = m.n_clusters
    if length(centroids) < n_clusters
        @warn "Number of spatial units ($(length(centroids))) is less than the requested number of clusters ($(n_clusters)). Adjusting n_clusters to $(length(centroids))."
        n_clusters = length(centroids)
    end
    
    centroids_matrix = hcat([c[1] for c in centroids], [c[2] for c in centroids])'
    kmeans_result = kmeans(centroids_matrix, n_clusters; maxiter=200, display=:none)
    
    cluster_assignments = assignments(kmeans_result)
    n_clusters_final = nclusters(kmeans_result)

    # Build the ICAR precision matrix template for the Leroux structure.
    template = build_structure_template(:icar, s_N; W=M.W)
    
    return (
        Q_icar=template.matrix,
        U=template.U,
        L=template.L,
        n_latent=s_N,
        n_clusters=n_clusters_final,
        cluster_assignments=cluster_assignments
    )
end

function get_priors(
    m::LocalAdaptive, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent
    n_clusters = spec.hyper.n_clusters
    
    return """
    $(_prior_or_constant(p_names.sigma, m.sigma))
    $(_prior_or_constant(p_names.rho, m.rho))
    $(p_names.innovations) ~ MvNormal(zeros(T, $(n_latent)), I)
    $(p_names.innovations_cluster) ~ MvNormal(zeros(T, $(n_clusters)), I)
    """
end

function get_updates(
    m::LocalAdaptive, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    n_latent = spec.hyper.n_latent
    
    common_code = """
        # --- LocalAdaptive Component: $(key) ($(m.method)) ---
        let
            hyper = spec_registry[:$(key)].hyper
            cluster_means_unscaled = $(p_names.innovations_cluster)
            cluster_means = cluster_means_unscaled .- mean(cluster_means_unscaled)
            mu_field = cluster_means[hyper.cluster_assignments]
    """

    spectral_code = """
        $(common_code)
            diag_D_leroux = $(p_names.sigma) ./ sqrt.((1.0 - $(p_names.rho)) .+
              $(p_names.rho) .* hyper.L .+ M.noise)
            latent_centered = hyper.U * (diag_D_leroux .* $(p_names.innovations))
            
            $(p_names.latent_field) = mu_field .+ latent_centered
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.s_idx)
        end
    """

    cholesky_code = """
        $(common_code)
            Q_leroux = (1.0 - $(p_names.rho)) .* I($(n_latent)) .+ $(p_names.rho) .* hyper.Q_icar
            F = cholesky(Symmetric(Matrix(Q_leroux) + M.noise * I))
            
            latent_centered = F.L' \\ $(p_names.innovations)
            
            $(p_names.latent_field) = mu_field .+ latent_centered .* $(p_names.sigma)
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.s_idx)
        end
    """

    cholesky_sparse_code = """
        $(common_code)
            Q_leroux = (1.0 - $(p_names.rho)) .* sparse(I, $(n_latent), $(n_latent)) .+
              $(p_names.rho) .* hyper.Q_icar
            F = cholesky(Symmetric(Q_leroux + M.noise * I))
            
            latent_centered = F.L' \\ $(p_names.innovations)
            
            $(p_names.latent_field) = mu_field .+ latent_centered .* $(p_names.sigma)
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.s_idx)
        end
    """
    
    if m.method == :spectral
        return spectral_code
    elseif m.method == :cholesky
        return cholesky_code
    elseif m.method == :cholesky_sparse
        return cholesky_sparse_code
    else; error("Unsupported method '$(m.method)' for LocalAdaptive component."); end
end

"""
    get_effects(m::LocalAdaptive, chain, spec::NamedTuple, M::NamedTuple, PS)

Reconstructs the `LocalAdaptive` spatial effect from posterior samples. This version
is CPU-only and uses modern chain accessors.
"""
function get_effects(
    m::LocalAdaptive, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions ---
    n_samples = if occursin("FlexiChain", string(typeof(chain)))
        size(chain, 1) * FlexiChains.nchains(chain)
    else
        size(chain, 1) * size(chain, 3)
    end
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = string.(keys(chain))
    
    hyper = spec.hyper
    noise = M.noise
    n_latent = hyper.n_latent
    n_clusters = hyper.n_clusters

    # --- Index Handling: Combine training and prediction sets on CPU ---
    s_idx_train = M.s_idx # Spatial indices for training data
    s_idx_full = if !isnothing(PS) && hasproperty(PS.data, :s_idx) # If prediction set is provided
        vcat(s_idx_train, PS.data.s_idx) # Combine training and prediction indices
    else
        s_idx_train # Otherwise, use only training indices
    end
    N_total = length(s_idx_full) # Total number of observations (training + prediction)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
            # Only `sigma` is pinnable here: `rho` and both innovation blocks are always
            # sampled. A pinned `sigma` is not a chain parameter, so resolve it with the
            # constant fallback rather than reading it by name.
            rho_name = _find_parameter(p_names, string(p_names_k.rho), k, is_multivariate_model)
            innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)
            innovations_cluster_name = _find_parameter(p_names, string(p_names_k.innovations_cluster), k,
                is_multivariate_model)
            sigma_samples = _resolve_hyper_samples(
                chain, p_names, p_names_k.sigma, m.sigma, k, is_multivariate_model,
                n_samples; as_matrix = true)

            if isnothing(sigma_samples) || isempty(rho_name) || isempty(innovations_name) ||
                isempty(innovations_cluster_name)
                @warn "Parameters for LocalAdaptive component $(spec.key) (outcome $k) not " *
                      "resolved, and sigma is not a pinned constant. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end

            # Extract posterior samples (these are on the CPU)
            rho_samples = get_params_vector(chain, rho_name, 1) # (n_samples, 1)
            innovations_samples = get_params_matrix(chain, innovations_name, n_latent) # (n_samples, n_latent)
        innovations_cluster_samples = get_params_matrix(chain, innovations_cluster_name,
            n_clusters) # (n_samples, n_clusters)
        
        # Initialize the output matrix for the full latent field
        latent_field_matrix = zeros(Float64, n_latent, n_samples)
        
        # --- Sample-wise Reconstruction ---
        for i in 1:n_samples # Iterate over each posterior sample
            sigma_s = sigma_samples[i, 1] # Sigma for current sample
            rho_s = rho_samples[i, 1] # Rho for current sample
            innov_s = innovations_samples[i, :] # Innovations for current sample
            cluster_innov_s = innovations_cluster_samples[i, :] # Cluster innovations for current sample

            # Reconstruct the mean field
            cluster_means = cluster_innov_s .- mean(cluster_innov_s)
            mu_field = cluster_means[hyper.cluster_assignments]
            
            local latent_centered
            if m.method == :spectral
                U = hyper.U
                L = hyper.L
                diag_D_leroux = sigma_s ./ sqrt.((1.0 - rho_s) .+ rho_s .* L .+ noise)
                latent_centered = U * (diag_D_leroux .* innov_s)
                latent_field_matrix[:, i] = mu_field .+ latent_centered
            else # :cholesky or :cholesky_sparse
                Q_icar = hyper.Q_icar
                I_mat = Matrix(I, n_latent, n_latent)
                Q_leroux = (1.0 - rho_s) .* I_mat .+ rho_s .* Q_icar
                F = cholesky(Symmetric(Q_leroux + noise * I_mat))
                latent_centered = F.L' \ innov_s
                latent_field_matrix[:, i] = mu_field .+ latent_centered .* sigma_s
            end
        end
        
        # Index the reconstructed effects for the full observation set
        indexed_effects = latent_field_matrix[s_idx_full, :]
        push!(structured_effects, indexed_effects)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end 
