"""
    Eigen <: ComponentModel

A component model for Bayesian Principal Component Analysis (PCA) or Factor Analysis.
It decomposes a set of multivariate outcomes into a smaller set of orthogonal latent
factors, and uses the dominant latent factor as a predictor in the main model.

# Version
v1.0.0

# Mathematical Summary
The `Eigen` component models a set of \$N_{vars}\$ observed variables 
\$Y \\in \\mathbb{R}^{N_{obs} \\times N_{vars}}\$ as a linear combination of 
\$N_{factors}\$ latent factors \$F \\in \\mathbb{R}^{N_{obs} \\times N_{factors}}\$
and a loadings matrix \$L \\in \\mathbb{R}^{N_{vars} \\times N_{factors}}\$, plus 
idiosyncratic noise \$\\Psi\$:
\$Y = F L^T + \\Psi\$
where \$F_{i,j} \\sim \\mathcal{N}(0, \\sigma_{pca,j}^2)\$ and 
\$\\Psi_{i,j} \\sim \\mathcal{N}(0, \\sigma_{pdef,j}^2)\$.

To ensure the loadings matrix \$L\$ is identifiable and orthonormal, it is constructed
using a sequence of Householder reflections. This parameterization ensures 
\$L^T L = I\$. The latent factors \$F\$ are typically sampled from standard normal 
distributions.

The effect added to the linear predictor `eta` of the main model is the first
(dominant) latent factor \$F_{:,1}\$.

# Computational Methods
- `:noncentered` (Default, AD-friendly): A non-centered parameterization where the
  latent factors are constructed from standard normal innovations. Recommended for NUTS.
- `:centered` (Didactic, Not AD-friendly): A centered parameterization where the
  latent factors are sampled directly from their scaled distribution. This can be
  less efficient for MCMC and is retained for didactic purposes.

# Inputs
- **Required**:
  - One or more variables passed to `eigen()` representing the multivariate data to 
    be decomposed.
- **Optional (in `eigen()` call)**:
  - `n_factors`: `Int`, the number of latent factors to extract. Default: `1`.
  - `pca_sd`: `UnivariateDistribution`, prior for the standard deviations of the 
    factors. Default: `Exponential(1.0)`.
  - `pdef_sd`: `UnivariateDistribution`, prior for the standard deviations of the 
    uniquenesses. Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:noncentered` or `:centered`). 
    Default: `:noncentered`.

# Outputs (Parameter Names)
- `v_raw_<key>`: Raw vectors for the Householder transformation.
- `pca_sd_<key>`: Standard deviations of the principal components (factors).
- `pdef_sd_<key>`: Standard deviations of the uniquenesses (residuals).
- `factors_flat_<key>`: Raw standard normal innovations for the latent factors 
  (for `:noncentered`).
- `latent_<key>`: The latent factors (for `:centered`).

# Key References
- Tipping, M. E., & Bishop, C. M. (1999). *Probabilistic principal component 
  analysis*. Journal of the Royal Statistical Society: Series B, 61(3), 611-622.
- Hoff, P. D. (2009). *A first course in Bayesian statistical methods*. Springer.
"""
struct Eigen <: ComponentModel
    pca_sd::Distribution
    pdef_sd::Distribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:eigen] = Eigen

COMPONENT_CONSTRUCTORS[:eigen] = (p, params) -> Eigen(
    p.pca_sd,
    p.pdef_sd,
    get(params, :method, :noncentered)
)

MODEL_TO_STRUCTURE_MAP[:eigen] = :any

# --- Helper functions for Householder transformations ---

"""
    householder_to_eigenvector(v_mat, p, k)

Constructs a `p x k` orthonormal matrix `U` from a `p x k` matrix of Householder
reflector vectors `v_mat`. This is used to build the loadings matrix in the `Eigen`
component.

# Arguments
- `v_mat`: A `p x k` matrix where each column is a Householder reflector vector.
- `p`: The number of original variables (`n_vars`).
- `k`: The number of latent factors. This is a local dimension symbol in the maths
  (`p` is likewise `n_vars`), not a TethysBayes parameter name; the sampled quantity derived
  from it is `factors_flat_<key>`.

# Returns
- A `p x k` matrix `U` with orthonormal columns.
"""
function householder_to_eigenvector(v_mat::AbstractMatrix{T}, p::Int, k::Int) where T
    # Start with the first k columns of the identity matrix
    U = zeros(T, p, k)
    for i in 1:k
        U[i, i] = one(T)
    end

    # Apply Householder reflections in reverse order
    for i in k:-1:1
        v = v_mat[:, i]
        v_norm_sq = sum(abs2, v)
        
        if v_norm_sq > 1e-9 # Avoid division by zero
            # Apply H_i = I - 2 * v * v' / ||v||^2 to U
            # This is equivalent to U - 2 * v * (v' * U) / ||v||^2
            vTU = v' * U
            U = U - (2 / v_norm_sq) * (v * vTU)
        end
    end
    return U
end

"""
    eigenvector_to_householder(U)

Performs the inverse operation of `householder_to_eigenvector`. Given an orthonormal
matrix `U`, it finds the corresponding Householder reflector vectors `v` that can
generate it. This is useful for initializing the `v_raw` parameters from a
classical PCA solution.

# Arguments
- `U`: A `p x k` matrix with orthonormal columns.

# Returns
- A `p x k` matrix `v_mat` of Householder reflector vectors.
"""
function eigenvector_to_householder(U::AbstractMatrix{T}) where T
    p, k = size(U)
    v_mat = zeros(T, p, k)
    A = copy(U)
    
    for i in 1:k
        # Find the reflector for the i-th column of the remaining matrix
        x = A[i:p, i]
        e1 = zeros(T, length(x))
        e1[1] = 1.0
        
        # The sign choice avoids catastrophic cancellation.
        v_sign = sign(x[1]) == 0 ? one(T) : sign(x[1])
        v = v_sign * norm(x) * e1 + x
        
        v_norm = norm(v)
        if v_norm > 1e-9
            v = v / v_norm
        end
        
        # Store the reflector vector (padded with zeros at the top)
        v_mat[i:p, i] = v
        
        # Apply the reflection to the remaining submatrix to continue the decomposition
        if i < k
            sub_A = A[i:p, (i+1):k]
            A[i:p, (i+1):k] = sub_A - 2 * v * (v' * sub_A)
        end
    end
    return v_mat
end

function get_precomputes(m::Eigen, M::NamedTuple, mod_data::Dict)::NamedTuple
    params = mod_data[:params]
    variables = mod_data[:variables]
    data = M.data # Access data from M.data

    if isempty(variables)
        error("The `eigen()` module was called without any variables specified.")
    end

    vars_sym = Symbol.(variables)
    if !all(hasproperty(data, v) for v in vars_sym)
        missing_vars = filter(v -> !hasproperty(data, v), vars_sym)
        error("Eigen module variables not found in data: $(missing_vars)")
    end

    # Extract the data and center it (a standard assumption for PCA).
    # This is performed on the CPU.
    #
    # Promote to Float64 explicitly. `Matrix(data[!, vars_sym])` preserves the column
    # eltype, so for integer-valued columns (an ordinal code, a count, or just an index
    # variable) the result is a `Matrix{Int}`, and the centering below then raises
    # `InexactError: Int64(-5.5)` on the first non-integral difference. Centering is float
    # arithmetic by definition, so the conversion belongs here rather than being left to
    # fail at the subtraction.
    eigen_data_matrix_cpu = Matrix{Float64}(data[!, vars_sym])
    eigen_data_matrix_cpu .-= mean(eigen_data_matrix_cpu, dims=1)

    n_vars = length(vars_sym)
    n_factors = get(params, :n_factors, 1)
    if n_factors >= n_vars
        @warn "Number of factors ($n_factors) for eigen() module should be less " *
              "than the number of variables ($n_vars). Setting to $(n_vars - 1)."
        n_factors = n_vars - 1
    end

    # Pre-calculate indices for the lower-triangular part of the Householder matrix.
    # These are small index vectors and can remain on the CPU.
    ltri_mask = [r >= c for r in 1:n_vars, c in 1:n_factors]
    ltri_indices = findall(vec(ltri_mask))
    ltri_indices_len = length(ltri_indices)

    n_obs = size(eigen_data_matrix_cpu, 1)

    return (
        eigen_data=eigen_data_matrix_cpu,
        n_latent=n_obs,
        n_vars=n_vars,
        n_factors=n_factors,
        ltri_indices=ltri_indices,
        ltri_indices_len=ltri_indices_len
    )
end

function get_priors(
    m::Eigen, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, 
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    pca_sd_prior_str = _distribution_to_string(m.pca_sd)
    pdef_sd_prior_str = _distribution_to_string(m.pdef_sd)
    
    hyper = spec.hyper
    n_obs = hyper.n_latent
    n_factors = hyper.n_factors
    n_vars = hyper.n_vars
    ltri_indices_len = hyper.ltri_indices_len

    priors = String[]
    push!(priors, "$(p_names.reflection_vector_unconstrained) ~ MvNormal(zeros(T, $(ltri_indices_len)), I)")
    push!(priors, "$(p_names.pca_sd) ~ filldist($(pca_sd_prior_str), $(n_factors))")
    push!(priors, "$(p_names.pdef_sd) ~ filldist($(pdef_sd_prior_str), $(n_vars))")
    
    if m.method == :noncentered
        push!(priors, "$(p_names.factors_flat) ~ MvNormal(zeros(T, $(n_obs * n_factors)), I)")
    end

    return join(priors, "\n    ")
end

function get_updates(
    m::Eigen, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, 
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    
    hyper = spec.hyper
    n_obs = hyper.n_latent
    n_factors = hyper.n_factors
    n_vars = hyper.n_vars
    
    common_code = """
        T_v = eltype($(p_names.reflection_vector_unconstrained))
        v_mat = zeros(T_v, $(n_vars), $(n_factors))
        v_mat[spec_registry[:$(key)].hyper.ltri_indices] .= $(p_names.reflection_vector_unconstrained)
        
        U = householder_to_eigenvector(v_mat, spec_registry[:$(key)].hyper.n_vars,
          spec_registry[:$(key)].hyper.n_factors)
        L = U * Diagonal($(p_names.pca_sd))
        pdef_sd_vec = $(p_names.pdef_sd)
        
        Y_eigen_data = spec_registry[:$(key)].hyper.eigen_data
    """

    noncentered_code = """
        # --- Factor Model for Eigen Component (Non-Centered): $(key) ---
        let
            $(common_code)
            F = reshape($(p_names.factors_flat), $(n_obs), $(n_factors))
            Y_hat = F * L'
            
            for j in 1:$(n_vars)
                Turing.@addlogprob! sum(Distributions.logpdf.(Distributions.Normal.(view(Y_hat, :, j), pdef_sd_vec[j]), view(Y_eigen_data, :, j)))
            end
            
            $(eta_target) = $(eta_target) .+ view(F, :, 1)
        end
    """


    centered_code = """
        # --- Factor Model for Eigen Component (Centered): $(key) ---
        let
            $(common_code)
            # Define the latent factors matrix that will be sampled row-by-row
            $(p_names.latent_field) = Matrix{T}(undef, $(n_obs), $(n_factors))
            Cov_F_row = Symmetric(L * L')
            
            for i in 1:$(n_obs)
                $(p_names.latent_field)[i, :] ~ MvNormal(zeros(T, $(n_factors)), Cov_F_row)
            end
            
            Y_hat = $(p_names.latent_field) * L'
            
            for i in 1:$(n_obs)
                Turing.@addlogprob! logpdf(MvNormal(Y_hat[i, :], Psi), Y_eigen_data[i, :])
            end
            
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), :, 1)
        end
    """

    if m.method == :noncentered
        return noncentered_code
    elseif m.method == :centered
        return centered_code
    else
        error("Unsupported method '$(m.method)' for Eigen component.")
    end
end

"""
    get_effects(m::Eigen, chain, spec::NamedTuple, M::NamedTuple, PS)

Reconstructs the `Eigen` component's effect from posterior samples. This version
is CPU-only and uses modern chain accessors. The dominant (first) latent factor is
returned as the effect.
"""
function get_effects(
    m::Eigen, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions ---
    n_samples = if occursin("FlexiChain", string(typeof(chain)))
        size(chain, 1) * FlexiChains.nchains(chain)
    else
        size(chain, 1) * size(chain, 3)
    end
    outcomes_N = M.outcomes_N
    p_names = string.(keys(chain))
    
    hyper = spec.hyper
    n_obs_train = hyper.n_latent
    n_factors = hyper.n_factors
    
    # Determine total number of observations (training + prediction)
    n_obs_full = M.y_N + (isnothing(PS) ? 0 : size(PS.data,
        1)) # Total observations (training + prediction)

    if !isnothing(PS)
        @warn "Prediction for the Eigen component '$(spec.key)' is not supported. " *
              "Returning zero effect for the prediction set."
    end

    # The Eigen component's parameters are not per-outcome. This flag ensures
    # that `_find_parameter` searches for the base parameter names without an
    # outcome-specific suffix (e.g., `_1`), even in a multivariate model.
    params_are_per_outcome = false
    p_names_k = generate_full_variable_names(spec, "univariate", nothing)

    # --- Reconstruct the latent factor effect (on CPU) ---
    local factor_effect
    if m.method == :noncentered
        factors_flat_name = _find_parameter(
            p_names, string(p_names_k.factors_flat), nothing, params_are_per_outcome
        )
        if isempty(factors_flat_name)
            @warn "Parameter 'factors_flat' for Eigen component $(spec.key) not found. " *
                  "Returning zero-matrix."
            factor_effect = zeros(Float64, n_obs_full, n_samples) # Initialize with zeros
        else
            factor_samples_train = get_params_vector(
                chain, factors_flat_name, n_obs_train * n_factors
            ) # (n_samples, n_obs_train * n_factors)
            # Reshape the flat [n_samples, n_params] matrix into a 3D tensor
            # [n_obs, n_factors, n_samples]
            F_tensor = reshape(factor_samples_train', n_obs_train, n_factors, n_samples)
            
            # Initialize full effect matrix with zeros (handles prediction set)
            factor_effect = zeros(Float64, n_obs_full, n_samples)
            
            # The effect is the first factor, applied only to training observations.
            # Slicing the tensor is efficient.
            factor_effect[1:n_obs_train, :] = F_tensor[:, 1, :]
        end
    else # :centered
        latent_field_name = _find_parameter(
            p_names, string(p_names_k.latent_field), nothing, params_are_per_outcome
        )
        if isempty(latent_field_name)
            @warn "Parameter 'latent_field' for Eigen component $(spec.key) not found. " *
                  "Returning zero-matrix."
            factor_effect = zeros(Float64, n_obs_full, n_samples)
        else
            latent_samples_train = get_params_matrix(
                chain, latent_field_name, n_obs_train * n_factors
            )
            # Reshape the flat [n_samples, n_params] matrix into a 3D tensor
            # [n_obs, n_factors, n_samples]
            F_tensor = reshape(latent_samples_train', n_obs_train, n_factors, n_samples)

            # Initialize full effect matrix with zeros (handles prediction set)
            factor_effect = zeros(Float64, n_obs_full, n_samples)

            # The effect is the first factor, applied only to training observations.
            factor_effect[1:n_obs_train, :] = F_tensor[:, 1, :]
        end
    end

    # The same dominant factor is applied as a shared effect to all outcomes.
    structured_effects = [factor_effect for _ in 1:outcomes_N]
    
    return (structured=structured_effects, noisy=structured_effects)
end
