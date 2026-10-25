"""
    NetworkFlow <: ComponentModel

A component model for network flow processes, where dependencies are directional or
influenced by underlying landscape features. It parameterizes a Gaussian Markov
Random Field (GMRF) on a graph with data-driven edge weights, often representing
habitat conductivity or resistance.

# Version
v1.0.0

# Mathematical Summary
The component models a latent spatial field \$\\phi\$ as a GMRF,
\$\\phi \\sim \\mathcal{N}(0, (\\frac{1}{\\sigma^2} Q_{\\beta})^{-1})\$. The precision
matrix \$Q_{\\beta}\$ is a graph Laplacian that depends on a learned parameter \$\\beta\$.

The edge weight \$w_{ij}\$ between two connected spatial units \$i\$ and \$j\$ is defined
as a function of a habitat variable \$H\$ at those locations:
\$w_{ij} = \\exp\\left( \\beta \\cdot \\frac{H_i + H_j}{2} \\right)\$
A positive \$\\beta\$ implies that higher habitat values correspond to stronger
connectivity (lower movement cost), while a negative \$\\beta\$ implies higher values
correspond to stronger resistance.

The precision matrix is then constructed as the weighted graph Laplacian:
\$Q_{\\beta} = D_{\\beta} - W_{\\beta}\$
where \$W_{\\beta}\$ is the matrix of weights \$w_{ij}\$ and \$D_{\\beta}\$ is the diagonal
matrix of row sums of \$W_{\\beta}\$. By estimating \$\\beta\$, the model learns the
degree to which the habitat influences the spatial correlation structure.

# Computational Methods
- `:cholesky` (Default, AD-friendly): An AD-safe method using dense Cholesky factorization.
- `:cholesky_sparse` (Didactic, Not AD-friendly): A more memory-efficient method using
  sparse Cholesky factorization, suitable for gradient-free samplers.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `region`) passed to `random()`.
  - An adjacency matrix `W` passed as a keyword argument to `@tbm`.
  - `habitat`: A `Symbol` pointing to a column in the data, or a `Vector` of length `s_N`.
- **Optional (in `random()` call)**:
  - `beta`: `UnivariateDistribution`, prior for the habitat effect parameter. Default:
    `Normal(0, 1)`.
  - `sigma`: `UnivariateDistribution`, prior for the marginal standard deviation. Default:
    `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:cholesky` or `:cholesky_sparse`). Default:
    `:cholesky`.

# Outputs (Parameter Names)
- `beta_<key>`: The habitat effect parameter.
- `sigma_<key>`: The marginal standard deviation of the latent field.
- `innovations_<key>`: The raw standard normal innovations for the latent field.
- `latent_<key>`: The reconstructed latent spatial field.
"""
struct NetworkFlow <: ComponentModel
    # `beta` accepts a `Real` as well as a Distribution, so the habitat effect can be pinned
    # to a constant. `_prior_or_constant` turns a Real into `name = value` rather than a
    # prior, and the struct field must accept it. Matches `sigma` below and the other
    # components; see the `FITC` note on the same class of fix.
    beta::Union{UnivariateDistribution, Real}
    sigma::Union{UnivariateDistribution, Real}
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:networkflow] = NetworkFlow

# `p.beta` with no fallback raised `FieldError: type NamedTuple has no field beta` for any
# model that did not spell `beta=` out explicitly -- but `beta` is documented as OPTIONAL
# with default `Normal(0, 1)` (see the docstring's "Optional (in `random()` call)" section),
# so omitting it is the documented usage, not a user error. Every other component reads its
# optional hyperparameter with `get(p, :name, default)`; this one did not.
COMPONENT_CONSTRUCTORS[:networkflow] = (p, params) -> NetworkFlow(
    get(p, :beta, Normal(0, 1)),
    get(p, :sigma, Exponential(1.0)),
    get(params, :method, :cholesky)
)

MODEL_TO_STRUCTURE_MAP[:networkflow] = :spatial

function get_precomputes(m::NetworkFlow, M::NamedTuple, mod_data::Dict)::NamedTuple
    # Ensure spatial context is established by process_random_module!
    s_N = M.s_N
    W = M.W
    
    params = mod_data[:params]
    data = M.data

    if !haskey(params, :habitat)
        error(
            "The `networkflow` model requires a `habitat` parameter specifying the " *
            "conductivity/resistivity data."
        )
    end

    habitat_val = params[:habitat]
    local habitat_data::Vector{Float64}

    if habitat_val isa Symbol
        if !hasproperty(data, habitat_val)
            error("Habitat variable ':$habitat_val' not found in the data frame.")
        end
        # Aggregate habitat data to spatial units if it's per-observation
        habitat_per_obs = data[!, habitat_val]
        habitat_aggregated = zeros(Float64, s_N)
        counts = zeros(Int, s_N)
        for i in 1:M.y_N
            s_i = M.s_idx[i]
            habitat_aggregated[s_i] += habitat_per_obs[i]
            counts[s_i] += 1
        end
        habitat_data = habitat_aggregated ./ max.(1, counts)

    elseif habitat_val isa AbstractVector
        if length(habitat_val) != s_N
            error(
                "Provided `habitat` vector length ($(length(habitat_val))) does not " *
                "match the number of spatial units s_N ($(s_N))."
            )
        end
        habitat_data = convert(Vector{Float64}, habitat_val)
    else
        error(
            "The `habitat` parameter must be a Symbol (column name) or a Vector of " *
            "length s_N."
        )
    end

    I, J, _ = findnz(W)
    return (W_I=I, W_J=J, n_latent=s_N, s_N=s_N, habitat_data=habitat_data)
end

function get_priors(
    m::NetworkFlow, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    priors = String[]

    push!(priors, "$(_prior_or_constant(p_names.beta, m.beta))")
    push!(priors, "$(_prior_or_constant(p_names.sigma, m.sigma))")
    push!(priors, "$(p_names.innovations) ~ MvNormal(zeros(T, $(spec.hyper.n_latent)), I)")

    return join(priors, "\n    ")
end

function get_updates(
    m::NetworkFlow, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key

    common_code = """
        # 1. Construct the weighted adjacency matrix based on habitat and beta.
        hyper = spec_registry[:$(key)].hyper
        W_I = hyper.W_I
        W_J = hyper.W_J
        habitat = hyper.habitat_data
        s_N = hyper.s_N
        
        V_beta = exp.($(p_names.beta) .* (habitat[W_I] .+ habitat[W_J]) ./ 2.0)
        W_beta = sparse(W_I, W_J, V_beta, s_N, s_N)
        
        # 2. Construct the weighted graph Laplacian (precision matrix).
        D_beta = Diagonal(vec(sum(W_beta, dims=2)))
        Q_beta = D_beta - W_beta
    """

    cholesky_code = """
        # --- NetworkFlow Component (Cholesky, AD-Safe): $(key) ---
        let
            $(common_code)
            
            F = cholesky(Symmetric(Matrix(Q_beta) + M.noise * I))
            $(p_names.latent_field) = $(p_names.sigma) .* (F.L' \\ $(p_names.innovations))
            
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.s_idx)
        end
    """

    cholesky_sparse_code = """
        # --- NetworkFlow Component (Sparse Cholesky, Not AD-Safe): $(key) ---
        let
            $(common_code)
            
            F = cholesky(Symmetric(Q_beta + M.noise * I))
            $(p_names.latent_field) = $(p_names.sigma) .* (F.L' \\ $(p_names.innovations))
            
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.s_idx)
        end
    """

    if m.method == :cholesky
        return cholesky_code
    elseif m.method == :cholesky_sparse
        return cholesky_sparse_code
    else
        error("Unsupported method '$(m.method)' for NetworkFlow component.")
    end
end

"""
    get_effects(m::NetworkFlow, chain, spec::NamedTuple, M::NamedTuple, PS)

Reconstructs the `NetworkFlow` component's effect from posterior samples. This
version is CPU-only and uses modern chain accessors.
"""
function get_effects(
    m::NetworkFlow, chain, spec::NamedTuple, M::NamedTuple,
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
    
    key = spec.key
    hyper = spec.hyper
    W_I = hyper.W_I
    W_J = hyper.W_J
    habitat_data = hyper.habitat_data
    s_N = hyper.s_N
    noise = M.noise

    # --- Index Handling: Combine training and prediction sets ---
    s_idx_train = M.s_idx # Spatial indices for training data
    s_idx_full = if !isnothing(PS) && hasproperty(PS.data, :s_idx) # If prediction set is provided
        vcat(s_idx_train, PS.data.s_idx) # Combine training and prediction indices
    else
        s_idx_train # Otherwise, use only training indices
    end
    N_total = length(s_idx_full) # Total number of observations (training + prediction)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k_outcome in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k_outcome)
        innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k_outcome,
            is_multivariate_model)

        # Both `beta` and `sigma` are pinnable here, and a pinned value is not a chain
        # parameter, so resolve them with the constant fallback rather than by name. Without
        # this, pinning either one reconstructs a ZERO field with only a warning.
        beta_samples = _resolve_hyper_samples(
            chain, p_names, p_names_k.beta, m.beta, k_outcome, is_multivariate_model, n_samples)
        sigma_samples = _resolve_hyper_samples(
            chain, p_names, p_names_k.sigma, m.sigma, k_outcome, is_multivariate_model, n_samples)

        if isnothing(beta_samples) || isnothing(sigma_samples) || isempty(innovations_name)
            @warn "Parameters for NetworkFlow component $(spec.key) (outcome $k_outcome) " *
                  "not resolved, and beta / sigma are not both pinned constants. " *
                  "Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        # Extract posterior samples (these are on the CPU)
        innovations_samples = get_params_matrix(chain, innovations_name, s_N)
        
        # Initialize the output matrix for the full latent field on the CPU
        reconstructed_effects_k = zeros(Float64, s_N, n_samples)

        # --- Sample-wise Reconstruction on the CPU ---
        for i in 1:n_samples
            # Construct precision matrix on CPU
            V_beta_i = exp.(beta_samples[i] .* (habitat_data[W_I] .+ habitat_data[W_J]) ./ 2.0)
            W_beta_i = sparse(W_I, W_J, V_beta_i, s_N, s_N)
            D_beta_i = Diagonal(vec(sum(W_beta_i, dims=2)))
            Q_beta_i = D_beta_i - W_beta_i
            
            # Perform factorization on CPU
            F_i = cholesky(Symmetric(Matrix(Q_beta_i) + noise * I))
            
            innov_i = innovations_samples[i, :]
            
            reconstructed_effects_k[:, i] = sigma_samples[i] .* (F_i.L' \ innov_i)
        end
        
        # Index the reconstructed effects for the full observation set
        indexed_effects = reconstructed_effects_k[s_idx_full, :]
        push!(structured_effects, indexed_effects)
    end

    return (structured=structured_effects, noisy=structured_effects)
end 
