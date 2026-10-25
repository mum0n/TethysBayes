"""
    DAG <: ComponentModel

A component for a Directed Acyclic Graph (DAG) structure, also known as a
unilateral autoregressive model. This model is useful for capturing causal or
directional dependencies between spatial or other units, such as in river networks
or epidemiological models.

# Version
v1.0.0

# Mathematical Summary
The DAG model defines a recursive relationship for the latent field \$\\phi\$:
\$\\phi_i = \\rho \\sum_{j \\in pa(i)} W_{ij} \\phi_j + \\epsilon_i\$
where \$pa(i)\$ are the parents of node \$i\$, \$W\$ is an adjacency matrix representing
the directed graph, \$\\rho\$ is a dependence parameter, and \$\\epsilon_i\$ are
independent innovations, \$\\epsilon_i \\sim N(0, \\sigma^2)\$.

This can be written in matrix form as \$(I - \\rho W)\\phi = \\epsilon\$. The corresponding
precision matrix is \$Q = (I - \\rho W)^T (I - \\rho W) / \\sigma^2\$.

# Computational Methods
- `:forward_substitution` (Default, AD-safe): If `W` is strictly triangular (representing a
  valid topological ordering), the latent field can be sampled efficiently using
  forward substitution. This is the recommended, AD-safe method.
- `:precision` (Didactic, AD-safe): Explicitly constructs the dense precision matrix \$Q\$ and
  samples the field from the corresponding `MvNormal`. This is less efficient but
  conceptually clear and also AD-safe.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `s_idx`).
  - An adjacency matrix `W` passed as a keyword argument to `@tbm`.
- **Optional (in `random()` call)**:
  - `rho`: `UnivariateDistribution`, prior for the autoregressive parameter. Default:
    `Normal(0, 0.5)`.
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation of the innovations.
    Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:forward_substitution` or `:precision`).
    Default: `:forward_substitution`.

# Outputs (Parameter Names)
- `rho_<key>`: The autoregressive parameter.
- `sigma_<key>`: The standard deviation of the innovations.
- `innovations_<key>`: The standard normal innovations for the latent field.
- `latent_<key>`: The reconstructed latent DAG effect.

# Key References
- Cressie, N. (1993). *Statistics for Spatial Data*. Wiley.
- Ver Hoef, J. M., Peterson, E. E., & Theobald, D. M. (2006). *Spatial statistical models
  that use flow and stream distance*. Environmental and Ecological Statistics, 13(4),
  449-464.
"""
struct DAG <: ComponentModel
    rho::Distribution
    sigma::Union{Distribution, Real}
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:dag] = DAG
COMPONENT_CONSTRUCTORS[:dag] = (p, params) -> DAG(
    get(p, :rho, Normal(0, 0.5)),
    get(p, :sigma, Exponential(1.0)),
    get(params, :method, :forward_substitution)
)

MODEL_TO_STRUCTURE_MAP[:dag] = :spatial

"""
    get_precomputes(m::DAG, M::NamedTuple, mod_data::Dict)::NamedTuple

For the `DAG` component, this function stores the adjacency matrix `W` as the
`Q_template`. The model's forward substitution logic relies on this matrix
representing the causal graph structure. This is a CPU-only implementation.
"""
function get_precomputes(m::DAG, M::NamedTuple, mod_data::Dict)::NamedTuple
    n = M.s_N
    W = sparse(M.W) # Ensure W is sparse for efficiency

    return (Q_template=W, n_latent=n)
end

"""
    get_priors(m::DAG, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for `rho`, `sigma`, and the raw innovations.
"""
function get_priors(
    m::DAG, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    return """
    $(_prior_or_constant(p_names.rho, m.rho))
    $(_prior_or_constant(p_names.sigma, m.sigma))
    $(p_names.innovations) ~ MvNormal(zeros(T, $(spec.hyper.n_latent)), I)
    """
end

"""
    get_updates(m::DAG, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates Turing code to construct the `DAG` effect, dispatching on the chosen method.
"""
function get_updates(
    m::DAG, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    
    forward_sub_code = """
        # --- DAG Component (Forward Substitution): $(key) ---
        let
            W_dag = spec_registry[:$(key)].hyper.Q_template
            innovations_vec = $(p_names.innovations)
            rho_val = $(p_names.rho)
            sigma_val = $(p_names.sigma)
            n_latent = spec_registry[:$(key)].hyper.n_latent

            T_num = promote_type(typeof(rho_val), eltype(innovations_vec))
            $(p_names.latent_field) = zeros(T_num, n_latent)

            # Iterate through nodes in topological order (assumed by W_dag structure)
            for i in 1:n_latent
                parent_effect = zero(T_num) # Initialize with zero of correct type
                # Sum contributions from parents (non-zero elements in the row of W_dag)
                for j_ptr in nzrange(W_dag, i)
                    parent_idx = W_dag.rowval[j_ptr]
                    parent_effect += W_dag.nzval[j_ptr] * $(p_names.latent_field)[parent_idx]
                end
                # Recursive relationship: phi_i = rho * sum(W_ij * phi_j) + epsilon_i
                $(p_names.latent_field)[i] = rho_val * parent_effect + innovations_vec[i]
            end
            $(p_names.latent_field) = $(p_names.latent_field) .* sigma_val # Scale the entire field by sigma
            
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.s_idx)  # Apply to linear predictor
        end
    """

    precision_code = """
        # --- DAG Component (Precision Matrix): $(key) ---
        let
            W_dag = spec_registry[:$(key)].hyper.Q_template
            L_op = I - $(p_names.rho) * W_dag # Operator (I - rho * W)
            Q = L_op' * L_op # Precision matrix Q = (I - rho * W)' * (I - rho * W)
            
            # Use dense Cholesky for AD-safety
            F = cholesky(Symmetric(Matrix(Q) + M.noise * I))
            
            # Non-centered parameterization: latent_field = sigma * L_inv * innovations
            $(p_names.latent_field) = $(p_names.sigma) .* (F.U \\ $(p_names.innovations))
            
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.s_idx)  # Apply to linear predictor
        end
    """

    if m.method == :forward_substitution
        return forward_sub_code
    elseif m.method == :precision
        return precision_code
    else
        error("Unsupported method '$(m.method)' for DAG component. Supported methods are `:forward_substitution` and `:precision`.")
    end
end


"""
    get_effects(m::DAG, chain, spec::NamedTuple, M::NamedTuple, PS)

Reconstructs the `DAG` component's effect from posterior samples, dispatching
on the method used during sampling. This version is CPU-only and uses modern
chain accessors.
"""
function get_effects(
    m::DAG, chain, spec::NamedTuple, M::NamedTuple,
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
    
    noise = M.noise
    
    n_latent = spec.hyper.n_latent
    W_dag = spec.hyper.Q_template

    # --- Coordinate/Index Handling: Combine training and prediction sets ---
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
        
            # Only `sigma` is pinnable here: `rho` and `innovations` are always sampled, so
            # they must come from the chain. A pinned `sigma` is not a chain parameter, so
            # resolve it with the constant fallback rather than reading it by name.
            rho_name = _find_parameter(p_names, string(p_names_k.rho), k, is_multivariate_model)
            innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)
            sigma_samples = _resolve_hyper_samples(
                chain, p_names, p_names_k.sigma, m.sigma, k, is_multivariate_model,
                n_samples; as_matrix = true)

            if isempty(rho_name) || isnothing(sigma_samples) || isempty(innovations_name)
                @warn "Parameters for DAG component $(spec.key) (outcome $k) not " *
                      "resolved, and sigma is not a pinned constant. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end

            # Extract posterior samples (these are on the CPU)
            rho_samples = get_params_vector(chain, rho_name, 1) # (n_samples, 1)
            innovations_samples = get_params_matrix(chain, innovations_name, n_latent) # (n_samples, n_latent)
        
        # Initialize the output matrix for latent effects
        latent_field_matrix = zeros(Float64, n_latent, n_samples)

        # --- Sample-wise Reconstruction ---
        if m.method == :forward_substitution
            for i in 1:n_samples
                innov_i = innovations_samples[i, :]
                latent_field_i = zeros(Float64, n_latent)
                
                for j in 1:n_latent
                    parent_effect = 0.0
                    for j_ptr in nzrange(W_dag, j)
                        parent_idx = W_dag.rowval[j_ptr]
                        parent_effect += W_dag.nzval[j_ptr] * latent_field_i[parent_idx]
                    end
                    latent_field_i[j] = rho_samples[i, 1] * parent_effect + innov_i[j]
                end
                latent_field_matrix[:, i] = latent_field_i .* sigma_samples[i, 1]
            end
        else # :precision
            for i in 1:n_samples
                innov_i = innovations_samples[i, :]
                
                L_op = I - rho_samples[i, 1] * W_dag
                Q = L_op' * L_op
                
                F = cholesky(Symmetric(Matrix(Q) + noise * I))
                latent_field_i = sigma_samples[i, 1] .* (F.U \ innov_i)
                latent_field_matrix[:, i] = latent_field_i
            end
        end

        # Index the reconstructed latent effects to match the observation indices
        indexed_effects = latent_field_matrix[s_idx_full, :]
        push!(structured_effects, indexed_effects)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
