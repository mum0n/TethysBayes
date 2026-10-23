"""
    Mixed <: ComponentModel

An operator component that models random effects (intercepts and/or slopes) for a
specified grouping variable. The correlation structure of the effects is determined
by an inner `ComponentModel`.

# Version
v1.0.0

# Mathematical Summary
The `Mixed` component models effects that vary across the levels of a grouping
variable. It supports both simple (uncorrelated) and correlated random effects.

1.  **Simple Random Effects** (e.g., `random(1 | group)`):
    A single random effect \$\\phi\$ (e.g., an intercept) is modeled for each of the
    \$G\$ levels of the grouping variable. The structure of these effects is
    determined by the inner model. For an `IID` inner model, this is:
    \$\\phi_g \\sim \\mathcal{N}(0, \\sigma^2)\$ for \$g = 1, \\dots, G\$.

2.  **Correlated Random Effects** (e.g., `random(1 + x | group)`):
    A vector of \$K\$ random effects, \$\\boldsymbol{\\beta}_g = [\\beta_{0g}, \\beta_{1g},
      \\dots]^T\$,
    is modeled for each group level \$g\$. These effects are assumed to be drawn from
    a multivariate normal distribution with a shared covariance structure:
    \$\\boldsymbol{\\beta}_g \\sim \\mathcal{N}(\\mathbf{0}, \\Sigma)\$
    The covariance matrix \$\\Sigma\$ is decomposed into a set of standard deviations
    \$\\boldsymbol{\\sigma}\$ and a correlation matrix \$\\mathbf{R}\$:
    \$\\Sigma = \\text{diag}(\\boldsymbol{\\sigma}) \\mathbf{R}
      \\text{diag}(\\boldsymbol{\\sigma})\$
    A prior is placed on the Cholesky factor of \$\\mathbf{R}\$ using the `LKJCholesky`
    distribution. The structure of the effects across group levels (e.g., IID, spatial)
    is determined by the inner `ComponentModel`.

# Computational Methods (for Correlated Effects)
- `:spectral` (default): An efficient, AD-safe method using spectral decomposition of
  the group-level precision matrix.
- `:cholesky`: An AD-safe didactic alternative using dense Cholesky factorization.
- `:cholesky_sparse`: A non-AD-safe didactic method using sparse Cholesky
  factorization, suitable for gradient-free samplers.

# Inputs
- **Required**:
  - A grouping variable (e.g., `group_id`) passed to `random()`.
  - One or more terms for the random effects (e.g., `1` for intercept, `covariate` for slope).
- **Optional (in `random()` call)**:
  - `model`: An inner `ComponentModel` defining the structure across groups (e.g., `iid()`,
    `ar1()`). Default: `iid()`.
  - `method`: `Symbol`, computational method for correlated effects (`:spectral`,
    `:cholesky`, `:cholesky_sparse`). Default: `:spectral`.

# Outputs (Parameter Names)
- **Simple Effects**: Same as the inner model (e.g., `sigma_<key>`, `innovations_<key>`).
- **Correlated Effects**:
  - `L_corr_<key>`: The Cholesky factor of the correlation matrix for the effects.
  - `sigma_effects_<key>`: The standard deviations for each random effect term.
  - `innovations_<key>`: The raw standard normal innovations for the coefficients.
"""
struct Mixed <: ComponentModel
    grouping_covariate::Symbol
    lhs::Vector{String}
    model::ComponentModel
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:mixed] = Mixed
COMPONENT_CONSTRUCTORS[:mixed] = (p, params) -> begin
    grouping_covariate = get(params, :grouping_covariate, :group)
    raw_lhs = get(params, :lhs, ["1"])
    lhs_terms = String[]
    for item in (raw_lhs isa AbstractVector ? raw_lhs : [raw_lhs])
        for sub_term in Base.split(string(item), "+")
            trimmed = Base.strip(sub_term)
            if !isempty(trimmed)
                push!(lhs_terms, trimmed)
            end
        end
    end
    if isempty(lhs_terms)
        lhs_terms = ["1"]
    end
    inner_model_obj = get(
        params, :inner_model_obj, IID(p.sigma, :noncentered)
    )
    method = get(params, :method, :spectral)
    
    Mixed(grouping_covariate, lhs_terms, inner_model_obj, method)
end

MODEL_TO_STRUCTURE_MAP[:mixed] = :mixed

function get_precomputes(m::Mixed, M::NamedTuple, mod_data::Dict)::NamedTuple
    n_cat = get(mod_data[:params], :n_cat, 0)
    if n_cat == 0
        error("Number of categories for Mixed not determined in get_datastructures!.")
    end

    inner_mod_data = Dict(
        :key => Symbol("$(mod_data[:key])_inner"),
        :type => :mixed,
        :variables => [m.grouping_covariate],
        :params => Dict(:n_cat => n_cat)
    )

    inner_precomputes = get_precomputes(m.model, M, inner_mod_data)
    
    return (
        inner_precomputes = inner_precomputes,
        n_latent = n_cat * length(m.lhs)
    )
end

function get_priors(
    m::Mixed, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_terms = length(m.lhs)
    n_groups = spec.hyper.inner_precomputes.n_latent

    if n_terms == 1
        inner_spec = (
            key = spec.key,
            structure = :mixed,
            var = spec.var,
            component_obj = m.model,
            params = spec.params,
            hyper = spec.hyper.inner_precomputes
        )
        return get_priors(m.model, inner_spec, arch, outcome_idx, M)
    else
        return """
        # Priors for Correlated Mixed Effects: $(spec.key)
        $(p_names.correlation_cholesky) ~ LKJCholesky($(n_terms), 1.0)
        $(p_names.sigma_effects) ~ filldist(Exponential(1.0), $(n_terms))
        $(p_names.innovations) ~ MvNormal(zeros(T, $(n_groups * n_terms)), I)
        """
    end
end

function get_updates(
    m::Mixed, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "mixed_idx_$(m.grouping_covariate)"
    n_terms = length(m.lhs)
    n_groups = spec.hyper.inner_precomputes.n_latent

    inner_hyper_access = "spec_registry[:$(spec.key)].hyper.inner_precomputes"

    if n_terms == 1
        lhs_str = m.lhs[1]
        inner_model = m.model
        
        local latent_field_code
        if inner_model isa IID
            latent_field_code = "$(p_names.latent_field) = $(p_names.innovations) .* $(p_names.sigma)"
        elseif inner_model isa Union{ICAR, Besag, RW1, RW2, Leroux}
            if m.method == :spectral
                latent_field_code = """
                diag_D = $(p_names.sigma) ./ sqrt.($(inner_hyper_access).L .+ M.noise)
                # The old code branched on the inner model to decide HOW MANY leading
                # entries to zero: one for ICAR/Besag/RW1, two for RW2. That count is a
                # property of the eigen-spectrum, not of the model type, so a disconnected
                # graph (one null direction per component) was under-deflated and RW2's
                # two-dimensional null space was hard-coded. Derive it from `L` instead.
                _zero_null_modes!(diag_D, $(inner_hyper_access).L)
                $(p_names.latent_field) = $(inner_hyper_access).U * (diag_D .* $(p_names.innovations))
                """
            else # :cholesky or :cholesky_sparse
                latent_field_code = """
                F_groups = $(inner_hyper_access).cholesky_factor
                latent_field_unscaled = F_groups.L' \\ $(p_names.innovations)
                if $(inner_model isa Union{ICAR, Besag, RW1, RW2})
                    Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * $(n_groups)), sum(latent_field_unscaled))
                end
                $(p_names.latent_field) = latent_field_unscaled .* $(p_names.sigma)
                """
            end
        else
            latent_field_code = "$(p_names.latent_field) = $(p_names.innovations) .* $(p_names.sigma)"
        end

        local application_code
        if lhs_str == "1" || lhs_str == "intercept()"
            application_code = "$(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.$(index_var))"
        else
            application_code = """
            let cov_data = M.data[!, :$(Symbol(lhs_str))]
                $(eta_target) = $(eta_target) .+ cov_data .* view($(p_names.latent_field), M.$(index_var))
            end
            """
        end

        return """
        # --- Mixed Effect (Single): $(lhs_str) | $(m.grouping_covariate) ---
        let
            $(latent_field_code)
            $(application_code)
        end
        """
    else
        application_loop = ""
        for i in 1:n_terms
            term = m.lhs[i]
            if term == "1" || term == "intercept()"
                application_loop *= "$(eta_target) = $(eta_target) .+ view(effects_matrix[:, $(i)], M.$(index_var))\n"
            else
                application_loop *= "let cov_data_$(i) = M.data[!, :$(Symbol(term))]; $(eta_target) = $(eta_target) .+ cov_data_$(i) .* view(effects_matrix[:, $(i)], M.$(index_var)); end\n"
            end
        end

        common_correlated_code = """
            L_effects_t = ($(p_names.correlation_cholesky).L' * Diagonal($(p_names.sigma_effects)))
            innovations_matrix = reshape($(p_names.innovations), $(n_groups), $(n_terms))
        """

        spectral_code = if m.model isa IID
            """
            # --- Correlated Mixed Effects (IID): $(spec.key) ---
            let
                $(common_correlated_code)
                gamma_matrix = innovations_matrix
                effects_matrix = gamma_matrix * L_effects_t
                
                $(application_loop)
            end
            """
        else
            """
            # --- Correlated Mixed Effects (Spectral): $(spec.key) ---
            let
                $(common_correlated_code)
                inner_hyper = $(inner_hyper_access)
                diag_D = 1.0 ./ sqrt.(abs.(inner_hyper.L) .+ M.noise)
                # Fixed-count deflation replaced by a tolerance test on the spectrum. The
                # argument must be the same vector the division used, i.e. `abs.(L)`.
                _zero_null_modes!(diag_D, abs.(inner_hyper.L))
                
                gamma_matrix = inner_hyper.U * (diag_D .* innovations_matrix)
                effects_matrix = gamma_matrix * L_effects_t
                
                $(application_loop)
            end
            """
        end

        cholesky_code = """
            # --- Correlated Mixed Effects (Cholesky): $(spec.key) ---
            let
                $(common_correlated_code)
                F_groups = $(inner_hyper_access).cholesky_factor
                gamma_matrix = F_groups.L' \\ innovations_matrix
                effects_matrix = gamma_matrix * L_effects_t
                
                $(application_loop)
            end
        """

        cholesky_sparse_code = """
            # --- Correlated Mixed Effects (Sparse Cholesky, Not AD-Safe): $(spec.key) ---
            let
                $(common_correlated_code)
                Q_groups = $(inner_hyper_access).Q_template
                F_groups = cholesky(Symmetric(Q_groups + M.noise * I))
                gamma_matrix = F_groups.L' \\ innovations_matrix
                effects_matrix = gamma_matrix * L_effects_t
                
                $(application_loop)
            end
        """

        if m.method == :spectral
            return spectral_code
        elseif m.method == :cholesky
            return cholesky_code
        elseif m.method == :cholesky_sparse
            return cholesky_sparse_code
        else
            error("Unsupported method '$(m.method)' for correlated Mixed component.")
        end
    end
end

function _get_model_symbol(m_obj::ComponentModel)
    for (sym, typ) in COMPONENT_TYPE_REGISTRY
        if m_obj isa typ
            return sym
        end
    end
    return :unknown
end


function get_effects(
    m::Mixed, chain, spec::NamedTuple, M::NamedTuple,
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
    
    n_terms = length(m.lhs)
    n_groups_train = spec.hyper.inner_precomputes.n_latent
    
    # --- Grouping Level and Index Handling ---
    grouping_covariate = m.grouping_covariate
    train_levels = unique(M.data[!, grouping_covariate])
    all_levels = train_levels
    has_new_levels = false

    if !isnothing(PS) && hasproperty(PS.data, grouping_covariate)
        pred_levels = unique(PS.data[!, grouping_covariate])
        if !isempty(setdiff(pred_levels, train_levels))
            has_new_levels = true
            all_levels = unique(vcat(train_levels, pred_levels))
        end
    end
    n_all_groups = length(all_levels)
    level_map = Dict(level => i for (i, level) in enumerate(all_levels))
    
    # Create the full index vector on the CPU
    full_indices_cpu = if !isnothing(PS) && hasproperty(PS.data, grouping_covariate)
        [level_map[v] for v in vcat(M.data[!, grouping_covariate], PS.data[!, grouping_covariate])]
    else
        [level_map[v] for v in M.data[!, grouping_covariate]]
    end

    # --- Reconstruction Logic ---
    if n_terms == 1
        # --- Case 1: Simple (Uncorrelated) Random Effects ---
        effects_per_outcome = Vector{Matrix{Float64}}()
        for k in 1:outcomes_N
            p_names_k = generate_full_variable_names(spec, M.model_arch, k)
            sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k,
                is_multivariate_model)
            innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)

            if isempty(sigma_name) || isempty(innovations_name)
                @warn "Parameters for simple Mixed component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(effects_per_outcome, zeros(Float64, length(full_indices_cpu),
                    n_samples)) # Use length(full_indices_cpu) for N_total
                continue
            end

            # Extract samples (CPU)
            sigma_samples = get_params_vector(chain, sigma_name, 1) # (n_samples, 1)
            innovations_samples = get_params_matrix(chain, innovations_name, n_groups_train) # (n_samples, n_groups_train)
            
            # Perform computation
            # latent_samples_train: [n_groups_train, n_samples]
            latent_samples_train_cpu = innovations_samples' .* sigma_samples'
            
            full_effects_cpu = zeros(Float64, n_all_groups, n_samples)
            train_indices_map_cpu = [level_map[level] for level in train_levels]
            full_effects_cpu[train_indices_map_cpu, :] = latent_samples_train_cpu

            if has_new_levels
                new_level_indices = setdiff(1:n_all_groups, train_indices_map_cpu)
                new_effects_cpu = randn(Float64, length(new_level_indices),
                    n_samples) .* sigma_samples'
                full_effects_cpu[new_level_indices, :] = new_effects_cpu
            end
            
            # Index the effects for the full observation set
            indexed_effects_cpu = full_effects_cpu[full_indices_cpu, :]
            push!(effects_per_outcome, indexed_effects_cpu)
        end
        # Wrap the single effect in a Dict for consistency with the correlated case
        effects_dict = Dict{Symbol, Vector{Matrix{Float64}}}()
        term_key = (m.lhs[1] == "1"||
            m.lhs[1] == "intercept()") ? :intercept : Symbol("slope_$(m.lhs[1])")
        effects_dict[term_key] = effects_per_outcome # This will be Vector{Matrix{Float64}} of length outcomes_N
        return (type=:simple, effects=effects_dict, lhs=m.lhs[1], indices=full_indices_cpu)
    else
        # --- Case 2: Correlated Random Effects ---
        effects_by_term = Dict{Symbol, Vector{Matrix{Float64}}}()
        for term in m.lhs
            term_key = (term == "1"||
                term == "intercept()") ? :intercept : Symbol("slope_$(term)")
            effects_by_term[term_key] = [zeros(Float64, length(full_indices_cpu),
                n_samples) for _ in 1:outcomes_N] # Use length(full_indices_cpu) for N_total
        end

        for k in 1:outcomes_N
            p_names_k = generate_full_variable_names(spec, M.model_arch, k)
            correlation_cholesky_name = _find_parameter(p_names, string(p_names_k.correlation_cholesky), k,
                is_multivariate_model)
            sigma_effects_name = _find_parameter(p_names, string(p_names_k.sigma_effects), k,
                is_multivariate_model)
            innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)

            if isempty(correlation_cholesky_name) || isempty(sigma_effects_name) || isempty(innovations_name)
                @warn "Parameters for correlated Mixed component $(spec.key) (outcome $k) not found. Skipping."
                continue
            end

            # Extract samples (CPU)
            l_corr_samples = get_params_matrix(chain, correlation_cholesky_name,
                n_terms * n_terms) # (n_samples, n_terms * n_terms)
            sigma_effects_samples = get_params_matrix(chain, sigma_effects_name,
                n_terms) # (n_samples, n_terms)
            innovations_samples = get_params_matrix(chain, innovations_name,
                n_groups_train * n_terms) # (n_samples, n_groups_train * n_terms)
            
            inner_precomputes = spec.hyper.inner_precomputes

            # Initialize matrix to store all effects for this outcome across all groups and samples
            all_effects_for_outcome = zeros(Float64, n_all_groups, n_terms, n_samples)

            for s in 1:n_samples # Iterate over each posterior sample
                # Current sample's parameters (CPU)
                l_corr_s = reshape(l_corr_samples[s,:], n_terms, n_terms)
                sigma_effects_s = sigma_effects_samples[s,:]
                innov_matrix_s = reshape(innovations_samples[s,:], n_groups_train, n_terms)

                # Perform computations on CPU
                L_effects_t = (l_corr_s' * Diagonal(sigma_effects_s))

                local gamma_matrix_cpu
                if m.method == :spectral
                    diag_D_cpu = 1.0 ./ sqrt.(inner_precomputes.L .+ noise)
                    # Zero EVERY null direction (a disconnected graph has more than one).
                    _zero_null_modes!(diag_D_cpu, inner_precomputes.L)
                    gamma_matrix_cpu = inner_precomputes.U * (diag_D_cpu .* innov_matrix_s)
                else # :cholesky or :cholesky_sparse
                    F_groups_cpu = inner_precomputes.cholesky_factor
                    gamma_matrix_cpu = F_groups_cpu.L' \ innov_matrix_s
                end

                effects_matrix_train = gamma_matrix_cpu * L_effects_t

                # Populate effects for this sample
                train_indices_map_cpu = [level_map[level] for level in train_levels]
                all_effects_for_outcome[train_indices_map_cpu, :, s] = effects_matrix_train

                if has_new_levels
                    new_level_indices = setdiff(1:n_all_groups, train_indices_map_cpu)
                    n_new = length(new_level_indices)
                    new_innovs_cpu = randn(Float64, n_new,
                        n_terms) # Generate new innovations for prediction
                    new_gamma_cpu = new_innovs_cpu # Simplified assumption for new levels
                    new_effects_cpu = new_gamma_cpu * L_effects_t
                    all_effects_for_outcome[new_level_indices, :, s] = new_effects_cpu
                end
            end

            # Distribute effects to the dictionary by term
            for i in 1:n_terms
                term_key = (m.lhs[i] == "1"||
                    m.lhs[i] == "intercept()") ? :intercept : Symbol("slope_$(m.lhs[i])")
                effects_by_term[term_key][k] = all_effects_for_outcome[full_indices_cpu, i, :]
            end
        end
        return (type=:correlated, effects=effects_by_term, lhs=m.lhs, indices=full_indices_cpu)
    end
end 
