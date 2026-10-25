"""
    Composed <: ComponentModel

An orchestrator component that combines multiple child components using a specified
operator. This component is the backbone for creating complex interactions, such as
spatiotemporal models, spatially varying coefficients, and non-stationary variance
models.

# Version
v1.0.0

# Mathematical Summary & Operators

The `Composed` component implements different model structures based on the `operator`
field:

1.  **`:pipe` (`|>`): Spatially Varying Curves & State-Space Models**
    -   **Use Case**: `random(t, model=pspline) |> random(s, model=icar)`
    -   **Math**: Models a set of basis coefficients \$\\beta(s)\$ as a spatial field.
        The final effect is \$f(s,t) = \\sum_j B_j(t) \\beta_j(s)\$, where \$B_j(t)\$ are
        temporal basis functions and \$\\beta_j(s)\$ are spatially correlated
        coefficients.

2.  **`:kronecker_product` (`⊗`): Spatiotemporal Interactions**
    -   **Use Case**: `random(s, model=icar) ⊗ random(t, model=ar1)`
    -   **Math**: Implements Knorr-Held interaction models. The joint precision
        matrix is the Kronecker product of the marginal precision matrices:
        \$Q_{st} = Q_s \\otimes Q_t\$. This creates a fully structured spatiotemporal
        random effect.

3.  **`:composition` (`∘`): Non-Stationary Variance & Functional Composition**
    -   **Use Case**: `random(s, model=icar) ∘ random(x, model=pspline)`
    -   **Math**: Models non-stationary processes where one component modulates the
        parameters of another. A common use is for non-stationary variance, where a
        smoother on a covariate `x` defines a spatially varying standard deviation
        \$\\sigma(x)\$ for a base spatial field \$\\phi(s)\$. The final effect is
        \$\\phi(s) \\cdot \\sigma(x)\$.

# Key References
- Knorr-Held, L. (2000). Bayesian modelling of inseparable space-time variation
  in disease risk. *Statistical Methods in Medical Research*. (For Kronecker
  product interactions).
- Gelfand, A. E., et al. (2003). Spatial modeling with spatially varying
  coefficient processes. *Journal of the American Statistical Association*. (For
  concepts related to spatially varying curves).

# Inputs
- **Required**:
  - Two or more `random()` modules combined with an operator (`|>`, `⊗`, `∘`).
- **Optional**:
  - `method`: A `Symbol` specifying the computational method.

# Outputs (Parameter Names)
- **`:pipe`**: `innovations_<key>`, `sigma_<state_key>`, `rho_<state_key>`
- **`:kronecker_product`**: `sigma_<key>`, `innovations_<key>`
- **`:composition`**: Parameters from the child components.
"""
struct Composed <: ComponentModel
    components::Vector{ComponentModel}
    operator::Symbol
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:composed] = Composed

COMPONENT_CONSTRUCTORS[:composed] = (p, params) -> begin
    components = get(params, :components, error("Composed requires child components."))
    operator = get(params, :operator, error("Composed requires an operator."))
    
    default_method = if operator == :kronecker_product
        :spectral
    elseif operator == :pipe
        :cholesky
    else
        :none
    end
    
    method = get(params, :method, default_method)
    Composed(components, operator, method)
end

MODEL_TO_STRUCTURE_MAP[:composed] = :any

function get_precomputes(m::Composed, M::NamedTuple, mod_data::Dict)::NamedTuple
    child_nodes = get(mod_data[:params], :components, [])
    child_precomputes_list = []
    child_specs_list = []

    for (i, child_node) in enumerate(child_nodes)
        child_component_obj = m.components[i]
        child_mod_data = Dict(
            :key => Symbol("$(mod_data[:key])_child_$(i)"),
            :type => child_node.module_type,
            :variables => get(child_node.args, :positional_args, []),
            :params => child_node.args
        )
        precomputes = get_precomputes(child_component_obj, M, child_mod_data)
        push!(child_precomputes_list, precomputes)

        child_spec = (
            key = child_mod_data[:key],
            structure = get_component_structure(child_node.module_type),
            var = join(string.(child_mod_data[:variables]), "_"),
            component_obj = child_component_obj,
            params = child_mod_data[:params],
            hyper = precomputes
        )
        push!(child_specs_list, child_spec)
    end

    return (
        operator = m.operator,
        child_precomputes = child_precomputes_list,
        child_specs = child_specs_list
    )
end

function get_priors(
    m::Composed, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    child_specs = spec.hyper.child_specs
    
    if m.operator == :pipe # Spatially varying curve
        dynamic_spec = child_specs[1]
        state_spec = child_specs[2]
        
        state_priors = get_priors(
            state_spec.component_obj, state_spec, arch, outcome_idx, M
        )
        
        n_spatial = state_spec.hyper.n_latent
        n_basis = dynamic_spec.hyper.n_latent
        coeffs_prior = "$(p_names.innovations) ~ MvNormal(zeros(T, " *
                       "$(n_spatial * n_basis)), I)"
        
        return """
        # Priors for Spatially Varying Curve: $(spec.key)
        $(state_priors)
        $(coeffs_prior)
        """
    elseif m.operator == :kronecker_product # Spatiotemporal interaction
        st_sigma_prior = get(spec.params, :sigma, Exponential(1.0))
        s_N = child_specs[1].hyper.n_latent
        t_N = child_specs[2].hyper.n_latent
        
        return """
        # Priors for Spatiotemporal Interaction: $(spec.key)
        $(_prior_or_constant(p_names.sigma, st_sigma_prior))
        $(p_names.innovations) ~ MvNormal(zeros(T, $(s_N * t_N)), I)
        """
    elseif m.operator == :composition # Non-stationary variance
        modifier_priors = get_priors(
            child_specs[1].component_obj, child_specs[1], arch, outcome_idx, M
        )
        base_priors = get_priors(
            child_specs[2].component_obj, child_specs[2], arch, outcome_idx, M
        )

        return """
        # Priors for Composition (NonStationaryVariance): $(spec.key)
        $(modifier_priors)
        $(base_priors)
        """
    else
        all_priors = [
            get_priors(cs.component_obj, cs, arch, outcome_idx, M)
            for cs in child_specs
        ]
        return join(filter(!isempty, all_priors), "\n    ")
    end
end

function get_updates(
    m::Composed, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    child_specs = spec.hyper.child_specs

    if m.operator == :pipe
        dynamic_spec = child_specs[1]
        state_spec = child_specs[2]
        
        state_p_names = generate_full_variable_names(state_spec, arch, outcome_idx)
        state_model_type = Symbol(lowercase(string(typeof(state_spec.component_obj))))
        rho_value = hasproperty(state_spec.component_obj, :rho) ? 
                    string(state_p_names.rho) : "nothing"
        
        basis_matrix = "spec_registry[:$(dynamic_spec.key)].hyper.basis_matrix"
        
        n_spatial = state_spec.hyper.n_latent
        n_basis = dynamic_spec.hyper.n_latent

        cholesky_base_code = """
            Q_spatial_template = spec_registry[:$(state_spec.key)].hyper.Q_template
            Q_spatial = recompose_precision(
                :$(state_model_type), Q_spatial_template, 1.0; extra_param=$(rho_value)
            )
        """

        cholesky_dense_code = """
            # --- Spatially Varying Curve (Cholesky): $(spec.key) ---
            let
                $(cholesky_base_code)
                F_spatial = cholesky(Symmetric(Matrix(Q_spatial) + M.noise * I))
                coeffs_unscaled_matrix = reshape($(p_names.innovations), $(n_spatial), $(n_basis))
                spatial_coeffs = F_spatial.L' \\ coeffs_unscaled_matrix
                $(p_names.latent_field) = sum($(basis_matrix) .* spatial_coeffs[M.s_idx, :], dims=2)
                $(eta_target) = $(eta_target) .+ $(p_names.latent_field)
            end
        """

        cholesky_sparse_code = """
            # --- Spatially Varying Curve (Sparse Cholesky, Not AD-Safe): $(spec.key) ---
            let
                $(cholesky_base_code)
                F_spatial = cholesky(Symmetric(Q_spatial + M.noise * I))
                coeffs_unscaled_matrix = reshape($(p_names.innovations), $(n_spatial), $(n_basis))
                spatial_coeffs = F_spatial.L' \\ coeffs_unscaled_matrix
                $(p_names.latent_field) = sum($(basis_matrix) .* spatial_coeffs[M.s_idx, :], dims=2)
                $(eta_target) = $(eta_target) .+ $(p_names.latent_field)
            end
        """
        
        if m.method == :cholesky
            return cholesky_dense_code
        elseif m.method == :cholesky_sparse
            return cholesky_sparse_code
        else; error("Unsupported method '$(m.method)' for pipe operator."); end

    elseif m.operator == :kronecker_product
        s_spec = child_specs[1]
        t_spec = child_specs[2]
        
        s_model_type = Symbol(lowercase(string(typeof(s_spec.component_obj))))
        t_model_type = Symbol(lowercase(string(typeof(t_spec.component_obj))))
        
        s_p_names = generate_full_variable_names(s_spec, arch, outcome_idx)
        t_p_names = generate_full_variable_names(t_spec, arch, outcome_idx)

        s_rho_val = hasproperty(s_spec.component_obj, :rho) ? 
                    string(s_p_names.rho) : "nothing"
        t_rho_val = hasproperty(t_spec.component_obj, :rho) ? 
                    string(t_p_names.rho) : "nothing"

        diag_Ls_expr = "s_hyper.L"
        diag_Lt_expr = "t_hyper.L"

        spectral_code = """
            # --- Spatiotemporal Interaction (Spectral): $(spec.key) ---
            let
                s_hyper = spec_registry[:$(s_spec.key)].hyper
                t_hyper = spec_registry[:$(t_spec.key)].hyper
                
                n_s = s_hyper.n_latent
                n_t = t_hyper.n_latent
                
                diag_Ls = $(diag_Ls_expr)
                diag_Lt = $(diag_Lt_expr)
                
                diag_D_s = $(p_names.sigma) ./ sqrt.(abs.(diag_Ls) .+ M.noise)
                diag_D_t = 1.0 ./ sqrt.(abs.(diag_Lt) .+ M.noise)
                
                Z_matrix = reshape($(p_names.innovations), n_s, n_t)
                transformed = (diag_D_s .* Z_matrix) .* diag_D_t'
                $(p_names.latent_field) = s_hyper.U * transformed * t_hyper.U'
                
                st_idx = (M.t_idx .- 1) .* n_s .+ M.s_idx
                $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), st_idx)
            end
        """

        cholesky_base_code = """
            Q_s = recompose_precision(:$(s_model_type), 
                spec_registry[:$(s_spec.key)].hyper.Q_template, 1.0; extra_param=$(s_rho_val))
            Q_t = recompose_precision(:$(t_model_type), 
                spec_registry[:$(t_spec.key)].hyper.Q_template, 1.0; extra_param=$(t_rho_val))
        """

        cholesky_dense_code = """
            # --- Spatiotemporal Interaction (Cholesky): $(spec.key) ---
            let
                $(cholesky_base_code)
                s_hyper = spec_registry[:$(s_spec.key)].hyper
                t_hyper = spec_registry[:$(t_spec.key)].hyper
                n_s = s_hyper.n_latent
                n_t = t_hyper.n_latent
                C_s = cholesky(Symmetric(Matrix(Q_s) + M.noise * I))
                C_t = cholesky(Symmetric(Matrix(Q_t) + M.noise * I))
                Z_matrix = reshape($(p_names.innovations), n_s, n_t)
                tmp_spatial = C_s.L' \\ Z_matrix
                st_field_unscaled = transpose(C_t.L' \\ transpose(tmp_spatial))
                Turing.@addlogprob! logpdf(Normal(0, 0.001 * (n_s * n_t)),
                  sum(st_field_unscaled))
                $(p_names.latent_field) = st_field_unscaled .* $(p_names.sigma)
                st_idx = (M.t_idx .- 1) .* n_s .+ M.s_idx
                $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), st_idx)
            end
        """

        if m.method == :spectral
            return spectral_code
        elseif m.method == :cholesky
            return cholesky_dense_code
        else; error("Unsupported method '$(m.method)' for Kronecker product operator."); end

    elseif m.operator == :composition # Non-stationary variance
        modifier_spec = child_specs[1]
        base_spec = child_specs[2]

        modifier_updates = get_updates(
            modifier_spec.component_obj, modifier_spec, arch, outcome_idx, M
        )
        modifier_latent_field_var = generate_full_variable_names(
            modifier_spec, arch, outcome_idx
        ).latent_field
        modifier_code = replace(modifier_updates,
            Regex("$(eta_target) (\\.\\+=|=|\\.\\=) .*") => "")

        base_updates = get_updates(
            base_spec.component_obj, base_spec, arch, outcome_idx, M
        )
        base_latent_field_var = generate_full_variable_names(
            base_spec, arch, outcome_idx
        ).latent_field
        base_code = replace(base_updates, Regex("$(eta_target) (\\.\\+=|=|\\.\\=) .*") => "")

        base_structure = base_spec.structure
        indexed_base_effect = if base_structure == :spatial
            "view($(base_latent_field_var), M.s_idx)"
        elseif base_structure == :temporal
            "view($(base_latent_field_var), M.t_idx)"
        elseif base_structure == :seasonal
            "view($(base_latent_field_var), M.u_idx)"
        else # :smooth or :any
            "$(base_latent_field_var)"
        end

        return """
        # --- Composition (NonStationaryVariance) Update: $(spec.key) ---
        let
            # Compute the modulating field (e.g., log_sigma(x))
            $(modifier_code)
            
            # Compute the base field (e.g., spatial effect z(s))
            $(base_code)

            # Modulate the base field and add to eta
            $(p_names.latent_field) = $(indexed_base_effect) .* exp.($(modifier_latent_field_var))
            $(eta_target) = $(eta_target) .+ $(p_names.latent_field)
        end
        """
    end
    
    return "# Operator '$(m.operator)' not implemented for get_updates in Composed"
end

function get_effects(
    m::Composed, chain, spec::NamedTuple, M::NamedTuple,
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
    child_specs = spec.hyper.child_specs

    # --- Operator-Specific Reconstruction ---
    if m.operator == :pipe # Spatially varying curve
        structured_effects = Vector{Matrix{Float64}}()
        dynamic_spec = child_specs[1]
        state_spec = child_specs[2] # Spatial component spec
        
        # Handle basis matrix for training and prediction sets (CPU operations)
        B_dynamic_train = dynamic_spec.hyper.basis_matrix
        B_dynamic_full = if !isnothing(PS) # If prediction set is provided
            coord_vars = get(dynamic_spec.params, :positional_args, [])
            if !isempty(coord_vars) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
                coords_pred = Matrix{Float64}(PS.data[!,
                    Symbol.(coord_vars)]) # Extract prediction coordinates
                dynamic_comp_obj = dynamic_spec.component_obj
                B_pred, _ = TethysBayes_bspline_basis(
                    coords_pred[:, 1], dynamic_comp_obj.nbins, dynamic_comp_obj.degree
                )
                vcat(B_dynamic_train, B_pred)
            else
                B_dynamic_train
            end
        else
            B_dynamic_train
        end
        N_total = size(B_dynamic_full, 1)
        
        n_spatial = state_spec.hyper.n_latent
        n_basis = dynamic_spec.hyper.n_latent
        s_idx_full = if !isnothing(PS) && hasproperty(PS.data, :s_idx) # Combine spatial indices
            vcat(M.s_idx, PS.data.s_idx) 
        else
            M.s_idx
        end

        for k in 1:outcomes_N
            p_names_k = generate_full_variable_names(spec, M.model_arch, k)
            innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)
            innovations_samples = get_params_matrix(chain, innovations_name, n_spatial * n_basis) # (n_samples, n_spatial * n_basis)
            state_p_names = generate_full_variable_names(state_spec, M.model_arch, k)
            state_model_type = Symbol(lowercase(string(typeof(state_spec.component_obj))))
            Q_spatial_template = state_spec.hyper.Q_template
            
            effect_k_matrix = zeros(Float64, N_total, n_samples) # Initialize output matrix

            for i in 1:n_samples # Iterate over each posterior sample
                rho_val = hasproperty(state_spec.component_obj, :rho) ?
                          get_params_vector(chain, string(state_p_names.rho), 1)[i,
                              1] : # Extract rho for current sample
                          nothing
                Q_spatial = recompose_precision(state_model_type, Q_spatial_template, 1.0;
                    extra_param=rho_val)
                F_spatial = cholesky(Symmetric(Matrix(Q_spatial) + M.noise * I))
                
                coeffs_unscaled_matrix = reshape(innovations_samples[i, :], n_spatial, n_basis)
                spatial_coeffs = F_spatial.L' \ coeffs_unscaled_matrix
                
                effect_k_matrix[:, i] = sum(B_dynamic_full .* spatial_coeffs[s_idx_full, :],
                    dims=2)
            end
            push!(structured_effects, effect_k_matrix)
        end
        return (structured=structured_effects, noisy=structured_effects)

    elseif m.operator == :kronecker_product
        structured_effects = Vector{Matrix{Float64}}()
        s_spec = child_specs[1]
        t_spec = child_specs[2] # Temporal component spec
        s_N = s_spec.hyper.n_latent # Number of spatial units
        t_N = t_spec.hyper.n_latent # Number of temporal units
        s_model_type = Symbol(lowercase(string(typeof(s_spec.component_obj))))
        t_model_type = Symbol(lowercase(string(typeof(t_spec.component_obj))))
        Q_s_template = s_spec.hyper.Q_template
        Q_t_template = t_spec.hyper.Q_template
        
        s_var = Symbol(s_spec.var)
        t_var = Symbol(t_spec.var)

        s_train = hasproperty(M, :data) && hasproperty(M.data, s_var) ? M.data[!, s_var] : M.s_idx
        t_train = hasproperty(M, :data) && hasproperty(M.data, t_var) ? M.data[!, t_var] : M.t_idx

        s_idx_full = if !isnothing(PS)
            s_pred = if hasproperty(PS.data, s_var)
                PS.data[!, s_var]
            elseif hasproperty(PS.data, :s_idx)
                PS.data.s_idx
            else
                fill(1, nrow(PS.data))
            end
            vcat(s_train, s_pred)
        else
            s_train
        end

        t_idx_full = if !isnothing(PS)
            t_pred = if hasproperty(PS.data, t_var)
                PS.data[!, t_var]
            elseif hasproperty(PS.data, :t_idx)
                PS.data.t_idx
            else
                fill(1, nrow(PS.data))
            end
            vcat(t_train, t_pred)
        else
            t_train
        end

        s_idx_int = clamp.(Int.(round.(s_idx_full)), 1, s_N)
        t_idx_int = clamp.(Int.(round.(t_idx_full)), 1, t_N)
        st_idx_full = (t_idx_int .- 1) .* s_N .+ s_idx_int
        N_total = length(st_idx_full)

        for k in 1:outcomes_N
            v = generate_full_variable_names(spec, M.model_arch, k)
            s_v = generate_full_variable_names(s_spec, M.model_arch, k)
            t_v = generate_full_variable_names(t_spec, M.model_arch, k)

            sigma_name = _find_parameter(p_names, string(v.sigma), k, is_multivariate_model)
            innovations_name = _find_parameter(p_names, string(v.innovations), k, is_multivariate_model)
            
            s_rho_name = hasproperty(s_spec.component_obj, :rho) ? _find_parameter(p_names,
                string(s_v.rho), k, is_multivariate_model) : ""
            t_rho_name = hasproperty(t_spec.component_obj, :rho) ? _find_parameter(p_names,
                string(t_v.rho), k, is_multivariate_model) : ""

            if isempty(sigma_name) || isempty(innovations_name)
                @warn "Parameters for Kronecker product component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end

            # Extract posterior samples (CPU)
            sigma_samples = get_params_vector(chain, sigma_name, 1) # (n_samples, 1)
            innovations_samples = get_params_matrix(chain, innovations_name, s_N * t_N) # (n_samples, s_N * t_N)
            
            s_rho_samples = !isempty(s_rho_name) ? get_params_vector(chain, s_rho_name,
                1) : nothing # (n_samples, 1)
            t_rho_samples = !isempty(t_rho_name) ? get_params_vector(chain, t_rho_name,
                1) : nothing # (n_samples, 1)

            # Initialize the output matrix for the full effect
            effect_k = zeros(Float64, N_total, n_samples)

            has_spectral = hasproperty(s_spec.hyper, :U) && hasproperty(s_spec.hyper, :L) &&
                           hasproperty(t_spec.hyper, :U) && hasproperty(t_spec.hyper, :L)

            if has_spectral
                s_U = s_spec.hyper.U
                s_L = s_spec.hyper.L
                t_U = t_spec.hyper.U
                t_L = t_spec.hyper.L
                
                diag_D_s_base = 1.0 ./ sqrt.(abs.(s_L) .+ noise)
                diag_D_t = 1.0 ./ sqrt.(abs.(t_L) .+ noise)

                for i in 1:n_samples
                    sigma_i = sigma_samples[i, 1]
                    innovations_i = innovations_samples[i, :]
                    
                    diag_D_s = sigma_i .* diag_D_s_base
                    Z_matrix = reshape(innovations_i, s_N, t_N)
                    transformed = (diag_D_s .* Z_matrix) .* diag_D_t'
                    st_field = s_U * transformed * t_U'
                    
                    effect_k[:, i] = vec(st_field)[st_idx_full]
                end
            else
                # Fallback to Cholesky reconstruction
                for i in 1:n_samples
                    sigma_i = sigma_samples[i, 1]
                    innovations_i = innovations_samples[i, :]
                    s_rho_val = isnothing(s_rho_samples) ? nothing : s_rho_samples[i, 1]
                    t_rho_val = isnothing(t_rho_samples) ? nothing : t_rho_samples[i, 1]
                    
                    Q_s = recompose_precision(s_model_type, Q_s_template, 1.0; extra_param=s_rho_val)
                    Q_t = recompose_precision(t_model_type, Q_t_template, 1.0; extra_param=t_rho_val)
                    
                    C_s = cholesky(Symmetric(Matrix(Q_s) + max(noise, 1e-4) * I))
                    C_t = cholesky(Symmetric(Matrix(Q_t) + max(noise, 1e-4) * I))
                    
                    Z_matrix = reshape(innovations_i, s_N, t_N)
                    tmp_spatial = C_s.L' \ Z_matrix
                    st_field_unscaled = transpose(C_t.L' \ transpose(tmp_spatial))
                    
                    st_field_unscaled .-= mean(st_field_unscaled)
                    st_field = st_field_unscaled .* sigma_i
                    
                    effect_k[:, i] = vec(st_field)[st_idx_full]
                end
            end
            
            push!(structured_effects, effect_k)
        end
        return (structured=structured_effects, noisy=structured_effects)

    elseif m.operator == :composition # Non-stationary variance
        structured_effects = Vector{Matrix{Float64}}()
        modifier_spec = child_specs[1]
        base_spec = child_specs[2]
        
        # Recursive calls with the correct, modern signature
        modifier_effects_all = get_effects(modifier_spec.component_obj, chain, modifier_spec,
            M, PS)
        base_effects_all = get_effects(base_spec.component_obj, chain, base_spec, M, PS)

        for k in 1:outcomes_N
            log_sigma_field = modifier_effects_all.structured[k]
            base_field = base_effects_all.structured[k]
            # Element-wise multiplication for non-stationary variance
            reconstructed_effects = base_field .* exp.(log_sigma_field)
            push!(structured_effects, reconstructed_effects)
        end
        return (structured=structured_effects, noisy=structured_effects)
    end
    
    N_total = isnothing(PS) ? M.y_N : M.y_N + size(PS.data, 1)
    structured_effects = [zeros(Float64, N_total, n_samples) for _ in 1:outcomes_N]
    @warn "Posterior reconstruction for Composed with operator :$(m.operator) is not fully implemented. Returning zero effects."
    return (structured=structured_effects, noisy=structured_effects)
end 
 
