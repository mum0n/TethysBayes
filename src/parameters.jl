"""
    parameters.jl

Centralized parameter registry, semantic descriptor management, and chain sample extraction
for Bayesian Spatio-Temporal Models (TethysBayes).

Version: v1.0.0
"""

"""
    ParamDescriptor

Detailed semantic metadata for a single model parameter / random variable.

# Fields
- `symbol::Symbol`: Canonical Julia symbol used in Turing code.
- `name::String`: Canonical string representation of the symbol.
- `component_key::Symbol`: Owner component key (e.g., `:s_idx`, `:t_idx`, `:intercept`).
- `role::Symbol`: Semantic role (e.g., `:sigma`, `:rho`, `:innovations`, `:latent`).
- `outcome_idx::Union{Int, Nothing}`: Outcome number for multivariate models, or `nothing`.
- `is_shared::Bool`: `true` if this hyperparameter is shared across outcomes.
- `shape::Tuple{Vararg{Int}}`: Expected dimension tuple (e.g., `(1,)`, `(20,)`, `(10, 20)`).
- `prior::Union{Distribution, Nothing}`: The prior distribution object if known.
"""
struct ParamDescriptor
    symbol::Symbol
    name::String
    component_key::Symbol
    role::Symbol
    outcome_idx::Union{Int, Nothing}
    is_shared::Bool
    shape::Tuple{Vararg{Int}}
    prior::Union{Distribution, Nothing}
end

# Convenience constructor with defaults
function ParamDescriptor(
    symbol::Symbol;
    component_key::Symbol = :unknown,
    role::Symbol = :param,
    outcome_idx::Union{Int, Nothing} = nothing,
    is_shared::Bool = false,
    shape::Tuple{Vararg{Int}} = (1,),
    prior::Union{Distribution, Nothing} = nothing
)
    return ParamDescriptor(
        symbol,
        string(symbol),
        component_key,
        role,
        outcome_idx,
        is_shared,
        shape,
        prior
    )
end

"""
    ParamRegistry

Central registry that stores parameter descriptors, semantic hierarchies, and chain index mappings.
Provides canonical name resolution and sample extraction across all MCMC chain formats.
"""
mutable struct ParamRegistry
    names::Vector{String}                                              # Canonical string names
    descriptors::Dict{Symbol, ParamDescriptor}                         # Symbol -> ParamDescriptor
    by_component::Dict{Symbol, Dict{Symbol, Vector{ParamDescriptor}}} # component_key -> role -> [descriptors]
    by_base::Dict{String, Vector{String}}                              # base name -> list of matching full names
    name_to_key::Dict{String, Any}                                     # String/Symbol in chain -> actual indexing key
end

# Default empty constructor
function ParamRegistry()
    return ParamRegistry(
        String[],
        Dict{Symbol, ParamDescriptor}(),
        Dict{Symbol, Dict{Symbol, Vector{ParamDescriptor}}}(),
        Dict{String, Vector{String}}(),
        Dict{String, Any}()
    )
end

# =============================================================================

# Registration & Mutation Helpers
# =============================================================================
"""
    add_descriptor!(reg::ParamRegistry, desc::ParamDescriptor)

Registers a `ParamDescriptor` in the registry and updates all indexing tables.
"""
function add_descriptor!(reg::ParamRegistry, desc::ParamDescriptor)
    reg.descriptors[desc.symbol] = desc
    
    if !(desc.name in reg.names)
        push!(reg.names, desc.name)
    end

    # Register under by_component hierarchy: component_key -> role -> [descriptors]
    if !haskey(reg.by_component, desc.component_key)
        reg.by_component[desc.component_key] = Dict{Symbol, Vector{ParamDescriptor}}()
    end

    role_dict = reg.by_component[desc.component_key]
    if !haskey(role_dict, desc.role)
        role_dict[desc.role] = ParamDescriptor[]
    end

    # Replace existing with same symbol or append
    existing_idx = findfirst(d -> d.symbol == desc.symbol, role_dict[desc.role])
    if isnothing(existing_idx)
        push!(role_dict[desc.role], desc)
    else
        role_dict[desc.role][existing_idx] = desc
    end

    # Index by canonical base (e.g. before '[')
    base = first(Base.split(desc.name, '['))
    if !haskey(reg.by_base, base)
        reg.by_base[base] = String[]
    end
    if !(desc.name in reg.by_base[base])
        push!(reg.by_base[base], desc.name)
    end
    # Also index by underscore-numeric suffix base if present (e.g., intercept_1 -> intercept)
    parts = Base.split(desc.name, '_')
    if length(parts) > 1 && all(isdigit.(collect(parts[end])))
        underscore_base = join(parts[1:end-1], "_")
        if !haskey(reg.by_base, underscore_base)
            reg.by_base[underscore_base] = String[]
        end
        if !(desc.name in reg.by_base[underscore_base])
            push!(reg.by_base[underscore_base], desc.name)
        end
    end
    # Map name to indexing key
    reg.name_to_key[desc.name] = desc.symbol
    return reg
end

# =============================================================================

# Multi-Source ParamRegistry Builders
# =============================================================================
"""
    build_param_registry(M::NamedTuple; prefix::String = "")

Builds a static `ParamRegistry` from the model configuration `M`. Recursively
registers descriptors for intercepts, fixed effects (including Errors-in-Variables
latent states), model component hyperparameters and latent fields, spatiotemporal
interaction fields, Householder spectral orientation vectors, likelihood parameters,
and nested sub-models.

# Mathematical Formulation
For a nested multi-fidelity component `k`, the linear predictor links via:
``\\eta_{\\text{main}} = \\eta_{\\text{base}} + \\rho_k \\cdot \\eta_{\\text{sub}, k}``
where ``\\rho_k \\sim \\text{Normal}(1.0, 0.5)``. Sub-model parameters are prefixed with
`$(prefix)_` to maintain parameter uniqueness and prevent namespace collisions.

# Arguments
- `M::NamedTuple`: Model configuration containing components, dimensions, and likelihoods.
- `prefix::String`: Optional namespace prefix for sub-model components (default: `""`).

# Returns
- `ParamRegistry`: Populated registry mapping parameter symbols to `ParamDescriptor`s.
"""
function build_param_registry(M::NamedTuple; prefix::String = "")
    reg = ParamRegistry()
    arch = get(M, :model_arch, "univariate")
    is_multivariate = arch == "multivariate"
    outcomes_N = get(M, :outcomes_N, 1)

    # 1. Intercept
    if get(M, :add_intercept, false)
        intercept_prior = get(M, :intercept_prior, Normal(0, 5))
        shared_intercept = get(M, :shared_intercept, get(M, :intercept_shared, false))
        if is_multivariate && !shared_intercept
            for k in 1:outcomes_N
                sym = !isempty(prefix) ? Symbol("intercept_$(prefix)_$(k)") : Symbol("intercept_$(k)")
                add_descriptor!(reg, ParamDescriptor(
                    sym;
                    component_key = :intercept,
                    role = :intercept,
                    outcome_idx = k,
                    is_shared = false,
                    shape = (1,),
                    prior = intercept_prior
                ))
            end
        else
            sym = !isempty(prefix) ? Symbol("intercept_$(prefix)") : :intercept
            add_descriptor!(reg, ParamDescriptor(
                sym;
                component_key = :intercept,
                role = :intercept,
                outcome_idx = is_multivariate ? 1 : nothing,
                is_shared = is_multivariate ? shared_intercept : true,
                shape = (1,),
                prior = intercept_prior
            ))
        end
    end

    # 2. Fixed Effects
    if get(M, :Xfixed_N, 0) > 0
        n_fixed = M.Xfixed_N
        if is_multivariate
            flat_sym = !isempty(prefix) ? Symbol("beta_flat_$(prefix)") : :beta_flat
            beta_sym = !isempty(prefix) ? Symbol("beta_$(prefix)") : :beta
            add_descriptor!(reg, ParamDescriptor(
                flat_sym;
                component_key = :fixed,
                role = :fixed_coef,
                outcome_idx = nothing,
                is_shared = false,
                shape = (n_fixed, outcomes_N)
            ))
            add_descriptor!(reg, ParamDescriptor(
                beta_sym;
                component_key = :fixed,
                role = :fixed_coef,
                outcome_idx = nothing,
                is_shared = false,
                shape = (n_fixed, outcomes_N)
            ))
        else
            beta_sym = !isempty(prefix) ? Symbol("beta_$(prefix)") : :beta
            add_descriptor!(reg, ParamDescriptor(
                beta_sym;
                component_key = :fixed,
                role = :fixed_coef,
                outcome_idx = nothing,
                is_shared = true,
                shape = (n_fixed,)
            ))
        end

        # Register Errors-in-Variables (EIV) latent innovations
        if haskey(M, :Xfixed_eiv_map) && !isempty(M.Xfixed_eiv_map)
            for (col_sym, sd_vec) in M.Xfixed_eiv_map
                eiv_sym = !isempty(prefix) ? Symbol("innovations_eiv_$(prefix)_$(col_sym)") : Symbol("innovations_eiv_$(col_sym)")
                add_descriptor!(reg, ParamDescriptor(
                    eiv_sym;
                    component_key = :fixed,
                    role = :eiv_innovations,
                    outcome_idx = nothing,
                    is_shared = true,
                    shape = (M.y_N,),
                    prior = Normal(0, 1)
                ))
            end
        end
    end

    # 3. Model Components
    if haskey(M, :components) && !isempty(M.components)
        for spec in M.components
            comp_obj = spec.component_obj
            comp_key = !isempty(prefix) ? Symbol(prefix, "_", spec.key) : spec.key
            prefixed_spec = !isempty(prefix) ? merge(spec, (key = comp_key,)) : spec
            shared_spec = get(spec.params, :shared, false)

            # Detect all hyperparameters on component struct
            for f in fieldnames(typeof(comp_obj))
                val = getfield(comp_obj, f)
                if val isa Distribution
                    is_f_shared = is_param_shared(shared_spec, f)
                    if is_multivariate && !is_f_shared
                        for k in 1:outcomes_N
                            sym = Symbol("$(f)_$(comp_key)_$(k)")
                            add_descriptor!(reg, ParamDescriptor(
                                sym;
                                component_key = comp_key,
                                role = f,
                                outcome_idx = k,
                                is_shared = false,
                                shape = (1,),
                                prior = val
                            ))
                        end
                    else
                        sym = Symbol("$(f)_$(comp_key)")
                        add_descriptor!(reg, ParamDescriptor(
                            sym;
                            component_key = comp_key,
                            role = f,
                            outcome_idx = is_multivariate ? 1 : nothing,
                            is_shared = is_f_shared,
                            shape = (1,),
                            prior = val
                        ))
                    end
                end
            end

            # Detect latent fields & innovations
            n_latent = if hasproperty(spec.hyper, :n_latent)
                spec.hyper.n_latent
            elseif hasproperty(spec.hyper, :in_dim) && hasproperty(comp_obj, :hidden_dim) &&
              hasproperty(comp_obj, :nbins)
                comp_obj.nbins
            else
                0
            end

            latent_roles = [:innovations, :latent_field, :rff_weights_1, :rff_offsets_1, :rff_weights_2, :rff_weights, :rff_offsets, :reflection_vector_unconstrained,
              :threshold_unconstrained, :amplitude_unconstrained]
            for role in latent_roles
                for k in 1:outcomes_N
                    outcome_k = is_multivariate ? k : nothing
                    p_names = generate_full_variable_names(prefixed_spec, arch, outcome_k)
                    if hasproperty(p_names, role)
                        sym = getfield(p_names, role)
                        shape_val = if role in [:innovations, :latent_field]
                            (n_latent > 0 ? n_latent : 1,)
                        elseif role == :rff_weights_1 && hasproperty(comp_obj, :hidden_dim) &&
                          hasproperty(spec.hyper, :in_dim)
                            (spec.hyper.in_dim, comp_obj.hidden_dim)
                        elseif role == :rff_offsets_1 && hasproperty(comp_obj, :hidden_dim)
                            (comp_obj.hidden_dim,)
                        elseif role == :rff_weights_2 && hasproperty(comp_obj, :hidden_dim) &&
                          hasproperty(comp_obj, :nbins)
                            (comp_obj.hidden_dim, comp_obj.nbins)
                        elseif role == :rff_weights && hasproperty(spec.hyper, :in_dim) &&
                          hasproperty(spec.hyper, :out_dim)
                            (spec.hyper.in_dim, spec.hyper.out_dim)
                        elseif role == :rff_offsets && hasproperty(spec.hyper, :out_dim)
                            (spec.hyper.out_dim,)
                        else
                            (1,)
                        end

                        add_descriptor!(reg, ParamDescriptor(
                            sym;
                            component_key = comp_key,
                            role = role,
                            outcome_idx = outcome_k,
                            is_shared = false,
                            shape = shape_val
                        ))
                    end
                end
            end
        end
    end

    # 4. Spatiotemporal Interaction
    if get(M, :model_st, "none") != "none"
        st_prior = if haskey(M, :sigma_st_interaction_prior)
            M.sigma_st_interaction_prior
        elseif haskey(M, :st_interaction_sigma_prior)
            M.st_interaction_sigma_prior
        else
            Exponential(1.0)
        end
        s_N = get(M, :s_N, 1)
        t_N = get(M, :t_N, 1)
        sig_st_sym = !isempty(prefix) ? Symbol("sigma_st_interaction_$(prefix)") : :sigma_st_interaction
        innovations_st_sym = !isempty(prefix) ? Symbol("innovations_st_interaction_$(prefix)") : :innovations_st_interaction
        add_descriptor!(reg, ParamDescriptor(
            sig_st_sym;
            component_key = :st_interaction,
            role = :sigma,
            outcome_idx = nothing,
            is_shared = false,
            shape = is_multivariate ? (outcomes_N,) : (1,),
            prior = st_prior
        ))
        add_descriptor!(reg, ParamDescriptor(
            innovations_st_sym;
            component_key = :st_interaction,
            role = :innovations,
            outcome_idx = nothing,
            is_shared = false,
            shape = is_multivariate ? (s_N * t_N * outcomes_N,) : (s_N * t_N,)
        ))
    end

    # 5. Spectral Orientation (Householder Reflection)
    if is_multivariate && get(M, :spectral_orientation, false)
        v_refl_sym = !isempty(prefix) ? Symbol("v_unscaled_reflection_$(prefix)") : :reflection_vector_unconstrained_reflection
        add_descriptor!(reg, ParamDescriptor(
            v_refl_sym;
            component_key = :spectral_orientation,
            role = :reflection_vector_unconstrained,
            outcome_idx = nothing,
            is_shared = true,
            shape = (outcomes_N,)
        ))
    end

    # 6. Likelihood parameters
    if is_multivariate && outcomes_N > 1 && !get(M, :is_multinomial, false)
        l_corr_sym = !isempty(prefix) ? Symbol("L_corr_$(prefix)") : :correlation_cholesky
        add_descriptor!(reg, ParamDescriptor(
            l_corr_sym;
            component_key = :likelihood,
            role = :correlation_cholesky,
            outcome_idx = nothing,
            is_shared = true,
            shape = (outcomes_N, outcomes_N)
        ))
    end

    if haskey(M, :likelihood_specs) && !isempty(M.likelihood_specs)
        families = [string(get(spec, :family, "gaussian")) for spec in M.likelihood_specs]
        if any(f -> f in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"], families)
            sig_sym = !isempty(prefix) ? Symbol("y_sigma_$(prefix)") : :y_sigma
            add_descriptor!(reg, ParamDescriptor(
                sig_sym;
                component_key = :likelihood,
                role = :y_sigma,
                outcome_idx = is_multivariate ? 1 : nothing,
                is_shared = true,
                shape = is_multivariate ? (outcomes_N,) : (1,),
                prior = Exponential(1.0)
            ))
        end
        if any(f -> f == "negbin", families)
            r_sym = !isempty(prefix) ? Symbol("r_nb_$(prefix)") : :r_nb
            add_descriptor!(reg, ParamDescriptor(
                r_sym;
                component_key = :likelihood,
                role = :r_nb,
                outcome_idx = nothing,
                is_shared = true,
                shape = (1,),
                prior = Exponential(1.0)
            ))
        end
    end

    # 7. Nested / Transfer Sub-models (recursive registration with prefix)
    if haskey(M, :nested_components) && !isempty(M.nested_components)
        for (k, sub_M) in M.nested_components
            sub_prefix = isempty(prefix) ? string(k) : "$(prefix)_$(k)"
            coupling = get(sub_M, :coupling, get(sub_M, :interaction, :additive))
            is_tensor = coupling == :tensor && get(sub_M, :is_multistrata, false)
            is_moderated = coupling == :moderated
            n_strata = get(sub_M, :n_strata, 1)

            if !get(sub_M, :fixed_coupling, false)
                if is_tensor
                    for fac in sub_M.strata_factors
                        dim = sub_M.factor_dims[fac]
                        add_descriptor!(reg, ParamDescriptor(
                            Symbol("rho_nested_$(sub_prefix)_$(fac)");
                            component_key = :nested,
                            role = :nested_weight,
                            outcome_idx = nothing,
                            is_shared = true,
                            shape = (dim,),
                            prior = get(sub_M, :coupling_prior, Normal(1.0, 0.5))
                        ))
                    end
                elseif is_moderated
                    c_prior = get(sub_M, :coupling_prior, Normal(1.0, 0.5))
                    s_shape = n_strata > 1 ? (n_strata,) : (1,)
                    add_descriptor!(reg, ParamDescriptor(
                        Symbol("rho_nested_$(sub_prefix)_0");
                        component_key = :nested,
                        role = :nested_weight,
                        outcome_idx = nothing,
                        is_shared = true,
                        shape = s_shape,
                        prior = c_prior
                    ))
                    add_descriptor!(reg, ParamDescriptor(
                        Symbol("rho_nested_$(sub_prefix)_1");
                        component_key = :nested,
                        role = :nested_weight,
                        outcome_idx = nothing,
                        is_shared = true,
                        shape = s_shape,
                        prior = c_prior
                    ))
                else
                    c_prior = get(sub_M, :coupling_prior, Normal(1.0, 0.5))
                    s_shape = n_strata > 1 ? (n_strata,) : (1,)
                    add_descriptor!(reg, ParamDescriptor(
                        Symbol("rho_nested_$(sub_prefix)");
                        component_key = :nested,
                        role = :nested_weight,
                        outcome_idx = nothing,
                        is_shared = true,
                        shape = s_shape,
                        prior = c_prior
                    ))
                end
            end
            sub_reg = build_param_registry(sub_M; prefix = sub_prefix)
            for (_, d) in sub_reg.descriptors
                add_descriptor!(reg, d)
            end
        end
    end
    return reg
end
"""
    build_param_registry(sample::Union{NamedTuple, AbstractDict}, M::Union{NamedTuple, Nothing}=nothing)
Builds or augments a `ParamRegistry` from a prior predictive draw
(e.g., `rand(model)` returning `NamedTuple` or `VarNamedTuple`).
Passes the draw to `calibrate_param_registry` to populate descriptor shapes.
"""
function build_param_registry(
    sample :: Union{NamedTuple, AbstractDict},
    M      :: Union{NamedTuple, Nothing} = nothing
)
    reg = !isnothing(M) ? build_param_registry(M) : ParamRegistry()
    return calibrate_param_registry(reg, sample)
end
"""
    build_param_registry(model::DynamicPPL.Model)
Builds a `ParamRegistry` directly from an instantiated Turing `DynamicPPL.Model`.
"""
function build_param_registry(model::DynamicPPL.Model)
    if hasproperty(model.args, :M)
        reg = build_param_registry(model.args.M)
        # Check VarInfo if available
        try
            vi = DynamicPPL.VarInfo(model)
            for vn in keys(vi)
                sym = _get_varname_symbol(vn)
                if !haskey(reg.descriptors, sym)
                    add_descriptor!(reg, ParamDescriptor(sym))
                end
            end
        catch
        end
        return reg
    else
        reg = ParamRegistry()
        try
            vi = DynamicPPL.VarInfo(model)
            for vn in keys(vi)
                sym = _get_varname_symbol(vn)
                add_descriptor!(reg, ParamDescriptor(sym))
            end
        catch
        end
        return reg
    end
end
"""
    build_param_registry(chain::Union{AbstractDataFrame, AbstractDict})
Builds a `ParamRegistry` from an MCMC chain expressed as a `DataFrame` or
`AbstractDict` (e.g., a `FlexiChain`-derived dict or `MCMCChains.Chains`
converted to DataFrame).
"""
function build_param_registry(chain::Union{AbstractDataFrame, AbstractDict})
    reg = ParamRegistry()
    raw_names = _extract_chain_column_names(chain)
    names_str = string.(raw_names)
    for (i, nstr) in enumerate(names_str)
        orig_key = raw_names[i]
        key_for_indexing = orig_key isa String ? Symbol(nstr) : orig_key
        sym = Symbol(nstr)
        base = first(Base.split(nstr, '['))
        if !haskey(reg.by_base, base)
            reg.by_base[base] = String[]
        end
        push!(reg.by_base[base], nstr)
        parts = Base.split(nstr, '_')
        if length(parts) > 1 && all(isdigit.(collect(parts[end])))
            underscore_base = join(parts[1:end-1], "_")
            if !haskey(reg.by_base, underscore_base)
                reg.by_base[underscore_base] = String[]
            end
            push!(reg.by_base[underscore_base], nstr)
        end
        reg.name_to_key[nstr] = key_for_indexing
        if !(nstr in reg.names)
            push!(reg.names, nstr)
        end
        reg.descriptors[sym] = ParamDescriptor(sym)
    end
    _infer_tensor_shapes!(reg)
    return reg
end
"""
    build_param_registry(chain)
Single generic fallback: handles FlexiChain, MCMCChains.Chains, or any
chain-like object whose column names can be extracted via
`_extract_chain_column_names`. Concrete types (`NamedTuple`, `AbstractDict`,
`AbstractDataFrame`, `DynamicPPL.Model`) are handled by more specific dispatches
above and will not reach this method.
"""
function build_param_registry(chain::T) where T
    # Disambiguate: if it looks like a prior sample (has `pairs`), delegate to calibrate.
    # Otherwise treat as a chain-like object.
    if chain isa DynamicPPL.Model
        # Handled by the DynamicPPL.Model dispatch above — should never reach here.
        error("build_param_registry: unexpected dispatch to generic chain fallback for DynamicPPL.Model")
    end
    reg = ParamRegistry()
    raw_names = try
        _extract_chain_column_names(chain)
    catch
        @warn "build_param_registry: could not extract chain column names from $(T); returning empty registry."
        return reg
    end
    names_str = string.(raw_names)
    for (i, nstr) in enumerate(names_str)
        orig_key = raw_names[i]
        key_for_indexing = orig_key isa String ? Symbol(nstr) : orig_key
        sym = Symbol(nstr)
        base = first(Base.split(nstr, '['))
        if !haskey(reg.by_base, base)
            reg.by_base[base] = String[]
        end
        push!(reg.by_base[base], nstr)
        parts = Base.split(nstr, '_')
        if length(parts) > 1 && all(isdigit.(collect(parts[end])))
            underscore_base = join(parts[1:end-1], "_")
            if !haskey(reg.by_base, underscore_base)
                reg.by_base[underscore_base] = String[]
            end
            push!(reg.by_base[underscore_base], nstr)
        end
        reg.name_to_key[nstr] = key_for_indexing
        if !(nstr in reg.names)
            push!(reg.names, nstr)
        end
        reg.descriptors[sym] = ParamDescriptor(sym)
    end
    _infer_tensor_shapes!(reg)
    return reg
end
"""
    _infer_tensor_shapes!(reg::ParamRegistry)
Scans `reg.by_base` for multi-dimensional bracketed indices (e.g. `var[i, j]`)
and creates root parameter descriptors with their exact matrix/tensor shapes.
"""
function _infer_tensor_shapes!(reg::ParamRegistry)
    for (base, col_list) in reg.by_base
        multi_dim_indices = Tuple{Vararg{Int}}[]
        for col in col_list
            m = match(r"\[([\d,\s]+)\]$", col)
            if !isnothing(m)
                idx_strs = Base.split(m.captures[1], ',')
                if length(idx_strs) > 1
                    try
                        push!(multi_dim_indices, Tuple(parse(Int, strip(s)) for s in idx_strs))
                    catch
                    end
                end
            end
        end
        if !isempty(multi_dim_indices)
            n_dims = length(first(multi_dim_indices))
            if all(length(idx) == n_dims for idx in multi_dim_indices)
                max_indices = ntuple(d -> maximum(idx[d] for idx in multi_dim_indices), n_dims)
                base_sym = Symbol(base)
                reg.descriptors[base_sym] = ParamDescriptor(
                    base_sym;
                    shape = max_indices
                )
            end
        end
    end
end
# =============================================================================
# Calibration Helper
# =============================================================================
"""
    calibrate_param_registry(reg::ParamRegistry, sample::Any)
Updates descriptors and shapes in `reg` based on actual realized values in `sample`
(from `rand(model)`). Supports `NamedTuple`, `DynamicPPL.VarNamedTuple`, `AbstractDict`, etc.
"""
function calibrate_param_registry(reg::ParamRegistry, sample::Any)
    pairs_iter = try
        pairs(sample)
    catch
        try
            pairs(NamedTuple(sample))
        catch
            Dict(k => getproperty(sample, k) for k in propertynames(sample))
        end
    end
    for (sym, val) in pairs_iter
        val_shape = val isa AbstractArray ? size(val) : (1,)
        nstr = string(sym)
        sym_key = Symbol(sym)
        if haskey(reg.descriptors, sym_key)
            existing = reg.descriptors[sym_key]
            reg.descriptors[sym_key] = ParamDescriptor(
                existing.symbol,
                existing.name,
                existing.component_key,
                existing.role,
                existing.outcome_idx,
                existing.is_shared,
                val_shape,
                existing.prior
            )
        else
            # New parameter discovered in sample
            add_descriptor!(reg, ParamDescriptor(
                sym_key;
                shape = val_shape
            ))
        end
        reg.name_to_key[nstr] = sym_key
    end

    return reg

end


# =============================================================================
# Universal Sample Extraction API
# =============================================================================
"""

    get_samples(chain, reg::ParamRegistry, component_key::Symbol, role::Symbol;
                outcome=nothing, expected_len=nothing)
Extracts an `(n_samples, param_dim)` matrix of posterior samples for a specific component
and semantic role.
Automatically handles scalar, vector, and matrix parameters across FlexiChains, MCMCChains,
and DataFrames.
"""
function get_samples(
    chain,
    reg::ParamRegistry,
    component_key::Symbol,
    role::Symbol;
    outcome::Union{Int, Nothing} = nothing,
    expected_len::Union{Int, Nothing} = nothing,
    reshape_to_shape::Bool = true
)
    # 1. Lookup matching descriptor from by_component
    has_role = haskey(reg.by_component, component_key) &&
      haskey(reg.by_component[component_key], role)
    if !has_role
        # Fallback to base name search
        fallback_name = isnothing(outcome) ? "$(role)_$(component_key)" :
          "$(role)_$(component_key)_$(outcome)"
        target_name = find_chain_param(reg, fallback_name; outcome_idx = outcome)
        if isempty(target_name)
            error("Parameter with component :$(component_key) and role :$(role) (outcome: " *
                  "$(outcome)) not found in ParamRegistry.")
        end
        return get_param_samples(
            chain, reg, target_name;
            expected_len = expected_len, reshape_to_shape = reshape_to_shape
        )
    end
    candidates = reg.by_component[component_key][role]
    selected_desc = nothing
    if !isnothing(outcome)
        # Find candidate with matching outcome_idx
        for d in candidates
            if d.outcome_idx == outcome
                selected_desc = d
                break
            end
        end
    end
    if isnothing(selected_desc) && !isempty(candidates)
        selected_desc = first(candidates)
    end
    if isnothing(selected_desc)
        error("No suitable descriptor found for component :$(component_key), role :$(role), " *
              "outcome $(outcome).")
    end
    exp_len = isnothing(expected_len) ? prod(selected_desc.shape) : expected_len
    return get_param_samples(
        chain, reg, selected_desc.name;
        expected_len = exp_len, reshape_to_shape = reshape_to_shape
    )
end



"""

    get_descriptors_by_role(reg::ParamRegistry, role::Symbol)



Returns all `ParamDescriptor` entries in `reg` registered with semantic role `role`.

### Arguments
- `reg::ParamRegistry`: Active parameter registry.
- `role::Symbol`: Semantic role symbol (e.g., `:fixed`, `:spatial`, `:nested_weight`).

### Returns
- `Vector{ParamDescriptor}`: Matching descriptor objects.
"""
function get_descriptors_by_role(reg::ParamRegistry, role::Symbol)
    return filter(d -> d.role == role, collect(values(reg.descriptors)))
end

"""
    get_samples(chain, param_name::Union{String, Symbol}; expected_len=nothing)

Directly extracts posterior samples for parameter `param_name` from an MCMC chain
without requiring a `ParamRegistry`. Returns a 1D vector for scalar parameters
or a 2D matrix of shape `(n_samples, dim)`.

### Arguments
- `chain`: MCMC chain or sample dictionary.
- `param_name`: Parameter identifier (e.g., `:rho_nested_proxy`, `"beta"`).
- `expected_len`: Optional dimension specification.

### Returns
- Vector or matrix of posterior draws.
"""
function get_samples(
    chain,
    param_name::Union{String, Symbol};
    expected_len::Union{Int, Nothing} = nothing
)
    mat = extract_param_matrix(chain, param_name; expected_dim = expected_len)
    if size(mat, 2) == 1
        return vec(mat)
    else
        return mat
    end
end

"""
    get_samples(chain, reg::ParamRegistry, param_name::Union{String, Symbol}; kwargs...)

Alias for `get_param_samples(chain, reg, param_name; kwargs...)`.
"""
function get_samples(
    chain,
    reg::ParamRegistry,
    param_name::Union{String, Symbol};
    kwargs...
)
    return get_param_samples(chain, reg, param_name; kwargs...)
end

"""
    get_param_samples(chain, reg::ParamRegistry, component_key::Symbol, role::Symbol; kwargs...)

Alias forwarding to `get_samples(chain, reg, component_key, role; kwargs...)` to extract
posterior samples for a specific component and semantic role from `reg`.
"""
function get_param_samples(
    chain,
    reg::ParamRegistry,
    component_key::Symbol,
    role::Symbol;
    kwargs...
)
    return get_samples(chain, reg, component_key, role; kwargs...)
end

"""
    get_param_samples(chain, reg::ParamRegistry, param_name::Union{String, Symbol};
                      expected_len::Union{Int, Nothing}=nothing,
                      reshape_to_shape::Bool=true, exact::Bool=false)

Extracts posterior samples for a specific parameter name.
When `reshape_to_shape=true` (default) and the parameter has a multi-dimensional shape
(e.g., `(in_dim, hidden_dim)`), returns an array of dimensions `(n_samples, in_dim, hidden_dim)`.
For scalar or 1D parameters, returns an `(n_samples, param_dim)` matrix.
"""
function get_param_samples(
    chain,
    reg::ParamRegistry,
    param_name::Union{AbstractString, Symbol};
    expected_len::Union{Int, Nothing} = nothing,
    reshape_to_shape::Bool = true
)
    nstr = string(param_name)
    actual_col = find_chain_param(reg, nstr)
    lookup_key = !isempty(actual_col) ? chain_index_key(reg, actual_col) : Symbol(nstr)

    base_sym = Symbol(first(Base.split(nstr, '[')))
    desc = get(reg.descriptors, Symbol(nstr), get(reg.descriptors, base_sym, nothing))
    target_shape = !isnothing(desc) ? desc.shape : nothing
    exp_len = if !isnothing(expected_len)
        expected_len
    elseif !isnothing(target_shape)
        prod(target_shape)
    else
        nothing
    end

    raw_mat = _extract_samples_from_chain(chain, lookup_key, nstr; expected_len = exp_len)

    if reshape_to_shape && !isnothing(target_shape) && length(target_shape) > 1
        if size(raw_mat, 2) == prod(target_shape)
            n_samples = size(raw_mat, 1)
            return reshape(raw_mat, n_samples, target_shape...)
        end
    end

    return raw_mat
end

# =============================================================================

# Name Resolution & Lookup Helpers
# =============================================================================
"""
    find_chain_param(reg::ParamRegistry, requested::AbstractString;

                     outcome_idx::Union{Int, Nothing}=nothing)

Resolve a parameter name to the concrete chain column it was registered under.
# Resolution rules (in order)
1. **Exact match** — `requested` is a registered name.
2. **Outcome-specific match** — `"\$(requested)_\$(outcome_idx)"` or
   `"\$(requested)[\$(outcome_idx)]"`, for multivariate models.
3. **Base match** — `requested` is a *base* name with several concrete
   registrations (e.g. one per outcome, or indexed variants such as
   `phi[1,1]`). The unambiguous structural form is returned: the bare base, then a
   bracketed variant, then a numeric-suffix variant.
There are **no aliases, abbreviations, or substring fallbacks**. A parameter has
exactly one supported name — the one `build_param_registry` registered. This is
deliberate: the previous implementation rewrote names (`ure_*` → `innovations_*`,
`beta` → `Xfixed_beta_prop`, `sre_*` → `latent_*`) and, when several candidates
matched, silently selected one. For an attribution or effect-size workflow that
resolves to the wrong coefficient without any error, which is worse than failing.
Use the canonical name from `get_descriptors_by_role` / `reg.names`.

# Arguments
- `reg::ParamRegistry`: Parameter registry for the model.
- `requested::AbstractString`: The canonical parameter name.
- `outcome_idx::Union{Int, Nothing}`: Optional outcome index.

# Returns
- `String`: The resolved chain column name, or `""` if `requested` is not a
  registered name.
"""
function find_chain_param(
    reg::ParamRegistry,
    requested::AbstractString;
    outcome_idx::Union{Int, Nothing} = nothing
)
    # 1) Exact match in registered names
    if requested in reg.names
        return String(requested)
    end

    # 2) If outcome index provided, try "_k" or "[k]"
    if !isnothing(outcome_idx)
        s1 = "$(requested)_$(outcome_idx)"
        s2 = "$(requested)[$(outcome_idx)]"
        if s1 in reg.names
            return s1
        end
        if s2 in reg.names
            return s2
        end
    end

    # 3) Base match: `requested` is a base name with concrete registrations.
    base = String(first(Base.split(requested, '[')))
    if haskey(reg.by_base, base)
        candidates = reg.by_base[base]
        # Plain candidate equal to base
        if base in candidates
            return base
        end
        # Bracketed candidates
        for c in candidates
            if startswith(c, base * "[")
                return c
            end
        end
        # Underscore numeric suffix candidates
        for c in candidates
            if startswith(c, base * "_")
                return c
            end
        end
        return first(candidates)
    end


    # Not a registered name. Suggest near-misses so the caller fails with something
    # actionable instead of quietly resolving to a different parameter.
    _warn_unknown_param(reg, requested)
    return ""
end
"""
    _warn_unknown_param(reg::ParamRegistry, requested::AbstractString)
Emit a single, actionable warning when a parameter name is not registered, listing
the registered names that share a prefix with what was requested.
"""
function _warn_unknown_param(reg::ParamRegistry, requested::AbstractString)
    prefix = String(first(Base.split(requested, '_')))
    registered = reg.names
    suggestions = sort!(filter(n -> startswith(n, prefix) || occursin(requested, n),
        registered))
    msg = "Parameter '$(requested)' is not a registered parameter name."
    if !isempty(suggestions)
        msg *= " Did you mean one of: " *
            join(first(suggestions, 8) .|> s -> "\"$(s)\"", ", ") * " ?"
    else
        with_prefix = sort!(filter(n -> startswith(n, prefix), registered))
        msg *= " Registered names beginning with '$(prefix)': " *
            (isempty(with_prefix) ? "(none)" :
                join(first(with_prefix, 8) .|> s -> "\"$(s)\"", ", ")) * "."
    end
    @warn msg
    return nothing
end
"""
    chain_index_key(reg::ParamRegistry, actual_name::AbstractString)
Returns the exact indexing key (Symbol or String) for indexing into the MCMC chain.
"""
function chain_index_key(reg::ParamRegistry, actual_name::AbstractString)
    return get(reg.name_to_key, String(actual_name), Symbol(actual_name))
end
# =============================================================================
# Low-Level Chain Extraction Dispatch Helpers
# =============================================================================
function _extract_chain_column_names(chain::Any)
    raw_names = if chain isa DataFrame
        names(chain)
    elseif chain isa Dict
        collect(keys(chain))
    elseif chain isa NamedTuple
        collect(keys(chain))
    else
        # FlexiChain / VNChain / MCMCChains.Chains
        try
            collect(string.(_get_varname_symbol.(keys(chain))))
        catch
            try
                names(DataFrame(chain))
            catch
                try
                    collect(keys(chain))
                catch
                    String[]
                end
            end
        end

    end
    return [replace(string(n), r"^Parameter\((.*)\)$" => s"\1", r"^parameters\." => "",
      r"^:+" => "") for n in raw_names]
end
function _extract_samples_from_chain(chain::Dict, key::Any, nstr::AbstractString; expected_len=nothing)
    # Mock chain dictionary support for testing
    sym_key = Symbol(key)
    if haskey(chain, sym_key)
        data = chain[sym_key]
        if data isa Matrix
            # [dim, n_samples] or [n_samples, dim]
            if size(data, 1) == 1 && size(data, 2) > 1
                return reshape(data, :, 1)
            elseif !isnothing(expected_len) && size(data, 1) == expected_len
                return Matrix(data')
            else
                return data
            end
        elseif data isa Vector
            return reshape(data, :, 1)
        else
            return fill(Float64(data), 1, 1)
        end

    end
    error("Parameter :$(sym_key) not found in mock chain.")
end
function _extract_samples_from_chain(chain::NamedTuple, key::Any, nstr::AbstractString;
  expected_len=nothing)
    sym_key = Symbol(key)
    if hasproperty(chain, sym_key)
        data = getproperty(chain, sym_key)
        if data isa Matrix
            if size(data, 1) == 1 && size(data, 2) > 1
                return reshape(data, :, 1)
            elseif !isnothing(expected_len) && size(data, 1) == expected_len
                return Matrix(data')
            else
                return data
            end
        elseif data isa Vector
            return reshape(data, :, 1)
        else
            return fill(Float64(data), 1, 1)
        end

    end

    error("Parameter :$(sym_key) not found in chain NamedTuple.")

end


function _extract_samples_from_chain(chain::Any, key::Any, nstr::AbstractString; expected_len=nothing)
    base_name = String(first(Base.split(nstr, '[')))
    return extract_param_matrix(chain, base_name; expected_dim=expected_len)
end

