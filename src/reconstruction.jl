"""
    reconstruction.jl

Posterior reconstruction, analytical post-processing, latent field synthesis,
denoised/noisy point predictions, WAIC computation, and MCMC convergence diagnostics for TethysBayes.

Version: v1.0.0
"""
# ==============================================================================
# SECTION 1: CORE UTILITIES FOR PARAMETER EXTRACTION
# ==============================================================================

"""
    get_kernel_from_string(kernel_name::String)::Kernel

Maps a string identifier to its corresponding `KernelFunctions.jl` kernel instance.
"""
function get_kernel_from_string(kernel_name::String)
    k_name = lowercase(kernel_name)
    if k_name == "constant"
        return ConstantKernel()
    elseif k_name == "linear"
        return LinearKernel()
    elseif k_name in ["matern12", "exponential"]
        return Matern12Kernel()
    elseif k_name == "matern32"
        return Matern32Kernel()
    elseif k_name == "matern52"
        return Matern52Kernel()
    elseif k_name == "spherical"
        return SphericalKernel()
    elseif k_name in ["squared_exponential", "se", "gaussian", "rbf"]
        return SqExponentialKernel()
    elseif k_name == "periodic"
        return PeriodicKernel()
    else
        @warn "Kernel '$kernel_name' not recognized. Defaulting to SqExponentialKernel."
        return SqExponentialKernel()
    end
end

"""
    _get_clean_chain_param_names(chain::Any)::Vector{String}

Extracts clean parameter names from chains of various formats (`VNChain`, `FlexiChain`,
  `DataFrame`, `Dict`, `NamedTuple`).
"""
function _get_clean_chain_param_names(chain::Any)::Vector{String}
    raw_names = if chain isa DataFrame
        string.(names(chain))
    elseif chain isa Dict
        string.(keys(chain))
    elseif chain isa NamedTuple
        string.(keys(chain))
    else
        try
            collect(string.(_get_varname_symbol.(keys(chain))))
        catch
            try
                string.(names(DataFrame(chain)))
            catch
                try
                    string.(collect(keys(chain)))
                catch
                    String[]
                end
            end
        end
    end

    clean_names = String[]
    for n in raw_names
        cleaned = replace(string(n), r"^Parameter\((.*)\)$" => s"\1", r"^parameters\." => "", r"^:+" => "")
        push!(clean_names, cleaned)
    end
    return unique(clean_names)
end

"""
    _get_chain_n_samples(chain::Any)::Int

Robustly extracts the total number of posterior samples from any chain type
(`VNChain`, `FlexiChain`, `MCMCChains.Chains`, `DataFrame`, `Dict`, `NamedTuple`).
"""
function _get_chain_n_samples(chain::Any)::Int
    if occursin("FlexiChain", string(typeof(chain))) || occursin("VNChain", string(typeof(chain)))
        try
            return size(chain, 1) * FlexiChains.nchains(chain)
        catch
            return size(chain, 1)
        end
    elseif chain isa DataFrame
        return nrow(chain)
    elseif chain isa AbstractDict || chain isa NamedTuple
        if isempty(chain)
            return 0
        end
        first_val = first(values(chain))
        if first_val isa AbstractArray
            if eltype(first_val) <: AbstractArray
                return length(first_val)
            elseif ndims(first_val) >= 2 && size(first_val, 1) == 1
                return size(first_val, 2)
            else
                return size(first_val, 1)
            end
        else
            return 1
        end
    else
        try
            return size(chain, 1) * size(chain, 3)
        catch
            try
                return size(chain, 1)
            catch
                return 1
            end
        end
    end
end

"""
    _resolve_hyper_samples(chain, p_names, target, value, k, is_multivariate_model, n_samples)

Posterior draws for one scalar hyperparameter of a component, or `nothing` when it cannot be
resolved.

**Why this exists.** Every component's `get_effects` used to bail to a zero matrix whenever a
hyperparameter it needed was missing from the chain:

    sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
    if isempty(sigma_name)
        @warn "Parameters for ... not found. Returning zero-matrix."
        push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
        continue
    end

That conflates two unrelated situations. A hyperparameter the user **pinned to a constant** is
not a chain parameter at all: `_prior_or_constant` emits `name = value` rather than a
`~` statement, so DynamicPPL never records it and `_find_parameter` can never find it. Those
runs were reconstructing a **zero field with only a warning** -- silently wrong, for exactly
the configuration a user pins a hyperparameter in order to hold it fixed. The 51 sites that
carried this pattern are listed in `todo.md`; the failure was confirmed end-to-end for `FITC`,
`ICAR` and `BYM2`.

So: look the parameter up first, and if it is absent, fall back to the component's own
declared value when that is a `Real`. A pinned hyperparameter has no posterior variation by
construction, so replicating the constant across draws is the exact answer, not an
approximation. `nothing` is returned only when the hyperparameter is genuinely unresolvable --
neither a chain parameter nor a constant -- and only then is a zero matrix defensible.
"""
function _resolve_hyper_samples(
    chain, p_names, target, value, k, is_multivariate_model::Bool, n_samples::Integer;
    as_matrix::Bool = false
)
    name = _find_parameter(p_names, string(target), k, is_multivariate_model)
    samples = if !isempty(name)
        get_params_vector(chain, name, 1)[:, 1]
    elseif value isa Real
        fill(Float64(value), Int(n_samples))
    else
        return nothing
    end
    # Some sites index their samples as `sigma_samples[i, 1]` and some as `sigma_samples[i]`.
    # Returning the requested shape here rather than reshaping at each call site keeps the
    # `isnothing` guard correct: `reshape(nothing, ...)` throws, so a reshape placed BEFORE
    # the guard would turn the genuinely-absent path into an error.
    return as_matrix ? reshape(samples, Int(n_samples), 1) : samples
end

"""
    _find_parameter(reg::ParamRegistry, target_name_base, outcome_idx=nothing,
      is_multivariate_model=false)
    _find_parameter(p_names, target_name_base, outcome_idx=nothing, is_multivariate_model=false)

Finds the canonical parameter name in a chain or `ParamRegistry` matching `target_name_base`
  and optional outcome index.
"""
function _find_parameter(
    reg::ParamRegistry, target_name_base::Union{String, Symbol}, outcome_idx=nothing, is_multivariate_model::Bool=false
)
    return find_chain_param(reg, string(target_name_base); outcome_idx=outcome_idx)
end

function _find_parameter(
    p_names, target_name_base::Union{String, Symbol}, outcome_idx=nothing, is_multivariate_model::Bool=false
)
    raw_base_str = string(target_name_base)
    base_str = replace(raw_base_str, r"^Parameter\((.*)\)$" => s"\1", r"^parameters\." => "",
        r"^:+" => "")
    p_names_str = [replace(string(n), r"^Parameter\((.*)\)$" => s"\1", r"^parameters\." => "", r"^:+" => "") for n in p_names]

    # Priority 1: Exact match with outcome index suffix (e.g., "sigma_s_idx_1").
    if is_multivariate_model && !isnothing(outcome_idx)
        specific_name_with_k_suffix = "$(base_str)_$(outcome_idx)"
        if specific_name_with_k_suffix in p_names_str
            return specific_name_with_k_suffix
        end
        
        indexed_name_with_k_bracket = "$(base_str)[$(outcome_idx)]"
        if indexed_name_with_k_bracket in p_names_str
            return indexed_name_with_k_bracket
        end
    end

    # Priority 2: Exact match for the base name
    if base_str in p_names_str
        return base_str
    end

    # Priority 3: Check for any bracketed indexed versions of the base name.
    re_indexed_any = Regex("^" * escape_string(base_str) * "\\[\\d+\\]")
    if any(n -> occursin(re_indexed_any, n), p_names_str)
        return base_str
    end

    # Priority 4: Substring match (return cleaned base_str)
    for n in p_names_str
        if occursin(base_str, n)
            return base_str
        end
    end

    return "" # Return empty string if no match is found.
end

"""
    _resolve_effect_indices(M::NamedTuple, PS::Union{NamedTuple, Nothing}, var_sym::Symbol)

Extracts and harmonizes spatial, temporal, or general unit observation indices
across training data and optional out-of-sample prediction sets (`PS`).

# Arguments
- `M::NamedTuple`: Main model configuration dictionary or NamedTuple.
- `PS::Union{NamedTuple, Nothing}`: Optional prediction set configuration.
- `var_sym::Symbol`: Target index variable (`:s_idx`, `:t_idx`, `:st_idx`, etc.).

# Returns
- `Tuple{Vector{Int}, Int}`: `(full_indices, N_total)` where `N_total = length(full_indices)`.
"""
function _resolve_effect_indices(
    M::NamedTuple,
    PS::Union{NamedTuple, Nothing},
    var_sym::Symbol = :s_idx
)::Tuple{Vector{Int}, Int}
    train_idx = if hasproperty(M, var_sym)
        collect(Int, getproperty(M, var_sym))
    elseif haskey(M, var_sym)
        collect(Int, M[var_sym])
    else
        ones(Float64, n_samples, outcomes_N)
    end

    full_idx = if !isnothing(PS)
        pred_idx = if hasproperty(PS, :data) && hasproperty(PS.data, var_sym)
            collect(Int, getproperty(PS.data, var_sym))
        elseif hasproperty(PS, :data) && haskey(PS.data, var_sym)
            collect(Int, PS.data[var_sym])
        elseif hasproperty(PS, var_sym)
            collect(Int, getproperty(PS, var_sym))
        elseif haskey(PS, var_sym)
            collect(Int, PS[var_sym])
        else
            Int[]
        end
        vcat(train_idx, pred_idx)
    else
        train_idx
    end

    return (full_idx, length(full_idx))
end

"""
    extract_chain_scalar(chain, param_prefix::Union{String, Symbol}; default=nothing)

Robustly extracts a scalar parameter vector across all posterior draws in an MCMC chain,
supporting `VNChain`, `FlexiChain`, `Chains`, `DataFrame`, `Dict`, and `NamedTuple`.

# Arguments
- `chain`: MCMC posterior chain or container.
- `param_prefix`: Base symbol or string name of the target parameter.
- `default`: Optional fallback value if parameter is not located. If `nothing`, throws error.

# Returns
- `Vector{Float64}`: Vector of posterior parameter values of length `n_samples`.
"""
function extract_chain_scalar(
    chain,
    param_prefix::Union{String, Symbol};
    default::Union{Nothing, Real, AbstractVector{<:Real}} = nothing
)::Vector{Float64}
    pref_str = string(param_prefix)

    # 1. Direct key lookup for Dict / NamedTuple
    if chain isa NamedTuple || chain isa AbstractDict
        for (k, v) in pairs(chain)
            sk = string(k)
            if sk == pref_str || sk == "$(pref_str)_1" || sk == "$(pref_str)_s_idx"
                return v isa AbstractVector ? Float64.(v) : [Float64(v)]
            end
        end
    end

    # 2. Extract clean parameter names
    p_names = _get_clean_chain_param_names(chain)
    matched = _find_parameter(p_names, pref_str, 1, false)

    if !isempty(matched)
        return get_params_vector(chain, matched, 1)[:, 1]
    end

    # 3. Check for common prefix variants
    for candidate in ["$(pref_str)_s_idx", "$(pref_str)_t_idx"]
        m_cand = _find_parameter(p_names, candidate, 1, false)
        if !isempty(m_cand)
            return get_params_vector(chain, m_cand, 1)[:, 1]
        end
    end

    # 4. Handle default or informative error
    if !isnothing(default)
        n_s = _get_chain_n_samples(chain)
        return default isa AbstractVector ? Float64.(default) : fill(Float64(default), n_s)
    end

    error("Parameter matching '$(pref_str)' was not found in the MCMC chain.")
end


"""
    _apply_multivariate_correlation(eta_latent, chain, outcomes_N)

Applies the estimated Cholesky correlation factor `correlation_cholesky` to independent multivariate
  latent fields.
"""
function _apply_multivariate_correlation(eta_latent, chain, outcomes_N)
    if outcomes_N == 1
        return eta_latent
    end
    N_tot, n_samples, _ = size(eta_latent)
    L_corr_samples = get_params_vector(chain, "correlation_cholesky", outcomes_N * outcomes_N)
    eta_final = zeros(N_tot, n_samples, outcomes_N)
    for s in 1:n_samples
        L_s = reshape(L_corr_samples[s, :], outcomes_N, outcomes_N)
        eta_final[:, s, :] = eta_latent[:, s, :] * L_s'
    end
    return eta_final
end


"""
    _summarize_effects_registry(registry, M, outcomes_N, alpha)

Computes posterior summary statistics (mean, median, std, lower/upper credible bounds) for
all structured and unstructured component effects.
"""
function _summarize_effects_registry(registry, M, outcomes_N, alpha)
    summarized_registry = Dict{Symbol, Any}()
    mixed_effects_summaries = Dict{Symbol, Any}()

    # 1. Summarize Component Effects
    comp_dict = haskey(registry, :components) ? registry.components : Dict{Symbol, Any}()
    for (key, effects) in pairs(comp_dict)
        spec_idx = findfirst(s -> s.key == key, M.components)
        if !isnothing(spec_idx) && M.components[spec_idx].component_obj isa Mixed
            # --- Handle Mixed Effects Separately ---
            summaries_per_outcome = [Dict{Symbol, Any}() for _ in 1:outcomes_N]
            if effects.type == :simple
                for k in 1:outcomes_N
                    summaries_per_outcome[k][Symbol(effects.lhs)] = summarize_array(effects.effects[k], alpha=alpha)
                end
            elseif effects.type == :correlated
                for (term_name, term_effects) in pairs(effects.effects)
                    for k in 1:outcomes_N
                        summaries_per_outcome[k][term_name] = summarize_array(term_effects[k],
                            alpha=alpha)
                    end
                end
            end
            
            summaries_final = outcomes_N > 1 ? [NamedTuple(s) for s in summaries_per_outcome] : NamedTuple(summaries_per_outcome[1])
            mixed_effects_summaries[key] = (grouping_covariate=M.components[spec_idx].var,
                summaries=summaries_final)
        else
            # --- Handle Standard Components ---
            component_summary = Dict{Symbol, Any}()
            for effect_type in keys(effects)
                effect_set = getproperty(effects, effect_type)
                if !(effect_set isa AbstractVector)
                    continue
                end # Skip non-effect fields like :indices
                if isempty(effect_set)
                    continue
                end

                if outcomes_N > 1
                    component_summary[effect_type] = [summarize_array(effect_set[k],
                        alpha=alpha) for k in 1:outcomes_N]
                else
                    component_summary[effect_type] = summarize_array(effect_set[1], alpha=alpha)
                end
            end
            if !isempty(component_summary)
                summarized_registry[key] = NamedTuple(component_summary)
            end
        end
    end
    if !isempty(mixed_effects_summaries)
        summarized_registry[:mixed_effects] = NamedTuple(mixed_effects_summaries)
    end

    # 2. Summarize Spatiotemporal Interaction Effects
    if haskey(registry, :st_interaction) && !isempty(registry.st_interaction) && !all(iszero,
        registry.st_interaction)
        st_mat = registry.st_interaction
        summarized_registry[:st_interaction] = outcomes_N > 1 ? [summarize_array(st_mat[:, :,
            k], alpha=alpha) for k in 1:outcomes_N] : summarize_array(st_mat[:, :, 1],
            alpha=alpha)
    end

    # 3. Summarize Fixed Effects
    if haskey(registry, :fixed_effects) && !isempty(registry.fixed_effects) && !all(iszero,
        registry.fixed_effects)
        fe_mat = registry.fixed_effects
        summarized_registry[:fixed] = outcomes_N > 1 ? [summarize_array(fe_mat[:, :, k],
            alpha=alpha) for k in 1:outcomes_N] : summarize_array(fe_mat[:, :, 1], alpha=alpha)
    end

    # 4. Summarize Intercept
    if haskey(registry, :intercept) && !isempty(registry.intercept) && !all(iszero,
        registry.intercept)
        int_mat = registry.intercept
        summarized_registry[:intercept] = outcomes_N > 1 ? [summarize_array(int_mat[:, k],
            alpha=alpha) for k in 1:outcomes_N] : summarize_array(int_mat[:, 1], alpha=alpha)
    end

    return NamedTuple(summarized_registry)
end


"""
    summarize_predictions(samples::AbstractArray; alpha=0.05)

Computes summary statistics (mean, median, std, lower/upper credible intervals) from a
  matrix of posterior samples.
"""
function summarize_predictions(samples::AbstractArray; alpha=0.05)
    # Ensure the input array is not empty or full of NaNs to prevent errors.
    if isempty(samples) || all(isnan, samples)
        return (mean = Float64[], median = Float64[], std = Float64[], lower = Float64[],
            upper = Float64[])
    end

    # The last dimension is assumed to be the dimension over which samples are drawn.
    sample_dim = ndims(samples)
    low_prob = alpha / 2.0
    high_prob = 1.0 - low_prob

    # Calculate standard summary statistics across the sample dimension.
    # `dropdims` removes the singleton dimension after the reduction.
    post_mean = dropdims(Statistics.mean(samples, dims=sample_dim), dims=sample_dim)
    post_median = dropdims(Statistics.median(samples, dims=sample_dim), dims=sample_dim)
    post_std = dropdims(Statistics.std(samples, dims=sample_dim), dims=sample_dim)

    # Calculate quantiles for the credible intervals.
    # `_quantile_along_last_dim` is a helper to compute quantiles over the last dimension.
    low_bound = _quantile_along_last_dim(samples, low_prob; sample_dim=sample_dim)
    high_bound = _quantile_along_last_dim(samples, high_prob; sample_dim=sample_dim)

    # Helper function to ensure all outputs are consistently formatted as vectors.
    to_vector(x) = x isa AbstractArray ? vec(collect(Float64, x)) : [Float64(x)]

    # Return the results as a NamedTuple.
    return (
        mean = to_vector(post_mean),
        median = to_vector(post_median),
        std = to_vector(post_std),
        lower = to_vector(low_bound),
        upper = to_vector(high_bound)
    )
end



function _discover_component_realizations(
    chain, M::NamedTuple, PS::Union{NamedTuple, Nothing},
    n_samples::Int, outcomes_N::Int, N_tot::Int; prefix::String = ""
)
    p_names = _get_clean_chain_param_names(chain)

    # --- Intercept ---
    intercept_samples = zeros(Float64, n_samples, outcomes_N)
    if M.add_intercept
        inter_base = !isempty(prefix) ? "intercept_$(prefix)" : "intercept"
        found_joint = _find_parameter(p_names, inter_base, nothing, false)
        if !isempty(found_joint) && outcomes_N > 1
            try
                all_inter = get_params_matrix(chain, found_joint, outcomes_N)
                if size(all_inter, 2) == outcomes_N
                    intercept_samples = all_inter
                end
            catch
            end
        end
        if all(intercept_samples .== 0.0)
            for k in 1:outcomes_N
                found_name = _find_parameter(p_names, inter_base, k, M.model_arch == "multivariate")
                if isempty(found_name)
                    found_name = _find_parameter(p_names, "$(inter_base)_$(k)", nothing, false)
                end
                if isempty(found_name) && !isempty(prefix)
                    found_name = _find_parameter(p_names, "intercept_$(k)", nothing, false)
                end
                if !isempty(found_name)
                    intercept_samples[:, k] = get_params_vector(chain, found_name, 1)[:, 1]
                else
                    @warn "Intercept parameter '$(inter_base)_$(k)' not found in the MCMC chain. Using zero for this effect."
                end
            end
        end
    end

    # --- Fixed Effects ---
    fixed_effects_samples = zeros(Float64, M.Xfixed_N, n_samples, outcomes_N)
    if M.Xfixed_N > 0
        param_name_base = if !isempty(prefix)
            (M.model_arch == "multivariate") ? "beta_flat_$(prefix)" : "beta_$(prefix)"
        else
            (M.model_arch == "multivariate") ? "beta_flat" : "beta"
        end
        found_base = _find_parameter(p_names, param_name_base, nothing,
            M.model_arch == "multivariate")
        if !isempty(found_base)
            if M.model_arch == "multivariate"
                all_fixed_beta_flat = get_params_matrix(chain, found_base, M.Xfixed_N * outcomes_N)
                reshaped_fixed_beta = reshape(all_fixed_beta_flat, n_samples, M.Xfixed_N,
                    outcomes_N)
                for k in 1:outcomes_N
                    fixed_effects_samples[:, :, k] = permutedims(reshaped_fixed_beta[:, :, k],
                        (2, 1))
                end
            else # Univariate
                fixed_beta = get_params_matrix(chain, found_base, M.Xfixed_N)
                fixed_effects_samples[:, :, 1] = permutedims(fixed_beta, (2, 1))
            end
        else
            @warn "Fixed effects parameter '$param_name_base' not found in the MCMC chain. Using zero for this effect."
        end
    end

    # --- Component Effects ---
    component_realizations = Dict{Symbol, Any}()
    for spec in M.components
        spec_for_effects = !isempty(prefix) ? merge(spec, (key=Symbol(prefix, "_", spec.key),)) : spec
        effects_result = get_effects(spec.component_obj, chain, spec_for_effects, M, PS)
        component_realizations[spec.key] = effects_result
    end



    # --- Spatiotemporal Interaction Effects ---
    st_interaction_effects_samples = zeros(Float64, M.s_N * M.t_N, n_samples, outcomes_N)
    if get(M, :model_st, "none") != "none"
        param_name_base = any(p -> occursin("innovations_st_interaction", string(p)),
            p_names) ? "innovations_st_interaction" : "st_interaction_raw"
        sigma_name_base = any(p -> occursin("sigma_st_interaction", string(p)),
            p_names) ? "sigma_st_interaction" : "sigma_st_interaction"
        has_param = any(p -> occursin(param_name_base, string(p)), p_names)
        has_sigma = any(p -> occursin(sigma_name_base, string(p)), p_names)
        if has_param && has_sigma
            if M.model_arch == "multivariate"
                sigma_samples = get_params_matrix(chain, sigma_name_base, outcomes_N)
                raw_samples = get_params_matrix(chain, param_name_base, M.s_N * M.t_N * outcomes_N)
                raw_samples_reshaped = reshape(raw_samples, n_samples, M.s_N * M.t_N, outcomes_N)
                for k in 1:outcomes_N
                    s_spec = M.components[findfirst(s -> s.structure == :spatial, M.components)]
                    t_spec = M.components[findfirst(s -> s.structure == :temporal, M.components)]
                    s_chol_factor = s_spec.hyper.cholesky_factor
                    t_chol_factor = t_spec.hyper.cholesky_factor
                    for i in 1:n_samples
                        Z_matrix = reshape(raw_samples_reshaped[i, :, k], M.s_N, M.t_N)
                        tmp_spatial = s_chol_factor.U \ Z_matrix
                        st_field_unscaled = transpose(t_chol_factor.U \ transpose(tmp_spatial))
                        st_field = st_field_unscaled .* sigma_samples[i, k]
                        st_interaction_effects_samples[:, i, k] = vec(st_field)
                    end
                end
            else # Univariate
                sigma_samples = get_params_vector(chain, sigma_name_base, 1)[:, 1]
                raw_samples = get_params_matrix(chain, param_name_base, M.s_N * M.t_N)
                s_spec = M.components[findfirst(s -> s.structure == :spatial, M.components)]
                t_spec = M.components[findfirst(s -> s.structure == :temporal, M.components)]
                s_chol_factor = s_spec.hyper.cholesky_factor
                t_chol_factor = t_spec.hyper.cholesky_factor
                for i in 1:n_samples
                    Z_matrix = reshape(raw_samples[i, :], M.s_N, M.t_N)
                    tmp_spatial = s_chol_factor.U \ Z_matrix
                    st_field_unscaled = transpose(t_chol_factor.U \ transpose(tmp_spatial))
                    st_field = st_field_unscaled .* sigma_samples[i]
                    st_interaction_effects_samples[:, i, 1] = vec(st_field)
                end
            end
        else
            @warn "Spatiotemporal interaction parameters not found. Using zero for this effect."
        end
    end

    # --- Householder Reflection Effects ---
    householder_effects_samples = zeros(Float64, outcomes_N, outcomes_N, n_samples)
    if get(M, :spectral_orientation, false) && M.model_arch == "multivariate"
        param_name = any(p -> occursin("v_unscaled_reflection", string(p)),
            p_names) ? "v_unscaled_reflection" : "v_raw_reflection"
        if any(p -> occursin(param_name, string(p)), p_names)
            v_reflection_samples = get_params_matrix(chain, param_name, outcomes_N)
            for i in 1:n_samples
                v_reflection = v_reflection_samples[i, :] / (norm(v_reflection_samples[i,
                    :]) + 1e-9)
                householder_effects_samples[:, :, i] = I - 2.0 * v_reflection * v_reflection'
            end
        else
            @warn "Householder reflection parameter '$param_name' not found. Using identity matrix for reflection."
            for i in 1:n_samples
                householder_effects_samples[:, :, i] = Matrix(I, outcomes_N, outcomes_N)
            end
        end
    end

    res_tuple = (
        intercept=intercept_samples,
        fixed_effects=fixed_effects_samples,
        components=component_realizations,
        st_interaction=st_interaction_effects_samples,
        householder_reflection=householder_effects_samples
    )

    return res_tuple
end


function _reconstruct(
    arch::UnivariateArchitecture, mode::String, chain, M::NamedTuple, PS,
    alpha::Float64; prefix::String = ""
)
    # --- 1. Metadata and Dimension Discovery ---
    n_samples_val = _get_chain_n_samples(chain)
    N_tot_val = isnothing(PS) ? M.y_N : M.y_N + PS.y_N
    outcomes_N_val = 1 # For univariate, this is always 1.

    # --- 2. Latent Field Reconstruction ---
    registry = _discover_component_realizations(
        chain, M, PS, n_samples_val, outcomes_N_val, N_tot_val; prefix = prefix
    )

    # --- 3. Linear Predictor Assembly ---
    eta_post_3d = _modular_eta_assembly(registry, M, PS, n_samples_val, outcomes_N_val)
    
    # Extract the 2D matrix for the single outcome
    eta_post_2d = eta_post_3d[:, :, 1]

    # --- 4. Prediction and Log-Likelihood Calculation ---
    pred_results = _process_ll_and_predictions(
        eta_post_2d, chain, M, PS, outcomes_N_val, 1; prefix = prefix
    )

    # --- 5. Final Result Consolidation ---
    pstats = (
        effects=_summarize_effects_registry(registry, M, outcomes_N_val, alpha),
        predictions_denoised=summarize_predictions(pred_results.p_denoised; alpha=alpha),
        predictions_noisy=summarize_predictions(pred_results.p_noisy; alpha=alpha),
        raw_predictions_denoised=pred_results.p_denoised,
        raw_predictions_noisy=pred_results.p_noisy,
        log_likelihood=pred_results.log_lik,
        waic=_compute_waic(pred_results.log_lik),
        arch=arch
    )

    return pstats
end


function _reconstruct(
    arch::MultivariateArchitecture, mode::String, chain, M::NamedTuple, PS,
    alpha::Float64; prefix::String = ""
)
    # --- 1. Metadata and Dimension Discovery ---
    n_samples_val = _get_chain_n_samples(chain)
    N_tot_val = isnothing(PS) ? M.y_N : M.y_N + PS.y_N
    outcomes_N_val = M.outcomes_N

    # --- 2. Latent Field Reconstruction ---
    registry = _discover_component_realizations(
        chain, M, PS, n_samples_val, outcomes_N_val, N_tot_val; prefix = prefix
    )
  
    # --- 3. Linear Predictor Assembly ---
    eta_latent_post = _modular_eta_assembly(registry, M, PS, n_samples_val, outcomes_N_val)

    is_multinomial_model = get(M, :is_multinomial, false) || any(
        s -> string(get(s, :family, "")) in [
            "multinomial", "categorical", "dirichlet_multinomial", "dirichlet"
        ],
        M.likelihood_specs
    )

    if is_multinomial_model
        pred_res = _process_multinomial_predictions(
            eta_latent_post, chain, M, PS; prefix = prefix
        )
        K_cats = get(M, :K_categories, outcomes_N_val + 1)
        cat_labels = get(M, :category_labels, [Symbol("cat_$k") for k in 1:K_cats])

        p_denoised_summaries = [
            summarize_array(pred_res.p_denoised[:, k, :], alpha=alpha)
            for k in 1:K_cats
        ]
        
        n_noisy_cols = size(pred_res.p_noisy, 2)
        p_noisy_summaries = [
            summarize_array(pred_res.p_noisy[:, k, :], alpha=alpha)
            for k in 1:n_noisy_cols
        ]

        # MAP classification: dominant category label per observation
        dominant_indices = [
            argmax([p_denoised_summaries[k].mean[i] for k in 1:K_cats])
            for i in 1:N_tot_val
        ]
        dominant_labels = [cat_labels[idx] for idx in dominant_indices]

        pstats = (
            effects=_summarize_effects_registry(registry, M, outcomes_N_val, alpha),
            predictions_denoised=p_denoised_summaries,
            predictions_noisy=p_noisy_summaries,
            raw_predictions_denoised=pred_res.p_denoised,
            raw_predictions_noisy=pred_res.p_noisy,
            dominant_category=dominant_labels,
            category_labels=cat_labels,
            log_likelihood=pred_res.log_lik,
            waic=_compute_waic(pred_res.log_lik),
            arch=arch
        )
        return pstats
    end

    # --- 4. Correlation Structure ---
    # The generated model does NOT rotate the linear predictor: for the joint Gaussian the
    # correlation lives in the residual covariance (D R D), not in the mean. Reconstruction
    # must mirror that exactly. Rotating here as well would put the two in different
    # outcome bases -- the exact mismatch that M2a was opened to remove.
    #
    # The factor is still extracted, and passed to the likelihood construction below, which
    # is where it belongs.
    correlation_cholesky_name = !isempty(prefix) ? "L_corr_$(prefix)" : "correlation_cholesky"
    L_corr_samples = get_params_matrix(chain, correlation_cholesky_name, outcomes_N_val * outcomes_N_val)
    if isempty(L_corr_samples) && !isempty(prefix)
        L_corr_samples = get_params_matrix(chain, "correlation_cholesky", outcomes_N_val * outcomes_N_val)
    end
    eta_post = eta_latent_post

    # --- 5. Prediction and Log-Likelihood Calculation ---
    all_pred_results = [
        _process_ll_and_predictions(
            eta_post[:, :, k], chain, M, PS, outcomes_N_val, k; prefix = prefix
        )
        for k in 1:outcomes_N_val
    ]
    
    p_denoised_summaries = [summarize_array(res.p_denoised,
        alpha=alpha) for res in all_pred_results]
    raw_denoised = [res.p_denoised for res in all_pred_results]
    raw_noisy = [res.p_noisy for res in all_pred_results]
    all_log_lik = hcat([res.log_lik for res in all_pred_results]...)

    # --- 5b. Joint (multivariate-Gaussian) noisy draws ---
    # `_process_ll_and_predictions` builds one SCALAR likelihood per outcome, so for a
    # multivariate Gaussian it drew each outcome independently and threw away the residual
    # correlation the model fits. That path is skipped for such models, leaving `p_noisy`
    # uninitialised -- so the noisy summaries MUST be computed after this block, not
    # before, or `summarize_array` reads uninitialised memory and reports NaN.
    # `nothing` (i.e. the per-outcome draws and their summaries are kept) for every other
    # family, where rotating the mean -- not a covariance -- carries the correlation.
    joint = _joint_noisy_draws_if_applicable(
        eta_post, L_corr_samples, chain, M, outcomes_N_val, n_samples_val, prefix
    )
    if !isnothing(joint)
        joint_noisy, joint_log_lik = joint
        raw_noisy = [joint_noisy[:, :, k] for k in 1:outcomes_N_val]
        # The per-outcome log-likelihood was skipped for this model, so substitute the joint
        # one -- it is the JOINT logpdf on the training rows, not a sum of marginals, or
        # WAIC would be computed for a factorised model that is no longer the one fitted.
        # Gated on shape rather than on `isnothing(PS)`: the per-outcome path allocates
        # `log_lik` uninitialised and skips filling it, so an un-substituted joint model
        # reports an all-NaN WAIC.
        if !isempty(joint_log_lik) && size(joint_log_lik, 1) == size(all_log_lik, 1)
            all_log_lik = joint_log_lik
        end
    end
    p_noisy_summaries = [summarize_array(a, alpha=alpha) for a in raw_noisy]

    # --- 6. Final Result Consolidation ---
    pstats = (
        effects=_summarize_effects_registry(registry, M, outcomes_N_val, alpha),
        predictions_denoised=p_denoised_summaries,
        predictions_noisy=p_noisy_summaries,
        raw_predictions_denoised=raw_denoised,
        raw_predictions_noisy=raw_noisy,
        log_likelihood=all_log_lik,
        waic=_compute_waic(all_log_lik),
        arch=arch
    )

    return pstats
end


"""
    _assemble_hierarchical_eta(chain, M, PS, n_samples, outcomes_N, N_tot; prefix="")

Recursively computes linear predictor `eta` for a model and all nested sub-models,
propagating lower-tier linear predictors through intermediate cascading parents.
"""
function _assemble_hierarchical_eta(
    chain, M::NamedTuple, PS, n_samples::Int, outcomes_N::Int, N_tot::Int;
    prefix::String = ""
)
    registry = _discover_component_realizations(
        chain, M, PS, n_samples, outcomes_N, N_tot; prefix = prefix
    )
    eta = _modular_eta_assembly(registry, M, PS, n_samples, outcomes_N)

    if haskey(M, :nested_components) && !isempty(M.nested_components)
        for (key, sub_M) in M.nested_components
            sub_PS = if !isnothing(PS) && haskey(PS, :nested_prediction_sets)
                get(PS.nested_prediction_sets, key, nothing)
            else
                nothing
            end

            sub_outcomes_N = get(sub_M, :outcomes_N, 1)
            sub_N_tot = isnothing(sub_PS) ? sub_M.y_N : sub_M.y_N + sub_PS.y_N
            sub_prefix = isempty(prefix) ? string(key) : "$(prefix)_$(key)"

            # Recursively assemble child's eta (with any cascading children)
            eta_sub = _assemble_hierarchical_eta(
                chain, sub_M, sub_PS, n_samples, sub_outcomes_N, sub_N_tot;
                prefix = sub_prefix
            )

            coupling = get(sub_M, :coupling, get(sub_M, :interaction, :additive))
            is_tensor = coupling == :tensor && get(sub_M, :is_multistrata, false)
            is_moderated = coupling == :moderated
            n_strata = get(sub_M, :n_strata, 1)
            is_stratified = n_strata > 1
            has_moderator = hasproperty(sub_M, :moderator) &&
                !isnothing(sub_M.moderator)
            moderator_vec = has_moderator ? sub_M.moderator : nothing
            strata_idx = get(sub_M, :strata_indices, nothing)

            # Sample extraction for tensor, moderated, or standard coupling
            fac_samples = Dict{Symbol, Matrix{Float64}}()
            rho_0_mat = nothing
            rho_1_mat = nothing
            rho_samples = nothing

            if is_tensor
                for fac in sub_M.strata_factors
                    dim = sub_M.factor_dims[fac]
                    rho_fac_name = "rho_nested_$(sub_prefix)_$(fac)"
                    v = extract_param_matrix(chain, rho_fac_name; expected_dim = dim)
                    fac_samples[fac] = isempty(v) ? ones(Float64, n_samples, dim) : v
                end
            elseif is_moderated
                rho_0_name = "rho_nested_$(sub_prefix)_0"
                rho_1_name = "rho_nested_$(sub_prefix)_1"
                exp_d = is_stratified ? n_strata : 1
                v0 = extract_param_matrix(chain, rho_0_name; expected_dim = exp_d)
                v1 = extract_param_matrix(chain, rho_1_name; expected_dim = exp_d)
                rho_0_mat = isempty(v0) ? ones(Float64, n_samples, exp_d) : v0
                rho_1_mat = isempty(v1) ? zeros(Float64, n_samples, exp_d) : v1
            else
                rho_name = "rho_nested_$(sub_prefix)"
                rho_samples = if get(sub_M, :fixed_coupling, false)
                    if is_stratified
                        ones(Float64, n_samples, n_strata)
                    else
                        ones(Float64, n_samples)
                    end
                else
                    if is_stratified
                        v = extract_param_matrix(chain, rho_name; expected_dim = n_strata)
                        isempty(v) ? ones(Float64, n_samples, n_strata) : v
                    else
                        v = extract_param_matrix(chain, rho_name; expected_dim = 1)
                        isempty(v) ? ones(Float64, n_samples) : vec(v[:, 1])
                    end
                end
            end

            eta_sub_mapped = if haskey(sub_M, :mapping) && !isnothing(sub_M.mapping)
                eta_sub[sub_M.mapping, :, :]
            elseif size(eta_sub, 1) == N_tot
                eta_sub
            else
                @warn "Size mismatch between parent observations ($N_tot) and " *
                      "nested model '$(key)' observations ($(size(eta_sub, 1)))." *
                      " Cannot apply nested effect."
                continue
            end

            # Apply coupling with interaction modes
            for i in 1:size(eta_sub_mapped, 1)
                w = if is_tensor
                    w_t = ones(Float64, n_samples)
                    for fac in sub_M.strata_factors
                        idx_f = sub_M.factor_indices[fac][i]
                        w_t .*= fac_samples[fac][:, idx_f]
                    end
                    if has_moderator
                        w_t .*= moderator_vec[i]
                    end
                    w_t
                elseif is_moderated
                    s_idx = !isnothing(strata_idx) ? strata_idx[i] : 1
                    mod_v = !isnothing(moderator_vec) ? moderator_vec[i] : 1.0
                    rho_0_mat[:, s_idx] .+ rho_1_mat[:, s_idx] .* mod_v
                elseif is_stratified
                    s_idx = !isnothing(strata_idx) ? strata_idx[i] : 1
                    mod_w = !isnothing(moderator_vec) ? moderator_vec[i] : 1.0
                    rho_samples[:, s_idx] .* mod_w
                else
                    mod_w = !isnothing(moderator_vec) ? moderator_vec[i] : 1.0
                    rho_samples .* mod_w
                end

                w_reshaped = reshape(w, 1, n_samples, 1)
                sub_row = eta_sub_mapped[i:i, :, :]

                if coupling == :multiplicative
                    eta[i:i, :, :] .*= (1.0 .+ w_reshaped .* sub_row)
                elseif coupling == :interaction
                    eta[i:i, :, :] .+= w_reshaped .* (eta[i:i, :, :] .* sub_row)
                else # :additive, :tensor, :moderated
                    eta[i:i, :, :] .+= w_reshaped .* sub_row
                end
            end
        end
    end

    return eta
end

function _reconstruct(
    arch::MultifidelityArchitecture, mode::String, chain, M::NamedTuple, PS,
    alpha::Float64; prefix::String = ""
)
    n_samples = _get_chain_n_samples(chain)
    N_tot = isnothing(PS) ? M.y_N : M.y_N + PS.y_N
    outcomes_N = M.outcomes_N

    # 1. Reconstruct full hierarchical eta (including all nested cascading levels)
    eta_main = _assemble_hierarchical_eta(
        chain, M, PS, n_samples, outcomes_N, N_tot; prefix = prefix
    )

    # 2. Main model registry for effects summary
    main_registry = _discover_component_realizations(
        chain, M, PS, n_samples, outcomes_N, N_tot; prefix = prefix
    )

    # 3. Individual reconstruction summaries for sub-models
    nested_results = Dict{Symbol, Any}()
    if haskey(M, :nested_components)
        for (key, sub_M) in M.nested_components
            sub_PS = if !isnothing(PS) && haskey(PS, :nested_prediction_sets)
                get(PS.nested_prediction_sets, key, nothing)
            else
                nothing
            end

            sub_prefix = isempty(prefix) ? string(key) : "$(prefix)_$(key)"
            sub_arch_raw = get(sub_M, :model_arch, "univariate")
            sub_arch_type = if (haskey(sub_M, :nested_components) &&
                               !isempty(sub_M.nested_components)) ||
                               sub_arch_raw == "multifidelity"
                MultifidelityArchitecture()
            elseif sub_arch_raw == "multivariate"
                MultivariateArchitecture()
            else
                UnivariateArchitecture()
            end

            nested_results[key] = _reconstruct(
                sub_arch_type, mode, chain, sub_M, sub_PS, alpha;
                prefix = sub_prefix
            )
        end
    end

    # 4. Apply correlation and generate predictions for the final main model
    eta_final = _apply_multivariate_correlation(eta_main, chain, outcomes_N)
    
    if outcomes_N > 1
        all_pred_results = [
            _process_ll_and_predictions(eta_final[:,:,k], chain, M, PS, outcomes_N, k)
            for k in 1:outcomes_N
        ]
        p_denoised_summaries = [summarize_array(res.p_denoised,
            alpha=alpha) for res in all_pred_results]
        p_noisy_summaries = [summarize_array(res.p_noisy, alpha=alpha) for res in all_pred_results]
        raw_denoised = [res.p_denoised for res in all_pred_results]
        raw_noisy = [res.p_noisy for res in all_pred_results]
        all_log_lik = hcat([res.log_lik for res in all_pred_results]...)
    else
        pred_results = _process_ll_and_predictions(
            eta_final[:,:,1], chain, M, PS, 1, 1
        )
        p_denoised_summaries = summarize_array(pred_results.p_denoised, alpha=alpha)
        p_noisy_summaries = summarize_array(pred_results.p_noisy, alpha=alpha)
        raw_denoised = pred_results.p_denoised
        raw_noisy = pred_results.p_noisy
        all_log_lik = pred_results.log_lik
    end

    summarized_effects = _summarize_effects_registry(
        main_registry, M, outcomes_N, alpha
    )
    waic = _compute_waic(all_log_lik)

    return (
        predictions_denoised = p_denoised_summaries, 
        predictions_noisy = p_noisy_summaries,
        raw_predictions_denoised = raw_denoised,
        raw_predictions_noisy = raw_noisy,
        log_likelihood = all_log_lik, 
        waic = waic, 
        effects = summarized_effects, 
        nested_results = nested_results, 
        transfer_results = nested_results,
        arch = arch
    )
end





"""
    _modular_eta_assembly(registry, M, PS, n_samples, outcomes_N)

Assembles the full linear predictor (`eta`) from all discovered latent effects.

# Version
v1.0.0

# Arguments
- `registry`: A NamedTuple containing raw posterior samples for each model component.
- `M`: The model configuration NamedTuple (for training data).
- `PS`: The prediction set configuration NamedTuple (for out-of-sample data), or `nothing`.
- `n_samples`: The total number of posterior samples.
- `outcomes_N`: The number of outcome variables.

# Returns
- A 3D array `eta_latent` of size `[N_total x n_samples x outcomes_N]`.
"""
function _modular_eta_assembly(registry, M, PS, n_samples, outcomes_N)
    N_tot = isnothing(PS) ? M.y_N : M.y_N + PS.y_N
    eta_latent = zeros(Float64, N_tot, n_samples, outcomes_N)

    # --- 1. Add Intercept ---
    # registry.intercept is [n_samples x outcomes_N]. Reshape to [1 x n_samples x
    #   outcomes_N] for broadcasting.
    eta_latent .+= reshape(registry.intercept, 1, n_samples, outcomes_N)

    # --- 2. Add Fixed Effects ---
    if haskey(registry, :fixed_effects) && size(registry.fixed_effects, 1) > 0
        Xfixed_full = if isnothing(PS)
            M.Xfixed
        else
            vcat(M.Xfixed, get(PS, :Xfixed, zeros(PS.y_N, size(M.Xfixed, 2))))
        end

        for k in 1:outcomes_N
            # Effect = X * beta_k
            eta_latent[:, :, k] .+= Xfixed_full * registry.fixed_effects[:, :, k]
        end
    end

    # Pre-compute full index vectors for spatial, temporal, and seasonal structures.
    # These map observations to their corresponding component units (e.g., spatial unit ID).
    s_idx_full = haskey(M, :s_idx) ? Array(isnothing(PS) || !haskey(PS,
        :s_idx) ? M.s_idx : vcat(M.s_idx, PS.s_idx)) : ones(Int, N_tot)
    t_idx_full = haskey(M, :t_idx) ? Array(isnothing(PS) || !haskey(PS,
        :t_idx) ? M.t_idx : vcat(M.t_idx, PS.t_idx)) : ones(Int, N_tot)
    u_idx_full = haskey(M, :u_idx) ? Array(isnothing(PS) || !haskey(PS,
        :u_idx) ? M.u_idx : vcat(M.u_idx, PS.u_idx)) : ones(Int, N_tot)

    # Iterate through all model components and add their effects to the linear predictor.
    for spec in M.components
        key = spec.key
        if !haskey(registry.components, key)
            continue
        end
        
        effects = registry.components[key]
        # Determine which set of effects to use (structured, noisy, or just the default
        #   `structured`).
        effect_set = hasproperty(effects, :noisy) ? effects.noisy : effects.structured
        if isempty(effect_set)
            continue
        end # Skip if the effect set is empty.
        
        for k in 1:outcomes_N
            effect_to_add = effect_set[k]
            if isempty(effect_to_add)
                continue
            end

            if size(effect_to_add, 1) == N_tot
                # Effect matrix is already expanded to full observation dimension [N_tot x
                #   n_samples]
                eta_latent[:, :, k] .+= effect_to_add
            elseif spec.structure in [:spatial, :temporal]
                # Unit-level effect matrix [n_units x n_samples] mapped via index vector
                idx_vec = spec.structure == :spatial ? s_idx_full : t_idx_full
                eta_latent[:, :, k] .+= effect_to_add[idx_vec, :]
            elseif spec.structure == :seasonal
                if spec.component_obj isa Harmonic
                    eta_latent[:, :, k] .+= effect_to_add
                else
                    idx_vec = u_idx_full
                    eta_latent[:, :, k] .+= effect_to_add[idx_vec, :]
                end
            elseif spec.structure in [:smooth, :interact, :nonstationaryvariance, :svc]
                eta_latent[:, :, k] .+= effect_to_add
            elseif spec.structure == :mixed
                idx_full = effects.indices
                if effects.type == :simple
                    if effects.lhs == "1"
                        eta_latent[:, :, k] .+= (size(effect_to_add,
                            1) == N_tot ? effect_to_add : effect_to_add[idx_full, :])
                    else
                        cov_name = Symbol(effects.lhs)
                        cov_vec = isnothing(PS) ? M.data[!, cov_name] : vcat(M.data[!,
                            cov_name], PS.data[!, cov_name])
                        eff = (size(effect_to_add,
                            1) == N_tot ? effect_to_add : effect_to_add[idx_full, :])
                        eta_latent[:, :, k] .+= eff .* cov_vec
                    end
                elseif effects.type == :correlated
                    for (term_name, term_effects) in pairs(effects.effects)
                        t_eff = term_effects[k]
                        if term_name == :intercept
                            eta_latent[:, :, k] .+= (size(t_eff,
                                1) == N_tot ? t_eff : t_eff[idx_full, :])
                        else
                            cov_name = Symbol(replace(string(term_name), "slope_" => ""))
                            cov_vec = isnothing(PS) ? M.data[!, cov_name] : vcat(M.data[!,
                                cov_name], PS.data[!, cov_name])
                            eff = (size(t_eff, 1) == N_tot ? t_eff : t_eff[idx_full, :])
                            eta_latent[:, :, k] .+= eff .* cov_vec
                        end
                    end
                end
            else # Fallback
                if size(effect_to_add, 1) == N_tot
                    eta_latent[:, :, k] .+= effect_to_add
                else
                    @warn "Component '$(key)' (structure $(spec.structure)) has unexpected dimension ($(size(effect_to_add, 1))) for outcome $k. Expected $N_tot."
                end
            end
        end
    end

    # Add log_offsets at the very end.
    if haskey(M, :log_offsets) && !isnothing(M.log_offsets)
        M_offsets = M.log_offsets isa AbstractVector ? reshape(M.log_offsets, :,
            1) : Matrix(M.log_offsets)
        PS_offsets = if isnothing(PS)
            zeros(Float64, 0, outcomes_N)
        else
            if haskey(PS, :log_offsets) && !isnothing(PS.log_offsets) && size(PS.log_offsets,
                1) == PS.y_N
                PS.log_offsets isa AbstractVector ? reshape(PS.log_offsets, :,
                    1) : Matrix(PS.log_offsets)
            else
                zeros(Float64, PS.y_N, outcomes_N)
            end
        end
        offset_full = vcat(M_offsets, PS_offsets)
        for k in 1:outcomes_N
            if k <= size(offset_full, 2)
                eta_latent[:, :, k] .+= offset_full[:, k]
            end
        end
    end

    # Apply the Householder reflection LAST, mirroring the generated model, where
    # `eta = eta * H_reflection` is the final update before the likelihood (model.jl:4294,
    # pushed after offsets, fixed effects, component updates and the spatiotemporal
    # interaction). A rotation does not commute with the additive terms, so the position
    # matters: it must come after `log_offsets` above, not before.
    #
    # Previously the reflection was computed into the registry and never read, so every
    # reconstructed field, prediction, log-likelihood and WAIC was produced in a different
    # outcome basis than the one the model was actually fitted in.
    #
    # The guard must mirror the condition at line 563 that actually *fits* a reflection.
    # `householder_effects_samples` is declared as a zeros array at line 562, before that
    # branch, and the key is therefore always present in the registry -- gating on the key
    # alone multiplies eta by zero and silently zeroes every multivariate prediction.
    if outcomes_N > 1 && get(M, :spectral_orientation, false) &&
            get(M, :model_arch, "") == "multivariate" &&
            haskey(registry, :householder_reflection)
        H_samples = registry.householder_reflection
        if H_samples isa AbstractArray && ndims(H_samples) == 3 &&
                size(H_samples, 1) == outcomes_N && size(H_samples, 2) == outcomes_N

    for s in 1:n_samples
                s <= size(H_samples, 3) || break
                eta_latent[:, s, :] = eta_latent[:, s, :] * view(H_samples, :, :, s)
            end
        end
    end

    return eta_latent
end




"""
    _optional_param_matrix(chain, name, n_samples, outcomes_N, default)

Fetch `name` from `chain` as an `(n_samples x outcomes_N)` matrix, returning `default` if
the parameter is absent. Unlike `get_params_matrix`, which errors, this tolerates families
that do not fit the parameter at all -- e.g. `lik_extra_params` exists only for the
dispersion families, and `lik_phi_hurdle` only when a hurdle is specified.
"""
function _optional_param_matrix(chain, name, n_samples::Int, outcomes_N::Int, default)
    try
        m = get_params_matrix(chain, name, outcomes_N)
        (size(m, 1) >= n_samples && size(m, 2) == outcomes_N) || return default
        return m[1:n_samples, :]
    catch
        return default
    end
end

"""
    _dispersion_or_nothing(v)

The dispersion sentinel: a `NaN` means the model fitted no dispersion parameter, and
`TethysBayes_Likelihood` expects `nothing` there so each family applies its own documented default
(`alpha = 1.0`, `phi = 10.0`, `nu = 5.0`). Passing `NaN` through would reach the kernels as
`extra_params` and be treated as a real, invalid value.
"""
_dispersion_or_nothing(v::Real) = isnan(v) ? nothing : v
_dispersion_or_nothing(v) = v

"""
    _process_ll_and_predictions(eta_samples, chain, M, PS, outcomes_N, k)

Generates predictions and log-likelihood values from the posterior `eta` samples.

# Version
v1.0.0

# Arguments
- `eta_samples`: A matrix of posterior linear predictor samples `[N_obs, n_samples]`.
- `chain`: The MCMC chain object.
- `M`: The main model configuration.
- `PS`: The prediction set configuration, or `nothing`.
- `outcomes_N`: The number of outcomes.
- `k`: The index of the current outcome.

# Returns
- A `NamedTuple` containing `p_denoised`, `p_noisy`, and `log_lik` matrices.
"""
function _process_ll_and_predictions(
    eta_samples, chain, M, PS, outcomes_N, k; prefix::String = ""
)    N_tot, n_samples = size(eta_samples)
    N_train = M.y_N
    
    y_obs_k = Array(outcomes_N > 1 ? M.y_obs[:, k] : M.y_obs)
    
    lik_spec = M.likelihood_specs[k]
    family = string(get(lik_spec, :family, "gaussian"))
    use_zi = get(M, :use_zi, false)
    
    phi_zi_name = !isempty(prefix) ? "lik_phi_zi_$(prefix)" : "lik_phi_zi"
    phi_zi_samples = if use_zi
        v = get_params_vector(chain, phi_zi_name, 1)
        if isempty(v) && !isempty(prefix)
            v = get_params_vector(chain, "lik_phi_zi", 1)
        end
        isempty(v) ? zeros(n_samples) : v[:, 1]
    else
        zeros(n_samples)
    end
    
    trials_full = if haskey(M, :trials)
        if isnothing(PS)
            M.trials[:, k]
        elseif N_tot == PS.y_N
            get(PS, :trials, ones(Int, PS.y_N, outcomes_N))[:, k]
        else
            vcat(M.trials[:, k], get(PS, :trials, ones(Int, PS.y_N, outcomes_N))[:, k])
        end
    else
        ones(Int, N_tot)
    end

    # --- Hurdle and dispersion parameters (L2) ---
    # The model fits these (model.jl:5299 samples `lik_phi_hurdle`, model.jl:5317 samples
    # `lik_extra_params`, passed through at 5471/5483) but they were never threaded to the
    # likelihood below, so it silently fell back to `phi_hurdle = -Inf` and
    # `extra_params = nothing` -- i.e. hardcoded alpha=1.0 / phi=10.0 / nu=5.0. Every
    # pointwise log-likelihood and noisy draw was then for a different model than the one
    # that was fitted. A family that does not use them simply has no such parameter in the
    # chain, in which case the previous default is kept and is correct.
    #
    # Extracted before `p_denoised_samples`, which needs `phi_hurdle_samples`.
    phi_hurdle_name = !isempty(prefix) ? "lik_phi_hurdle_$(prefix)" : "lik_phi_hurdle"
    extra_params_name = !isempty(prefix) ? "lik_extra_params_$(prefix)" : "lik_extra_params"
    phi_hurdle_samples = _optional_param_matrix(
        chain, phi_hurdle_name, n_samples, outcomes_N, fill(-Inf, n_samples, outcomes_N)
    )
    extra_params_samples = _optional_param_matrix(
        chain, extra_params_name, n_samples, outcomes_N, fill(NaN, n_samples, outcomes_N)
    )

    p_denoised_samples = similar(eta_samples)
    for s in 1:n_samples
        p_denoised_samples[:, s] = _apply_link_and_lik(
            family, view(eta_samples, :, s), use_zi, phi_zi_samples[s], 1.0, trials_full;
            phi_hurdle=phi_hurdle_samples[s, k]
        )
    end

    p_noisy_samples = similar(eta_samples)
    # `PS` supplies a *prediction* dataset, so `eta_samples` is indexed at prediction
    # locations while `y_obs_k` (== M.y_obs) and N_train are the *training* quantities --
    # different index spaces. The pointwise log-likelihood is therefore not computable here
    # unless the caller passed the training data itself. Leave it as NaN rather than zeros:
    # zeros make `_compute_waic` return a plausible-looking 0.0 for a model that was never
    # evaluated, which is worse than reporting that it is unavailable.
    log_lik_samples = fill(NaN, min(N_train, N_tot), n_samples)

    y_sigma_name = !isempty(prefix) ? "y_sigma_$(prefix)" : "y_sigma"
    r_nb_name = !isempty(prefix) ? "r_nb_$(prefix)" : "r_nb"

    y_sigma_samples = if family in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"]
        if outcomes_N > 1
            m = get_params_matrix(chain, y_sigma_name, outcomes_N)
            if isempty(m) && !isempty(prefix)
                m = get_params_matrix(chain, "y_sigma", outcomes_N)
            end
            m
        else
            v = get_params_vector(chain, y_sigma_name, 1)
            if isempty(v) && !isempty(prefix)
                v = get_params_vector(chain, "y_sigma", 1)
            end
            v
        end
    else
        ones(Float64, n_samples, outcomes_N)
    end

    r_nb_samples = if family == "negbin"
        if outcomes_N > 1
            m = get_params_matrix(chain, r_nb_name, outcomes_N)
            if isempty(m) && !isempty(prefix)
                m = get_params_matrix(chain, "r_nb", outcomes_N)
            end
            m
        else
            v = get_params_vector(chain, r_nb_name, 1)
            if isempty(v) && !isempty(prefix)
                v = get_params_vector(chain, "r_nb", 1)
            end
            v
        end
    else
        ones(Float64, n_samples, outcomes_N)
    end

    # A joint multivariate Gaussian draws each observation from ONE `MvNormal`, which needs
    # the full K-vector of etas -- but this function is handed outcome `k` alone. Building a
    # scalar likelihood for an `mvnormal` family is an outright type error
    # (`Cannot convert Vector{Float64} to Float64`), so skip the scalar work below and let
    # the caller (`_reconstruct`) fill `p_noisy` and `log_lik` from the joint draw. The
    # denoised pass above is per-outcome and still correct, so it is not skipped.
    #
    # A marginal draw would look correct and silently discard the fitted correlation, which
    # is the defect this whole path exists to close.
    joint_model = _is_joint_gaussian_model(M, outcomes_N)

    for s in 1:n_samples
        phi_zi_s = phi_zi_samples isa AbstractMatrix ? phi_zi_samples[s, 1] : phi_zi_samples[s]
        y_sigma_s = y_sigma_samples[s, k]
        r_nb_s = r_nb_samples[s, k]
        
        # Explicitly loop to avoid broadcasting issues with keyword arguments.
        # `phi_zi` must carry the -Inf "inactive" sentinel when zero-inflation is off: the
        # likelihood treats any value > -Inf as an active ZI, and `TethysBayes_Likelihood` rejects
        # ZI together with a hurdle (likelihoods.jl:499-510). Passing 0.0 unconditionally
        # made every hurdle model fail validation.
        if joint_model
            continue
        end
        lik_obj_vec = [
            TethysBayes_Likelihood(family, eta_samples[i, s];
                phi_zi=(use_zi ? phi_zi_s : -Inf),
                phi_hurdle=phi_hurdle_samples[s, k], r_nb=r_nb_s,
                sigma_y=y_sigma_s, trial=trials_full[min(i, length(trials_full))],
                extra_params=_dispersion_or_nothing(extra_params_samples[s, k]))
            for i in 1:N_tot
        ]
        
        p_noisy_samples[:, s] = rand.(lik_obj_vec)

        if isnothing(PS) && N_train > 0
            lik_obj_train = view(lik_obj_vec, 1:N_train)
            log_lik_samples[:, s] = logpdf.(lik_obj_train, view(y_obs_k, 1:N_train))
        end
    end

    return (p_denoised = p_denoised_samples, p_noisy = p_noisy_samples, log_lik = log_lik_samples)
end


"""
    _is_joint_gaussian_model(M, outcomes_N)

`true` when the model is a plain multivariate Gaussian whose correlation lives in the
residual covariance `D R D` (rather than in a mean rotation), so predictions must be drawn
jointly across outcomes.

The guards mirror `_generate_multivariate_likelihood_block` in `model.jl`: the family
symbols, and `!is_multinomial` with a likelihood spec per outcome. A multinomial/Dirichlet
model has several outcomes but a SINGLE spec -- the whole composition is one draw -- so
iterating `outcomes_N` over `likelihood_specs` would index past the end.
"""
function _is_joint_gaussian_model(M, outcomes_N::Int)
    outcomes_N <= 1 && return false
    gaus_families = ("gaussian", "mvnormal", "multivariate_normal")
    haskey(M, :likelihood_specs) || return false
    nspec = length(M.likelihood_specs)
    get(M, :is_multinomial, false) && return false
    nspec >= outcomes_N || return false
    return all(string(get(M.likelihood_specs[k], :family, "")) in gaus_families
               for k in 1:outcomes_N)
end


"""
    _joint_noisy_draws_if_applicable(eta_post, L_corr_samples, chain, M, outcomes_N,
                                     n_samples, prefix)

Returns an `N_tot x n_samples x outcomes_N` array of joint draws, or `nothing` when the model
is not a plain multivariate Gaussian and the per-outcome draws must be kept.
"""
function _joint_noisy_draws_if_applicable(
    eta_post::AbstractArray, L_corr_samples::AbstractMatrix, chain, M,
    outcomes_N::Int, n_samples::Int, prefix::String
)
    _is_joint_gaussian_model(M, outcomes_N) || return nothing
    isempty(L_corr_samples) && return nothing

    y_sigma_name = !isempty(prefix) ? "y_sigma_$(prefix)" : "y_sigma"
    sig = get_params_matrix(chain, y_sigma_name, outcomes_N)
    if isempty(sig) && !isempty(prefix)
        sig = get_params_matrix(chain, "y_sigma", outcomes_N)
    end
    (size(sig, 1) >= n_samples && size(sig, 2) == outcomes_N) || return nothing
    size(L_corr_samples, 1) >= n_samples || return nothing

    return _joint_noisy_draws(
        eta_post, L_corr_samples, sig[1:n_samples, :], size(eta_post, 1), n_samples,
        outcomes_N, M, Int(M.y_N)
    )
end


"""
    _joint_noisy_draws(eta_post, L_corr_samples, y_sigma_samples, N_tot, n_samples, outcomes_N)

Joint draws of the noisy multivariate-Gaussian prediction, returned as an
`N_tot x n_samples x outcomes_N` array.

`eta_post` is `N_tot x n_samples x outcomes_N`. The generated model does **not** rotate the
mean: for the joint Gaussian the correlation lives in the residual covariance `D R D`, so
each observation must be drawn once from a single `MvNormal`, not once per outcome.

Drawing per outcome instead is what made predictions silently lose their correlation: the
marginals look right, which is exactly why the defect survived. The output is still
*stored* per outcome (so `summarize_array` and the public result shape are unchanged), but
the draws are correlated, so `raw_predictions_noisy` now carries the joint structure.

Returns `nothing` when the model is not a plain multivariate Gaussian, so the caller falls
back to the per-outcome path.
"""
function _joint_noisy_draws(
    eta_post::AbstractArray, L_corr_samples::AbstractMatrix,
    y_sigma_samples::AbstractMatrix, N_tot::Int, n_samples::Int, outcomes_N::Int,
    M, N_train::Int
)
    joint = Array{Float64}(undef, N_tot, n_samples, outcomes_N)
    # Joint log-likelihood on the training rows. Must be the JOINT logpdf, not the sum of
    # marginals, or WAIC would be computed for a factorised model that is no longer the one
    # being fitted. Left at zero by the per-outcome path, which is skipped for this model.
    log_lik = zeros(Float64, N_train, n_samples)
    for s in 1:n_samples
        L_s = reshape(L_corr_samples[s, :], outcomes_N, outcomes_N)
        sig_s = collect(y_sigma_samples[s, :])
        for i in 1:N_tot
            d = TethysBayes_Likelihood(
                :mvnormal, collect(view(eta_post, i, s, :));
                sigma_y=sig_s,
                extra_params=Dict(:correlation_cholesky => L_s)
            )
            if i <= N_train
                log_lik[i, s] = logpdf(d, collect(view(M.y_obs, i, :)))
            end
            draw = rand(d)
            for k in 1:outcomes_N
                joint[i, s, k] = draw[k]
            end
        end
    end
    return joint, log_lik
end



"""
    _process_multinomial_predictions(eta_samples, chain, M, PS)

Generates posterior predictions (proportions, simulated counts/classes) and pointwise
log-likelihood for multinomial, categorical, and Dirichlet-multinomial models.

# Mathematical Formulation
For \$K\$ categories with reference category 1 (\$k_{\\text{ref}} = 1\$):
\$\\boldsymbol{\\eta}_i = [0, \\eta_{i, 1}, \\dots, \\eta_{i, K-1}]^\\top\$
\$\\boldsymbol{\\pi}_i = \\operatorname{softmax}(\\boldsymbol{\\eta}_i) \\in \\Delta^{K-1}\$

# Returns
A `NamedTuple` `(p_denoised, p_noisy, log_lik)`.
"""
function _process_multinomial_predictions(
    eta_samples, chain, M, PS; prefix::String = ""
)
    n_samples = size(eta_samples, 2)
    N_train = M.y_N
    N_pred = isnothing(PS) ? 0 : PS.y_N
    N_tot = N_train + N_pred

    K_free = size(eta_samples, 3)
    K_tot = get(M, :K_categories, K_free + 1)
    fam = Symbol(get(M, :multinomial_family, :multinomial))

    y_obs_train = M.y_obs

    # 1. Denoised simplex probabilities [N_tot, K_tot, n_samples]
    p_denoised_samples = zeros(Float64, N_tot, K_tot, n_samples)
    for s in 1:n_samples 
        for i in 1:N_tot
            eta_row = if K_free < K_tot
                vcat(0.0, eta_samples[i, s, :])
            else
                collect(eta_samples[i, s, :])
            end
            p_denoised_samples[i, :, s] = NNlib.softmax(eta_row)
        end
    end

    # 2. Noisy predictions and pointwise log-likelihood
    log_lik_samples = zeros(Float64, N_train, n_samples)

    if fam == :categorical
        p_noisy_samples = zeros(Int, N_tot, 1, n_samples)
        for s in 1:n_samples
            for i in 1:N_tot
                probs = p_denoised_samples[i, :, s]
                dist = Categorical(probs)
                p_noisy_samples[i, 1, s] = rand(dist)
                if i <= N_train
                    y_val = y_obs_train isa AbstractVector ?
                        y_obs_train[i] : argmax(y_obs_train[i, :])
                    log_lik_samples[i, s] = logpdf(dist, Int(y_val))
                end
            end
        end

    elseif fam == :dirichlet
        p_noisy_samples = zeros(Float64, N_tot, K_tot, n_samples)
        phi_name = !isempty(prefix) ? "dirichlet_phi_$(prefix)" : "dirichlet_phi"
        phi_samples = try
            v = get_params_vector(chain, phi_name, 1)
            if isempty(v) && !isempty(prefix)
                v = get_params_vector(chain, "dirichlet_phi", 1)
            end
            isempty(v) ? fill(1.0, n_samples) : v[:, 1]
        catch
            fill(1.0, n_samples)
        end

        for s in 1:n_samples
            phi_s = isempty(phi_samples) ? 1.0 : max(phi_samples[s], 1e-4)
            for i in 1:N_tot
                alpha_vec = max.(phi_s .* p_denoised_samples[i, :, s], 1e-6)
                dist = Dirichlet(alpha_vec)
                p_noisy_samples[i, :, s] = rand(dist)
                if i <= N_train
                    log_lik_samples[i, s] = logpdf(dist, y_obs_train[i, :])
                end
            end
        end

    else # :multinomial or :dirichlet_multinomial
        p_noisy_samples = zeros(Int, N_tot, K_tot, n_samples)
        
        # Total trials per observation
        trials_train = if haskey(M, :trials)
            vec(M.trials isa AbstractMatrix ? M.trials[:, 1] : M.trials)
        else
            vec(sum(y_obs_train, dims=2))
        end
        trials_pred = if N_pred > 0
            if !isnothing(PS) && haskey(PS, :trials)
                vec(PS.trials)
            else
                fill(Int(trials_train[1]), N_pred)
            end
        else
            Int[]
        end
        trials_full = vcat(trials_train, trials_pred)

        phi_name = !isempty(prefix) ? "dirichlet_phi_$(prefix)" : "dirichlet_phi"
        phi_samples = if fam == :dirichlet_multinomial
            try
                v = get_params_vector(chain, phi_name, 1)
                if isempty(v) && !isempty(prefix)
                    v = get_params_vector(chain, "dirichlet_phi", 1)
                end
                isempty(v) ? fill(1.0, n_samples) : v[:, 1]
            catch
                fill(1.0, n_samples)
            end
        else
            Float64[]
        end

        for s in 1:n_samples
            phi_s = (fam == :dirichlet_multinomial && !isempty(phi_samples)) ?
                max(phi_samples[s], 1e-4) : 1.0

            for i in 1:N_tot
                n_tr = Int(trials_full[i])
                probs = p_denoised_samples[i, :, s]

                dist = if fam == :dirichlet_multinomial
                    alpha_vec = max.(phi_s .* probs, 1e-6)
                    DirichletMultinomial(n_tr, alpha_vec)
                else
                    Multinomial(n_tr, probs)
                end

                p_noisy_samples[i, :, s] = rand(dist)
                if i <= N_train
                    log_lik_samples[i, s] = logpdf(dist, y_obs_train[i, :])
                end
            end
        end
    end

    return (p_denoised = p_denoised_samples, p_noisy = p_noisy_samples,
            log_lik = log_lik_samples)
end

function _quantile_along_last_dim(A::AbstractArray, q::Real; sample_dim=ndims(A))
    other_dims = size(A)[1:end-1]
    out = Array{Float64}(undef, other_dims)
    
    for I in CartesianIndices(out) 
        slice_view = view(A, I.I..., :)
        out[I] = quantile(slice_view, q)
    end
    return out
end

"""
    summarize_array(samples::AbstractArray; alpha=0.05)

Computes mean, median, std, and credible interval bounds along the last dimension of `samples`.
"""
function summarize_array(samples::AbstractArray; alpha=0.05)
    if isempty(samples) || all(isnan, samples)
        return (mean = Float64[], median = Float64[], std = Float64[], lower = Float64[],
            upper = Float64[])
    end 

    sample_dim = ndims(samples)
    low_prob = alpha / 2.0
    high_prob = 1.0 - low_prob

    post_mean = dropdims(Statistics.mean(samples, dims=sample_dim), dims=sample_dim)
    post_median = dropdims(Statistics.median(samples, dims=sample_dim), dims=sample_dim)
    post_std = dropdims(Statistics.std(samples, dims=sample_dim), dims=sample_dim)
    
    low_bound = _quantile_along_last_dim(samples, low_prob; sample_dim=sample_dim)
    high_bound = _quantile_along_last_dim(samples, high_prob; sample_dim=sample_dim)

    to_vector(x) = x isa AbstractArray ? vec(collect(Float64, x)) : [Float64(x)]

    return (
        mean = to_vector(post_mean),
        median = to_vector(post_median),
        std = to_vector(post_std),
        lower = to_vector(low_bound),
        upper = to_vector(high_bound)
    )
end

"""
    _compute_waic(log_lik::AbstractMatrix)

Computes the Watanabe-Akaike (Widely Applicable) Information Criterion (WAIC) from pointwise
  log-likelihood evaluations.
"""
function _compute_waic(log_lik)
    if !(log_lik isa AbstractMatrix)
        @warn "log_lik passed to _compute_waic is not a matrix. Returning NaN."
        return NaN
    end
    if isempty(log_lik)
        return NaN
    end

    nobs, nsamples = size(log_lik)
    lppd = sum(LogExpFunctions.logsumexp(view(log_lik, i, :)) - log(nsamples) for i in 1:nobs)
    p_waic = nsamples > 1 ? sum(var(view(log_lik, i, :)) for i in 1:nobs) : 0.0
    
    return -2 * (lppd - p_waic)
end

function _apply_link_and_lik(family::String, eta::AbstractArray, use_zi::Bool, phi=0.0, r=1.0,
    trials=nothing; phi_hurdle = -Inf)
    local mu
    if family in ["poisson", "negbin", "gamma", "exponential", "inverse_gaussian", "pareto", "lognormal"]
        # Must use the SAME bound as `get_dist_ref` in likelihoods.jl, or the reported
        # denoised mean and the mean of its own simulated noise disagree in the tails.
        # See `ETA_CLAMP_BOUND`: these were 20 and 30, giving an e^10 discrepancy for
        # eta in (20, 30]. (L8)
        clamped_eta = clamp.(eta, -ETA_CLAMP_BOUND, ETA_CLAMP_BOUND)
        mu = exp.(clamped_eta)
    elseif family in ["bernoulli", "beta"]
        mu = LogExpFunctions.logistic.(eta)
    elseif family == "binomial"
        prob = LogExpFunctions.logistic.(eta)
        mu = !isnothing(trials) ? Float64.(trials) .* prob : prob
    else
        mu = copy(eta)
    end
    if use_zi
        mu = (1.0 .- phi) .* mu
    end
    # A hurdle places probability `phi_hurdle` on the hurdle value, so the population mean is
    # `(1 - phi_hurdle) * E[y | not hurdle]` -- the same combination the kernels use
    # (likelihoods.jl:675). Without this branch `p_denoised` reported the above-hurdle
    # conditional mean as though it were the population mean. The `-Inf` default means the
    # hurdle is inactive and must leave the mean untouched, mirroring how the kernels gate it.
    if phi_hurdle > -Inf
        mu = (1.0 .- phi_hurdle) .* mu
    end
    return mu
end

"""
    post_stratification_weights(M, strata_ids, samples_denoised, totals;
        normalize=true)

**Model-based post-stratification weights**, one column per posterior draw.

Returns a matrix \`[n_obs x n_samples]\` such that, within every stratum \$h\$ and every
draw \$s\$,

sum over i in h of  w[i,s] * yhat[i,s]  ==  T[h]

where \$T_h\$ is the **externally known** total for that stratum and
\$\\hat{y}_{i,s}\$ is the model's posterior prediction. That identity is the whole
definition of post-stratification: the weight rescales the model's stratum total onto the
known external total, so a weighted re-analysis reproduces the external control totals.

## What this replaces, and why

The previous implementation computed \`w_i = stratum_mean / pred_i\` — the mean prediction
across a stratum divided by each unit's own prediction. That is **not**
post-stratification and cannot be made into it:

- it references **no external total at all**, so there is nothing to calibrate against;
- the weights are not constant within a stratum, so they cannot shift a stratum total;
- as a reweighting they satisfy no useful property, and for observations near zero the
  ratio is dominated by the guard constant in the denominator.

It is retained in name only; the function now requires the external totals, because
post-stratification without a target is not post-stratification. **Callers that do not
supply \`totals\` get \`nothing\`, not a number** — previously they got a number that looked
authoritative and was meaningless.

## Arguments
- `M`: model configuration, used only for its spatial index when `strata_ids` is omitted.
- `strata_ids`: stratum label per observation. Defaults to \`M.s_idx\`.
- `samples_denoised`: \`[n_obs x n_samples]\$ posterior predictions.
- `totals`: the external stratum totals, as a \`Dict\`/\`NamedTuple\` keyed by stratum, or a
  vector ordered to match \`unique(strata_ids)\`. \`nothing\` disables post-stratification.
- `normalize`: divide by the mean weight so the weights average 1, making them read as
  "relative to an unweighted analysis".

  ⚠️ This rescales the weighted totals by a single common factor, so the identity above
  becomes `sum_{i in h} w[i,s]*yhat[i,s] == T[h] / c` for a common `c`. **Use
  `normalize=false` when you need exact reconciliation to the external totals** — which is
  the point of post-stratification. `normalize=true` is the right choice when the weights
  are used for a *relative* reweighting (a weighted mean, a weighted fit, or comparing a
  weighted analysis against an unweighted one), where a common factor cancels.

## Returns
\`[n_obs x n_samples]\` of weights, or \`nothing\` when no external totals are available.

A draw whose stratum total is zero or non-finite gets \`NaN\` rather than a fabricated
weight, because a silently-chosen substitute for a degenerate total would bias exactly the
reweighted quantity it is meant to correct.
"""
function post_stratification_weights(
    M,
    strata_ids,
    samples_denoised,
    totals;
    normalize::Bool = true
)
    ids = isnothing(strata_ids) ? get(M, :s_idx, nothing) : strata_ids
    if isnothing(ids)
        @warn "Post-stratification needs a stratum label per observation " *
              "(`strata_ids`, or a spatial index `:s_idx` in the model configuration). " *
              "Returning `nothing`." maxlog = 1
        return nothing
    end

    # No external totals => no post-stratification. Do NOT fall back to a ratio.
    if isnothing(totals)
        @info "No external stratum totals supplied, so no post-stratification weights " *
              "were computed. Pass `totals` (stratum => known total) to reweight " *
              "observations. This is not an error; weights are simply undefined here."
        return nothing
    end

    n_obs, n_samples = size(samples_denoised)
    length(ids) == n_obs || begin
        @error "Dimension mismatch: `samples_denoised` has $n_obs rows but there are " *
               "$(length(ids)) stratum labels. Returning `nothing`." maxlog = 1
        return nothing
    end

    unique_strata = unique(ids)
    T = resolve_stratum_totals(totals, unique_strata, M, ids)

    # Which external total applies to each observation?
    total_per_obs = similar(unique_strata, Float64, n_obs)
    for (k, h) in enumerate(unique_strata)
        idx = findall(==(h), ids)
        total_per_obs[idx] .= T[k]
    end

    weights = Matrix{Float64}(undef, n_obs, n_samples)
    degenerate = 0
    for s in 1:n_samples
        ŷ = view(samples_denoised, :, s)
        for (k, h) in enumerate(unique_strata)
            idx = findall(==(h), ids)
            ŷ_h = sum(ŷ[idx])
            if !isfinite(ŷ_h) || abs(ŷ_h) <= _POST_STRAT_DEGENERATE_TOL
                # Undefined: the model predicts no total for this stratum, so there is
                # nothing to rescale. NaN rather than a fabricated weight.
                weights[idx, s] .= NaN
                degenerate += 1
            else
                weights[idx, s] .= T[k] / ŷ_h
            end
        end
    end

    degenerate > 0 && @warn "Post-stratification: $degenerate of $(n_obs * n_samples) " *
        "stratum-draw combinations have a zero or non-finite predicted total, so their " *
        "weight is undefined and set to NaN. Check that the strata are populated in the " *
        "region of interest." maxlog = 1

    if normalize
        for s in 1:n_samples
            col = view(weights, :, s)
            m = mean(col)
            # Only rescale if every weight is usable; a column containing NaN stays NaN.
            if isfinite(m) && m != 0
                col ./= m
            end
        end
    end

    return weights
end

# A predicted stratum total below this is treated as degenerate. Relative rather than
# absolute, so it means the same thing for a rate model and a count model.
const _POST_STRAT_DEGENERATE_TOL = 1e-12

"""
    resolve_stratum_totals(totals, unique_strata, M, ids)::Vector{Float64}

Normalise the several accepted forms of `totals` into one external total per stratum, in
the order of `unique_strata`, and check that every stratum is covered.

A missing stratum is an error rather than a default: silently post-stratifying some strata
and not others produces a reweighted total that matches no control at all.
"""
function resolve_stratum_totals(totals, unique_strata, M, ids)
    if totals isa AbstractDict || totals isa NamedTuple
        pairs_ = collect(pairs(totals))
        T = Vector{Float64}(undef, length(unique_strata))
        seen = Dict{Any, Float64}()
        for (k, v) in pairs_
            seen[k] = Float64(v)
        end
        missing_strata = Any[]
        for (k, h) in enumerate(unique_strata)
            if haskey(seen, h)
                T[k] = seen[h]
            else
                push!(missing_strata, h)
            end
        end
        if !isempty(missing_strata)
            error("`totals` is missing external totals for $(length(missing_strata)) " *
                  "stratum/strata: $(first(missing_strata, 5)). Every stratum must have a " *
                  "known total, otherwise the reweighted total reconciles to nothing.")
        end
        return T
    elseif totals isa AbstractVector
        length(totals) == length(unique_strata) || error(
            "`totals` is a vector of length $(length(totals)) but there are " *
            "$(length(unique_strata)) strata. A vector must be ordered to match " *
            "`unique(strata_ids)`.")
        return Float64.(totals)
    else
        error("`totals` must be a Dict/NamedTuple keyed by stratum, a vector ordered to " *
              "match `unique(strata_ids)`, or `nothing`. Got $(typeof(totals)).")
    end
end


"""
    reconstruct(chain, M::NamedTuple; alpha::Float64=0.05, PS=nothing)
    reconstruct(chain, model::DynamicPPL.Model; alpha::Float64=0.05, PS=nothing)
    reconstruct(model::DynamicPPL.Model, chain; alpha::Float64=0.05, PS=nothing)

Reconstruct latent fields, compute denoised and noisy predictions, prediction intervals,
goodness-of-fit metrics, and nested transfer components from an MCMC chain and model
specification.

### Mathematical Formulation
For each posterior draw \$\\theta^{(s)}\$, the linear predictor \$\\eta^{(s)}\$ is assembled
from fixed effects and latent components \$\\eta_k^{(s)}\$. In nested or multi-fidelity
specifications, sub-model predictors \$\\eta_{m, \\text{sub}}^{(s)}\$ are coupled via transfer
weights \$\\rho_m^{(s)}\$ (with optional index mapping \$\\mathcal{M}\$):
\$\$ \\eta^{(s)} = \\eta_{\\text{main}}^{(s)} + \\sum_m \\rho_m^{(s)} \\eta_{m, \\text{sub}}^{(s)}[\\mathcal{M}] \$\$
Point predictions and intervals are derived by evaluating the inverse link function
\$\\mathbb{E}[y^{(s)}] = g^{-1}(\\eta^{(s)})\$ across samples \$s = 1, \\dots, S\$.

### Arguments
- `chain`: MCMC sample collection (e.g., Turing `Chains` or `VNChain`).
- `M`: Model configuration `NamedTuple` (from `TethysBayes_config` or `TethysBayes_core`) or `DynamicPPL.Model`.
- `alpha::Float64`: Significance level for \$(1 - \\alpha)\$ credible intervals (default `0.05`).
- `PS`: Optional prediction set `NamedTuple` for out-of-sample synthesis (default `nothing`).

### Returns
A `NamedTuple` containing:
- `predictions_denoised`: Summary statistics (`mean`, `median`, `std`, `ci_lower`, `ci_upper`)
  for expected values \$g^{-1}(\\eta)\$.
- `predictions_noisy`: Summary statistics incorporating observation dispersion/noise.
- `raw_predictions_denoised`: Array of raw Monte Carlo draws for denoised predictions.
- `raw_predictions_noisy`: Array of raw Monte Carlo draws for noisy predictions.
- `log_likelihood`: Observation-level log-likelihood evaluation matrix.
- `waic`: Widely Applicable Information Criterion diagnostic.
- `effects`: Summarized posterior effects for all registered components.
- `nested_results` / `transfer_results`: Reconstructed outputs for each nested sub-model.
"""
function reconstruct(chain, M::NamedTuple; alpha::Float64=0.05, PS=nothing)
    raw_arch = get(M, :model_arch, "univariate")
    arch_type = if raw_arch == "multivariate"
        MultivariateArchitecture()
    elseif raw_arch == "multifidelity" ||
           (haskey(M, :nested_components) && !isempty(M.nested_components))
        MultifidelityArchitecture()
    else
        UnivariateArchitecture()
    end
    mode = isnothing(PS) ? "model_results" : "prediction"
    return _reconstruct(arch_type, mode, chain, M, PS, alpha)
end

function reconstruct(chain, model::DynamicPPL.Model; alpha::Float64=0.05, PS=nothing)
    return reconstruct(chain, model.args.M; alpha=alpha, PS=PS)
end

function reconstruct(model::DynamicPPL.Model, chain; alpha::Float64=0.05, PS=nothing)
    return reconstruct(chain, model.args.M; alpha=alpha, PS=PS)
end


"""
    model_results_comprehensive(model::DynamicPPL.Model, chain; data=nothing, alpha=0.05,
      strata_info=nothing)

The primary analytical post-processing engine that reconstructs latent fields, computes
point predictions, prediction intervals, goodness-of-fit metrics, and MCMC convergence diagnostics.
Generates pure analytical data without executing graphical rendering.
"""
function _is_optimization_or_vi_result(chain)::Bool
    if chain isa DataFrame || chain isa AbstractDict
        return false
    end
    t_str = string(typeof(chain))
    if occursin("ModeResult", t_str) ||
       occursin("VIResult", t_str) ||
       occursin("VariationalPosterior", t_str)
        return true
    end
    if (hasproperty(chain, :params) && hasproperty(chain, :optim_result)) ||
       hasproperty(chain, :q)
        return true
    end
    return false
end

function _is_area_unit_object(obj)::Bool
    isnothing(obj) && return false
    if obj isa DataFrame || (isdefined(Main, :AbstractDataFrame) && obj isa AbstractDataFrame)
        return false
    end
    return hasproperty(obj, :centroids) || hasproperty(obj, :W) ||
           hasproperty(obj, :polygons) || hasproperty(obj, :n_units)
end

function model_results_comprehensive(model::DynamicPPL.Model, chain; data=nothing, alpha=0.05,
    strata_info=nothing, strata_totals=nothing, au=nothing, kwargs...)
    if _is_optimization_or_vi_result(chain)
        n_samples_req = get(kwargs, :n_samples, 100)
        chain = convert_to_chains(chain, model, n_samples_req)
    end
    n_samples = _get_chain_n_samples(chain)
    
    # --- 1. Metadata and Architecture Extraction ---
    M = model.args.M
    y_obs = M.y_obs
    raw_arch = get(M, :model_arch, "univariate")

    arch_type = if raw_arch == "multivariate"
        MultivariateArchitecture()
    elseif raw_arch == "multifidelity" ||
           (haskey(M, :nested_components) && !isempty(M.nested_components))
        MultifidelityArchitecture()
    else
        UnivariateArchitecture()
    end

    # --- 2. Core Reconstruction ---
    res = _reconstruct(arch_type, "model_results", chain, M, nothing, alpha)

    # --- 2.5 Post-Stratification Weight Calculation (if applicable) ---
    #
    # Weights are only defined when the caller supplies externally known stratum totals.
    # They are returned in `draws_obj.weights` for a downstream reweighted analysis; note
    # that this function does NOT itself reweight anything -- applying the weights is the
    # caller's step, and the weights satisfy
    #     sum_{i in h} w[i, s] * yhat[i, s] == T[h]
    # for every stratum h and draw s, which is what makes that reweighting reconcile to
    # the external totals.
    post_strat_weights = nothing

    local M_for_post_strat = M
    if !haskey(M, :strata_info) && !isnothing(strata_info)
        M_for_post_strat = merge(M, (strata_info=strata_info,))
    end

    is_multinomial_model = get(M, :is_multinomial, false) || any(
        s -> string(get(s, :family, "")) in [
            "multinomial", "categorical", "dirichlet_multinomial", "dirichlet"
        ],
        M.likelihood_specs
    )

    if !isnothing(strata_totals) && hasproperty(res, :raw_predictions_denoised)
        samples_denoised = if res.raw_predictions_denoised isa AbstractArray{<:Real, 3}
            res.raw_predictions_denoised[:, 1, :]
        elseif res.raw_predictions_denoised isa AbstractVector
            res.raw_predictions_denoised[1]
        else
            res.raw_predictions_denoised
        end
        post_strat_weights = post_stratification_weights(
            M_for_post_strat, get(M, :s_idx, nothing), samples_denoised, strata_totals)
    end

    # --- 3. Performance Metric Calculation ---
    pred_summary = res.predictions_denoised
    local rmse_val, r_pearson
    if is_multinomial_model
        accuracy = if hasproperty(res, :dominant_category)
            if y_obs isa AbstractVector
                cat_labels = get(M, :category_labels, string.(1:get(M, :K_categories, 2)))
                obs_labels = [string(cat_labels[y_obs[i]]) for i in 1:length(y_obs)]
                mean(res.dominant_category .== obs_labels)
            else
                dominant_obs = [argmax(y_obs[i, :]) for i in 1:size(y_obs, 1)]
                cat_labels = get(M, :category_labels, string.(1:size(y_obs, 2)))
                obs_labels = [cat_labels[idx] for idx in dominant_obs]
                mean(res.dominant_category .== obs_labels)
            end
        else
            NaN
        end
        rmse_val = [1.0 - accuracy]
        r_pearson = [accuracy]
    elseif arch_type isa MultivariateArchitecture
        rmse_val = Float64[]
        r_pearson = Float64[]
        for k in 1:M.outcomes_N
            y_obs_k = Array(y_obs[:, k]) # Ensure y_obs is on CPU
            y_pred_k = pred_summary[k].mean
            valid_idx_k = findall(x -> !isnan(x) && !isnothing(x), y_obs_k)
            if !isempty(valid_idx_k)
                obs_v = y_obs_k[valid_idx_k]
                pred_v = y_pred_k[valid_idx_k]
                push!(rmse_val, sqrt(mean((obs_v .- pred_v).^2)))
                try; push!(r_pearson, cor(obs_v, pred_v)); catch; push!(r_pearson, 0.0); end
            else
                push!(rmse_val, NaN); push!(r_pearson, NaN)
            end
        end
    else # Univariate
        y_pred = hasproperty(pred_summary, :mean) ? pred_summary.mean : []
        y_obs_cpu = Array(y_obs) # Ensure y_obs is on CPU
        valid_idx = findall(x -> !isnan(x) && !isnothing(x), y_obs_cpu)
        if !isempty(valid_idx)
            obs_v = y_obs_cpu[valid_idx]
            pred_v = y_pred[valid_idx]
            rmse_val = sqrt(mean((obs_v .- pred_v).^2))
            try; r_pearson = cor(obs_v, pred_v); catch; r_pearson = 0.0; end
        else
            rmse_val = NaN; r_pearson = NaN
        end
    end

    # --- 4. MCMC Diagnostics ---
    # --- 4. MCMC Diagnostics (Pooled across all chains) ---
    df_stats = _compute_direct_parameter_summary(chain, model; alpha=alpha)

    mean_rhat, min_ess, sampling_time = 1.0, 0.0, 0.0 
    if hasproperty(df_stats, :rhat)
        r_vals = filter(x -> !isnan(x) && x > 0, df_stats.rhat)
        mean_rhat = isempty(r_vals) ? 1.0 : mean(r_vals)
    end
    e_col = hasproperty(df_stats, :ess_bulk) ? :ess_bulk : (hasproperty(df_stats,
        :ess) ? :ess : nothing)
    if !isnothing(e_col)
        e_vals = filter(x -> !isnan(x) && x >= 0, df_stats[!, e_col])
        min_ess = isempty(e_vals) ? 0.0 : minimum(e_vals)
    end
    if hasproperty(chain, :info) && haskey(chain.info, :stop_time)
        sampling_time = (chain.info.stop_time - chain.info.start_time)
    end
    
    # --- 5. Final Assembly (Structured Analytical Results) ---
    y_obs_clean = if is_multinomial_model
        y_obs
    elseif arch_type isa MultivariateArchitecture
        [Array(y_obs[:, k]) for k in 1:M.outcomes_N]
    else
        Array(y_obs)
    end

    pred_obj = if is_multinomial_model
        K_cats = get(M, :K_categories, length(res.predictions_denoised))
        cat_labels = get(M, :category_labels, [Symbol("cat_$k") for k in 1:K_cats])
        probs_dict = Dict{Symbol, Any}()
        for (k, lbl) in enumerate(cat_labels)
            probs_dict[Symbol("p_$(lbl)")] = res.predictions_denoised[k]
        end
        (
            observed = y_obs_clean,
            probabilities = NamedTuple(probs_dict),
            predicted_category = (
                mean = res.dominant_category,
                mode = res.dominant_category,
                median = res.dominant_category
            ),
            denoised = res.predictions_denoised,
            noisy = res.predictions_noisy
        )
    else
        (
            observed = y_obs_clean,
            denoised = res.predictions_denoised,
            noisy = res.predictions_noisy
        )
    end

    draws_obj = (
        predictions_denoised = res.raw_predictions_denoised,
        predictions_noisy = res.raw_predictions_noisy,
        weights = post_strat_weights,
        log_likelihood = res.log_likelihood
    )

    waic_raw = get(res, :waic, 0.0)
    waic_val = waic_raw isa Real ? waic_raw : (hasproperty(waic_raw, :waic) ? waic_raw.waic : 0.0)

    return (
        metrics = (
            rmse = rmse_val, 
            r_pearson = r_pearson, 
            waic = waic_val, 
            rhat = mean_rhat, 
            ess = min_ess, 
            time = sampling_time
        ),
        parameters = df_stats,
        effects = res.effects,
        predictions = pred_obj,
        draws = draws_obj,
        nested_results = get(res, :nested_results, nothing),
        transfer_results = get(res, :transfer_results, nothing),
        arch = res.arch,
        M = M,
        data = get(M, :data, data),
        au = au,
        model = model,
        chain = chain
    )
end

"""
    model_results_comprehensive(model::DynamicPPL.Model, chain, arg3, arg4=nothing; kwargs...)

Positional argument convenience overload supporting `(model, chain, data, au)` and
`(model, chain, au)` call signatures.
"""
function model_results_comprehensive(
    model::DynamicPPL.Model,
    chain,
    arg3,
    arg4 = nothing;
    data = nothing,
    au = nothing,
    kwargs...
)
    resolved_au = !isnothing(au) ? au : (_is_area_unit_object(arg3) ? arg3 : arg4)
    resolved_data = !isnothing(data) ? data : (_is_area_unit_object(arg3) ? arg4 : arg3)

    return model_results_comprehensive(
        model, chain; data = resolved_data, au = resolved_au, kwargs...
    )
end

function _get_chain_n_chains(chain::Any)::Int
    if chain isa DataFrame || chain isa AbstractDict
        return 1
    end
    try
        if occursin("FlexiChain", string(typeof(chain))) || occursin("VNChain",
            string(typeof(chain)))
            return FlexiChains.nchains(chain)
        end
    catch
    end
    try
        if hasproperty(chain, :chains)
            return length(chain.chains)
        end
    catch
    end
    try
        if hasproperty(chain, :value) && ndims(chain.value) == 3
            return size(chain.value, 3)
        end
    catch
    end
    try
        if hasproperty(chain, :info) && haskey(chain.info, :n_chains)
            return Int(chain.info.n_chains)
        end
    catch
    end
    return 1
end

"""
    _compute_direct_parameter_summary(chain::Any, model=nothing; alpha=0.05)::DataFrame

Computes standard MCMC parameter summary statistics directly from extracted parameter arrays.
Guarantees a pooled, well-formed DataFrame with exactly one row per parameter containing:
`parameters`, `mean`, `std`, `median`, `lower_95`, `upper_95`, `ess`, `rhat`.
"""
function _compute_direct_parameter_summary(chain::Any, model=nothing; alpha=0.05)::DataFrame
    p_names = _get_clean_chain_param_names(chain)
    if isempty(p_names) && !isnothing(model) && hasproperty(model.args, :M)
        reg = build_param_registry(model.args.M)
        p_names = reg.names
    end

    n_chains_val = _get_chain_n_chains(chain)

    low_q = alpha / 2.0
    high_q = 1.0 - low_q

    rows = []
    for pname in p_names
        try
            samples_mat = extract_param_matrix(chain, pname)
            dim = size(samples_mat, 2)
            for d in 1:dim
                v = samples_mat[:, d]
                valid_v = filter(x -> !isnan(x) && !isinf(x), v)
                if isempty(valid_v)
                    continue
                end
                
                param_label = dim == 1 ? pname : "$(pname)[$d]"
                m_val = Statistics.mean(valid_v)
                s_val = length(valid_v) > 1 ? Statistics.std(valid_v) : 0.0
                med_val = Statistics.median(valid_v)
                q_low = Statistics.quantile(valid_v, low_q)
                q_high = Statistics.quantile(valid_v, high_q)
                
                # --- Gelman-Rubin Rhat and ESS ---
                n_total = length(valid_v)
                rhat_val = 1.0
                ess_val = Float64(n_total)

                if n_chains_val >= 2 && n_total % n_chains_val == 0&&
                    (n_total ÷ n_chains_val) >= 4
                    n_iter_per_chain = n_total ÷ n_chains_val
                    chain_means = Float64[]
                    chain_vars = Float64[]
                    for c in 1:n_chains_val
                        sub_v = valid_v[(c-1)*n_iter_per_chain + 1 : c*n_iter_per_chain]
                        push!(chain_means, Statistics.mean(sub_v))
                        push!(chain_vars, length(sub_v) > 1 ? Statistics.var(sub_v) : 0.0)
                    end
                    overall_mean = Statistics.mean(chain_means)
                    B = (Float64(n_iter_per_chain) / (n_chains_val - 1)) * sum((chain_means .- overall_mean).^2)
                    W = Statistics.mean(chain_vars)
                    if W > 1e-12
                        var_plus = ((n_iter_per_chain - 1.0) / n_iter_per_chain) * W + (1.0 / n_iter_per_chain) * B
                        rhat_val = clamp(sqrt(var_plus / W), 1.0, 10.0)
                    end
                elseif n_total >= 20 # Split-Rhat for single chain
                    half_n = n_total ÷ 2
                    sub1 = valid_v[1:half_n]
                    sub2 = valid_v[half_n+1 : 2*half_n]
                    m1, m2 = Statistics.mean(sub1), Statistics.mean(sub2)
                    v1, v2 = Statistics.var(sub1), Statistics.var(sub2)
                    B = (Float64(half_n) / 1.0) * ((m1 - (m1+m2)/2.0)^2 + (m2 - (m1+m2)/2.0)^2)
                    W = (v1 + v2) / 2.0
                    if W > 1e-12
                        var_plus = ((half_n - 1.0) / half_n) * W + (1.0 / half_n) * B
                        rhat_val = clamp(sqrt(var_plus / W), 1.0, 10.0)
                    end
                end

                # Effective sample size via autocorrelation
                if n_total >= 10 && s_val > 1e-12
                    try
                        rho_regime_1 = StatsBase.autocor(valid_v, [1])[1]
                        if !isnan(rho_regime_1) && rho_regime_1 < 0.99 && rho_regime_1 > -0.99
                            act = (1.0 + rho_regime_1) / (1.0 - rho_regime_1)
                            ess_val = clamp(n_total / act, 1.0, Float64(n_total))
                        end
                    catch
                    end
                end

                push!(rows, (
                    parameters = Symbol(param_label),
                    mean = round(m_val, digits=4),
                    std = round(s_val, digits=4),
                    median = round(med_val, digits=4),
                    lower_95 = round(q_low, digits=4),
                    upper_95 = round(q_high, digits=4),
                    rhat = round(rhat_val, digits=4),
                    ess = round(ess_val, digits=1)
                ))
            end
        catch
        end
    end

    if isempty(rows)
        return DataFrame(
            parameters = Symbol[],
            mean = Float64[],
            std = Float64[],
            median = Float64[],
            lower_95 = Float64[],
            upper_95 = Float64[],
            rhat = Float64[],
            ess = Float64[]
        )
    end

    return DataFrame(rows)
end



"""
    _generate_conditional_predictions(model_obj, chain, M, target_cov::Symbol; second_cov::Union{Symbol, Nothing}=nothing, n_points::Int=50, alpha::Float64=0.05)

Generates conditional predictions for covariates across the spatial and temporal domain.
"""
function _generate_conditional_predictions(
    model_obj,
    chain,
    M,
    target_cov::Symbol;
    second_cov::Union{Symbol, Nothing} = nothing,
    n_points::Int = 50,
    alpha::Float64 = 0.05
)
    # Conditional predictions require both the compiled model and posterior chain
    if isnothing(model_obj) || isnothing(chain) || isnothing(M)
        return nothing
    end

    n_samps = try
        _get_chain_n_samples(chain)
    catch
        50
    end


    # --- Standard TethysBayes Conditional Predictions ---
    if !hasproperty(M, :data) || isnothing(M.data) || !hasproperty(M.data, target_cov)
        return nothing
    end

    base_df = DataFrame(M.data[1:1, :])
    for col in names(M.data)
        col_sym = Symbol(col)
        if col_sym != target_cov && (isnothing(second_cov) || col_sym != second_cov)
            col_type = eltype(M.data[!, col_sym])
            if col_type <: Integer
                base_df[1, col_sym] = round(col_type, Statistics.median(M.data[!, col_sym]))
            elseif col_type <: AbstractFloat
                base_df[1, col_sym] = Statistics.mean(M.data[!, col_sym])
            elseif col_type <: Number
                base_df[1, col_sym] = round(col_type, Statistics.mean(M.data[!, col_sym]))
            else
                if hasmethod(levels, Tuple{typeof(M.data[!, col_sym])}) && !isempty(levels(M.data[!, col_sym]))
                    base_df[1, col_sym] = first(levels(M.data[!, col_sym]))
                else
                    base_df[1, col_sym] = first(M.data[!, col_sym])
                end
            end
        end
    end

    if isnothing(second_cov)
        if eltype(M.data[!, target_cov]) <: Number
            min_val, max_val = extrema(M.data[!, target_cov])
            cov_range = if eltype(M.data[!, target_cov]) <: Integer
                step_val = max(1, round(Int, (max_val - min_val) / max(1, n_points - 1)))
                collect(min_val:step_val:max_val)
            else
                collect(range(min_val, stop=max_val, length=n_points))
            end
            n_actual = length(cov_range)
            pred_df = vcat([deepcopy(base_df) for _ in 1:n_actual]...)
            pred_df[!, target_cov] = cov_range
        else
            unique_levels = unique(M.data[!, target_cov])
            n_levels = length(unique_levels)
            pred_df = vcat([deepcopy(base_df) for _ in 1:n_levels]...)
            pred_df[!, target_cov] = unique_levels
            cov_range = unique_levels
        end

        preds = predict(model_obj, chain, pred_df; n_samples = n_samps, alpha = alpha)
        return (preds.predictions_denoised, cov_range)
    else
        if !(eltype(M.data[!, target_cov]) <: Number && eltype(M.data[!, second_cov]) <: Number)
            @warn "2D conditional plots are currently only supported for two continuous covariates."
            return nothing
        end
        min_val1, max_val1 = extrema(M.data[!, target_cov])
        min_val2, max_val2 = extrema(M.data[!, second_cov])
        range1 = eltype(M.data[!, target_cov]) <: Integer ? collect(min_val1:max(1, round(Int, (max_val1-min_val1)/max(1, n_points-1))):max_val1) : collect(range(min_val1, stop=max_val1, length=n_points))
        range2 = eltype(M.data[!, second_cov]) <: Integer ? collect(min_val2:max(1, round(Int, (max_val2-min_val2)/max(1, n_points-1))):max_val2) : collect(range(min_val2, stop=max_val2, length=n_points))
        
        pred_df_rows = DataFrame()
        for val1 in range1
            for val2 in range2
                row = deepcopy(base_df)
                row[1, target_cov] = val1
                row[1, second_cov] = val2
                append!(pred_df_rows, row)
            end
        end
        preds = predict(model_obj, chain, pred_df_rows; n_samples = n_samps, alpha = alpha)
        return (preds.predictions_denoised, range1, range2)
    end
end


"""
    _prepare_prediction_inputs(M_train, new_data::DataFrame)::NamedTuple

Prepares the prediction set configuration `PS` from the training model specification `M_train`
and new input observations `new_data`. Automatically resolves and maps user-supplied spatial
and temporal variable names to 1-based internal indices.
"""
function _prepare_prediction_inputs(M_train, new_data::DataFrame)::NamedTuple
    # 1. Initialize the Prediction Set (PS) configuration
    PS_dict = Dict(pairs(M_train))
    PS_dict[:data] = new_data
    PS_dict[:y_obs] = zeros(nrow(new_data)) # Placeholder
    PS_dict[:y_N] = nrow(new_data)

    # 2. Re-create fixed effects design matrix for the new data
    if haskey(M_train, :formula)
        decomposed_formula = decompose_TethysBayes_formula(M_train.formula, M_train.data)
        
        fixed_effects_vars = String[]
        append!(fixed_effects_vars, decomposed_formula.fixed_effects)
        for (_, mod_data_nt) in decomposed_formula.modules
            if mod_data_nt.module_type == :fixed && haskey(mod_data_nt.args, :positional_args)
                append!(fixed_effects_vars, string.(mod_data_nt.args[:positional_args]))
            end
        end
        fixed_effects_vars = unique(fixed_effects_vars)
        
        if !isempty(fixed_effects_vars)
            rhs = "0 + " * join(fixed_effects_vars, " + ")
            Xfixed_pred, _ = create_fixed_design(
                rhs, 
                new_data, 
                get(M_train, :calling_module, @__MODULE__); 
                contrasts=get(M_train, :contrasts, Dict())
            )
            PS_dict[:Xfixed] = Matrix(Xfixed_pred)
            PS_dict[:Xfixed_N] = size(Xfixed_pred, 2)
            PS_dict[:Xfixed_names] = names(Xfixed_pred, 2)
        end
    end

    # 3. Update indices and offsets from new_data
    if haskey(M_train, :s_idx_var) && hasproperty(new_data, M_train.s_idx_var)
        raw_new_s = new_data[!, M_train.s_idx_var]
        if haskey(M_train, :s_values) && !isnothing(M_train.s_values)
            s_val_map = Dict(v => i for (i, v) in enumerate(M_train.s_values))
            PS_dict[:s_idx] = [get(s_val_map, v, 1) for v in raw_new_s]
        elseif eltype(raw_new_s) <: Integer
            PS_dict[:s_idx] = Int.(raw_new_s)
        else
            PS_dict[:s_idx] = raw_new_s
        end
    elseif hasproperty(new_data, :s_idx)
        PS_dict[:s_idx] = Int.(new_data[!, :s_idx])
    end 

    if haskey(M_train, :t_idx_var) && hasproperty(new_data, M_train.t_idx_var)
        raw_new_t = new_data[!, M_train.t_idx_var]
        if haskey(M_train, :t_values) && !isnothing(M_train.t_values)
            t_vals = M_train.t_values
            PS_dict[:t_idx] = [
                begin
                    fidx = findfirst(==(v), t_vals)
                    if !isnothing(fidx)
                        fidx
                    elseif v isa Real && eltype(t_vals) <: Real
                        argmin(abs.(t_vals .- v))
                    else
                        1
                    end
                end for v in raw_new_t
            ]
        elseif eltype(raw_new_t) <: Integer
            PS_dict[:t_idx] = Int.(raw_new_t)
        else
            PS_dict[:t_idx] = raw_new_t
        end
    elseif hasproperty(new_data, :t_idx)
        PS_dict[:t_idx] = Int.(new_data[!, :t_idx])
    end 

    if haskey(M_train, :u_idx_var) && hasproperty(new_data, M_train.u_idx_var)
        raw_new_u = new_data[!, M_train.u_idx_var]
        if haskey(M_train, :u_values) && !isnothing(M_train.u_values)
            u_val_map = Dict(v => i for (i, v) in enumerate(M_train.u_values))
            PS_dict[:u_idx] = [get(u_val_map, v, 1) for v in raw_new_u]
        elseif eltype(raw_new_u) <: Integer
            PS_dict[:u_idx] = Int.(raw_new_u)
        else
            PS_dict[:u_idx] = raw_new_u
        end
    elseif hasproperty(new_data, :u_idx)
        PS_dict[:u_idx] = Int.(new_data[!, :u_idx])
    end 

    if haskey(PS_dict, :s_idx) && haskey(PS_dict, :t_idx) && haskey(M_train, :s_N)
        PS_dict[:st_idx] = (PS_dict[:t_idx] .- 1) .* M_train.s_N .+ PS_dict[:s_idx]
    end 

    if haskey(M_train, :log_offsets) && !isnothing(M_train.log_offsets)
        offset_var = get(M_train, :log_offsets_var, nothing)
        if !isnothing(offset_var) && hasproperty(new_data, Symbol(offset_var))
            PS_dict[:log_offsets] = Matrix(reshape(Float64.(new_data[!, Symbol(offset_var)]),
                :, M_train.outcomes_N))
        elseif hasproperty(new_data, :log_offsets)
            PS_dict[:log_offsets] = Matrix(reshape(Float64.(new_data[!, :log_offsets]), :,
                M_train.outcomes_N))
        else
            PS_dict[:log_offsets] = zeros(Float64, nrow(new_data), M_train.outcomes_N)
        end
    end

    # 4. Re-create basis matrices for smoothers on the new data
    if haskey(M_train, :components)
        ps_basis_registry = Dict{Symbol, Any}()
        smooth_specs = filter(s -> s.structure == :smooth, M_train.components)
        
        for spec in smooth_specs
            key_sym = Symbol(spec.var)
            vars = get(spec.params, :positional_args, [])
            n_vars = length(vars)
            if haskey(M_train.basis_matrices, key_sym) && all(hasproperty(new_data,
                Symbol(v)) for v in vars)
                m_obj = spec.component_obj
                model_type_str = lowercase(string(typeof(m_obj)))
                nb = size(M_train.basis_matrices[key_sym], 2)
                local_kwargs = Dict(spec.params)

                B_matrix, _ = if n_vars == 1 
                    TethysBayes_smooth_basis_1D(model_type_str, new_data[!, Symbol(vars[1])], nb;
                        local_kwargs...)
                elseif n_vars > 1
                    coords_new = Matrix{Float64}(new_data[!, Symbol.(vars)])
                    B_matrix_raw = if n_vars == 2
                        TethysBayes_smooth_basis_2D(model_type_str, coords_new, nb; local_kwargs...)
                    elseif n_vars == 3
                        TethysBayes_smooth_basis_3D(model_type_str, coords_new, nb; local_kwargs...)
                    elseif n_vars == 4
                        TethysBayes_smooth_basis_4D(model_type_str, coords_new, nb; local_kwargs...)
                    else
                        error("Smoothers with more than 4 dimensions are not supported for prediction.")
                    end
                    (B_matrix_raw, size(B_matrix_raw, 2))
                end
                ps_basis_registry[key_sym] = B_matrix
            end
        end
        PS_dict[:basis_matrices] = ps_basis_registry
    end

    # 5. Create prediction sets for nested sub-models
    if haskey(M_train, :nested_components) && !isempty(M_train.nested_components)
        PS_dict[:nested_prediction_sets] = Dict{Symbol, Any}()
        for (key, sub_M) in M_train.nested_components
            sub_PS_dict = Dict(pairs(sub_M))
            sub_PS_dict[:data] = new_data
            sub_PS_dict[:y_obs] = zeros(nrow(new_data))
            sub_PS_dict[:y_N] = nrow(new_data)

            if haskey(sub_M, :formula)
                sub_decomposed = decompose_TethysBayes_formula(sub_M.formula, sub_M.data) 
                sub_fixed_effects_vars = String[]
                append!(sub_fixed_effects_vars, sub_decomposed.fixed_effects)
                for (_, mod_data_nt) in sub_decomposed.modules
                    if mod_data_nt.module_type == :fixed && haskey(mod_data_nt.args,
                        :positional_args)
                        append!(sub_fixed_effects_vars,
                            string.(mod_data_nt.args[:positional_args]))
                    end
                end
                sub_fixed_effects_vars = unique(sub_fixed_effects_vars)
                
                if !isempty(sub_fixed_effects_vars)
                    rhs = "0 + " * join(sub_fixed_effects_vars, " + ")
                    Xfixed_sub, _ = create_fixed_design(
                        rhs, 
                        new_data, 
                        get(sub_M, :calling_module, @__MODULE__); 
                        contrasts=get(sub_M, :contrasts, Dict())
                    )
                    sub_PS_dict[:Xfixed] = Matrix(Xfixed_sub)
                    sub_PS_dict[:Xfixed_N] = size(Xfixed_sub, 2)
                    sub_PS_dict[:Xfixed_names] = names(Xfixed_sub, 2)
                else
                    sub_PS_dict[:Xfixed] = zeros(nrow(new_data), 0)
                    sub_PS_dict[:Xfixed_N] = 0
                    sub_PS_dict[:Xfixed_names] = Symbol[]
                end
            end

            if haskey(sub_M, :components)
                sub_ps_basis_registry = _recreate_basis_matrices_for_prediction(sub_M, new_data)
                sub_PS_dict[:basis_matrices] = sub_ps_basis_registry
            end

            if haskey(sub_M, :likelihood_specs) && !isempty(sub_M.likelihood_specs) 
                sub_lik_params = sub_M.likelihood_specs[1]
                _resolve_obs_param!(sub_PS_dict, sub_lik_params, new_data, [:log_offsets],
                    :log_offsets)
                _resolve_obs_param!(sub_PS_dict, sub_lik_params, new_data, [:weights], :weights)
                _resolve_obs_param!(sub_PS_dict, sub_lik_params, new_data, [:trials], :trials)
            end
            _precompute_likelihood_params!(sub_PS_dict)

            PS_dict[:nested_prediction_sets][key] = NamedTuple(sub_PS_dict)
        end
    end

    # 6. Finalize PS
    return NamedTuple(PS_dict)
end

"""
    predict(model_obj::DynamicPPL.Model, chain, new_data::DataFrame; n_samples::Int=100,
      alpha=0.05)

The primary engine for projecting a fitted `TethysBayes` model onto a new dataset to
generate out-of-sample predictions.

# Version
v1.0.0

# Arguments
- `model_obj::DynamicPPL.Model`: The fitted Turing model object.
- `chain`: The `MCMCChains.Chains` object from the fitted model.
- `new_data::DataFrame`: A `DataFrame` with the same column names as the training data.
- `n_samples::Int`: The number of posterior samples to use for prediction.
- `alpha::Float64`: The significance level for credible intervals.

# Returns
- A `NamedTuple` containing the summarized `predictions_denoised` and
  `predictions_noisy`, the full posterior statistics object `pstats`, and the
  prediction set configuration `PS`.
"""
function predict(model_obj::DynamicPPL.Model, chain, new_data::DataFrame; n_samples::Int=100,
    alpha=0.05)
    M_train = model_obj.args.M
    n_samps = min(_get_chain_n_samples(chain), n_samples)

 
    # Prepare Prediction Set (PS) configuration
    PS = _prepare_prediction_inputs(M_train, new_data)

    raw_arch = get(M_train, :model_arch, "univariate")
    arch_type = if raw_arch == "multivariate"
        MultivariateArchitecture()
    elseif raw_arch == "multifidelity" ||
           (haskey(M_train, :nested_components) && !isempty(M_train.nested_components))
        MultifidelityArchitecture()
    else
        UnivariateArchitecture()
    end 

    res = _reconstruct(arch_type, "prediction", chain, M_train, PS, alpha)

    # 7. Slice the prediction part from the full summary and sample matrices
    N_train = M_train.y_N
    
    function slice_summary(summary, raw_samples)
        if summary isa AbstractVector # Multivariate case 
            return [(
                mean = s.mean[(N_train+1):end],
                median = s.median[(N_train+1):end],
                std = s.std[(N_train+1):end],
                lower = s.lower[(N_train+1):end],
                upper = s.upper[(N_train+1):end],
                samples = (raw_samples isa AbstractArray && ndims(raw_samples) == 3 &&
                    size(raw_samples, 1) >= N_train + 1) ?
                    raw_samples[(N_train+1):end, :, k] : nothing
            ) for (k, s) in enumerate(summary)]
        else # Univariate case
            samps = (raw_samples isa AbstractArray && ndims(raw_samples) >= 2 &&
                size(raw_samples, 1) >= N_train + 1) ?
                raw_samples[(N_train+1):end, :] : nothing
            return (
                mean = summary.mean[(N_train+1):end],
                median = summary.median[(N_train+1):end],
                std = summary.std[(N_train+1):end],
                lower = summary.lower[(N_train+1):end],
                upper = summary.upper[(N_train+1):end],
                samples = samps
            )
        end
    end

    raw_denoised = hasproperty(res, :raw_predictions_denoised) ? res.raw_predictions_denoised : nothing
    raw_noisy = hasproperty(res, :raw_predictions_noisy) ? res.raw_predictions_noisy : nothing

    return (
        predictions_denoised = slice_summary(res.predictions_denoised, raw_denoised),
        predictions_noisy = slice_summary(res.predictions_noisy, raw_noisy),
        pstats = res,
        PS = PS
    )
end



"""
    TethysBayes_cv_orchestrator(formula::String, data::DataFrame; ...)

An orchestration utility for performing cross-validation on `TethysBayes` models. It
supports several strategies designed to handle the dependent nature of
spatiotemporal data.

# Version
v1.0.0

# Arguments
- `formula::String`: The `TethysBayes` model formula.
- `data::DataFrame`: The full dataset.
- `method::Symbol`: The CV strategy to use. Default: `:kfold`.
- `cv_var::Symbol`: The column in `data` used for grouping/blocking. Default: `:s_idx`.
- `n_folds::Int`: The number of folds or blocks. Default: `5`.
- `sampler`: The Turing sampler used to fit the model in each fold. Default: `NUTS(500, 0.65)`.
- `n_samples::Int`: The number of posterior samples to draw in each fold. Default: `500`.
- `alpha::Float64`: The significance level for credible intervals in prediction. Default: `0.05`.
- `cv_space_vars::Vector{Symbol}`: Coordinate columns for `:spatial_block`. Default: `[:s_x,
  :s_y]`.
- `kwargs...`: Additional keyword arguments passed to the underlying `TethysBayes_core` call.

# Returns
- A `NamedTuple` containing:
  - `folds`: A vector of `NamedTuple`s with `rmse` and `r2` for each fold.
  - `mean_rmse`: The average RMSE across all folds.
  - `mean_r2`: The average R-squared across all folds.
  - `response_var`: The name of the response variable.
  - `method`: The CV method used.
  - `n_folds`: The number of folds executed.
""" 
function TethysBayes_cv_orchestrator(
    formula::String,
    data::DataFrame;
    method::Symbol = :kfold,
    cv_var::Symbol = :s_idx,
    n_folds::Int = 5,
    n_samples::Int = 500,
    sampler = NUTS(500, 0.65),
    alpha = 0.05,
    cv_space_vars::Union{Vector{Symbol}, Nothing} = nothing,
    kwargs...
)
    
    
    meta_discovery = decompose_TethysBayes_formula(formula, data)
    response_name = !isnothing(meta_discovery) && !isempty(meta_discovery.outcomes) ? Symbol(meta_discovery.outcomes[1][:var]) : (hasproperty(data, :recapture) ? :recapture : :y)
    
    folds_indices = Vector{Vector{Int}}()
    is_forward_chain = false

    if method == :lolo
        eff_cv_var = if hasproperty(data, cv_var)
            cv_var
        else
            _detect_spatial_unit_column(data; allow_nothing=true)
        end
        if isnothing(eff_cv_var) || !hasproperty(data, eff_cv_var)
            error(
                "LOLO cross-validation requires a spatial unit column in data " *
                "(e.g. $(join(STANDARD_SPATIAL_UNIT_CANDIDATES, ", ")))."
            )
        end
        unique_locs = unique(data[!, eff_cv_var])
        for loc in unique_locs
            push!(folds_indices, findall(x -> x == loc, data[!, eff_cv_var]))
        end
    elseif method == :spatial_block
        space_cols = if !isnothing(cv_space_vars)
            cv_space_vars
        else
            cx, cy = _detect_xy_columns(data)
            [cx, cy]
        end
        if !all(hasproperty(data, v) for v in space_cols)
            error("Spatial block cross-validation requires coordinate columns: $space_cols.")
        end
        coords = Matrix(data[!, space_cols])' 
        R = Clustering.kmeans(coords, n_folds; maxiter=200, display=:none)
        assignments = R.assignments
        for k in 1:n_folds
            fold_k_indices = findall(x -> x == k, assignments)
            if !isempty(fold_k_indices)
                push!(folds_indices, fold_k_indices)
            end
        end
    elseif method == :temporal_block
        eff_cv_var = if cv_var != :s_idx && hasproperty(data, cv_var)
            cv_var
        else
            _detect_time_column(data; allow_nothing=true)
        end
        if isnothing(eff_cv_var) || !hasproperty(data, eff_cv_var)
            error(
                "Temporal block cross-validation requires a temporal column in data " *
                "(e.g. $(join(STANDARD_TEMPORAL_CANDIDATES, ", ")))."
            )
        end
        unique_times = sort(unique(data[!, eff_cv_var]))
        fold_size = cld(length(unique_times), n_folds)
        for i in 1:n_folds
            start_idx = (i - 1) * fold_size + 1
            end_idx = min(i * fold_size, length(unique_times))
            if start_idx > length(unique_times)
                continue
            end
            time_block = unique_times[start_idx:end_idx]
            push!(folds_indices, findall(t -> t in time_block, data[!, eff_cv_var]))
        end
    elseif method == :temporal_forward_chain
        eff_cv_var = if cv_var != :s_idx && hasproperty(data, cv_var)
            cv_var
        else
            _detect_time_column(data; allow_nothing=true)
        end
        if isnothing(eff_cv_var) || !hasproperty(data, eff_cv_var)
            error(
                "Forward-chaining cross-validation requires a temporal column in data " *
                "(e.g. $(join(STANDARD_TEMPORAL_CANDIDATES, ", ")))."
            )
        end
        is_forward_chain = true
        unique_times = sort(unique(data[!, eff_cv_var]))
        test_times = unique_times[end-n_folds+1:end]
        for t in test_times
            push!(folds_indices, findall(x -> x == t, data[!, eff_cv_var]))
        end
    else # Default to k-fold
        n_obs = size(data, 1)
        row_indices = Random.randperm(n_obs)
        fold_size = cld(n_obs, n_folds)
        for i in 1:n_folds
            idx_start = (i - 1) * fold_size + 1
            idx_end = min(i * fold_size, n_obs)
            if idx_start > n_obs
                continue
            end
            push!(folds_indices, row_indices[idx_start:idx_end])
        end
    end

    fold_results = []
    n_actual_folds = length(folds_indices)

    for (f_idx, test_idx) in enumerate(folds_indices)
        test_data = data[test_idx, :]
        train_data = if is_forward_chain
            min_test_time = minimum(test_data[!, cv_var])
            train_idx = findall(t -> t < min_test_time, data[!, cv_var])
            data[train_idx, :]
        else
            train_mask = trues(size(data, 1))
            train_mask[test_idx] .= false
            data[train_mask, :]
        end

        if nrow(train_data) == 0
            @warn "Fold $f_idx created an empty training set. Skipping."
            continue
        end

        cv_kwargs = Dict{Symbol, Any}(pairs(kwargs))
        cv_kwargs[:verbose] = false
        
        model_train = TethysBayes_core(formula, train_data; cv_kwargs...)
        chain_train = Base.invokelatest(sample, model_train, sampler, n_samples; progress=false)
        fit_obj = model_train

        res_pred = predict(fit_obj, chain_train, test_data; n_samples=div(n_samples, 2), alpha=alpha)
        y_test_obs  = test_data[!, response_name]
        y_test_pred = res_pred.predictions_denoised.mean

        if length(y_test_obs) == length(y_test_pred)
            residuals = y_test_obs .- y_test_pred
            rmse = sqrt(Statistics.mean(residuals.^2))
            ss_res = sum(residuals.^2)
            ss_tot = sum((y_test_obs .- Statistics.mean(y_test_obs)).^2)
            r2 = 1.0 - (ss_res / (ss_tot + 1e-15))
            push!(fold_results, (fold=f_idx, rmse=rmse, r2=r2))
        else
            @warn "Fold $f_idx: Prediction length mismatch. Observed: $(length(y_test_obs)), Predicted: $(length(y_test_pred))"
        end
    end

    mean_rmse = Statistics.mean([r.rmse for r in fold_results])
    mean_r2 = Statistics.mean([r.r2 for r in fold_results])
    return (folds = fold_results, mean_rmse = mean_rmse, mean_r2 = mean_r2, response_var = response_name, method = method, n_folds = n_actual_folds)
end


"""
    TethysBayes_loo(model_obj::DynamicPPL.Model, chain; alpha=0.05)

A utility for performing Leave-One-Out Cross-Validation using Pareto Smoothed
Importance Sampling (PSIS-LOO) to assess a model's out-of-sample predictive accuracy.

# Version
v1.0.0

# Arguments
- `model_obj::DynamicPPL.Model`: The fitted Turing model object.
- `chain`: The `MCMCChains.Chains` object from the fitted model.
- `alpha::Float64`: The significance level for credible intervals (not directly
  used in LOO calculation but maintained for API consistency).

# Returns
- A `NamedTuple` containing:
  - `loo_obj`: The full results object from `PosteriorStats.loo`.
  - `metrics`: A `NamedTuple` with the key estimates (`elpd`, `p_loo`, `looic`).
  - `log_likelihood`: The original `[n_obs, n_samples]` log-likelihood matrix.
  - `pareto_k`: A vector of the Pareto-k diagnostic values for each observation.
""" 
function TethysBayes_loo(model_obj::DynamicPPL.Model, chain; alpha=0.05)
    M = model_obj.args.M
    raw_arch = get(M, :model_arch, "univariate")
    arch_type = if raw_arch == "univariate"
        UnivariateArchitecture()
    elseif raw_arch == "multivariate"
        MultivariateArchitecture()
    elseif raw_arch == "multifidelity"
        MultifidelityArchitecture()
    else
        UnivariateArchitecture()
    end

    res = _reconstruct(arch_type, "loo_recovery", chain, M, nothing, alpha)
    log_lik = res.log_likelihood

    if isempty(log_lik)
        @warn "Log-likelihood matrix is empty. Cannot compute LOO."
        return nothing
    end

    n_obs, n_samples = size(log_lik)
    loo_result = nothing
    try
        loo_result = loo(Matrix(log_lik'))
    catch e
        @error "TethysBayes Selection Error: PSIS-LOO calculation failed. Error: " * string(e)
        return nothing
    end

    println("\n--- TethysBayes Model Selection Report ---")
    println("Expected Log Pointwise Predictive Density (ELPD): ", round(loo_result.estimates[:elpd_loo, :estimate], digits=2))
    println("Effective Number of Parameters (p_loo):          ", round(loo_result.estimates[:p_loo, :estimate], digits=2))
    println("LOO Information Criterion:                       ", round(loo_result.estimates[:looic, :estimate], digits=2))

    pareto_k = loo_result.pointwise[:pareto_k]
    influential_count = count(x -> x > 0.7, pareto_k)
    if influential_count > 0
        @warn "TethysBayes: " * string(influential_count) * " influential observations detected (Pareto k > 0.7)."
    end

    return (
        loo_obj = loo_result,
        metrics = (
            elpd = loo_result.estimates[:elpd_loo, :estimate],
            p_loo = loo_result.estimates[:p_loo, :estimate],
            looic = loo_result.estimates[:looic, :estimate]
        ),
        log_likelihood = log_lik,
        pareto_k = pareto_k
    )
end


"""
    compare_models(loo_a_report, loo_b_report; model_names=["Model_A", "Model_B"])

A utility for formal model comparison between two fitted `TethysBayes` models. It uses
their PSIS-LOO results to compute the difference in Expected Log Pointwise
Predictive Density (ELPD) and provides a statistical basis for model selection.

# Arguments
- `loo_a_report`, `loo_b_report`: The output `NamedTuple` from `TethysBayes_loo` for each model.
- `model_names`: A vector of strings with names for the models being compared.

# Returns
- A `NamedTuple` containing the comparison table, ELPD difference, and the original LOO objects.
"""
function compare_models(loo_a_report, loo_b_report; model_names=["Model_A", "Model_B"])
    println("--- Starting TethysBayes Model Comparison ---")

    # 1. LOO Object Extraction
    loo_a = loo_a_report.loo_obj
    loo_b = loo_b_report.loo_obj

    # 2. Formal Selection Metric Calculation
    comparison_stats = nothing
    try
        comparison_stats = compare([loo_a, loo_b])
    catch e
        @error "TethysBayes Comparison Error: Selection suite failed. Error: " * string(e)
        return nothing
    end

    # 3. Parameter and Diagnostic Extraction
    p_loo_a = loo_a_report.metrics.p_loo
    p_loo_b = loo_b_report.metrics.p_loo
    elpd_a = loo_a_report.metrics.elpd
    elpd_b = loo_b_report.metrics.elpd

    # 4. Report Generation
    println("\n--- TethysBayes Model Selection Registry ---")
    println("Model A (", model_names[1], "): ELPD = ", round(elpd_a, digits=2), " | p_loo = ",
        round(p_loo_a, digits=2))
    println("Model B (", model_names[2], "): ELPD = ", round(elpd_b, digits=2), " | p_loo = ",
        round(p_loo_b, digits=2))
    diff_elpd = elpd_a - elpd_b
    println("\nELPD Delta (A - B): ", round(diff_elpd, digits=2))

    if abs(diff_elpd) > 4.0
        winning_model = diff_elpd > 0 ? model_names[1] : model_names[2]
        println("CONCLUSION: ", winning_model,
            " is statistically preferred based on predictive density.")
    else
        println("CONCLUSION: Competing models provide indistinguishable predictive density.")
    end

    # 5. Table Construction
    comparison_df = DataFrame(
        Metric = ["ELPD (LOO)", "Effective Parameters (p_loo)", "LOO-IC"],
        Model_A = [elpd_a, p_loo_a, loo_a_report.metrics.looic],
        Model_B = [elpd_b, p_loo_b, loo_b_report.metrics.looic]
    )
    comparison_df[!, :Delta] = comparison_df.Model_A .- comparison_df.Model_B
    display(comparison_df)

    return (
        comparison_table = comparison_df,
        elpd_diff = diff_elpd,
        loo_objects = (loo_a, loo_b)
    )
end

