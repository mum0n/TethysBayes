"""
    definitions.jl

Core type definitions, abstract hierarchies, registries, and interface function declarations
for Bayesian Spatio-Temporal Models (BSTM).

Version: v1.0.0
"""

abstract type Component end
abstract type ComponentOperator <: Component end
abstract type ComponentModel <: Component end

struct None <: ComponentModel end

const COMPONENT_CONSTRUCTORS = Dict{Symbol, Function}(
    :none => (p, params) -> None()
)

const MODEL_TO_STRUCTURE_MAP = Dict{Union{Symbol, DataType}, Symbol}(
    :none => :none
)

"""
    get_component_structure(component)::Symbol

Resolves the structural category (`:spatial`, `:temporal`, `:seasonal`, `:smooth`,
  `:spacetime`, `:mixed`, `:any`)
for a component model symbol, type, or instance.
"""
function get_component_structure(component)::Symbol
    if component isa Symbol
        return get(MODEL_TO_STRUCTURE_MAP, component, :any)
    elseif component isa DataType
        if haskey(MODEL_TO_STRUCTURE_MAP, component)
            return MODEL_TO_STRUCTURE_MAP[component]
        end
        sym = Symbol(lowercase(string(nameof(component))))
        return get(MODEL_TO_STRUCTURE_MAP, sym, :any)
    else
        T = typeof(component)
        if haskey(MODEL_TO_STRUCTURE_MAP, T)
            return MODEL_TO_STRUCTURE_MAP[T]
        end
        sym = Symbol(lowercase(string(nameof(T))))
        return get(MODEL_TO_STRUCTURE_MAP, sym, :any)
    end
end

"""
    is_param_shared(shared_spec, param_sym::Symbol)::Bool

Determines whether a model hyperparameter or latent component is shared across
outcomes in a multivariate model.

# Mathematical & Design Logic
In multivariate BSTM formulations with \$K\$ outcomes, parameters may either vary
independently per outcome (\$k \\in \\{1, \\dots, K\\}\$) or be shared across all
outcomes to enforce parsimony or joint structure.

The `shared_spec` argument supports fine-grained control:
- `true`: All parameters of the component are shared across outcomes.
- `false`: All parameters of the component vary per outcome.
- `:all`: Alias for `true`.
- `Symbol` (e.g. `:sigma`): Only the matching parameter symbol is shared.
- `AbstractVector` or `Tuple` of Symbols/Strings (e.g. `[:sigma, :range]`): Only the
  specified parameter names in the collection are shared.

# Arguments
- `shared_spec`: Sharing configuration (`Bool`, `Symbol`, or collection of `Symbol`s).
- `param_sym::Symbol`: The parameter name being checked (e.g., `:sigma`, `:rho`).

# Returns
- `Bool`: `true` if `param_sym` is shared across outcomes, `false` otherwise.
"""
function is_param_shared(shared_spec, param_sym::Symbol)::Bool
    if shared_spec isa Bool
        return shared_spec
    elseif shared_spec isa Symbol
        return shared_spec == :all || shared_spec == param_sym
    elseif shared_spec isa AbstractVector || shared_spec isa Tuple
        spec_syms = Symbol.(shared_spec)
        return :all in spec_syms || param_sym in spec_syms
    else
        return false
    end
end

"""
    _get_varname_symbol(vn)::Symbol

Extracts the base Symbol from a `DynamicPPL.VarName` or `Symbol` across DynamicPPL and
  AbstractPPL versions.
"""
function _get_varname_symbol(vn)::Symbol
    if vn isa Symbol
        return vn
    elseif hasfield(typeof(vn), :name)
        return vn.name isa Symbol ? vn.name : Symbol(vn.name)
    else
        try
            return DynamicPPL.getsym(vn)
        catch
            try
                return Symbol(first(Base.split(string(vn), '[')))
            catch
                return Symbol(vn)
            end
        end
    end
end

"""
    _model_float_type(vi)::Type

Extracts the active scalar floating-point number type from `DynamicPPL.VarInfo` (e.g.
  `Float64`, `ForwardDiff.Dual`, `ReverseDiff.TrackedReal`).
Ensures automatic differentiation type-stability when allocating intermediate arrays in
  generated Turing models.
"""
function _model_float_type(vi)::Type
    # 1. Check OnlyAccsVarInfo (vi.accs)
    if hasfield(typeof(vi), :accs)
        try
            accs = getfield(vi, :accs)
            if hasproperty(accs, :LogPrior) && hasproperty(accs.LogPrior, :val)
                val_type = typeof(accs.LogPrior.val)
                if val_type !== Union{} && val_type !== Any && val_type <: Number
                    return val_type
                end
            end
            for acc in accs
                if hasproperty(acc, :val)
                    val_type = typeof(acc.val)
                    if val_type !== Union{} && val_type !== Any && val_type <: Number
                        return val_type
                    end
                end
            end
        catch
        end
    end

    # 2. Check VarInfo / SimpleVarInfo / UntypedVarInfo (vi.values)
    if hasfield(typeof(vi), :values)
        try
            vals = getfield(vi, :values)
            if vals isa AbstractArray && length(vals) > 0
                et = eltype(vals)
                if et !== Union{} && et !== Any && et <: Number
                    return et
                end
            elseif vals isa NamedTuple && length(vals) > 0
                first_val = first(values(vals))
                if first_val isa AbstractArray && length(first_val) > 0
                    et = eltype(first_val)
                    if et !== Union{} && et !== Any && et <: Number
                        return et
                    end
                elseif first_val isa Number
                    val_type = typeof(first_val)
                    if val_type !== Union{} && val_type !== Any && val_type <: Number
                        return val_type
                    end
                end
            end
        catch
        end
    end

    # 3. Check TypedVarInfo (vi.metadata)
    if hasfield(typeof(vi), :metadata)
        try
            meta = getfield(vi, :metadata)
            if meta isa NamedTuple && length(meta) > 0
                first_meta = first(values(meta))
                if hasproperty(first_meta, :vals) && length(first_meta.vals) > 0
                    et = eltype(first_meta.vals)
                    if et !== Union{} && et !== Any && et <: Number
                        return et
                    end
                end
            end
        catch
        end
    end

    # 4. Check DynamicPPL.float_type
    try
        ft = DynamicPPL.float_type(vi)
        if ft !== Union{} && ft !== Any && ft isa Type && ft <: Number
            return ft
        end
    catch
    end

    # 5. Check eltype
    try
        et = eltype(vi)
        if et !== Union{} && et !== Any && et isa Type && et <: Number
            return et
        end
    catch
    end

    return Float64
end

abstract type AbstractModelArchitecture end
struct UnivariateArchitecture <: AbstractModelArchitecture end
struct MultivariateArchitecture <: AbstractModelArchitecture end
struct MultifidelityArchitecture <: AbstractModelArchitecture end
struct ExampleArchitecture <: AbstractModelArchitecture end
struct UnknownArchitecture <: AbstractModelArchitecture end


const BSTM_MODULE_KEYWORDS = Set([ 
    :intercept, :fixed, :mixed, :random, :nested, :transfer, :fidelity, :eigen,
    :dynamics, :pointprocess, :custom, :zscore, :log, :center, :scale, :sciml
])
  
const TRANSFORMATION_FUNCTIONS = Set([:zscore, :log, :center, :scale])


 
# Define simple structs for geometric primitives
struct Point2D
    x::Float64
    y::Float64
end

struct Point4D
    x::Float64
    y::Float64
    z::Float64
    t::Float64
end

struct Triangle
    v1::Int
    v2::Int
    v3::Int
end
 

const COMPONENT_TYPE_REGISTRY = Dict{Symbol, Type{<:ComponentModel}}(
    :none => None
)

  
const PC_PRIORS = Dict(
    "sigma" => Exponential(1.0),
    "rho" => Beta(1, 1),
    "rho_regime_1" => Normal(0, 0.5),
    "rho_regime_2" => Normal(0, 0.5),
    "length_scale" => InverseGamma(3, 3),
    "amplitude" => Normal(0, 1),
    "phase" => Beta(1, 1),
    "pca_sd" => Exponential(1.0), 
    "pdef_sd" => Exponential(1.0),
    "range" => InverseGamma(3,3),
    "velocity" => truncated(Normal(0.2, 0.2), 0.0, 0.95),
    "diffusion" => truncated(Normal(0.1, 0.2), 0.0, Inf),
    "gamma" => Normal(1.0, 1.0)
)

const INFORMATIVE_PRIORS = Dict(
    "sigma" => Exponential(0.5),
    "rho" => Beta(2, 2),
    "rho_regime_1" => Normal(0, 1.0),
    "rho_regime_2" => Normal(0, 1.0),
    "length_scale" => InverseGamma(5, 5),
    "amplitude" => Normal(0, 0.5),
    "phase" => Beta(2, 2),
    "pca_sd" => Exponential(0.5), 
    "pdef_sd" => Exponential(0.5),
    "range" => InverseGamma(5,5),
    "velocity" => truncated(Normal(0.2, 0.1), 0.0, 0.95),
    "diffusion" => truncated(Normal(0.1, 0.1), 0.0, Inf),
    "gamma" => Normal(1.0, 0.5),
    "precision_field_scale" => Exponential(0.5),
    "precision_field_mixing" => Beta(2, 2)
)

const UNINFORMATIVE_PRIORS = Dict(
    "sigma" => Normal(0, 1e6),
    "rho" => Uniform(-1, 1),
    "rho_regime_1" => Normal(0, 10),
    "rho_regime_2" => Normal(0, 10),
    "length_scale" => InverseGamma(0.01, 0.01),
    "amplitude" => Normal(0, 100),
    "phase" => Uniform(0, 1),
    "pca_sd" => Normal(0, 1e6), 
    "pdef_sd" => Normal(0, 1e6),
    "range" => InverseGamma(0.01, 0.01),
    "velocity" => Uniform(0.0, 0.95),
    "diffusion" => truncated(Normal(0.0, 10.0), 0.0, Inf),
    "gamma" => Normal(0.0, 10.0),
    "precision_field_scale" => Exponential(10.0),
    "precision_field_mixing" => Uniform(0, 1)
)


const AMBIGUOUS_MODELS = Set([
    :iid, # Can be spatial (with W), temporal (with time var), or smooth (generic var)
    :gp,  # Can be spatial (coords), temporal (time var), or smooth (generic var)
    :rff, # Same as gp
    :fft, # Can be spatial (coords) or temporal (time var)
])

 

"""
    COMPONENT_CONFIG_ARGS

A registry holding the default values for non-prior configuration arguments
for various `random()` model types. This allows the parameter summary to show
all applicable settings, even those not explicitly set by the user.
"""
const COMPONENT_CONFIG_ARGS = Dict(
    # Spline models
    :pspline => Dict(:nbins => 20, :degree => 3, :penalty_order => 2, :knot_method => :quantile),
    :bspline => Dict(:nbins => 10, :degree => 3, :knot_method => :quantile),
    :tps => Dict(:nbins => 20, :knot_method => :quantile),
    
    # Continuous kernel models
    :gp => Dict(:kernel => "se", :anisotropic => false),
    :kriging => Dict(:kernel => "se", :anisotropic => false),
    :rff => Dict(:n_features => 20, :kernel => "se", :anisotropic => false),
    :fitc => Dict(:n_inducing => 20, :kernel => "se", :anisotropic => false),
    :svgp => Dict(:n_inducing => 20, :kernel => "se", :anisotropic => false),
    :nystrom => Dict(:n_inducing => 20, :kernel => "se", :anisotropic => false),
    :warp => Dict(:n_features => 20, :kernel => "se", :anisotropic => false),
    :spde => Dict(:anisotropic => false),

    # Temporal models
    :harmonic => Dict(:nharmonics => 1, :period => 12.0),
    :cyclic => Dict(:period => 12),
    
    # Other models
    :dynamics => Dict(:model => "none"),
    :svar => Dict(),
    :tar => Dict(), 
    :pointprocess => Dict(:model => :lgcp, :inner_model => :icar, :grid_areas => "unit"),
    :localadaptive => Dict(:n_clusters => 5),
    :eigen => Dict(:n_factors => 1)
)



# interface definitions

"""
    get_precomputes(m::ComponentModel, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-independent pre-calculations for a component instance.

This method is dispatched on the `ComponentModel` instance and runs after the
component has been instantiated. It is responsible for:
1.  Building template matrices (e.g., `Q_template` for GMRFs).
2.  Computing spectral decompositions (`U`, `L` for AD-friendly sampling).
3.  Generating fixed basis functions or other static structures.

# Arguments
- `m`: The `ComponentModel` instance.
- `M`: The main model configuration `NamedTuple`.
- `mod_data`: A dictionary containing parsed module data.

# Returns
- A `NamedTuple` containing all precomputed items necessary for code generation
  and posterior reconstruction (e.g., `(Q_template=..., U=..., L=...)`).

# Assumptions
- All data-dependent setup (e.g., `s_N`, `t_N`, `W`) has already been performed
  by `get_datastructures!`.
"""
function get_precomputes end

"""
    get_priors(m::ComponentModel, spec::NamedTuple, arch::String, outcome_idx::Union{Int,
      Nothing}, M::NamedTuple)::String

Generates the Turing code string for the component's priors.

This method is dispatched on the `ComponentModel` instance and is responsible for
producing Julia/Turing code snippets that define:
1.  Priors for the component's hyperparameters (e.g., `sigma`, `rho`).
2.  Priors for the component's latent fields (e.g., `raw ~ MvNormal(...)`).

# Arguments
- `m`: The `ComponentModel` instance.
- `spec`: A `NamedTuple` containing the component's full specification (key, structure, etc.).
- `arch`: The model architecture (`"univariate"`, `"multivariate"`).
- `outcome_idx`: The index of the outcome for multivariate models, `nothing` otherwise.
- `M`: The main model configuration `NamedTuple`.

# Returns
- A `String` containing the generated Turing code for priors.
"""
function get_priors end

"""
    get_updates(m::ComponentModel, spec::NamedTuple, arch::String, outcome_idx::Union{Int,
      Nothing}, M::NamedTuple)::String

Generates the Turing code string for constructing the component's effect and adding it to
  the linear predictor.

This method is dispatched on the `ComponentModel` instance and is responsible for
producing Julia/Turing code snippets that define:
1.  The logic for transforming sampled latent variables into the component's effect.
2.  Adding this effect to the linear predictor (`eta`).

# Arguments
- `m`: The `ComponentModel` instance.
- `spec`: A `NamedTuple` containing the component's full specification.
- `arch`: The model architecture (`"univariate"`, `"multivariate"`).
- `outcome_idx`: The index of the outcome for multivariate models, `nothing` otherwise.
- `M`: The main model configuration `NamedTuple`.

# Returns
- A `String` containing the generated Turing code for the component's update logic.
"""
function get_updates end

"""
    get_effects(m::ComponentModel, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
      p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing},
      N_total::Int)::NamedTuple

Reconstructs the component's effect from the MCMC chain's posterior samples for diagnostics
  and visualization.

This method is dispatched on the `ComponentModel` instance and is responsible for:
1.  Extracting relevant parameter samples from the MCMC `chain`.
2.  Reconstructing the component's contribution to the linear predictor for each sample.
3.  Returning structured results (e.g., mean, lower, upper credible intervals) of the effect.

# Arguments
- `m`: The `ComponentModel` instance.
- `chain`: The MCMC chain object containing posterior samples.
- `M`: The main model configuration `NamedTuple`.
- `n_samples`: The total number of posterior samples.
- `outcomes_N`: The total number of outcomes in the model.
- `p_names`: A `NamedTuple` of parameter names for easy access.
- `spec`: A `NamedTuple` containing the component's full specification.
- `PS`: A `NamedTuple` containing prediction set data, or `nothing` for in-sample reconstruction.
- `N_total`: The total number of observations (training + prediction).

# Returns
- A `NamedTuple` (e.g., `(structured=..., noisy=...)`) containing the reconstructed effects.
"""
function get_effects end

# -----------------------------------------------------------------------------
# Interface contract
#
# Every registered component must implement all four of these. They were bare global
# `function ... end` declarations, so a component missing one only failed later, as a
# MethodError deep inside model construction, naming the internal call site rather than
# the component at fault. These helpers turn that into an immediate, component-attributed
# error.
# -----------------------------------------------------------------------------

"""
    COMPONENT_INTERFACE_FUNCTIONS

The four functions every component type must implement, in call order.
"""
const COMPONENT_INTERFACE_FUNCTIONS = (:get_precomputes, :get_priors, :get_updates, :get_effects)

"""
    missing_interface_methods(T::Type) -> Vector{Symbol}

The interface functions `T` does not implement.

This inspects each method's *first argument* type rather than calling `hasmethod`, which
cannot be used here: the interface takes five arguments, so a one-element
`Tuple{T}` never matches a five-argument method signature.
"""
function missing_interface_methods(@nospecialize(T::Type))
    return [f for f in COMPONENT_INTERFACE_FUNCTIONS if !_implements_for(T, getfield(bstm, f))]
end

function _implements_for(@nospecialize(T::Type), f::Function)
    for m in methods(f)
        params = Base.unwrap_unionall(m.sig).parameters
        length(params) >= 2 || continue
        first_arg = params[2]
        first_arg isa Type && T <: first_arg && return true
    end
    return false
end

"""
    validate_component_interface(T::Type) -> Type

Assert that `T` implements the whole component interface. Throws an error naming the
component and exactly which functions are missing.
"""
function validate_component_interface(@nospecialize(T::Type))
    missing = missing_interface_methods(T)
    if !isempty(missing)
        error(
            "Component `$(T)` does not implement the bstm component interface. " *
            "Missing: $(join(string.(missing), ", ")). " *
            "Every component must define $(join(string.(COMPONENT_INTERFACE_FUNCTIONS), ", ")).")
    end
    return T
end

"""
    validate_all_component_interfaces() -> Dict{Symbol, Vector{Symbol}}

Check every type in `COMPONENT_TYPE_REGISTRY`. Returns a map of component key to the
interface functions it is missing, and errors if any is incomplete. Called when a model is
assembled, so an incomplete component fails at model-build time with the component named,
rather than at sample time with a `MethodError`.
"""
function validate_all_component_interfaces()
    report = Dict{Symbol, Vector{Symbol}}()
    for (key, T) in COMPONENT_TYPE_REGISTRY
        missing = missing_interface_methods(T)
        isempty(missing) || (report[key] = missing)
    end
    if !isempty(report)
        details = join(("  `$(k)` -> $(join(string.(v), ", "))" for (k, v) in sort(collect(report), by = first)),
            "\n")
        error("Incomplete component interface(s):\n$(details)")
    end
    return report
end

# -----------------------------------------------------------------------------
# Coordinate & Variable Resolvers
# -----------------------------------------------------------------------------

# The spatial/temporal coordinate-time column-name vocabulary and its two
# resolvers (STANDARD_SPATIAL_COORDINATE_PAIRS, STANDARD_TEMPORAL_CANDIDATES,
# _detect_xy_columns, _detect_time_column) moved to the GeoPartitioning
# package (they were the only package-internal coupling the partitioning engine
# had). bstm.jl imports the same names from GeoPartitioning, which keeps every
# internal call site and the module export statements working unchanged. The
# remaining _detect_* resolvers below are bstm modelling vocabulary and stay.


"""
    _detect_response_column(df; target_var=nothing, exclude=Symbol[])::Symbol

Automatically identifies the primary numeric response variable column for modeling
or kriging interpolation.

# Process
1. If `target_var` is explicitly provided, validates and returns it.
2. If `:z`, `:y`, `:val`, or `:value` exists and is not excluded, selects it.
3. Otherwise, picks the first numeric column in `df` not contained in `exclude`.

# Inputs
- `df`: Tabular dataset.
- `target_var`: Optional explicit response column name.
- `exclude`: List of column symbols to exclude (e.g. spatial coordinates, IDs).

# Outputs
- `Symbol`: Resolved response column name.
"""
function _detect_response_column(df; target_var=nothing, exclude=Symbol[])::Symbol
    if !isnothing(target_var)
        sv = Symbol(target_var)
        if !hasproperty(df, sv)
            error(
                "Specified target variable :$sv not found in data: " *
                "$(propertynames(df))."
            )
        end
        return sv
    end

    # Standard candidate response names
    for cand in [:z, :y, :val, :value, :response, :obs]
        if hasproperty(df, cand) && !(cand in exclude)
            return cand
        end
    end

    # First numeric non-excluded column
    for col in propertynames(df)
        sym = Symbol(col)
        if !(sym in exclude)
            col_data = df[!, sym]
            if eltype(col_data) <: Number
                return sym
            end
        end
    end

    error(
        "Could not automatically detect response column in dataset. " *
        "Available columns: $(propertynames(df)). " *
        "Please specify target variable explicitly (e.g., var=:z)."
    )
end

"""
Standard candidate column names for discrete spatial units (regions, areas, districts, etc.).
"""
const STANDARD_SPATIAL_UNIT_CANDIDATES = [
    :s_idx,
    :region,
    :district,
    :county,
    :area,
    :area_id,
    :zone,
    :unit,
    :unit_id,
    :spatial_unit,
    :au,
    :au_idx,
    :polygon_id,
    :id,
    :site,
    :location,
    :station
]

"""
    _detect_spatial_unit_column(df; s_idx_var=nothing, allow_nothing=false)::Union{Symbol, Nothing}

Automatically identifies or validates the spatial unit or areal index column in a dataset.

# Process
1. If `s_idx_var` is explicitly specified, validates that it exists in `df` and returns it.
2. Otherwise, matches against `STANDARD_SPATIAL_UNIT_CANDIDATES`.
3. If not found and `allow_nothing=false`, raises a descriptive `error`.

# Inputs
- `df`: Tabular dataset.
- `s_idx_var`: Optional explicit column name.
- `allow_nothing`: If true, returns `nothing` when not detected instead of error.

# Outputs
- `Union{Symbol, Nothing}`: Resolved column symbol or nothing.
"""
function _detect_spatial_unit_column(
    df; s_idx_var=nothing, allow_nothing::Bool=false
)::Union{Symbol, Nothing}
    if !isnothing(s_idx_var)
        sym = Symbol(s_idx_var)
        if !hasproperty(df, sym)
            allow_nothing && return nothing
            error(
                "Specified spatial unit column s_idx_var=:$sym not found in data: " *
                "$(propertynames(df))."
            )
        end
        return sym
    end

    for cand in STANDARD_SPATIAL_UNIT_CANDIDATES
        if hasproperty(df, cand)
            return cand
        end
    end

    if allow_nothing
        return nothing
    end

    error(
        "Could not automatically detect spatial unit column in dataset. " *
        "Available columns: $(propertynames(df)). " *
        "Please specify spatial unit variable explicitly " *
        "(e.g., `random(region, model=:icar)` or s_idx_var=:region)."
    )
end

"""
Standard candidate column names for seasonal and cyclic indexing, ordered by preference.
"""
const STANDARD_SEASONAL_CANDIDATES = [
    :u_idx,
    :month,
    :season,
    :quarter,
    :week,
    :doy,
    :day_of_year,
    :hour,
    :tod,
    :time_of_day,
    :period,
    :cycle,
    :step
]

"""
    _detect_seasonal_column(df; u_idx_var=nothing, allow_nothing=false)::Union{Symbol, Nothing}

Automatically identifies or validates the seasonal/cyclic index column in a dataset.

# Process
1. If `u_idx_var` is explicitly specified, validates that it exists in `df` and returns it.
2. Otherwise, matches against `STANDARD_SEASONAL_CANDIDATES`.
3. If not found and `allow_nothing=false`, raises a descriptive `error`.

# Inputs
- `df`: Tabular dataset (`DataFrame`, `NamedTuple`, or table-like object).
- `u_idx_var`: Optional explicit column name.
- `allow_nothing`: If true, returns `nothing` when not detected instead of error.

# Outputs
- `Union{Symbol, Nothing}`: Resolved column symbol or `nothing`.
"""
function _detect_seasonal_column(
    df; u_idx_var=nothing, allow_nothing::Bool=false
)::Union{Symbol, Nothing}
    if !isnothing(u_idx_var)
        sym = Symbol(u_idx_var)
        if !hasproperty(df, sym)
            allow_nothing && return nothing
            error(
                "Specified seasonal index column u_idx_var=:$sym not found in data: " *
                "$(propertynames(df))."
            )
        end
        return sym
    end

    for cand in STANDARD_SEASONAL_CANDIDATES
        if hasproperty(df, cand)
            return cand
        end
    end

    if allow_nothing
        return nothing
    end

    error(
        "Could not automatically detect seasonal/cyclic column in dataset. " *
        "Available columns: $(propertynames(df)). " *
        "Please specify seasonal variable explicitly " *
        "(e.g., `random(month, model=:cyclic)` or u_idx_var=:month)."
    )
end

"""
Standard candidate column names for grouping / clustering, ordered by preference.
"""
const STANDARD_GROUP_CANDIDATES = [
    :group,
    :group_id,
    :grp,
    :g_idx,
    :subject,
    :subject_id,
    :id,
    :cluster,
    :cluster_id,
    :batch,
    :category,
    :stratum,
    :strata
]

"""
    _detect_group_column(df; grouping_covariate=nothing, allow_nothing=false)::Union{Symbol, Nothing}

Automatically identifies or validates the grouping or cluster column in a dataset.

# Process
1. If `grouping_covariate` is explicitly specified, validates that it exists in `df` and returns it.
2. Otherwise, matches against `STANDARD_GROUP_CANDIDATES`.
3. If not found and `allow_nothing=false`, raises a descriptive `error`.

# Inputs
- `df`: Tabular dataset (`DataFrame`, `NamedTuple`, or table-like object).
- `grouping_covariate`: Optional explicit column name.
- `allow_nothing`: If true, returns `nothing` when not detected instead of error.

# Outputs
- `Union{Symbol, Nothing}`: Resolved column symbol or `nothing`.
"""
function _detect_group_column(
    df; grouping_covariate=nothing, allow_nothing::Bool=false
)::Union{Symbol, Nothing}
    if !isnothing(grouping_covariate)
        sym = Symbol(grouping_covariate)
        if !hasproperty(df, sym)
            allow_nothing && return nothing
            error(
                "Specified group column grouping_covariate=:$sym not found in data: " *
                "$(propertynames(df))."
            )
        end
        return sym
    end

    for cand in STANDARD_GROUP_CANDIDATES
        if hasproperty(df, cand)
            return cand
        end
    end

    if allow_nothing
        return nothing
    end

    error(
        "Could not automatically detect grouping column in dataset. " *
        "Available columns: $(propertynames(df)). " *
        "Please specify grouping variable explicitly (e.g., `random(group, model=:iid)`)."
    )
end

"""
Standard supported coupling and interaction modes for nested multi-fidelity models:
- `:additive`       : Standard additive link η_parent = η_base + ρ * η_sub[mapping].
- `:multiplicative` : Proportional modulator η_parent = η_base * (1 + ρ * η_sub[mapping]).
- `:interaction`    : Cross-product interaction η_parent = η_base + ρ * (η_base .* η_sub[mapping]).
- `:tensor`         : Kronecker tensor product coupling across multiple strata factors.
- `:moderated`      : Continuous moderator interaction ρ = ρ_0[strata] + ρ_1[strata] * mod.
"""
const NESTED_COUPLING_MODES = [
    :additive,
    :multiplicative,
    :interaction,
    :tensor,
    :moderated
]

"""
    _resolve_nested_strata(data, strata_arg; calling_mod::Module = Main)

Resolves single or compound strata definitions for nested multi-fidelity coupling.

# Mathematical Formulation
When multiple strata columns ``\\{C_1, \\dots, C_K\\}`` are provided (e.g. `[:region, :season]`),
each observation ``i`` is mapped to a compound joint stratum tuple:
``s[i] = (C_1[i], \\dots, C_K[i])``
Unique observed combinations define discrete compound levels ``\\{1, \\dots, S_{\\text{tot}}\\}``,
allowing coupling parameters ``\\boldsymbol{\\rho}`` to vary across joint parent strata or
to form factorial / Kronecker tensor interactions.

# Inputs
- `data`: Tabular dataset (`DataFrame`, `NamedTuple`, or table-like object).
- `strata_arg`: Strata specification. Supports:
  - `Symbol` or `AbstractString`: Single column name in `data`.
  - `Vector{Symbol}`, `Vector{String}`, `Tuple`: Multiple column names in `data`.
  - `AbstractVector{<:Integer}`: Pre-computed integer stratum indices.
  - `nothing`: Unstratified coupling.
- `calling_mod`: Calling module scope for variable evaluation.

# Outputs
- `NamedTuple` containing:
  - `strata_indices`: `Vector{Int}` of length ``N`` mapping each row to joint stratum ``1 \\dots S``.
  - `n_strata`: `Int` total count of unique joint strata.
  - `strata_levels`: `Vector{Any}` of unique level values or compound tuples.
  - `strata_factors`: `Vector{Symbol}` names of strata factors.
  - `factor_indices`: `Dict{Symbol, Vector{Int}}` marginal indices per factor.
  - `factor_levels`: `Dict{Symbol, Vector{Any}}` unique marginal levels per factor.
  - `factor_dims`: `Dict{Symbol, Int}` counts of marginal levels per factor.
  - `is_multistrata`: `Bool` indicating if multiple factors define the strata.
"""
function _resolve_nested_strata(data, strata_arg; calling_mod::Module = Main)
    isnothing(strata_arg) && return nothing

    N_obs = size(data, 1)

    # 1. Direct integer index vector
    if strata_arg isa AbstractVector{<:Integer}
        s_idx = Vector{Int}(strata_arg)
        length(s_idx) == N_obs || throw(DimensionMismatch(
            "Length of pre-computed strata_indices ($(length(s_idx))) " *
            "does not match data row count ($(N_obs))."
        ))
        n_s = isempty(s_idx) ? 0 : maximum(s_idx)
        return (
            strata_indices = s_idx,
            n_strata = n_s,
            strata_levels = collect(1:n_s),
            strata_factors = Symbol[],
            factor_indices = Dict{Symbol, Vector{Int}}(),
            factor_levels = Dict{Symbol, Vector{Any}}(),
            factor_dims = Dict{Symbol, Int}(),
            is_multistrata = false
        )
    end

    # 2. Extract column symbol list
    col_syms = if strata_arg isa Symbol || strata_arg isa AbstractString
        [Symbol(strata_arg)]
    elseif strata_arg isa AbstractVector
        [Symbol(c) for c in strata_arg]
    elseif strata_arg isa Tuple
        [Symbol(c) for c in strata_arg]
    else
        throw(ArgumentError(
            "Unsupported type for nested strata specification: $(typeof(strata_arg)). " *
            "Expected Symbol, Vector{Symbol}, Tuple, or Vector{<:Integer}."
        ))
    end

    isempty(col_syms) && return nothing

    # Validate all strata columns exist in data
    for col in col_syms
        if !hasproperty(data, col)
            throw(ArgumentError(
                "Nested strata column ':$col' not found in data. " *
                "Available columns: $(propertynames(data))."
            ))
        end
    end

    # 3. Compute marginal factor levels and indices
    factor_indices = Dict{Symbol, Vector{Int}}()
    factor_levels = Dict{Symbol, Vector{Any}}()
    factor_dims = Dict{Symbol, Int}()

    for col in col_syms
        raw_vals = data[!, col]
        u_vals = unique(raw_vals)
        val_map = Dict(v => i for (i, v) in enumerate(u_vals))
        factor_indices[col] = [val_map[v] for v in raw_vals]
        factor_levels[col] = Vector{Any}(u_vals)
        factor_dims[col] = length(u_vals)
    end

    # 4. Compute joint compound strata levels and indices
    if length(col_syms) == 1
        col = col_syms[1]
        s_idx = factor_indices[col]
        n_s = factor_dims[col]
        u_levels = factor_levels[col]
        is_multi = false
    else
        compound_tuples = [
            Tuple(data[i, col] for col in col_syms) for i in 1:N_obs
        ]
        u_levels = unique(compound_tuples)
        n_s = length(u_levels)
        compound_map = Dict(tup => i for (i, tup) in enumerate(u_levels))
        s_idx = [compound_map[tup] for tup in compound_tuples]
        is_multi = true
    end

    return (
        strata_indices = s_idx,
        n_strata = n_s,
        strata_levels = u_levels,
        strata_factors = col_syms,
        factor_indices = factor_indices,
        factor_levels = factor_levels,
        factor_dims = factor_dims,
        is_multistrata = is_multi
    )
end
