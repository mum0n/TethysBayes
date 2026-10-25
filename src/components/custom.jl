"""
    Custom <: ComponentModel

A component for injecting user-defined arbitrary code into the `TethysBayes` model.
This provides an "escape hatch" for advanced users who need to implement logic
not covered by the standard components.

# Version
v1.0.0

# Mathematical Summary
This component does not have a fixed mathematical form. It is a "blank canvas"
where the user provides a `code_fragment` string containing valid Turing.jl model
code. This allows for the implementation of novel statistical relationships, custom
priors, or likelihoods not covered by the standard `TethysBayes` components. The user is
responsible for defining the mathematical logic within this code fragment.

For example, a user could implement a custom non-linear function:
`y ~ Normal(a * x^b, sigma)`
by providing a `code_fragment` that defines priors for `a` and `b` and adds the
result to the linear predictor `eta`.

# Assumptions
- The user-provided `code_fragment` is syntactically correct Turing.jl code.
- The user is responsible for ensuring parameter names are unique to avoid
  collisions with other model components.
- The code must correctly modify the `eta` variable if it is intended to be part
  of the linear predictor.
- The user is responsible for ensuring the provided code is AD-compatible if using
  gradient-based samplers like NUTS.

# Best Use Case
Prototyping new models, implementing highly specialized statistical processes, or
integrating logic from other libraries directly into a `TethysBayes` model without needing
to create a full custom component file from scratch. It serves as a powerful didactic
tool for learning how `TethysBayes` constructs models.

# Inputs
- **Required**:
  - `code_fragment`: A `String` containing valid Turing.jl model code. This can be
    passed directly or as a variable name (Symbol) that resolves to a string in the
    calling module.
- **Optional**:
  - `params`: A `Dict{Symbol, Any}` for user-defined parameters. A special key,
    `:reconstruct_func`, can be included. Its value should be a function that
    implements the posterior reconstruction logic for the custom component.

# Outputs (Parameter Names)
- All parameter names are defined by the user within the `code_fragment`. There are
  no standard output parameter names for this component.

# Key References
- Turing.jl Documentation: https://turing.ml/dev/docs/using-turing/
"""
struct Custom <: ComponentModel
    code_fragment::String
    params::Dict{Symbol, Any}
end

COMPONENT_TYPE_REGISTRY[:custom] = Custom

COMPONENT_CONSTRUCTORS[:custom] = (p, params) -> begin
    code_fragment_val = get(params, :code_fragment, "")
    Custom(string(code_fragment_val), get(params, :params, Dict{Symbol, Any}()))
end

MODEL_TO_STRUCTURE_MAP[:custom] = :any

"""
    get_precomputes(m::Custom, M::NamedTuple, mod_data::Dict)::NamedTuple

Returns an empty `NamedTuple`. All pre-computation must be handled within the
user's code fragment.
"""
function get_precomputes(m::Custom, M::NamedTuple, mod_data::Dict)::NamedTuple
    return (;)
end

"""
    get_priors(m::Custom, spec::NamedTuple, arch::String, outcome_idx, M)::String

Returns an empty string. All priors must be defined within the user's
`code_fragment`.
"""
function get_priors(
    m::Custom, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    return ""
end

"""
    get_updates(m::Custom, spec::NamedTuple, arch::String, outcome_idx, M)::String

Injects the user-provided `code_fragment` directly into the model's update block,
wrapped in a `let` block to prevent variable scope leakage.
"""
function get_updates(
    m::Custom, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    user_code = m.code_fragment
    
    if isempty(strip(user_code))
        @warn "Custom component '$(spec.key)' has an empty `code_fragment` and will have no effect."
        return ""
    end

    return """
        # --- Custom Code Block for $(spec.key) ---
        let
            $(user_code)
        end
    """
end

"""
    get_effects(m::Custom, chain, spec::NamedTuple, M::NamedTuple, PS)

Calls a user-provided `reconstruct_func` for posterior reconstruction. If no
function is provided, it returns a zero-effect with a warning.

The user's function is expected to have the signature:
`(m::Custom, chain, spec::NamedTuple, M::NamedTuple, PS::Union{NamedTuple, Nothing}) -> NamedTuple`

The user's function will receive standard CPU arrays within `M` and `spec` and
should return a `NamedTuple` containing standard CPU `Array`s.
"""
function get_effects(
    m::Custom, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    reconstruct_func = get(m.params, :reconstruct_func, nothing)

    if !isnothing(reconstruct_func) && isa(reconstruct_func, Function)
        try
            # The user's function is called with the modern, standardized arguments.
            return reconstruct_func(m, chain, spec, M, PS)
        catch e
            @error "The custom reconstruction function for component '$(spec.key)' failed."
            rethrow(e)
        end
    else
        @warn "Reconstruction for custom component '$(spec.key)' is not defined. " *
              "Returning a zero-effect. Provide a `reconstruct_func` to the " *
              "`custom()` module to enable posterior reconstruction."
        
        n_samples = if occursin("FlexiChain", string(typeof(chain)))
            size(chain, 1) * FlexiChains.nchains(chain)
        else
            size(chain, 1) * size(chain, 3)
        end
        outcomes_N = M.outcomes_N
        N_total = M.y_N + (isnothing(PS) ? 0 : size(PS.data, 1))

        structured_effects = [
            zeros(Float64, N_total, n_samples) for _ in 1:outcomes_N
        ]
        return (structured=structured_effects, noisy=structured_effects)
    end
end 
 