 
"""
    Warp <: ComponentModel
 
A component for a non-stationary Gaussian Process using a warping function. This
component implements a simple Deep GP structure with one hidden layer. The input
coordinates are first "warped" by a non-linear function (approximated by RFFs),
and then a standard RFF-based GP is applied to these warped coordinates.

# Version
v1.0.0

# Mathematical Summary
A Warped Gaussian Process models a non-stationary function \$f(x)\$ by composing a
standard, stationary GP \$h(\\cdot)\$ with a non-linear warping function \$g(x)\$.
The model is defined as:
 
\$f(x) = h(g(x))\$
 
The warping function \$g(x)\$ is typically defined as an offset from the original
coordinates:
 
\$g(x) = x + \\text{offset}(x)\$
 
Both the main process \$h(\\cdot)\$ and the offset function \$\\text{offset}(x)\$ are
modeled as GPs, which are approximated using Random Fourier Features (RFFs) for
computational scalability. This allows the model to learn how to stretch and
compress the input space to best fit the data, effectively making the length_scale
of the main GP dependent on the input location \$x\$.

# Computational Methods
- `:noncentered` (default): A non-centered parameterization where the RFF coefficients
  for the main GP are constructed from standard normal innovations. Recommended for AD.
- `:centered`: A centered parameterization where the RFF coefficients for the main GP
  are sampled directly from a scaled Normal distribution. Didactic, can be less efficient.

# Inputs
- **Required**:
  - One or more coordinate variables (e.g., `x`, `y`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `n_features`: `Int`, number of random features for both the warping and main RFF layers.
    Default: `20`.
  - `kernel`: `String`, name of the kernel to approximate (e.g., `"se"`, `"matern32"`).
    Default: `"se"`.
  - `length_scale`: `UnivariateDistribution` or `Vector{<:UnivariateDistribution}`, prior for
    the kernel length_scale(s) of the main GP.
  - `sigma`: `UnivariateDistribution`, prior for the std. dev. of the main GP's coefficients.
  - `method`: `Symbol`, the computational method (`:noncentered` or `:centered`). Default:
    `:noncentered`.

# Outputs (Parameter Names)
- `sigma_<key>`: The standard deviation of the main GP's coefficients.
- `ls_<key>`: The kernel length_scale(s).
- `W_warp_<key>`: RFF weights for the warping layer.
- `b_warp_<key>`: RFF biases for the warping layer.
- `beta_warp_<key>`: RFF coefficients for the warping layer.
- `W_main_<key>`: RFF weights for the main GP layer.
- `b_main_<key>`: RFF biases for the main GP layer.
- `innovations_<key>`: Raw standard normal innovations for main GP coefficients (non-centered).
- `latent_<key>`: Main GP coefficients (centered).
"""
struct Warp <: ComponentModel
    length_scale::Union{Distribution, Vector{<:Distribution}}
    sigma::Union{Distribution, Real}
    n_features::Int
    kernel::String
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:warp] = Warp
 
COMPONENT_CONSTRUCTORS[:warp] = (p, params) -> Warp(
    p.length_scale,
    p.sigma,
    get(params, :n_features, 20),
    string(get(params, :kernel, "se")),
    get(params, :method, :noncentered)
)

MODEL_TO_STRUCTURE_MAP[:warp] = :smooth
 
"""
    get_precomputes(m::Warp, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-independent pre-calculations for a Warp component instance.
This includes validating coordinate variables and storing the coordinate matrix
and its dimensions. All RFF parameters are learned, so no fixed features are
pre-generated.

# Arguments
- `m`: The `Warp` component instance.
- `M`: The main model configuration `NamedTuple`.
- `mod_data`: A dictionary containing parsed module data.

# Returns
- A `NamedTuple` containing precomputed items: `coords`, `in_dims`, `n_latent`.
"""
function get_precomputes(m::Warp, M::NamedTuple, mod_data::Dict)::NamedTuple
    variables = mod_data[:variables]
    if isempty(variables)
        error("The Warp model requires coordinate variables, e.g., `random(x, y, model=:warp)`.")
    end
    for var_sym in variables
        if !hasproperty(M.data, Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for Warp model not found in data.")
        end
    end
    
    coords = Matrix{Float64}(M.data[!, Symbol.(variables)])
    in_dims = size(coords, 2)
    n_latent = size(coords, 1)
    return (
        coords=coords,
        in_dims=in_dims,
        n_latent=n_latent
    )
end

"""
    get_priors(m::Warp, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code string for the Warp component's priors.
This includes priors for the overall standard deviation (`sigma`),
the kernel length_scale(s), and all Random Fourier Features (RFF)
parameters for both the warping layer and the main GP layer.

# Arguments
- `m`: The `Warp` component instance.
- `spec`: A `NamedTuple` containing the component's full specification.
- `arch`: The model architecture (`"univariate"` or `"multivariate"`).
- `outcome_idx`: The index of the outcome for multivariate models, `nothing` otherwise.
- `M`: The main model configuration `NamedTuple`.

# Returns
- A `String` containing the generated Turing code for priors.
"""
function get_priors(m::Warp, spec::NamedTuple, arch::String, outcome_idx, M)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    rff_weights_warp_name = Symbol("$(p_names.rff_weights)_warp")
    rff_offsets_warp_name = Symbol("$(p_names.rff_offsets)_warp")
    beta_warp_name = Symbol("$(p_names.beta)_warp")
    rff_weights_main_name = Symbol("$(p_names.rff_weights)_main")
    rff_offsets_main_name = Symbol("$(p_names.rff_offsets)_main")
    
    key = spec.key
    # `spec_registry` is a name that exists only inside the generated model body, where the
    # model receives it as an argument. Referencing it from `get_priors`/`get_updates` --
    # which run in `TethysBayes`'s own scope -- raised
    # `UndefVarError: spec_registry not defined in TethysBayes`, so `warp` could not be built at
    # all. The value lives on `spec` directly.
    in_dims = spec.hyper.in_dims
    n_features = m.n_features

    priors = String[]
    push!(priors, "$(_prior_or_constant(p_names.sigma, m.sigma))")

    if m.length_scale isa Vector
        length_scale_priors_str = join([_distribution_to_string(p) for p in m.length_scale], ", ")
        push!(priors, "$(p_names.length_scale) ~ Product([$(length_scale_priors_str)])")
    else
        length_scale_prior_str = _distribution_to_string(m.length_scale)
        push!(priors, "$(p_names.length_scale) ~ $(length_scale_prior_str)")
    end

    # Priors for warping layer RFF parameters (always sampled)
    push!(priors, """

        $(rff_weights_warp_name) ~ MvNormal(zeros(T, $(in_dims * n_features)), I)
    """)

    push!(priors, """
        $(rff_offsets_warp_name) ~ MvNormal(zeros(T, $(n_features)), I)
    """)
    push!(priors, """
        $(beta_warp_name) ~ MvNormal(zeros(T, $(n_features)), I)
    """)
    # Priors for main GP layer RFF parameters (always sampled)
    push!(priors, """
        $(rff_weights_main_name) ~ MvNormal(zeros(T, $(in_dims * n_features)), I)
    """)
    push!(priors, """
        $(rff_offsets_main_name) ~ MvNormal(zeros(T, $(n_features)), I)
    """)
    # Prior for main GP coefficients (depends on method)
    if m.method == :noncentered
        push!(priors, """
            $(p_names.innovations) ~ MvNormal(zeros(T, $(n_features)), I)
        """)
    elseif m.method == :centered
        push!(priors, """
            $(p_names.latent_field) ~ MvNormal(
                zeros(T, $(n_features)), $(p_names.sigma)^2 * I
            )
        """)
    end
    
    return join(priors, "\n    ")
end

"""
    get_updates(m::Warp, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code string for constructing the `Warp` smooth effect
and adding it to the linear predictor.

This function dispatches on the chosen computational method (`:noncentered` or
`:centered`). It first constructs a non-linear warping function using RFFs,
then applies a main GP (also RFF-based) to these warped coordinates.

# Arguments
- `m`: The `Warp` component instance.
- `spec`: A `NamedTuple` containing the component's full specification.
- `arch`: The model architecture (`"univariate"` or `"multivariate"`).
- `outcome_idx`: The index of the outcome for multivariate models, `nothing` otherwise.
- `M`: The main model configuration `NamedTuple`.

# Returns
- A `String` containing the generated Turing code for the component's update logic.
"""
function get_updates(m::Warp, spec::NamedTuple, arch::String, outcome_idx, M)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    rff_weights_warp_name = Symbol("$(p_names.rff_weights)_warp")
    rff_offsets_warp_name = Symbol("$(p_names.rff_offsets)_warp")
    beta_warp_name = Symbol("$(p_names.beta)_warp")
    rff_weights_main_name = Symbol("$(p_names.rff_weights)_main")
    rff_offsets_main_name = Symbol("$(p_names.rff_offsets)_main")

    key = spec.key
    # `spec_registry` is a name that exists only inside the generated model body, where the
    # model receives it as an argument. Referencing it from `get_priors`/`get_updates` --
    # which run in `TethysBayes`'s own scope -- raised
    # `UndefVarError: spec_registry not defined in TethysBayes`, so `warp` could not be built at
    # all. The value lives on `spec` directly.
    in_dims = spec.hyper.in_dims
    n_features = m.n_features

    length_scale_scaling_code = if m.length_scale isa Vector
        "local W_main_matrix = reshape($(rff_weights_main_name), $(in_dims), $(n_features)) ./ $(p_names.length_scale)'"
    else
        "local W_main_matrix = reshape($(rff_weights_main_name), $(in_dims), $(n_features)) ./ $(p_names.length_scale)"
    end

    common_code = """ # Common code for both noncentered and centered methods
        local hyper = spec_registry[:$(spec.key)].hyper
        local coords = hyper.coords
        
        # 1. Construct and apply the warping function
        local W_warp_matrix = reshape($(rff_weights_warp_name), $(in_dims), $(n_features))
        local Phi_warp = sqrt(2.0 / $(n_features)) .* cos.((coords * W_warp_matrix) .+
          $(rff_offsets_warp_name)')
        local warping_effect = Phi_warp * $(beta_warp_name)
        local coords_warped = coords .+ warping_effect

        # 2. Construct the main GP on the warped coordinates
        $(length_scale_scaling_code)
        local Phi_main = sqrt(2.0 / $(n_features)) .* cos.((coords_warped * W_main_matrix)
          .+ $(rff_offsets_main_name)')
    """

    noncentered_code = """
        # --- Warp (Deep GP) Component (Non-Centered): $(spec.key) ---
        let
            $(common_code)
            
            # 3. Scale coefficients and compute final effect
            local scaled_beta_main = $(p_names.innovations) .* $(p_names.sigma)
            $(p_names.latent_field) = Phi_main * scaled_beta_main
            
            $(eta_target) = $(eta_target) .+ $(p_names.latent_field)
        end
    """

    centered_code = """
        # --- Warp (Deep GP) Component (Centered): $(spec.key) ---
        let
            $(common_code)
            
            # 3. Sample coefficients directly and compute final effect
            $(eta_target) = $(eta_target) .+ Phi_main * $(p_names.latent_field)
        end
    """

    if m.method == :noncentered
        return noncentered_code
    elseif m.method == :centered
        return centered_code
    else
        error("Unsupported method '$(m.method)' for Warp component.")
    end
end

function get_effects(
    m::Warp, chain, spec::NamedTuple, M::NamedTuple,
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
    
    # --- Coordinate Handling: Combine training and prediction sets on CPU ---
    coords_train = spec.hyper.coords
    coord_vars = get(spec.params, :positional_args, [])
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        coords_pred_cpu = Matrix{Float64}(PS.data[!, Symbol.(coord_vars)])
        vcat(coords_train, coords_pred_cpu)
    else
        coords_train
    end
    n_obs_full = size(coords_full, 1)

    in_dims = spec.hyper.in_dims
    n_features = m.n_features
    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k_outcome in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k_outcome)
        
        # Define parameter names for this outcome
        rff_weights_warp_name = string(Symbol("$(p_names_k.rff_weights)_warp"))
        rff_offsets_warp_name = string(Symbol("$(p_names_k.rff_offsets)_warp"))
        beta_warp_name = string(Symbol("$(p_names_k.beta)_warp"))
        rff_weights_main_name = string(Symbol("$(p_names_k.rff_weights)_main"))
        rff_offsets_main_name = string(Symbol("$(p_names_k.rff_offsets)_main"))
        sigma_name = string(p_names_k.sigma)
        length_scale_name = string(p_names_k.length_scale)

        # Extract posterior samples (these are on the CPU)
        sigma_samples_cpu = get_params_vector(chain, sigma_name, 1)[:, 1]
        length_scale_dim = m.length_scale isa Vector ? length(m.length_scale) : 1
        ls_samples_cpu = get_params_matrix(chain, length_scale_name, length_scale_dim)
        W_warp_samples_cpu = get_params_matrix(chain, rff_weights_warp_name, in_dims * n_features)
        b_warp_samples_cpu = get_params_matrix(chain, rff_offsets_warp_name, n_features)
        beta_warp_samples_cpu = get_params_matrix(chain, beta_warp_name, n_features)
        W_main_samples_cpu = get_params_matrix(chain, rff_weights_main_name, in_dims * n_features)
        b_main_samples_cpu = get_params_matrix(chain, rff_offsets_main_name, n_features)

        # Initialize the output matrix for the full effect on the CPU
        effect_k_cpu = zeros(Float64, n_obs_full, n_samples)

        # --- Sample-wise Reconstruction on the CPU ---
        for i in 1:n_samples
            # 1. Construct and apply the warping function on the CPU
            W_warp_i_cpu = reshape(W_warp_samples_cpu[i, :], in_dims, n_features)
            b_warp_i_cpu = b_warp_samples_cpu[i, :]
            beta_warp_i_cpu = beta_warp_samples_cpu[i, :]
            
            Phi_warp_i_cpu = sqrt(2.0 / n_features) .* cos.((coords_full * W_warp_i_cpu) .+ b_warp_i_cpu')
            warping_effect_i_cpu = Phi_warp_i_cpu * beta_warp_i_cpu
            coords_warped_i_cpu = coords_full .+ warping_effect_i_cpu

            # 2. Construct the main GP on the warped coordinates on the CPU
            ls_i_cpu = length_scale_dim > 1 ? ls_samples_cpu[i, :] : ls_samples_cpu[i, 1]
            W_main_i_unscaled_cpu = reshape(W_main_samples_cpu[i, :], in_dims, n_features)
            W_main_i_cpu = W_main_i_unscaled_cpu ./ (ls_i_cpu isa AbstractVector ? ls_i_cpu' : ls_i_cpu)

            b_main_i_cpu = b_main_samples_cpu[i, :]
            Phi_main_i_cpu = sqrt(2.0 / n_features) .* cos.((coords_warped_i_cpu * W_main_i_cpu) .+ b_main_i_cpu')
            
            # 3. Scale coefficients and compute final effect on the CPU
            local beta_main_i_cpu
            if m.method == :noncentered
                innovations_name = string(p_names_k.innovations)
                beta_main_unscaled_samples_cpu = get_params_matrix(chain, innovations_name, n_features)
                beta_main_unscaled_i_cpu = beta_main_unscaled_samples_cpu[i, :]
                beta_main_i_cpu = beta_main_unscaled_i_cpu .* sigma_samples_cpu[i]
            else # :centered
                latent_field_name = string(p_names_k.latent_field)
                beta_main_samples_cpu = get_params_matrix(chain, latent_field_name, n_features)
                beta_main_i_cpu = beta_main_samples_cpu[i, :]
            end
            
            effect_k_cpu[:, i] = Phi_main_i_cpu * beta_main_i_cpu
        end
        
        push!(structured_effects, effect_k_cpu)
    end

    return (structured=structured_effects, noisy=structured_effects)
end
   
