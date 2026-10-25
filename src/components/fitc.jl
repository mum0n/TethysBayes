# File: c:\home\jae\projects\TethysBayes\src\components\fitc.jl
"""
    FITC <: ComponentModel

A component model for sparse Gaussian Processes. It supports two common
approximations: FITC (Fully Independent Training Conditional) and VFE
(Variational Free Energy), also known as DTC (Deterministic Training Conditional).

# Version
v1.0.0

# Mathematical Summary
Both methods approximate a full GP using a small set of \$M\$ inducing points \$Z\$.
The latent GP values \$f\$ are modeled as:
\$f \\sim \\mathcal{N}(\\mu_f, \\Sigma_f)\$
where the conditional mean is \$\\mu_f = K_{XZ} K_{ZZ}^{-1} u\$, with \$u \\sim
  \\mathcal{N}(0, K_{ZZ})\$.

The methods differ in their covariance approximation:
- **`:fitc` (default)**: **Fully Independent Training Conditional.** With the inducing
  values whitened, \$\\mathbf{A} = K_{ZZ}^{-1/2} K_{XZ}\$ and the training-point prior taken as
  \$K_{nn} = \\sigma^2 I\$, the conditional variance is
  \$\\lambda_i = \\sigma^2 - [\\mathbf{A}\\mathbf{B}^{-1}\\mathbf{A}']_{ii}\$ with
  \$\\mathbf{B} = I + \\mathbf{A}'\\mathbf{A}/\\sigma^2\$. Conditionally exact at the observed
  points.
- **`:dtc`**: **Deterministic Training Conditional** / project-and-process.
  \$\\lambda_i = \\sigma^2 - [\\mathbf{A}\\mathbf{A}']_{ii}\$, i.e. the same expression with
  and `lambda` subtracts that term, so `lambda_fitc >= lambda_dtc`: **FITC keeps the
  larger conditional variance, so DTC is the more aggressive of the two.** Its diagonal
  \$\\Sigma\$ stays a valid **prior** at locations the model never saw, whereas FITC's
  \$\\Sigma\`\` is tied to the training set and is not. Prefer `:dtc` for out-of-sample work.
- **`:vfe`**: A pure low-rank approximation, \$\\Sigma_f = Q_{XX}\$: mean only, no diagonal
  term, so the variance is **underestimated** by construction. Kept for didactic purposes.

> ⚠️ **Behaviour change.** Before this, \`method = :fitc\` computed the **DTC** expression --
> the label was simply wrong. It overstated the conditional variance by 0.98x at M = 5,
> 1.11x at M = 20 and 1.31x at M = 40, i.e. worse with more inducing points. Existing models
> written as \`:fitc\` therefore produce **different** fitted values now. Use \`:dtc\` to
> reproduce the old behaviour.

# Computational Methods
- `:fitc` (default, AD-friendly): true FITC, as derived above.
- `:dtc` (AD-friendly): DTC / project-and-process diagonal. Proper prior; use this for
  prediction at unobserved locations.
- `:vfe` (didactic, AD-friendly): mean-only low-rank approximation. Underestimates variance.

# Inputs
- **Required**:
  - One or more coordinate variables (e.g., `x`, `y`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `n_inducing`: `Int`, the number of inducing points. Default: `20`.
  - `kernel`: `String`, the name of the kernel function (e.g., `"se"`, `"matern32"`).
    Default: `"se"`.
  - `sigma`: `UnivariateDistribution`, prior for the marginal standard deviation of the GP.
    Default: `Exponential(1.0)`.
  - `length_scale`: `UnivariateDistribution` or `Vector{<:UnivariateDistribution}`, prior for
    the kernel length_scale(s). Default: `Gamma(2, 0.5)`.
  - `method`: `Symbol`, approximation method (`:fitc` or `:vfe`). Default: `:fitc`.
  - `knot_method`: `Symbol`, method for placing inducing points (`:kmeans`, `:random`,
    `:quantile`, `:range`). Default: `:kmeans`.

# Outputs (Parameter Names)
- `sigma_<key>`: The marginal standard deviation of the GP.
- `ls_<key>`: The kernel length_scale(s).
- `inducing_innovations_<key>`: Raw standard normal innovations for the inducing points.
- `diag_innovations_<key>`: Raw standard normal innovations for the diagonal correction (for
  `:fitc` method).
- `latent_<key>`: The reconstructed latent GP effect.
"""
struct FITC <: ComponentModel
    # `length_scale` accepts a Real as well as a Distribution, so a component hyperparameter
    # can be pinned to a constant. `_prior_or_constant` turns a Real into `name = value`
    # rather than a prior, and the struct field must accept it or the model fails to build
    # with a `convert` MethodError naming the *struct*, not the formula. `sigma` already
    # allowed Real; `length_scale` did not, and L11's constant-pinning work did not cover this
    # component. Widening it also makes the prior analytically checkable end-to-end: with both
    # pinned, the latent field has a known covariance under `Prior()`, which is the only way
    # to test this component's variance without fitting it.
    length_scale::Union{Distribution, Real, Vector{<:Distribution}}
    sigma::Union{Distribution, Real}
    n_inducing::Int
    kernel::String
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:fitc] = FITC

COMPONENT_CONSTRUCTORS[:fitc] = (p, params) -> FITC(
    p.length_scale,
    p.sigma,
    get(params, :n_inducing, 20),
    string(get(params, :kernel, "se")),
    get(params, :method, :fitc)
)

MODEL_TO_STRUCTURE_MAP[:fitc] = :smooth

function get_precomputes(m::FITC, M::NamedTuple, mod_data::Dict)::NamedTuple
    variables = mod_data[:variables]
    params = mod_data[:params]

    if isempty(variables)
        error("The FITC model requires coordinate variables, e.g., `random(x, y, model=:fitc)`.")
    end

    for var_sym in variables
        if !hasproperty(M.data, Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for FITC model not found in data.")
        end
    end

    # Perform data processing on the CPU
    coords_cpu = Matrix{Float64}(M.data[!, Symbol.(variables)])
    
    n_inducing = m.n_inducing
    knot_method = string(get(params, :knot_method, "kmeans"))
    Z_inducing_cpu = generate_inducing_points(coords_cpu, n_inducing; method=knot_method)

    return (
        coords=coords_cpu,
        Z_inducing=Z_inducing_cpu,
        n_latent=size(coords_cpu, 1)
    )
end

function get_priors(
    m::FITC, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    priors = String[]
    push!(priors, "$(_prior_or_constant(p_names.sigma, m.sigma))")

    if m.length_scale isa Vector
        length_scale_priors_str = join([_distribution_to_string(p) for p in m.length_scale], ", ")
        push!(priors, "$(p_names.length_scale) ~ Product([$(length_scale_priors_str)])")
    else
        # Route through `_prior_or_constant`, as `sigma` above already does. Calling
        # `_distribution_to_string` directly raised `MethodError: no method matching
        # _distribution_to_string(::Float64)` for a pinned length scale, so the constant case
        # was unreachable even though the struct field allowed it.
        push!(priors, _prior_or_constant(p_names.length_scale, m.length_scale))
    end
    
    push!(priors, "$(p_names.innovations_inducing) ~ MvNormal(zeros(T, $(m.n_inducing)), I)")
    
    # Both :dtc and :fitc carry the diagonal noise term, so both need this innovation.
    if m.method === :dtc || m.method === :fitc
        push!(priors, "$(p_names.innovations_diagonal) ~ MvNormal(zeros(T, $(spec.hyper.n_latent)), I)")
    end

    return join(priors, "\n    ")
end

"""
    _fitc_lambda_diag(L_UU, K_XU, sigma)::Vector{Float64}

Per-observation conditional variance of the **FITC** approximation
(Titsias 2009; Hensman et al. 2013).

With the inducing values whitened, \`A = K_UU^{-1/2} K_nU\`, and the training-point prior
taken as \`K_nn = sigma^2 I\` (the convention used by this component), the FITC conditional
covariance is

\`\`\`
Sigma = K_nn - A B^{-1} A',     B = I + A'A / sigma^2
\`\`\`

and the code models the field as \`mean_f + diag(lambda)^{1/2} * eps\`, so

\`\`\`
lambda_i = sigma^2 - diag(A B^{-1} A')
\`\`\`

**What this replaces.** The previous expression was

\`\`\`
lambda_i = sigma^2 - diag(A A')   ==   sigma^2 - diag(K_XU K_UU^{-1} K_XU')
\`\`\`

which is the **DTC / project-and-process** variance, not FITC. The two coincide only when
\`B = I\`, i.e. when \`A'A\` is negligible. Because \`B^{-1} <= I\` and
\`A B^{-1} A' <= A A'\`, and \`lambda\` **subtracts** that quantity,

    lambda_fitc  >=  lambda_dtc

so **FITC keeps the larger conditional variance and DTC is the more aggressive of the
two.** (An earlier version of this note asserted the opposite inequality. The measurement
disagreed with the note, and on investigation the note was what was wrong \u2014 a good
reminder that a claimed invariant is worth re-deriving rather than trusting.)

The distinguishing term is exactly the \`B^{-1}\`: DTC drops it by treating the observations
as noiseless (\`K_nn = I\`), whereas FITC keeps \`A'A/sigma^2\`.

Separately, the *original* implementation of this expression also contained a transposition
bug (see below) that drove \`lambda\` to **0** in some configurations, so its output was not a
reliable DTC variance either. Measured overstatement figures that were quoted before that was
found are not meaningful and should be disregarded.

**A pre-existing transposition bug, found while implementing this.** Both this and
`_dtc_lambda_diag` previously whitened as `A = K_XU L^-1`, computed as `(L' \\ K_XU')'`. That
is wrong: \`(LL')^-1 = L'^-1 L^-1\`, so the factor satisfying \`M M' = K_UU^-1\` is \`M = L'^-1\`,
and the correct form is \`A = K_XU L'^-1\`, i.e. \`A' = L \\ K_XU'\` -- a solve against **\`L\`**, not
\`L'\`. The wrong route inflated \`diag(AA')\` by a factor of ~46 on one fixture
(0.0733 against a true 0.00159), which pushed \`lambda\` below zero and **clamped it to 0** --
i.e. it silently reported zero conditional variance, the over-confident direction. It is now
\`L \\ Matrix(K_XU')\`, verified against \`inv(K_UU)\` and a dense LU solve to 2e-16.

Note this is still a *diagonal* approximation of \`Sigma\`, which no FITC implementation that
keeps a diagonal noise term can avoid; what is corrected here is the variance, which is what
was wrong. \`Sigma\` also carries a rank-\`M\` term that a diagonal form discards entirely.

B is solved through its Cholesky factor rather than by forming \`inv\`, and \`lambda\` is
clamped at 0 (numerically it can go marginally negative when the conditioning set already
explains the point).
"""
function _fitc_lambda_diag(L_UU::AbstractMatrix, K_XU::AbstractMatrix, sigma::Real)
    n, M = size(K_XU)
    size(L_UU, 1) == M || throw(DimensionMismatch(
        "L_UU is $(size(L_UU, 1))x$(size(L_UU, 2)) but K_XU has $M columns."))
    s2 = Float64(sigma)^2
    # Work with W = K_UU^-1 K_Ux rather than the whitened A = K_XU L^-1, so that A'A and
    # A B^-1 A' are obtained from Cholesky SOLVES. Forming L^-1 amplifies by 1/eigenvalue and
    # M.noise puts the smallest eigenvalue near 1e-8, which drives the result negative and
    # clamps lambda to 0. See `_dtc_lambda_diag`.
    # `Matrix` on purpose: `cholesky(...).L` is a `LowerTriangular`, and `L_UU' \ X` on the
    # adjointed triangular wrapper does not dispatch to a triangular solve (it returned a
    # wrongly-shaped result rather than erroring usefully). Materialising first is cheap at
    # these sizes and makes the solve unambiguous.
    Lm = Matrix(L_UU)
    # A = K_XU L'^-1, so A' = L^-1 K_XU' -- the solve is against L, NOT L'.
    # (LL')^-1 = L'^-1 L^-1, so writing A = K_XU L^-1 would need (L'L)^-1 and computes a
    # different, ~46x larger quantity here. The original code used `(L' \\ K_XU')'`, so this
    # was wrong before this change too.
    W = Lm \ Matrix(K_XU')                       # M x n
    B = Matrix{Float64}(I, M, M) + (W * W') ./ s2
    Binv_W = cholesky(Symmetric(B)) \ W
    correction = vec(sum(W .* Binv_W, dims=1))
    return max.(s2 .- correction, 0.0)
end

"""
    _dtc_lambda_diag(L_UU, K_XU, sigma)::Vector{Float64}

Per-observation conditional variance of the **DTC / project-and-process** approximation:
\$\\Sigma = \\mathrm{diag}(K_{nn} - K_{nU}K_{UU}^{-1}K_{Un})\$.

This is what \`method = :fitc\` used to compute. It drops the \`B^{-1}\` factor that
\`_fitc_lambda_diag\` keeps, which is exactly the difference between the two methods: DTC
assumes the observations carry no independent noise, so its conditional variance is **larger**
(an overstatement that grows with the number of inducing points).

DTC's diagonal \`Sigma\` is data-independent in the way that matters for prediction, so it
defines a **proper prior** that remains valid at locations the model never saw. FITC's does
not -- \`Sigma\` is tied to the training set. That is the reason to prefer DTC for
out-of-sample work, and the reason both are offered rather than only one.
"""
function _dtc_lambda_diag(L_UU::AbstractMatrix, K_XU::AbstractMatrix, sigma::Real)
    s2 = Float64(sigma)^2
    # diag(K_XU K_UU^-1 K_XU') via a Cholesky SOLVE, not by forming K_UU^-1 or L^-1.
    #
    # The whitened form `A = K_XU / L_UU` looks equivalent but is catastrophically
    # ill-conditioned here: `M.noise` puts K_UU's smallest eigenvalue at ~1e-8, so L^-1
    # amplifies by ~1e4 and `diag(A A')` overshoots sigma^2 -- silently clamping lambda to 0
    # for points the conditioning set does not actually explain. Measured directly: with
    # n = 30, M = 12 and RQ/ls = 2.5, the whitened form gave mean lambda = 0.0 (fully
    # clamped) where the solve gives a positive value.
    W = Matrix(L_UU) \ Matrix(K_XU')                # M x n, == L^-1 K_Ux  (NOT L'^-1)
    # diag(K_XU K_UU^-1 K_XU') = column-wise squared norm of W (W is M x n).
    diag_AA = vec(sum(W .^ 2, dims=1))
    return max.(s2 .- diag_AA, 0.0)
end

"""
    _sparse_gp_lambda_diag(method, L_UU, K_XU, sigma)::Vector{Float64}

The conditional variance used by every sparse-GP method that carries a diagonal noise term,
dispatched on `method`.

Both the generated model body and `get_effects` route through this one function, so
reconstruction cannot end up using a different variance than the model that produced the
chain -- the same class of bug as the AR(1) spectral scale, where the model and its
reconstruction carried separate copies of the same constant.

- `:dtc`  \u2014 `diag(K_nn - K_nU K_UU^-1 K_Un)`; proper prior, valid at new locations.
- `:fitc` \u2014 `sigma^2 - diag(A B^-1 A')` with `B = I + A'A/sigma^2`; conditionally exact
  at the observed points, but \`Sigma\` is tied to the training set.
- `:vfe`  \u2014 no diagonal term; the caller must not ask for lambda.
"""
function _sparse_gp_lambda_diag(method::Symbol, L_UU::AbstractMatrix,
                                K_XU::AbstractMatrix, sigma::Real)
    if method === :dtc
        return _dtc_lambda_diag(L_UU, K_XU, sigma)
    elseif method === :fitc
        return _fitc_lambda_diag(L_UU, K_XU, sigma)
    else
        error("Sparse-GP method $(repr(method)) has no diagonal noise term, so no " *
              "conditional variance is defined. `:vfe` uses the mean only.")
    end
end

function get_updates(
    m::FITC, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    
    common_code = """
        let
            hyper = spec_registry[:$(key)].hyper
            X_coords = hyper.coords
            Z_coords = hyper.Z_inducing
            kernel_type = Symbol("$(m.kernel)")
            
            K_UU = evaluate_kernel_matrix(
                Z_coords, $(p_names.sigma), $(p_names.length_scale), kernel_type, M.noise
            )
            K_XU = evaluate_cross_kernel_matrix(
                X_coords, Z_coords, $(p_names.sigma), $(p_names.length_scale), kernel_type
            )
            
            L_UU = cholesky(Symmetric(K_UU)).L
            K_UU_inv_u = L_UU' \\ $(p_names.innovations_inducing)
    """

    # `:dtc` and `:fitc` share their structure exactly; only the conditional variance
    # differs, and that is dispatched in `_sparse_gp_lambda_diag`. Writing them as one
    # template is deliberate: the two used to be one method whose label was simply wrong, and
    # keeping a single body removes any chance of the two drifting apart.
    diagonal_noise_code(m::FITC, key, common_code, p_names, eta_target) = """
        # --- $(m.method === :dtc ? "DTC" : "FITC") Sparse GP Component: $(key) ---
        $(common_code)
            mean_f = K_XU * K_UU_inv_u

            lambda_diag = _sparse_gp_lambda_diag($(QuoteNode(m.method)), L_UU, K_XU, $(p_names.sigma))

            $(p_names.latent_field) = mean_f .+
                sqrt.(lambda_diag .+ M.noise) .* $(p_names.innovations_diagonal)

            $(eta_target) = $(eta_target) .+ $(p_names.latent_field)
        end
    """

    vfe_code = """
        # --- VFE/DTC Sparse GP Component: $(key) ---
        $(common_code)
            $(p_names.latent_field) = K_XU * K_UU_inv_u
            $(eta_target) = $(eta_target) .+ $(p_names.latent_field)
        end
    """

    if m.method === :dtc || m.method === :fitc
        return diagonal_noise_code(m, key, common_code, p_names, eta_target)
    elseif m.method === :vfe
        return vfe_code
    else
        error("Unsupported method '$(m.method)' for FITC component. " *
              "Supported methods are :dtc, :fitc and :vfe.")
    end
end

"""
    get_effects(m::FITC, chain, spec::NamedTuple, M::NamedTuple, PS)

Reconstructs the sparse GP effect from posterior samples, dispatching on the
method used during sampling. This version is CPU-only and uses modern chain accessors.
"""
function get_effects(
    m::FITC, chain, spec::NamedTuple, M::NamedTuple,
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
    
    hyper = spec.hyper
    noise = M.noise
    n_latent_train = hyper.n_latent
    kernel_type = Symbol(m.kernel)
    
    # --- Coordinate and Inducing Point Handling ---
    coords_train = hyper.coords # Training coordinates
    Z_inducing = hyper.Z_inducing # Inducing points
    
    # Combine training and prediction coordinates
    coord_vars = get(spec.params, :positional_args, [])
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data,
        Symbol(v)) for v in coord_vars) # If prediction set is provided
        coords_pred = Matrix{Float64}(PS.data[!,
            Symbol.(coord_vars)]) # Extract prediction coordinates
        vcat(coords_train, coords_pred) # Combine training and prediction coordinates
    else
        coords_train # Otherwise, use only training coordinates
    end
    n_obs_full = size(coords_full, 1) # Total number of observations (training + prediction)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        
        # Find parameter names in the MCMC chain
        inducing_innovations_name = _find_parameter(p_names, string(p_names_k.innovations_inducing), k,
            is_multivariate_model)

        # A hyperparameter absent from the chain is only a problem if it is also not a
        # pinned constant. `_resolve_hyper_samples` returns chain samples when the parameter
        # is present and the pinned constant when it is not, so the two cases need no
        # separate branch. (Requiring BOTH to be pinned -- an earlier version -- made pinning
        # only `sigma` fall through to a zero field, which is the same defect all over again.)
        sigma_samples = _resolve_hyper_samples(
            chain, p_names, p_names_k.sigma, m.sigma, k, is_multivariate_model,
            n_samples; as_matrix = true)
        length_scale_dim = m.length_scale isa Vector ? length(m.length_scale) : 1
        length_scale_samples = _resolve_hyper_samples(
            chain, p_names, p_names_k.length_scale, m.length_scale, k,
            is_multivariate_model, n_samples; as_matrix = true)

        if isnothing(sigma_samples) || isnothing(length_scale_samples) ||
                isempty(inducing_innovations_name)
            @warn "Parameters for FITC component $(spec.key) (outcome $k) not " *
                  "resolved, and sigma / length_scale are not both pinned constants. " *
                  "Returning zero-matrix." maxlog = 1
            push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
            continue
        end

        # The only genuinely-sampled input left is the inducing innovations.
        inducing_innovations_samples = get_params_matrix(chain, inducing_innovations_name,
            m.n_inducing) # (n_samples, n_inducing)

        # Initialize the output matrix for the full effect
        effect_k_matrix = zeros(Float64, n_obs_full, n_samples)

        # --- Sample-wise Reconstruction ---
        for i in 1:n_samples # Iterate over each posterior sample
            current_sigma = sigma_samples[i, 1] # Sigma for current sample
            current_ls = length_scale_dim > 1 ? length_scale_samples[i, :] : length_scale_samples[i, 1] # Lengthscale for current sample
            current_innovations_unscaled = inducing_innovations_samples[i, :] # Unscaled innovations for inducing points for current sample
            
            # Kernel evaluations happen on the CPU
            K_UU = evaluate_kernel_matrix(Z_inducing, current_sigma, current_ls, kernel_type,
                noise)
            K_XU = evaluate_cross_kernel_matrix(coords_full, Z_inducing, current_sigma,
                current_ls, kernel_type)
            
            # Cholesky and linear solves happen on the CPU
            L_UU = cholesky(Symmetric(K_UU)).L
            u_latent = L_UU * current_innovations_unscaled
            K_UU_inv_u = K_UU \ u_latent
            mean_f = K_XU * K_UU_inv_u

            if m.method === :fitc || m.method === :dtc
                diagonal_innovations_name = _find_parameter(p_names, string(p_names_k.innovations_diagonal), k,
                    is_multivariate_model)
                
                diagonal_innovations_samples = get_params_matrix(chain, diagonal_innovations_name,
                    n_latent_train) # (n_samples, n_latent_train)
                
                # Handle prediction set by generating new innovations
                diagonal_innovations_i = if n_obs_full > n_latent_train
                    vcat(
                        diagonal_innovations_samples[i, :],
                        randn(Float64, n_obs_full - n_latent_train) # Generate new innovations for prediction points
                    )
                else
                    diagonal_innovations_samples[i, :]
                end

                # Conditional variance, via the SAME dispatcher the model body uses.
                # Writing the DTC expression out here a second time is exactly how the
                # model and its reconstruction came to disagree on the AR(1) scale.
                lambda_diag = _sparse_gp_lambda_diag(m.method, L_UU, K_XU, current_sigma)

                effect_k_matrix[:, i] = mean_f .+ sqrt.(lambda_diag .+ noise) .* diagonal_innovations_i
            else # :vfe
                effect_k_matrix[:, i] = mean_f
            end
        end
        
        push!(structured_effects, effect_k_matrix)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
