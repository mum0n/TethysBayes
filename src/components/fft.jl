# File: c:\home\jae\projects\TethysBayes\src\components\fft.jl
"""
    FFT <: ComponentModel

A component model for a Fourier-based smoother. This component creates a basis of
sine and cosine functions at different frequencies. The effect is a linear
combination of these basis functions, with coefficients regularized by a random
walk prior to ensure smoothness.

# Version
v1.0.0

# Mathematical Summary
The component models a smooth function \$f(x)\$ as a linear combination of Fourier
basis functions:
\$f(x) = \\sum_{j=1}^{M} \\left( \\beta_{s,j} \\sin\\left(\\frac{2\\pi j x}{\\ell}\\right) +
  \\beta_{c,j} \\cos\\left(\\frac{2\\pi j x}{\\ell}\\right) \\right)\$
where:
- \$M\$ is half the number of bins (`nbins`), representing the number of sine/cosine pairs.
- \$\\ell\$ is the `length_scale` that controls the periodicity of the basis functions.
- \$\\beta_{s,j}\$ and \$\\beta_{c,j}\$ are the Fourier coefficients.

To ensure smoothness, a penalty is applied to the coefficients, typically a
second-order random walk (RW2) prior, which penalizes deviations from a linear
trend in the coefficient space. The coefficients are sampled from a Gaussian
Markov Random Field (GMRF) with a precision matrix derived from an RW2 penalty,
scaled by \$\\sigma^2\$.

# Computational Methods
- `:spectral` (Default, AD-friendly): Regularizes coefficients using a spectral
  decomposition of the RW2 penalty matrix. This method is fully compatible with
  automatic differentiation (AD) and thus suitable for gradient-based samplers
  like NUTS.
- `:cholesky` (AD-friendly): Uses a dense Cholesky factorization of the RW2
  penalty matrix, computed on-the-fly. This method is also AD-compatible.
- `:cholesky_sparse` (Didactic, Not AD-friendly): Uses sparse Cholesky
  factorization, computed on-the-fly. This method is generally not AD-compatible.

# Inputs
- **Required**:
  - One or more coordinate variables (e.g., `x`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `nbins`: `Int`, the total number of basis functions (sine/cosine pairs). Default: `20`.
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation of the Fourier
    coefficients. Default: `Exponential(1.0)`.
  - `length_scale`: `UnivariateDistribution` or `Vector{<:UnivariateDistribution}`,
    prior for the length_scale(s) controlling the periodicity. Default: `LogNormal(0, 1)`.
  - `method`: `Symbol`, computational method (`:spectral`, `:cholesky`, `:cholesky_sparse`).
    Default: `:spectral`.

# Outputs (Parameter Names)
- `sigma_<key>`: The standard deviation of the Fourier coefficients.
- `ls_<key>`: The length_scale(s) controlling the periodicity of the basis functions.
- `innovations_<key>`: The raw standard normal innovations for the Fourier coefficients.
- `latent_<key>`: The reconstructed latent smooth effect.
"""
struct FFT <: ComponentModel
    sigma::Union{Distribution, Real}
    nbins::Int
    length_scale::Union{Distribution, Vector{<:Distribution}}
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:fft] = FFT

COMPONENT_CONSTRUCTORS[:fft] = (p, params) -> FFT(
    p.sigma,
    get(params, :nbins, 20),
    get(p, :length_scale, LogNormal(0.0, 1.0)), # Default prior for length_scale
    get(params, :method, :spectral)
)

MODEL_TO_STRUCTURE_MAP[:fft] = :smooth

"""
    get_precomputes(m::FFT, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs all data-dependent setup and pre-computation for the `FFT` component.
This function validates that coordinate variables exist, extracts them, and then
pre-computes the penalty matrix and its spectral decomposition for the Fourier
coefficients. This is a CPU-only implementation.
"""
function get_precomputes(m::FFT, M::NamedTuple, mod_data::Dict)::NamedTuple
    variables = mod_data[:variables]
    if isempty(variables)
        error("The FFT model requires coordinate variables, e.g., `random(x, model=:fft)`.")
    end

    for var_sym in variables
        if !hasproperty(M.data, Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for FFT model not found in data.")
        end
    end

    # Ensure coords are on CPU for initial processing
    coords_cpu = Matrix{Float64}(M.data[!, Symbol.(variables)])
    
    n_latent = m.nbins
    n_dims = size(coords_cpu, 2)

    # Determine nbins_per_dim for tensor product basis
    nbins_per_dim = fill(round(Int, n_latent^(1/n_dims)), n_dims)
    while prod(nbins_per_dim) < n_latent
        nbins_per_dim[1] += 1
    end

    # The RW2 penalty is commonly used for Fourier coefficients to ensure smoothness
    # build_structure_template returns CPU arrays
    template = build_structure_template(:rw2, n_latent)
    Q_template_cpu = template.matrix
    
    rank_deficiency = 2 # RW2 penalty has a rank deficiency of 2
    eig_decomp = eigen(Symmetric(Matrix(Q_template_cpu)))
    U_cpu = eig_decomp.vectors
    L_cpu = eig_decomp.values
    scaling_factor = _compute_scaling_factor(L_cpu, rank_deficiency)
    
    Q_template_scaled_cpu = Q_template_cpu ./ scaling_factor
    L_scaled_cpu = L_cpu ./ scaling_factor

    # All pre-computed arrays are kept on the CPU.
    return (
        coords = coords_cpu,
        nbins_per_dim = nbins_per_dim,
        Q_template = Q_template_scaled_cpu,
        scaling_factor = scaling_factor,
        U = U_cpu,
        L = L_scaled_cpu,
        n_latent = n_latent
    )
end

"""
    get_priors(m::FFT, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for `sigma`, `length_scale`, and the `innovations` coefficients.
"""
function get_priors(
    m::FFT, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    priors = String[]
    push!(priors, "$(_prior_or_constant(p_names.sigma, m.sigma))")

    if m.length_scale isa Vector
        length_scale_priors_str = join([_distribution_to_string(p) for p in m.length_scale], ", ")
        push!(priors, "$(p_names.length_scale) ~ Product([$(length_scale_priors_str)])")
    else
        length_scale_prior_str = _distribution_to_string(m.length_scale)
        push!(priors, "$(p_names.length_scale) ~ $(length_scale_prior_str)")
    end
    
    push!(
        priors,
        "$(p_names.innovations) ~ MvNormal(zeros(T, $(spec.hyper.n_latent)), I)"
    )

    return join(priors, "\n    ")
end

"""
    TethysBayes_fourier_basis(coords, nbins_per_dim, length_scale)

Helper function to generate a tensor product Fourier basis matrix. This is a CPU-only
implementation.
"""
function TethysBayes_fourier_basis(
    coords::AbstractMatrix, nbins_per_dim::Vector{Int},
    length_scale::Union{Real, AbstractVector}
)
    n_obs, n_dims = size(coords)
    
    length_scale_vec = if length_scale isa Real
        fill(Float64(length_scale), n_dims)
    else
        if length(length_scale) != n_dims
            error("Length of length_scale vector must match coordinate dimensions.")
        end
        length_scale
    end

    basis_matrices_1D = []
    for i in 1:n_dims
        vals = coords[:, i]
        ls_val = length_scale_vec[i]
        t_coords = vals ./ ls_val
        n_basis_1d = nbins_per_dim[i]
        B_1d = similar(t_coords, n_obs, n_basis_1d)
        
        idx = 1
        for m in 1:div(n_basis_1d, 2)
            arg = (2.0 * pi * m) .* t_coords
            B_1d[:, idx] = sin.(arg)
            B_1d[:, idx+1] = cos.(arg)
            idx += 2
        end
        if isodd(n_basis_1d) && idx <= n_basis_1d
            m = div(n_basis_1d, 2) + 1
            arg = (2.0 * pi * m) .* t_coords
            B_1d[:, idx] = sin.(arg)
        end
        push!(basis_matrices_1D, B_1d)
    end
    
    if isempty(basis_matrices_1D)
        return similar(coords, n_obs, 0)
    end

    # Combine 1D basis matrices using broadcasting for a row-wise Kronecker product
    B_final = basis_matrices_1D[1]
    for i in 2:n_dims
        B_next = basis_matrices_1D[i]
        n_obs_i, n_cols_final = size(B_final)
        _, n_cols_next = size(B_next)
        
        B_final_reshaped = reshape(B_final, n_obs_i, n_cols_final, 1)
        B_next_reshaped = reshape(B_next, n_obs_i, 1, n_cols_next)
        
        tensor_prod = B_final_reshaped .* B_next_reshaped
        B_final = reshape(tensor_prod, n_obs_i, n_cols_final * n_cols_next)
    end
    
    return B_final
end

"""
    get_updates(m::FFT, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for the `FFT` component. This is a CPU-only implementation.
"""
function get_updates(
    m::FFT, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    n_latent = spec.hyper.n_latent
    
    common_basis_code = """
        B_fft = TethysBayes_fourier_basis(
            spec_registry[:$(key)].hyper.coords,
            spec_registry[:$(key)].hyper.nbins_per_dim,
            $(p_names.length_scale)
        )
    """

    spectral_code = """
        # --- FFT Smoother Component (Spectral): $(key) ---
        let
            $(common_basis_code)
            hyper = spec_registry[:$(key)].hyper
            
            # Construct diag_D on the CPU
            diag_D = $(p_names.sigma) ./ sqrt.(hyper.L .+ M.noise)
            # Zero every null direction: the count depends on the spectrum, and a
            # disconnected graph has more than one. See _zero_null_modes!.
            #
            # The argument MUST be `hyper.L` here, not a bare `L`. This is the generated
            # model body, where the eigenvalues are reached through `hyper`; the `get_effects`
            # reconstruction below binds a local `L = hyper.L` instead. An automated rewrite
            # assumed the two shared a name, got them backwards, and the result was
            # `UndefVarError: L not defined in _GeneratedModelRuntime` -- i.e. `fft` could not
            # be sampled at all.
            _zero_null_modes!(diag_D, hyper.L)
            
            coeffs = hyper.U * (diag_D .* $(p_names.innovations))
            $(p_names.latent_field) = B_fft * coeffs
            
            $(eta_target) = $(eta_target) .+ $(p_names.latent_field)
        end
    """

    cholesky_code = """
        # --- FFT Smoother Component (Cholesky, AD-Safe): $(key) ---
        let
            $(common_basis_code)
            Q_penalty = spec_registry[:$(key)].hyper.Q_template
            F = cholesky(Symmetric(Matrix(Q_penalty) + M.noise * I))
            
            coeffs_unscaled = F.L' \\ $(p_names.innovations)
            
            # Apply soft sum-to-zero constraint for RW2 penalty
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * $(n_latent)), sum(coeffs_unscaled)
            )
            
            coeffs = $(p_names.sigma) .* coeffs_unscaled
            $(p_names.latent_field) = B_fft * coeffs
            
            $(eta_target) = $(eta_target) .+ $(p_names.latent_field)
        end
    """

    cholesky_sparse_code = """
        # --- FFT Smoother Component (Sparse Cholesky, Not AD-Safe): $(key) ---
        let
            $(common_basis_code)
            Q_penalty = spec_registry[:$(key)].hyper.Q_template
            F = cholesky(Symmetric(Q_penalty + M.noise * I))
            
            coeffs_unscaled = F.L' \\ $(p_names.innovations)
            
            # Apply soft sum-to-zero constraint for RW2 penalty
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * $(n_latent)), sum(coeffs_unscaled)
            )
            
            coeffs = $(p_names.sigma) .* coeffs_unscaled
            $(p_names.latent_field) = B_fft * coeffs
            
            $(eta_target) = $(eta_target) .+ $(p_names.latent_field)
        end
    """

    if m.method == :spectral
        return spectral_code
    elseif m.method == :cholesky
        return cholesky_code
    elseif m.method == :cholesky_sparse
        return cholesky_sparse_code
    else; error("Unsupported method '$(m.method)' for FFT component."); end
end

"""
    get_effects(m::FFT, chain, spec::NamedTuple, M::NamedTuple, PS)

Reconstructs the `FFT` component's effect from posterior samples, dispatching on
the method used during sampling. This is a CPU-only implementation.
"""
function get_effects(
    m::FFT, chain, spec::NamedTuple, M::NamedTuple,
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
    n_latent = hyper.n_latent
    nbins_per_dim = hyper.nbins_per_dim

    # --- Coordinate Handling: Combine training and prediction sets on CPU ---
    coords_train_cpu = hyper.coords
    coord_vars = get(spec.params, :positional_args, [])
    coords_full_cpu = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        vcat(coords_train_cpu, Matrix{Float64}(PS.data[!, Symbol.(coord_vars)]))
    else
        coords_train_cpu
    end
    N_total = size(coords_full_cpu, 1)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        length_scale_name = _find_parameter(p_names, string(p_names_k.length_scale), k, is_multivariate_model)
        innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)

        # Only `sigma` is pinnable here: `length_scale` is
        # `Union{Distribution, Vector{<:Distribution}}` and `innovations` is always sampled,
        # so neither can carry a fixed value. A pinned `sigma` is not a chain parameter, so
        # resolve it with the constant fallback rather than reading it by name.
        sigma_samples_cpu = _resolve_hyper_samples(
            chain, p_names, p_names_k.sigma, m.sigma, k, is_multivariate_model, n_samples)

        if isnothing(sigma_samples_cpu) || isempty(length_scale_name) || isempty(innovations_name)
            @warn "Parameters for FFT component $(spec.key) (outcome $k) not " *
                  "resolved, and sigma is not a pinned constant. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        # Extract posterior samples (these are on the CPU)
        length_scale_dim = m.length_scale isa Vector ? length(m.length_scale) : 1
        ls_samples_cpu = get_params_matrix(chain, length_scale_name, length_scale_dim)
        innovations_samples_cpu = get_params_matrix(chain, innovations_name, n_latent)

        # Initialize the output matrix for the full effect on the CPU
        effect_k_cpu = zeros(Float64, N_total, n_samples)

        # --- Sample-wise Reconstruction ---
        for i in 1:n_samples
            # 1. Generate basis matrix on CPU
            current_ls_cpu = length_scale_dim > 1 ? ls_samples_cpu[i, :] : ls_samples_cpu[i, 1]
            B_fft_i_cpu = TethysBayes_fourier_basis(
                coords_full_cpu, nbins_per_dim, current_ls_cpu
            )
            
            innov_i_cpu = innovations_samples_cpu[i, :]
            sigma_i_cpu = sigma_samples_cpu[i]
            
            # 3. Reconstruct coefficients on CPU
            local coeffs_cpu
            if m.method == :spectral
                U = hyper.U
                L = hyper.L
                diag_D = sigma_i_cpu ./ sqrt.(L .+ noise)
                # Zero every null direction: the count depends on the spectrum, and a
                # disconnected graph has more than one. See _zero_null_modes!.
                _zero_null_modes!(diag_D, hyper.L)
                coeffs_cpu = U * (diag_D .* innov_i_cpu)
            else # :cholesky or :cholesky_sparse
                Q_penalty = hyper.Q_template
                F = cholesky(Symmetric(Matrix(Q_penalty) + noise * I))
                coeffs_unscaled = F.L' \ innov_i_cpu
                coeffs_centered = coeffs_unscaled .- mean(coeffs_unscaled)
                coeffs_cpu = sigma_i_cpu .* coeffs_centered
            end
            # 4. Compute effect for this sample
            effect_k_cpu[:, i] = B_fft_i_cpu * coeffs_cpu
        end
        
        push!(structured_effects, effect_k_cpu)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
 
