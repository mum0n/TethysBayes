"""
    RW2 <: ComponentModel

A component for a second-order random walk (RW2) model. This model assumes that the
second differences of a latent temporal field follow a random innovation. It is an
intrinsic Gaussian Markov Random Field (GMRF) with a rank deficiency of 2, implying
two sum-to-zero constraints for identifiability. It produces a smoother field than
an RW1 model.

# Version
v1.0.0

# Mathematical Summary
The RW2 model defines a latent temporal field \$\\phi\$ where the value at time \$t\$ is
a linear extrapolation from its two immediate predecessors, plus a random innovation:
\$\\phi_t | \\phi_{t-1}, \\phi_{t-2} \\sim \\mathcal{N}(2\\phi_{t-1} - \\phi_{t-2}, \\sigma^2)\$
This can be written as \$\\phi_t - 2\\phi_{t-1} + \\phi_{t-2} = \\epsilon_t\$, where
\$\\epsilon_t \\sim \\mathcal{N}(0, \\sigma^2)\$.

The joint precision matrix \$\\mathbf{Q}\$ for this process is singular (rank-deficient),
making it an "intrinsic" GMRF. To ensure the model is identifiable from a global
intercept and linear trend, two sum-to-zero constraints are imposed on the latent
field.

# Computational Methods
- `:statespace` (Default, AD-friendly): The most efficient method, constructing the RW2 process
  via a state-space recurrence relation.
- `:spectral` (AD-friendly): An efficient method using spectral decomposition of the
  precision matrix.
- `:cholesky` (AD-friendly): A didactic alternative using dense Cholesky factorization.
- `:cholesky_sparse` (Didactic, Not AD-friendly): A didactic method using sparse Cholesky
  factorization, suitable for gradient-free samplers.

# Inputs
- **Required**:
  - A temporal index variable (e.g., `year`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation of the
    innovations. Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:statespace`, `:spectral`, `:cholesky`,
    or `:cholesky_sparse`). Default: `:statespace`.

# Outputs (Parameter Names)
- `sigma_<key>`: The standard deviation of the innovations.
- `innovations_<key>`: The raw standard normal innovations for the latent field.
- `latent_<key>`: The reconstructed latent temporal field.

# Key References
- Rue, H., & Held, L. (2005). *Gaussian Markov Random Fields: Theory and
  Applications*. CRC Press.
- Wikipedia: Random walk
"""
struct RW2 <: ComponentModel
    sigma::Union{UnivariateDistribution, Real}
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:rw2] = RW2

COMPONENT_CONSTRUCTORS[:rw2] = (p, params) -> RW2(
    p.sigma, get(params, :method, :statespace)
)

MODEL_TO_STRUCTURE_MAP[:rw2] = :temporal

function get_precomputes(m::RW2, M::NamedTuple, mod_data::Dict)::NamedTuple
    # Data validation
    variables = mod_data[:variables]
    if isempty(variables)
        error("The RW2 model requires a time index variable, e.g., `random(year, model=:rw2)`.")
    end

    time_var_sym = Symbol(variables[1])
    if !hasproperty(M.data, time_var_sym)
        error("Time index variable ':$time_var_sym' for RW2 model not found in data.")
    end

    t_N = get(M, :t_N, nothing)
    if isnothing(t_N)
        error(
            "RW2 component '$(mod_data[:key])' failed: t_N not found in model " *
            "configuration. This should have been set by the model processor."
        )
    end
    
    if t_N == 0
        @warn "Number of time steps for RW2 component '$(mod_data[:key])' is zero. " *
              "The component will have no effect."
    end

    # build_structure_template returns CPU arrays
    template = build_structure_template(:rw2, t_N)
    
    # All arrays remain on the CPU.
    Q_template_cpu = template.matrix
    U_cpu = template.U
    L_cpu = template.L
    
    # Pre-compute the dense Cholesky factor for the :cholesky method on the CPU
    F_cpu = cholesky(Symmetric(Matrix(Q_template_cpu) + M.noise * I))
    
    return (
        Q_template=Q_template_cpu,
        U=U_cpu,
        L=L_cpu,
        scaling_factor=template.scaling_factor,
        n_latent=t_N,
        cholesky_factor=F_cpu
    )
end

"""
    _rw2_log_marginal_likelihood(y_residual, t_idx, t_N, Q_template, sigma, y_sigma, noise=1e-6)

Computes the exact log marginal likelihood for an RW2 process integrated out analytically.
"""
function _rw2_log_marginal_likelihood(
    y_residual::AbstractVector{T},
    t_idx::AbstractVector{Int},
    t_N::Int,
    Q_template::AbstractMatrix,
    sigma::T,
    y_sigma::T,
    noise::Real=1e-6
) where {T}
    N = length(y_residual)
    T_num = promote_type(T, typeof(noise))
    
    # Pre-accumulate observation counts and sums per time index
    N_t = zeros(T_num, t_N)
    S_t = zeros(T_num, t_N)
    for i in 1:N
        t = t_idx[i]
        if 1 <= t <= t_N
            N_t[t] += one(T_num)
            S_t[t] += y_residual[i]
        end
    end
    
    inv_sigma_y2 = one(T_num) / (y_sigma^2 + T_num(noise))
    scale = sigma^2 + T_num(noise)
    
    Q_base = Matrix{T_num}(Q_template)
    for t in 1:t_N
        Q_base[t, t] += T_num(noise) + N_t[t] * inv_sigma_y2 * scale
    end
    
    F = cholesky(Symmetric(Q_base))
    
    # Determinant term (RW2 has rank deficiency 2)
    log_det_diff = - max(t_N - 2, 1) * log(scale) - 2 * sum(log.(diag(F.U)))
    
    # Quadratic term
    b = S_t .* inv_sigma_y2
    v = F.L \ b
    quad_term = scale * dot(v, v)
    
    log_lik = - (N / 2) * log(2 * T_num(pi) * (y_sigma^2 + T_num(noise))) -
              (inv_sigma_y2 / 2) * dot(y_residual, y_residual) +
              (1 / 2) * log_det_diff +
              (1 / 2) * quad_term
              
    return log_lik
end

function get_priors(
    m::RW2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    key = spec.key
    sigma_prior_str = _prior_or_constant(p_names.sigma, m.sigma)
    if m.method == :marginalized
        return "$(sigma_prior_str)"
    else
        return """
            $(sigma_prior_str)
            $(p_names.innovations) ~ MvNormal(
                zeros(T, spec_registry[:$(key)].hyper.n_latent), I
            )
        """
    end
end

function get_updates(
    m::RW2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    inner_hyper_access = "spec_registry[:$(key)].hyper"

    statespace_code = """
        # --- RW2 Component: $(key) (State-Space Method) ---
        let
            innovations = $(p_names.innovations)
            T_num = eltype(innovations)
            n_latent = $(inner_hyper_access).n_latent
            latent_field_unscaled = similar(innovations, T_num, n_latent)
            
            if n_latent > 0
                latent_field_unscaled[1] = innovations[1]
            end
            if n_latent > 1
                latent_field_unscaled[2] = 2 * latent_field_unscaled[1] + innovations[2]
            end
            for t in 3:n_latent
                latent_field_unscaled[t] = 2 * latent_field_unscaled[t-1] - latent_field_unscaled[t-2] + innovations[t]
            end
            if n_latent > 0
                Turing.@addlogprob! logpdf(
                    Normal(0.0, 0.001 * n_latent), sum(latent_field_unscaled)
                )
            end
            
            $(p_names.latent_field) = latent_field_unscaled .* $(p_names.sigma)
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.t_idx)
        end
    """

    spectral_code = """
        # --- RW2 Component: $(key) (Spectral Method) ---
        let
            inner_hyper = $(inner_hyper_access)
            diag_D = $(p_names.sigma) ./ sqrt.(inner_hyper.L .+ M.noise)
            # Zero every null direction: the count depends on the spectrum, and a
            # disconnected graph has more than one. See _zero_null_modes!.
            _zero_null_modes!(diag_D, inner_hyper.L)
            $(p_names.latent_field) = inner_hyper.U * (diag_D .* $(p_names.innovations))
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.t_idx)
        end
    """

    cholesky_code = """
        # --- RW2 Component: $(key) (Cholesky Method) ---
        let
            F = $(inner_hyper_access).cholesky_factor
            latent_field_unscaled = F.L' \\ $(p_names.innovations)
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * spec_registry[:$(key)].hyper.n_latent),
                sum(latent_field_unscaled)
            )
            $(p_names.latent_field) = latent_field_unscaled .* $(p_names.sigma)
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.t_idx)
        end
    """

    cholesky_sparse_code = """
        # --- RW2 Component: $(key) (Sparse Cholesky Method, Not AD-Safe) ---
        let
            F = $(inner_hyper_access).cholesky_factor
            latent_field_unscaled = F.L' \\ $(p_names.innovations)
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * spec_registry[:$(key)].hyper.n_latent),
                sum(latent_field_unscaled)
            )
            $(p_names.latent_field) = latent_field_unscaled .* $(p_names.sigma)
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.t_idx)
        end
    """

    marginalized_code = """
        # --- RW2 Component: $(key) (Marginalized Analytic Normal Method) ---
        let
            N_t = $(inner_hyper_access).N_t
            S_t = $(inner_hyper_access).S_t
            n_t = $(inner_hyper_access).n_latent
            y_obs_k = $((isnothing(outcome_idx) || outcome_idx <= M.outcomes_N) ? (M.model_arch == "multivariate" ?
              "M.y_obs[:, $(outcome_idx)]" : "M.y_obs") : "zeros(M.y_N)")
            y_sig = $((isnothing(outcome_idx) || outcome_idx <= M.outcomes_N) ? "y_sigma" : "1.0")
            
            log_lik_marginalized_$(key) = _marginalized_latent_gaussian_logpdf(
                y_obs_k,
                $(inner_hyper_access).Q_template,
                $(p_names.sigma),
                y_sig,
                N_t,
                S_t,
                n_t,
                M.noise
            )
            Turing.@addlogprob! log_lik_marginalized_$(key)
        end
    """

    if m.method == :statespace
        return statespace_code
    elseif m.method == :spectral
        return spectral_code
    elseif m.method == :cholesky
        return cholesky_code
    elseif m.method == :cholesky_sparse
        return cholesky_sparse_code
    elseif m.method == :marginalized
        return marginalized_code
    else
        error("Unsupported method '$(m.method)' for RW2. Use :statespace, :spectral, :cholesky, :cholesky_sparse, or :marginalized.")
    end
end

"""
    get_effects(m::RW2, chain, spec::NamedTuple, M::NamedTuple, PS)

Reconstructs the RW2 effect from posterior samples. This version is CPU-only and
vectorized for efficiency.
"""
function get_effects(
    m::RW2, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    n_samples = _get_chain_n_samples(chain)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = string.(keys(chain))
    
    noise = M.noise

    # --- Get precomputed data ---
    hyper = spec.hyper
    n_latent_train = hyper.n_latent

    # --- Index Handling: Combine training and prediction sets ---
    t_idx_train_cpu = M.t_idx
    t_idx_full_cpu = if !isnothing(PS) && hasproperty(PS.data, :t_idx)
        vcat(t_idx_train_cpu, PS.data.t_idx)
    else
        t_idx_train_cpu
    end
    t_N_full = isempty(t_idx_full_cpu) ? 0 : maximum(t_idx_full_cpu)
    N_total = length(t_idx_full_cpu)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop ---
    for k in 1:outcomes_N
        v = generate_full_variable_names(spec, M.model_arch, k)
        sigma_samples_cpu = _resolve_hyper_samples(
            chain, p_names, v.sigma, m.sigma, k,
            is_multivariate_model, n_samples)

        if isnothing(sigma_samples_cpu)
            @warn "Parameters for $(spec.key) (outcome $k) not resolved, and " *
                  "sigma is not a pinned constant. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end


        # Initialize output matrix for the full latent field on the CPU
        effect_k_latent_cpu = zeros(Float64, t_N_full, n_samples)
        
        # --- Vectorized Reconstruction on CPU ---
        local latent_field_train_cpu
        if m.method == :marginalized
            y_sigma_name = _find_parameter(p_names, "y_sigma", k, is_multivariate_model)
            y_sigma_samples = if !isempty(y_sigma_name)
                get_params_vector(chain, y_sigma_name, 1)[:, 1]
            else
                fill(1.0, n_samples)
            end
            
            y_vec = M.y_obs isa AbstractMatrix ? M.y_obs[:, k] : M.y_obs
            
            N_t = zeros(Float64, n_latent_train)
            S_t = zeros(Float64, n_latent_train)
            for i in 1:length(t_idx_train_cpu)
                t = t_idx_train_cpu[i]
                if 1 <= t <= n_latent_train
                    N_t[t] += 1.0
                    S_t[t] += y_vec[i]
                end
            end
            
            for j in 1:n_samples
                sig = sigma_samples_cpu[j]
                y_sig = y_sigma_samples[j]
                
                scale = sig^2 + noise
                inv_sigma_y2 = 1.0 / (y_sig^2 + noise)
                
                Q_base = Matrix{Float64}(hyper.Q_template)
                for t in 1:n_latent_train
                    Q_base[t, t] += noise + N_t[t] * inv_sigma_y2 * scale
                end
                
                F = cholesky(Symmetric(Q_base))
                b = S_t .* inv_sigma_y2
                mu = scale .* (F \ b)
                
                z = randn(n_latent_train)
                x_train = mu .+ sqrt(max(scale, 1e-12)) .* (F.U \ z)
                x_train .-= mean(x_train)
                effect_k_latent_cpu[1:n_latent_train, j] = x_train
            end
        else
            innovations_name = _find_parameter(p_names, string(v.innovations), k, is_multivariate_model)
            innovations_samples_cpu = get_params_matrix(chain, innovations_name, n_latent_train)

            if m.method == :statespace
                innovations_T = innovations_samples_cpu' # [n_latent_train, n_samples]
                latent_field_unscaled_cpu = similar(innovations_T)
                if n_latent_train > 0
                    latent_field_unscaled_cpu[1, :] = innovations_T[1, :]
                end
                if n_latent_train > 1
                    latent_field_unscaled_cpu[2, :] = 2 .* latent_field_unscaled_cpu[1, :] .+ innovations_T[2, :]
                end
                for t in 3:n_latent_train
                    latent_field_unscaled_cpu[t, :] = 2 .* latent_field_unscaled_cpu[t-1, :] .- latent_field_unscaled_cpu[t-2, :] .+ innovations_T[t, :]
                end
                latent_field_centered_cpu = latent_field_unscaled_cpu .- mean(latent_field_unscaled_cpu, dims=1)
                latent_field_train_cpu = latent_field_centered_cpu .* sigma_samples_cpu'
            elseif m.method == :spectral
                U_cpu = hyper.U
                L_cpu = hyper.L
                diag_D = (sigma_samples_cpu' ./ sqrt.(L_cpu .+ noise))
                # Zero EVERY null direction (a disconnected graph has more than one).
                for _ni in _zero_null_modes!(zeros(length(L_cpu)), L_cpu)
                    diag_D[_ni, :] .= 0.0
                end
                latent_field_train_cpu = U_cpu * (diag_D .* innovations_samples_cpu')
            else # :cholesky or :cholesky_sparse
                F_cpu = hyper.cholesky_factor
                latent_field_unscaled = F_cpu.L' \ innovations_samples_cpu'
                latent_field_centered_cpu = latent_field_unscaled .- mean(latent_field_unscaled, dims=1)
                latent_field_train_cpu = latent_field_centered_cpu .* sigma_samples_cpu'
            end
            effect_k_latent_cpu[1:n_latent_train, :] = latent_field_train_cpu
        end

        # Forecasting step (vectorized over samples)
        if t_N_full > n_latent_train
            innov_pred_cpu = randn(Float32, t_N_full - n_latent_train, n_samples)
            
            for t in (n_latent_train + 1):t_N_full
                innov_t = view(innov_pred_cpu, t - n_latent_train, :)
                effect_k_latent_cpu[t, :] = 2 .* effect_k_latent_cpu[t-1, :] .- effect_k_latent_cpu[t-2, :] .+ innov_t' .* sigma_samples_cpu'
            end
        end

        # Indexing on the CPU
        indexed_effects_cpu = effect_k_latent_cpu[t_idx_full_cpu, :]
        push!(structured_effects, indexed_effects_cpu)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
 
