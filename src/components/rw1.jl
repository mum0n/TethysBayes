"""
    RW1 <: ComponentModel

A component for a first-order random walk (RW1) model. This model assumes that the
current value of a latent temporal field is the previous value plus a random
innovation. It is an intrinsic Gaussian Markov Random Field (GMRF) with a rank
deficiency of 1, implying a sum-to-zero constraint for identifiability.

# Version
v1.0.0

# Mathematical Summary
The RW1 model defines a latent temporal field \$\\phi\$ where the value at time \$t\$ is
conditionally dependent on its immediate predecessor:
\$\\phi_t | \\phi_{t-1} \\sim \\mathcal{N}(\\phi_{t-1}, \\sigma^2)\$
This can be written as \$\\phi_t - \\phi_{t-1} = \\epsilon_t\$, where
\$\\epsilon_t \\sim \\mathcal{N}(0, \\sigma^2)\$.

The joint precision matrix \$\\mathbf{Q}\$ for this process is singular (rank-deficient),
making it an "intrinsic" GMRF. To ensure the model is identifiable from a global
intercept, a sum-to-zero constraint (\$\\sum_i \\phi_i = 0\$) is imposed on the
latent field.

# Computational Methods
- `:statespace` (Default, AD-friendly): The most efficient method, constructing the RW1 process
  via a cumulative sum of innovations.
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
struct RW1 <: ComponentModel
    sigma::Union{UnivariateDistribution, Real}
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:rw1] = RW1

COMPONENT_CONSTRUCTORS[:rw1] = (p, params) -> RW1(
    p.sigma, get(params, :method, :statespace)
)

MODEL_TO_STRUCTURE_MAP[:rw1] = :temporal

function get_precomputes(m::RW1, M::NamedTuple, mod_data::Dict)::NamedTuple
    raw_vars = get(mod_data, :variables, [])
    variables = raw_vars isa AbstractVector ? raw_vars : [raw_vars]
    
    t_N = get(M, :t_N, get(M, :N_time, 0))
    if t_N == 0 && !isempty(variables)
        time_var_sym = Symbol(variables[1])
        if hasproperty(M, :data) && hasproperty(M.data, time_var_sym)
            t_N = length(unique(M.data[!, time_var_sym]))
        end
    end
    
    if t_N == 0
        t_N = 10
    end

    # build_structure_template returns CPU arrays
    template = build_structure_template(:rw1, t_N)
    
    # All arrays remain on the CPU.
    Q_template_cpu = template.matrix
    U_cpu = template.U
    L_cpu = template.L
    noise_val = get(M, :noise, 1e-6)
    
    # Pre-compute the dense Cholesky factor for the :cholesky method on the CPU
    F_cpu = cholesky(Symmetric(Matrix(Q_template_cpu) + noise_val * I))
    
    return (
        Q_template=Q_template_cpu,
        U=U_cpu,
        L=L_cpu,
        scaling_factor=template.scaling_factor,
        n_latent=t_N,
        cholesky_factor=F_cpu,
        model_type=:rw1
    )
end

"""
    _rw1_log_marginal_likelihood(y_residual, t_idx, t_N, Q_template, sigma, y_sigma, noise=1e-6)

Computes the exact log marginal likelihood for an RW1 process integrated out analytically.
"""
function _rw1_log_marginal_likelihood(
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
    
    # Determinant term
    log_det_diff = - (t_N - 1) * log(scale) - 2 * sum(log.(diag(F.U)))
    
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
    m::RW1, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
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
    m::RW1, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key

    statespace_code = """
        # --- RW1 Component: $(key) (State-Space Method) ---
        let
            latent_field_unscaled = cumsum($(p_names.innovations))
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * spec_registry[:$(key)].hyper.n_latent), 
                sum(latent_field_unscaled)
            )
            $(p_names.latent_field) = latent_field_unscaled .* $(p_names.sigma)
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.t_idx)
        end
    """

    spectral_code = """
        # --- RW1 Component: $(key) (Spectral Method) ---
        let
            hyper = spec_registry[:$(key)].hyper
            diag_D = $(p_names.sigma) ./ sqrt.(hyper.L .+ M.noise)
            # Zero every null direction: the count depends on the spectrum, and a
            # disconnected graph has more than one. See _zero_null_modes!.
            _zero_null_modes!(diag_D, hyper.L)
            $(p_names.latent_field) = hyper.U * (diag_D .* $(p_names.innovations))
            $(eta_target) = $(eta_target) .+ view($(p_names.latent_field), M.t_idx)
        end
    """

    cholesky_code = """
        # --- RW1 Component: $(key) (Cholesky Method, AD-Safe) ---
        let
            F = spec_registry[:$(key)].hyper.cholesky_factor
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
        # --- RW1 Component: $(key) (Sparse Cholesky, Not AD-Safe) ---
        let
            Q = spec_registry[:$(key)].hyper.Q_template
            F = cholesky(Symmetric(Q + M.noise * I))
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
        # --- RW1 Component: $(key) (Marginalized Method) ---
        let
            hyper = spec_registry[:$(key)].hyper
            y_residual = M.y_obs .- $(eta_target)
            log_lik_marginalized_$(key) = _rw1_log_marginal_likelihood(
                y_residual,
                M.t_idx,
                hyper.n_latent,
                hyper.Q_template,
                $(p_names.sigma),
                @isdefined(y_sigma) ? y_sigma : 0.0,
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
    else; error("Unsupported method '$(m.method)' for RW1. Use :statespace, :spectral, :cholesky, :cholesky_sparse, or :marginalized."); end
end


"""
    get_effects(m::RW1, chain, spec, M, PS)

Reconstructs the RW1 effect from posterior samples. This version is CPU-only.
"""
function get_effects(
    m::RW1, chain, spec::NamedTuple, M::NamedTuple,
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
                latent_field_unscaled_cpu = cumsum(innovations_samples_cpu', dims=1)
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
                latent_field_unscaled_cpu = F_cpu.L' \ innovations_samples_cpu'
                latent_field_centered_cpu = latent_field_unscaled_cpu .- mean(latent_field_unscaled_cpu, dims=1)
                latent_field_train_cpu = latent_field_centered_cpu .* sigma_samples_cpu'
            end
            effect_k_latent_cpu[1:n_latent_train, :] = latent_field_train_cpu
        end

        # Forecasting step (vectorized over samples)
        if t_N_full > n_latent_train
            innov_pred_cpu = randn(Float32, t_N_full - n_latent_train, n_samples)
            
            for t in (n_latent_train + 1):t_N_full
                innov_t = view(innov_pred_cpu, t - n_latent_train, :)
                effect_k_latent_cpu[t, :] = effect_k_latent_cpu[t-1, :] .+ innov_t' .* sigma_samples_cpu'
            end
        end

        # Indexing on the CPU
        indexed_effects_cpu = effect_k_latent_cpu[t_idx_full_cpu, :]
        push!(structured_effects, indexed_effects_cpu)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
 
