"""
    Dynamics <: ComponentModel

A component for embedding mechanistic, process-based models within the `bstm`
framework. It simulates the evolution of a latent field over space and time
according to a user-specified differential or difference equation.

# Version
v1.0.0

# Mathematical Summary
Dynamics models describe the evolution of a latent field \$N(s, t)\$ over space \$s\$
and time \$t\$. The state at time \$t\$ is a function of the state at \$t-1\$, often
including process noise \$\\epsilon(s, t)\$.

1.  **General Form (Implicit Euler Scheme)**:
    \$N(s, t) = P^{-1} (N(s, t-1) + \\epsilon(s, t))\$
    where \$P\$ is a propagator matrix (e.g., \$I - \\text{operator} \\cdot \\Delta t\$).

2.  **Advection-Diffusion**:
    Models transport and spreading. The operator combines an advection term
    (e.g., \$v \\cdot A\$) and a diffusion term (e.g., \$D \\cdot L\$), where \$A\$ is an
    advection operator and \$L\$ is the graph Laplacian.
    \$P = I - v \\cdot A - D \\cdot L\$

3.  **Population Dynamics (Logistic, Delay-Difference, Lotka-Volterra, Leslie Matrix)**:
    These models describe population changes based on biological processes (growth,
    mortality, recruitment, competition) and can include exploitation (effort, removal).
    They are typically formulated as difference equations:
    \$N(s, t) = N(s, t-1) + \\text{Growth}(N(s, t-1)) - \\text{Exploitation}(N(s, t-1)) +
      \\epsilon(s, t)\$

    *   **Logistic**: \$N_{t+1} = N_t + r N_t (1 - N_t/K) - C_t\$
    *   **Delay-Difference**: \$N_{t+1} = (N_t - C_t)e^{-M} + R_{t+1}\$
    *   **Lotka-Volterra**: Predator-prey or competition dynamics.
    *   **Leslie Matrix**: Age-structured population dynamics.

# Computational Methods
- **`:explicit` (Default, AD-friendly)**: Uses an explicit Euler time-stepping
  scheme. This method is fully compatible with automatic differentiation (AD) and
  thus suitable for gradient-based samplers like NUTS. However, it is only
  conditionally stable and may require small time steps or strong priors on
  `velocity` and `diffusion` to prevent numerical instability.
  The update rule is:
  `u_t = u_{t-1} + dt * (v*A*u_{t-1} + D*L*u_{t-1})`

- **`:implicit` (Didactic, Not AD-friendly)**: Uses an implicit Euler time-stepping
  scheme, which is unconditionally stable and often more robust for stiff problems
  (e.g., high diffusion). This method requires solving a linear system at each
  time step, which is done via an `lu` decomposition. This decomposition is not
  differentiable, making this method incompatible with AD. It is retained as a
  didactic alternative for use with gradient-free samplers.
  The update rule is:
  `(I - dt*(v*A + D*L)) * u_t = u_{t-1}`

# Assumptions
- The spatial domain is either provided as a graph (`W`) or can be discretized
  from coordinates into a regular grid.
- The evolution is deterministic given parameters, with additive process noise.
- Exploitation data (effort, removal) can be mapped to the spatiotemporal grid.

# Best Use Case
Modeling ecological population dynamics, disease spread, or other processes where
mechanistic understanding of change over space and time is critical. It allows
estimation of physical or biological parameters within a Bayesian framework.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `s_idx`).
  - A temporal index variable (e.g., `year`).
  - Either an adjacency matrix `W` passed as a keyword argument to `@bstm`, or
    spatial coordinate variables (e.g., `x_coord`, `y_coord`) for continuous mode.
- **Optional (in `random()` call)**:
  - `model`: `String`, the specific dynamics model (`"logistic"`, `"advection_diffusion"`,
    etc.). Default: `"none"`.
  - `velocity`: `UnivariateDistribution`, prior for advection velocity. Default: `Normal(0, 0.5)`.
  - `diffusion`: `UnivariateDistribution`, prior for diffusion rate. Default: `LogNormal(-1, 1)`.
  - `sigma`: `UnivariateDistribution`, prior for process noise standard deviation. Default:
    `Exponential(1.0)`.
  - `habitat`: `Symbol` or `Vector{Float64}`, a habitat covariate influencing diffusion.
  - `resolution`: `Int`, grid resolution for continuous mode. Default: `30`.
  - `method`: `Symbol`, numerical method (`:explicit` or `:implicit`). Default: `:explicit`.
  - `intrinsic_growth_rate`, `carrying_capacity`, `natural_mortality_rate`,
    `exploitation_rate`, `alpha`, `beta`, `gamma`, `delta`: Priors for biological
    parameters, depending on the model.
  - `effort`, `removal`: `Symbol` or `Array`, data for exploitation.
  - `spatially_varying_carrying_capacity`, `spatially_varying_intrinsic_growth_rate`,
    `spatially_varying_rates`: `Bool`, flags for spatially varying parameters.

# Outputs (Parameter Names)
- `velocity_<key>`: The global advection velocity parameter.
- `diffusion_<key>`: The base diffusion parameter.
- `beta_habitat_diffusion_<key>`: The coefficient for the effect of the habitat covariate on
  diffusion (if `habitat` is provided).
- `sigma_<key>`: The marginal standard deviation of the movement process.
- `innovations_<key>`: The latent innovations driving the process.
- `r_<key>`, `K_<key>`, `M_nat_<key>`, `q_<key>`, `alpha_<key>`, `beta_<key>`,
  `gamma_<key>`, `delta_<key>`: Biological parameters, depending on the model.
- `innovations_predator_<key>`: Innovations for predator population (Lotka-Volterra).
- `log_fecundity_mean_<key>`, `sigma_fecundity_<key>`, `fecundity_raw_<key>`: Fecundity
  parameters (Leslie).
- `logit_survival_mean_<key>`, `sigma_survival_<key>`, `survival_raw_<key>`: Survival
  parameters (Leslie).
- `sigma_process_<key>`, `innov_process_<key>`: Process noise parameters (Leslie, GLV).
- `alpha_raw_<key>`: Raw interaction coefficients (GLV).
- `log_K_mean_<key>`, `K_raw_<key>`: Spatially varying K parameters.
- `log_r_mean_<key>`, `r_raw_<key>`: Spatially varying r parameters.

# Key References
- Wikle, C. K. (2003). Hierarchical Bayesian models for predicting the spread of
  ecological processes. *Ecology*, 84(6), 1382-1394.
- Knorr-Held, L. (2000). Bayesian modelling of inseparable space-time variation
  in disease risk. *Statistical Methods in Medical Research*, 9(3), 205-220.
- Wikipedia: Population dynamics
"""
struct Dynamics <: ComponentModel
    model::String
    params::Dict{Symbol, Any}
    resolution::Int

    function Dynamics(
        model::String,
        params::Dict{Symbol, Any} = Dict{Symbol, Any}(),
        resolution::Int = 30
    )
        return new(model, params, resolution)
    end
end

COMPONENT_TYPE_REGISTRY[:dynamics] = Dynamics

COMPONENT_CONSTRUCTORS[:dynamics] = (p, params) -> Dynamics(
    string(get(params, :model, "none")), params, get(params, :resolution, 30)
)

MODEL_TO_STRUCTURE_MAP[:dynamics] = :spacetime

"""
    get_precomputes(m::Dynamics, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs all data-dependent setup and pre-computation for the `Dynamics` component.
This function is the main data processing entry point for this component and is
CPU-only.

This function handles two main modes for spatial setup:
1.  **Graph Mode (Default)**: If an adjacency matrix `W` is provided in the main
    `@bstm` call, it uses the provided graph structure.
2.  **Continuous Mode**: If `W` is not provided, it discretizes the domain into a
    regular grid based on coordinate variables and the `resolution` parameter.

It also processes exploitation data (`effort`, `removal`) into a spatiotemporal
grid format and computes the necessary operator matrices (`L_template`, `A_template`).
"""
function get_precomputes(
    m::Dynamics, M::NamedTuple, mod_data::Dict
)::NamedTuple
    params = mod_data[:params]
    data = M.data
    variables = mod_data[:variables]
    
    local W, s_N, s_idx, t_idx, t_N, centroids, grid_areas

    W_provided = haskey(M, :W)

    if W_provided
        # --- Graph-based method (W is provided) ---
        if length(variables) == 1 && haskey(M, :s_idx) && haskey(M, :t_idx) && !isempty(M.s_idx) && !isempty(M.t_idx)
            W = M.W
            s_N = size(W, 1)
            s_idx = M.s_idx
            t_idx = M.t_idx
            t_N = length(unique(t_idx))
            centroids = get(M, :centroids, nothing)
        elseif length(variables) >= 2
            spatial_idx_var_sym = Symbol(variables[1])
            temporal_idx_var_sym = Symbol(variables[2])

            W = M.W
            s_N = size(W, 1)
            
            if !hasproperty(data, spatial_idx_var_sym)
                error("Spatial index variable ':$spatial_idx_var_sym' not found for graph-based dynamics.")
            end
            s_idx = data[!, spatial_idx_var_sym]
            
            if !hasproperty(data, temporal_idx_var_sym)
                error("Temporal index variable ':$temporal_idx_var_sym' not found.")
            end
            t_idx = data[!, temporal_idx_var_sym]
            t_N = length(unique(t_idx))
            centroids = get(M, :centroids, nothing)
        else
            error("Graph-based dynamics requires at least two positional arguments: a spatial index and a temporal index.")
        end
    else
        # --- Continuous/Grid-based method (W is not provided) ---
        @info "Adjacency matrix 'W' not provided for Dynamics component. Creating a regular grid from coordinates."
        
        if length(variables) < 3
            error("Continuous dynamics requires three positional arguments: x-coordinate, y-coordinate, and a temporal index.")
        end
        x_coord_sym = Symbol(variables[1])
        y_coord_sym = Symbol(variables[2])
        temporal_idx_var_sym = Symbol(variables[3])

        if !hasproperty(data, x_coord_sym) || !hasproperty(data, y_coord_sym)
            error("Coordinate variables ':$x_coord_sym' or ':$y_coord_sym' not found for continuous dynamics.")
        end
        
        res = m.resolution
        s_N = res * res

        x_coords = data[!, x_coord_sym]
        y_coords = data[!, y_coord_sym]
        grid_x = range(minimum(x_coords), maximum(x_coords), length=res)
        grid_y = range(minimum(y_coords), maximum(y_coords), length=res)
        
        W_grid = spzeros(Int, s_N, s_N)
        centroids_grid = Tuple{Float64, Float64}[]
        for c in 1:res, r in 1:res
            idx = (c-1)*res + r
            push!(centroids_grid, (grid_x[r], grid_y[c]))
            for dr in -1:1, dc in -1:1
                if dr == 0 && dc == 0
                    continue
                end
                nr, nc = r + dr, c + dc
                if 1 <= nr <= res && 1 <= nc <= res
                    n_idx = (nc-1)*res + nr
                    W_grid[idx, n_idx] = 1
                end
            end
        end
        W = W_grid
        centroids = centroids_grid

        s_idx_new = zeros(Int, nrow(data))
        for i in 1:nrow(data)
            obs_x, obs_y = x_coords[i], y_coords[i]
            best_r = searchsortedfirst(grid_x, obs_x)
            best_c = searchsortedfirst(grid_y, obs_y)
            if best_r > 1 && abs(grid_x[best_r-1] - obs_x) < abs(grid_x[best_r] - obs_x)
                best_r -= 1
            end
            if best_c > 1 && abs(grid_y[best_c-1] - obs_y) < abs(grid_y[best_c] - obs_y)
                best_c -= 1
            end
            s_idx_new[i] = (best_c-1)*res + best_r
        end
        s_idx = s_idx_new

        if !hasproperty(data, temporal_idx_var_sym)
            error("Temporal index variable ':$temporal_idx_var_sym' not found.")
        end
        t_idx = data[!, temporal_idx_var_sym]
        t_N = length(unique(t_idx))
    end

    # Process grid areas
    if haskey(params, :grid_areas)
        ga_val = params[:grid_areas]
        if ga_val isa Symbol && hasproperty(data, ga_val)
            grid_areas = data[!, ga_val]
        elseif ga_val isa AbstractVector
            grid_areas = ga_val
        else
            try
                grid_areas = Core.eval(get(M, :calling_module, Main), ga_val)
            catch
                grid_areas = ones(s_N)
            end
        end
    else
        grid_areas = ones(s_N)
    end

    # Process exploitation data (effort or removal)
    processed_params = Dict{Symbol, Any}()
    for param_base_name in [:effort, :removal]
        if haskey(params, param_base_name)
            val = params[param_base_name]
            vals_to_process = val isa Vector && !(val isa AbstractVector{<:Real}) ? val : [val]
            for (i, v) in enumerate(vals_to_process)
                storage_key = length(vals_to_process) > 1 ? Symbol("$(param_base_name)_$(i)") : param_base_name
                covariate_data = if v isa AbstractArray{<:Real}
                    v
                elseif v isa Symbol && hasproperty(data, v)
                    data[!, v]
                else
                    nothing
                end
                if !isnothing(covariate_data)
                    if ndims(covariate_data) == 1
                        cov_matrix = zeros(s_N, t_N)
                        counts = zeros(Int, s_N, t_N)
                        for obs_i in 1:M.y_N
                            si, ti = s_idx[obs_i], t_idx[obs_i]
                            cov_matrix[si, ti] += covariate_data[obs_i]
                            counts[si, ti] += 1
                        end
                        cov_matrix ./= max.(1, counts)
                        processed_params[storage_key] = cov_matrix
                    elseif ndims(covariate_data) >= 2
                        processed_params[storage_key] = covariate_data
                    end
                end
            end
        end
    end

    # Build operator templates
    L_template = build_structure_template(:besag, s_N; W=W).matrix
    A_template = if m.model in ["advection", "advection_diffusion"]
        W_dir = tril(W, -1)
        out_degree = sum(W_dir, dims=2)[:]
        D_inv = spdiagm(0 => 1.0 ./ (out_degree .+ 1e-9))
        D_inv * W_dir
    else
        spzeros(Float64, s_N, s_N)
    end
    
    effort_keys = [k for k in keys(processed_params) if startswith(string(k), "effort")]
    removal_keys = [k for k in keys(processed_params) if startswith(string(k), "removal")]

    return (
        W=W,
        s_N=s_N,
        t_N=t_N,
        s_idx=s_idx,
        t_idx=t_idx,
        centroids=centroids,
        L_template=L_template,
        A_template=A_template,
        areas=grid_areas,
        effort_keys=effort_keys,
        removal_keys=removal_keys,
        processed_params=processed_params
    )
end

"""
    generate_exploitation_block(spec, time_var)

Generates a Turing DSL code block for calculating exploitation (e.g., catch or removals)
from `effort` and `removal` parameters within a univariate `Dynamics` component.
"""
function generate_exploitation_block(spec, time_var)
    effort_keys = get(spec.hyper, :effort_keys, [])
    removal_keys = get(spec.hyper, :removal_keys, [])
    
    if isempty(effort_keys) && isempty(removal_keys)
        # If no exploitation is specified, return a zero-initialized vector.
        # This is initialized with the correct numeric type for AD safety.
        return "exploitation = similar(N_prev, eltype(N_prev), $(spec.hyper.s_N))"
    end
    
    # Initialize the exploitation vector.
    lines = ["exploitation = similar(N_prev, eltype(N_prev), $(spec.hyper.s_N))"]

    # Add exploitation from effort-based removals.
    # Assumes a catchability coefficient `q_...` is defined in the model.
    for key in effort_keys
        push!(lines, "exploitation .+= exploitation_rate_$(key) .* spec_registry[:$(spec.key)].hyper.processed_params[:$(key)][:, $(time_var)] .* N_prev")
    end

    # Add exploitation from direct removals.
    for key in removal_keys
        push!(lines, "exploitation .+= spec_registry[:$(spec.key)].hyper.processed_params[:$(key)][:, $(time_var)]")
    end

    return join(lines, "\n    ")
end

"""
    get_priors(m::Dynamics, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code string for the `Dynamics`'s priors.
"""
function get_priors(
    m::Dynamics, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    params = m.params
    priors_acc = String[]
    key_str = string(spec.key)
    s_N = spec.hyper.s_N
    t_N = spec.hyper.t_N

    if arch == "univariate"
        if m.model in ["advection", "advection_diffusion"]
            vel_prior = get(params, :velocity, Normal(0.0, 0.5))
            push!(priors_acc, "$(_prior_or_constant(p_names.velocity, vel_prior))")
        end
        if m.model in ["diffusion", "advection_diffusion"]
            diff_prior = get(params, :diffusion, LogNormal(-1.0, 1.0))
            push!(priors_acc, "$(_prior_or_constant(p_names.diffusion, diff_prior))")
        end
        
        sigma_prior = get(params, :sigma, Exponential(1.0))
        push!(priors_acc, "$(_prior_or_constant(p_names.sigma, sigma_prior))")
        
        if m.model in ["logistic", "delay_difference"]
            if get(params, :spatially_varying_intrinsic_growth_rate, false)
                intrinsic_growth_rate_mean_log_prior = get(params, :intrinsic_growth_rate_mean_log, Normal(0.0, 0.5))
                intrinsic_growth_rate_sd_prior = get(params, :intrinsic_growth_rate_sd, Exponential(1.0))
                push!(priors_acc, "sigma_r_$(_prior_or_constant(key_str, intrinsic_growth_rate_sd_prior))")
                push!(priors_acc,
                    "log_r_mean_$(_prior_or_constant(key_str, intrinsic_growth_rate_mean_log_prior))")
                push!(priors_acc, "r_unscaled_$(key_str) ~ MvNormal(zeros(T, $(s_N)), I)")
            else
                intrinsic_growth_rate_prior = get(params, :intrinsic_growth_rate, LogNormal(0.0, 1.0))
                push!(priors_acc, "$(_prior_or_constant(p_names.intrinsic_growth_rate, intrinsic_growth_rate_prior))")
            end
            if get(params, :spatially_varying_carrying_capacity, false)
                carrying_capacity_mean_log_prior = get(params, :carrying_capacity_mean_log, Normal(log(100.0), 0.5))
                carrying_capacity_sd_prior = get(params, :carrying_capacity_sd, Exponential(1.0))
                push!(priors_acc, "sigma_K_$(_prior_or_constant(key_str, carrying_capacity_sd_prior))")
                push!(priors_acc,
                    "log_K_mean_$(_prior_or_constant(key_str, carrying_capacity_mean_log_prior))")
                push!(priors_acc, "K_unscaled_$(key_str) ~ MvNormal(zeros(T, $(s_N)), I)")
            else
                carrying_capacity_prior = get(params, :carrying_capacity, LogNormal(log(100.0), 1.0))
                push!(priors_acc, "$(_prior_or_constant(p_names.carrying_capacity, carrying_capacity_prior))")
            end
            for key in spec.hyper.effort_keys
                exploitation_rate_prior = get(params, Symbol("exploitation_rate_$(key)"), LogNormal(-2.0, 1.0))
                push!(priors_acc, "exploitation_rate_$(_prior_or_constant(key, exploitation_rate_prior))")
            end
        end
        if m.model == "delay_difference"
            natural_mortality_rate_prior = get(params, :natural_mortality_rate, LogNormal(-1.0, 0.5))
            push!(priors_acc, "$(_prior_or_constant(p_names.natural_mortality_rate, natural_mortality_rate_prior))")
        end
        
        if m.model == "lotka_volterra"
            alpha_prior = get(params, :alpha, LogNormal(0.0, 0.5))
            beta_prior = get(params, :beta, LogNormal(-1.0, 0.5))
            gamma_prior = get(params, :gamma, LogNormal(-1.0, 0.5))
            delta_prior = get(params, :delta, LogNormal(0.0, 0.5))
            push!(priors_acc, "$(_prior_or_constant(p_names.alpha, alpha_prior))")
            push!(priors_acc, "$(_prior_or_constant(p_names.beta, beta_prior))")
            push!(priors_acc, "$(_prior_or_constant(p_names.gamma, gamma_prior))")
            push!(priors_acc, "$(_prior_or_constant(p_names.delta, delta_prior))")
            push!(priors_acc, "$(p_names.innovations_predator) ~ MvNormal(zeros(T, $(s_N) * $(t_N)), I)")
        end

        push!(priors_acc, "$(p_names.innovations) ~ MvNormal(zeros(T, $(s_N) * $(t_N)), I)")
    
    elseif arch == "multivariate"
        key_str = string(spec.key)
        if m.model == "leslie_matrix"
            n_classes = get(params, :n_age_classes, M.outcomes_N)
            if get(params, :spatially_varying_rates, false)
                push!(priors_acc,
                    "log_fecundity_mean_$(key_str) ~ filldist(Normal(0.0, 1.0), $(n_classes))")
                push!(priors_acc,
                    "sigma_fecundity_$(key_str) ~ filldist(Exponential(1.0), $(n_classes))")
                push!(priors_acc,
                    "fecundity_unscaled_$(key_str) ~ MvNormal(zeros(T, $(s_N) * $(n_classes)), I)")
                push!(priors_acc,
                    "logit_survival_mean_$(key_str) ~ filldist(Normal(1.5, 1.0), $(n_classes-1))")
                push!(priors_acc,
                    "sigma_survival_$(key_str) ~ filldist(Exponential(1.0), $(n_classes-1))")
                push!(priors_acc, "survival_unscaled_$(key_str) ~ MvNormal(zeros(T, $(s_N) * ($(n_classes)-1)), I)")
            else
                push!(priors_acc,
                    "survival_rates_$(key_str) ~ filldist(Beta(9.0, 1.0), $(n_classes - 1))")
                push!(priors_acc,
                    "fecundity_rates_$(key_str) ~ filldist(LogNormal(0.0, 1.0), $(n_classes))")
            end
            if get(params, :spatially_varying_carrying_capacity, false)
                push!(priors_acc, "sigma_K_$(key_str) ~ Exponential(1.0)")
                push!(priors_acc, "log_K_mean_$(key_str) ~ Normal(log(100.0), 0.5)")
                push!(priors_acc, "K_unscaled_$(key_str) ~ MvNormal(zeros(T, $(s_N)), I)")
            else
                push!(priors_acc, "K_$(key_str) ~ LogNormal(log(100.0), 1.0)")
            end
            for key in spec.hyper.effort_keys
                exploitation_rate_prior = get(params, Symbol("exploitation_rate_$(key)"), filldist(LogNormal(-4.0, 1.0),
                    n_classes))
                push!(priors_acc, "exploitation_rate_$(_prior_or_constant(key, exploitation_rate_prior))")
            end
            push!(priors_acc,
                "$(p_names.sigma_process) ~ filldist(Exponential(1.0), $(n_classes))")
            push!(priors_acc,
                "$(p_names.innovations) ~ MvNormal(zeros(T, $(s_N) * $(t_N) * $(n_classes)), I)")
        
        elseif m.model == "generalized_lotka_volterra"
            n_species = M.outcomes_N
            push!(priors_acc, "r_$(key_str) ~ filldist(LogNormal(0.0, 1.0), $(n_species))")
            push!(priors_acc, "alpha_unscaled_$(key_str) ~ MvNormal(zeros(T, $(n_species * (n_species - 1))), I)")
            if get(params, :spatially_varying_carrying_capacity, false)
                push!(priors_acc,
                    "log_K_mean_$(key_str) ~ filldist(Normal(log(100.0), 1.0), $(n_species))")
                push!(priors_acc, "sigma_K_$(key_str) ~ filldist(Exponential(1.0), $(n_species))")
                push!(priors_acc,
                    "K_unscaled_$(key_str) ~ MvNormal(zeros(T, $(s_N) * $(n_species)), I)")
            else
                push!(priors_acc,
                    "K_$(key_str) ~ filldist(LogNormal(log(100.0), 1.0), $(n_species))")
            end
            push!(priors_acc,
                "$(p_names.sigma_process) ~ filldist(Exponential(1.0), $(n_species))")
            push!(priors_acc,
                "$(p_names.innovations) ~ MvNormal(zeros(T, $(s_N) * $(t_N) * $(n_species)), I)")
        end
    end

    return join(priors_acc, "\n    ")
end

"""
    get_updates(m::Dynamics, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code string for simulating the `Dynamics`'s effect.
"""
function get_updates(
    m::Dynamics, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key_str = string(spec.key)
    params = m.params
    s_N = spec.hyper.s_N
    t_N = spec.hyper.t_N

    # Determine the numeric type for dynamic fields, ensuring AD compatibility.
    T_num_dyn = "eltype($(p_names.innovations))"

    if arch == "univariate"
        propagator_setup = ""
        if m.model in ["advection", "advection_diffusion"]
            propagator_setup *=
                "propagator = lu(I($(s_N)) - $(p_names.velocity) * " *
                "spec_registry[:$(key_str)].hyper.A_template + M.noise * I)\n"
        end
        if m.model in ["diffusion", "advection_diffusion"]
            propagator_setup *=
                "propagator = cholesky(Symmetric(I($(s_N)) - " *
                "$(p_names.diffusion) * spec_registry[:$(key_str)].hyper.L_template + M.noise * I))\n"
        end
        
        field_setup =
            "dyn_field = similar($(p_names.innovations), $(T_num_dyn), $(s_N), $(t_N))\n    " *
            "innovations_matrix = reshape($(p_names.innovations), $(s_N), $(t_N))\n    " *
            "dyn_field[:, 1] = innovations_matrix[:, 1]"
        
        K_setup_block = ""
        K_var = string(p_names.carrying_capacity)
        if get(params, :spatially_varying_carrying_capacity, false)
            K_setup_block =
                "Q_K = spec_registry[:$(key_str)].hyper.L_template\n" *
                "F_K = cholesky(Symmetric(Matrix(Q_K) + M.noise * I))\n" *
                "K_field_unscaled = F_K.L' \\ K_unscaled_$(key_str)\n" *
                "Turing.@addlogprob! logpdf(Normal(0.0,0.001 * $(s_N)), sum(K_field_unscaled))\n" *
                "K_spatial = exp.(log_K_mean_$(key_str) .+ K_field_unscaled .* sigma_K_$(key_str))"
            K_var = "K_spatial"
        end

        r_setup_block = ""
        r_var = string(p_names.intrinsic_growth_rate)
        if get(params, :spatially_varying_intrinsic_growth_rate, false)
            r_setup_block =
                "Q_r = spec_registry[:$(key_str)].hyper.L_template\n" *
                "F_r = cholesky(Symmetric(Matrix(Q_r) + M.noise * I))\n" *
                "r_field_unscaled = F_r.L' \\ r_unscaled_$(key_str)\n" *
                "Turing.@addlogprob! logpdf(Normal(0.0,0.001 * $(s_N)), sum(r_field_unscaled))\n" *
                "r_spatial = exp.(log_r_mean_$(key_str) .+ r_field_unscaled .* sigma_r_$(key_str))"
            r_var = "r_spatial"
        end
        
        propagator_logic = if m.model in ["advection", "diffusion", "advection_diffusion"]
            "dyn_field[:, t] = (propagator \\ N_intermediate) .+ innovations_matrix[:, t]"
        else
            "dyn_field[:, t] = N_intermediate .+ innovations_matrix[:, t]"
        end

        local evolution_loop_body
        if m.model == "logistic"
            exploitation_logic = generate_exploitation_block(spec, "t")
            evolution_loop_body =
                "areas = spec_registry[:$(key_str)].hyper.areas\n" *
                "for t in 2:$(t_N)\n    " *
                "N_prev = dyn_field[:, t-1]; D_prev = N_prev ./ areas; " *
                "K_density = $(K_var) ./ areas; " *
                "growth = $(r_var) .* D_prev .* (1.0 .- D_prev ./ K_density); " *
                "$(exploitation_logic); " *
                "N_intermediate = N_prev .+ (growth .* areas) .- exploitation; " *
                "$(propagator_logic); " *
                "dyn_field[:, t] = max.($(T_num_dyn)(0.0), dyn_field[:, t]);\n" *
                "end"
        elseif m.model == "delay_difference"
            exploitation_logic = generate_exploitation_block(spec, "t-1")
            evolution_loop_body =
                "areas = spec_registry[:$(key_str)].hyper.areas\n" *
                "for t in 2:$(t_N)\n    " *
                "N_prev = dyn_field[:, t-1]; D_prev = N_prev ./ areas; " *
                "K_density = $(K_var) ./ areas; " *
                "growth = $(r_var) .* D_prev .* (1.0 .- D_prev ./ K_density); " *
                "$(exploitation_logic); " *
                "N_survived = (N_prev .- exploitation) .* exp.(-$(p_names.natural_mortality_rate)); " *
                "N_intermediate = N_survived .+ (growth .* areas); " *
                "$(propagator_logic); " *
                "dyn_field[:, t] = max.($(T_num_dyn)(0.0), dyn_field[:, t]);\n" *
                "end"
        elseif m.model == "lotka_volterra"
            output_species = get(params, :output_species, :prey)
            field_setup =
                "dyn_field_prey = similar($(p_names.innovations), $(T_num_dyn), $(s_N), $(t_N)); " *
                "dyn_field_predator = similar($(p_names.innovations), $(T_num_dyn), $(s_N), $(t_N)); " *
                "innovations_matrix_prey = reshape($(p_names.innovations), $(s_N), $(t_N)); " *
                "innovations_matrix_predator = reshape($(p_names.innovations_predator), $(s_N), $(t_N)); " *
                "dyn_field_prey[:, 1] = innovations_matrix_prey[:, 1]; " *
                "dyn_field_predator[:, 1] = innovations_matrix_predator[:, 1]"
            evolution_loop_body =
                "for t in 2:$(t_N)\n    " *
                "N_prey_prev = dyn_field_prey[:, t-1]; " *
                "N_pred_prev = dyn_field_predator[:, t-1]; " *
                "d_prey = ($(p_names.alpha) .* N_prey_prev) .- ($(p_names.beta) .* N_prey_prev .* N_pred_prev); " *
                "d_pred = ($(p_names.gamma) .* N_prey_prev .* N_pred_prev) .- ($(p_names.delta) .* N_pred_prev); " *
                "dyn_field_prey[:, t] = max.($(T_num_dyn)(0.0), N_prey_prev .+ d_prey .+ innovations_matrix_prey[:, t]); " *
                "dyn_field_predator[:, t] = max.($(T_num_dyn)(0.0), N_pred_prev .+ d_pred .+ innovations_matrix_predator[:, t]);\n" *
                "end\n" *
                "dyn_field = $(output_species == :prey ? "dyn_field_prey" : "dyn_field_predator")"
        else
            evolution_loop_body =
                "for t in 2:$(t_N)\n    " *
                "dyn_field[:, t] = (propagator \\ dyn_field[:, t-1]) + innovations_matrix[:, t];\n" *
                "end"
        end

        return """
        begin
            # Dynamics model: $(m.model) for $(spec.key)
            $(K_setup_block)
            $(r_setup_block)
            $(propagator_setup)
            $(field_setup)
            $(evolution_loop_body)
            dyn_field .*= $(p_names.sigma)
            for i in 1:M.y_N
                $(eta_target)[i] += log(dyn_field[spec.hyper.s_idx[i], spec.hyper.t_idx[i]]
                  + 1.0e-6)
            end
        end
        """
    elseif arch == "multivariate"
        key_str = string(spec.key)
        if m.model == "leslie_matrix"
            n_classes = get(params, :n_age_classes, M.outcomes_N)
            spatially_varying_carrying_capacity = get(params, :spatially_varying_carrying_capacity, false)
            spatially_varying_rates = get(params, :spatially_varying_rates, false)
            
            # Inlined exploitation logic for multivariate model
            exploitation_lines = ["exploitation = similar(N_prev, T, $(n_classes));"]
            for e_key in spec.hyper.effort_keys
                push!(exploitation_lines, "exploitation .+= q_$(e_key) .* spec_registry[:$(key_str)].hyper.processed_params[Symbol(\"$(e_key)\")][s, t-1] .* N_prev;")
            end
            for r_key in spec.hyper.removal_keys
                push!(exploitation_lines, "exploitation .+= spec_registry[:$(key_str)].hyper.processed_params[Symbol(\"$(r_key)\")][s, t-1, :];")
            end
            exploitation_block = join(exploitation_lines, "\n                        ")

            return """
            begin
                Q_spatial = spec_registry[:$(key_str)].hyper.L_template;
                F_spatial = cholesky(Symmetric(Matrix(Q_spatial) + M.noise * I));
                areas = spec_registry[:$(key_str)].hyper.areas
                local survival_rates_spatial, fecundity_rates_spatial;
                if $(spatially_varying_rates)
                    fecundity_unscaled_matrix = reshape(fecundity_unscaled_$(key_str),
                      $(s_N), $(n_classes));
                    fecundity_field = F_spatial.L' \\ fecundity_unscaled_matrix;
                    fecundity_rates_spatial = exp.(log_fecundity_mean_$(key_str)' .+
                      fecundity_field .* sigma_fecundity_$(key_str)');
                    survival_unscaled_matrix = reshape(survival_unscaled_$(key_str), $(s_N),
                      $(n_classes-1));
                    survival_field = F_spatial.L' \\ survival_unscaled_matrix;
                    survival_rates_spatial =
                      LogExpFunctions.logistic.(logit_survival_mean_$(key_str)' .+
                      survival_field .* sigma_survival_$(key_str)');
                end
                local K_values_$(key_str);
                if $(spatially_varying_carrying_capacity)
                    K_field_unscaled = F_spatial.L' \\ K_unscaled_$(key_str);
                    Turing.@addlogprob! logpdf(Normal(0.0,0.001 * $(s_N)), sum(K_field_unscaled));
                    K_values_$(key_str) = exp.(log_K_mean_$(key_str) .+ K_field_unscaled .*
                      sigma_K_$(key_str));
                else
                    K_values_$(key_str) = fill(K_$(key_str), $(s_N));
                end
                innovations_tensor_$(key_str) = reshape($(p_names.innovations), $(s_N), $(t_N), $(n_classes));
                population_field_$(key_str) = similar($(p_names.innovations), T, $(s_N), $(t_N),
                  $(n_classes))
                for a in 1:$(n_classes)
                    population_field_$(key_str)[:, 1, a] = max.(0.0,
                      innovations_tensor_$(key_str)[:, 1, a] .* sigma_process_$(key_str)[a]);
                end
                for s in 1:$(s_N)
                    L_s = zeros(T, $(n_classes), $(n_classes));
                    if $(spatially_varying_rates)
                        for i in 1:($(n_classes)-1)
                            L_s[i+1, i] = survival_rates_spatial[s, i]
                        end
                        L_s[1, :] = fecundity_rates_spatial[s, :]
                    else
                        for i in 1:($(n_classes)-1)
                            L_s[i+1, i] = survival_rates_$(key_str)[i]
                        end
                        L_s[1, :] = fecundity_rates_$(key_str)
                    end
                    for t in 2:$(t_N)
                        N_prev = view(population_field_$(key_str), s, t-1, :);
                        $(exploitation_block)
                        N_after_removal = max.(0.0, N_prev - exploitation);
                        L_effective = copy(L_s)
                        if $(spatially_varying_carrying_capacity) || haskey(params, :carrying_capacity)
                            total_pop_prev = sum(N_after_removal);
                            K_density = K_values_$(key_str)[s] / areas[s];
                            dd_factor = max(0.0, 1.0 - (total_pop_prev / areas[s]) / K_density);
                            L_effective[1, :] .*= dd_factor;
                        end
                        N_projected = L_effective * N_after_removal;
                        current_innov = view(innovations_tensor_$(key_str), s, t, :) .*
                          sigma_process_$(key_str);
                        population_field_$(key_str)[s, t, :] = max.(0.0, N_projected .+
                          current_innov)
                    end
                end
                for k in 1:$(n_classes)
                    for i in 1:M.y_N
                        $(eta_target)[i, k] +=
                          log(population_field_$(key_str)[spec.hyper.s_idx[i],
                          spec.hyper.t_idx[i], k] + 1.0e-6);
                    end
                end
            end
            """
        elseif m.model == "generalized_lotka_volterra"
            n_species = M.outcomes_N
            spatially_varying_carrying_capacity = get(params, :spatially_varying_carrying_capacity, false)
            return """
            begin
                areas = spec_registry[:$(key_str)].hyper.areas;
                alpha_$(key_str) = diagm(0 => ones(T, $(n_species)));
                off_diag_indices = [i for i in 1:($(n_species)^2) if mod(i-1,
                  $(n_species)+1) != 0];
                alpha_$(key_str)[off_diag_indices] = alpha_unscaled_$(key_str)
                local K_values_$(key_str);
                if $(spatially_varying_carrying_capacity)
                    Q_spatial = spec_registry[:$(key_str)].hyper.L_template;
                    F_spatial = cholesky(Symmetric(Matrix(Q_spatial) + M.noise * I));
                    K_unscaled_matrix = reshape(K_unscaled_$(key_str), $(s_N), $(n_species));
                    K_field = F_spatial.L' \\ K_unscaled_matrix;
                    K_values_$(key_str) = exp.(log_K_mean_$(key_str)' .+ K_field .*
                      sigma_K_$(key_str)');
                else
                    K_values_$(key_str) = repeat(K_$(key_str)', $(s_N), 1);
                end
                innovations_tensor = reshape($(p_names.innovations), $(s_N), $(t_N), $(n_species));
                population_field = similar($(p_names.innovations), T, $(s_N), $(t_N), $(n_species));
                population_field[:, 1, :] = max.(0.0, innovations_tensor[:, 1, :] .*
                  sigma_process_$(key_str)')
                for s in 1:$(s_N), t in 2:$(t_N)
                    N_prev = view(population_field, s, t-1, :);
                    D_prev = N_prev ./ areas[s];
                    K_density = K_values_$(key_str)[s, :] ./ areas[s];
                    N_intermediate = similar(N_prev, T, $(n_species))
                    for i in 1:$(n_species)
                        interaction_sum_density = dot(alpha_$(key_str)[i, :], D_prev);
                        growth_density = r_$(key_str)[i] * D_prev[i] * (1.0 -
                          interaction_sum_density / K_density[i]);
                        N_intermediate[i] = N_prev[i] + growth_density * areas[s];
                    end
                    current_innov = view(innovations_tensor, s, t, :) .* sigma_process_$(key_str);
                    population_field[s, t, :] = max.(0.0, N_intermediate .+ current_innov)
                end
                for k in 1:$(n_species)
                    for i in 1:M.y_N
                        $(eta_target)[i, k] += log(population_field[spec.hyper.s_idx[i],
                          spec.hyper.t_idx[i], k] + 1.0e-6);
                    end
                end
            end
            """
        end
    end
    throw(ArgumentError(
        "Dynamics model '$(m.model)' is not supported for architecture '$(arch)'."
    ))
end

"""
    get_effects(m::Dynamics, chain, spec::NamedTuple, M::NamedTuple, PS)

Reconstructs the `Dynamics`'s effect from the MCMC chain's posterior samples.
This version is CPU-only and uses modern chain accessors.
"""
function get_effects(
    m::Dynamics, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions ---
    n_samples = _get_chain_n_samples(chain)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = string.(keys(chain))
    
    key_str = string(spec.key)
    model_type = m.model
    params = m.params
    hyper = spec.hyper

    # --- Coordinate/Index Handling: Combine training and prediction sets on CPU ---
    s_idx_full, N_total = _resolve_effect_indices(M, PS, :s_idx)
    t_idx_full, _ = _resolve_effect_indices(M, PS, :t_idx)
    t_N_full = isempty(t_idx_full) ? 0 : maximum(t_idx_full)

    L_op = hyper.L_template
    A_op = hyper.A_template
    areas = hyper.areas
    noise = M.noise
    s_N = hyper.s_N
    t_N = hyper.t_N

    # Pre-calculate flat spatiotemporal index for efficient lookups
    st_idx_full = (t_idx_full .- 1) .* s_N .+ s_idx_full

    structured_effects = Vector{Matrix{Float64}}()
    simulated_pop_tensor = nothing

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k_outcome in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k_outcome)

        if model_type in ["advection", "diffusion", "advection_diffusion"]
            sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k_outcome,
                is_multivariate_model)
            innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k_outcome,
                is_multivariate_model)
            
            rate_samples = if model_type == "advection"
                get_params_vector(chain, _find_parameter(p_names, string(p_names_k.velocity),
                    k_outcome, is_multivariate_model), 1)
            elseif model_type == "diffusion"
                get_params_vector(chain, _find_parameter(p_names, string(p_names_k.diffusion),
                    k_outcome, is_multivariate_model), 1) # (n_samples, 1)
            else # advection_diffusion
                v_samples = get_params_vector(chain, _find_parameter(p_names,
                    string(p_names_k.velocity), k_outcome, is_multivariate_model),
                    1) # (n_samples, 1)
                d_samples = get_params_vector(chain, _find_parameter(p_names,
                    string(p_names_k.diffusion), k_outcome, is_multivariate_model),
                    1) # (n_samples, 1)
                hcat(v_samples, d_samples)
            end

            if isnothing(rate_samples) || isempty(sigma_name) || isempty(innovations_name)
                @warn "Parameters for Dynamics component $(key_str) (model: $(model_type), outcome $(k_outcome)) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end

            sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
            innovations_samples = get_params_vector(chain, innovations_name, s_N * t_N)

            dyn_field_all_samples = zeros(Float64, s_N * t_N_full, n_samples)
            I_s = Matrix(I, s_N, s_N)

            for j in 1:n_samples # Iterate over each posterior sample
                innov_matrix_train = reshape(innovations_samples[j, :], s_N,
                    t_N) # Innovations for training period
                innov_matrix_full = if t_N_full > t_N
                    hcat(innov_matrix_train, randn(s_N,
                        t_N_full - t_N)) # Append random innovations for prediction
                else
                    innov_matrix_train[:, 1:t_N_full]
                end

                dyn_field_sample = zeros(Float64, s_N, t_N_full)
                
                local propagator # Propagator matrix for current sample
                if model_type == "advection"
                    propagator = lu(I_s - rate_samples[j,
                        1] * A_op + noise * I_s) # Use lu for advection
                elseif model_type == "diffusion"
                    propagator = cholesky(Symmetric(I_s - rate_samples[j, 1] * L_op + noise * I_s))
                else # advection_diffusion
                    propagator = lu(I_s - rate_samples[j, 1] * A_op - rate_samples[j,
                        2] * L_op + noise * I_s)
                end

                dyn_field_sample[:, 1] = innov_matrix_full[:, 1]
                for t in 2:t_N_full
                    dyn_field_sample[:, t] = (propagator \ dyn_field_sample[:,
                        t-1]) + innov_matrix_full[:, t]
                end
                
                dyn_field_sample .*= sigma_samples[j, 1] # Scale by sigma for current sample
                dyn_field_all_samples[:, j] = vec(dyn_field_sample)
            end
            # Take log of the dynamics field and index to observation locations
            log_dyn_field = log.(dyn_field_all_samples .+ 1e-6)
            effect_k = log_dyn_field[st_idx_full, :]
            push!(structured_effects, effect_k)

        elseif model_type == "logistic"
            sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k_outcome,
                is_multivariate_model)
            innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k_outcome,
                is_multivariate_model)
            r_name = _find_parameter(p_names, string(p_names_k.r), k_outcome,
                is_multivariate_model)
            K_name = _find_parameter(p_names, string(p_names_k.K), k_outcome,
                is_multivariate_model)
            
            if isempty(sigma_name) || isempty(innovations_name) || isempty(r_name) || isempty(K_name)
                @warn "Parameters for Dynamics component $(key_str) (model: $(model_type), outcome $(k_outcome)) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end

            sigma_samples = get_params_vector(chain, sigma_name, 1) # (n_samples, 1)
            innovations_samples = get_params_matrix(chain, innovations_name, s_N * t_N) # (n_samples, s_N * t_N)
            r_samples = get_params_vector(chain, r_name, 1) # (n_samples, 1)
            K_samples = get_params_vector(chain, K_name, 1) # (n_samples, 1)
            
            dyn_field_all_samples = zeros(Float64, s_N * t_N_full, n_samples)

            for j in 1:n_samples # Iterate over each posterior sample
                innov_matrix_train = reshape(innovations_samples[j, :], s_N,
                    t_N) # Innovations for training period
                innov_matrix_full = if t_N_full > t_N
                    hcat(innov_matrix_train, randn(s_N,
                        t_N_full - t_N)) # Append random innovations for prediction
                else
                    innov_matrix_train[:, 1:t_N_full]
                end

                dyn_field_sample = zeros(Float64, s_N, t_N_full)
                dyn_field_sample[:, 1] = innov_matrix_full[:, 1]

                for t in 2:t_N_full
                    N_prev = dyn_field_sample[:, t-1]
                    D_prev = N_prev ./ areas # Population density
                    K_density = K_samples[j, 1] ./ areas # Carrying capacity density
                    growth = r_samples[j, 1] .* D_prev .* (1.0 .- D_prev ./ K_density) # Logistic growth
                    N_intermediate = N_prev .+ (growth .* areas) # Update population
                    dyn_field_sample[:, t] = max.(0.0, N_intermediate .+ innov_matrix_full[:, t])
                end
                
                dyn_field_sample .*= sigma_samples[j, 1] # Scale by sigma for current sample
                dyn_field_all_samples[:, j] = vec(dyn_field_sample)
            end
            
            log_dyn_field = log.(dyn_field_all_samples .+ 1e-6)
            effect_k = log_dyn_field[st_idx_full, :]
            push!(structured_effects, effect_k)

        elseif model_type == "delay_difference"
            sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k_outcome,
                is_multivariate_model)
            innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k_outcome,
                is_multivariate_model)
            r_name = _find_parameter(p_names, string(p_names_k.r), k_outcome,
                is_multivariate_model)
            K_name = _find_parameter(p_names, string(p_names_k.K), k_outcome,
                is_multivariate_model)
            natural_mortality_rate_name = _find_parameter(p_names, string(p_names_k.natural_mortality_rate), k_outcome,
                is_multivariate_model)
            
            if isempty(sigma_name) || isempty(innovations_name) || isempty(r_name) || isempty(K_name)||
                isempty(natural_mortality_rate_name)
                @warn "Parameters for Dynamics component $(key_str) (model: $(model_type), outcome $(k_outcome)) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end

            sigma_samples = get_params_vector(chain, sigma_name, 1) # (n_samples, 1)
            innovations_samples = get_params_matrix(chain, innovations_name, s_N * t_N) # (n_samples, s_N * t_N)
            r_samples = get_params_vector(chain, r_name, 1) # (n_samples, 1)
            K_samples = get_params_vector(chain, K_name, 1) # (n_samples, 1)
            M_nat_samples = get_params_vector(chain, natural_mortality_rate_name, 1) # (n_samples, 1)
            
            effort_keys = spec.hyper.effort_keys
            exploitation_rate_samples_dict = Dict(key => get_params_vector(chain, _find_parameter(p_names,
                "exploitation_rate_$(key)", k_outcome, is_multivariate_model),
                1) for key in effort_keys) # (n_samples, 1)
            
            dyn_field_all_samples = zeros(Float64, s_N * t_N_full, n_samples)

            for j in 1:n_samples
                innov_matrix_train = reshape(innovations_samples[j, :], s_N, t_N)
                innov_matrix_full = if t_N_full > t_N
                    hcat(innov_matrix_train, randn(s_N, t_N_full - t_N))
                else
                    innov_matrix_train[:, 1:t_N_full]
                end

                dyn_field_sample = zeros(Float64, s_N, t_N_full)
                dyn_field_sample[:, 1] = innov_matrix_full[:, 1]

                for t in 2:t_N_full
                    N_prev = dyn_field_sample[:, t-1]
                    D_prev = N_prev ./ areas # Population density
                    K_density = K_samples[j, 1] ./ areas # Carrying capacity density
                    growth = r_samples[j, 1] .* D_prev .* (1.0 .- D_prev ./ K_density) # Logistic growth
                    
                    C_prev = zeros(Float64, s_N)
                    for e_key in effort_keys
                        effort_data = spec.hyper.processed_params[e_key][:, t-1]
                        C_prev .+= exploitation_rate_samples_dict[e_key][j, 1] .* effort_data .* N_prev # Exploitation from effort
                    end
                    for r_key in spec.hyper.removal_keys
                        removal_data = spec.hyper.processed_params[r_key][:, t-1]
                        C_prev .+= removal_data
                    end

                    N_survived = (N_prev .- C_prev) .* exp(-M_nat_samples[j])
                    N_intermediate = N_survived .+ (growth .* areas) # Update population
                    dyn_field_sample[:, t] = max.(0.0, N_intermediate .+ innov_matrix_full[:, t])
                end
                
                dyn_field_sample .*= sigma_samples[j, 1] # Scale by sigma for current sample
                dyn_field_all_samples[:, j] = vec(dyn_field_sample)
            end
            
            log_dyn_field = log.(dyn_field_all_samples .+ 1e-6)
            effect_k = log_dyn_field[st_idx_full, :]
            push!(structured_effects, effect_k)

        elseif model_type == "lotka_volterra"
            if is_multivariate_model
                @warn "Lotka-Volterra reconstruction is currently only supported for univariate models. Skipping."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end

            alpha_name = _find_parameter(p_names, string(p_names_k.alpha), k_outcome,
                is_multivariate_model)
            beta_name = _find_parameter(p_names, string(p_names_k.beta), k_outcome,
                is_multivariate_model)
            gamma_name = _find_parameter(p_names, string(p_names_k.gamma), k_outcome,
                is_multivariate_model)
            delta_name = _find_parameter(p_names, string(p_names_k.delta), k_outcome,
                is_multivariate_model)
            sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k_outcome,
                is_multivariate_model)
            innovations_prey_name = _find_parameter(p_names, string(p_names_k.innovations), k_outcome,
                is_multivariate_model)
            innovations_predator_name = _find_parameter(p_names, string(p_names_k.innovations)_predator,
                k_outcome, is_multivariate_model)
            
            if any(isempty, [alpha_name, beta_name, gamma_name, delta_name, sigma_name,
                innovations_prey_name, innovations_predator_name])
                @warn "Parameters for Dynamics component $(key_str) (model: $(model_type), outcome $(k_outcome)) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end

            alpha_samples = get_params_vector(chain, alpha_name, 1) # (n_samples, 1)
            beta_samples = get_params_vector(chain, beta_name, 1)[:, 1]
            gamma_samples = get_params_vector(chain, gamma_name, 1)[:, 1]
            delta_samples = get_params_vector(chain, delta_name, 1)[:, 1]
            sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
            innovations_prey_samples = get_params_vector(chain, innovations_prey_name, s_N * t_N)
            innovations_predator_samples = get_params_vector(chain, innovations_predator_name, s_N * t_N)
            
            output_species = get(params, :output_species, :prey)
            dyn_field_all_samples = zeros(Float64, s_N * t_N_full, n_samples)

            for j in 1:n_samples
                innov_matrix_prey_train = reshape(innovations_prey_samples[j, :], s_N, t_N)
                innov_matrix_predator_train = reshape(innovations_predator_samples[j, :], s_N, t_N)
                
                innov_matrix_prey_full = if t_N_full > t_N
                    hcat(innov_matrix_prey_train, randn(s_N, t_N_full - t_N))
                else
                    innov_matrix_prey_train[:, 1:t_N_full]
                end
                innov_matrix_predator_full = if t_N_full > t_N
                    hcat(innov_matrix_predator_train, randn(s_N, t_N_full - t_N))
                else
                    innov_matrix_predator_train[:, 1:t_N_full]
                end

                dyn_field_prey = zeros(Float64, s_N, t_N_full)
                dyn_field_predator = zeros(Float64, s_N, t_N_full)
                
                dyn_field_prey[:, 1] = innov_matrix_prey_full[:, 1]
                dyn_field_predator[:, 1] = innov_matrix_predator_full[:, 1]

                for t in 2:t_N_full
                    N_prey_prev = dyn_field_prey[:, t-1]
                    N_pred_prev = dyn_field_predator[:, t-1]
                    
                    d_prey = (alpha_samples[j, 1] .* N_prey_prev) .- (beta_samples[j,
                        1] .* N_prey_prev .* N_pred_prev)
                    d_pred = (gamma_samples[j,
                        1] .* N_prey_prev .* N_pred_prev) .- (delta_samples[j,
                        1] .* N_pred_prev)
                    
                    dyn_field_prey[:, t] = max.(0.0,
                        N_prey_prev .+ d_prey .+ innov_matrix_prey_full[:, t])
                    dyn_field_predator[:, t] = max.(0.0,
                        N_pred_prev .+ d_pred .+ innov_matrix_predator_full[:, t])
                end
                
                final_dyn_field = (output_species == :prey) ? dyn_field_prey : dyn_field_predator # Select output species
                final_dyn_field .*= sigma_samples[j, 1] # Scale by sigma for current sample
                dyn_field_all_samples[:, j] = vec(final_dyn_field)
            end

            log_dyn_field = log.(dyn_field_all_samples .+ 1e-6)
            effect_k = log_dyn_field[st_idx_full, :]
            push!(structured_effects, effect_k)

        elseif model_type == "leslie_matrix"
            if isnothing(simulated_pop_tensor)
                n_classes = get(params, :n_age_classes, outcomes_N)
                spatially_varying_carrying_capacity = get(params, :spatially_varying_carrying_capacity, false)
                spatially_varying_rates = get(params, :spatially_varying_rates, false)

                sigma_p_name = _find_parameter(
                    p_names, string(p_names_k.sigma_process), k_outcome,
                    is_multivariate_model
                )
                if isempty(sigma_p_name)
                    sigma_p_name = _find_parameter(
                        p_names, "sigma_process_$(key_str)", 1, is_multivariate_model
                    )
                end
                innovations_name = _find_parameter(
                    p_names, string(p_names_k.innovations), k_outcome, is_multivariate_model
                )

                if isempty(sigma_p_name) || isempty(innovations_name)
                    @warn "Parameters for Leslie Dynamics $(key_str) not found."
                    simulated_pop_tensor = zeros(
                        Float64, s_N, t_N_full, n_classes, n_samples
                    )
                else
                    sigma_p_samples = get_params_matrix(
                        chain, sigma_p_name, n_classes
                    )
                    innovations_samples = get_params_matrix(
                        chain, innovations_name, s_N * t_N * n_classes
                    )

                    local surv_rates_fixed, fec_rates_fixed
                    local fec_unscaled_samples, fec_mean_samples, fec_sig_samples
                    local surv_unscaled_samples, surv_mean_samples, surv_sig_samples
                    if spatially_varying_rates
                        fec_unscaled_samples = get_params_matrix(
                            chain, "fecundity_unscaled_$(key_str)", s_N * n_classes
                        )
                        fec_mean_samples = get_params_matrix(
                            chain, "log_fecundity_mean_$(key_str)", n_classes
                        )
                        fec_sig_samples = get_params_matrix(
                            chain, "sigma_fecundity_$(key_str)", n_classes
                        )
                        surv_unscaled_samples = get_params_matrix(
                            chain, "survival_unscaled_$(key_str)",
                            s_N * (n_classes - 1)
                        )
                        surv_mean_samples = get_params_matrix(
                            chain, "logit_survival_mean_$(key_str)", n_classes - 1
                        )
                        surv_sig_samples = get_params_matrix(
                            chain, "sigma_survival_$(key_str)", n_classes - 1
                        )
                    else
                        surv_name = _find_parameter(
                            p_names, "survival_rates_$(key_str)", 1,
                            is_multivariate_model
                        )
                        fec_name = _find_parameter(
                            p_names, "fecundity_rates_$(key_str)", 1,
                            is_multivariate_model
                        )
                        surv_rates_fixed = !isempty(surv_name) ?
                            get_params_matrix(chain, surv_name, n_classes - 1) :
                            fill(0.8, n_samples, n_classes - 1)
                        fec_rates_fixed = !isempty(fec_name) ?
                            get_params_matrix(chain, fec_name, n_classes) :
                            fill(1.0, n_samples, n_classes)
                    end

                    local K_samples_mat, carrying_capacity_unconstrained, carrying_capacity_mean_log, sig_K
                    if spatially_varying_carrying_capacity
                        carrying_capacity_unconstrained = get_params_matrix(
                            chain, "K_unscaled_$(key_str)", s_N
                        )
                        carrying_capacity_mean_log = get_params_vector(
                            chain, "log_K_mean_$(key_str)", 1
                        )[:, 1]
                        sig_K = get_params_vector(
                            chain, "sigma_K_$(key_str)", 1
                        )[:, 1]
                    elseif haskey(params, :carrying_capacity)
                        K_name = _find_parameter(
                            p_names, "K_$(key_str)", 1, is_multivariate_model
                        )
                        K_samples_mat = !isempty(K_name) ?
                            get_params_vector(chain, K_name, 1)[:, 1] :
                            fill(100.0, n_samples)
                    end

                    effort_keys = spec.hyper.effort_keys
                    exploitation_rate_samples_dict = Dict(
                        key => get_params_matrix(
                            chain,
                            _find_parameter(
                                p_names, "exploitation_rate_$(key)", 1, is_multivariate_model
                            ),
                            n_classes
                        ) for key in effort_keys
                    )

                    pop_tensor = zeros(
                        Float64, s_N, t_N_full, n_classes, n_samples
                    )
                    F_sp_obj = cholesky(
                        Symmetric(Matrix(hyper.L_template) + noise * I)
                    )

                    for j in 1:n_samples
                        innovations_train = reshape(
                            innovations_samples[j, :], s_N, t_N, n_classes
                        )
                        innovations_full = if t_N_full > t_N
                            cat(
                                innovations_train,
                                randn(s_N, t_N_full - t_N, n_classes),
                                dims = 2
                            )
                        else
                            innovations_train[:, 1:t_N_full, :]
                        end

                        sig_proc = sigma_p_samples[j, :]

                        for a in 1:n_classes
                            pop_tensor[:, 1, a, j] = max.(
                                0.0, innovations_full[:, 1, a] .* sig_proc[a]
                            )
                        end

                        local S_rates, F_rates
                        if spatially_varying_rates
                            fec_field = F_sp_obj.L' \ reshape(
                                fec_unscaled_samples[j, :], s_N, n_classes
                            )
                            F_rates = exp.(
                                fec_mean_samples[j, :]' .+
                                fec_field .* fec_sig_samples[j, :]'
                            )
                            surv_field = F_sp_obj.L' \ reshape(
                                surv_unscaled_samples[j, :], s_N, n_classes - 1
                            )
                            S_rates = 1.0 ./ (
                                1.0 .+ exp.-(
                                    surv_mean_samples[j, :]' .+
                                    surv_field .* surv_sig_samples[j, :]'
                                )
                            )
                        end

                        local K_spatial
                        if spatially_varying_carrying_capacity
                            K_field = F_sp_obj.L' \ carrying_capacity_unconstrained[j, :]
                            K_spatial = exp.(
                                carrying_capacity_mean_log[j] .+ K_field .* sig_K[j]
                            )
                        elseif haskey(params, :carrying_capacity)
                            K_spatial = fill(K_samples_mat[j], s_N)
                        end

                        for s in 1:s_N
                            L_s = zeros(Float64, n_classes, n_classes)
                            if spatially_varying_rates
                                for i in 1:(n_classes - 1)
                                    L_s[i+1, i] = S_rates[s, i]
                                end
                                L_s[1, :] = F_rates[s, :]
                            else
                                for i in 1:(n_classes - 1)
                                    L_s[i+1, i] = surv_rates_fixed[j, i]
                                end
                                L_s[1, :] = fec_rates_fixed[j, :]
                            end

                            for t in 2:t_N_full
                                N_prev = pop_tensor[s, t-1, :, j]
                                exploit = zeros(Float64, n_classes)
                                for e_key in effort_keys
                                    eff = spec.hyper.processed_params[
                                        Symbol(e_key)
                                    ][s, min(t-1, t_N)]
                                    exploit .+= exploitation_rate_samples_dict[e_key][j, :] .*
                                        eff .* N_prev
                                end
                                for r_key in spec.hyper.removal_keys
                                    remov = spec.hyper.processed_params[
                                        Symbol(r_key)
                                    ][s, min(t-1, t_N), :]
                                    exploit .+= remov
                                end
                                N_rem = max.(0.0, N_prev .- exploit)
                                L_eff = copy(L_s)
                                if spatially_varying_carrying_capacity || haskey(params, :carrying_capacity)
                                    tot_pop = sum(N_rem)
                                    K_dens = K_spatial[s] / areas[s]
                                    dd = max(
                                        0.0,
                                        1.0 - (tot_pop / areas[s]) / K_dens
                                    )
                                    L_eff[1, :] .*= dd
                                end
                                N_proj = L_eff * N_rem
                                innov = innovations_full[s, t, :] .* sig_proc
                                pop_tensor[s, t, :, j] = max.(
                                    0.0, N_proj .+ innov
                                )
                            end
                        end
                    end
                    simulated_pop_tensor = pop_tensor
                end
            end
            pop_k = simulated_pop_tensor[:, :, k_outcome, :]
            log_dyn_field = log.(
                reshape(pop_k, s_N * t_N_full, n_samples) .+ 1e-6
            )
            effect_k = log_dyn_field[st_idx_full, :]
            push!(structured_effects, effect_k)

        elseif model_type == "generalized_lotka_volterra"
            if isnothing(simulated_pop_tensor)
                n_species = outcomes_N
                spatially_varying_carrying_capacity = get(params, :spatially_varying_carrying_capacity, false)

                sigma_p_name = _find_parameter(
                    p_names, string(p_names_k.sigma_process), k_outcome,
                    is_multivariate_model
                )
                if isempty(sigma_p_name)
                    sigma_p_name = _find_parameter(
                        p_names, "sigma_process_$(key_str)", 1, is_multivariate_model
                    )
                end
                innovations_name = _find_parameter(
                    p_names, string(p_names_k.innovations), k_outcome, is_multivariate_model
                )

                if isempty(sigma_p_name) || isempty(innovations_name)
                    @warn "Parameters for GLV Dynamics $(key_str) not found."
                    simulated_pop_tensor = zeros(
                        Float64, s_N, t_N_full, n_species, n_samples
                    )
                else
                    sigma_p_samples = get_params_matrix(
                        chain, sigma_p_name, n_species
                    )
                    innovations_samples = get_params_matrix(
                        chain, innovations_name, s_N * t_N * n_species
                    )

                    r_samples = get_params_matrix(
                        chain, "r_$(key_str)", n_species
                    )
                    n_off = n_species * (n_species - 1)
                    alpha_unscaled_samples = get_params_matrix(
                        chain, "alpha_unscaled_$(key_str)", n_off
                    )
                    off_diag_indices = [
                        i for i in 1:(n_species^2) if mod(i-1, n_species+1) != 0
                    ]

                    local K_samples_mat, log_K_mean_mat, sig_K_mat, K_unscaled_mat
                    if spatially_varying_carrying_capacity
                        log_K_mean_mat = get_params_matrix(
                            chain, "log_K_mean_$(key_str)", n_species
                        )
                        sig_K_mat = get_params_matrix(
                            chain, "sigma_K_$(key_str)", n_species
                        )
                        K_unscaled_mat = get_params_matrix(
                            chain, "K_unscaled_$(key_str)", s_N * n_species
                        )
                    else
                        K_samples_mat = get_params_matrix(
                            chain, "K_$(key_str)", n_species
                        )
                    end

                    pop_tensor = zeros(
                        Float64, s_N, t_N_full, n_species, n_samples
                    )
                    F_sp_obj = cholesky(
                        Symmetric(Matrix(hyper.L_template) + noise * I)
                    )

                    for j in 1:n_samples
                        innovations_train = reshape(
                            innovations_samples[j, :], s_N, t_N, n_species
                        )
                        innovations_full = if t_N_full > t_N
                            cat(
                                innovations_train,
                                randn(s_N, t_N_full - t_N, n_species),
                                dims = 2
                            )
                        else
                            innovations_train[:, 1:t_N_full, :]
                        end

                        sig_proc = sigma_p_samples[j, :]

                        alpha_mat = Matrix{Float64}(I, n_species, n_species)
                        alpha_mat[off_diag_indices] = alpha_unscaled_samples[j, :]

                        local K_values
                        if spatially_varying_carrying_capacity
                            K_un_mat = reshape(
                                K_unscaled_mat[j, :], s_N, n_species
                            )
                            K_field = F_sp_obj.L' \ K_un_mat
                            K_values = exp.(
                                log_K_mean_mat[j, :]' .+
                                K_field .* sig_K_mat[j, :]'
                            )
                        else
                            K_values = repeat(K_samples_mat[j, :]', s_N, 1)
                        end

                        pop_tensor[:, 1, :, j] = max.(
                            0.0, innovations_full[:, 1, :] .* sig_proc'
                        )

                        r_j = r_samples[j, :]

                        for s in 1:s_N
                            area_s = areas[s]
                            K_dens = K_values[s, :] ./ area_s
                            for t in 2:t_N_full
                                N_prev = pop_tensor[s, t-1, :, j]
                                D_prev = N_prev ./ area_s
                                N_inter = copy(N_prev)
                                for i in 1:n_species
                                    inter_sum = dot(alpha_mat[i, :], D_prev)
                                    growth = r_j[i] * D_prev[i] * (
                                        1.0 - inter_sum / K_dens[i]
                                    )
                                    N_inter[i] += growth * area_s
                                end
                                innov = innovations_full[s, t, :] .* sig_proc
                                pop_tensor[s, t, :, j] = max.(
                                    0.0, N_inter .+ innov
                                )
                            end
                        end
                    end
                    simulated_pop_tensor = pop_tensor
                end
            end
            pop_k = simulated_pop_tensor[:, :, k_outcome, :]
            log_dyn_field = log.(
                reshape(pop_k, s_N * t_N_full, n_samples) .+ 1e-6
            )
            effect_k = log_dyn_field[st_idx_full, :]
            push!(structured_effects, effect_k)

        else
            @warn "Reconstruction for Dynamics model '$(model_type)' is " *
                  "not implemented. Returning zero effects."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
        end
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
