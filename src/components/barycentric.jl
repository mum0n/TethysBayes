"""
    Barycentric <: ComponentModel

A component for non-linear smoothing using barycentric interpolation on a 2D Delaunay
triangulation of knot points. This method is particularly well-suited for modeling
smooth spatial effects on irregular domains.

# Version
v1.0.0

# Mathematical Summary
The component models a function \$f(s)\$ where \$s\$ is a 2D coordinate.
1.  **Knot Triangulation**: A set of \$N_{\\text{knots}}\$ knot points are defined over the
    spatial domain. A Delaunay triangulation is performed on these knots to create a
    mesh of non-overlapping triangles.
2.  **Barycentric Coordinates**: For any observation point \$s_{\\text{obs}}\$, the model finds
    the triangle in the mesh that encloses it. It then computes the barycentric
    coordinates \$(\\lambda_1, \\lambda_2, \\lambda_3)\$ of \$s_{\\text{obs}}\$ with respect
      to the triangle's vertices
    \$ (v_1, v_2, v_3) \$. These coordinates are non-negative weights that sum to 1.
3.  **Basis Construction**: The barycentric coordinates form the basis functions. For
    an observation \$i\$ falling in a triangle with vertices \$ (j, k, l) \$, the
    corresponding row in the basis matrix \$B\$ will have non-zero values only at
    columns \$j, k, l\$, where \$B[i,j] = \\lambda_1\$, \$B[i,k] = \\lambda_2\$, and
      \$B[i,l] = \\lambda_3\$.
4.  **Final Effect**: The final smooth effect is a linear combination of the basis
    functions, with coefficients \$\\beta\$ (representing the latent field values at the
    knots) scaled by a standard deviation \$\\sigma\$:
    \$f(s) = (B \\cdot \\beta)(s)\$, where the prior on \$\\beta\$ depends on the chosen method.

# Computational Methods
- `:noncentered` (default): A non-centered parameterization where coefficients are
  sampled from a standard normal and scaled by `sigma`. Recommended for AD.
- `:centered`: A centered parameterization where coefficients are sampled directly
  from `N(0, sigma^2)`. Didactic, can be less efficient and not AD-friendly.
- `:gmrfsmooth`: Imposes a spatial ICAR prior on the knot coefficients, encouraging
  a smoother interpolation surface. AD-safe via spectral decomposition.

# Inputs
- **Required**:
  - Exactly two coordinate variables (e.g., `x`, `y`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `nbins`: `Int`, the approximate number of knot points to use. Default: `25`.
  - `knot_method`: `Symbol`, method for placing knots (`:quantile` or `:range`). Default:
    `:quantile`.
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation of the basis
    function coefficients. Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:noncentered`, `:centered`, `:gmrfsmooth`).
    Default: `:noncentered`.

# Outputs (Parameter Names)
- `sigma_<key>`: The standard deviation of the basis coefficients.
- `innovations_<key>`: The latent standard normal innovations for basis coefficients (for
  `:noncentered` and `:gmrfsmooth`).
- `latent_<key>`: The latent basis coefficients (for `:centered`).

# Key References
- de Berg, M., van Kreveld, M., Overmars, M., & Schwarzkopf, O. (2008).
  *Computational Geometry: Algorithms and Applications*. Springer.
- Wikipedia: Barycentric coordinate system
"""
struct Barycentric <: ComponentModel
    sigma::Union{UnivariateDistribution, Real}
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:barycentric] = Barycentric
COMPONENT_CONSTRUCTORS[:barycentric] = (p, params) -> Barycentric(
    p.sigma, get(params, :method, :noncentered)
)
MODEL_TO_STRUCTURE_MAP[:barycentric] = :smooth

"""
    get_precomputes(m::Barycentric, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes the barycentric basis matrix. For the `:gmrfsmooth` method, it also
computes the precision matrix template and spectral decomposition for the knot grid.
This version is CPU-only.
"""
function get_precomputes(
    m::Barycentric, M::NamedTuple, mod_data::Dict
)::NamedTuple
    variables = mod_data[:variables]
    if length(variables) != 2
        error("The Barycentric model requires exactly two coordinate variables, " *
              "e.g., `random(x, y, model=:barycentric)`.")
    end

    for var_sym in variables
        if !hasproperty(M.data, Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for Barycentric model not found " *
                  "in data.")
        end
    end

    coords = Matrix{Float64}(M.data[!, Symbol.(variables)])
    
    nbins = get(mod_data[:params], :nbins, 25)
    knot_method = get(mod_data[:params], :knot_method, :quantile)
    n_marginal = Int(floor(sqrt(nbins)))
    
    local kx, ky
    if knot_method == :range
        kx = collect(range(extrema(coords[:, 1])..., length=n_marginal))
        ky = collect(range(extrema(coords[:, 2])..., length=n_marginal))
    else # Default to :quantile
        kx = quantile(coords[:, 1], range(0, 1, length=n_marginal))
        ky = quantile(coords[:, 2], range(0, 1, length=n_marginal))
    end
    
    knot_points = [Point2D(x, y) for x in kx for y in ky]
    basis_matrix = TethysBayes_barycentric_basis_2D(coords, knot_points)
    n_knots = length(knot_points)

    precomputes = Dict{Symbol, Any}(
        :B => basis_matrix,
        :n_knots => n_knots,
        :knots => knot_points
    )

    if m.method == :gmrfsmooth
        W_knots = spzeros(Int, n_knots, n_knots)
        for i in 1:n_knots, j in (i+1):n_knots
            dist_sq = (knot_points[i].x - knot_points[j].x)^2 + 
                      (knot_points[i].y - knot_points[j].y)^2
            # Connect adjacent knots on the grid
            if dist_sq <= ((kx[2]-kx[1])^2 + (ky[2]-ky[1])^2) * 1.1
                W_knots[i, j] = W_knots[j, i] = 1
            end
        end
        
        template = build_structure_template(:besag, n_knots; W=W_knots)
        precomputes[:Q_template] = template.matrix
        precomputes[:U] = template.U
        precomputes[:L] = template.L
    end

    return NamedTuple(precomputes)
end

"""
    _barycentric_log_marginal_likelihood(y_residual, B_basis, Q_prior, L_eig, sigma,
      y_sigma, noise=1e-6)

Computes the exact log marginal likelihood for a Barycentric smoother with knot coefficients
  integrated out analytically.
"""
function _barycentric_log_marginal_likelihood(
    y_residual::AbstractVector{T},
    B_basis::AbstractMatrix,
    Q_prior::Union{AbstractMatrix, Nothing},
    L_eig::Union{AbstractVector, Nothing},
    sigma::T,
    y_sigma::T,
    noise::Real=1e-6
) where {T}
    N = length(y_residual)
    K = size(B_basis, 2)
    T_num = promote_type(T, typeof(noise))
    
    inv_sigma_y2 = one(T_num) / (y_sigma^2 + T_num(noise))
    scale = sigma^2 + T_num(noise)
    
    BTB = Matrix{T_num}(B_basis' * B_basis)
    BTy = Vector{T_num}(B_basis' * y_residual)
    
    Q_base = if !isnothing(Q_prior)
        Matrix{T_num}(Q_prior) .+ (scale * inv_sigma_y2) .* BTB
    else
        Matrix{T_num}(I, K, K) .+ (scale * inv_sigma_y2) .* BTB
    end
    
    for k in 1:K
        Q_base[k, k] += T_num(noise)
    end
    
    F = cholesky(Symmetric(Q_base))
    
    log_det_diff = if !isnothing(L_eig)
        # Rank-deficient GMRF
        valid_eigs = L_eig[2:end]
        log_det_prior = isempty(valid_eigs) ? zero(T_num) : sum(log.(valid_eigs .+ T_num(noise)))
        - max(K - 1, 1) * log(scale) + log_det_prior - 2 * sum(log.(diag(F.U)))
    else
        - K * log(scale) - 2 * sum(log.(diag(F.U)))
    end
    
    b = BTy .* inv_sigma_y2
    v = F.L \ b
    quad_term = scale * dot(v, v)
    
    log_lik = - (N / 2) * log(2 * T_num(pi) * (y_sigma^2 + T_num(noise))) -
              (inv_sigma_y2 / 2) * dot(y_residual, y_residual) +
              (1 / 2) * log_det_diff +
              (1 / 2) * quad_term
              
    return log_lik
end

"""
    get_priors(m::Barycentric, spec::NamedTuple, arch::String, outcome_idx, M)

Generates priors for the basis coefficients and scale, dispatching on the method.
"""
function get_priors(
    m::Barycentric, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_knots = spec.hyper.n_knots
    
    priors = ["$(p_names.sigma) ~ " * 
              "$(_distribution_to_string(m.sigma))"]

    if m.method in [:noncentered, :gmrfsmooth]
        push!(priors, "$(p_names.innovations) ~ MvNormal(zeros(T, $(n_knots)), I)")
    elseif m.method == :centered
        push!(priors, "$(p_names.latent_field) ~ MvNormal(zeros(T, $(n_knots)), I)")
    end

    return join(priors, "\n    ")
end

"""
    get_updates(m::Barycentric, spec::NamedTuple, arch::String, outcome_idx, M)

Generates code to compute the smooth effect, dispatching on the chosen method.
"""
function get_updates(
    m::Barycentric, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key

    common_code = """
        let
            hyper = spec_registry[:$(key)].hyper
            B = hyper.B
    """

    noncentered_code = """
        # --- Barycentric Component (Non-Centered): $(key) ---
        $(common_code)
            scaled_coeffs = $(p_names.innovations) .* $(p_names.sigma)
            $(p_names.latent_field) = B * scaled_coeffs
            $(eta_target) = $(eta_target) .+ $(p_names.latent_field)
        end
    """

    centered_code = """
        # --- Barycentric Component (Centered): $(key) ---
        $(common_code)
            scaled_coeffs = $(p_names.latent_field) .* $(p_names.sigma)
            $(p_names.latent_field) = B * scaled_coeffs
            $(eta_target) = $(eta_target) .+ $(p_names.latent_field)
        end
    """

    gmrfsmooth_code = """
        # --- Barycentric Component (GMRF Smooth): $(key) ---
        $(common_code)
            U = hyper.U
            L = hyper.L
            diag_D = $(p_names.sigma) ./ sqrt.(L .+ M.noise)
            # Zero every null direction: the count depends on the spectrum, and a
            # disconnected graph has more than one. See _zero_null_modes!.
            _zero_null_modes!(diag_D, L)
            coeffs = U * (diag_D .* $(p_names.innovations))
            $(p_names.latent_field) = B * coeffs
            $(eta_target) = $(eta_target) .+ $(p_names.latent_field)
        end
    """

    marginalized_code = """
        # --- Barycentric Component (Marginalized): $(key) ---
        $(common_code)
            y_residual = M.y_obs .- $(eta_target)
            Q_mat = haskey(hyper, :Q_template) ? hyper.Q_template : nothing
            L_vec = haskey(hyper, :L) ? hyper.L : nothing
            log_lik_marginalized_$(key) = _barycentric_log_marginal_likelihood(
                y_residual,
                B,
                Q_mat,
                L_vec,
                $(p_names.sigma),
                @isdefined(y_sigma) ? y_sigma : 0.0,
                M.noise
            )
            Turing.@addlogprob! log_lik_marginalized_$(key)
        end
    """

    if m.method == :noncentered
        return noncentered_code
    elseif m.method == :centered
        return centered_code
    elseif m.method == :gmrfsmooth
        return gmrfsmooth_code
    elseif m.method == :marginalized
        return marginalized_code
    else; error("Unsupported method '$(m.method)' for Barycentric component. Use :noncentered, :centered, :gmrfsmooth, or :marginalized."); end
end

"""
    get_effects(m::Barycentric, chain, spec::NamedTuple, M::NamedTuple, PS)

Reconstructs the barycentric smooth effect from posterior samples. This version is
CPU-only and uses modern chain accessors.
"""
function get_effects(
    m::Barycentric, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions ---
    n_samples = _get_chain_n_samples(chain)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = string.(keys(chain))
    noise_val = get(M, :noise, 1e-6)
    n_knots = spec.hyper.n_knots

    # --- Basis Matrix Handling (CPU only) ---
    B_train = spec.hyper.B
    B_full = if !isnothing(PS)
        coord_vars = get(spec.params, :positional_args, [])
        if all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
            coords_pred = Matrix{Float64}(PS.data[!, Symbol.(coord_vars)])
            B_pred = TethysBayes_barycentric_basis_2D(coords_pred, spec.hyper.knots)
            vcat(B_train, B_pred)
        else
            @warn "Prediction coordinates not found for Barycentric component $(spec.key). Returning effects for training data only."
            B_train
        end
    else
        B_train
    end
    n_obs_full = size(B_full, 1)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_samples = _resolve_hyper_samples(
            chain, p_names, p_names_k.sigma, m.sigma, k,
            is_multivariate_model, n_samples)

        if isnothing(sigma_samples)
            @warn "Parameters for $(spec.key) (outcome $k) not resolved, and " *
                  "sigma is not a pinned constant. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
            continue
        end

        
        # Initialize the output matrix for coefficients on the CPU
        coeffs_samples_matrix = zeros(Float64, n_knots, n_samples)
        
        if m.method == :marginalized
            y_sigma_name = _find_parameter(p_names, "y_sigma", k, is_multivariate_model)
            y_sigma_samples = if !isempty(y_sigma_name)
                get_params_vector(chain, y_sigma_name, 1)[:, 1]
            else
                fill(1.0, n_samples)
            end
            
            y_vec = M.y_obs isa AbstractMatrix ? M.y_obs[:, k] : M.y_obs
            BTB = Matrix{Float64}(B_train' * B_train)
            BTy = Vector{Float64}(B_train' * y_vec)
            Q_prior = haskey(spec.hyper,
                :Q_template) ? spec.hyper.Q_template : Matrix{Float64}(I, n_knots, n_knots)
            
            for i in 1:n_samples
                sig = sigma_samples[i]
                y_sig = y_sigma_samples[i]
                
                scale = sig^2 + noise_val
                inv_sigma_y2 = 1.0 / (y_sig^2 + noise_val)
                
                Q_base = Matrix{Float64}(Q_prior) .+ (scale * inv_sigma_y2) .* BTB
                for j in 1:n_knots
                    Q_base[j, j] += noise_val
                end
                
                F = cholesky(Symmetric(Q_base))
                b = BTy .* inv_sigma_y2
                mu = scale .* (F \ b)
                
                z = randn(n_knots)
                coeffs_samples_matrix[:, i] = mu .+ sqrt(max(scale, 1e-12)) .* (F.U \ z)
            end
        elseif m.method == :centered
            latent_field_name = _find_parameter(p_names, string(p_names_k.latent_field), k, is_multivariate_model)
            if !isempty(latent_field_name)
                # get_params_vector returns [n_samples x n_params], so we transpose it
                latent_field_samples = get_params_vector(chain, latent_field_name, n_knots)'
                coeffs_samples_matrix = latent_field_samples .* sigma_samples'
            else
                @warn "Latent coefficients (latent_field) for centered Barycentric component $(spec.key) (outcome $k) not found. Using zeros."
            end
        else # :noncentered or :gmrfsmooth
            innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)
            if !isempty(innovations_name)
                innovations_samples = get_params_vector(chain, innovations_name,
                    n_knots)' # Transpose to [n_knots x n_samples]
                
                if m.method == :noncentered
                    coeffs_samples_matrix = innovations_samples .* sigma_samples'
                else # :gmrfsmooth
                    U = spec.hyper.U
                    L = spec.hyper.L
                    for i in 1:n_samples
                        diag_D = sigma_samples[i] ./ sqrt.(L .+ noise_val)
                        # Zero every null direction: the count depends on the spectrum, and a
                        # disconnected graph has more than one. See _zero_null_modes!.
                        _zero_null_modes!(diag_D, L) # Assuming L[1] corresponds to the rank-deficient mode
                        coeffs_samples_matrix[:, i] = U * (diag_D .* innovations_samples[:, i])
                    end
                end
            else
                 @warn "Innovations (innovations) for Barycentric component $(spec.key) (outcome $k) not found. Using zeros."
            end
        end
        
        # Perform matrix multiplication on the CPU
        effect_k = B_full * coeffs_samples_matrix
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end 




# --- Helper functions for Delaunay triangulation, restored from archive/model.jl ---

function _get_barycentric_coords(p::Point2D, p1::Point2D, p2::Point2D, p3::Point2D)
    area_total = abs((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))
    if area_total < 1e-9 return nothing end
    area1 = abs((p2.x - p.x) * (p3.y - p.y) - (p3.x - p.x) * (p2.y - p.y))
    area2 = abs((p3.x - p.x) * (p1.y - p.y) - (p1.x - p.x) * (p3.y - p.y))
    area3 = abs((p1.x - p.x) * (p2.y - p.y) - (p2.x - p.x) * (p1.y - p.y))
    w_sum = area1 + area2 + area3
    if abs(w_sum - area_total) > 1e-6 return nothing end # Point is outside
    return (area1 / area_total, area2 / area_total, area3 / area_total)
end

function _is_inside_triangle(p::Point2D, p1::Point2D, p2::Point2D, p3::Point2D)
    coords = _get_barycentric_coords(p, p1, p2, p3)
    return !isnothing(coords) && all(c -> c >= -1e-9, coords)
end

function _get_circumcircle(p1::Point2D, p2::Point2D, p3::Point2D)
    D = 2 * (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))
    if abs(D) < 1e-9 return nothing, nothing end
    p1_sq, p2_sq, p3_sq = p1.x^2 + p1.y^2, p2.x^2 + p2.y^2, p3.x^2 + p3.y^2
    center_x = (p1_sq * (p2.y - p3.y) + p2_sq * (p3.y - p1.y) + 
                p3_sq * (p1.y - p2.y)) / D
    center_y = (p1_sq * (p3.x - p2.x) + p2_sq * (p1.x - p3.x) + 
                p3_sq * (p2.x - p1.x)) / D
    center = Point2D(center_x, center_y)
    radius_sq = (p1.x - center.x)^2 + (p1.y - center.y)^2
    return center, radius_sq
end

function _is_in_circumcircle(p::Point2D, p1::Point2D, p2::Point2D, p3::Point2D)
    center, radius_sq = _get_circumcircle(p1, p2, p3)
    if isnothing(center) return false end
    dist_sq = (p.x - center.x)^2 + (p.y - center.y)^2
    return dist_sq < radius_sq
end


"""
    _delaunay_triangulation(points::Vector{Point2D})

Computes the Delaunay triangulation of a set of 2D points using the Bowyer-Watson algorithm.

# Version
v1.0.0

# Arguments
- `points::Vector{Point2D}`: A vector of points to triangulate.

# Returns
- `Vector{Triangle}`: A vector of `Triangle` structs representing the Delaunay triangulation.
"""
function _delaunay_triangulation(points::Vector{Point2D})
    n = length(points)
    if n < 3
        return Triangle[]
    end

    # Determine a "super-triangle" that encloses all points.
    min_x = minimum(p.x for p in points)
    max_x = maximum(p.x for p in points)
    min_y = minimum(p.y for p in points)
    max_y = maximum(p.y for p in points)
    
    dx = max_x - min_x
    dy = max_y - min_y
    delta_max = max(dx, dy)
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2

    # Define vertices of the super-triangle, ensuring it's large enough.
    p_super1 = Point2D(mid_x - 20 * delta_max, mid_y - delta_max)
    p_super2 = Point2D(mid_x + 20 * delta_max, mid_y - delta_max)
    p_super3 = Point2D(mid_x, mid_y + 20 * delta_max)
    
    super_triangle = Triangle(n + 1, n + 2, n + 3)
    all_points = [points; p_super1; p_super2; p_super3]

    triangulation = [super_triangle]

    for i in 1:n
        point = points[i]
        bad_triangles = Vector{Triangle}()
        
        for tri in triangulation
            p1 = all_points[tri.v1]
            p2 = all_points[tri.v2]
            p3 = all_points[tri.v3]
            if _is_in_circumcircle(point, p1, p2, p3)
                push!(bad_triangles, tri)
            end
        end

        edge_counts = Dict{Tuple{Int, Int}, Int}()
        for tri in bad_triangles
            edges = [(tri.v1, tri.v2), (tri.v2, tri.v3), (tri.v3, tri.v1)]
            for edge in edges
                normalized_edge = minmax(edge[1], edge[2])
                edge_counts[normalized_edge] = get(edge_counts, normalized_edge, 0) + 1
            end
        end
        
        polygon_edges = Vector{Tuple{Int, Int}}()
        for (edge, count) in edge_counts
            if count == 1
                push!(polygon_edges, edge)
            end
        end

        filter!(t -> !(t in bad_triangles), triangulation)

        for edge in polygon_edges
            push!(triangulation, Triangle(edge[1], edge[2], i))
        end
    end

    filter!(t -> !(t.v1 > n || t.v2 > n || t.v3 > n), triangulation)

    return triangulation
end

"""
    TethysBayes_barycentric_basis_2D(coords::AbstractMatrix, knots::Vector{Point2D})

Generates a 2D barycentric basis matrix based on a Delaunay triangulation of knot points.

# Version
v1.0.0

# Arguments
- `coords`: An `N x 2` matrix of data points.
- `knots`: A vector of `Point2D` knot points (vertices for the triangulation).

# Returns
- A sparse basis matrix of size `(N, length(knots))`.
"""
function TethysBayes_barycentric_basis_2D(coords::AbstractMatrix, knots::Vector{Point2D})
    n_obs = size(coords, 1)
    n_knots = length(knots)
    B = spzeros(Float64, n_obs, n_knots)

    # 1. Perform Delaunay triangulation on the knot points
    triangles = _delaunay_triangulation(knots)
    if isempty(triangles)
        @warn "Delaunay triangulation failed or resulted in no triangles. " *
              "Returning an empty basis."
        return B
    end

    # 2. For each observation, find its enclosing triangle and barycentric coordinates
    for i in 1:n_obs
        obs_point = Point2D(coords[i, 1], coords[i, 2])
        
        for tri in triangles
            v1_idx, v2_idx, v3_idx = tri.v1, tri.v2, tri.v3
            p1, p2, p3 = knots[v1_idx], knots[v2_idx], knots[v3_idx]

            if _is_inside_triangle(obs_point, p1, p2, p3)
                bary_coords = _get_barycentric_coords(obs_point, p1, p2, p3)
                if !isnothing(bary_coords)
                    w1, w2, w3 = bary_coords
                    B[i, v1_idx] = w1
                    B[i, v2_idx] = w2
                    B[i, v3_idx] = w3
                end
                break # Found the enclosing triangle
            end
        end
    end
    return B
end
