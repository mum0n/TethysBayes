





"""

    model.jl


Core formula compilation, AST decomposition, technical primitive resolution,
dynamic Turing code generator, and sampling execution engine for Bayesian Spatio-Temporal
  Models (TethysBayes).
Version: v1.0.0

"""



function Base.:|>(m1::Component, m2::Component)

    return Composed([m1, m2], :pipe)

end



composition(m1::Component, m2::Component) = Composed([m1, m2], :composition)

∘(m1::Component, m2::Component) = Composed([m1, m2], :composition)



otimes(m1::Component, m2::Component) = Composed([m1, m2], :kronecker_product)

⊗(m1::Component, m2::Component) = Composed([m1, m2], :kronecker_product)



"""

    apply_transformation(func_name::Symbol, data_vector::AbstractVector;

                         offset::Union{Real, Nothing} = nothing, kwargs...)



Applies a specified mathematical transformation to a data vector.



# Mathematical Formulations

- **`:zscore`**: \$ z = \\frac{x - \\text{mean}(x)}{\\text{std}(x)} \$

- **`:log`**:

  - For strictly positive data (\$\\min(x) > 0\$) without an explicit offset: computes \$ \\log(x) \$.

  - For non-negative data containing zeros (\$\\min(x) = 0\$) without an explicit offset:
    uses standard pseudocount \$ \\log(x + 1) = \\text{log1p}(x) \$.
  - For user-specified `offset`: computes \$ \\log(x + \\text{offset}) \$.
  - For data with negative values without an offset: raises an informative `ArgumentError`.
- **`:center`**: \$ x - \\text{mean}(x) \$
- **`:scale`**: \$ \\frac{x - \\min(x)}{\\max(x) - \\min(x)} \$



# Arguments

- `func_name::Symbol`: The transformation to apply (`:zscore`, `:log`, `:center`, `:scale`).

- `data_vector::AbstractVector`: The input data column.

- `offset::Union{Real, Nothing}`: Optional user-specified scalar offset for `:log`.
- `kwargs...`: Additional keyword arguments forwarded to the transformation.

# Returns
- `AbstractVector`: The transformed vector.
"""

function apply_transformation(

    func_name::Symbol,

    data_vector::AbstractVector;
    offset::Union{Real, Nothing} = nothing,
    kwargs...
)
    if func_name == :zscore
        return standardize(ZScoreTransform, data_vector)
    elseif func_name == :log
        valid_vals = collect(skipmissing(data_vector))
        if isempty(valid_vals)
            return log.(data_vector)
        end

        min_val = minimum(valid_vals)
        effective_offset = if !isnothing(offset)
            Float64(offset)
        elseif min_val > 0.0
            0.0
        elseif min_val == 0.0
            1.0
        else
            throw(ArgumentError(
                "Cannot apply :log transformation to data containing negative values " *
                "(minimum = $(min_val)). Please provide an explicit positive offset shift, " *
                "e.g. `log(x, offset=$(ceil(abs(min_val) + 1.0)))`."
            ))
        end

        if min_val + effective_offset <= 0.0
            throw(ArgumentError(
                "Log transformation offset ($(effective_offset)) is insufficient for " *
                "data minimum ($(min_val)). The shifted minimum must be strictly positive (> 0), " *
                "but got $(min_val + effective_offset)."
            ))
        end

        

        if effective_offset == 0.0

            return log.(data_vector)

        elseif effective_offset == 1.0
            return log1p.(data_vector)
        else
            return log.(data_vector .+ effective_offset)
        end
    elseif func_name == :center
        valid_vals = collect(skipmissing(data_vector))
        return isempty(valid_vals) ? data_vector : data_vector .- mean(valid_vals)
    elseif func_name == :scale
        return standardize(UnitRangeTransform, data_vector)
    else
        @warn "Unknown transformation function ':$func_name'. Returning original data."
        return data_vector
    end
end







"""

    _rewrite_transformations!(nodes::Vector, data::DataFrame)



Recursively traverses the formula's Abstract Syntax Tree (AST), applies data
transformations defined by pipe operators (`|>`), and rewrites the AST nodes to
use the newly created data columns.
# Version
v1.0.0

# Arguments
- `nodes::Vector`: A vector of AST nodes to be processed.
- `data::DataFrame`: The input data frame. **This argument is mutated** by adding
  new columns for the transformed data.

# Returns
- `Vector`: A new vector of rewritten AST nodes.
"""
function _rewrite_transformations!(nodes::Vector, data::DataFrame)
    new_nodes = []
    for node in nodes
        if hasproperty(node,
            :type) && node.type == :operator && node.op == :pipe && length(node.children) == 2
            lhs, rhs = node.children
            if hasproperty(lhs, :module_type) && lhs.module_type in TRANSFORMATION_FUNCTIONS
                # This is a transformation pipe, e.g., `zscore(temp) |> fixed()`
                transform_func_name = lhs.module_type
                vars_to_transform = get(lhs.args, :positional_args, [])
                if isempty(vars_to_transform)
                    @warn "Transformation module '$(transform_func_name)' called without a variable. Skipping."
                    push!(new_nodes, rhs) # Keep the RHS component
                    continue
                end
                var_sym = vars_to_transform[1]



                if !hasproperty(data, var_sym)

                    error("Variable ':$var_sym' for transformation not found in data.")

                end

                

                # Extract any optional keyword arguments passed to the transformation (e.g. offset)
                transform_kwargs = Dict{Symbol, Any}()
                for (k, v) in lhs.args
                    if k != :positional_args
                        transform_kwargs[k] = v
                    end
                end
                # Apply transformation and add new column to the DataFrame
                original_data = data[!, var_sym]
                transformed_data = apply_transformation(transform_func_name, original_data;
                                                        transform_kwargs...)
                # Validation of transformed_data (Items 4 & 7)
                if !(transformed_data isa AbstractVector)
                    throw(ArgumentError(
                        "Transformation '$(transform_func_name)' on column ':$var_sym' " *
                        "must return an AbstractVector, but returned $(typeof(transformed_data))."
                    ))
                end
                if length(transformed_data) != nrow(data)
                    throw(DimensionMismatch(
                        "Transformation '$(transform_func_name)' on column ':$var_sym' returned " *
                        "$(length(transformed_data)) elements, but expected $(nrow(data)) elements " *
                        "to match DataFrame row count."
                    ))
                end
                # Check for unexpected NaN / Inf introduced by transformation
                orig_finite = all(x -> ismissing(x) || (x isa Real && isfinite(x)), original_data)
                if orig_finite
                    bad_nan = any(x -> !ismissing(x) && x isa Real && isnan(x), transformed_data)
                    bad_inf = any(x -> !ismissing(x) && x isa Real && isinf(x), transformed_data)
                    if bad_nan || bad_inf
                        bad_type = bad_nan ? "NaN" : "Inf"
                        throw(ArgumentError(
                            "Transformation '$(transform_func_name)' on column ':$var_sym' produced " *
                            "invalid $(bad_type) values from valid finite data. Please inspect the input " *
                            "values or provide appropriate offset/parameters."
                        ))
                    end
                end
                new_col_name = Symbol("$(var_sym)_$(transform_func_name)")
                if hasproperty(data, new_col_name)
                    counter = 2
                    unique_col_name = Symbol("$(var_sym)_$(transform_func_name)_$(counter)")
                    while hasproperty(data, unique_col_name)
                        counter += 1
                        unique_col_name = Symbol("$(var_sym)_$(transform_func_name)_$(counter)")
                    end
                    @warn "Column ':$new_col_name' already exists in DataFrame. " *
                          "Allocating unique column ':$unique_col_name' for transformed data."
                    new_col_name = unique_col_name
                end
                data[!, new_col_name] = transformed_data
                # Rewrite the RHS node to use the new variable
                new_rhs_args = deepcopy(rhs.args)
                new_rhs_args[:positional_args] = [new_col_name]
                new_rhs = (module_type=rhs.module_type, args=new_rhs_args)
                # Recursively process the rewritten node
                rewritten_children = _rewrite_transformations!([new_rhs], data)
                append!(new_nodes, rewritten_children)
            else
                # Not a transformation pipe, so process children recursively
                rewritten_children = _rewrite_transformations!(node.children, data)
                push!(new_nodes, (type=:operator, op=:pipe, children=rewritten_children))
            end
        elseif hasproperty(node, :type) && node.type == :operator
            # Handle other operators like ⊗ and ∘
            rewritten_children = _rewrite_transformations!(node.children, data)
            push!(new_nodes, (type=node.type, op=node.op, children=rewritten_children))
        else
            # It's a terminal node (a standard component)
            push!(new_nodes, node)
        end

    end

    return new_nodes

end



"""
    _halton_sequence(dim::Int, n::Int; skip::Int = 10) -> Matrix{Float64}
Generates an `n`-point low-discrepancy Halton sequence across `dim` dimensions
using prime radices.
"""
function _halton_sequence(dim::Int, n::Int; skip::Int = 10)::Matrix{Float64}
    primes = Int[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    seq = Matrix{Float64}(undef, dim, n)
    for d in 1:dim
        base = primes[min(d, length(primes))]
        for i in 1:n
            idx = i + skip
            f = 1.0
            r = 0.0
            while idx > 0
                f /= base
                r += f * (idx % base)
                idx = div(idx, base)
            end
            seq[d, i] = clamp(r, 1e-7, 1.0 - 1e-7)
        end
    end
    return seq
end
"""
    generate_rff_params(
        in_dims::Int,
        n_features::Int,
        length_scale::Union{Real, AbstractVector},
        kernel_name::String = "se";
        sampling::Symbol = :orthogonal,
        coords::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
        rng::AbstractRNG = Random.default_rng()
    ) -> Tuple{Matrix{Float64}, Vector{Float64}}
Generates random projection frequencies `W` (size `in_dims × n_features`) and phase
offsets `b` (length `n_features`) for Random Fourier Features (RFF) approximations
of stationary spatial Gaussian process covariance kernels.
# Mathematical Formulation
By Bochner's theorem, any continuous, shift-invariant, positive-definite kernel
k(x - y) on R^D can be represented as the Fourier transform of a non-negative
spectral probability measure p(w):
    k(x - y) = integral_{R^D} exp(i w^T (x - y)) p(w) dw
The randomized trigonometric feature map:
    phi(x) = sqrt(2 / M) [ cos(w_1^T x + b_1), ..., cos(w_M^T x + b_M) ]^T
with b_j ~ Uniform(0, 2pi) provides an unbiased Monte Carlo estimator:
    E[phi(x)^T phi(y)] = k(x, y)
# Sampling Strategies (`sampling`):
1. `:orthogonal` (Default - Orthogonal Random Features, ORF; Yu et al., 2016):
   Partitions the M features into blocks of size D x D. For each block:
     - Draw G ~ Normal(0, 1)^{D x D} and compute QR factorization G = Q R.
     - Normalize Q = Q * diag(sign(diag(R))) to guarantee uniform Haar distribution
       on the orthogonal group O(D).
     - Sample radial norms s_j from the radial spectral density:
       * Gaussian (SE/RBF): s_j ~ Chi(D) = sqrt(Chi^2(D)).
       * Matern(nu): s_j ~ Chi(D) / sqrt(u / (2nu)), where u ~ Chi^2(2nu).
     - Form W_block = S * Q / length_scale.
   Enforcing mutual orthogonality across feature vectors provably reduces kernel
   approximation variance from O(1/M) to O(1/M^2) for local spatial distances.
2. `:adaptive` (Adaptive Spatial Bandwidth Spectral Sampling):
   When coordinate matrix `coords` (N x D) is provided, computes spatial domain
   bounding extents L_d = max(x_d) - min(x_d) and minimum inter-point spacing.
   Truncates and scales spectral frequencies to the active spatial band:
       w in [pi / L_d, pi / d_min]
   preventing aliasing of unresolvable high frequencies and under-representation of
   domain-scale spatial trends.
3. `:quasi` (Quasi-Monte Carlo Stratification):
   Draws low-discrepancy Halton sequences mapped through inverse CDF quantiles
   of the spectral measure, eliminating frequency clumping and gaps.
4. `:iid` (Classical Monte Carlo RFF; Rahimi & Recht, 2007):
   Independent and identically distributed draws from Normal or Student-t
   distributions.
# Arguments
- `in_dims::Int`: Input coordinate dimensionality (e.g. 2 for 2D spatial coordinates).
- `n_features::Int`: Total number of random Fourier features M.
- `length_scale`: Kernel length_scale parameter (scalar for isotropic, vector of length
  `in_dims` for anisotropic / ARD kernels).
- `kernel_name::String`: Name of kernel ("se", "rbf", "gaussian", "matern12",
  "matern32", "matern52"). Default: `"se"`.
# Keywords
- `sampling::Symbol`: Spectral sampling method (`:orthogonal`, `:adaptive`, `:quasi`,
  or `:iid`). Default: `:orthogonal`.
- `coords`: Optional spatial coordinate matrix (N × D) used for adaptive bandwidth
  scaling.
- `rng::AbstractRNG`: Random number generator for reproducible sampling.
# Returns
- `Tuple{Matrix{Float64}, Vector{Float64}}`: `(W, b)` where `W` has size
  `(in_dims, n_features)` and `b` has length `n_features`.
# References
- Rahimi, A., & Recht, B. (2007). Random features for large-scale kernel machines.
  Advances in Neural Information Processing Systems, 20.
- Yu, F. X., Suresh, A. T., Choromanski, K. M., Holtmann-Rice, D. N., & Kumar, S.
  (2016). Orthogonal random features. Advances in Neural Information Processing
  Systems, 29, 1975-1983.
"""
function generate_rff_params(
    in_dims::Int,
    n_features::Int,
    length_scale::Union{Real, AbstractVector},
    kernel_name::String = "se";
    sampling::Symbol = :orthogonal,
    coords::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    rng::AbstractRNG = Random.default_rng()
)::Tuple{Matrix{Float64}, Vector{Float64}}
    in_dims >= 1 || throw(ArgumentError(
        "in_dims must be >= 1, got $in_dims"
    ))
    n_features >= 1 || throw(ArgumentError(
        "n_features must be >= 1, got $n_features"
    ))
    length_scale_vec = if length_scale isa Real
        Float64(length_scale) > 0.0 || throw(ArgumentError(
            "length_scale must be positive, got $length_scale"
        ))
        fill(Float64(length_scale), in_dims)
    else
        length(length_scale) == in_dims || throw(ArgumentError(
            "ARD length_scale vector length mismatch: expected $in_dims, " *
            "got $(length(length_scale))"
        ))
        all(length_scale .> 0.0) || throw(ArgumentError(
            "All ARD lengthscales must be positive"
        ))
        Float64.(length_scale)
    end
    k_name = lowercase(kernel_name)
    is_se = k_name in ["se", "gaussian", "rbf"]
    is_matern = occursin("matern", k_name)
    if !is_se && !is_matern
        @warn "Kernel '$kernel_name' not recognized for RFF. Defaulting to SE."
        is_se = true
    end
    nu = is_matern ? (
        k_name == "matern12" ? 0.5 :
        k_name == "matern32" ? 1.5 : 2.5
    ) : 1.0
    df = 2.0 * nu
    b = rand(rng, Uniform(0.0, 2.0 * pi), n_features)
    W = Matrix{Float64}(undef, in_dims, n_features)
    if sampling == :orthogonal
        n_blocks = ceil(Int, n_features / in_dims)
        W_blocks = Matrix{Float64}[]
        for _ in 1:n_blocks
            G = randn(rng, in_dims, in_dims)
            F = qr(G)
            Q = Matrix(F.Q)
            R_diag = diag(F.R)
            for j in 1:in_dims
                if R_diag[j] < 0.0
                    Q[:, j] .*= -1.0
                end
            end
            for j in 1:in_dims
                rad = if is_se
                    sqrt(rand(rng, Chisq(in_dims)))
                else
                    u_t = rand(rng, Chisq(df))
                    sqrt(rand(rng, Chisq(in_dims))) /
                        sqrt(max(1e-12, u_t / df))
                end
                Q[:, j] .*= rad
            end
            push!(W_blocks, Q)
        end
        W_all = hcat(W_blocks...)
        W .= W_all[:, 1:n_features]
        for d in 1:in_dims
            W[d, :] ./= length_scale_vec[d]
        end
    elseif sampling == :adaptive
        base_w, _ = generate_rff_params(
            in_dims, n_features, length_scale_vec, kernel_name;
            sampling = :orthogonal, rng = rng
        )
        if coords !== nothing && size(coords, 2) == in_dims && size(coords, 1) >= 2
            N_pts = size(coords, 1)
            for d in 1:in_dims
                L_d = maximum(coords[:, d]) - minimum(coords[:, d])
                L_d = L_d > 1e-6 ? L_d : 1.0
                omega_min = pi / L_d
                delta_approx = L_d / (N_pts ^ (1.0 / in_dims))
                omega_max = pi / max(1e-6, delta_approx)
                for j in 1:n_features
                    s_val = sign(base_w[d, j])
                    s_val = s_val == 0.0 ? 1.0 : s_val
                    mag = abs(base_w[d, j])
                    clamped_mag = clamp(mag, omega_min, omega_max)
                    base_w[d, j] = s_val * clamped_mag
                end
            end
        end
        W .= base_w
    elseif sampling == :quasi
        u_seq = _halton_sequence(in_dims, n_features)
        for d in 1:in_dims
            std_d = 1.0 / length_scale_vec[d]
            for j in 1:n_features
                u_val = u_seq[d, j]
                W[d, j] = if is_se
                    quantile(Normal(0.0, std_d), u_val)
                else
                    (sqrt(df) / length_scale_vec[d]) * quantile(TDist(df), u_val)
                end
            end
        end
    elseif sampling == :iid
        for d in 1:in_dims
            std_d = 1.0 / length_scale_vec[d]
            W[d, :] .= if is_se
                rand(rng, Normal(0.0, std_d), n_features)
            else
                (sqrt(df) / length_scale_vec[d]) .* rand(rng, TDist(df), n_features)
            end
        end
    else
        throw(ArgumentError(
            "Unknown sampling strategy :$sampling. " *
            "Expected :orthogonal, :adaptive, :quasi, or :iid."
        ))
    end
    return W, b
end
"""
    _is_escaped(s::AbstractString, idx::Int)::Bool
Returns `true` if the character at byte index `idx` in string `s` is escaped by an odd
number of preceding backslashes. If preceded by an even number of backslashes (including 0),
the backslashes are themselves escaped and this function returns `false`.
# Arguments
- `s::AbstractString`: The source string.
- `idx::Int`: The byte index of the character to check.
# Returns
- `Bool`: `true` if escaped, `false` otherwise.
"""
function _is_escaped(s::AbstractString, idx::Int)::Bool
    count = 0
    p = prevind(s, idx)
    while p >= 1 && s[p] == '\\'
        count += 1
        p = prevind(s, p)
    end
    return isodd(count)
end
"""
    split_terms_at_depth(input::AbstractString, sep::AbstractString)
Splits a string by a given separator, but only when the separator is not nested
inside parentheses `()`, brackets `[]`, braces `{}`, or string literal quotes.
Correctly handles consecutive escape sequences (e.g. `\\\\\\"`).
# Arguments
- `input::AbstractString`: The string to be split.
- `sep::AbstractString`: The separator string to split by.
# Returns
- `Vector{String}`: A vector of the resulting terms.
"""
function split_terms_at_depth(input::AbstractString, sep::AbstractString)
    terms = String[]
    current_term = IOBuffer()
    depth = 0
    in_quotes = false
    quote_char = '"'
    i = 1
    n_units = ncodeunits(input)
    while i <= n_units
        char = input[i]
        # Check for quote boundary, correctly respecting consecutive backslash escapes
        if (char == '"' || char == '\'') && !_is_escaped(input, i)
            if in_quotes && char == quote_char
                in_quotes = false
            elseif !in_quotes
                in_quotes = true
                quote_char = char
            end
        end
        # Check for separator at current position, but only if not inside delimiters or quotes
        if depth == 0 && !in_quotes && startswith(SubString(input, i), sep)
            push!(terms, strip(String(take!(current_term))))
            i = nextind(input, i, length(sep))
            continue
        end
        # Append character to current term and update depth
        if !in_quotes
            if char == '(' || char == '[' || char == '{'
                depth += 1
            elseif char == ')' || char == ']' || char == '}'
                depth = max(0, depth - 1)
            end
        end
        write(current_term, char)
        i = nextind(input, i)
    end
    push!(terms, strip(String(take!(current_term))))
    return filter!(!isempty, terms)
end
"""
    _infer_structure_from_args(variables, params)
Infers the structural type of a `random()` component (e.g., :spatial, :temporal)
based on the model name and the variables provided.
"""
_infer_structure_from_args(params::Dict) = _infer_structure_from_args(get(params,
    :positional_args, []), params)
_infer_structure_from_args(variables, params::Dict)::Symbol = begin
    vars_vec = variables isa AbstractVector ? variables : [variables]
    model_name_raw = get(params, :model, :iid)
    model_name = model_name_raw isa Symbol ? model_name_raw : Symbol(model_name_raw)
    # Use the central registry to find the structure for unambiguous models
    if haskey(MODEL_TO_STRUCTURE_MAP, model_name)
        return MODEL_TO_STRUCTURE_MAP[model_name]
    end
    # For truly ambiguous models, infer from variable names or standard candidates
    if model_name in AMBIGUOUS_MODELS
        if any(
            v -> Symbol(lowercase(string(v))) in STANDARD_SEASONAL_CANDIDATES ||
                 occursin(r"month|season|quarter|cycle|doy"i, string(v)),
            vars_vec
        )
            return :seasonal
        elseif any(
            v -> Symbol(lowercase(string(v))) in STANDARD_TEMPORAL_CANDIDATES ||
                 occursin(r"year|time|date|day|week|period"i, string(v)),
            vars_vec
        )
            return :temporal
        elseif any(
            v -> Symbol(lowercase(string(v))) in STANDARD_SPATIAL_UNIT_CANDIDATES ||
                 occursin(
                     r"region|district|county|zone|area|site|cluster|poly|idx"i,
                     string(v)
                 ),
            vars_vec
        )
            return :spatial
        elseif any(v -> occursin(r"lon|lat|coord|east|north|plat|plon"i, string(v)), vars_vec) ||
               length(vars_vec) == 2
            return :smooth
        end
    end
    # Default to a smooth structure if no other context is available
    return :smooth 
end
"""
    _parse_arguments_from_expr(args::Vector{Any})
Parses the arguments from a Julia expression (specifically, the `args` field of a
`:call` expression) into a dictionary of keyword arguments and a list of positional
arguments.
# Version
v1.0.0
# Arguments
- `args`: A vector of arguments from an `Expr` object.
# Returns
- A `Dict{Symbol, Any}` where keyword arguments are stored by their key, and
  positional arguments are stored under the `:positional_args` key.
"""
function _parse_arguments_from_expr(args::Vector{Any})
    parsed_args = Dict{Symbol, Any}()
    positional_args = []
    for arg in args
        if arg isa Expr && arg.head == :kw
            # This is a keyword argument, e.g., `model=:bym2`.
            key = arg.args[1]
            value = arg.args[2]
            # If the value is a QuoteNode, extract the inner symbol.
            # Otherwise, keep it as is (it could be a literal or another expression).
            if value isa QuoteNode
                parsed_args[key] = value.value # Extract the Symbol from QuoteNode
            else
                parsed_args[key] = value
            end
        else
            # This is a positional argument, e.g., `s_idx`.
            push!(positional_args, arg)
        end
    end
    # Store positional arguments under the conventional `:positional_args` key for consistency.
    if !isempty(positional_args)
        parsed_args[:positional_args] = positional_args
    end
    return parsed_args
end
"""
    _parse_value(val_str::AbstractString)
Parses a string value from a formula argument into an appropriate Julia type.
# Version
v1.0.0
# Arguments
- `val_str::AbstractString`: The string value to parse.
# Returns
- The parsed Julia object. The type can be `Symbol`, `String`, `Bool`, `Number`,
  or `Expr`.
"""
function _parse_value(val_str::AbstractString)
    val_str = strip(val_str)
    # Priority 1: Handle symbol literals like `:foo`.
    if startswith(val_str, ":")
        return Symbol(val_str[2:end])
    # Priority 2: Handle quoted string literals.
    elseif (startswith(val_str, "'") && endswith(val_str, "'")) || (startswith(val_str,
        "\"") && endswith(val_str, "\""))
        return String(val_str[2:end-1])
    # Priority 3: Handle boolean literals.
    elseif val_str == "true"
        return true
    elseif val_str == "false"
        return false
    # Priority 4: Handle bare words that are valid identifiers (e.g., variable names).
    elseif occursin(r"^[a-zA-Z_][a-zA-Z0-9_]*$", val_str)
        return Symbol(val_str)
    else
        # Fallback: Attempt to parse as a Julia expression (e.g., a number, a vector, or a
        #   function call like `Normal(0,1)`).
        try
            return Meta.parse(val_str)
        catch
            # If all else fails, treat it as a string that was not quoted.
            return String(val_str)
        end
    end
end
_parse_value(val_str::SubString{String}) = _parse_value(String(val_str))
"""
    _add_parsed_arg!(args_dict::Dict{Symbol, Any}, positional_args::Vector{Any},
      arg_val::AbstractString)
A helper function that parses a single argument string and adds it to either the
keyword argument dictionary or the positional argument list.
# Version
v1.0.0
# Arguments
- `args_dict::Dict{Symbol, Any}`: The dictionary to which keyword arguments will be added.
- `positional_args::Vector{Any}`: The vector to which positional arguments will be added.
- `arg_val::AbstractString`: The raw argument string to parse (e.g., "model=bym2" or "s_idx").
# Returns
- `nothing`. The `args_dict` and `positional_args` collections are mutated.
"""
function _add_parsed_arg!(args_dict::Dict{Symbol, Any}, positional_args::Vector{Any},
    arg_val::AbstractString)
    if contains(arg_val, "=")
        # It's a keyword argument.
        key_val = Base.split(arg_val, "=", limit=2)
        key = Symbol(Base.strip(key_val[1]))
        val_str = String(Base.strip(key_val[2]))
        args_dict[key] = _parse_value(val_str)
    else
        # It's a positional argument.
        push!(positional_args, _parse_value(arg_val))
    end
end
"""
    _parse_arguments_string(args_str::String)
Parses the inner content of a module call string (e.g., "s_idx, model=bym2") into
a dictionary of keyword arguments and a list of positional arguments.
# Version
v1.0.0
# Arguments
- `args_str::String`: The string of arguments from inside a module's parentheses.
# Returns
- `Dict{Symbol, Any}`: A dictionary containing parsed keyword arguments and a
  `:positional_args` key for positional arguments.
"""
function _parse_arguments_string(args_str::String)
    args_dict = Dict{Symbol, Any}()
    positional_args = []
    current_arg = IOBuffer()
    depth = 0
    in_string = false
    string_char = ' '
    for char in args_str
        if char == '"' || char == '\''
            if !in_string
                in_string = true
                string_char = char
            elseif char == string_char
                in_string = false
            end
        end
        if char == ',' && depth == 0 && !in_string
            arg_val = strip(String(take!(current_arg)))
            if !isempty(arg_val)
                _add_parsed_arg!(args_dict, positional_args, arg_val)
            end
        else
            write(current_arg, char)
            if !in_string
                if char == '(' || char == '['
                    depth += 1
                elseif char == ')' || char == ']'
                    depth -= 1
                end
            end
        end
    end
    arg_val = strip(String(take!(current_arg)))
    if !isempty(arg_val)
        _add_parsed_arg!(args_dict, positional_args, arg_val)
    end
    if !isempty(positional_args)
        args_dict[:positional_args] = positional_args
    end
    return args_dict
end
const CONFIG_RESERVED_SYMBOLS = Set([
    :model, :method, :structure, :family, :relationship, :habitat_relationship,
    :type, :operator, :penalty, :time_method, :link, :positional_args
])
"""
    _resolve_param_references!(params::Dict{Symbol,Any}, data::Union{DataFrame,Nothing}, calling_mod::Module=Main)
Resolve parameter references for a component's `args` dictionary:
- Skips configuration keywords (:model, :method, :structure, etc.) to preserve them as Symbols.
- If a value is a Symbol and matches a column name in `data`, resolves to that column (except for
  `:habitat` where a Symbol is preserved so component aggregation can be performed).
- Else if a value is a Symbol, resolves by evaluating in `calling_mod`.
- If a value is an Expr, evaluates it in `calling_mod` (e.g., prior distributions).
- Recurses into vectors/tuples/dicts to resolve nested references.
This mutates `params` in place.
"""
function _resolve_param_references!(
    params::AbstractDict{Symbol, <:Any},
    data::Union{DataFrame, Nothing} = nothing,
    calling_mod::Module = Main
)
    for k in collect(keys(params))
        if k in CONFIG_RESERVED_SYMBOLS
            continue
        end
        v = params[k]
        # Helper to resolve a single value
        resolve_one = function(x)
            if x isa Symbol
                if k == :habitat && data !== nothing && hasproperty(data, x)
                    return x
                elseif data !== nothing && hasproperty(data, x)
                    return data[!, x]
                else
                    try
                        return Core.eval(calling_mod, x)
                    catch
                        return x   # leave as symbol if not found
                    end
                end
            elseif x isa Expr
                try
                    return Core.eval(calling_mod, x)
                catch
                    return x
                end
            elseif x isa AbstractVector
                return [resolve_one(el) for el in x]
            elseif x isa Tuple
                return tuple((resolve_one(e) for e in x)...)
            elseif x isa AbstractDict
                newd = Dict{Any, Any}()
                for (kk, vv) in x
                    newd[kk] = resolve_one(vv)
                end
                return newd
            else
                return x
            end
        end
        params[k] = resolve_one(v)
    end
    return nothing
end
"""
    _sanitize_variablename(name::String)
Sanitizes a string to be a valid Julia variable name, suitable for use in
dynamically generated code.
# Version
v1.0.0
# Arguments
- `name::String`: The raw string to be sanitized.
# Returns
- `String`: A sanitized version of the name suitable for use as a variable.
"""
function _sanitize_variablename(name::String)
    s = name
    # ORDER MATTERS: the two named symbols must be expanded BEFORE the blanket
    # complement replace, because neither is a letter and would otherwise be turned into
    # a bare "_" first, silently losing the distinction the expansion exists to preserve.
    # Replace kronecker product symbol
    s = replace(s, "⊗" => "_kron_")
    # Replace composition symbol
    s = replace(s, "∘" => "_comp_")
    # A Julia identifier may contain only letters, digits, `_`, and `!`. The original
    # character class here covered `\s | ( ) + * & :` but not `- / ^ = > <`, so a column
    # named e.g. `a-b` or `rate/100` produced a name that is not a valid identifier and
    # then failed deep inside generated code (L1b). The class is now the complement of
    # the permitted set, so it cannot fall behind the grammar again.
    s = replace(s, r"[^\p{L}\p{N}_!]" => "_")
    # Remove any leading or trailing underscores that may result
    s = Base.strip(s, '_')
    # Consolidate sequences of multiple underscores into a single one
    s = replace(s, r"__+" => "_")
    # A name made entirely of punctuation (e.g. "***") collapses to the empty string,
    # which is not a valid identifier either; fall back to a stable placeholder.
    return isempty(s) ? "var" : s
end
"""
    _parse_single_component_term(term_str::AbstractString)
Parses a single module call string (e.g., "random(s_idx, model=bym2)") into its
constituent parts: the module name and its arguments.
# Version
v1.0.0
# Arguments
- `term_str::AbstractString`: The string representing a single module call.
# Returns
- A `NamedTuple` of the form `(module_type::Symbol, args::Dict)`.
"""
function _parse_single_component_term(term_str::AbstractString)
    term_str = Base.strip(term_str)
    # Regex to capture the module name and the content inside the parentheses.
    m = match(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)\s*$", term_str)
    if m === nothing
        # This error is thrown when a term without parentheses is incorrectly passed
        # to this function, indicating a logic error in the parent parser.
        error("Internal Parser Error: _parse_single_component_term was called with a non-module term '$term_str'.")
    end
    module_name = Symbol(m.captures[1])
    args_str = String(m.captures[2])
    args_dict = _parse_arguments_string(args_str)
    return (module_type = module_name, args = args_dict)
end
"""
    resolve_hyperpriors(model_name::String, global_priors::Dict, local_params::Dict,
      scheme::Symbol, calling_mod::Module)
Resolves the prior distribution for each hyperparameter of a model component by
checking for specifications at the local, global, and scheme levels.
# Version
v1.0.0
# Arguments
- `model_name::String`: The name of the component model being processed.
- `global_priors::Dict`: A dictionary of globally specified hyperpriors.
- `local_params::Dict`: A dictionary of parameters from the component's formula call.
- `scheme::Symbol`: The active prior scheme (e.g., `:pcpriors`).
- `calling_mod::Module`: The module context for evaluating symbols or expressions.
# Returns
- A `NamedTuple` containing the resolved prior distributions for the component.
"""
function resolve_hyperpriors(model_name::String, global_priors::Dict, local_params::Dict,
    scheme::Symbol, calling_mod::Module)
    prior_defaults = if scheme == :pcpriors
        PC_PRIORS
    elseif scheme == :informative
        INFORMATIVE_PRIORS
    else
        UNINFORMATIVE_PRIORS
    end
    # One name per parameter. There is deliberately no alias normaliser here: a
    # misspelled or legacy key should reach the vocabulary check below and be reported,
    # not silently rewritten into a different parameter's name.
    local_params_norm = Dict(local_params)
    is_anisotropic = get(local_params_norm, :anisotropic, false)
    in_dims = get(local_params_norm, :in_dims, 0)
    # Every hyperprior a component can carry, in one place. This used to be a second
    # hand-maintained list, which silently drifted from the naming vocabulary; it now
    # *is* the vocabulary, so a new stem is picked up automatically.
    possible_priors = copy(HYPERPARAMETER_STEMS)
    resolved = Dict{Symbol, Any}()
    for p_sym in possible_priors
        is_ard_param = p_sym in [:length_scale, :range]
        if is_ard_param && is_anisotropic
            if in_dims == 0
                error("Cannot resolve anisotropic prior for '$p_sym' because input dimensionality is unknown.")
            end
            prior_val = get(local_params_norm, p_sym, nothing)
            if prior_val isa Expr && prior_val.head == :vect
                # Case: length_scale=[Normal(0,1), Normal(0,1)]
                resolved_priors = [Core.eval(calling_mod, p) for p in prior_val.args]
                if length(resolved_priors) != in_dims
                    error("Anisotropic prior vector for '$p_sym' has length $(length(resolved_priors)), but expected $in_dims.")
                end
                resolved[p_sym] = resolved_priors
            else
                # Case: length_scale=Normal(0,1) or default
                single_prior = if !isnothing(prior_val)
                    prior_val isa Expr ? Core.eval(calling_mod, prior_val) : prior_val
                elseif haskey(global_priors, Symbol(model_name, "_",
                    p_sym)); global_priors[Symbol(model_name, "_", p_sym)]
                elseif haskey(global_priors, p_sym); global_priors[p_sym]
                else; get(prior_defaults, string(p_sym), nothing); end
                if !(single_prior isa Distribution)
                    error("Resolved prior for '$p_sym' is not a Distribution.")
                end
                resolved[p_sym] = [single_prior for _ in 1:in_dims]
            end
            continue
        end
        if haskey(local_params_norm, p_sym)
            prior_val = local_params_norm[p_sym]
            if prior_val isa Tuple
                resolved[p_sym] = create_pc_prior(p_sym, prior_val)
            elseif prior_val isa Expr
                try
                    resolved[p_sym] = Core.eval(calling_mod, prior_val)
                catch e
                    error("Could not evaluate `prior` argument `$(prior_val)` for component '$model_name'. Error: $e")
                end
            else
                resolved[p_sym] = prior_val
            end
            continue
        end
        global_key_model = Symbol(model_name, "_", p_sym)
        global_key_param = p_sym
        if haskey(global_priors, global_key_model)
            resolved[p_sym] = global_priors[global_key_model]
        elseif haskey(global_priors, global_key_param)
            resolved[p_sym] = global_priors[global_key_param]
        elseif haskey(prior_defaults, string(p_sym))
            resolved[p_sym] = prior_defaults[string(p_sym)]
        end
    end
    return NamedTuple(resolved)
end
"""
    _is_outermost_grouping_parentheses(s::AbstractString)::Bool
Checks if a string is fully and exclusively enclosed by a single matching pair of outer
grouping parentheses `(...)`, correctly accounting for nested parentheses, quotes, and
escaped characters.
# Arguments
- `s::AbstractString`: The string to check.
# Returns
- `Bool`: `true` if the string is fully enclosed by outer grouping parentheses, `false` otherwise.
"""
function _is_outermost_grouping_parentheses(s::AbstractString)::Bool
    s_str = Base.strip(s)
    if !startswith(s_str, "(") || !endswith(s_str, ")")
        return false
    end
    depth = 0
    in_quotes = false
    quote_char = '"'
    n_units = ncodeunits(s_str)
    i = 1
    while i <= n_units
        c = s_str[i]
        # Check string literal quote boundaries respecting backslash escaping
        if (c == '"' || c == '\'') && !_is_escaped(s_str, i)
            if in_quotes && c == quote_char
                in_quotes = false
            elseif !in_quotes
                in_quotes = true
                quote_char = c
            end
        elseif !in_quotes
            if c == '('
                depth += 1
            elseif c == ')'
                depth -= 1
                # If depth becomes 0 before the end of the string, parentheses closed prematurely
                # e.g., "(a) + (b)" closes at index 3 while total length is 9.
                if depth == 0 && i < n_units
                    return false
                end
            end
        end
        i = nextind(s_str, i)
    end
    return depth == 0
end
"""
    _parse_rhs_expression(term_str::AbstractString)
Recursively parses a term from the right-hand side (RHS) of a formula into an
Abstract Syntax Tree (AST) node, strictly respecting operator precedence and associativity.
# Operator Precedence Hierarchy (from lowest binding to highest binding)
1. Addition `+` (lowest precedence, additive model components):
   `a + b` combines additive terms of the linear predictor \$\\eta\$.
2. Pipe `|>` (low precedence, left-associative):
   `a |> b |> c` parses as `((a |> b) |> c)`. Passes upstream output to downstream component.
3. Composition `∘` (middle precedence, right-associative):
   `a ∘ b ∘ c` parses as `(a ∘ (b ∘ c))`. Functional composition of stochastic processes.
4. Kronecker Product `⊗` (highest binary precedence, associative):
   `a ⊗ b` evaluates tensor product / spatiotemporal interaction before composition or pipe.
5. Grouping `(...)` / invocations (highest precedence):
   Parentheses explicitly override precedence, e.g. `(a + b) |> c`.
# Precedence Examples
- `a ⊗ b |> c` parses as `(a ⊗ b) |> c`
- `a |> b ⊗ c` parses as `a |> (b ⊗ c)`
- `a ∘ b ⊗ c` parses as `a ∘ (b ⊗ c)`
- `a ⊗ b ∘ c` parses as `(a ⊗ b) ∘ c`
- `(a + b) |> c` parses as `pipe(add(a, b), c)`
- `intercept + fixed(x) |> random(s)` parses as `intercept + (fixed(x) |> random(s))`
"""
function _parse_rhs_expression(term_str::AbstractString)
    term_str_stripped = Base.strip(term_str)
    # If the expression is wrapped in grouping parentheses, parse the inner content recursively.
    if _is_outermost_grouping_parentheses(term_str_stripped)
        inner_content = strip(term_str_stripped[nextind(term_str_stripped,
            1):prevind(term_str_stripped, lastindex(term_str_stripped))])
        return _parse_rhs_expression(inner_content)
    end
    # Proceed with parsing based on operator precedence (lowest precedence first).
    # Precedence order (from lowest to highest binding):
    # 0. Addition `+` (lowest precedence, additive model components): a + b -> add(a, b)
    # 1. Pipe `|>` (low precedence, left-associative): a ⊗ b |> c -> (a ⊗ b) |> c
    # 2. Composition `∘` (middle precedence, right-associative): a ∘ b ⊗ c -> a ∘ (b ⊗ c)
    # 3. Kronecker product `⊗` (highest binary precedence, associative): a ⊗ b
    parts_plus = [Base.strip(p) for p in split_terms_at_depth(term_str_stripped, "+")
                  if !isempty(Base.strip(p))]
    if length(parts_plus) > 1
        return (type=:operator, op=:add,
            children=[_parse_rhs_expression(p) for p in parts_plus])
    end
    parts = split_terms_at_depth(term_str_stripped, " |> ")
    if length(parts) > 1
        # Pipe is left-associative: a |> b |> c -> (a |> b) |> c
        return (type=:operator, op=:pipe, children=[_parse_rhs_expression(join(parts[1:end-1],
            " |> ")), _parse_rhs_expression(parts[end])])
    end
    parts = split_terms_at_depth(term_str_stripped, " ∘ ")
    if length(parts) > 1
        # Composition is right-associative: a ∘ b ∘ c -> a ∘ (b ∘ c)
        return (type=:operator, op=:composition, children=[_parse_rhs_expression(parts[1]),
            _parse_rhs_expression(join(parts[2:end], " ∘ "))])
    end
    parts = split_terms_at_depth(term_str_stripped, " ⊗ ")
    if length(parts) > 1
        child_nodes = [_parse_rhs_expression(p) for p in parts]
        for (idx, ch) in enumerate(child_nodes)
            m_type = hasproperty(ch, :module_type) ? ch.module_type : :unknown
            if m_type in (:fixed, :intercept)
                throw(ArgumentError(
                    "Invalid Kronecker product (⊗) composition: component #$(idx) " *
                    "('$(parts[idx])') is a $(m_type) term. Kronecker interactions require " *
                    "multidimensional structured random process components (e.g. spatial ⊗ temporal), " *
                    "not scalar fixed effects or intercepts."
                ))
            end
        end
        return (type=:operator, op=:kronecker_product, children=child_nodes)
    end
    # If no operators are found, parse as a single module or a fixed effect.
    if occursin(r"\(.*\)", term_str_stripped)
        return _parse_single_component_term(term_str_stripped)
    else
        # Treat bare terms as fixed effects.
        return (module_type = :fixed, args = Dict{Symbol, Any}(:positional_args => Any[term_str_stripped]))
    end
end
"""
    _generate_unique_module_key(base_key::String, existing_modules::Dict)
Generates a unique, sanitized module key by ensuring it does not conflict with
existing keys in the `existing_modules` dictionary.
# Arguments
- `base_key::String`: The initial, unsanitized key string.
- `existing_modules::Dict`: The dictionary of modules already processed.
# Returns
- `String`: A unique and sanitized module key.
"""
function _generate_unique_module_key(base_key::String, existing_modules::Dict)
    sanitized_base_key = _sanitize_variablename(base_key)
    module_key = sanitized_base_key
    counter = 1
    while haskey(existing_modules, module_key)
        counter += 1
        module_key = "$(sanitized_base_key)_$(counter)"
    end
    return module_key
end
"""
    _categorize_rhs_nodes!(nodes, modules, fixed_effects; data=nothing, calling_mod::Module=Main)
Recursively traverses the Abstract Syntax Tree (AST) of the right-hand side (RHS)
of a formula, categorizing its nodes into `TethysBayes` modules (components) or bare
fixed effects.
# Version
v1.0.0
# Arguments
- `nodes`: A vector of AST nodes representing the RHS of the formula.
- `modules`: A dictionary to store categorized `TethysBayes` modules.
- `fixed_effects`: A list to store bare fixed effect terms.
# Returns
- `nothing`. The `modules` and `fixed_effects` collections are mutated.
"""
function _categorize_rhs_nodes!(nodes, modules, fixed_effects; data=nothing, calling_mod::Module=Main)
    for node in nodes
        is_pp_composition = false
        if hasproperty(node, :type) && node.type == :operator && node.op == :composition&&
            length(node.children) == 2
            outer_node, inner_node = node.children[1], node.children[2]
            if outer_node.module_type == :pointprocess && inner_node.module_type == :random
                is_pp_composition = true
                pp_module_type = get(outer_node.args, :model, :lgcp)
                final_params = copy(inner_node.args)
                for (k, v) in outer_node.args
                    if k != :model
                        final_params[k] = v
                    end
                end
                vars = get(inner_node.args, :positional_args, [])
                _resolve_param_references!(final_params, data, calling_mod) 
                new_module_data = (module_type = pp_module_type, args = final_params)
                key_parts = [string(pp_module_type), join(string.(vars), "_")]
                raw_key = join(filter(!isempty, key_parts), "_")
                module_key = _generate_unique_module_key(raw_key, modules)
                modules[module_key] = new_module_data
            end
        end
        if is_pp_composition
            continue
        end
        if hasproperty(node, :type) && node.type == :operator && node.op == :add
            _categorize_rhs_nodes!(node.children, modules, fixed_effects; data=data, calling_mod=calling_mod)
            continue
        end
        if hasproperty(node, :type) && node.type == :operator
            # Simplified key generation for composed nodes.
            function _get_simplified_composed_node_key(n)
                if hasproperty(n, :type) && n.type == :operator
                    op_str = string(n.op)
                    child_keys = [_get_simplified_composed_node_key(child) for child in n.children]
                    return join(child_keys, "_$(op_str)_")
                elseif hasproperty(n, :module_type)
                    pos_args = get(n.args, :positional_args, [])
                    # Include module type in the key for better specificity
                    return isempty(pos_args) ? string(n.module_type) : "$(string(n.module_type))_$(join(string.(pos_args), "_"))"
                else
                    return "unknown"
                end
            end
            raw_key = _get_simplified_composed_node_key(node)
            module_key = _generate_unique_module_key(raw_key, modules)
            for child in node.children 
                if hasproperty(child, :args) && child.args isa AbstractDict 
                    _resolve_param_references!(child.args, data, calling_mod) 
                end 
            end 
            interact_args = Dict{Symbol, Any}(:operator => node.op, :components => node.children) 
            _resolve_param_references!(interact_args, data, calling_mod) 
            modules[module_key] = (module_type = :interact, args = interact_args)
        elseif hasproperty(node, :module_type)
            m_type = node.module_type
            if m_type in TethysBayes_MODULE_KEYWORDS || haskey(COMPONENT_TYPE_REGISTRY, m_type)
                local raw_key
                pos_args = get(node.args, :positional_args, [])
                if m_type == :mixed
                    # For `mixed(effect | group)`, the key is based on the group variable.
                    if !isempty(pos_args) && pos_args[1] isa Expr && pos_args[1].head == :call&&
                        pos_args[1].args[1] == :|
                        group_var_str = string(pos_args[1].args[3])
                        raw_key = group_var_str
                    else
                        raw_key = isempty(pos_args) ? "mixed_unknown" : string(pos_args[1])
                    end
                elseif !isempty(pos_args)
                    # For most modules, the key is based on the variables it acts on.
                    raw_key = join([string(a) for a in pos_args], "_")
                else
                    # Fallback for modules with no positional args (e.g., intercept).
                    raw_key = string(m_type)
                end
                module_key = _generate_unique_module_key(raw_key, modules)
                _resolve_param_references!(node.args, data, calling_mod)
                modules[module_key] = node
            else
                push!(fixed_effects, string(m_type))
            end
        end
    end
end
"""
    _extract_symbols_from_expr!(sym_set::Set{Symbol}, ex::Expr)
Recursively traverses a Julia AST expression and extracts all unique `Symbol` instances into
  `sym_set`.
"""
function _extract_symbols_from_expr!(sym_set::Set{Symbol}, ex::Expr)
    for arg in ex.args
        if arg isa Symbol
            push!(sym_set, arg)
        elseif arg isa Expr
            _extract_symbols_from_expr!(sym_set, arg)
        end
    end
end
"""
    _extract_outcome_vars(raw_str::AbstractString)
Extracts individual outcome variable identifiers from an outcome string. Supports
parenthesized tuples (e.g., `"(y, y_rate)"`), bracketed vectors (e.g., `"[y1, y2]"`),
and additive expressions (e.g., `"y1 + y2"`).
# Arguments
- `raw_str::AbstractString`: Raw string representing one or more outcome variables.
# Returns
- `Vector{String}`: Cleaned list of variable names.
"""
function _extract_outcome_vars(raw_str::AbstractString)
    s = Base.strip(raw_str)
    if (startswith(s, "(") && endswith(s, ")")) ||
       (startswith(s, "[") && endswith(s, "]"))
        s = Base.strip(s[2:end-1])
    end
    outcomes = String[]
    for comma_part in split_terms_at_depth(s, ",")
        comma_part_stripped = Base.strip(comma_part)
        if !isempty(comma_part_stripped)
            for plus_part in split_terms_at_depth(comma_part_stripped, "+")
                var_name = Base.strip(plus_part)
                if !isempty(var_name)
                    push!(outcomes, var_name)
                end
            end
        end
    end
    return outcomes
end
"""
    _parse_lhs_term(term::String)
Parses a single term from the left-hand side (LHS) of the formula string into one
or more outcome specifications. Supports bare variables (`"y"`), additive outcomes
(`"y1 + y2"`), grouped outcomes (`"(y1, y2)"` or `"[y1, y2]"`), and explicit
`likelihood(...)` blocks with per-outcome or shared options.
# Version
v1.1.0
# Arguments
- `term::String`: A string representing one part of the LHS (e.g., `"y"`,
  `"y1 + y2"`, `"(y, y_rate)"`, or `"likelihood((y, y_rate), family=(poisson, gaussian))"`).
# Returns
- `Vector{Dict{Symbol, Any}}`: A vector where each dictionary represents a single
  outcome specification. Each dictionary contains:
  - `:var`: The name of the outcome variable as a `String`.
  - `:params`: A `Dict` of parameters parsed from the `likelihood()` block. This
    is empty if the outcome was specified as a bare variable.
"""
function _parse_lhs_term(term::String)
    term = Base.strip(term)
    m = match(r"likelihood\((.*)\)", term)
    specs = Dict{Symbol, Any}[]
    if !isnothing(m)
        # Case 1: The term is a likelihood() block.
        inner_content = m.captures[1]
        args = split_terms_at_depth(inner_content, ",")
        if isempty(args)
            return specs
        end
        # The first argument(s) are the outcome variables.
        outcome_var_str = Base.strip(args[1])
        params_str = join(args[2:end], ",")
        params = _parse_arguments_string(params_str)
        # Validate mutual exclusivity of zero-inflation and hurdle specifications
        is_zi = get(params, :zero_inflated, false) in (true, :true, "true") ||
                haskey(params, :phi_zi)
        is_hurdle = haskey(params, :hurdle) || haskey(params, :phi_hurdle)
        if is_zi && is_hurdle
            throw(ArgumentError("zero_inflated and hurdle specifications are mutually exclusive."))
        end
        raw_vars = _extract_outcome_vars(outcome_var_str)
        K = length(raw_vars)
        _clean_ast_node(val) = val isa QuoteNode ? val.value : val
        if K <= 1
            var_name = isempty(raw_vars) ? outcome_var_str : raw_vars[1]
            cleaned_params = Dict{Symbol, Any}()
            for (pk, pv) in params
                cleaned_params[pk] = _clean_ast_node(pv)
            end
            push!(specs, Dict{Symbol, Any}(:var => var_name, :params => cleaned_params))
        else
            # Unpack per-outcome options if provided as tuples/vectors matching K outcomes
            for k in 1:K
                p_k = Dict{Symbol, Any}()
                for (pkey, pval) in params
                    p_k[pkey] = if pval isa Expr && (pval.head == :tuple || pval.head == :vect)
                        elements = [_clean_ast_node(el) for el in pval.args]
                        if length(elements) == K
                            elements[k]
                        else
                            throw(ArgumentError(
                                "Parameter '$(pkey)' has $(length(elements)) elements, " *
                                "expected $K matching outcomes $(raw_vars)."
                            ))
                        end
                    elseif pval isa Tuple || pval isa AbstractVector
                        elements = [_clean_ast_node(el) for el in pval]
                        if length(elements) == K
                            collect(elements)[k]
                        else
                            throw(ArgumentError(
                                "Parameter '$(pkey)' has $(length(elements)) elements, " *
                                "expected $K matching outcomes $(raw_vars)."
                            ))
                        end
                    else
                        _clean_ast_node(pval)
                    end
                end
                push!(specs, Dict{Symbol, Any}(:var => raw_vars[k], :params => p_k))
            end
        end
    else
        # Case 2: The term is one or more bare outcome variables.
        raw_vars = _extract_outcome_vars(term)
        for ov in raw_vars
            push!(specs, Dict{Symbol, Any}(:var => ov, :params => Dict{Symbol, Any}()))
        end
    end
    return specs
end
"""
    _normalize_rhs_subtraction(rhs_str::AbstractString)::String

Rewrite binary subtraction `a - b` into `a + -b` so that the depth-aware term splitter
treats the two operands as separate RHS terms.

A blanket `replace(r"\\s*-\\s*" => " + -")` corrupts **scientific notation**: the exponent
sign is a `-` too, so `1e-8` became `1e + -8`, which no longer parses as a float and was
rebuilt by `Meta.parse` as `1 * e + -8` -- an undefined variable `e` (L10). Every
scientific-notation literal in a prior (`sigma=Normal(0, 1e-8)`) was therefore unusable.

Scientific-notation numbers are substituted out to inert placeholders before the rewrite and
restored afterwards, so their sign is never touched. The guard is deliberately tight:
a digit must precede the `e`, the sign must be adjacent, and a digit must follow it, while
the negative lookbehind rejects `A-Za-z_.`. That keeps genuine subtraction working --
`scale - 8` has a space, and `rate-8` as a variable reference has a non-digit before the `e`
-- while protecting real float literals.
"""
function _normalize_rhs_subtraction(rhs_str::AbstractString)::String
    saved = String[]
    guarded = replace(
        String(rhs_str),
        r"(?<![A-Za-z_.])[0-9]+(?:\.[0-9]+)?[eE][+-]?[0-9]+" => function (m)
            push!(saved, m isa RegexMatch ? m.match : String(m))
            return "\x1b$(length(saved))\x1b"
        end
    )
    normalized = replace(guarded, r"\s*-\s*" => " + -")
    for (i, original) in enumerate(saved)
        normalized = replace(normalized, "\x1b$(i)\x1b" => original)
    end
    return normalized
end

"""
    decompose_TethysBayes_formula(formula_str::String, data::DataFrame;
                           calling_mod::Module = Main, copy_data::Bool = true)

Decomposes a `TethysBayes` formula string into its constituent parts, including outcomes,
likelihood specifications, model components, and fixed effects. Applies transformations
safely without mutating caller input data when `copy_data = true`.

# Version
v1.1.0

# Arguments
- `formula_str::String`: The model formula string (e.g., `"y ~ intercept() + fixed(x)"`).
- `data::DataFrame`: The input DataFrame. If `copy_data = true` (default), transformations
  are applied to an isolated working copy to preserve caller data integrity.
- `calling_mod::Module`: Calling module scope for evaluating symbols/expressions.
- `copy_data::Bool`: If `true` (default), copies `data` before transformation.

# Returns
- A `NamedTuple` with fields `:outcomes`, `:modules`, `:fixed_effects`,
  `:has_intercept`, `:intercept_prior`, and `:data` (containing the working DataFrame
  with any generated transformation columns).
"""

function decompose_TethysBayes_formula(
    formula_str::String,
    data::DataFrame;
    calling_mod::Module = Main,
    copy_data::Bool = true
)
    parts = split_terms_at_depth(formula_str, "~")
    lhs_str = Base.strip(parts[1])
    
    # If there is no '~', default the RHS to "1" (intercept only).
    rhs_str = if length(parts) > 1
        Base.strip(parts[2])
    else
        "1" 
    end

    outcome_specs = vcat([_parse_lhs_term(term) for term in split_terms_at_depth(lhs_str, "+")]...)

    rhs_normalized = _normalize_rhs_subtraction(rhs_str)
    rhs_terms = split_terms_at_depth(rhs_normalized, "+")
    
    has_intercept = !("0" in rhs_terms || "-1" in rhs_terms)
    intercept_prior = nothing
    module_terms = String[]
    intercept_module_found = false

    for term in rhs_terms
        term_stripped = Base.strip(term)
        if term_stripped == "0" || term_stripped == "-1" || term_stripped == "1"
            continue
        elseif startswith(term_stripped, "intercept(") && !occursin(r"[⊗∘|>]", term_stripped)
            if !intercept_module_found
                intercept_module_found = true
                intercept_data = _parse_single_component_term(term_stripped)
                has_intercept = get(intercept_data.args, :positional_args, [true])[1] != false
                if haskey(intercept_data.args, :prior)
                    intercept_prior = intercept_data.args[:prior]
                end
            end
        else
            push!(module_terms, term_stripped)
        end
    end

    top_level_nodes = [_parse_rhs_expression(term) for term in module_terms]
    
    working_data = copy_data ? copy(data) : data
    rewritten_nodes = _rewrite_transformations!(top_level_nodes, working_data)

    modules = Dict{String, Any}()
    fixed_effects = String[]
    _categorize_rhs_nodes!(
        rewritten_nodes, modules, fixed_effects;
        data = working_data, calling_mod = calling_mod
    )

    return (
        outcomes = outcome_specs,
        modules = modules,
        fixed_effects = unique(fixed_effects),
        has_intercept = has_intercept,
        intercept_prior = intercept_prior,
        data = working_data
    )

end



 

"""

    _precompute_likelihood_params!(M::Dict)



Ensures all observation-level likelihood parameters are consistently formatted as

matrices of size `(N, K)`.



# Version

v1.0.0



# Arguments
- `M::Dict`: The model configuration dictionary, which is mutated by this function.
# Returns
- `nothing`.

"""

function _precompute_likelihood_params!(M::Dict)

    N = M[:y_N]

    K = M[:outcomes_N]


    param_specs = [
        (key=:censor_lower, default=-Inf),
        (key=:censor_upper, default=Inf),
        (key=:hurdle, default=-Inf),
        (key=:trials, default=1),
        (key=:weights, default=1.0),
        (key=:log_offsets, default=0.0)
    ]
    for spec in param_specs
        key = spec.key
        default_val = spec.default
        final_matrix = Matrix{typeof(default_val)}(undef, N, K)
        if !haskey(M, key)
            fill!(final_matrix, default_val)
        else
            val = M[key]
            if val isa Real
                # Uniform scalar across all observations and outcomes
                fill!(final_matrix, val)
            elseif val isa AbstractMatrix && size(val) == (N, K)
                # Exact observation-by-outcome matrix
                final_matrix = Matrix{typeof(default_val)}(val)
            elseif (val isa AbstractVector && length(val) == N) ||
                   (val isa AbstractMatrix && size(val) == (N, 1))
                # Per-observation vector or (N, 1) column broadcast across outcomes
                vec_val = Vector{typeof(default_val)}(vec(val))
                final_matrix = repeat(reshape(vec_val, N, 1), 1, K)
            elseif (val isa Tuple || val isa AbstractVector) && length(val) == K
                # Per-outcome vector broadcast across observations
                final_matrix = repeat(Matrix{typeof(default_val)}(collect(val)'), N, 1)
            else
                throw(DimensionMismatch(
                    "Likelihood parameter `:$key` has invalid dimensions $(size(val)). " *
                    "Expected scalar, vector of length $N (observations), " *
                    "vector of length $K (outcomes), or matrix of size ($N, $K)."
                ))
            end

        end

        M[key] = final_matrix
    end
end










"""

    TethysBayes_config(formula::String, data::DataFrame; calling_module::Module=Main, kwargs...)


Constructs the complete model configuration from a formula and data. 
"""
TethysBayes_config(formula::Union{Expr, Symbol}, data::DataFrame; calling_module::Module=Main,
    kwargs...) = TethysBayes_config(string(formula), data; calling_module=calling_module, kwargs...)


"""

    _process_nested_link(s_spec, sub_cfg, parent_data, parent_key, child_key, calling_module)
Resolves strata, coupling/interaction modes, moderator vectors, and observation mappings
for a nested sub-model linking into a parent equation.
"""

function _process_nested_link(

    s_spec::Union{NamedTuple, AbstractDict},
    sub_cfg::NamedTuple,
    parent_data::DataFrame,
    parent_key::Symbol,
    child_key::Symbol,
    calling_module::Module;
    parent_scope::Union{AbstractDict, NamedTuple, Nothing} = nothing
)::NamedTuple
    parent_N = size(parent_data, 1)
    sub_N = size(sub_cfg.data, 1)
    # 1. Strata resolution supporting single or multiple factors
    strata_arg = get(s_spec, :strata, nothing)
    strata_res = _resolve_nested_strata(
        parent_data, strata_arg; calling_mod = calling_module
    )
    # 2. Coupling and interaction mode
    raw_coupling = get(s_spec, :coupling, get(s_spec, :interaction, :additive))
    coupling_sym = Symbol(raw_coupling)
    if coupling_sym ∉ NESTED_COUPLING_MODES && coupling_sym != :bidirectional
        throw(ArgumentError(
            "Unsupported nested coupling mode ':$coupling_sym' for sub-model " *
            "':$child_key'. Supported modes: $(NESTED_COUPLING_MODES)."
        ))
    end
    # 3. Continuous interactive moderator
    moderator_arg = get(s_spec, :moderator, nothing)
    moderator_vec = nothing
    if !isnothing(moderator_arg)
        if moderator_arg isa Symbol || moderator_arg isa AbstractString
            mod_sym = Symbol(moderator_arg)
            if hasproperty(parent_data, mod_sym)
                moderator_vec = Vector{Float64}(parent_data[!, mod_sym])
            elseif !isnothing(parent_scope) && haskey(parent_scope, mod_sym)
                moderator_vec = Vector{Float64}(parent_scope[mod_sym])
            elseif isdefined(calling_module, mod_sym)
                moderator_vec = Vector{Float64}(getfield(calling_module, mod_sym))
            else
                throw(ArgumentError(
                    "Nested moderator column or variable ':$mod_sym' not found in " *
                    "parent ':$parent_key' data or scope."
                ))
            end
        elseif moderator_arg isa AbstractVector{<:Real}
            moderator_vec = Vector{Float64}(moderator_arg)
        else
            throw(ArgumentError(
                "Unsupported type for 'moderator' in sub-model ':$child_key': " *
                "$(typeof(moderator_arg))."
            ))
        end

        length(moderator_vec) == parent_N || throw(DimensionMismatch(

            "Nested moderator length ($(length(moderator_vec))) does not match " *
            "parent observation count ($parent_N)."
        ))
    end



    # 4. Observation alignment and mapping

    mapping_arg = get(s_spec, :mapping, nothing)

    mapping_indices = nothing

    if !isnothing(mapping_arg)

        if mapping_arg isa Symbol || mapping_arg isa AbstractString

            col_sym = Symbol(mapping_arg)

            if hasproperty(parent_data, col_sym)

                mapping_indices = Vector{Int}(parent_data[!, col_sym])

            elseif !isnothing(parent_scope) && haskey(parent_scope, col_sym)

                mapping_indices = Vector{Int}(parent_scope[col_sym])

            elseif isdefined(calling_module, col_sym)

                mapping_indices = Vector{Int}(getfield(calling_module, col_sym))
            else
                throw(ArgumentError(
                    "Nested mapping column or variable ':$col_sym' not found in parent " *
                    "model ':$parent_key' data or scope."
                ))
            end

        elseif mapping_arg isa AbstractVector{<:Integer}

            mapping_indices = Vector{Int}(mapping_arg)

        else

            throw(ArgumentError(

                "Unsupported type for 'mapping' in sub-model ':$child_key': " *

                "$(typeof(mapping_arg))."

            ))

        end

        if length(mapping_indices) != parent_N

            throw(DimensionMismatch(

                "Nested mapping length ($(length(mapping_indices))) does not " *

                "match parent observation count ($parent_N)."

            ))

        end

        if any(idx -> idx < 1 || idx > sub_N, mapping_indices)
            throw(BoundsError(
                "Nested mapping indices out of bounds (valid range: 1..$sub_N)."
            ))
        end

    elseif parent_N != sub_N
        throw(ArgumentError(
            "Observation count mismatch in sub-model ':$child_key': parent " *
            "model ':$parent_key' has $parent_N observations, but sub-model " *
            "data has $sub_N observations. Specify explicit mapping."
        ))
    end


    coupling_prior = get(s_spec, :prior, Normal(1.0, 0.5))
    fixed_coupling = get(s_spec, :fixed, false)
    augmented = merge(sub_cfg, (
        coupling = coupling_sym,
        coupling_prior = coupling_prior,
        fixed_coupling = fixed_coupling,
        strata_indices = isnothing(strata_res) ? nothing : strata_res.strata_indices,
        n_strata = isnothing(strata_res) ? 1 : strata_res.n_strata,
        strata_levels = isnothing(strata_res) ? nothing : strata_res.strata_levels,
        strata_factors = isnothing(strata_res) ? Symbol[] : strata_res.strata_factors,
        factor_indices = isnothing(strata_res) ? Dict{Symbol, Vector{Int}}() :
            strata_res.factor_indices,
        factor_levels = isnothing(strata_res) ? Dict{Symbol, Vector{Any}}() :
            strata_res.factor_levels,
        factor_dims = isnothing(strata_res) ? Dict{Symbol, Int}() :
            strata_res.factor_dims,
        is_multistrata = isnothing(strata_res) ? false : strata_res.is_multistrata,
        moderator = moderator_vec,
        parent_key = parent_key
    ))
    if !isnothing(mapping_indices)
        augmented = merge(augmented, (mapping = mapping_indices,))
    end



    return augmented

end



"""

    _infer_equation_dag(pair_dict, primary_key)



Constructs the multi-fidelity equation dependency graph from paired equations.
Identifies parent-child hierarchies, multi-target linkages, and bidirectional pairs.
"""
function _infer_equation_dag(pair_dict::Dict{Symbol, Any}, primary_key::Symbol)
    all_keys = collect(keys(pair_dict))
    parents = Dict{Symbol, Vector{Symbol}}(k => Symbol[] for k in all_keys)
    bidirectional_pairs = Set{Tuple{Symbol, Symbol}}()
    for k in all_keys
        spec = pair_dict[k]
        raw_parent = get(spec, :parent, get(spec, :links_to, nothing))
        raw_target = get(spec, :target, nothing)
        if isnothing(raw_parent) && !isnothing(raw_target) && !(raw_target isa Bool)
            raw_parent = raw_target
        end
        is_bi = get(spec, :bidirectional, false) == true ||
                get(spec, :coupling, :none) == :bidirectional ||
                get(spec, :interaction, :none) == :bidirectional
        if !isnothing(raw_parent)
            par_list = raw_parent isa AbstractVector ?
                [Symbol(p) for p in raw_parent] : [Symbol(raw_parent)]
            for p in par_list
                if p in all_keys && p != k
                    push!(parents[k], p)
                    if is_bi
                        push!(parents[p], k)
                        push!(bidirectional_pairs, (min(k, p), max(k, p)))
                    end
                end
            end
        else
            for other_k in all_keys
                other_k == k && continue
                other_spec = pair_dict[other_k]
                other_f_raw = get(other_spec, :formula, "")
                other_f_str = other_f_raw isa String ?
                    other_f_raw : string(other_f_raw)
                pat = Regex("\\b(transfer|nested|fidelity)\\s*\\(\\s*[:\"]?" *
                    string(k) * "\\b")
                if occursin(pat, other_f_str)
                    push!(parents[k], other_k)
                end
            end
        end
        if isempty(parents[k]) && k != primary_key
            push!(parents[k], primary_key)
        end
    end

    for k in all_keys
        for p in parents[k]
            if k in parents[p]
                push!(bidirectional_pairs, (min(k, p), max(k, p)))
            end
        end
    end
    children = Dict{Symbol, Vector{Symbol}}(k => Symbol[] for k in all_keys)
    for (k, pars) in pairs(parents)
        for p in pars
            push!(children[p], k)
        end
    end
    return (
        parents = parents,
        children = children,
        bidirectional_pairs = bidirectional_pairs
    )
end



"""

    TethysBayes_config(pairs::Pair{Symbol, <:Union{NamedTuple, AbstractDict}}...; kwargs...)

Constructs a multi-fidelity model configuration from a system of paired equations.
# Mathematical Formulation
Couples sub-model linear predictors into the primary linear predictor:
``\\eta_{\\text{primary}} = \\eta_{\\text{base}} + \\sum_k \\rho_k \\cdot \\eta_{\\text{sub}, k}[\\text{mapping}_k]``
where ``\\rho_k \\sim \\text{coupling\\_prior}`` (or ``\\rho_k \\equiv 1.0`` when `fixed = true`).

# Arguments
- `pairs`: One or more pairs `key => (formula = ..., data = ..., ...)`.
  The primary tier is identified by `:primary`, `:main`, `:target`, or default `pairs[1]`.
  Secondary pairs configure sub-models (e.g. `:proxy => (formula=..., data=..., mapping=..., prior=...)`).
- `kwargs`: Global model options passed to all component models.

# Returns
- A comprehensive `NamedTuple` model configuration.
"""
function TethysBayes_config(
    equation_pairs::Pair{Symbol, <:Union{NamedTuple, AbstractDict}}...;
    calling_module::Module = Main,
    kwargs...
)
    if isempty(equation_pairs)
        throw(ArgumentError("TethysBayes_config requires at least one model specification pair."))
    end



    pair_dict = Dict{Symbol, Any}(p.first => p.second for p in equation_pairs)

    all_keys = [p.first for p in equation_pairs]


    primary_key = if haskey(pair_dict, :primary)
        :primary
    elseif haskey(pair_dict, :main)
        :main
    elseif haskey(pair_dict, :target)
        :target
    else
        all_keys[1]
    end
    primary_spec = pair_dict[primary_key]
    primary_data = get(primary_spec, :data, nothing)
    if isnothing(primary_data)
        throw(ArgumentError(
            "Primary specification ':$primary_key' must include a `data` DataFrame."
        ))
    end
    # Build dependency DAG
    dag = _infer_equation_dag(pair_dict, primary_key)
    # Compute dependency depth for topological sorting (leaves first)
    depths = Dict{Symbol, Int}()
    function get_depth(node::Symbol, visited::Set{Symbol}=Set{Symbol}())
        haskey(depths, node) && return depths[node]
        node in visited && return 0
        push!(visited, node)
        ch = dag.children[node]
        d = isempty(ch) ? 0 : 1 + maximum(get_depth(c, copy(visited)) for c in ch)
        depths[node] = d
        return d
    end
    for k in all_keys
        get_depth(k)
    end
    # Sort sub-models in ascending depth: leaves (depth 0) before parents
    non_primary_keys = filter(k -> k != primary_key, all_keys)
    sorted_sub_keys = sort(non_primary_keys, by = k -> depths[k])
    configured_models = Dict{Symbol, NamedTuple}()
    for sk in sorted_sub_keys
        s_spec = pair_dict[sk]
        s_formula_raw = get(s_spec, :formula, nothing)
        s_data = get(s_spec, :data, nothing)
        if isnothing(s_formula_raw) || isnothing(s_data)
            throw(ArgumentError(
                "Sub-model specification ':$sk' must include both `formula` and `data`."
            ))
        end
        s_formula_str = s_formula_raw isa String ? s_formula_raw : string(s_formula_raw)
        sub_kwargs = Dict{Symbol, Any}(kwargs)
        for (k, v) in pairs(s_spec)
            if k ∉ (:formula, :data, :mapping, :prior, :fixed, :strata,
                   :moderator, :coupling, :interaction, :parent, :target, :links_to)
                sub_kwargs[k] = v
            end
        end
        sub_kwargs[:calling_module] = calling_module
        # Attach child sub-models if this sub-model is an intermediate parent
        ch_keys = dag.children[sk]
        if !isempty(ch_keys)
            nested_for_sk = Dict{Symbol, Any}()
            for ch_k in ch_keys
                if haskey(configured_models, ch_k)
                    ch_cfg = configured_models[ch_k]
                    ch_spec = pair_dict[ch_k]
                    linked_ch = _process_nested_link(
                        ch_spec, ch_cfg, s_data, sk, ch_k, calling_module
                    )
                    nested_for_sk[ch_k] = linked_ch
                end
            end
            if !isempty(nested_for_sk)
                sub_kwargs[:nested_components] = nested_for_sk
            end
        end
        sub_cfg = TethysBayes_config(s_formula_str, s_data; sub_kwargs...)
        configured_models[sk] = sub_cfg
    end
    # Attach direct children of primary_key to primary_kwargs
    primary_nested = Dict{Symbol, Any}()
    for ch_k in dag.children[primary_key]
        if haskey(configured_models, ch_k)
            ch_cfg = configured_models[ch_k]
            ch_spec = pair_dict[ch_k]
            linked_ch = _process_nested_link(
                ch_spec, ch_cfg, primary_data, primary_key, ch_k, calling_module
            )
            primary_nested[ch_k] = linked_ch
        end
    end
    primary_formula_raw = get(primary_spec, :formula, nothing)
    if isnothing(primary_formula_raw)
        throw(ArgumentError(
            "Primary specification ':$primary_key' must include a `formula`."
        ))
    end
    primary_formula_str = primary_formula_raw isa String ?
        primary_formula_raw : string(primary_formula_raw)
    primary_kwargs = Dict{Symbol, Any}(kwargs)
    for (k, v) in pairs(primary_spec)
        if k ∉ (:formula, :data, :parent, :target, :links_to)
            primary_kwargs[k] = v
        end
    end
    primary_kwargs[:calling_module] = calling_module
    primary_kwargs[:nested_components] = primary_nested
    return TethysBayes_config(primary_formula_str, primary_data; primary_kwargs...)
end



function TethysBayes_config(

    formula::String, data::DataFrame; calling_module::Module=Main, kwargs...

)

    decomposed_formula = decompose_TethysBayes_formula(

        formula, data; calling_mod = calling_module, copy_data = true

    )

    df_processed = decomposed_formula.data



    M = _initialize_config(

        df_processed,

        merge(Dict(kwargs), Dict(:calling_module => calling_module))

    )

    M[:formula] = formula



    # Support submodels keyword argument dictionary if provided

    if haskey(M, :submodels)

        if !haskey(M, :nested_components)

            M[:nested_components] = Dict{Symbol, Any}()

        end

        for (sk, s_spec) in pairs(M[:submodels])

            s_sym = Symbol(sk)

            if !haskey(M[:nested_components], s_sym)

                s_formula_raw = get(s_spec, :formula, "")

                s_data = get(s_spec, :data, nothing)

                if !isnothing(s_data) && !isempty(string(s_formula_raw))

                    s_kwargs = Dict{Symbol, Any}(kwargs)
                    for (k, v) in pairs(s_spec)
                        if k ∉ (:formula, :data, :mapping, :prior, :fixed,
                               :strata, :moderator)
                            s_kwargs[k] = v
                        end
                    end
                    s_kwargs[:calling_module] = calling_module
                    if haskey(s_spec, :submodels)
                        s_kwargs[:submodels] = s_spec.submodels
                    end
                    s_cfg = TethysBayes_config(string(s_formula_raw), s_data; s_kwargs...)
                    s_cfg = _process_nested_link(
                        s_spec, s_cfg, data, :main, s_sym, calling_module
                    )
                    M[:nested_components][s_sym] = s_cfg
                end
            end
        end
    end


    _process_lhs!(M, decomposed_formula.outcomes, decomposed_formula.modules)


    is_multivariate = get(M, :model_arch, "univariate") == "multivariate"

    if is_multivariate

        for (key, mod_data_nt) in decomposed_formula.modules

            model_name = get(mod_data_nt.args, :model, :none)

            if mod_data_nt.module_type == :dynamics && model_name in [

                :leslie_matrix, :delay_difference, :generalized_lotka_volterra,

                :generalized_leslie_matrix

            ]

                M[:is_multivariate_dynamics] = true

                M[:multivariate_dynamics_key] = key

                break

            end

        end

    end



    _precompute_likelihood_params!(M)



    M[:add_intercept] = decomposed_formula.has_intercept

    if !isnothing(decomposed_formula.intercept_prior)

        prior_val = decomposed_formula.intercept_prior

        if prior_val isa Expr
            try
                M[:intercept_prior] = Core.eval(calling_module, prior_val)
            catch e
                error(
                    "Could not evaluate `prior` argument `$(prior_val)` " *
                    "in intercept() module. Error: $e"
                )
            end
        else
            M[:intercept_prior] = prior_val
        end
    end

    for (key, mod_data_nt) in decomposed_formula.modules
        mod_type = mod_data_nt.module_type
        mod_data_dict = Dict(
            :key => key,
            :type => mod_type,
            :variables => get(mod_data_nt.args, :positional_args, []),
            :params => mod_data_nt.args
        )
        processor! = get(MODULE_PROCESSORS, mod_type, nothing)
        create_component = true
        if !isnothing(processor!)
            create_component = processor!(M, mod_data_dict, M, M[:hyperpriors])
            if mod_type == :random && !haskey(mod_data_dict[:params], :structure)
                mod_data_dict[:params][:structure] = _infer_structure_from_args(mod_data_dict[:params])
            end
        end
        if !create_component
            continue
        end
        component_obj = resolve_technical_primitive(
            mod_data_dict, M, M[:hyperpriors], M[:prior_scheme]
        )
        mod_data_dict[:component_obj] = component_obj
        M_nt = NamedTuple(M)
        precomputes = get_precomputes(component_obj, M_nt, mod_data_dict)
        spec = (
            key=Symbol(key), 
            structure=get(mod_data_dict[:params], :structure, get(MODEL_TO_STRUCTURE_MAP,
                mod_type, :any)),
            var=join(string.(mod_data_dict[:variables]), "_"), 
            component_obj=component_obj, 
            params=mod_data_dict[:params], 
            hyper=precomputes
        )
        push!(M[:components], spec)
    end

    all_fixed_effects = copy(decomposed_formula.fixed_effects)
    if haskey(M, :fixed_effects_from_modules)
        append!(all_fixed_effects, M[:fixed_effects_from_modules])
    end
    _process_fixed_effects!(M, unique(all_fixed_effects))
    _process_fixed_effects_priors!(M)
    # Pre-compute Cholesky factorizations for static components.
    _precompute_static_components!(M)
    _finalize_config!(M)
    

    return NamedTuple(M)

end

 











# =============================================================================

# CANONICAL PARAMETER NAME VOCABULARY

#

# Every sampled parameter is named `<role>[_<qualifier>]_<component_key>[_<outcome>]`.

# The role words below are the *only* supported spellings: there are no aliases, no

# abbreviations and no back-compat shims. `find_chain_param` resolves a parameter by

# its registered name or not at all.

#

# The two role words that carry the most weight:

#

#   innovations  A vector of i.i.d. standard-normal draws. It is *not* the effect

#                itself -- components scale and correlate it into whatever latent

#                quantity they need. Historically abbreviated `innovations` ("unstructured

#                random effect"), which read as if it were the spatial effect.

#   latent_field The reconstructed, already-structured field (a GMRF/smoothed draw,

#                or the structured half of a BYM2 decomposition). Historically

#                abbreviated `latent_field`.

#

# Keeping these two distinct matters: `innovations` is what NUTS samples and what

# diagnostics should check; `latent_field` is the interpretable quantity you plot.

# =============================================================================



"""

    HYPERPARAMETER_STEMS
Hyperparameters are shared across outcomes in a multivariate model unless the
component marks them otherwise. Each entry is a role word chosen from the vocabulary
described above; component-specific detail belongs in the component's own
constructor arguments, not in the sampled name.
"""

const HYPERPARAMETER_STEMS = [

    # ── Field scale and correlation ───────────────────────────────────────────

    # `sigma` is the one word for "the overall SD of whatever this component draws".

    # It is deliberately not `scale` or `sigma0`: one concept, one word, everywhere.

    :sigma,

    :rho,                       # spatial or temporal correlation

    :rho_unconstrained,         # ... on the unconstrained (logit) scale

    :nu,                        # Matern smoothness

    # ── Two-regime autoregression (AR2, TAR) ──────────────────────────────────
    # Scoped by regime rather than numbered, so the rho/sigma pair is obvious.
    :rho_regime_1, :rho_regime_2,
    :rho_regime_1_unconstrained, :rho_regime_2_unconstrained,
    :sigma_regime_1, :sigma_regime_2,
    :sigma_regime_1_unconstrained, :sigma_regime_2_unconstrained,
    # ── Kernel geometry ──────────────────────────────────────────────────────
    # `range` is shared with the GP length scale because an SPDE's kappa *is* a
    # range: the distance over which the process decorrelates.
    :length_scale,              # kernel length scale
    :range,                     # spatial range (SPDE `kappa`; same idea, one word)
    :period, :amplitude, :phase, :curvature,
    # ── Correlation structure ────────────────────────────────────────────────
    :correlation_cholesky,      # Cholesky factor of the LKJ correlation matrix
    :correlation_lkj,           # LKJ shape parameter for that matrix
    # ── Population dynamics ──────────────────────────────────────────────────
    # Spelled out: `r`, `K`, `q` and `M_nat` were unsearchable in a chain column and
    # collided with the matrix names `K` and `W`. The role suffixes
    # (`_mean_log`, `_sd`, `_unconstrained`) are the same shape used everywhere else.
    :intrinsic_growth_rate,             # per-capita rate of increase
    :intrinsic_growth_rate_mean_log,    # its hierarchical mean, on the log scale
    :intrinsic_growth_rate_sd,          # between-process SD of that mean
    :intrinsic_growth_rate_unconstrained, # the raw spatially varying log-rate field
    :carrying_capacity,                 # saturation level of the growth term
    :carrying_capacity_mean_log,
    :carrying_capacity_sd,
    :carrying_capacity_unconstrained,
    :exploitation_rate,                 # per-unit-effort removal coefficient
    :natural_mortality_rate,            # background death rate
    # ── Process description ──────────────────────────────────────────────────
    :velocity, :diffusion, :shape, :friction_power,
    :sigma_process,                    # process-noise SD
    :sigma_effects,                    # effect-size SD
    :sigma_st_interaction,             # SD of the space-time interaction
    # ── Network, eigen and graph-spectral ────────────────────────────────────
    :alpha,                            # graph-wavelet decay exponent
    :beta, :gamma, :delta,
    :beta_habitat_diffusion,
    :pca_sd,                           # SD from truncating principal components
    :pdef_sd,                          # SD from eliciting quantiles
    # ── Point process ────────────────────────────────────────────────────────
    :n_parents,                        # how many parents drive a case event
    # ── Spatio-varying autoregression ────────────────────────────────────────
    # Named as a family with `innovations_precision` so the three read together.
    :precision_field_scale,   # marginal SD of the precision field
    :precision_field_mixing,  # BYM2 / LeRoux mixing weight for it
    :threshold_unconstrained, # TAR threshold, on the unconstrained scale
]



"""

    LATENT_FIELD_STEMS
Latent fields and innovation vectors. These are always unique per outcome in a
multivariate model, because each outcome gets its own draw.
The distinction that matters: `innovations` are i.i.d. standard-normal draws that carry
no structure of their own, while `latent_field` is the reconstructed field *after* the
component has scaled and correlated those draws. Neither name implies the other.
"""

const LATENT_FIELD_STEMS = [

    :innovations,           # standard-normal innovation vector (was `innovations`)

    :latent_field,          # structured latent field (was `latent_field`)

    # Qualified innovation vectors: `innovations` for a specific role.

    :innovations_diagonal,      # scaled by sqrt(eigenvalue) (was `innovations_diagonal`)

    :innovations_coupling,      # parent-independent coupling, sparse GP (was `innovations_coupling`)

    :innovations_inducing,      # inducing-point innovations (was `innovations_inducing`)

    :innovations_precision,     # drives a precision field (was `innovations_precision`)

    :innovations_cluster,       # mixture component innovations (was `innovations_cluster`)
    :innovations_predator,      # predator population innovations (was `innovations_predator`)
    :innovations_habitat,       # habitat preference innovations (was `innovations_habitat`)
    # Random Fourier features
    :rff_weights, :rff_weights_1, :rff_weights_2,   # Random Fourier weights
    :rff_offsets, :rff_offsets_1,                   # Random Fourier offsets
    # Eigen / spectral orientation
    :reflection_vector_unconstrained,   # Householder vector (was `v_unscaled`)
    :factors_flat,                      # spectral factor loadings
    :beta_cos, :beta_sin,               # harmonic cosine/sine coefficients
    :parent_locs_x, :parent_locs_y,     # parent locations for the point process
]



"""

    generate_full_variable_names(spec::NamedTuple, arch::String, outcome_idx::Union{Int,

      Nothing}; prefix::String="")



Generates a NamedTuple of full variable names for a given component.
# Naming scheme
See `HYPERPARAMETER_STEMS` and `LATENT_FIELD_STEMS` for the canonical role words.
The resulting chain column is `<stem>_<component_key>` (plus `_<outcome_idx>` for
per-outcome quantities in a multivariate model).
# Arguments
- `spec`: The component's specification.
- `arch`: The model architecture (`"univariate"` or `"multivariate"`).
- `outcome_idx`: The index of the outcome for multivariate models.
- `prefix`: An optional prefix for nested models.
# Returns
- A `NamedTuple` containing all necessary variable names as Symbols.

"""

function generate_full_variable_names(spec::NamedTuple, arch::String, outcome_idx::Union{Int,

    Nothing}; prefix::String="")
    base_key = string(spec.key)
    full_key = isempty(prefix) ? base_key : "$(prefix)_$(base_key)"
    is_multivariate = arch == "multivariate"
    shared_spec = get(spec.params, :shared, false)
    # Suffix for latent fields, which are always per-outcome in a multivariate model.
    latent_field_suffix = is_multivariate ? "_$(outcome_idx)" : ""
    names = Dict{Symbol, Symbol}()
    # --- Hyperparameters ---
    # These parameters may be shared across outcomes in a multivariate model.
    for p in HYPERPARAMETER_STEMS
        p_is_shared = is_param_shared(shared_spec, p)
        p_suffix = (is_multivariate && !p_is_shared) ? "_$(outcome_idx)" : ""
        names[p] = Symbol("$(p)_$(full_key)$(p_suffix)")
    end

    # --- Latent Fields & Innovations ---
    # These are always unique per outcome in a multivariate model.
    for p in LATENT_FIELD_STEMS
        names[p] = Symbol("$(p)_$(full_key)$(latent_field_suffix)")
    end



    return NamedTuple(names)

end









"""

    _generate_st_interaction_block(M::NamedTuple, s_spec, t_spec, is_multivariate::Bool,
      eta_name::String)
Generates Turing code for a spatiotemporal interaction effect. 
# Version
v1.0.0
# Arguments
- `M`: The main model configuration `NamedTuple`.
- `s_spec`, `t_spec`: The specifications for the spatial and temporal components.
- `is_multivariate`: A boolean indicating if the model is multivariate.
- `eta_name`: The name of the linear predictor variable.
# Returns
- A `String` containing the generated Turing code for the interaction block.
"""

function _generate_st_interaction_block(M::NamedTuple, s_spec, t_spec, is_multivariate::Bool,
    eta_name::String)
    has_composed_kronecker = any(spec -> hasproperty(spec, :component_obj) && 
        spec.component_obj isa Composed && spec.component_obj.operator == :kronecker_product,
        get(M, :components, []))
    if has_composed_kronecker || get(M, :model_st, "none") == "none" 
        return ""
    end

    if isnothing(s_spec) || isnothing(t_spec)
        @warn "Spatiotemporal interaction requested but marginal specifications are missing."
        return ""
    end


    s_key = string(s_spec.key)
    t_key = string(t_spec.key)
    
    s_chol_access = get(s_spec, :is_static, false) ? "spec_registry[:$(s_key)].cholesky_factor" : (hasproperty(s_spec.hyper, :cholesky_factor) ? "spec_registry[:$(s_key)].hyper.cholesky_factor" : "cholesky(Symmetric(Matrix(spec_registry[:$(s_key)].hyper.Q_template) + noise * I))")
    t_chol_access = get(t_spec, :is_static, false) ? "spec_registry[:$(t_key)].cholesky_factor" : (hasproperty(t_spec.hyper, :cholesky_factor) ? "spec_registry[:$(t_key)].hyper.cholesky_factor" : "cholesky(Symmetric(Matrix(spec_registry[:$(t_key)].hyper.Q_template) + noise * I))")
    K = get(M, :outcomes_N, 1)
    st_sigma_prior_dist_str = haskey(M, :sigma_st_interaction_prior) ?
      _distribution_to_string(M.sigma_st_interaction_prior) : (haskey(M,
      :st_interaction_sigma_prior) ? _distribution_to_string(M.st_interaction_sigma_prior) :
      "Exponential(1.0)")
    if is_multivariate
        interaction_code = """
    # --- Spatiotemporal Interaction Priors ---
    sigma_st_interaction ~ filldist($(st_sigma_prior_dist_str), $K)
    # --- Spatiotemporal Interaction Innovations ---
    innovations_st_interaction ~ MvNormal(fill!(Array{T}(undef, M.s_N * M.t_N * $K), 0),
      I)
    let
        C_s = $s_chol_access
        C_t = $t_chol_access
        Z_tensor = reshape(innovations_st_interaction, M.s_N, M.t_N, $K)
        for k in 1:$K
            Z_k = view(Z_tensor, :, :, k)
            tmp_spatial = C_s.U \\ Z_k
            st_field_k_unscaled = (transpose(C_t.U \\ transpose(tmp_spatial)))
            Turing.@addlogprob! logpdf(Normal(0, 0.001 * (M.s_N * M.t_N)),
              sum(st_field_k_unscaled))
            st_field_k = st_field_k_unscaled .* sigma_st_interaction[k]
            # Vectorized update to the linear predictor
            effect_k = vec(st_field_k)[M.st_idx]
            @views $(eta_name)[:, k] .+= effect_k
        end

    end

    """
    else
        interaction_code = """
    # --- Spatiotemporal Interaction Priors ---
    sigma_st_interaction ~ $(st_sigma_prior_dist_str)
    innovations_st_interaction ~ MvNormal(fill!(Array{T}(undef, M.s_N * M.t_N), 0), I)
    let
        C_s = $s_chol_access
        C_t = $t_chol_access
        Z_matrix = reshape(innovations_st_interaction, M.s_N, M.t_N)
        tmp_spatial = C_s.U \\ Z_matrix
        st_field_unscaled = (transpose(C_t.U \\ transpose(tmp_spatial)))
        Turing.@addlogprob! logpdf(Normal(0, 0.001 * (M.s_N * M.t_N)), sum(st_field_unscaled))
        st_field = st_field_unscaled .* sigma_st_interaction
        # Vectorized update to the linear predictor
        effect = vec(st_field)[M.st_idx]
        $(eta_name) = $(eta_name) .+ effect
    end

    """
    end


    return interaction_code

end

 


"""

    _generate_householder_reflection_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)



Generates Turing code for the Householder reflection (spectral orientation) feature.

This allows for rotating the latent space in multivariate models to better align signals,
which can be useful for processes with directional dependencies. This is controlled by
the `spectral_orientation=true` keyword argument.
"""
function _generate_householder_reflection_block(M::NamedTuple, is_multivariate::Bool,
    eta_name::String)
    if !is_multivariate || !get(M, :spectral_orientation, false)
        return "", ""
    end



    K = M[:outcomes_N]
    priors_str = """
    # Householder reflection for spectral orientation
    v_unscaled_reflection ~ MvNormal(fill!(Array{T}(undef, $(K)), 0), I)
    """
    update_str = """
    let
        v_reflection = v_unscaled_reflection / (norm(v_unscaled_reflection) + 1e-9)
        H_reflection = I - 2.0 * v_reflection * v_reflection'
        $(eta_name) = $(eta_name) * H_reflection
    end

    """
    return priors_str, update_str
end



"""

    _generate_nested_model_block(M::NamedTuple, is_multivariate::Bool, main_eta_name::String)


Generates the code block for nested sub-models.
# Version
v1.0.0
"""

function _generate_nested_model_block(

    M::NamedTuple, is_multivariate::Bool, main_eta_name::String;
    parent_prefix::String = "", parent_data_ref::String = "M"
)
    if haskey(M, :nested_components) && !isempty(M.nested_components)
        priors_acc = String[]
        updates_acc = String[]
        likelihood_acc = String[]
        for (key, sub_M) in M.nested_components
            prefix = isempty(parent_prefix) ? string(key) : "$(parent_prefix)_$(key)"
            sub_data_var = "sub_M_$(prefix)"
            sub_data_accessor = "$(parent_data_ref).nested_components[:$(key)]"
            # 1. Define linking parameters based on coupling mode and strata
            coupling = get(sub_M, :coupling, get(sub_M, :interaction, :additive))
            is_tensor = coupling == :tensor && get(sub_M, :is_multistrata, false)
            is_moderated = coupling == :moderated
            n_strata = get(sub_M, :n_strata, 1)
            is_stratified = n_strata > 1
            has_moderator = hasproperty(sub_M, :moderator) && !isnothing(sub_M.moderator)
            c_prior = get(sub_M, :coupling_prior, Normal(1.0, 0.5))
            dist_str = _distribution_to_string(c_prior)
            fixed_c = get(sub_M, :fixed_coupling, false)
            rho_term = ""
            if is_tensor
                for fac in sub_M.strata_factors
                    dim = sub_M.factor_dims[fac]
                    rho_fac_name = "rho_nested_$(prefix)_$(fac)"
                    if fixed_c
                        push!(sub_updates_acc, "$(rho_fac_name) = fill(1.0, $(dim))")
                    else
                        push!(priors_acc,
                            "$(rho_fac_name) ~ " *
                            "filldist($(dist_str), $(dim))")
                    end
                end
                fac_terms = [
                    "rho_nested_$(prefix)_$(fac)[$(sub_data_var).factor_indices[:$(fac)]]"
                    for fac in sub_M.strata_factors
                ]
                rho_term = "(" * join(fac_terms, " .* ") * ")"
            elseif is_moderated
                rho_0_name = "rho_nested_$(prefix)_0"
                rho_1_name = "rho_nested_$(prefix)_1"
                if fixed_c
                    if is_stratified
                        push!(sub_updates_acc, "$(rho_0_name) = fill(1.0, $(n_strata))")
                        push!(sub_updates_acc, "$(rho_1_name) = fill(0.0, $(n_strata))")
                    else
                        push!(sub_updates_acc, "$(rho_0_name) = 1.0")
                        push!(sub_updates_acc, "$(rho_1_name) = 0.0")
                    end
                else
                    if is_stratified
                        push!(priors_acc,
                            "$(rho_0_name) ~ " *
                            "filldist($(dist_str), $(n_strata))")
                        push!(priors_acc,
                            "$(rho_1_name) ~ " *
                            "filldist($(dist_str), $(n_strata))")
                    else
                        push!(priors_acc,
                            "$(rho_0_name) ~ $(dist_str)")
                        push!(priors_acc,
                            "$(rho_1_name) ~ $(dist_str)")
                    end
                end
                rho_0_expr = is_stratified ?
                    "$(rho_0_name)[$(sub_data_var).strata_indices]" : rho_0_name
                rho_1_expr = is_stratified ?
                    "$(rho_1_name)[$(sub_data_var).strata_indices]" : rho_1_name
                rho_term = "($(rho_0_expr) .+ $(rho_1_expr) .* $(sub_data_var).moderator)"
            else
                rho_name = "rho_nested_$(prefix)"
                if fixed_c
                    if is_stratified
                        push!(sub_updates_acc, "$(rho_name) = fill(1.0, $(n_strata))")
                    else
                        push!(sub_updates_acc, "$(rho_name) = 1.0")
                    end
                else
                    if is_stratified
                        push!(priors_acc,
                            "$(rho_name) ~ " *
                            "filldist($(dist_str), $(n_strata))")
                    else
                        push!(priors_acc,
                            "$(rho_name) ~ $(dist_str)")
                    end
                end
                rho_term = is_stratified ?
                    "$(rho_name)[$(sub_data_var).strata_indices]" : rho_name
                if has_moderator
                    rho_term = "($(rho_term) .* $(sub_data_var).moderator)"
                end
            end
            # --- Start generating code for the sub-model ---
            sub_priors_acc = String[]
            sub_updates_acc = String[]
            sub_arch = get(sub_M, :model_arch, "univariate")
            is_sub_multivariate = sub_arch == "multivariate"
            sub_eta_name = is_sub_multivariate ? "eta_latent_sub_$(prefix)" : "eta_sub_$(prefix)"
            # --- Generate Priors for sub-model ---
            # Intercept
            if get(sub_M, :add_intercept, false)
                intercept_var_name = "intercept_$(prefix)"
                dist_str_ic = is_sub_multivariate ?
                    "filldist($(_distribution_to_string(sub_M.intercept_prior)), $(sub_M.outcomes_N))" :
                    _distribution_to_string(sub_M.intercept_prior)
                push!(sub_priors_acc,
                    "$(intercept_var_name) ~ $(dist_str_ic)")
            end
            # Fixed Effects (priors & updates)
            if get(sub_M, :Xfixed_N, 0) > 0
                fe_priors, fe_updates = _generate_fixed_effects_block(
                    sub_M, is_sub_multivariate, sub_eta_name; prefix = prefix
                )
                if !isempty(strip(fe_priors))
                    push!(sub_priors_acc, fe_priors)
                end
                if !isempty(strip(fe_updates))
                    push!(sub_updates_acc, fe_updates)
                end
            end
            # Components
            for (i, sub_spec) in enumerate(sub_M.components)
                prefixed_sub_spec = merge(sub_spec, (key=Symbol(prefix, "_", sub_spec.key),))
                for k in 1:sub_M.outcomes_N
                    sub_outcome_idx = is_sub_multivariate ? k : nothing
                    priors_str = get_priors(sub_spec.component_obj, prefixed_sub_spec,
                        sub_arch, sub_outcome_idx, sub_M)
                    push!(sub_priors_acc, replace(
                        priors_str,
                        "spec_registry[:$(prefixed_sub_spec.key)]" => "$(sub_data_var).components[$(i)]"
                    ))
                end
            end
            # Sub-model Likelihood Priors
            sub_lik_priors = _generate_likelihood_section(
                sub_M, is_sub_multivariate; prefix = prefix
            )
            if !isempty(strip(sub_lik_priors))
                push!(sub_priors_acc, sub_lik_priors)
            end
            # --- Assemble sub-model updates ---
            sub_eta_init = if get(sub_M, :add_intercept, false)
                is_sub_multivariate ?
                    "intercept_$(prefix)' .+ fill!(Array{T}(undef, $(sub_M.y_N), $(sub_M.outcomes_N)), 0)" :
                    "intercept_$(prefix) .+ fill!(Array{T}(undef, $(sub_M.y_N)), 0)"
            else
                is_sub_multivariate ?
                    "fill!(Array{T}(undef, $(sub_M.y_N), $(sub_M.outcomes_N)), 0)" :
                    "fill!(Array{T}(undef, $(sub_M.y_N)), 0)"
            end
            # Add component updates
            for (i, sub_spec) in enumerate(sub_M.components)
                prefixed_sub_spec = merge(sub_spec, (key=Symbol(prefix, "_", sub_spec.key),))
                for k in 1:sub_M.outcomes_N
                    sub_outcome_idx = is_sub_multivariate ? k : nothing
                    updates_str = get_updates(sub_spec.component_obj, prefixed_sub_spec,
                        sub_arch, sub_outcome_idx, sub_M)
                    updates_str_final = replace(
                        updates_str,
                        r"\b(eta_latent|eta)\b" => sub_eta_name,
                        "spec_registry[:$(prefixed_sub_spec.key)]" => "$(sub_data_var).components[$(i)]",
                        r"\bM\." => "$(sub_data_var)."
                    )
                    push!(sub_updates_acc, updates_str_final)
                end
            end
            # Recursive child sub-models for cascading hierarchies
            if haskey(sub_M, :nested_components) && !isempty(sub_M.nested_components)
                child_p, child_u, child_l = _generate_nested_model_block(
                    sub_M, is_sub_multivariate, sub_eta_name;
                    parent_prefix = prefix,
                    parent_data_ref = sub_data_var
                )
                if !isempty(strip(child_p))
                    push!(sub_priors_acc, child_p)
                end
                if !isempty(strip(child_u))
                    push!(sub_updates_acc, child_u)
                end
                if !isempty(strip(child_l))
                    push!(likelihood_acc, child_l)
                end
            end
            # --- Sub-model Likelihood ---
            sub_lik_code = _generate_final_likelihood_block(
                sub_M, is_sub_multivariate; prefix = prefix
            )
            sub_lik_code_final = replace(
                sub_lik_code,
                r"\b(eta_latent|eta)\b" => sub_eta_name,
                r"\bM\." => "$(sub_data_var)."
            )
            # --- Linear Predictor Coupling with interaction modes ---
            has_map = haskey(sub_M, :mapping) && !isnothing(sub_M.mapping)
            map_ref = has_map ? "$(sub_data_var).mapping" : ""
            eta_coupling = if is_multivariate
                if is_sub_multivariate
                    sub_expr = has_map ? "view($(sub_eta_name), $(map_ref), :)" : sub_eta_name
                else
                    sub_expr = has_map ? "view($(sub_eta_name), $(map_ref))" : sub_eta_name
                end
                if coupling == :multiplicative
                    "@views $(main_eta_name) .*= (1.0 .+ $(rho_term) .* $(sub_expr))"
                elseif coupling == :interaction
                    "@views $(main_eta_name) .+= $(rho_term) .* ($(main_eta_name) .* $(sub_expr))"
                else
                    "@views $(main_eta_name) .+= $(rho_term) .* $(sub_expr)"
                end
            else
                if is_sub_multivariate
                    sub_expr = has_map ? "$(sub_eta_name)[$(map_ref), :]" : sub_eta_name
                else
                    sub_expr = has_map ? "$(sub_eta_name)[$(map_ref)]" : sub_eta_name
                end
                if coupling == :multiplicative
                    "$(main_eta_name) = $(main_eta_name) .* (1.0 .+ $(rho_term) .* $(sub_expr))"
                elseif coupling == :interaction
                    "$(main_eta_name) = $(main_eta_name) .+ $(rho_term) .* ($(main_eta_name) .* $(sub_expr))"
                else
                    "$(main_eta_name) = $(main_eta_name) .+ $(rho_term) .* $(sub_expr)"
                end
            end
            # --- Assemble final code blocks ---
            push!(priors_acc, join(filter(!isempty, sub_priors_acc), "\n"))
            updates_block = """
            $(sub_data_var) = $(sub_data_accessor)
            $(sub_eta_name) = $(sub_eta_init)
            $(join(filter(!isempty, sub_updates_acc), "\n"))
            $(eta_coupling)
            """
            push!(updates_acc, updates_block)
            push!(likelihood_acc, sub_lik_code_final)
        end
        return join(priors_acc, "\n\n"), join(updates_acc, "\n\n"), join(likelihood_acc, "\n\n")
    end

    return "", "", ""

end











"""
    _process_fixed_effects!(M::Dict, fixed_effects_vars::Vector{String})
Processes all fixed effect variables from the formula.
This function is updated to pass the `calling_module` from the main configuration `M`
to the `create_fixed_design` function. This ensures that the formula parsing within
`create_fixed_design` occurs in the correct module context, resolving the `MethodError`
related to world age issues.
"""
function _process_fixed_effects!(M::Dict, fixed_effects_vars::Vector{String})
    if isempty(fixed_effects_vars)
        M[:Xfixed] = zeros(M[:y_N], 0)
        M[:Xfixed_N] = 0
        M[:Xfixed_names] = Symbol[]
        M[:Xfixed_applied_formula] = nothing
        return
    end

    rhs_vars = join(fixed_effects_vars, " + ")
    # Explicitly add "0" to prevent StatsModels from creating its own intercept.
    # The intercept is handled separately by the `intercept()` module.
    rhs = "0 + " * rhs_vars
    # Pass the calling_module to create_fixed_design.
    Xfixed_named, applied_formula = create_fixed_design(
        rhs, 
        M[:data], 
        M[:calling_module]; 
        contrasts=get(M, :contrasts, Dict())
    )
    if size(Xfixed_named, 1) != M[:y_N]
        @warn "Dimension mismatch in fixed effects design matrix: Expected $(M[:y_N]) rows, but got $(size(Xfixed_named, 1)). This can happen if there are missing values in the fixed effect covariates. Attempting to reconcile."
        # This is a simple reconciliation; a more robust solution might involve
        # filtering the main data frame based on complete cases for all model variables.
        if size(Xfixed_named, 1) < M[:y_N]
            padded_Xfixed = zeros(M[:y_N], size(Xfixed_named, 2))
            # This assumes the rows align, which might not be safe without row indices.
            # A more robust implementation would use indices from `completecases`.
            padded_Xfixed[1:size(Xfixed_named, 1), :] = Matrix(Xfixed_named)
            M[:Xfixed] = padded_Xfixed
        else
            M[:Xfixed] = Matrix(Xfixed_named[1:M[:y_N], :])
        end
    else
        M[:Xfixed] = Matrix(Xfixed_named)
    end


    M[:Xfixed_N] = size(M[:Xfixed], 2)
    M[:Xfixed_names] = size(Xfixed_named, 2) > 0 ? names(Xfixed_named, 2) : Symbol[]
    M[:Xfixed_applied_formula] = applied_formula



    # Process Errors-in-Variables (EIV) standard deviations
    eiv_dict = get(M, :fixed_effects_eiv, Dict{Symbol, Any}())
    eiv_map = Dict{Symbol, Vector{Float64}}()
    if !isempty(eiv_dict) && !isempty(M[:Xfixed_names])
        for col_sym in M[:Xfixed_names]
            col_str = string(col_sym)
            base_sym = Symbol(replace(col_str, r"^.*:" => ""))
            matched_key = if haskey(eiv_dict, col_sym)
                col_sym
            elseif haskey(eiv_dict, base_sym)
                base_sym
            else
                nothing
            end
            if !isnothing(matched_key)
                err_spec = eiv_dict[matched_key]
                sd_vec = if err_spec isa Symbol
                    if hasproperty(M[:data], err_spec)
                        Vector{Float64}(M[:data][!, err_spec])
                    else
                        error("Errors-in-Variables error_sd column :$err_spec not found in model DataFrame.")
                    end
                elseif err_spec isa AbstractString
                    sd_sym = Symbol(err_spec)
                    if hasproperty(M[:data], sd_sym)
                        Vector{Float64}(M[:data][!, sd_sym])
                    else
                        error("Errors-in-Variables error_sd column '$err_spec' not found in model DataFrame.")
                    end
                elseif err_spec isa Real
                    fill(Float64(err_spec), M[:y_N])
                elseif err_spec isa AbstractVector
                    if length(err_spec) != M[:y_N]
                        error("Errors-in-Variables error_sd vector length ($(length(err_spec))) does not match data row count ($(M[:y_N])).")
                    end
                    Vector{Float64}(err_spec)
                else
                    error("Unsupported error_sd specification for Errors-in-Variables: $(typeof(err_spec))")
                end
                eiv_map[col_sym] = sd_vec
            end
        end
    end

    M[:Xfixed_eiv_map] = eiv_map

end





"""

    _canonical_term_string(term::StatsModels.AbstractTerm)
Creates a canonical string representation for a `StatsModels.AbstractTerm`. This is
used internally to map priors to the correct fixed-effect coefficients, especially
for interaction terms where the order of variables does not matter.
# Arguments
- `term::StatsModels.AbstractTerm`: A term from a `StatsModels.FormulaTerm`.
# Returns
- `String`: A standardized string representation of the term.
"""
function _canonical_term_string(term::StatsModels.AbstractTerm)
    if term isa StatsModels.InteractionTerm
        # Sort term names for canonical representation, e.g., "a&b" is the same as "b&a".
        term_names = sort([string(t.sym) for t in term.terms])
        return join(term_names, "&")
    elseif term isa StatsModels.Term
        return string(term.sym)
    elseif term isa StatsModels.ConstantTerm
        return "(Intercept)"
    else
        # Fallback for other term types like FunctionTerm, etc.
        # This might not be perfectly canonical but is a reasonable default.
        return string(term)
    end

end


"""
    _precompute_static_components!(M::Dict)
Pre-computes matrix factorizations for static model components. This version is
CPU-only.
"""
function _precompute_static_components!(M::Dict)
    noise = M[:noise]
    new_components = []
    static_component_types = [IID, ICAR, Besag, RW1, RW2, Cyclic, PSpline, TPS, BSpline, Eigen, Moran, Barycentric, TensorProductSmooth]
    for spec_in in M[:components]
        current_spec = spec_in
        m_obj = current_spec.component_obj
        if m_obj isa Mixed
            inner_model = m_obj.model
            is_inner_static = any(T -> inner_model isa T, static_component_types)
            if is_inner_static && hasproperty(current_spec.hyper, :Q_template)&&
                !isnothing(current_spec.hyper.Q_template)&&
                size(current_spec.hyper.Q_template, 1) > 0
                try
                    Q_concrete = current_spec.hyper.Q_template
                    Q_dense = Matrix(Q_concrete)
                    F = cholesky(Symmetric(Q_dense + noise * I))
                    final_spec = merge(current_spec, (is_static=true, cholesky_factor=F))
                    push!(new_components, final_spec)
                    continue
                catch e
                    @warn "Cholesky factorization failed for static inner model in $(current_spec.key). Reverting to dynamic computation. Error: $e"
                end
            end
        end
        is_main_static = !(current_spec.component_obj isa Composed)&&
            any(T -> current_spec.component_obj isa T, static_component_types)
        if is_main_static && hasproperty(current_spec.hyper, :Q_template)&&
            !isnothing(current_spec.hyper.Q_template)&&
            size(current_spec.hyper.Q_template, 1) > 0
            try
                Q_concrete = current_spec.hyper.Q_template
                Q_dense = Matrix(Q_concrete)
                F = cholesky(Symmetric(Q_dense + noise * I))
                final_spec = merge(current_spec, (is_static=true, cholesky_factor=F))
                push!(new_components, final_spec)
            catch e
                @warn "Cholesky factorization failed for static component $(current_spec.key). Reverting to dynamic computation. Error: $e"
                final_spec = merge(current_spec, (is_static=false,))
                push!(new_components, final_spec)
            end
        else
            final_spec = merge(current_spec, (is_static=get(current_spec, :is_static, false),))
            push!(new_components, final_spec)
        end
    end

    M[:components] = new_components
end
"""
    _initialize_config(data::DataFrame, kwargs)
Creates the initial model configuration dictionary (`M`) 
# Version
v1.0.0
# Arguments
- `data::DataFrame`: The input data for the model.
- `kwargs`: A dictionary of keyword arguments passed from the main `@tbm` call.
# Returns
- `Dict{Symbol, Any}`: The initial model configuration dictionary `M`.
""" 
function _initialize_config(data::DataFrame, kwargs)
    M = Dict{Symbol, Any}()
    M[:data] = data
    M[:y_N] = size(data, 1)
    # Set defaults that can be overridden by user-provided kwargs.
    M[:noise] = 1e-6
    M[:hyperpriors] = Dict{Symbol, Any}()
    M[:prior_scheme] = :pcpriors
    M[:fixed_effects_priors] = Dict{Symbol, Any}()
    M[:spectral_orientation] = true
    # Merge user-provided keyword arguments, overriding defaults.
    for (k, v) in kwargs; M[k] = v; end
    # Initialize containers for components and basis matrices.
    M[:calling_module] = get(kwargs, :calling_module, Main)
    M[:components] = []
    M[:basis_matrices] = Dict{Symbol, Any}()
    return M
end
"""
    _process_lhs!(M::Dict, outcome_specs::Vector{Dict{Symbol, Any}})
Processes the Left-Hand Side (LHS) of the formula, setting up outcomes, likelihoods,
and observation-level parameters in the main model configuration.
# Version
v1.0.0
# Arguments
- `M::Dict`: The main model configuration dictionary, which is mutated by this function.
- `outcome_specs::Vector{Dict{Symbol, Any}}`: A vector of parsed outcome specifications
  from the formula parser.
# Returns
- `nothing`.
"""
function _process_lhs!(
    M::Dict, outcome_specs::Vector{Dict{Symbol, Any}}, modules::Dict=Dict()
)
    outcomes = [Symbol(spec[:var]) for spec in outcome_specs]
    likelihood_specs = [spec[:params] for spec in outcome_specs]
    # Check the family from the first spec, assuming it's consistent for a `+` separated group.
    family_type = string(get(likelihood_specs[1], :family, "gaussian"))
    if family_type in [
        "multinomial", "categorical", "dirichlet_multinomial", "dirichlet"
    ]
        # --- Special Handling for Multinomial, Categorical & Dirichlet Models ---
        spec1 = likelihood_specs[1]
        ref_kw = get(spec1, :reference, nothing)
        if length(outcomes) >= 2
            # Wide format: multiple columns represent categories
            for out_sym in outcomes
                if !hasproperty(M[:data], out_sym)
                    error("Outcome category column ':$out_sym' for $family_type not found in data frame.")
                end
            end
            # Determine reference category index (default: 1)
            ref_idx = 1
            if ref_kw isa Integer
                if 1 <= ref_kw <= length(outcomes)
                    ref_idx = Int(ref_kw)
                else
                    @warn "Reference category index $ref_kw out of range 1:$(length(outcomes)). Defaulting to 1."
                end
            elseif ref_kw isa Symbol || ref_kw isa String
                found = findfirst(==(Symbol(ref_kw)), outcomes)
                if !isnothing(found)
                    ref_idx = found
                else
                    @warn "Reference category '$ref_kw' not found in outcomes $(outcomes). Defaulting to 1."
                end
            end
            # Order outcomes so reference category is placed at index 1
            ordered_outcomes = if ref_idx == 1
                outcomes
            else
                vcat([outcomes[ref_idx]], [outcomes[i] for i in 1:length(outcomes) if i != ref_idx])
            end
            K = length(ordered_outcomes)
            M[:category_levels] = ordered_outcomes
            M[:category_labels] = string.(ordered_outcomes)
            M[:ref_category] = ordered_outcomes[1]
            M[:K_categories] = K
            M[:outcomes] = ordered_outcomes
            M[:outcomes_N] = K - 1 # K - 1 free linear predictors for non-reference categories!
            M[:model_arch] = "multivariate"
            M[:is_multinomial] = true
            M[:multinomial_family] = Symbol(family_type)
            raw_y = Matrix(M[:data][!, ordered_outcomes])
            if family_type in ["multinomial", "dirichlet_multinomial"]
                M[:y_obs] = round.(Int, raw_y)
                M[:trials] = vec(sum(M[:y_obs], dims=2))
            elseif family_type == "categorical"
                M[:y_obs] = [argmax(raw_y[i, :]) for i in 1:size(raw_y, 1)]
                M[:trials] = ones(Int, size(raw_y, 1))
            else # dirichlet
                M[:y_obs] = Float64.(raw_y)
                M[:trials] = ones(Int, size(raw_y, 1))
            end
            merged_params = Dict{Symbol, Any}()
            for spec in reverse(likelihood_specs); merge!(merged_params, spec); end
            M[:likelihood_specs] = [merged_params]
        else
            # Single outcome column: long categorical factor or integer classes
            out_var = outcomes[1]
            if !hasproperty(M[:data], out_var)
                error("Multinomial/Categorical outcome column ':$out_var' not found in data frame.")
            end
            col_data = M[:data][!, out_var]
            uniq_levels = sort(unique(skipmissing(col_data)))
            if length(uniq_levels) < 2
                error("Multinomial/Categorical outcome variable ':$out_var' must have at least 2 unique levels, found $(length(uniq_levels)).")
            end
            # Place reference level at index 1
            ordered_levels = if !isnothing(ref_kw)
                ref_str = string(ref_kw)
                found = findfirst(x -> string(x) == ref_str, uniq_levels)
                if !isnothing(found)
                    vcat([uniq_levels[found]], [uniq_levels[i] for i in 1:length(uniq_levels) if i != found])
                else
                    @warn "Reference category '$ref_kw' not found in unique levels of '$out_var'. Using $(first(uniq_levels))."
                    uniq_levels
                end
            else
                uniq_levels
            end
            K = length(ordered_levels)
            level_map = Dict(lvl => i for (i, lvl) in enumerate(ordered_levels))
            y_indices = [level_map[v] for v in col_data]
            M[:category_levels] = ordered_levels
            M[:category_labels] = string.(ordered_levels)
            M[:ref_category] = ordered_levels[1]
            M[:K_categories] = K
            M[:outcomes] = [Symbol("cat_$(l)") for l in ordered_levels]
            M[:outcomes_N] = K - 1 # K - 1 free linear predictors for non-reference categories!
            M[:model_arch] = "multivariate"
            M[:is_multinomial] = true
            M[:multinomial_family] = Symbol(family_type)
            if family_type == "categorical"
                M[:y_obs] = y_indices
                M[:trials] = ones(Int, length(y_indices))
            else
                # Single column multinomial / dirichlet_multinomial: encode as one-hot count matrix
                y_onehot = zeros(Int, length(y_indices), K)
                for i in 1:length(y_indices)
                    y_onehot[i, y_indices[i]] = 1
                end
                M[:y_obs] = y_onehot
                M[:trials] = ones(Int, length(y_indices))
            end
            merged_params = Dict{Symbol, Any}()
            for spec in reverse(likelihood_specs); merge!(merged_params, spec); end
            M[:likelihood_specs] = [merged_params]
        end



    else

        # --- Standard Handling for Other Families ---

        M[:outcomes] = outcomes
        M[:outcomes_N] = length(outcomes)
        M[:likelihood_specs] = likelihood_specs
        for (i, spec) in enumerate(M[:likelihood_specs])
            if !haskey(spec, :family)
                spec[:family] = "gaussian"
                @warn "Likelihood `family` not specified for outcome '$(outcomes[i])'. Defaulting to `family=gaussian`."
            end
            if string(get(spec, :family, "")) == "ordinal"
                if M[:outcomes_N] > 1
                    error("The `ordinal` family is currently only supported for univariate models.")
                end
                outcome_var = M[:outcomes][i]
                if !hasproperty(M[:data], outcome_var)
                    error("Ordinal outcome variable ':$outcome_var' not found in data.")
                end
                outcome_data = M[:data][!, outcome_var]
                if !(eltype(outcome_data) <: Integer)
                    @warn "Ordinal outcome variable ':$outcome_var' is not of integer type. Attempting to convert."
                    try; M[:data][!, outcome_var] = round.(Int, outcome_data); catch; error("Could not convert ordinal outcome variable ':$outcome_var' to integers."); end
                end
                unique_levels = sort(unique(M[:data][!, outcome_var]))
                K = length(unique_levels)
                if K < 2
                    error("Ordinal outcome variable ':$outcome_var' must have at least 2 unique levels.")
                end
                spec[:latent_dist] = get(spec, :latent_dist, :logistic)
                spec[:K] = K
                level_map = Dict(level => i for (i, level) in enumerate(unique_levels))
                M[:data][!, outcome_var] = [level_map[val] for val in M[:data][!, outcome_var]]
                @info "Ordinal outcome '$outcome_var' recoded to integers 1:$K."
            end
        end

        for out_sym in M[:outcomes]
            if !hasproperty(M[:data], out_sym)
                error("Outcome variable ':$out_sym' specified in the formula was not found as a column in the provided data frame. Please check for typos or ensure the column exists.")
            end
        end
        if M[:outcomes_N] > 1
            M[:model_arch] = "multivariate"
            M[:y_obs] = Matrix(M[:data][!, M[:outcomes]])
        else
            M[:model_arch] = get(M, :model_arch, "univariate")
            M[:y_obs] = M[:data][!, M[:outcomes][1]]
        end
    end
    # --- Resolve Observation-Level Parameters ---
    calling_mod = get(M, :calling_module, Main)
    # Merge all likelihood parameters to resolve global settings like offsets, weights, etc.
    merged_params = Dict{Symbol, Any}()
    for spec_params in M[:likelihood_specs]
        merge!(merged_params, spec_params)
    end
    obs_param_configs = [
        (:log_offsets, [:log_offsets, :offsets], 0.0),
        (:weights, [:weights], 1.0),
        (:trials, [:trials], 1)
    ]
    for (target_key, param_aliases, default_val) in obs_param_configs
        has_per_outcome = any(
            spec -> any(k -> haskey(spec, k), param_aliases),
            M[:likelihood_specs]
        )
        if M[:outcomes_N] > 1 && has_per_outcome
            N_obs = nrow(M[:data])
            col_matrix = Matrix{Float64}(undef, N_obs, M[:outcomes_N])
            any_provided = false
            for (k, spec) in enumerate(M[:likelihood_specs])
                temp_dict = Dict{Symbol, Any}(:calling_module => calling_mod)
                _resolve_obs_param!(temp_dict, spec, M[:data], param_aliases, target_key)
                if haskey(temp_dict, target_key)
                    any_provided = true
                    val = temp_dict[target_key]
                    if val isa Real
                        col_matrix[:, k] .= Float64(val)
                    elseif val isa AbstractVector && length(val) == N_obs
                        col_matrix[:, k] .= Float64.(val)
                    else
                        col_matrix[:, k] .= Float64(default_val)
                    end
                else
                    col_matrix[:, k] .= Float64(default_val)
                end
            end
            if any_provided
                M[target_key] = target_key == :trials ? round.(Int, col_matrix) : col_matrix
                M[Symbol("user_provided_", target_key)] = true
            end
        else
            _resolve_obs_param!(M, merged_params, M[:data], param_aliases, target_key)
        end
    end
    scalar_param_keys = [:censor_lower, :censor_upper, :hurdle]
    for key in scalar_param_keys
        if haskey(merged_params, key)
            _resolve_obs_param!(M, merged_params, M[:data], [key], key)
        else
            values_per_outcome = []
            any_provided = false
            for spec_params in M[:likelihood_specs]
                val = _resolve_outcome_scalar_param(spec_params, key, calling_mod)
                if !isnothing(val)
                    any_provided = true
                end
                push!(values_per_outcome, val)
            end
            if any_provided
                default_val = if key == :censor_lower || key == :hurdle
                    -Inf
                else
                    Inf
                end
                final_values = [isnothing(v) ? default_val : v for v in values_per_outcome]
                M[key] = M[:outcomes_N] == 1 ? final_values[1] : final_values
                M[Symbol("user_provided_", key)] = true
            end
        end
    end
    _resolve_boolean_obs_param!(M, merged_params, :zero_inflated, :use_zi)
    if get(M, :user_provided_hurdle, false) && get(M, :use_zi, false)
        throw(ArgumentError(
            "Model specification error: `zero_inflated=true` and `hurdle` cannot both be specified. " *
            "Zero-inflation (mixture at zero) and hurdle (two-part conditional threshold) formulations " *
            "are mutually exclusive. Please choose either zero-inflation or hurdle."
        ))
    end
    _resolve_boolean_obs_param!(M, merged_params, :volatility, :volatility)
end
"""
    _resolve_outcome_scalar_param(params::Dict, key::Symbol, calling_mod::Module)
Resolves a likelihood parameter that must be a scalar value for a given outcome.
# Version
v1.0.0
# Arguments
- `params::Dict`: The dictionary of parameters from the parsed `likelihood()` module.
- `key::Symbol`: The symbol for the parameter to resolve (e.g., `:censor_lower`).
- `calling_mod::Module`: The module context for evaluating symbols or expressions.
# Returns
- The resolved scalar `Number`, or `nothing` if the parameter is not found or invalid.
"""
function _resolve_outcome_scalar_param(params::Dict, key::Symbol, calling_mod::Module)
    if !haskey(params, key)
        return nothing
    end
    val = params[key]
    if val isa Number
        return val
    elseif val isa Symbol || val isa Expr
        try
            evaluated_val = Core.eval(calling_mod, val)
            if evaluated_val isa Number
                return evaluated_val
            else
                @warn "Parameter '$val' for '$key' must be a scalar number, but evaluated to type '$(typeof(evaluated_val))'. Ignoring."
                return nothing
            end
        catch
            @warn "Parameter '$val' for '$key' could not be evaluated as a scalar variable in the calling module. Ignoring."
            return nothing
        end
    else
        @warn "Parameter for '$key' has an unsupported type '$(typeof(val))'. It must be a scalar number or a variable that evaluates to one. Ignoring."
        return nothing
    end
end
"""
    _resolve_obs_param!(opt_dict, params, data, param_keys, target_key)
Resolves an observation-level parameter (e.g., offsets, weights) from the
likelihood parameters and sets it in the main configuration dictionary.
# Version
v1.0.0
# Arguments
- `opt_dict`: The main model configuration dictionary (`M`), which is mutated.
- `params`: The dictionary of parameters from the parsed `likelihood()` module.
- `data`: The input `DataFrame`.
- `param_keys`: A list of possible keys for the parameter (e.g., `[:log_offsets, :offsets]`).
- `target_key`: The key to set in `opt_dict` (e.g., `:log_offsets`).
# Returns
- `nothing`.
"""
function _resolve_obs_param!(opt_dict, params, data, param_keys, target_key)
    for key in param_keys
        if haskey(params, key)
            val = params[key]
            if val isa Symbol
                if hasproperty(data, val)
                    # Case 1: The value is a symbol that directly matches a column name.
                    opt_dict[target_key] = data[!, val]
                    opt_dict[Symbol("user_provided_", target_key)] = true
                elseif haskey(opt_dict, val)
                    # Case 1b: The value was passed as a keyword argument in opt_dict
                    opt_dict[target_key] = opt_dict[val]
                    opt_dict[Symbol("user_provided_", target_key)] = true
                else
                    # Case 2: The symbol might refer to a variable in the calling scope.
                    calling_mod = get(opt_dict, :calling_module, Main)
                    try
                        evaluated_val = Core.eval(calling_mod, val)
                        if evaluated_val isa Symbol && hasproperty(data, evaluated_val)
                            # e.g., my_var = :my_col; log_offsets = my_var
                            opt_dict[target_key] = data[!, evaluated_val]
                            opt_dict[Symbol("user_provided_", target_key)] = true
                        elseif evaluated_val isa String && hasproperty(data, Symbol(evaluated_val))
                            # e.g., my_var = "my_col"; log_offsets = my_var
                            opt_dict[target_key] = data[!, Symbol(evaluated_val)]
                            opt_dict[Symbol("user_provided_", target_key)] = true
                        elseif evaluated_val isa Number || evaluated_val isa AbstractVector
                            # e.g., my_vec = [0.1, ...]; log_offsets = my_vec
                            opt_dict[target_key] = evaluated_val
                            opt_dict[Symbol("user_provided_", target_key)] = true
                        else
                            @warn "Parameter '$val' for '$target_key' evaluated to an unsupported type '$(typeof(evaluated_val))'. Ignoring."
                        end
                    catch
                        @warn "Parameter '$val' for '$target_key' is not a valid column name and could not be evaluated as a variable in the calling module '$(calling_mod)'. Ignoring."
                    end
                end
            elseif val isa Number || val isa AbstractVector
                # Case 3: The value is a literal number or vector.
                opt_dict[target_key] = val
                opt_dict[Symbol("user_provided_", target_key)] = true
            elseif val isa Tuple || (val isa Expr && (val.head == :tuple || val.head == :vect))
                # Case 4: A tuple or vector of columns/values for multiple outcomes
                raw_elems = val isa Tuple ? collect(val) : val.args
                calling_mod = get(opt_dict, :calling_module, Main)
                resolved_cols = []
                all_valid = true
                for elem in raw_elems
                    el = elem isa QuoteNode ? elem.value : elem
                    if el isa Symbol && hasproperty(data, el)
                        push!(resolved_cols, data[!, el])
                    elseif el isa String && hasproperty(data, Symbol(el))
                        push!(resolved_cols, data[!, Symbol(el)])
                    elseif el isa Number
                        push!(resolved_cols, fill(Float64(el), nrow(data)))
                    elseif el isa AbstractVector && length(el) == nrow(data)
                        push!(resolved_cols, Float64.(el))
                    else
                        try
                            ev = Core.eval(calling_mod, el)
                            if ev isa Symbol && hasproperty(data, ev)
                                push!(resolved_cols, data[!, ev])
                            elseif ev isa Number
                                push!(resolved_cols, fill(Float64(ev), nrow(data)))
                            elseif ev isa AbstractVector && length(ev) == nrow(data)
                                push!(resolved_cols, Float64.(ev))
                            else
                                all_valid = false
                                break
                            end
                        catch
                            all_valid = false
                            break
                        end
                    end
                end
                if all_valid && !isempty(resolved_cols)
                    mat = hcat(resolved_cols...)
                    opt_dict[target_key] = target_key == :trials ? round.(Int, mat) : mat
                    opt_dict[Symbol("user_provided_", target_key)] = true
                    return
                else
                    @warn "Observation parameter '$val' for '$target_key' could not be resolved. Ignoring."
                end
            else
                @warn "Observation parameter '$val' for '$target_key' is not a valid column name, vector, or scalar. Ignoring."
            end
            return # Stop after finding the first matching key.
        end
    end
end
"""
    _resolve_boolean_obs_param!(opt_dict, params, param_key, target_key)
Resolves a boolean flag from the likelihood parameters and sets it in the main
configuration dictionary.
# Version
v1.0.0
# Arguments
- `opt_dict`: The main model configuration dictionary (`M`), which is mutated.
- `params`: The dictionary of parameters from the parsed `likelihood()` module.
- `param_key`: The key for the boolean flag to look for in `params`.
- `target_key`: The key to set in `opt_dict`.
# Returns
- `nothing`.
"""
function _resolve_boolean_obs_param!(opt_dict, params, param_key, target_key) 
    if haskey(params, param_key)
        val = params[param_key]
        if val isa Bool
            opt_dict[target_key] = val
        elseif val isa Symbol || val isa Expr
            calling_mod = get(opt_dict, :calling_module, Main)
            try
                evaluated_val = Core.eval(calling_mod, val)
                if evaluated_val isa Bool
                    opt_dict[target_key] = evaluated_val
                else
                    @warn "Parameter '$val' for '$param_key' evaluated to a non-boolean type '$(typeof(evaluated_val))'. Ignoring."
                end
            catch
                @warn "Parameter '$val' for '$param_key' could not be evaluated as a boolean variable in the calling module. Ignoring."
            end
        else
            @warn "Parameter for '$param_key' has an unsupported type '$(typeof(val))'. It must be a boolean or a variable that evaluates to one. Ignoring."
        end
    end
end
"""
    _process_fixed_effects_priors!(M::Dict)
Resolves and stores the prior distributions for each fixed effect coefficient.
# Version
v1.0.0
# Arguments
- `M::Dict`: The main model configuration dictionary, which is mutated by this function.
# Returns
- `nothing`.
"""
function _process_fixed_effects_priors!(M::Dict) 
    n_fixed = get(M, :Xfixed_N, 0) 
    if n_fixed == 0
        M[:Xfixed_priors_vec] = UnivariateDistribution[]
        return
    end
    calling_mod = get(M, :calling_module, Main)
    custom_priors = M[:fixed_effects_priors]
    intercept_prior_val = get(M, :intercept_prior, nothing)
    default_prior = Normal(0, 5)
    priors_vec = Vector{Union{UnivariateDistribution, Nothing}}(undef, n_fixed)
    fill!(priors_vec, nothing)
    normalized_priors = Dict{String, Any}()
    for (key, prior) in custom_priors
        norm_key = replace(string(key), r"\s*[\*:]\s*" => "&")
        norm_key = replace(norm_key, r"\s*&\s*" => "&")
        
        if occursin("&", norm_key)
            parts = sort(Base.split(norm_key, '&'))
            norm_key = join(parts, '&')
        end
        normalized_priors[norm_key] = prior
    end
    applied_formula = get(M, :Xfixed_applied_formula, nothing)
    if isnothing(applied_formula)
        @warn "Could not find the applied formula for fixed effects. Prior assignment may be incomplete. This is an internal issue."
        M[:Xfixed_priors_vec] = fill(default_prior, n_fixed)
        return
    end
    all_coef_names = string.(coefnames(applied_formula.rhs))
    coef_name_to_idx = Dict(name => i for (i, name) in enumerate(all_coef_names))
    processed_indices = Set{Int}()
    for term in applied_formula.rhs.terms
        canonical_name = _canonical_term_string(term)
        if haskey(normalized_priors, canonical_name)
            prior_val = normalized_priors[canonical_name]
            prior_obj = if prior_val isa Expr
                try
                    Core.eval(calling_mod, prior_val)
                catch e
                    error("Could not evaluate `prior` argument `$(prior_val)` for fixed effect '$canonical_name'. Error: $e")
                end
            else
                prior_val
            end
            term_coef_names = coefnames(term)
            term_coef_names_vec = term_coef_names isa AbstractString ? [term_coef_names] : term_coef_names
            for coef_name in term_coef_names_vec
                if haskey(coef_name_to_idx, coef_name)
                    idx = coef_name_to_idx[coef_name]
                    priors_vec[idx] = prior_obj isa Tuple ? create_pc_prior(Symbol(canonical_name), prior_obj) : prior_obj
                    push!(processed_indices, idx)
                else
                    @warn "Coefficient name '$coef_name' for term '$canonical_name' not found in the full coefficient list. Prior may not be applied."
                end
            end
        end
    end
    # Assign default priors to any coefficients that were not explicitly assigned one.
    for i in 1:n_fixed
        if !(i in processed_indices)
            coef_name_str = all_coef_names[i]
            if coef_name_str == "(Intercept)"
                prior = intercept_prior_val
                priors_vec[i] = isnothing(prior) ? default_prior : (prior isa Tuple ? create_pc_prior(:intercept, prior) : prior)
            else
                priors_vec[i] = default_prior
            end
        end
    end
    M[:Xfixed_priors_vec] = convert(Vector{UnivariateDistribution}, priors_vec) 
end
"""
    _finalize_config!(M::Dict)
Ensures the model configuration dictionary has all necessary keys with default values
before being passed to the code generator.
# Version
v1.0.0
# Arguments
- `M::Dict`: The model configuration dictionary, which is mutated by this function.
# Returns
- `nothing`.
"""
function _finalize_config!(M::Dict)
    # This function provides safe defaults for keys that may not have been set
    # during the main configuration process.
    defaults = Dict(
        :s_N => 0, 
        :t_N => 0, 
        :u_N => 0,
        :s_idx => ones(Int, M[:y_N]),
        :t_idx => ones(Int, M[:y_N]),
        :u_idx => ones(Int, M[:y_N]),
        :hyperpriors => Dict(),
        :prior_scheme => :pcpriors,
        :intercept_prior => Normal(0, 5)
    )
    for (key, val) in defaults
        if !haskey(M, key)
            M[key] = val
        end
    end
end
"""
    _replace_TethysBayes_modules_in_expr(ex)
Recursively traverses a Julia expression and replaces `TethysBayes`-specific modules
with their `StatsModels.jl` equivalents for parsing within other modules like `mixed()`.
# Version
v1.0.0
# Arguments
- `ex`: A Julia expression, symbol, or literal.
# Returns
- The modified expression, ready for parsing by `StatsModels.jl`.
"""
function _replace_TethysBayes_modules_in_expr(ex)
    if ex isa Expr && ex.head == :call
        if ex.args[1] == :intercept
            return 1
        elseif ex.args[1] == :fixed && length(ex.args) > 1
            # Return the variable inside fixed(), e.g., fixed(x) -> x
            return ex.args[2]
        end
        # Recursively process the arguments of any other function call.
        return Expr(ex.head, _replace_TethysBayes_modules_in_expr.(ex.args)...)
    elseif ex isa Expr
        # Recursively process the arguments of any other expression type.
        return Expr(ex.head, _replace_TethysBayes_modules_in_expr.(ex.args)...)
    else
        # Base case: return non-expression atoms (symbols, literals) as is.
        return ex
    end
end
"""
    _TethysBayes_error_handler(e, model)
Provides detailed, user-friendly diagnostics when the initial prior predictive check
(`rand(model)`) fails.
# Version
v1.0.0
# Arguments
- `e`: The exception object caught during the prior predictive check.
- `model`: The instantiated Turing model object.
# Returns
- `nothing`. The function prints a diagnostic report to the console.
"""
function _TethysBayes_error_handler(e, model)
    println("\nERROR during prior predictive check (rand(m)):")
    showerror(stdout, e, stacktrace(catch_backtrace()))
    println("\n\n--- TethysBayes Diagnosis ---")
    if e isa DimensionMismatch
        println("A `DimensionMismatch` error occurred. This often points to an issue in the model's structure.")
        println("Potential Causes:")
        println("  1. Latent Field vs. Index Mismatch: The number of latent variables (e.g., `s_N`) does not match the number of unique levels in the corresponding index variable (e.g., `s_idx`).")
        println("  2. Matrix Multiplication: An operation like `X * beta` has incompatible dimensions.")
    elseif e isa BoundsError
        println("A `BoundsError` occurred. This means an index is out of range for an array.")
        println("Potential Causes:")
        println("  - This is common with `mixed()` or `spatial()` effects. The latent field vector is smaller than the maximum index in the corresponding index vector (e.g., `M.s_idx`).")
        println("  - Check for off-by-one errors or miscalculated numbers of levels (`n_cat`, `s_N`).")
    elseif e isa PosDefException
        println("A `PosDefException` occurred. A matrix that must be positive definite is not.")
        println("Potential Causes:")
        println("  1. GMRF Precision Matrix: For `icar` or `besag` models, ensure your adjacency matrix `W` corresponds to a single connected graph. Disconnected spatial 'islands' will cause this error.")
        println("  2. GP Covariance Matrix: A kernel matrix `K` is not positive definite, often due to very close data points. Try increasing the `noise` keyword argument (e.g., `noise=1e-5`).")
    elseif e isa KeyError
        println("A `KeyError` occurred. The model tried to access a parameter in the configuration that does not exist.")
        println("Potential Causes:")
        println("  - A typo in a variable name within the formula string.")
        println("  - A required parameter (e.g., `W` for a spatial model) was not passed as a keyword argument.")
    elseif e isa MethodError
        println("A `MethodError` occurred. This is often due to a type instability, especially with Automatic Differentiation (AD).")
        println("Potential Causes:")
        println("  - A custom function or prior distribution is not compatible with Turing's AD backend (e.g., ForwardDiff.jl). Ensure custom code handles `Dual` number types.")
        println("  - A parameter was expected to be one type (e.g., a Vector) but was another (e.g., a scalar).")
    elseif e isa ArgumentError
        println("An `ArgumentError` occurred. A function or distribution was called with an invalid argument (e.g., a negative standard deviation).")
        println("Potential Causes:")
        println("  - A prior distribution is misspecified (e.g., `Normal(0, -1)`). Check priors in your formula.")
    elseif e isa UndefVarError
        println("An `UndefVarError` occurred: `$(e.var)` is not defined.")
        println("Potential Causes:")
        println("  - A variable in the formula (e.g., `fixed(my_var)`) is not a column in the DataFrame.")
        println("  - A variable for a keyword argument (e.g., `W=my_matrix`) is not defined in the script's scope.")
    else
        println("An unexpected error occurred. General debugging tips:")
        println("  - Review the generated model code printed above the error.")
        println("  - Use `show_model(m)` to inspect the full model configuration.")
        println("  - Simplify your model formula by removing components one by one to isolate the error.")
    end
    println("------------------------")
    # Suggest simplified formulas to help the user debug.
    println("\n--- Suggested Debugging Steps ---")
    try
        formula_str = model.args.M.formula
        lhs, rhs_raw = Base.split(formula_str, '~')
        lhs = Base.strip(lhs)
        rhs_normalized = replace(Base.strip(rhs_raw), r"\s*-\s*" => " + -")
        all_terms = split_terms_at_depth(rhs_normalized, " + ")
        has_intercept = !any(in.(Base.strip.(all_terms), (["0", "-1"],
            ))) && !any(startswith.(Base.strip.(all_terms), "intercept(false"))
        base_rhs = has_intercept ? "1" : "0"
        println("1. Start with the simplest possible model to isolate the issue.")
        println("   This helps determine if the error is in your `likelihood()` definition or in the model components.")
        println("\n   Suggested base model:")
        println("   @tbm(\n       $lhs ~ $base_rhs,\n       data, ...\n   )")
        structural_terms = filter(t -> !in(Base.strip(t), ["1", "0", "-1"])&&
            !startswith(Base.strip(t), "intercept("), all_terms)
        if !isempty(structural_terms)
            println("\n2. If the base model works, add components back one by one to find the problematic term.")
            println("   For example, try the following formulas in order:")
            current_formula_rhs = base_rhs
            for (i, term) in enumerate(structural_terms)
                current_formula_rhs *= " + " * term
                println("\n   Step $i: Add '$term'")
                println("   @tbm(\n       $lhs ~ $current_formula_rhs,\n       data, ...\n   )")
            end
        end
    catch e_sugg
        println("\nError during suggestion generation: $e_sugg")
    end
    println("---------------------------------\n")
end
"""
    _transform_pair_macro_expr(pair_ex)
Transforms a pair AST expression `key => (formula = ..., data = ...)` so formula
expressions are cleanly converted to strings at macro expansion time, while runtime
variables (like DataFrames, priors, and mapping vectors) are properly escaped.
"""
function _transform_pair_macro_expr(pair_ex)
    if !(pair_ex isa Expr && pair_ex.head == :call && length(pair_ex.args) == 3 &&
         pair_ex.args[1] == :(=>))
        return esc(pair_ex)
    end
    key_part = esc(pair_ex.args[2])
    spec_part = pair_ex.args[3]
    if spec_part isa Expr && spec_part.head == :tuple
        transformed_args = Any[]
        for item in spec_part.args
            if item isa Expr && (item.head == :(=) || item.head == :kw)
                k = item.args[1]
                v = item.args[2]
                if k == :formula
                    v_str = v isa String ? v : string(v)
                    push!(transformed_args, Expr(:(=), k, v_str))
                else
                    push!(transformed_args, Expr(:(=), k, esc(v)))
                end
            else
                push!(transformed_args, esc(item))
            end
        end
        return Expr(:call, :(=>), key_part, Expr(:tuple, transformed_args...))
    else
        return Expr(:call, :(=>), key_part, esc(spec_part))
    end
end
macro tbm(exprs...)
    # --- detect assignment form: @tbm name = rest... ---
    local var_name = nothing
    local expressions_to_parse = exprs
    if !isempty(exprs) && exprs[1] isa Expr && exprs[1].head == :(=)
        var_name = exprs[1].args[1]
        expressions_to_parse = (exprs[1].args[2], exprs[2:end]...)
    end
    # --- collect positional args and raw keyword expressions (preserve order) ---
    positional = Any[]
    raw_kwargs = Expr[]    # will hold Expr(:kw, key, val) or elements from a :parameters node
    for ex in expressions_to_parse
        if ex isa Expr && ex.head == :parameters
            # things after semicolon — ex.args is a list of Expr(:kw, key, val)
            append!(raw_kwargs, ex.args)
        elseif ex isa Expr && ex.head == :kw
            push!(raw_kwargs, ex)
        elseif ex isa Expr && ex.head == :(=)
            # Keyword argument before semicolon (e.g., verbose = true)
            push!(raw_kwargs, Expr(:kw, ex.args[1], ex.args[2]))
        else
            push!(positional, ex)
        end
    end
    # --- check if positional arguments are pairs: @tbm :primary => (...), :proxy => (...) ---
    _is_pair_expr(ex) = (ex isa Expr && ex.head == :call && length(ex.args) == 3 &&
                         ex.args[1] == :(=>))
    if !isempty(positional) && all(_is_pair_expr, positional)
        transformed_pairs = Expr[]
        for p in positional
            push!(transformed_pairs, _transform_pair_macro_expr(p))
        end
        final_kwargs = Expr[]
        for kw in raw_kwargs
            if kw isa Expr && kw.head == :kw
                push!(final_kwargs, kw)
            end
        end
        kwargs_esc = [esc(kw) for kw in final_kwargs]
        core_logic = :(TethysBayes_core($(transformed_pairs...); calling_module = $(__module__), $(kwargs_esc...)))
        if !isnothing(var_name)
            return :($(esc(var_name)) = $core_logic)
        else
            return core_logic
        end
    end
    # --- convert raw_kwargs into a Dict for lookup (but keep raw_kwargs for order) ---
    kwdict = Dict{Symbol, Any}()
    for kw in raw_kwargs
        if kw isa Expr && kw.head == :kw
            key = kw.args[1]
            val = kw.args[2]
            kwdict[key] = val
        end
    end
    # --- resolve formula and data: keywords take precedence over positional ---
    formula_pos_idx = 0
    if haskey(kwdict, :formula)
        formula_expr = kwdict[:formula]
        delete!(kwdict, :formula)
    elseif length(positional) >= 1
        formula_expr = positional[1]
        formula_pos_idx = 1
    else
        error("The @tbm macro requires a formula; supply it positionally or with `formula=`.")
    end
    data_pos_idx = 0
    if haskey(kwdict, :data)
        data_expr = kwdict[:data]
        delete!(kwdict, :data)
    elseif length(positional) >= (formula_pos_idx + 1)
        data_pos_idx = formula_pos_idx + 1
        data_expr = positional[data_pos_idx]
    else
        error("The @tbm macro requires a data frame; supply it positionally or with `data=`.")
    end
    # Warn if there are unexpected extra positional arguments
    consumed_pos = max(formula_pos_idx, data_pos_idx)
    if length(positional) > consumed_pos
        extra_args = positional[(consumed_pos + 1):end]
        @warn "Ignoring extra positional arguments: $(extra_args)"
    end
    # Promote spatial graph / data parameters inside formula_expr to macro keyword arguments
    # so they are cleanly resolved in the caller's lexical scope (including local function variables)
    function _promote_context_args!(ex)
        if ex isa Expr
            if (ex.head == :kw || ex.head == :(=)) && length(ex.args) == 2
                k = ex.args[1]
                v = ex.args[2]
                promote_keys = (
                    :W, :Q, :habitat, :centroids, :sources, :sinks
                )
                if k in promote_keys && !haskey(kwdict, k)
                    kwdict[k] = v
                    push!(raw_kwargs, Expr(:kw, k, v))
                end
            end
            for a in ex.args
                _promote_context_args!(a)
            end
        end
    end
    _promote_context_args!(formula_expr)
    # --- reconstruct ordered keyword list excluding formula/data ---
    final_kwargs = Expr[]
    for kw in raw_kwargs
        # each kw is Expr(:kw, key, val); skip formula/data
        if kw isa Expr && kw.head == :kw
            ksym = kw.args[1]
            if ksym == :formula || ksym == :data
                continue
            end
            push!(final_kwargs, kw)
        end
    end
    # --- prepare for insertion into generated call ---
    #
    # `formula_expr` can be three different things:
    #
    #   1. DSL syntax written inline -- `likelihood(y) ~ intercept()`.
    #   2. A `String` literal -- `formula = "y ~ x"`. Used directly.
    #   3. A variable or expression evaluated at run time.
    formula_runtime = if formula_expr isa String
        nothing
    elseif formula_expr isa Symbol
        :($(esc(formula_expr))::String)
    elseif formula_expr isa Expr && formula_expr.head == :string
        :($(esc(formula_expr))::String)
    else
        nothing
    end
    formula_str = if isnothing(formula_runtime)
        string(formula_expr)
    else
        formula_runtime
    end
    data_esc = esc(data_expr)
    kwargs_esc = [esc(kw) for kw in final_kwargs]
    core_logic = :(TethysBayes_core($formula_str, $data_esc, $(__module__); $(kwargs_esc...)))
    if !isnothing(var_name)
        return :($(esc(var_name)) = $core_logic)
    else
        return core_logic
    end
end

# Backward-compatibility aliases
const var"@TethysBayes" = var"@tbm"
const var"@bstm" = var"@tbm"
"""
    _print_param(name, value, status; indent=4)
A helper function to print a single parameter with its value and status in a
standardized format. It ensures consistent indentation and safely truncates long
string representations to maintain readability.
# Arguments
- `name`: The name of the parameter.
- `value`: The value of the parameter.
- `status`: A `Symbol` indicating if the value was `:user` provided or a `:default`.
- `indent`: The indentation level for printing.
"""
function _print_param(name, value, status; indent=4)
    indent_str = " " ^ indent
    status_str = status == :user ? "(User-provided)" : "(Default)"
    value_str = string(value)
    # Safely truncate long values to prevent cluttering the console.
    if length(value_str) > 70
        value_str = first(value_str, 67) * "..."
    end
    println("$indent_str- $(rpad(name, 20)): $(value_str)  $status_str")
end



"""

    _print_finalized_parameters(config::NamedTuple)
Prints a comprehensive and well-formatted summary of all finalized parameters for each
module used in the `TethysBayes` model. This function is called when `verbose=true`.
It details the configuration for the likelihood, intercept, fixed effects, and all
random/smooth components. For each parameter, it shows the final value that will be
used in the model and indicates whether this value was explicitly provided by the user
or if it was assigned a system default. This provides clarity on the model's exact
specification before sampling begins.
"""
function _print_finalized_parameters(config::NamedTuple)
    println("\n--- Finalized Model Configuration ---")
    # 1. Likelihood Configuration
    println("\n[ Likelihood ]")
    lik_param_defs = [
        (:family, "gaussian"), (:log_offsets, "0.0"), (:weights, "1.0"),
        (:trials, "1"), (:zero_inflated, false), (:volatility, false),
        (:censor_lower, -Inf), (:censor_upper, Inf), (:hurdle, -Inf)
    ]
    # Add latent_dist for ordinal family
    push!(lik_param_defs, (:latent_dist, :logistic))
    for (i, spec) in enumerate(config.likelihood_specs)
        outcome = config.outcomes[i]
        println("  Outcome: $outcome")
        user_params = spec # `spec` itself is already the params dictionary
        for (p_name, p_default) in lik_param_defs
            final_val = get(user_params, p_name, p_default)
            status = haskey(user_params, p_name) ? :user : :default
            _print_param(p_name, final_val, status)
        end
    end
    # 2. Intercept Configuration
    println("\n[ Intercept ]")
    if config.add_intercept
        is_user_provided = haskey(config,
            :intercept_prior) && config.intercept_prior != Normal(0, 5)
        _print_param(:prior, get(config, :intercept_prior, Normal(0, 5)),
            is_user_provided ? :user : :default; indent=2)
    else
        println("  - Intercept removed from model.")
    end
    # 3. Fixed Effects Configuration
    if get(config, :Xfixed_N, 0) > 0
        println("\n[ Fixed Effects ]")
        println("  Formula: ~ $(config.Xfixed_applied_formula.rhs)")
        if haskey(config, :contrasts) && !isempty(config.contrasts)
            println("  Contrasts:")
            for (var, cont) in config.contrasts
                println("    - $var: $(typeof(cont))")
            end
        end
        println("  Priors per Coefficient:")
        for (i, name) in enumerate(config.Xfixed_names)
            prior_obj = config.Xfixed_priors_vec[i]
            is_default = prior_obj == Normal(0, 5)
            _print_param(name, prior_obj, is_default ? :default : :user; indent=4)
        end
    end
    # 4. Model Components (random, smooth, etc.)
    if !isempty(config.components)
        println("\n[ Model Components ]")
        for spec in config.components
            println("  --- Component: $(spec.key) ---")
            component_obj = spec.component_obj
            model_type_sym = Symbol(lowercase(string(typeof(component_obj))))
            println("    - Type:      $(typeof(component_obj))")
            println("    - Structure: $(spec.structure)")
            println("    - Variable:  $(spec.var)")
            latent_dim_val = 0
            # Corrected access: Q_template is inside spec.hyper
            if hasproperty(spec.hyper, :Q_template) && spec.hyper.Q_template isa AbstractMatrix
                latent_dim_val = size(spec.hyper.Q_template, 1)
            elseif hasproperty(component_obj, :nbins)
                latent_dim_val = component_obj.nbins
            elseif hasproperty(component_obj, :n_features)
                latent_dim_val = component_obj.n_features
            elseif hasproperty(component_obj, :n_inducing)
                latent_dim_val = component_obj.n_inducing
            elseif hasproperty(spec.hyper,
                :n_latent) # Fallback for components that explicitly define n_latent in hyper
                latent_dim_val = spec.hyper.n_latent
            end
            if latent_dim_val > 0
                println("    - Latent Field Dimension: $(latent_dim_val)")
            end
            println("    - Parameters:")
            user_provided_params_raw = spec.params 
            # Combine struct fields (priors) and config args
            all_param_names = Set(fieldnames(typeof(component_obj)))
            if haskey(COMPONENT_CONFIG_ARGS, model_type_sym)
                union!(all_param_names, keys(COMPONENT_CONFIG_ARGS[model_type_sym]))
            end
            for param_name in sort(collect(all_param_names))
                if param_name in [:positional_args, :structure, :in_dims]
                    continue
                end
                local final_val, status
                if param_name in fieldnames(typeof(component_obj))
                    # It's a hyperparameter (prior)
                    final_val = getfield(component_obj, param_name)
                    status = haskey(user_provided_params_raw, param_name) ? :user : :default
                else
                    # It's a configuration argument
                    default_val = get(COMPONENT_CONFIG_ARGS, model_type_sym,
                        Dict()) |> d -> get(d, param_name, "N/A")
                    final_val = get(user_provided_params_raw, param_name, default_val)
                    status = haskey(user_provided_params_raw, param_name) ? :user : :default
                end
                if final_val isa Vector{<:UnivariateDistribution}
                    println("      - $(rpad(param_name, 20)): [")
                    for (idx, p_dist) in enumerate(final_val)
                        println("        $idx: $p_dist")
                    end
                    println("      ] $(status == :user ? "(User-provided)" : "(Default)")")
                else
                    _print_param(param_name, final_val, status; indent=6)
                end
            end
        end
    end
    println("\n-------------------------------------\n")
end
"""
    WorldAgeTrampoline{F}
Callable wrapper that defers to a runtime-generated function via `Base.invokelatest`.
`TethysBayes` generates the Turing model body at runtime with `Core.eval`, which defines a
brand-new *method* at call time. Julia resolves world age at the call site, so
already-compiled code — including DynamicPPL's `_evaluate!!` — cannot see that method
and raises
    MethodError: no method matching TethysBayes_dynamic_model_XXXX(::Model, ::AbstractVarInfo, ...)
    Closest candidates are: ... (method too new to be called from this world context.)
which breaks the ordinary
    function run()
        m = @tbm(...)
        sample(m, NUTS())   # <- MethodError
    end
pattern whenever the model is constructed inside a function rather than at top level.
Routing every model evaluation through this type fixes it: the *call method* of
`WorldAgeTrampoline` is defined once at precompile time, so it is visible in every
world, while the wrapped closure is reached through `invokelatest`. The cost is
measured at ~25 ns per call, against a DynamicPPL density evaluation on the order of
10-100 µs — well under 0.25% of a single MCMC leapfrog step.
A struct with its own call overload is used rather than `Base.Fix2` because `Fix2` is a
`map` adapter with no call method, so calling it directly falls into broadcasting.
"""
struct WorldAgeTrampoline{F}
    f::F
end
@inline (t::WorldAgeTrampoline)(args...; kwargs...) =
    Base.invokelatest(t.f, args...; kwargs...)
"""
    make_world_age_safe(model::DynamicPPL.Model) -> DynamicPPL.Model
Return `model` with its evaluation function wrapped in a [`WorldAgeTrampoline`](@ref)
so it can be sampled from inside any function scope, not just top level. Idempotent:
a model whose function is already a `WorldAgeTrampoline` is returned unchanged.
"""
function make_world_age_safe(model::DynamicPPL.Model)
    model.f isa WorldAgeTrampoline && return model
    tramp = WorldAgeTrampoline(model.f)
    # `requires_threadsafe` is a DynamicPPL internal; fall back to the (unthreaded)
    # default if a future version renames it.
    threaded = if isdefined(DynamicPPL, :requires_threadsafe)
        DynamicPPL.requires_threadsafe(model)
    else
        false
    end
    return DynamicPPL.Model{threaded}(tramp, model.args, model.defaults, model.context)
end
"""
    TethysBayes_core(formula::String, data::DataFrame, calling_module::Module; kwargs...)
The main entry point for the `@tbm` macro. This function orchestrates the configuration,
code generation, and instantiation of a Turing model.
# Version
v1.0.0

# Arguments
- `formula::String`: The model formula.
- `data::DataFrame`: The input data.
- `calling_module::Module`: The module in which the model was called (used for context).
- `kwargs...`: Additional keyword arguments passed to `TethysBayes_config`.

# Returns
- An instantiated and validated Turing model object.
"""
function TethysBayes_core(formula::String, data::DataFrame, calling_module::Module; kwargs...)
    # --- 1. Configuration ---
    # Generate model configuration dictionary based on formula syntax and data schema.
    options = TethysBayes_config(formula, data; calling_module=calling_module,
        kwargs...) # Pass kwargs to TethysBayes_config
    # --- 2. Code Generation ---
    # Generate a unique name for the model function to avoid world age issues.
    random_suffix = rand(10000:99999)
    model_func_name = Symbol("TethysBayes_dynamic_model_$(random_suffix)")
    # Call the text assembler to get the model's source code, expression, and registry.
    model_string, expr, registry = TethysBayes_text_assembler(options, model_func_name)
    # Update the configuration with the generated model code for inspection.
    config_dict = Dict(pairs(options))
    config_dict[:generated_model_code] = model_string
    delete!(config_dict, :calling_module)
    new_config = NamedTuple(config_dict)
    if get(new_config, :verbose, true)
        _print_finalized_parameters(new_config)
    end

    # --- 3. Model Evaluation and Instantiation ---
    model_function = Core.eval(_GeneratedModelRuntime, quote
        $(expr)
        $(model_func_name)
    end)
    # Instantiate the Turing Model Object using invokelatest to prevent world age issues
    model_instance = Base.invokelatest(model_function, new_config, registry)
    # Route evaluation through a precompiled trampoline so the model can also be
    # sampled from inside a function scope (see `WorldAgeTrampoline`).
    model_instance = make_world_age_safe(model_instance)
    # --- 4. Prior Predictive Check and Validation ---
    if get(new_config, :verbose, true)
        println("\n--- Running prior predictive check ---")
    end
    prior_sample = nothing
    try
        # Run a single draw from the prior to validate the model structure.
        prior_sample = Base.invokelatest(rand, model_instance)
        if get(new_config, :verbose, true) && !isnothing(prior_sample)
            println("Prior sample check successful. Sample values:")
            display(prior_sample)
        end
        # Calibrate and store centralized ParamRegistry in spec_registry
        current_reg = get(registry, :parameters, build_param_registry(new_config))
        if !isnothing(prior_sample)
            registry[:parameters] = calibrate_param_registry(current_reg, prior_sample)
        else
            registry[:parameters] = current_reg
        end
    catch e 
        # Provide detailed, user-friendly error diagnostics if the check fails.
        _TethysBayes_error_handler(e, model_instance)
    end
    if get(new_config, :verbose, true)
        println("--------------------------------------\n")
    end
    # Return the fully configured and validated model object.
    return model_instance
end




"""

    TethysBayes_core(formula::String, data::DataFrame; kwargs...)
A convenience overload for the main `TethysBayes` constructor. This method defaults the
model's evaluation scope to the `Main` module.

# Arguments
- `formula::String`: The model formula.
- `data::DataFrame`: The input data.
- `kwargs...`: Additional keyword arguments passed to the main `TethysBayes` constructor.

# Returns
- An instantiated Turing model object.
"""

function TethysBayes_core(formula::String, data::DataFrame; kwargs...)

    return TethysBayes_core(formula, data, Main; kwargs...)

end


function TethysBayes_core(formula::Union{Expr, Symbol}, data::DataFrame, calling_module::Module=Main;
    kwargs...)
    return TethysBayes_core(string(formula), data, calling_module; kwargs...)
end



"""

    TethysBayes_core(equation_pairs::Pair{Symbol, <:Union{NamedTuple, AbstractDict}}...; calling_module::Module=Main, kwargs...)
Constructs and instantiates a Turing model from a system of paired multi-fidelity equations.

"""
function TethysBayes_core(
    equation_pairs::Pair{Symbol, <:Union{NamedTuple, AbstractDict}}...;
    calling_module::Module = Main,
    kwargs...
)
    options = TethysBayes_config(equation_pairs...; calling_module = calling_module, kwargs...)
    random_suffix = rand(10000:99999)
    model_func_name = Symbol("TethysBayes_dynamic_model_$(random_suffix)")
    model_string, expr, registry = TethysBayes_text_assembler(options, model_func_name)
    config_dict = Dict(pairs(options))
    config_dict[:generated_model_code] = model_string
    delete!(config_dict, :calling_module)
    new_config = NamedTuple(config_dict)
    if get(new_config, :verbose, true)
        _print_finalized_parameters(new_config)
    end

    model_function = Core.eval(_GeneratedModelRuntime, quote
        $(expr)
        $(model_func_name)
    end)
    model_instance = Base.invokelatest(model_function, new_config, registry)
    # Route evaluation through a precompiled trampoline so the model can also be
    # sampled from inside a function scope (see `WorldAgeTrampoline`).
    model_instance = make_world_age_safe(model_instance)
    if get(new_config, :verbose, true)
        println("\n--- Running prior predictive check ---")
    end
    prior_sample = nothing
    try
        prior_sample = Base.invokelatest(rand, model_instance)
        if get(new_config, :verbose, true) && !isnothing(prior_sample)
            println("Prior sample check successful. Sample values:")
            display(prior_sample)
        end
        current_reg = get(registry, :parameters, build_param_registry(new_config))
        if !isnothing(prior_sample)
            registry[:parameters] = calibrate_param_registry(current_reg, prior_sample)
        else
            registry[:parameters] = current_reg
        end
    catch e
        _TethysBayes_error_handler(e, model_instance)
    end
    if get(new_config, :verbose, true)
        println("--------------------------------------\n")
    end
    return model_instance
end


"""
    _streamline_multivariate_updates(text::String)::String
Streamlines generated Turing code for multivariate models to eliminate dynamic array
allocations during vectorized linear predictor accumulations. Converts indexed
assignments `eta_latent[:, k] = eta_latent[:, k] .+ ...` into
`@views eta_latent[:, k] .+= ...` and whole matrix updates
`eta_latent = eta_latent .+ ...` into `eta_latent .+= ...`.
This reduces memory footprint during HMC/NUTS iterations and ensures compatibility
with ReverseDiff AD (which does not support `setindex!` on sliced TrackedArrays).
"""
function _streamline_multivariate_updates(text::String)::String
    s1 = replace(
        text,
        r"(\b\w*eta_latent\w*\[[^\]]+\])\s*=\s*\1\s*\.\+\s*" => s"@views \1 .+= "
    )
    s2 = replace(
        s1,
        r"\b(\w*eta_latent\w*)\s*=\s*\1\s*\.\+\s*" => s"\1 .+= "
    )
    return s2
end


"""

    TethysBayes_text_assembler(config::NamedTuple, model_func_name::Symbol)
Assembles the full Turing model code as a string and a Julia `Expr` from the
provided configuration.
# Version
v1.0.0
# Arguments
- `M::NamedTuple`: The complete model configuration object.
- `model_func_name::Symbol`: The unique name for the generated Turing model function.
# Returns
- A tuple `(model_string, expr, registry)`.
"""
function TethysBayes_text_assembler(M::NamedTuple, model_func_name::Symbol)
    arch = get(M, :model_arch, "univariate")
    is_multivariate = arch == "multivariate"
    eta_name = is_multivariate ? "eta_latent" : "eta"
    eta_init = if get(M, :add_intercept, false)
        if is_multivariate
            "zeros(T, N, K) .+ reshape(intercept, 1, K)"
        else
            "intercept .+ zeros(T, N)"
        end
    else
        is_multivariate ? "zeros(T, N, K)" : "zeros(T, N)"
    end

    outcomes_N = get(M, :outcomes_N, 1)
    spec_registry = Dict{Symbol, Any}()
    
    priors_acc = String[]
    updates_acc = String[]
    likelihood_acc = String[]
    main_spatial_spec = nothing
    main_temporal_spec = nothing
    

    has_custom_likelihood_from_component = any(
        spec -> (spec.component_obj isa PointProcess || (hasproperty(spec.component_obj,
            :method) && spec.component_obj.method == :marginalized)),
        M.components
    )
    has_custom_likelihood_from_family = any(spec -> string(get(spec, :family,
        "")) == "ordinal", M.likelihood_specs)
    has_custom_likelihood = has_custom_likelihood_from_component ||
        has_custom_likelihood_from_family
    # --- Generate all code fragments ---
    intercept_priors, _ = _generate_intercept_block(M, is_multivariate, eta_name)
    push!(priors_acc, intercept_priors)
    push!(updates_acc, _generate_offset_block(M, is_multivariate, eta_name))
    fixed_effects_priors, fixed_effects_update = _generate_fixed_effects_block(M,
        is_multivariate, eta_name)
    push!(priors_acc, fixed_effects_priors)
    push!(updates_acc, fixed_effects_update)
    if get(M, :is_multivariate_dynamics, false)
        mv_dyn_key = M[:multivariate_dynamics_key]
        spec_idx = findfirst(s -> string(s.key) == mv_dyn_key, M.components)
        if !isnothing(spec_idx)
            spec = M.components[spec_idx]
            spec_registry[spec.key] = spec
            push!(priors_acc, get_priors(spec.component_obj, spec, arch, nothing, M))
            push!(updates_acc, get_updates(spec.component_obj, spec, arch, nothing, M))
        end
    end



    for spec in M.components

        if get(M, :is_multivariate_dynamics, false) && string(spec.key) == M[:multivariate_dynamics_key]
            continue
        end

        spec_registry[spec.key] = spec

        if hasproperty(spec, :hyper) && hasproperty(spec.hyper, :child_specs)
            for cs in spec.hyper.child_specs
                spec_registry[cs.key] = cs
                if hasproperty(cs, :hyper) && hasproperty(cs.hyper, :child_specs)
                    for gcs in cs.hyper.child_specs
                        spec_registry[gcs.key] = gcs
                    end
                end
            end
        end

        for k in 1:outcomes_N
            outcome_idx = is_multivariate ? k : nothing
            push!(priors_acc, get_priors(spec.component_obj, spec, arch, outcome_idx, M))
            push!(updates_acc, get_updates(spec.component_obj, spec, arch, outcome_idx, M))
        end

        if spec.structure == :spatial && isnothing(main_spatial_spec)
            main_spatial_spec = spec
        end
        if spec.structure == :temporal && isnothing(main_temporal_spec)
            main_temporal_spec = spec
        end
    end


    push!(priors_acc, _generate_likelihood_section(M, is_multivariate))
    st_interaction_block = _generate_st_interaction_block(M, main_spatial_spec,
        main_temporal_spec, is_multivariate, eta_name)
    push!(updates_acc, st_interaction_block)
    householder_priors, householder_update = _generate_householder_reflection_block(M,
        is_multivariate, eta_name)
    push!(priors_acc, householder_priors)
    push!(updates_acc, householder_update)
    nested_priors, nested_updates, nested_likelihoods = _generate_nested_model_block(M,
        is_multivariate, eta_name)
    push!(priors_acc, nested_priors)
    push!(updates_acc, nested_updates)
    push!(likelihood_acc, nested_likelihoods)
    final_likelihood = if has_custom_likelihood
        has_custom_likelihood_from_family ? _generate_final_likelihood_block(M,
            is_multivariate) : ""
    else
        _generate_final_likelihood_block(M, is_multivariate)
    end

    push!(likelihood_acc, final_likelihood)
    
    # --- Assemble the final model string ---
    function _indent_block(text::String, level=1)
        if isempty(strip(text)) return "" end
        indent_str = "    " ^ level
        return indent_str * replace(strip(text), "\n" => "\n" * indent_str)
    end
    priors_code = join(filter(s -> !isempty(strip(s)), priors_acc), "\n\n")
    updates_raw = join(filter(s -> !isempty(strip(s)), updates_acc), "\n\n")
    updates_code = if is_multivariate
        _streamline_multivariate_updates(updates_raw)
    else
        updates_raw
    end
    likelihood_code = join(filter(s -> !isempty(strip(s)), likelihood_acc), "\n\n")
    model_string = """
    @model function $(model_func_name)(M, spec_registry)
        noise = M.noise
        N = M.y_N
        K = $(outcomes_N)
        T = _model_float_type(__varinfo__)
        # --- Priors & Hyperparameters ---
    $(_indent_block(priors_code))
        # --- Linear Predictor Assembly ---
        $(eta_name) = $(eta_init)
    $(_indent_block(updates_code))
        # --- Likelihood ---
    $(_indent_block(likelihood_code))
    end
    """
    # The literal above ends with a blank line, and `Meta.parse` rejects a complete
    # expression followed by one ("extra token after end of expression"). Trim it here
    # rather than relying on the layout of the triple-quoted block staying as-is.
    model_string = rstrip(model_string)
    spec_registry[:parameters] = build_param_registry(M)
    # A component that omits part of the interface fails here, attributed to the
    # component, instead of as a MethodError from deep inside model construction.
    validate_all_component_interfaces()
    try
        return model_string, parse_generated_model(model_string), spec_registry
    catch e
        report_generated_model_error(model_string, e)
        rethrow(e)
    end
end

"""
    parse_generated_model(model_string::AbstractString)::Expr

Parse assembled model code into an `Expr`, and fail with a located diagnostic rather than
an opaque `ParseError`.

`Meta.parseall` is used deliberately instead of `Meta.parse`: `parse` stops after the first
expression and its "extra token after end of expression" message says nothing about *where*
the problem is, which is exactly the failure the old `try Meta.parse(model_string)` produced
whenever a component emitted a stray blank line. Any `:error`/`:incomplete` node in the
result is reported with its line and the source line itself.
"""
function parse_generated_model(model_string::AbstractString)
    parsed = Meta.parseall(model_string)
    for node in parsed.args
        if node isa Expr && node.head in (:error, :incomplete)
            # `haskey(::Expr, ::line)` is a MethodError on Julia 1.13 -- `Expr` supports
            # `:line`/`:file` as properties, not as keys. Using it here meant the
            # parse-error reporter itself threw a MethodError, replacing a located
            # diagnostic with an unrelated one, and only when something was already wrong.
            line = try
                Int(node.line)
            catch
                0
            end
            detail = node.args[1]
            src = ""
            if 0 < line
                all = collect(eachline(IOBuffer(model_string)))
                line <= length(all) && (src = "\n  line $line: " * all[line])
            end
            error("Generated model code is not valid Julia: $(detail)$(src)")
        end
    end
    return parsed
end

"""
    report_generated_model_error(model_string, err)

Print the offending generated code with line numbers before rethrowing, so a codegen
regression can be read without re-running under a debugger.
"""
function report_generated_model_error(model_string::AbstractString, err)
    println("TethysBayes Assembler Error: the generated model code failed to parse.")
    numbered = String[]
    for (i, l) in enumerate(eachline(IOBuffer(model_string)))
        push!(numbered, string(lpad(i, 5), " | ", l))
    end
    println(join(numbered, "\n"))
    return err
end
"""
    resolve_technical_primitive(module_metadata::Dict{Symbol, Any}, M, priors_dict, scheme::Symbol)
Instantiates a `ComponentModel` object from its parsed formula representation. This function
acts as a factory, resolving hyperpriors and calling the appropriate constructor from the
`COMPONENT_CONSTRUCTORS` registry.
# Version
v1.0.0
# Arguments
- `module_metadata::Dict`: The parsed data for the module from the formula.
- `M`: The main model configuration dictionary.
- `priors_dict::Dict`: A dictionary of globally specified hyperpriors.
- `scheme::Symbol`: The active prior scheme (e.g., `:pcpriors`).
# Returns
- A `ComponentModel` object (e.g., an instance of `BYM2`, `AR1`, or `Composed`).
"""
function resolve_technical_primitive(module_metadata::Dict{Symbol, Any}, M, priors_dict,
    scheme::Symbol)
    m_type = module_metadata[:type]
    m_params = module_metadata[:params]
    calling_mod = get(M, :calling_module, Main)
    # Handle composed components recursively.
    if m_type == :interact
        op = m_params[:operator]
        components_data = m_params[:components]
        components_metadata = map(c_node -> Dict(:key=>"temp", :type => c_node.module_type,
            :params => c_node.args, :variables => get(c_node.args, :positional_args, [])),
            components_data)
        resolved_components = [resolve_technical_primitive(comp_meta, M, priors_dict,
            scheme) for comp_meta in components_metadata]
        default_method = op == :kronecker_product ? :spectral : (op == :pipe ? :cholesky : :none)
        return Composed(resolved_components, op, get(m_params, :method, default_method))
    end
    # Handle standard components.
    model_name = if haskey(COMPONENT_CONSTRUCTORS, m_type)
        m_type
    elseif haskey(m_params, :model)
        m_params[:model] isa Symbol ? m_params[:model] : Symbol(m_params[:model])
    else
        # Infer default model based on the structure if no model is specified.
        if m_type == :spatial
            haskey(M, :W) ? :bym2 : :iid
        elseif m_type == :temporal
            :rw2
        else
            :iid
        end
    end
    model_name_str = string(model_name)
    resolved_priors = resolve_hyperpriors(model_name_str, priors_dict, m_params, scheme,
        calling_mod)
    if !haskey(COMPONENT_CONSTRUCTORS, model_name)
        error("Component model ':$model_name' is not a recognized model type.")
    end
    constructor_func = COMPONENT_CONSTRUCTORS[model_name]
    return constructor_func(resolved_priors, m_params)
end


"""
    _zero_null_modes!(diag_D::AbstractVector, L::AbstractVector; rel_tol=1e-10)

Zero every eigen-direction whose precision eigenvalue is numerically zero, in place, and
return the indices zeroed.

**Why not just `diag_D[1] = 0`.** That idiom imposed the intrinsic sum-to-zero constraint by
zeroing a *SINGLE* eigen-direction. It is correct only when the precision has exactly ONE null
direction. That holds for a **connected** spatial graph, but not in general: a disconnected
graph has as many null directions as it has connected components, and zeroing only the first
leaves the rest carrying variance \`sigma^2 / noise\` because a null eigenvalue \`L ~ 0\` makes
\`diag_D = sigma / sqrt(L + noise)\` *large*, not small. The opposite error is just as real: the
\`diag_D[1] = 0.0; diag_D[2] = 0.0\` form, hard-coded in the 2-D second-order penalty
components, deletes a genuine degree of freedom for a 1-D path penalty such as RW2, whose null
space is one-dimensional however the index set is arranged.

**All such sites have been converted.** Every \`diag_D\` deflation in \`src/components/\` now calls
this helper, in both the generated model body and the \`get_effects\` CPU reconstruction (they
must agree, or reconstruction stops reproducing what the model sampled). To add a component
back to the old idiom would reintroduce the defect; the count is not a property of the source
file, it is a property of the spectrum.

Measured with \`scripts/_oracle_icar.jl\` on a 5-unit, 2-component chain: the connected graph
matches \`pinv(Q)\` to 4e-9, but the disconnected one puts **1e9** of variance on the null
space and yields a field whose mean is **2.0e8** instead of 0, at the default \`noise\`. The
\`island_handling\` option does not help: all four modes produced byte-identical templates,
despite the Info message claiming "component deflation" was applied.

The tolerance is **relative** to the largest eigenvalue, so it means the same thing for a
precision scaled to 1e6 as for one scaled to 1e-6. An absolute threshold would silently
classify every direction of a small-scale precision as null. A degenerate template with no
positive eigenvalue at all (fully isolated units) has no scale to compare against, so every
direction is treated as null rather than dividing by ~0.

On a connected graph this is a no-op, since the single null direction is index 1 anyway, so
it is safe wherever the constraint is intended.
"""
function _zero_null_modes!(diag_D::AbstractVector, L::AbstractVector; rel_tol::Real = 1e-10)
    isempty(L) && return Int[]
    lmax = maximum(L)
    tol = lmax > 0 ? rel_tol * lmax : rel_tol
    null_idx = findall(<=(tol), L)
    for i in null_idx
        diag_D[i] = 0.0
    end
    return null_idx
end


"""
    build_structure_template(
        model_type::Symbol,
        n::Int;
        W::Union{AbstractMatrix, Nothing} = nothing,
        island_handling::Symbol = :normalize,
        deflation_gamma::Real = 1.0,
        check_components::Bool = true
    )
Creates a precision matrix template and its spectral decomposition for a GMRF model.
For spatial models with an adjacency matrix `W` (:icar, :besag, :bym2, :leroux,
:localadaptive), automatically performs connected component analysis to detect
disconnected components (sub-graphs) and isolated units (graph islands).
# Mathematical Formulation
1. **Single Connected Graph (K = 1)**:
   The singular spatial Laplacian is:
   Q = D - W_sym
   where W_sym = (W + W') / 2 (symmetrized binary adjacency) and
   D = diag(sum_j W_sym[i, j]).
   With rank deficiency 1 and null vector 1_n / sqrt(n), the scaling
   factor s is the geometric mean of the n - 1 positive eigenvalues
   (Riebler et al., 2016):
   s = exp( (1 / (n - 1)) * sum_{j=2}^n log(lambda_j) )
   The normalized precision matrix is Q* = Q / s.
2. **Disconnected Graphs & Islands (K > 1, Freni-Sterrantino et al., 2018)**:
   When the adjacency graph consists of K disjoint components C_1, ..., C_K:
   - **Sub-graph Island Normalization (`island_handling = :normalize`)**:
     For each sub-graph C_k with size n_k = |C_k| >= 2, the sub-graph
     Laplacian Q_k is extracted and scaled by its individual sub-graph factor:
     s_k = exp( (1 / (n_k - 1)) * sum_{j=2}^{n_k} log(lambda_{k,j}) )
     ensuring that each island has standardized average marginal variance of 1.0.
     For isolated singletons (n_k = 1, degree 0), assigning unit precision
     Q*[i, i] = 1.0 provides an independent standard Gaussian prior
     N(0, 1), preventing singular pivot failures during Cholesky
     factorization.
   - **Component Deflation**:
     The null space of Q* is spanned by the K_null orthonormal vectors:
     v_k[i] = 1 / sqrt(n_k) if i in C_k, and 0 otherwise.
     The orthogonal deflation projector:
     P_perp = I - V_0 * V_0' = I - sum_{k=1}^{K_null} v_k * v_k'
     enforces the sum-to-zero constraint sum_{i in C_k} phi_i = 0 independently
     on each spatial island.
     The deflated, strictly positive definite precision matrix:
     Q_deflated = Q* + gamma * V_0 * V_0'
     satisfies P_perp * inv(Q_deflated) * P_perp = (Q*)^+, matching the
     Moore-Penrose pseudo-inverse on the range space while being directly invertible.
# Arguments
- `model_type::Symbol`: Latent model type (`:icar`, `:besag`, `:bym2`, `:leroux`,
  `:localadaptive`, `:rw1`, `:rw2`, `:cyclic`, `:ar1`, `:iid`).
- `n::Int`: Dimension of the latent field (number of spatial units / time points).
- `W::Union{AbstractMatrix, Nothing}`: Spatial adjacency matrix (required for spatial models).
- `island_handling::Symbol`: Strategy for handling disconnected components (`:normalize`
  for sub-graph normalization + singleton regularization, or `:global` for single global scale).
- `deflation_gamma::Real`: Regularization parameter \$\\gamma\$ for shifting null space
  eigenvalues in `matrix_deflated` (default: 1.0).
- `check_components::Bool`: Whether to run automated connected component analysis (default: true).
# Returns
- `NamedTuple` containing:
  - `matrix`: Normalized structure / precision matrix \$Q^*\$ (`SparseMatrixCSC{Float64, Int}`).
  - `scaling_factor`: Effective global geometric mean scaling factor (`Float64`).
  - `U`: Orthonormal eigenvectors (`Matrix{Float64}`).
  - `L`: Eigenvalues in ascending order (`Vector{Float64}`).
  - `components`: Connected components as lists of node indices (`Vector{Vector{Int}}`).
  - `n_components`: Number of connected components \$K\$ (`Int`).
  - `subgraph_scaling_factors`: Vector of individual scaling factors \$s_k\$ (`Vector{Float64}`).
  - `rank_deficiency`: Number of zero eigenvalues / singular nullity (`Int`).
  - `null_vectors`: Orthonormal null-space basis matrix \$V_0\$ (`Matrix{Float64}`).
  - `projection_matrix`: Sparse orthogonal deflation projector \$P_\\perp\$
    (`SparseMatrixCSC{Float64, Int}`).
  - `matrix_deflated`: Deflated invertible precision matrix \$\\tilde{Q}\$
    (`SparseMatrixCSC{Float64, Int}`).
# References
- Riebler, A., Sørbye, S. H., Simpson, D., & Rue, H. (2016). An intuitive Bayesian spatial
  model for disease mapping that accounts for scaling. *Statistical Methods in Medical Research*,
  25(4), 1145-1165.
- Freni-Sterrantino, A., Ventrucci, M., & Rue, H. (2018). A note on intrinsic conditional
  autoregressive models for disconnected graphs. *Spatial and
  Spatio-temporal Epidemiology*, 26, 25-34.
"""
function build_structure_template(
    model_type::Symbol,
    n::Int;
    W::Union{AbstractMatrix, Nothing} = nothing,
    island_handling::Symbol = :normalize,
    deflation_gamma::Real = 1.0,
    check_components::Bool = true
)
    Q_template = spzeros(Float64, n, n)
    rank_deficiency = 0
    if n == 0
        return (
            matrix = Q_template,
            scaling_factor = 1.0,
            U = spzeros(Float64, 0, 0),
            L = Float64[],
            components = Vector{Int}[],
            n_components = 0,
            subgraph_scaling_factors = Float64[],
            rank_deficiency = 0,
            null_vectors = zeros(Float64, 0, 0),
            projection_matrix = spzeros(Float64, 0, 0),
            matrix_deflated = spzeros(Float64, 0, 0)
        )
    end
    if model_type in [:icar, :besag, :bym2, :leroux, :localadaptive]
        if isnothing(W)
            error("Spatial model '$model_type' requires an adjacency matrix `W`.")
        end
        if size(W, 1) != n || size(W, 2) != n
            error("Adjacency matrix `W` dimensions ($(size(W))) do not match `n` ($n).")
        end
        # Symmetrize binary adjacency and clear self-loops
        W_sym = sparse((W + W') .> 0)
        for i in 1:n
            W_sym[i, i] = false
        end
        dropzeros!(W_sym)
        # Automated connected component analysis
        g = SimpleGraph(W_sym)
        comps = connected_components(g)
        n_comps = length(comps)
        subgraphs = [c for c in comps if length(c) > 1]
        singletons = [c[1] for c in comps if length(c) == 1]
        n_subgraphs = length(subgraphs)
        n_singletons = length(singletons)
        if n_comps > 1 && check_components
            @info "Spatial adjacency W has $(n_comps) components " *
                  "($(n_subgraphs) sub-graphs, $(n_singletons) islands). " *
                  "Applying sub-graph island normalization and component deflation."
        end
        if island_handling == :normalize && n_comps > 1
            Q_norm = spzeros(Float64, n, n)
            s_factors = zeros(Float64, n_comps)
            # Sub-graph island normalization
            for (k, c) in enumerate(comps)
                n_k = length(c)
                if n_k == 1
                    # Isolated singleton island: assign unit precision
                    idx = c[1]
                    Q_norm[idx, idx] = 1.0
                    s_factors[k] = 1.0
                else
                    W_k = W_sym[c, c]
                    D_k = spdiagm(0 => vec(sum(W_k, dims=2)))
                    Q_k = D_k - W_k
                    eig_k = eigen(Symmetric(Matrix(Q_k)))
                    vals_k = eig_k.values
                    pos_vals_k = vals_k[2:end]
                    s_k = exp(mean(log.(pos_vals_k)))
                    s_factors[k] = s_k
                    Q_k_scaled = Q_k ./ s_k
                    for (local_i, global_i) in enumerate(c)
                        for (local_j, global_j) in enumerate(c)
                            val = Q_k_scaled[local_i, local_j]
                            if !iszero(val)
                                Q_norm[global_i, global_j] = val
                            end
                        end
                    end
                end
            end
            # Effective global scaling factor (weighted geometric mean of sub-graphs)
            multi_indices = [k for k in 1:n_comps if length(comps[k]) > 1]
            if isempty(multi_indices)
                effective_scale = 1.0
            else
                weights = [length(comps[k]) - 1 for k in multi_indices]
                effective_scale = exp(
                    sum(weights .* log.(s_factors[multi_indices])) / sum(weights)
                )
            end
            # Null space basis V0: one vector per multi-node sub-graph
            rank_deficiency = n_subgraphs
            V0 = zeros(Float64, n, rank_deficiency)
            for (col_idx, k) in enumerate(multi_indices)
                c = comps[k]
                norm_val = 1.0 / sqrt(length(c))
                for node_idx in c
                    V0[node_idx, col_idx] = norm_val
                end
            end
            P_perp = sparse(I, n, n) - sparse(V0 * V0')
            gamma_val = Float64(deflation_gamma)
            Q_deflated = Q_norm + sparse(gamma_val .* (V0 * V0'))
            eig_decomp = eigen(Symmetric(Matrix(Q_norm)))
            U = eig_decomp.vectors
            L = eig_decomp.values
            tol = max(1e-10, 1e-8 * maximum(abs, L))
            for i in 1:length(L)
                if abs(L[i]) <= tol
                    L[i] = 0.0
                end
            end
            return (
                matrix = Q_norm,
                scaling_factor = effective_scale,
                U = U,
                L = L,
                components = comps,
                n_components = n_comps,
                subgraph_scaling_factors = s_factors,
                rank_deficiency = rank_deficiency,
                null_vectors = V0,
                projection_matrix = P_perp,
                matrix_deflated = Q_deflated
            )
        else
            # Single connected component or global scaling fallback
            D = spdiagm(0 => vec(sum(W_sym, dims=2)))
            Q_template = D - W_sym
            rank_deficiency = max(1, n_subgraphs)
            eig_decomp = eigen(Symmetric(Matrix(Q_template)))
            U = eig_decomp.vectors
            L = eig_decomp.values
            scaling_factor = _compute_scaling_factor(L, rank_deficiency)
            Q_template = Q_template ./ scaling_factor
            L = L ./ scaling_factor
            multi_indices = [k for k in 1:n_comps if length(comps[k]) > 1]
            K_null = length(multi_indices)
            V0 = zeros(Float64, n, K_null)
            for (col_idx, k) in enumerate(multi_indices)
                c = comps[k]
                norm_val = 1.0 / sqrt(length(c))
                for node_idx in c
                    V0[node_idx, col_idx] = norm_val
                end
            end
            P_perp = sparse(I, n, n) - sparse(V0 * V0')
            gamma_val = Float64(deflation_gamma)
            Q_deflated = Q_template + sparse(gamma_val .* (V0 * V0'))
            return (
                matrix = Q_template,
                scaling_factor = scaling_factor,
                U = U,
                L = L,
                components = comps,
                n_components = n_comps,
                subgraph_scaling_factors = fill(scaling_factor, n_comps),
                rank_deficiency = rank_deficiency,
                null_vectors = V0,
                projection_matrix = P_perp,
                matrix_deflated = Q_deflated
            )
        end
    elseif model_type == :rw1
        if n > 1
            Q_template = spdiagm(
                0 => fill(2.0, n),
                -1 => fill(-1.0, n-1),
                1 => fill(-1.0, n-1)
            )
            Q_template[1, 1] = 1.0
            Q_template[n, n] = 1.0
        elseif n == 1
            Q_template[1, 1] = 1.0
        end
        rank_deficiency = 1
        V0 = reshape(fill(1.0 / sqrt(n), n), n, 1)
        P_perp = sparse(I, n, n) - sparse(V0 * V0')
    elseif model_type == :rw2
        if n > 1
            Q_template = spdiagm(
                0 => fill(6.0, n),
                -1 => fill(-4.0, n-1),
                1 => fill(-4.0, n-1),
                -2 => fill(1.0, n-2),
                2 => fill(1.0, n-2)
            )
            Q_template[1, 1] = 1.0
            Q_template[2, 2] = 5.0
            Q_template[1, 2] = -2.0
            Q_template[2, 1] = -2.0
            Q_template[n-1, n-1] = 5.0
            Q_template[n, n] = 1.0
            Q_template[n-1, n] = -2.0
            Q_template[n, n-1] = -2.0
        elseif n == 1
            Q_template[1, 1] = 1.0
        end
        rank_deficiency = 2
        # Orthonormal constant and linear null vectors for RW2
        v1 = fill(1.0 / sqrt(n), n)
        t_raw = collect(1:n) .- (n + 1) / 2.0
        v2 = t_raw ./ norm(t_raw)
        V0 = hcat(v1, v2)
        P_perp = sparse(I, n, n) - sparse(V0 * V0')
    elseif model_type == :cyclic
        if n > 0
            Q_template = spdiagm(
                0 => fill(2.0, n),
                1 => fill(-1.0, n-1),
                -1 => fill(-1.0, n-1)
            )
            Q_template[1, n] = -1.0
            Q_template[n, 1] = -1.0
        end
        rank_deficiency = 1
        V0 = reshape(fill(1.0 / sqrt(n), n), n, 1)
        P_perp = sparse(I, n, n) - sparse(V0 * V0')
    elseif model_type == :ar1
        if n > 1
            Q_template = spdiagm(
                0 => zeros(n),
                -1 => fill(-1.0, n-1),
                1 => fill(-1.0, n-1)
            )
        end
        rank_deficiency = 0
        V0 = zeros(Float64, n, 0)
        P_perp = sparse(I, n, n)
    elseif model_type == :iid
        Q_template = sparse(I, n, n)
        rank_deficiency = 0
        V0 = zeros(Float64, n, 0)
        P_perp = sparse(I, n, n)
    else
        @warn "Unknown model type '$model_type'. Returning identity matrix as template."
        Q_template = sparse(I, n, n)
        rank_deficiency = 0
        V0 = zeros(Float64, n, 0)
        P_perp = sparse(I, n, n)
    end
    eig_decomp = eigen(Symmetric(Matrix(Q_template)))
    U = eig_decomp.vectors
    L = eig_decomp.values
    if rank_deficiency > 0
        scaling_factor = _compute_scaling_factor(L, rank_deficiency)
        Q_template = Q_template ./ scaling_factor
        L = L ./ scaling_factor
    else
        scaling_factor = 1.0
    end
    gamma_val = Float64(deflation_gamma)
    Q_deflated = isempty(V0) ? Q_template : Q_template + sparse(gamma_val .* (V0 * V0'))
    return (
        matrix = Q_template,
        scaling_factor = scaling_factor,
        U = U,
        L = L,
        components = [collect(1:n)],
        n_components = 1,
        subgraph_scaling_factors = [scaling_factor],
        rank_deficiency = rank_deficiency,
        null_vectors = V0,
        projection_matrix = P_perp,
        matrix_deflated = Q_deflated
    )
end


"""
    _compute_scaling_factor(evals::Vector{Float64}, rank_deficiency::Int)
Computes a robust scaling factor for a precision matrix from its eigenvalues.
# Version
v1.0.0
# Mathematical Formulation
The scaling factor `c` is defined as the geometric mean of the `n - rank_deficiency`
non-zero eigenvalues. This ensures that the determinant of the scaled precision
matrix is 1.
\$c = \\exp\\left( \\frac{1}{n - \\text{rank\\_deficiency}}
  \\sum_{i=\\text{rank\\_deficiency}+1}^{n} \\log(\\lambda_i) \\right)\$
where \$\\lambda_i\$ are the non-zero eigenvalues.
# Arguments
- `evals::Vector{Float64}`: A vector of eigenvalues from a symmetric matrix.
- `rank_deficiency::Int`: The known rank deficiency of the matrix (number of zero eigenvalues).
# Returns
- `Float64`: The computed scaling factor.
"""
function _compute_scaling_factor(evals::Vector{Float64}, rank_deficiency::Int)
    # Sort eigenvalues in ascending order to easily discard the smallest ones,
    # which correspond to the null space.
    sorted_evals = sort(evals)
    n = length(sorted_evals)
    if n <= rank_deficiency
        # If the number of eigenvalues is less than or equal to the rank deficiency,
        # it implies all eigenvalues are effectively zero or the matrix is too small.
        return 1.0
    end
    # Select candidate eigenvalues after discarding expected null space rank deficiency
    candidate_evals = sorted_evals[(rank_deficiency + 1):end]
    # Check for additional zero eigenvalues (e.g. disconnected graph components)
    max_λ = isempty(sorted_evals) ? 1.0 : maximum(abs, sorted_evals)
    tol = max(1e-10, 1e-8 * max_λ)
    num_near_zero = count(λ -> λ <= tol, candidate_evals)
    if num_near_zero > 0
        @warn "Precision matrix has $(num_near_zero) additional near-zero eigenvalue(s) " *
              "(<= $tol), indicating disconnected spatial graph components. " *
              "Filtering from scaling factor calculation."
    end
    positive_evals = filter(λ -> λ > tol, candidate_evals)
    if isempty(positive_evals)
        @warn "No positive eigenvalues found above tolerance $tol. Defaulting to 1.0."
        return 1.0
    end
    # The scaling factor is the geometric mean of the positive eigenvalues.
    # This is a standard method for ensuring the determinant of the scaled
    # precision matrix is 1.
    return exp(mean(log.(positive_evals)))
end
"""
    evaluate_cross_kernel_matrix(coords1::AbstractMatrix, coords2::AbstractMatrix,
      param_val::Real, length_scale::Union{Real, AbstractVector}, kernel_type::Symbol)
# Version
v1.0.0
# Arguments
- `coords1::AbstractMatrix`: An `N1 x D` matrix of data points.
- `coords2::AbstractMatrix`: An `N2 x D` matrix of data points.
- `param_val::Real`: The signal variance (\$\\sigma^2\$) of the kernel.
- `length_scale::Union{Real, AbstractVector}`: The length_scale(s) (\$\\ell\$) of the kernel.
  A `Real` value assumes an isotropic kernel, while a `Vector` of length `D` enables
  ARD with a separate length_scale for each dimension.
- `kernel_type::Symbol`: The type of kernel to evaluate.
# Supported Kernels and Mathematical Formulation
- `:gaussian`, `:se`, `:rbf`: Squared Exponential kernel.
  \$k(x, x') = \\sigma^2 \\exp\\left(-\\frac{\\|x - x'\\|^2}{2\\ell^2}\\right)\$
- `:exponential`, `:matern12`: Exponential kernel (Matérn with \$\\nu=1/2\$).
  \$k(x, x') = \\sigma^2 \\exp\\left(-\\frac{\\|x - x'\\|}{\\ell}\\right)\$
- `:matern32`: Matérn kernel with \$\\nu=3/2\$.
  \$k(x, x') = \\sigma^2 \\left(1 + \\frac{\\sqrt{3}\\|x - x'\\|}{\\ell}\\right)
    \\exp\\left(-\\frac{\\sqrt{3}\\|x - x'\\|}{\\ell}\\right)\$
- `:matern52`: Matérn kernel with \$\\nu=5/2\$.
  \$k(x, x') = \\sigma^2 \\left(1 + \\frac{\\sqrt{5}\\|x - x'\\|}{\\ell} + \\frac{5\\|x -
    x'\\|^2}{3\\ell^2}\\right) \\exp\\left(-\\frac{\\sqrt{5}\\|x - x'\\|}{\\ell}\\right)\$
- `:spherical`: Spherical kernel.
  \$k(x, x') = \\sigma^2 \\left(1 - \\frac{3}{2}\\frac{\\|x - x'\\|}{\\ell} +
    \\frac{1}{2}\\left(\\frac{\\|x - x'\\|}{\\ell}\\right)^3\\right)\$ for \$\\|x - x'\\| <
    \\ell\$, else \$0\$.
- `:cosine`: Cosine kernel.
  \$k(x, x') = \\sigma^2 \\cos\\left(\\frac{2\\pi\\|x - x'\\|}{\\ell}\\right)\$
- `:linear`: Linear kernel.
  \$k(x, x') = \\sigma^2 x^T x'\$
- `:constant`: Constant kernel.
  \$k(x, x') = \\sigma^2\$
"""
function evaluate_cross_kernel_matrix(coords1::AbstractMatrix, coords2::AbstractMatrix,
    param_val::Real, length_scale::Union{Real, AbstractVector}, kernel_type::Symbol)
    T = promote_type(eltype(coords1), eltype(coords2), typeof(param_val), eltype(length_scale))
    coords1_T = convert(AbstractMatrix{T}, coords1)
    coords2_T = convert(AbstractMatrix{T}, coords2)
    length_scale_T = convert(typeof(length_scale) <: Real ? T : AbstractVector{T}, length_scale)
    if kernel_type == :linear
        return param_val^2 .* (coords1_T * coords2_T')
    end
    function _sqeuclidean_broadcast_cross(X1::AbstractMatrix, X2::AbstractMatrix)
        sum(X1.^2, dims=2) .- 2 * (X1 * X2') .+ sum(X2.^2, dims=2)'
    end
    local dist_sq
    if length_scale isa AbstractVector # ARD case
        if size(coords1_T, 2) != length(length_scale_T) || size(coords2_T, 2) != length(length_scale_T)
            error("Dimension mismatch for ARD kernel: Number of coordinate dimensions ($(size(coords1_T, 2))) does not match number of lengthscales ($(length(length_scale_T))).")
        end
        dist_sq = _sqeuclidean_broadcast_cross(coords1_T ./ length_scale_T', coords2_T ./ length_scale_T')
    else # Isotropic case
        dist_sq = _sqeuclidean_broadcast_cross(coords1_T, coords2_T) ./ length_scale_T^2
    end
    dist_sq .= max.(zero(T), dist_sq)
    if kernel_type == :gaussian || kernel_type == :se || kernel_type == :rbf
        return param_val^2 .* exp.(-one(T)/2 .* dist_sq)
    elseif kernel_type == :exponential || kernel_type == :matern12
        d = sqrt.(dist_sq)
        return param_val^2 .* exp.(-d)
    elseif kernel_type == :matern32
        d = sqrt.(dist_sq)
        val = sqrt(convert(T, 3.0)) .* d
        return param_val^2 .* (one(T) .+ val) .* exp.(-val)
    elseif kernel_type == :matern52
        d = sqrt.(dist_sq)
        val = sqrt(convert(T, 5.0)) .* d
        return param_val^2 .* (one(T) .+ val .+ (val.^2 ./ convert(T, 3.0))) .* exp.(-val)
    elseif kernel_type == :spherical
        d = sqrt.(dist_sq)
        K = zeros(T, size(d))
        mask = d .< one(T)
        K[mask] = param_val^2 .* (one(T) .- 1.5 .* d[mask] .+ 0.5 .* d[mask].^3)
        return K
    elseif kernel_type == :cosine
        if length_scale isa AbstractVector
            @warn "Cosine kernel with ARD length_scale is not standard. Using the first length_scale for an isotropic kernel."
            length_scale_T = length_scale_T[1]
        end
        d_euclidean = sqrt.(_sqeuclidean_broadcast_cross(coords1_T, coords2_T))
        return param_val^2 .* cos.(2.0 * pi .* d_euclidean ./ length_scale_T)
    elseif kernel_type == :constant
        return fill(convert(T, param_val^2), size(dist_sq))
    else
        @warn "Kernel '$(kernel_type)' not explicitly handled in evaluate_cross_kernel_matrix. Defaulting to Squared Exponential."
        return param_val^2 .* exp.(-one(T)/2 .* dist_sq)
    end
end
"""
    observation_volatility(M::NamedTuple)
Generates Turing code fragments for the observation error variance, handling both
constant variance and a spatiotemporal stochastic volatility (SV) model.
# Version
v1.0.0
# Mathematical Formulation
- **Constant Variance**: \$\\sigma_y\$ is a single parameter.
- **Stochastic Volatility**: The log-variance is modeled as a GP approximated by RFFs:
  `log_var(s, t) = Z(s, t) * β`
  where `Z` is the RFF basis matrix and `β` are coefficients. The standard deviation
  is then `σ_y(s, t) = exp(log_var(s, t) / 2)`.
# Arguments
- `M::NamedTuple`: The model configuration object.
# Returns
- A `NamedTuple` with code strings for `:priors` and `:calculation`.
"""
function observation_volatility(M::NamedTuple)
    is_multivariate = get(M, :model_arch, "univariate") == "multivariate"
    if get(M, :volatility, false)
        required_keys = [:M_rff_sigma, :W_sigma_fixed, :b_sigma_fixed, :coords_st]
        if !all(k -> haskey(M, k), required_keys)
            error("Stochastic volatility is enabled, but required keys are missing from the model configuration: $required_keys.")
        end
        priors_str = """
        sigma_log_var ~ Exponential(1.0)
        beta_vol ~ MvNormal(fill!(Array{T}(undef, M.M_rff_sigma), 0),
          sigma_log_var^2 * I)
        """
        calc_str = """
        # Stochastic Volatility Calculation
        vol_proj = (M.coords_st * M.W_sigma_fixed) .+ M.b_sigma_fixed'
        log_var_latent = sqrt(2.0 / M.M_rff_sigma) .* cos.(vol_proj) * beta_vol
        y_sigma_sv = exp.(log_var_latent ./ 2.0)
        """
        final_calc_str = is_multivariate ? "y_sigma = y_sigma_sv .* y_sigma_const'" : "y_sigma = y_sigma_const .* y_sigma_sv"
        return (priors=priors_str, calculation="$(calc_str)\n    $(final_calc_str)")
    else
        priors_str = ""
        calc_str = if is_multivariate
            "y_sigma = y_sigma_const'"
        else
            "y_sigma = fill(y_sigma_const, N)"
        end
        return (priors=priors_str, calculation=calc_str)
    end
end
"""
    generate_inducing_points(coords::AbstractMatrix, n_inducing::Int;
      method::String="kmeans", seed::Int=42)
Selects a representative subset of coordinates to serve as inducing points for sparse
Gaussian Process (GP) models.
# Version
v1.0.0
# Arguments
- `coords::AbstractMatrix`: An `N x D` matrix of data point coordinates.
- `n_inducing::Int`: The number of inducing points to select.
- `method::String`: The selection method to use.
- `seed::Int`: A random seed for reproducibility of `:random` and `:kmeans`.
# Returns
- An `M x D` matrix of inducing point coordinates, where `M <= n_inducing`.
"""
function generate_inducing_points(
    coords::AbstractMatrix, 
    n_inducing::Int; 
    method::String="kmeans", 
    seed::Int=42
)
    n_obs, n_dims = size(coords)
    if n_inducing >= n_obs
        return coords
    end
    Random.seed!(seed)
    if method == "random"
        # Simple stochastic selection without replacement.
        selected_idx = StatsBase.sample(1:n_obs, n_inducing, replace=false)
        return coords[selected_idx, :]
    elseif method == "kmeans"
        # Centroid-based selection via Clustering.jl.
        # kmeans expects observations in columns: [dims x obs].
        kmeans_res = Clustering.kmeans(coords', n_inducing; maxiter=200, display=:none)
        return kmeans_res.centers'
    elseif method == "quantile" || method == "regular"
        # Systematic mapping methods requiring KDTree for efficiency.
        target_pts = zeros(Float64, n_inducing, n_dims)
        if method == "quantile"
            # Density-aware target generation using marginal quantiles.
            probs = range(0.0, stop=1.0, length=n_inducing)
            for d in 1:n_dims
                target_pts[:, d] = Statistics.quantile(coords[:, d], probs)
            end
        else # method == "regular"
            # Grid-like target generation across marginal ranges.
            for d in 1:n_dims
                v_min, v_max = extrema(coords[:, d])
                target_pts[:, d] = range(v_min, stop=v_max, length=n_inducing)
            end
        end
        # Efficient Nearest Neighbor Search using KDTree.
        tree = KDTree(coords')
        # Find the single nearest observation for each target coordinate.
        nn_indices_vec, _ = knn(tree, target_pts', 1, true)
        # Extract the scalar index from each neighbor search result and deduplicate.
        unique_nn_indices = unique([idx_list[1] for idx_list in nn_indices_vec])
        return coords[unique_nn_indices, :]
    else
        @warn "Inducing point method '$method' not recognized. Falling back to random selection."
        selected_idx = StatsBase.sample(1:n_obs, n_inducing, replace=false)
        return coords[selected_idx, :]
    end
end
"""
    create_pc_prior(param_name::Symbol, constraint::Tuple)
Creates a Penalized Complexity (PC) prior distribution from a user-specified quantile constraint.
# Version
v1.0.0
# Mathematical Formulation
The function maps a quantile constraint `(U, α)` to the hyperparameter `λ` of an
`Exponential(λ)` prior. The specific formula depends on the parameter type:
- **For `sigma` or `range` (scale parameters)**:
  - Constraint: \$P(\\text{param} > U) = \\alpha\$
  - Derivation: \$e^{-\\lambda U} = \\alpha \\implies \\lambda = -\\log(\\alpha) / U\$
- **For `rho` (correlation parameter on [0, 1])**:
  - The prior is placed on a transformed parameter \$\\theta = -\\log(1-\\rho) \\sim
    \\text{Exponential}(\\lambda)\$.
  - Constraint: \$P(\\rho > U) = \\alpha\$
  - Derivation: \$P(\\theta > -\\log(1-U)) = e^{-\\lambda(-\\log(1-U))} = (1-U)^{\\lambda} =
    \\alpha \\implies \\lambda = \\log(\\alpha) / \\log(1-U)\$
- **For `length_scale`**:
  - The prior is placed on the inverse \$\\theta = 1/\\ell \\sim \\text{Exponential}(\\lambda)\$.
  - Constraint: \$P(\\ell < U) = \\alpha\$
  - Derivation: \$P(\\theta > 1/U) = e^{-\\lambda/U} = \\alpha \\implies \\lambda = -U
    \\log(\\alpha)\$
- **For other parameters**:
  - A symmetric `Normal(0, σ)` prior is assumed, where the standard deviation `σ` is
    derived from a two-sided constraint \$P(|\\text{param}| > U) = \\alpha\$.
# Arguments
- `param_name::Symbol`: The base name of the parameter (e.g., `:sigma`, `:rho`).
- `constraint::Tuple`: A tuple `(U, α)` or `(U, α, direction)` defining the quantile constraint.
# Returns
- A `Distribution` object representing the calculated prior.
"""
function create_pc_prior(param_name::Symbol, constraint::Tuple)
    direction = :upper
    if length(constraint) == 2
        U, α = constraint
    elseif length(constraint) == 3
        U, α, direction = constraint
    else
        error("PC prior constraint must be a tuple of (U, α) or (U, α, direction).")
    end
    if param_name == :sigma || endswith(string(param_name), "_sigma")
        direction != :upper && error("PC prior for sigma only supports upper tail constraints.")
        λ = -log(α) / U
        return Exponential(λ)
    elseif param_name == :rho || endswith(string(param_name), "_rho")
        direction != :upper && error("PC prior for 'rho' only supports upper tail constraints.")
        λ = log(α) / log(1.0 - U)
        return Exponential(λ)
    elseif param_name == :length_scale || endswith(string(param_name), "_length_scale")
        direction != :lower&&
            error("PC prior for 'length_scale' only supports lower tail constraints.")
        λ = -U * log(α)
        return Exponential(λ)
    elseif param_name == :range || endswith(string(param_name), "_range")
        direction != :upper && error("PC prior for 'range' only supports upper tail constraints.")
        λ = -log(α) / U
        return Exponential(λ)
    else
        # Fallback for other parameters, assuming a symmetric Normal prior.
        # P(|param| > U) = α  => P(Z > U/σ) = α/2
        sigma = -U / quantile(Normal(0, 1), α / 2)
        return Normal(0, sigma)
    end
end
"""
    create_fixed_design(formula_rhs::AbstractString, data::DataFrame,
      calling_module::Module; contrasts=Dict{Symbol, Any}())
Creates a fixed-effects design matrix (`X`) from a formula string using `StatsModels.jl`.
# Version
v1.0.0
# Arguments
- `formula_rhs::AbstractString`: A string representing the right-hand side of the model
  formula (e.g., "0 + x + y*z").
- `data::DataFrame`: The input data frame containing the variables.
- `calling_module::Module`: The module in which the formula should be evaluated (used by
  `StatsModels` to resolve variables).
- `contrasts`: An optional dictionary specifying contrast coding for categorical variables.
# Returns
- A tuple `(NamedArray, Union{StatsModels.FormulaTerm, Nothing})`.
"""
function create_fixed_design(
    formula_rhs::AbstractString, 
    data::DataFrame, 
    calling_module::Module; 
    contrasts=Dict{Symbol, Any}()
)
    df_internal = copy(data)
    final_rhs_string = strip(formula_rhs)
    if isempty(final_rhs_string)
        return NamedArray(zeros(size(df_internal, 1), 0), (1:size(df_internal, 1),
            Symbol[])), nothing
    end
    if final_rhs_string == "1"
        return NamedArray(ones(size(df_internal, 1), 1), (1:size(df_internal, 1),
            [:Intercept])), nothing
    end
    try
        placeholder_name = :__y_placeholder
        if !hasproperty(df_internal, placeholder_name)
            df_internal[!, placeholder_name] = zeros(size(df_internal, 1))
        end
        # Explicitly qualify the @formula macro to prevent LoadError.
        formula_expression = Meta.parse("StatsModels.@formula($placeholder_name ~ $final_rhs_string)")
        # Evaluate in the current module's scope to ensure StatsModels is found.
        # The variables in the formula string are resolved later by apply_schema.
        dynamic_formula = Core.eval(@__MODULE__, formula_expression)
        data_schema = StatsModels.schema(dynamic_formula, df_internal, contrasts)
        applied_formula = StatsModels.apply_schema(dynamic_formula, data_schema,
            StatsModels.RegressionModel)
        _, model_matrix_numeric = StatsModels.modelcols(applied_formula, df_internal)
        coefficient_labels = StatsModels.coefnames(applied_formula.rhs)
        label_vector = coefficient_labels isa AbstractString ? [Symbol(coefficient_labels)] : Symbol.(coefficient_labels)
        return NamedArray(model_matrix_numeric, (1:size(model_matrix_numeric, 1),
            label_vector)), applied_formula
    catch design_error
        @warn "TethysBayes Registry: create_fixed_design expansion failed for: '$final_rhs_string'. Error of type '$(typeof(design_error))' occurred. Check formula syntax and variable names."
        return NamedArray(zeros(size(df_internal, 1), 0), (1:size(df_internal, 1),
            Symbol[])), nothing
    end
end
"""
    show_model(m::DynamicPPL.Model)
Displays a comprehensive and well-formatted summary of the `TethysBayes` model configuration,
including likelihoods, priors, and component-specific parameters.
# Version
v1.0.0
# Arguments
- `m`: The Turing model instance generated by `@tbm`.
# Returns
- `nothing`. The function prints the summary to the console.
"""
function show_model(m::DynamicPPL.Model)
    println("\n--- Model Summary ---\n")
    config = m.args.M
    println("Model Name:           ", get(config, :model_name, nameof(m.f)))
    println("Model Architecture:   ", get(config, :model_arch, "N/A"))
    # Likelihood Configuration
    println("\n[ Likelihood ]")
    for (i, spec) in enumerate(config.likelihood_specs)
        outcome = config.outcomes[i]
        println("  Outcome: $outcome")
        user_params = spec
        lik_params_to_show = [
            :family, :log_offsets, :weights, :trials, :zero_inflated, 
            :volatility, :censor_lower, :censor_upper, :hurdle, :latent_dist
        ]
        for p_name in lik_params_to_show
            if haskey(user_params, p_name)
                 _print_param(p_name, user_params[p_name], :user; indent=4)
            end
        end
    end
    # Intercept Configuration
    println("\n[ Intercept ]")
    if get(config, :add_intercept, false)
        is_user_provided = haskey(config,
            :intercept_prior) && config.intercept_prior != Normal(0, 5)
        _print_param(:prior, get(config, :intercept_prior, Normal(0, 5)),
            is_user_provided ? :user : :default; indent=2)
    else
        println("  - Intercept removed from model.")
    end
    # Fixed Effects Configuration
    if get(config, :Xfixed_N, 0) > 0
        println("\n[ Fixed Effects ]")
        println("  Formula: ~ $(config.Xfixed_applied_formula.rhs)")
        if haskey(config, :contrasts) && !isempty(config.contrasts)
            println("  Contrasts:")
            for (var, cont) in config.contrasts
                println("    - $var: $(typeof(cont))")
            end
        end
        println("  Priors per Coefficient:")
        for (i, name) in enumerate(config.Xfixed_names)
            prior_obj = config.Xfixed_priors_vec[i]
            is_default = prior_obj == Normal(0, 5)
            _print_param(name, prior_obj, is_default ? :default : :user; indent=4)
        end
    end
    # Model Components
    if haskey(config, :components) && !isempty(config.components)
        println("\n[ Model Components ]")
        for spec in config.components
            println("  --- Component: $(spec.key) ---")
            component_obj = spec.component_obj
            println("    - Type:      $(typeof(component_obj))")
            println("    - Structure: $(spec.structure)")
            println("    - Variable:  $(spec.var)")
            println("    - Parameters:")
            user_provided_params_raw = spec.params 
            all_param_names = Set(fieldnames(typeof(component_obj)))
            union!(all_param_names, keys(user_provided_params_raw))
            for param_name in sort(collect(all_param_names))
                if param_name in [:positional_args, :structure, :in_dims]
                    continue
                end
                local final_val, status
                if param_name in fieldnames(typeof(component_obj))
                    final_val = getfield(component_obj, param_name)
                    status = haskey(user_provided_params_raw, param_name) ? :user : :default
                else
                    final_val = get(user_provided_params_raw, param_name, "N/A")
                    status = haskey(user_provided_params_raw, param_name) ? :user : :default
                end
                if final_val isa Vector{<:UnivariateDistribution}
                    println("      - $(rpad(param_name, 20)): [")
                    for (idx, p_dist) in enumerate(final_val)
                        println("        $idx: $p_dist")
                    end
                    println("      ] $(status == :user ? "(User-provided)" : "(Default)")")
                else
                    _print_param(param_name, final_val, status; indent=6)
                end
            end
        end
    else
        println("\n[ Model Components ]")
        println("  None")
    end
    # Generated Code
    if haskey(config, :generated_model_code)
        println("\n--- Generated Model Source ---\n")
        println(config.generated_model_code)
        println("\n--- End Generated Model Source ---")
    else
        println("\n--- Reconstructed Model Source (Pseudo-code) ---\n")
        println(model_pseudocode(m))
        println("\n--- End Reconstructed Model Source ---")
    end
    println("\n--- End Model Summary ---")
    return nothing
end
function model_pseudocode(m::DynamicPPL.Model)
    config = m.args.M
    model_name = get(config, :model_name, nameof(m.f))
    lines = String[]
    push!(lines, "@model function $(model_name)(M; T::Type=Float64)")
    push!(lines, "    # --- Priors & Hyperparameters ---")
    # Likelihood-specific priors
    family = string(get(config.likelihood_specs[1], :family, "gaussian"))
    if family == "negbin"
        push!(lines, "    r_nb ~ $(_distribution_to_string(Exponential(1.0)))")
    end
    if get(config, :use_zi, false)
        push!(lines, "    lik_phi_zi ~ $(_distribution_to_string(Beta(1, 1)))")
    end
    if get(config, :user_provided_hurdle, false)
        push!(lines, "    lik_phi_hurdle ~ $(_distribution_to_string(Beta(1,1)))")
    end
    if family in ["gaussian", "lognormal", "student_t", "laplace", "half_normal",
        "half_student_t"] && !get(config, :volatility, false)
        push!(lines, "    y_sigma ~ $(_distribution_to_string(Exponential(1.0)))")
    end
    if get(config, :volatility, false)
        push!(lines, "    sigma_log_var ~ $(_distribution_to_string(Exponential(1.0)))")
        # Simplified representation of beta_vol for pseudo-code
        push!(lines, "    beta_vol ~ MvNormal(zeros(T, M.M_rff_sigma), sigma_log_var^2 * I)")
    end
    if get(config, :outcomes_N, 1) > 1
        push!(lines, "    correlation_cholesky ~ $(_distribution_to_string(LKJCholesky(get(config, :outcomes_N, 1), 1.0)))")
    end
    if family == "student_t"
        push!(lines, "    lik_nu_student_t ~ $(_distribution_to_string(Exponential(1.0)))")
    end
    if family in ["gamma", "beta", "inverse_gaussian", "pareto", "half_student_t"]
        push!(lines, "    lik_extra_params ~ $(_distribution_to_string(Exponential(1.0)))")
    end
    # Ordinal-specific priors
    ordinal_spec_idx = findfirst(s -> string(get(s, :family, "")) == "ordinal",
        config.likelihood_specs)
    if !isnothing(ordinal_spec_idx)
        spec = config.likelihood_specs[ordinal_spec_idx]
        K_ordinal = get(spec, :K, 0)
        if K_ordinal > 2
            push!(lines, "    ordinal_alpha_unscaled_1 ~ $(_distribution_to_string(Normal(0, 5)))")
            push!(lines, "    ordinal_alpha_diffs ~ $(_distribution_to_string(Fill(Exponential(1.0), K_ordinal - 2)))")
        elseif K_ordinal == 2
            push!(lines, "    ordinal_alpha_unscaled_1 ~ $(_distribution_to_string(Normal(0, 5)))")
        end
        if get(spec, :latent_dist, :logistic) == :student_t
            push!(lines, "    ordinal_df ~ $(_distribution_to_string(Exponential(1.0)))")
        end
    end
    # Intercept prior
    if get(config, :add_intercept, false)
        intercept_prior_obj = get(config, :intercept_prior, Normal(0, 5))
        push!(lines, "    intercept ~ $(_distribution_to_string(intercept_prior_obj))")
    end
    # Fixed effects priors
    if get(config, :Xfixed_N, 0) > 0
        push!(lines, "    # Priors for fixed effects coefficients")
        is_multivariate = config.model_arch == "multivariate"
        n_fixed = config.Xfixed_N
        outcomes_N = config.outcomes_N
        # Proportional fixed effects
        prop_indices = collect(1:n_fixed)
        npo_indices = Int[]
        if !isnothing(ordinal_spec_idx) && haskey(config,
            :non_proportional_effects) && !isempty(config.non_proportional_effects)
            npo_indices = findall(x -> x in config.non_proportional_effects, config.Xfixed_names)
            prop_indices = setdiff(prop_indices, npo_indices)
        end
        if !isempty(prop_indices)
            priors_prop = get(config, :Xfixed_priors_vec, [Normal(0,
                5) for _ in 1:n_fixed])[prop_indices]
            if is_multivariate
                # Simplified for pseudo-code: assume all outcomes have the same prior structure
                prior_str = _distribution_to_string(priors_prop[1]) # Take first as representative
                push!(lines,
                    "    beta_flat ~ filldist($(prior_str), $(length(prop_indices) * outcomes_N))")
            else
                prior_str_list = [_distribution_to_string(p) for p in priors_prop]
                push!(lines, "    beta ~ Product([$(join(prior_str_list, ", "))])")
            end
        end
        # Non-proportional fixed effects (for ordinal)
        if !isempty(npo_indices) && K_ordinal > 1
            priors_npo = get(config, :Xfixed_priors_vec, [Normal(0,
                5) for _ in 1:n_fixed])[npo_indices]
            prior_str_list = [_distribution_to_string(p) for p in priors_npo]
            push!(lines, "    beta_npo ~ Product([$(join(prior_str_list, ", "))])")
        end
    end
    # Component-specific priors
    if haskey(config, :components) && !isempty(config.components)
        for spec in config.components
            m_obj = spec.component_obj
            m_type_str = string(typeof(m_obj).name.name)
            key = spec.key
            push!(lines, "\n    # Priors for component: $(key) ($(m_type_str))")
            # Every hyperprior a component can carry. Derived from the vocabulary rather
            # than hand-listed, so this cannot drift from the names components use.
            all_possible_hyperpriors = HYPERPARAMETER_STEMS
            for field_sym in all_possible_hyperpriors
                if hasproperty(m_obj, field_sym)
                    prior_dist = getfield(m_obj, field_sym)
                    if prior_dist isa Distribution || (prior_dist isa Vector&&
                        !isempty(prior_dist) && all(d -> d isa Distribution, prior_dist))
                        p_names = generate_full_variable_names(spec, config.model_arch,
                            1) # Use 1 for outcome_idx for pseudo-code simplicity
                        param_name_sym = get(p_names, field_sym,
                            Symbol("$(field_sym)_$(key)")) # Fallback if not in p_names
                        if prior_dist isa Vector
                            dist_str = "Product([$(join([_distribution_to_string(d) for d in prior_dist], ", "))])"
                            push!(lines, "    $(param_name_sym) ~ $(dist_str)")
                        else
                            push!(lines,
                                "    $(param_name_sym) ~ $(_distribution_to_string(prior_dist))")
                        end
                    end
                end
            end
            # Add innovations/innovations prior for components that have them
            p_names = generate_full_variable_names(spec, config.model_arch, 1)
            if hasproperty(spec.hyper, :n_latent) && spec.hyper.n_latent > 0
                push!(lines, "    $(p_names.innovations) ~ MvNormal(zeros(T, $(spec.hyper.n_latent)), I)")
            end
        end
    end
    push!(lines, "\n    # --- Linear Predictor Assembly ---")
    eta_parts = String[]
    is_multivariate = config.model_arch == "multivariate"
    eta_var_name = is_multivariate ? "eta_latent" : "eta"
    # Initialize eta with intercept and offsets
    if get(config, :add_intercept, false)
        push!(eta_parts, "intercept")
    end
    if haskey(config, :log_offsets) && !all(iszero, config.log_offsets)
        push!(eta_parts, "M.log_offsets")
    end
    eta_init_str = isempty(eta_parts) ? "zeros(T, M.y_N, $(config.outcomes_N))" : join(eta_parts,
        " .+ ")
    push!(lines, "    $(eta_var_name) = $(eta_init_str)")
    # Add fixed effects
    if get(config, :Xfixed_N, 0) > 0
        if is_multivariate
            push!(lines,
                "    $(eta_var_name) .+= M.Xfixed * reshape(beta_flat, M.Xfixed_N, M.outcomes_N)")
        else
            push!(lines, "    $(eta_var_name) .+= M.Xfixed * beta")
        end
    end
    # Add component effects
    if haskey(config, :components) && !isempty(config.components)
        for spec in config.components
            p_names = generate_full_variable_names(spec, config.model_arch,
                1) # Use 1 for pseudo-code
            if hasproperty(p_names, :latent_field)
                # This is a simplification; actual update logic is more complex and depends
                #   on structure.
                # For pseudo-code, we show a generic addition.
                if is_multivariate
                    push!(lines, "    $(eta_var_name) .+= $(p_names.latent_field)[M.s_idx, :]") # Example for spatial/temporal
                else
                    push!(lines, "    $(eta_var_name) .+= $(p_names.latent_field)[M.s_idx]")
                end
            end
        end
    end
    # Add spacetime interaction
    model_st = get(config, :model_st, "none")
    if model_st != "none"
        if is_multivariate
            push!(lines, "    $(eta_var_name) .+= spacetime_interaction[M.s_idx, M.t_idx, :]")
        else
            push!(lines, "    $(eta_var_name) .+= spacetime_interaction[M.s_idx, M.t_idx]")
        end
    end
    # Apply multivariate correlation if applicable
    if is_multivariate
        push!(lines, "    eta = $(eta_var_name) * correlation_cholesky.L'")
    end
    push!(lines, "\n    # --- Likelihood ---")
    # Construct a more complete TethysBayes_Likelihood call for pseudo-code
    lik_kwargs_parts = String[]
    if family == "negbin"
        push!(lik_kwargs_parts, "r_nb=r_nb")
    end
    if get(config, :user_provided_hurdle, false)
        push!(lik_kwargs_parts, "phi_hurdle=lik_phi_hurdle")
    end
    if get(config, :use_zi, false)
        push!(lik_kwargs_parts, "phi_zi=phi_zi")
    end
    if family in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"]
        push!(lik_kwargs_parts, "sigma_y=y_sigma")
    end
    if family == "student_t"
        push!(lik_kwargs_parts, "nu=lik_nu_student_t")
    end
    if family in ["gamma", "beta", "inverse_gaussian", "pareto", "half_student_t"]
        push!(lik_kwargs_parts, "extra_params=lik_extra_params")
    end
    if get(config, :user_provided_trials, false)
        push!(lik_kwargs_parts, "trial=M.trials")
    end
    if get(config, :user_provided_weights, false)
        push!(lik_kwargs_parts, "weight=M.weights")
    end
    if get(config, :user_provided_censor_lower, false)
        push!(lik_kwargs_parts, "censor_lower=M.censor_lower")
    end
    if get(config, :user_provided_censor_upper, false)
        push!(lik_kwargs_parts, "censor_upper=M.censor_upper")
    end
    lik_kwargs_str = isempty(lik_kwargs_parts) ? "" : "; $(join(lik_kwargs_parts, ", "))"
    if is_multivariate
        push!(lines, "    M.y_obs ~ TethysBayes_Likelihood(\"$(family)\", eta $(lik_kwargs_str))")
    else
        push!(lines, "    M.y_obs ~ TethysBayes_Likelihood(\"$(family)\", eta $(lik_kwargs_str))")
    end
    push!(lines, "end")
    return join(lines, "\n")
end

"""
    extract_model_body(text::String)::String

Extract the body of a `@model` function from its full source text (e.g., obtained
via CodeTracking.@code_string). Strips the `@model function ... end` wrapper and
returns only the inner code fragment, suitable for use with `custom(code_fragment=...)`.
"""
function extract_model_body(text::String)::String
    lines = split(text, '\n')
    depth = 0
    sig_end = 0
    in_sig = false
    for (i, line) in enumerate(lines)
        if occursin("@model function", line)
            in_sig = true
        end
        if !in_sig
            continue
        end
        for c in line
            if c == '('
                depth += 1
            elseif c == ')'
                depth -= 1
                if depth == 0
                    sig_end = i
                    break
                end
            end
        end
        sig_end > 0 && break
    end
    sig_end == 0 && error("Could not find end of function signature")
    body_lines = lines[sig_end+1:end]
    while !isempty(body_lines) && strip(last(body_lines)) == "end"
        pop!(body_lines)
    end
    while !isempty(body_lines) && strip(first(body_lines)) == ""
        popfirst!(body_lines)
    end
    return join(body_lines, "\n")
end

"""
    TethysBayes_bspline_basis(x::AbstractVector, n_basis::Int, degree::Int; ...)

Generates a B-spline basis matrix. This version is CPU-only.
"""
function TethysBayes_bspline_basis(x::AbstractVector, n_basis::Int, degree::Int;
    knot_method::Symbol=:quantile, custom_knots::Union{AbstractVector, Nothing}=nothing)
    p = degree
    if n_basis <= p
        error("Number of basis functions (nbins) must be greater than the spline degree. Got n_basis=$n_basis, degree=$p.")
    end
    n_interior_knots = n_basis - p
    knots = if !isnothing(custom_knots)
        custom_knots
    else
        if n_interior_knots > 0
            if knot_method == :quantile
                probs = range(0, 1, length=n_interior_knots + 2)[2:end-1]
                quantile(x, probs)
            else
                range(minimum(x), maximum(x), length=n_interior_knots + 2)[2:end-1]
            end
        else
            Float64[]
        end
    end
    boundary_knots = [minimum(x), maximum(x)]
    all_knots = sort(unique(vcat(boundary_knots, knots)))
    n_basis_possible = length(all_knots) + p - 1
    if n_basis > n_basis_possible
        @warn "Requested n_basis ($n_basis) is too high for the number of unique knots ($(length(all_knots))) and degree ($p). Reducing to $n_basis_possible."
        n_basis = n_basis_possible
    end
    t = vcat(fill(all_knots[1], p), all_knots, fill(all_knots[end], p))
    N = length(x)
    num_total_basis = length(t) - p - 1
    B = Matrix{Float64}(undef, N, num_total_basis); fill!(B, 0.0)
    for j in 1:num_total_basis
        B[:, j] = (t[j] .<= x .< t[j+1])
    end
    if !isempty(x) && t[end] == maximum(x)
        B[x .== t[end], num_total_basis] .= 1.0
    end
    for d in 1:p
        for j in 1:(num_total_basis - d)
            w1 = Vector{Float64}(undef, N); fill!(w1, 0.0)
            denom1 = t[j+d] - t[j]
            if denom1 > 1e-9
                w1 = (x .- t[j]) ./ denom1
            end
            w2 = Vector{Float64}(undef, N); fill!(w2, 0.0)
            denom2 = t[j+d+1] - t[j+1]
            if denom2 > 1e-9
                w2 = (t[j+d+1] .- x) ./ denom2
            end
            B[:, j] = w1 .* B[:, j] + w2 .* B[:, j+1]
        end
    end
    return (B[:, 1:n_basis], n_basis)
end
"""
    TethysBayes_tensor_product_basis(coords::AbstractMatrix, nbins_per_dim::Vector{Int},
      degrees_per_dim::Vector{Int}; ...)
Generates a tensor product B-spline basis matrix. This version is CPU-only.
"""
function TethysBayes_tensor_product_basis(coords::AbstractMatrix, nbins_per_dim::Vector{Int},
    degrees_per_dim::Vector{Int}; knot_method::Symbol=:quantile, kwargs...)
    n_dims = size(coords, 2)
    if length(nbins_per_dim) != n_dims || length(degrees_per_dim) != n_dims
        error("Number of dimensions in coords must match length of nbins_per_dim and degrees_per_dim.")
    end
    bspline_kwargs = Dict{Symbol, Any}()
    if haskey(kwargs, :custom_knots)
        bspline_kwargs[:custom_knots] = kwargs[:custom_knots]
    end
    basis_matrices_1D = Vector{AbstractMatrix{Float64}}(undef, n_dims)
    for i in 1:n_dims
        local_bspline_kwargs = copy(bspline_kwargs)
        if haskey(local_bspline_kwargs,
            :custom_knots) && local_bspline_kwargs[:custom_knots] isa Tuple
            local_bspline_kwargs[:custom_knots] = local_bspline_kwargs[:custom_knots][i]
        end
        basis_mat, _ = TethysBayes_bspline_basis(
            coords[:, i], 
            nbins_per_dim[i], 
            degrees_per_dim[i]; 
            knot_method=knot_method, 
            local_bspline_kwargs...
        )
        basis_matrices_1D[i] = basis_mat
    end
    if isempty(basis_matrices_1D)
        return Matrix{Float64}(undef, size(coords, 1), 0)
    end
    B_final = basis_matrices_1D[1]
    for i in 2:n_dims
        B_next = basis_matrices_1D[i]
        n_obs, n_cols_final = size(B_final)
        _, n_cols_next = size(B_next)
        B_final_reshaped = reshape(B_final, n_obs, n_cols_final, 1)
        B_next_reshaped = reshape(B_next, n_obs, 1, n_cols_next)
        tensor_prod = B_final_reshaped .* B_next_reshaped
        B_final = reshape(tensor_prod, n_obs, n_cols_final * n_cols_next)
    end
    return B_final
end
"""
    TethysBayes_wavelet_basis_1D(vals::AbstractVector, nbins::Int, family::Symbol, length_scale::Float64)
Generates a 1D wavelet basis matrix. This version is CPU-only.
"""
function TethysBayes_wavelet_basis_1D(vals::AbstractVector, nbins::Int, family::Symbol,
    length_scale::Float64)
    Interpolations = Base.require(Base.Main, :Interpolations)
    n_obs = length(vals)
    B = Matrix{Float64}(undef, n_obs, nbins); fill!(B, 0.0)
    v_min, v_max = minimum(vals), maximum(vals)
    v_range = v_max - v_min
    if v_range < 1e-9
        v_range = 1.0
    end
    local wt_type
    try
        wt_type = getfield(Wavelets.WT, family)
    catch e
        @error "Could not resolve wavelet family ':$family'. Error: $e. Defaulting to db4."
        wt_type = Wavelets.WT.db4
    end
    local wt_instance
    try
        wt_instance = Wavelets.wavelet(wt_type)
    catch e
        error("Failed to instantiate wavelet object from type '$wt_type'. Error: $e")
    end
    h_filter = wt_instance.qmf
    L = length(h_filter)
    g_filter = similar(h_filter)
    for i in 1:L
        g_filter[i] = (-1.0)^(i-1) * h_filter[L - (i-1)]
    end
    n_reconstruction_iterations = 8
    x_psi_grid, psi_vals = _reconstruct_wavelet_function_from_filters(h_filter, g_filter,
        n_reconstruction_iterations)
    itp = Interpolations.linear_interpolation(x_psi_grid, psi_vals,
        extrapolation_bc=Interpolations.Flat())
    n_scales = max(1, floor(Int, log2(nbins/4)))
    bins_per_scale = div(nbins, n_scales)
    current_bin = 1
    for j in 1:n_scales
        scale_factor = length_scale * (2.0^(j-1))
        n_translations = (j == n_scales) ? (nbins - current_bin + 1) : bins_per_scale
        if n_translations <= 0
            continue
        end
        probs = n_translations == 1 ? [0.5] : range(0, 1, length=n_translations)
        centers = quantile(vals, probs)
        for k in 1:n_translations
            if current_bin > nbins
                break
            end
            transformed_vals = (vals .- centers[k]) ./ (scale_factor * v_range)
            B[:, current_bin] = itp.(transformed_vals)
            current_bin += 1
        end
    end
    return B
end
"""
    _reconstruct_wavelet_function_from_filters(h::Vector{Float64}, g::Vector{Float64},
      n_iterations::Int)
Reconstructs the mother wavelet function from its quadrature mirror filters using the
  cascade algorithm.
# Version
v1.0.0
# Mathematical Formulation
The algorithm starts with the scaling function \$\\phi_0(t)\$ as a box function
(\$1\$ on `[0,1)`, \$0\$ otherwise) and iteratively refines it using the two-scale relation:
\$\\phi_{j+1}(t) = \\sqrt{2} \\sum_k h_k \\phi_j(2t - k)\$
In each iteration, the wavelet function \$\\psi(t)\$ is computed from the current scaling
function:
\$\\psi_{j+1}(t) = \\sqrt{2} \\sum_k g_k \\phi_j(2t - k)\$
After `n_iterations`, the function returns the final approximation of \$\\psi(t)\$.
# Arguments
- `h::Vector{Float64}`: The low-pass filter coefficients (scaling function filter).
- `g::Vector{Float64}`: The high-pass filter coefficients (wavelet function filter).
- `n_iterations::Int`: The number of refinement iterations to perform.
# Returns
- A tuple `(x_grid_final, psi_next_vals)` where `x_grid_final` is the coordinate
  grid and `psi_next_vals` are the corresponding values of the wavelet function.
"""
function _reconstruct_wavelet_function_from_filters(h::Vector{Float64}, g::Vector{Float64},
    n_iterations::Int)
    # Dynamically load Interpolations to ensure it's available in the execution scope.
    Interpolations = Base.require(Base.Main, :Interpolations)
    L = length(h) 
    x_min_support = 0.0
    x_max_support = L > 1 ? L - 1.0 : 1.0
    # Define a fine grid for the final reconstruction.
    num_points_final_grid = max(2, (2^n_iterations) * max(1, L - 1) + 1)
    x_grid_final = collect(range(x_min_support, stop=x_max_support, length=num_points_final_grid))
    # Initialize the scaling function phi_0 as a box function.
    phi_current_vals = zeros(length(x_grid_final))
    for i in eachindex(x_grid_final)
        if 0.0 <= x_grid_final[i] < 1.0
            phi_current_vals[i] = 1.0
        end
    end
    # Create an interpolant for the current scaling function.
    phi_itp = Interpolations.linear_interpolation(x_grid_final, phi_current_vals,
        extrapolation_bc=Interpolations.Flat())
    psi_next_vals = zeros(length(x_grid_final))
    # Iteratively refine the scaling and wavelet functions.
    for iter in 1:n_iterations
        phi_next_vals = zeros(length(x_grid_final))
        for idx in eachindex(x_grid_final)
            x_val = x_grid_final[idx]
            phi_sum = 0.0
            psi_sum = 0.0
            # Apply the two-scale relation using the filter coefficients.
            for k_filter in 0:(L-1)
                arg = 2.0 * x_val - k_filter
                phi_val_at_arg = phi_itp(arg)
                phi_sum += h[k_filter+1] * phi_val_at_arg
                psi_sum += g[k_filter+1] * phi_val_at_arg
            end
            phi_next_vals[idx] = sqrt(2.0) * phi_sum
            psi_next_vals[idx] = sqrt(2.0) * psi_sum
        end
        # Update the interpolant for the next iteration.
        phi_itp = Interpolations.linear_interpolation(x_grid_final, phi_next_vals,
            extrapolation_bc=Interpolations.Flat())
    end
    return x_grid_final, psi_next_vals
end
"""
    TethysBayes_tensor_product_wavelet_basis(coords::AbstractMatrix, nbins_per_dim::Vector{Int},
      family::Symbol, length_scale::Union{Real, AbstractVector})
Generates a multi-dimensional wavelet basis matrix via a tensor product of 1D bases.
# Version
v1.0.0
# Mathematical Formulation
Given 1D basis matrices \$B_1, B_2, \\dots, B_D\$, the tensor product basis \$B\$ is
constructed such that each column of \$B\$ is the element-wise product of one column
from each of the 1D basis matrices. This is equivalent to the Kronecker product of
the rows of the 1D basis matrices.
# Arguments
- `coords::AbstractMatrix`: An `N x D` matrix of data points.
- `nbins_per_dim::Vector{Int}`: A vector specifying the number of basis functions for each
  dimension.
- `family::Symbol`: The wavelet family to use for the 1D bases.
- `length_scale::Union{Real, AbstractVector}`: The length_scale(s) for the wavelets.
# Returns
- A basis matrix of size `(N, prod(nbins_per_dim))`.
"""
function TethysBayes_tensor_product_wavelet_basis(coords::AbstractMatrix, nbins_per_dim::Vector{Int},
    family::Symbol, length_scale::Union{Real, AbstractVector})
    n_dims = size(coords, 2)
    if length(nbins_per_dim) != n_dims
        error("Length of `nbins_per_dim` must match coordinate dimensions.")
    end
    length_scale_vec = if length_scale isa Real
        fill(Float64(length_scale), n_dims)
    else
        if length(length_scale) != n_dims
            error("Length of length_scale vector must match coordinate dimensions.")
        end
        length_scale
    end
    # Generate a 1D wavelet basis matrix for each dimension.
    basis_matrices_1D = [TethysBayes_wavelet_basis_1D(coords[:, i], nbins_per_dim[i], family,
        length_scale_vec[i]) for i in 1:n_dims]
    if isempty(basis_matrices_1D)
        return zeros(size(coords, 1), 0)
    end
    # Initialize the final basis with the matrix from the first dimension.
    B_final = basis_matrices_1D[1]
    # Iteratively compute the tensor product with the remaining basis matrices.
    for i in 2:n_dims
        B_next = basis_matrices_1D[i]
        n_obs, n_cols_final = size(B_final)
        _, n_cols_next = size(B_next)
        # Reshape for broadcasting to compute row-wise outer products.
        B_final_reshaped = reshape(B_final, n_obs, n_cols_final, 1)
        B_next_reshaped = reshape(B_next, n_obs, 1, n_cols_next)
        # The element-wise product creates the tensor product of the rows.
        tensor_prod = B_final_reshaped .* B_next_reshaped
        # Reshape the result into the final 2D basis matrix.
        B_final = reshape(tensor_prod, n_obs, n_cols_final * n_cols_next)
    end
    return B_final
end
"""
    TethysBayes_smooth_basis_1D(type::String, vals::AbstractVector, nbins::Int, degree::Int; ...)
Generates a 1D basis matrix. This version is CPU-only.
"""
function TethysBayes_smooth_basis_1D(
    type::String, 
    vals::AbstractVector, 
    nbins::Int, 
    degree::Int; 
    W=nothing, 
    knot_method::Symbol = :quantile, 
    custom_knots::Union{AbstractVector, Nothing} = nothing, 
    kwargs...
)
    n_obs = length(vals)
    v_min = minimum(vals)
    v_max = maximum(vals)
    v_std = std(vals) + 1e-9
    use_regular_grid = type in ["invdist", "kriging", "tps", "gp"]
    knots = if knot_method == :custom && !isnothing(custom_knots)
        custom_knots
    elseif knot_method == :range || use_regular_grid
        collect(range(v_min, stop=v_max, length=nbins))
    else
        quantile(vals, range(0, 1, length=nbins))
    end
    if type in ["pspline", "bspline"]
        return TethysBayes_bspline_basis(vals, nbins, degree; knot_method=knot_method,
            custom_knots=custom_knots)
    end
    B_out = Matrix{Float64}(undef, n_obs, nbins); fill!(B_out, 0.0)
    actual_nbins_generated = nbins
    if type in ["smooth", "barycentric", "linear"]
        h = (v_max - v_min) / (nbins > 1 ? (nbins - 1) : 1.0)
        h = h > 0 ? h : 1.0
        for m in 1:nbins
            dist = abs.(vals .- knots[m]) ./ h
            mask = dist .< 1.0
            B_out[mask, m] .= 1.0 .- dist[mask]
        end
    elseif type == "tps"
        for m in 1:nbins
            r = abs.(vals .- knots[m])
            B_out[:, m] .= r.^3
        end
    elseif type == "rff"
        length_scale = get(kwargs, :length_scale, v_std)
        Omega = randn(1, nbins) ./ length_scale
        Phi_phases = rand(nbins) .* (2.0 * pi)
        B_out .= sqrt(2.0 / nbins) .* cos.((vals * Omega) .+ Phi_phases')
    elseif type == "fft"
        length_scale = get(kwargs, :length_scale, v_std)
        t_coords = vals ./ length_scale
        idx = 1
        for m in 1:div(nbins, 2)
            arg = m .* t_coords
            B_out[:, idx] = sin.(2.0 * pi * arg)
            B_out[:, idx+1] = cos.(2.0 * pi * arg)
            idx += 2
        end
        if isodd(nbins) && idx <= nbins
            m = div(nbins, 2) + 1
            arg = m .* t_coords
            B_out[:, idx] = sin.(2.0 * pi * arg)
        end
    elseif type == "wavelet"
        family = get(kwargs, :family, :db4)
        length_scale = get(kwargs, :length_scale, 0.1)
        B_out = TethysBayes_wavelet_basis_1D(vals, nbins, family, length_scale)
    elseif type == "spherical"
        range_r = get(kwargs, :range, v_std * 2.0)
        for m in 1:nbins
            h = abs.(vals .- knots[m]) ./ range_r
            mask = h .< 1.0
            B_out[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end
    else
        B_out = ones(Float64, n_obs, 1)
        actual_nbins_generated = 1
    end
    return B_out, actual_nbins_generated
end
"""
    TethysBayes_smooth_basis_2D(type::String, coords::AbstractMatrix, nbins::Union{Int, Vector{Int}}; ...)
Generates a 2D basis matrix. This version is CPU-only.
"""
function TethysBayes_smooth_basis_2D(
    type::String, 
    coords::AbstractMatrix, 
    nbins::Union{Int, Vector{Int}}; 
    W=nothing, 
    knot_method::Symbol = :quantile, 
    custom_knots::Union{Tuple{AbstractVector, AbstractVector}, Nothing} = nothing, 
    kwargs...
)
    n_obs = size(coords, 1)
    n_marginal_x, n_marginal_y = if nbins isa Int
        (nbins, nbins)
    elseif nbins isa Vector{Int} && length(nbins) == 2
        (nbins[1], nbins[2])
    else
        error("For a 2D smooth, `nbins` must be an Int or a Vector{Int} of length 2.")
    end
    total_bins = n_marginal_x * n_marginal_y
    c_min = [minimum(coords[:, 1]), minimum(coords[:, 2])]
    c_max = [maximum(coords[:, 1]), maximum(coords[:, 2])]
    c_std = [std(coords[:, 1]), std(coords[:, 2])] .+ 1e-9
    ls_x = get(kwargs, :ls_x, c_std[1])
    ls_y = get(kwargs, :ls_y, c_std[2])
    use_regular_grid = type in ["invdist", "kriging", "tps", "spherical"]
    kx, ky = if knot_method == :custom && !isnothing(custom_knots)
        custom_knots
    elseif knot_method == :quantile && !use_regular_grid
        (quantile(coords[:, 1], range(0, 1, length=n_marginal_x)),
         quantile(coords[:, 2], range(0, 1, length=n_marginal_y)))
    else
        (collect(range(c_min[1], stop=c_max[1], length=n_marginal_x)),
         collect(range(c_min[2], stop=c_max[2], length=n_marginal_y)))
    end
    B = Matrix{Float64}(undef, n_obs, total_bins); fill!(B, 0.0)
    if type == "barycentric"
        knot_points = [Point2D(kx[i], ky[j]) for j in 1:n_marginal_y for i in 1:n_marginal_x]
        B = TethysBayes_barycentric_basis_2D(coords, knot_points)
    elseif type in ["pspline", "bspline"]
        degree_val = get(kwargs, :degree, 3)
        B = TethysBayes_tensor_product_basis(coords, [n_marginal_x, n_marginal_y], [degree_val,
            degree_val]; knot_method=knot_method, kwargs...)
    elseif type == "wavelet"
        family = get(kwargs, :family, :db4)
        length_scale = get(kwargs, :length_scale, 0.1)
        B = TethysBayes_tensor_product_wavelet_basis(coords, [n_marginal_x, n_marginal_y], family,
            length_scale)
    elseif type in ["smooth", "linear"]
        hx = (c_max[1] - c_min[1]) / (n_marginal_x > 1 ? (n_marginal_x - 1) : 1.0); hx = hx > 0 ? hx : 1.0
        hy = (c_max[2] - c_min[2]) / (n_marginal_y > 1 ? (n_marginal_y - 1) : 1.0); hy = hy > 0 ? hy : 1.0
        idx = 1
        for j in 1:n_marginal_y, i in 1:n_marginal_x
            if idx > total_bins
                break
            end
            b_x = max.(0.0, 1.0 .- abs.(coords[:, 1] .- kx[i]) ./ hx)
            b_y = max.(0.0, 1.0 .- abs.(coords[:, 2] .- ky[j]) ./ hy)
            B[:, idx] .= b_x .* b_y
            idx += 1
        end
    elseif type == "tps"
        centers = [(x, y) for y in ky for x in kx][:]
        for m in 1:total_bins
            dx = coords[:, 1] .- centers[m][1]
            dy = coords[:, 2] .- centers[m][2]
            r = sqrt.(dx.^2 .+ dy.^2)
            B[:, m] .= (r.^2) .* log.(r .+ 1e-9)
        end
    elseif type == "rff" || type == "anisotropic"
        Omega = randn(2, total_bins)
        Omega[1, :] ./= ls_x
        Omega[2, :] ./= ls_y
        Phi_phases = rand(total_bins) .* (2.0 * pi)
        B .= sqrt(2.0 / total_bins) .* cos.((coords * Omega) .+ Phi_phases')
    else
        @warn "Basis type '$type' not recognized for 2D smooth. Returning an empty basis matrix."
        return Matrix{Float64}(undef, n_obs, 0)
    end
    return B[:, 1:min(total_bins, size(B, 2))]
end
"""
    TethysBayes_smooth_basis_3D(type::String, coords::AbstractMatrix, nbins::Union{Int, Vector{Int}}; ...)
Generates a 3D basis matrix. This version is CPU-only.
"""
function TethysBayes_smooth_basis_3D(
    type::String, 
    coords::AbstractMatrix, 
    nbins::Union{Int, Vector{Int}}; 
    W=nothing, 
    knot_method::Symbol = :quantile, 
    custom_knots::Union{Tuple{AbstractVector, AbstractVector, AbstractVector}, Nothing} = nothing, 
    kwargs...
)
    n_obs = size(coords, 1)
    n_marginal_x, n_marginal_y, n_marginal_z = if nbins isa Int
        (nbins, nbins, nbins)
    elseif nbins isa Vector{Int} && length(nbins) == 3
        (nbins[1], nbins[2], nbins[3])
    else
        error("For a 3D smooth, `nbins` must be an Int or a Vector{Int} of length 3.")
    end
    total_bins = n_marginal_x * n_marginal_y * n_marginal_z
    c_min = [minimum(coords[:, i]) for i in 1:3]
    c_max = [maximum(coords[:, i]) for i in 1:3]
    c_std = [std(coords[:, i]) for i in 1:3] .+ 1e-9
    ls_x = get(kwargs, :ls_x, c_std[1])
    ls_y = get(kwargs, :ls_y, c_std[2])
    ls_z = get(kwargs, :ls_z, c_std[3])
    kx, ky, kz = if knot_method == :custom && !isnothing(custom_knots)
        custom_knots
    elseif knot_method == :quantile
        (quantile(coords[:, 1], range(0, 1, length=n_marginal_x)),
         quantile(coords[:, 2], range(0, 1, length=n_marginal_y)),
         quantile(coords[:, 3], range(0, 1, length=n_marginal_z)))
    else
        (collect(range(c_min[1], stop=c_max[1], length=n_marginal_x)),
         collect(range(c_min[2], stop=c_max[2], length=n_marginal_y)),
         collect(range(c_min[3], stop=c_max[3], length=n_marginal_z)))
    end
    B = Matrix{Float64}(undef, n_obs, total_bins); fill!(B, 0.0)
    if type in ["pspline", "bspline"]
        degree_val = get(kwargs, :degree, 3)
        B = TethysBayes_tensor_product_basis(coords, [n_marginal_x, n_marginal_y, n_marginal_z],
            fill(degree_val, 3); knot_method=knot_method, kwargs...)
    elseif type == "wavelet"
        family = get(kwargs, :family, :db4)
        length_scale = get(kwargs, :length_scale, 0.1)
        B = TethysBayes_tensor_product_wavelet_basis(coords, [n_marginal_x, n_marginal_y,
            n_marginal_z], family, length_scale)
    elseif type in ["smooth", "barycentric", "linear"]
        hx = (c_max[1] - c_min[1]) / (n_marginal_x > 1 ? (n_marginal_x - 1) : 1.0); hx = hx > 0 ? hx : 1.0
        hy = (c_max[2] - c_min[2]) / (n_marginal_y > 1 ? (n_marginal_y - 1) : 1.0); hy = hy > 0 ? hy : 1.0
        hz = (c_max[3] - c_min[3]) / (n_marginal_z > 1 ? (n_marginal_z - 1) : 1.0); hz = hz > 0 ? hz : 1.0
        idx = 1
        for k_idx in 1:n_marginal_z, j_idx in 1:n_marginal_y, i_idx in 1:n_marginal_x
            if idx > total_bins
                break
            end
            b_x = max.(0.0, 1.0 .- abs.(coords[:, 1] .- kx[i_idx]) ./ hx)
            b_y = max.(0.0, 1.0 .- abs.(coords[:, 2] .- ky[j_idx]) ./ hy)
            b_z = max.(0.0, 1.0 .- abs.(coords[:, 3] .- kz[k_idx]) ./ hz)
            B[:, idx] .= b_x .* b_y .* b_z
            idx += 1
        end
    elseif type == "rff"
        Omega = randn(3, total_bins)
        Omega[1, :] ./= ls_x; Omega[2, :] ./= ls_y; Omega[3, :] ./= ls_z
        Phi_phases = rand(total_bins) .* (2.0 * pi)
        B .= sqrt(2.0 / total_bins) .* cos.((coords * Omega) .+ Phi_phases')
    else
        B = ones(Float64, n_obs, total_bins)
    end
    return B[:, 1:min(total_bins, size(B, 2))]
end
"""
    TethysBayes_smooth_basis_4D(type::String, coords::AbstractMatrix, nbins::Union{Int, Vector{Int}}; ...)
Generates a 4D basis matrix. This version is CPU-only.
"""
function TethysBayes_smooth_basis_4D(
    type::String, 
    coords::AbstractMatrix, 
    nbins::Union{Int, Vector{Int}}; 
    W=nothing, 
    knot_method::Symbol = :quantile, 
    custom_knots::Union{Tuple{AbstractVector, AbstractVector, AbstractVector, AbstractVector}, Nothing} = nothing, 
    kwargs...
)
    n_obs = size(coords, 1)
    n_marginal_1, n_marginal_2, n_marginal_3, n_marginal_4 = if nbins isa Int
        (nbins, nbins, nbins, nbins)
    elseif nbins isa Vector{Int} && length(nbins) == 4
        (nbins[1], nbins[2], nbins[3], nbins[4])
    else
        error("For a 4D smooth, `nbins` must be an Int or a Vector{Int} of length 4.")
    end
    total_bins = n_marginal_1 * n_marginal_2 * n_marginal_3 * n_marginal_4
    c_min = [minimum(coords[:, i]) for i in 1:4]
    c_max = [maximum(coords[:, i]) for i in 1:4]
    c_std = [std(coords[:, i]) for i in 1:4] .+ 1e-9
    ls_1 = get(kwargs, :ls_1, c_std[1])
    ls_2 = get(kwargs, :ls_2, c_std[2])
    ls_3 = get(kwargs, :ls_3, c_std[3])
    ls_4 = get(kwargs, :ls_4, c_std[4])
    k1, k2, k3, k4 = if knot_method == :custom && !isnothing(custom_knots)
        custom_knots
    elseif knot_method == :quantile
        (quantile(coords[:, 1], range(0, 1, length=n_marginal_1)),
         quantile(coords[:, 2], range(0, 1, length=n_marginal_2)),
         quantile(coords[:, 3], range(0, 1, length=n_marginal_3)),
         quantile(coords[:, 4], range(0, 1, length=n_marginal_4)))
    else
        (collect(range(c_min[1], stop=c_max[1], length=n_marginal_1)),
         collect(range(c_min[2], stop=c_max[2], length=n_marginal_2)),
         collect(range(c_min[3], stop=c_max[3], length=n_marginal_3)),
         collect(range(c_min[4], stop=c_max[4], length=n_marginal_4)))
    end
    B = Matrix{Float64}(undef, n_obs, total_bins); fill!(B, 0.0)
    if type in ["pspline", "bspline"]
        degree_val = get(kwargs, :degree, 3)
        B = TethysBayes_tensor_product_basis(coords, [n_marginal_1, n_marginal_2, n_marginal_3,
            n_marginal_4], fill(degree_val, 4); knot_method=knot_method, kwargs...)
    elseif type == "wavelet"
        family = get(kwargs, :family, :db4)
        length_scale = get(kwargs, :length_scale, 0.1)
        B = TethysBayes_tensor_product_wavelet_basis(coords, [n_marginal_1, n_marginal_2,
            n_marginal_3, n_marginal_4], family, length_scale)
    elseif type in ["smooth", "linear", "barycentric"]
        hx1 = (c_max[1] - c_min[1]) / (n_marginal_1 > 1 ? (n_marginal_1 - 1) : 1.0); hx1 = hx1 > 0 ? hx1 : 1.0
        hx2 = (c_max[2] - c_min[2]) / (n_marginal_2 > 1 ? (n_marginal_2 - 1) : 1.0); hx2 = hx2 > 0 ? hx2 : 1.0
        hx3 = (c_max[3] - c_min[3]) / (n_marginal_3 > 1 ? (n_marginal_3 - 1) : 1.0); hx3 = hx3 > 0 ? hx3 : 1.0
        hx4 = (c_max[4] - c_min[4]) / (n_marginal_4 > 1 ? (n_marginal_4 - 1) : 1.0); hx4 = hx4 > 0 ? hx4 : 1.0
        idx = 1
        for l_idx in 1:n_marginal_4, k_idx in 1:n_marginal_3, j_idx in 1:n_marginal_2, i_idx in 1:n_marginal_1
            if idx > total_bins
                break
            end
            b1 = max.(0.0, 1.0 .- abs.(coords[:, 1] .- k1[i_idx]) ./ hx1)
            b2 = max.(0.0, 1.0 .- abs.(coords[:, 2] .- k2[j_idx]) ./ hx2)
            b3 = max.(0.0, 1.0 .- abs.(coords[:, 3] .- k3[k_idx]) ./ hx3)
            b4 = max.(0.0, 1.0 .- abs.(coords[:, 4] .- k4[l_idx]) ./ hx4)
            B[:, idx] .= b1 .* b2 .* b3 .* b4
            idx += 1
        end
    elseif type == "rff"
        Omega = randn(4, total_bins)
        Omega[1, :] ./= ls_1; Omega[2, :] ./= ls_2; Omega[3, :] ./= ls_3; Omega[4, :] ./= ls_4
        Phi_phases = rand(total_bins) .* (2.0 * pi)
        B .= sqrt(2.0 / total_bins) .* cos.((coords * Omega) .+ Phi_phases')
    else
        B = ones(Float64, n_obs, total_bins)
    end
    return B[:, 1:min(total_bins, size(B, 2))]
end
"""
    evaluate_kernel_matrix(coords::AbstractMatrix, param_val::Real, length_scale::Union{Real,
      AbstractVector}, kernel_type::Symbol, noise::Real; wavelet_levels=3)
Computes the covariance kernel matrix for a given set of coordinates.
# Version
v1.0.0
# Arguments
- `coords::AbstractMatrix`: An `N x D` matrix of data points, where `N` is the
  number of points and `D` is the number of dimensions.
- `param_val::Real`: The signal variance (\$\\sigma^2\$) of the kernel. This controls
  the overall amplitude of the function.
- `length_scale::Union{Real, AbstractVector}`: The length_scale(s) (\$\\ell\$) of the kernel.
  Controls the "wiggliness" or correlation distance. A `Real` value assumes an
  isotropic kernel, while a `Vector` of length `D` enables Automatic Relevance
  Determination (ARD) with a separate length_scale for each dimension.
- `kernel_type::Symbol`: The type of kernel to evaluate.
- `noise::Real`: A small jitter or "nugget" term added to the diagonal for
  numerical stability, representing observation noise.
- `wavelet_levels`: The number of levels for the wavelet kernel.
# Supported Kernels
- `:gaussian`, `:se`, `:rbf`: Squared Exponential kernel.
  \$k(x, x') = \\sigma^2 \\exp\\left(-\\frac{\\|x - x'\\|^2}{2\\ell^2}\\right)\$
- `:exponential`, `:matern12`: Exponential kernel (Matérn with \$\\nu=1/2\$).
  \$k(x, x') = \\sigma^2 \\exp\\left(-\\frac{\\|x - x'\\|}{\\ell}\\right)\$
- `:matern32`: Matérn kernel with \$\\nu=3/2\$.
- `:matern52`: Matérn kernel with \$\\nu=5/2\$.
- `:spherical`: Spherical kernel, which is compactly supported (zero beyond range \$\\ell\$).
- `:cosine`: Cosine kernel for periodic functions.
- `:linear`: Linear kernel.
- `:constant`: Constant kernel.
- `:wavelet`: A multi-scale kernel constructed from a sum of SE kernels.
# Returns
- A dense `N x N` covariance matrix.
Computes the covariance kernel matrix for a given set of coordinates 
"""
function evaluate_kernel_matrix(coords::AbstractMatrix, param_val::Real, length_scale::Union{Real,
    AbstractVector}, kernel_type::Symbol, noise::Real; wavelet_levels=3)
    T = promote_type(eltype(coords), typeof(param_val), eltype(length_scale), typeof(noise))
    coords_T = convert(AbstractMatrix{T}, coords)
    length_scale_T = convert(typeof(length_scale) <: Real ? T : AbstractVector{T}, length_scale)
    N = size(coords_T, 1)
    if kernel_type == :linear
        return param_val^2 .* (coords_T * coords_T') .+ (noise * I)
    end
    function _sqeuclidean_broadcast(X::AbstractMatrix)
        sum(X.^2, dims=2) .- 2 * (X * X') .+ sum(X.^2, dims=2)'
    end
    local dist_sq
    if length_scale isa AbstractVector # ARD case
        if size(coords_T, 2) != length(length_scale_T)
            error("Dimension mismatch for ARD kernel: Number of coordinate dimensions ($(size(coords_T, 2))) does not match number of lengthscales ($(length(length_scale_T))).")
        end
        dist_sq = _sqeuclidean_broadcast(coords_T ./ length_scale_T')
    else # Isotropic case
        dist_sq = _sqeuclidean_broadcast(coords_T) ./ length_scale_T^2
    end
    dist_sq .= max.(zero(T), dist_sq)
    if kernel_type == :gaussian || kernel_type == :se || kernel_type == :rbf
        return (param_val^2 .* exp.(-one(T)/2 .* dist_sq)) + (noise * I)
    elseif kernel_type == :exponential || kernel_type == :matern12
        d = sqrt.(dist_sq)
        return (param_val^2 .* exp.(-d)) + (noise * I)
    elseif kernel_type == :matern32
        d = sqrt.(dist_sq)
        val = sqrt(convert(T, 3.0)) .* d
        return (param_val^2 .* (one(T) .+ val) .* exp.(-val)) + (noise * I)
    elseif kernel_type == :matern52
        d = sqrt.(dist_sq)
        val = sqrt(convert(T, 5.0)) .* d
        return (param_val^2 .* (one(T) .+ val .+ (val.^2 ./ convert(T,
            3.0))) .* exp.(-val)) + (noise * I)
    elseif kernel_type == :spherical
        d = sqrt.(dist_sq)
        K = zeros(T, size(d))
        mask = d .< one(T)
        K[mask] = param_val^2 .* (one(T) .- 1.5 .* d[mask] .+ 0.5 .* d[mask].^3)
        return K + (noise * I)
    elseif kernel_type == :cosine
        if length_scale isa AbstractVector
            @warn "Cosine kernel with ARD length_scale is not standard. Using the first length_scale for an isotropic kernel."
            length_scale_T = length_scale_T[1]
        end
        d_euclidean = sqrt.(_sqeuclidean_broadcast(coords_T))
        return (param_val^2 .* cos.(2.0 * pi .* d_euclidean ./ length_scale_T)) + (noise * I)
    elseif kernel_type == :constant
        return fill(convert(T, param_val^2), size(dist_sq)) + (noise * I)
    elseif kernel_type == :wavelet
        local_length_scale = length_scale isa Real ? length_scale_T : length_scale_T[1]
        if length_scale isa AbstractVector
            @warn "Wavelet kernel with ARD length_scale is not standard. Using the first length_scale for decay."
        end
        K_accum = zeros(T, size(dist_sq))
        for wv_scale in 1:wavelet_levels
            length_scale_sq = (length_scale isa Real ? length_scale_T^2 : one(T)) / (convert(T, 4.0)^(wv_scale-1))
            weight_scale = param_val^2 * exp(convert(T, -wv_scale) / local_length_scale)
            K_accum .+= weight_scale .* exp.(-one(T)/2 .* dist_sq ./ length_scale_sq)
        end
        return K_accum + (noise * I)
    else
        @warn "Kernel '$(kernel_type)' not explicitly handled in evaluate_kernel_matrix. Defaulting to Squared Exponential."
        return param_val^2 .* exp.(-one(T)/2 .* dist_sq) .+ (noise * I)
    end
end
"""
    recompose_precision(m_type::Symbol, template_s::AbstractMatrix, param_val::Real; ...)
Constructs a final precision matrix from a template and sampled hyperparameters.
This version is CPU-only.
"""
function recompose_precision(m_type::Symbol, template_s::AbstractMatrix, param_val::Real;
    extra_param=nothing, noise=1e-4, kwargs...)
    n_s = size(template_s, 1)
    T_num = promote_type(typeof(param_val), typeof(noise), eltype(template_s), typeof(extra_param))
    if m_type == :SPDE
        spde_range = isnothing(extra_param) ? one(T_num) : extra_param
        Q_range = if spde_range isa Real
            spde_range^2 * I
        else
            if length(spde_range) != n_s
                error("Anisotropic spde_range vector length must match number of spatial units.")
            end
            Diagonal(spde_range.^2)
        end
        L_spde = Q_range + template_s
        return Symmetric(L_spde' * L_spde)
    end
    if m_type == :None || m_type == :FIXED
        return Symmetric(sparse(I, n_s, n_s))
    end
    if m_type == :Besag || m_type == :ICAR || m_type == :Cyclic
        return Symmetric(template_s)
    end
    if m_type == :AR1
        rho = isnothing(extra_param) ? zero(T_num) : extra_param
        Q = (one(T_num) + rho^2) * I + rho .* template_s
        if n_s > 0
            Q[1, 1] = one(T_num)
            Q[n_s, n_s] = one(T_num)
        end
        return Symmetric(Q)
    end
    if m_type == :Leroux || m_type == :LocalAdaptive
        lambda_val = isnothing(extra_param) ? convert(T_num, 0.5) : extra_param
        I_prom = I
        return Symmetric(lambda_val .* template_s + (one(T_num) - lambda_val) .* I_prom)
    end
    if m_type == :NetworkFlow
        rho_net = isnothing(extra_param) ? convert(T_num, 0.8) : extra_param
        W_net = template_s
        flow_direction = get(kwargs, :flow_direction, :bidirectional)
        L_op = if flow_direction == :upstream
            I - rho_net .* W_net'
        elseif flow_direction == :downstream
            I - rho_net .* W_net
        else
            W_symm = (W_net + W_net') ./ 2
            I - rho_net .* W_symm
        end
        return Symmetric(L_op' * L_op)
    end
    if m_type == :SAR || m_type == :DAG
        rho_p = isnothing(extra_param) ? convert(T_num, 0.8) : extra_param
        L_op = I - rho_p .* template_s
        return Symmetric(L_op' * L_op)
    end
    if m_type == :GP
        length_scale = isnothing(extra_param) ? one(T_num) : extra_param
        K = param_val^2 .* exp.(-(template_s) ./ (convert(T_num, 2.0) * length_scale^2))
        return inv(Symmetric(K + (noise * I)))
    end
    if m_type in [:RFF, :FFT, :BSpline, :PSpline, :TPS]
        return Symmetric(template_s)
    end
    return Symmetric(template_s)
end
"""
    _prior_or_constant(param_name, value)

Emit a Turing prior line for a hyperparameter that may be either a prior or a fixed value.

A `Real` is a *pinned constant*, not a prior: the generated model gets a plain binding so
the parameter name refers to a constant instead of a sampled variable. That makes
`sigma=2.0` mean what it looks like, instead of forcing callers to spell a degenerate
distribution to pin a hyperparameter. Anything else is a prior and is emitted as `~`.

Used by every component's `get_priors`, so a constant hyperparameter is also visible in the
generated model rather than being silently sampled.
"""
function _prior_or_constant(param_name, value)
    value isa Real && return "$(param_name) = $(value)"
    return "$(param_name) ~ $(_distribution_to_string(value))"
end
"""
    _distribution_to_string(d::Distribution)
Converts a `Distribution` object into a type-stable string representation of its
constructor call, suitable for dynamic code generation within a Turing `@model`.
# Arguments
- `d::Distribution`: The distribution object to convert.
# Returns
- `String`: A string representing the constructor call for the distribution.
"""
function _distribution_to_string(d::Distribution)
    dist_name = string(typeof(d).name.name)
    if d isa Exponential
        return "$(dist_name)($(rate(d)))"
    elseif d isa Normal
        return "$(dist_name)($(mean(d)), $(std(d)))"
    elseif d isa LogNormal
        return "$(dist_name)($(d.μ), $(d.σ))"
    elseif d isa Beta
        # Access alpha and beta parameters directly for Beta distribution
        return "$(dist_name)($(d.α), $(d.β))"
    elseif d isa InverseGamma
        return "$(dist_name)($(Distributions.shape(d)), $(Distributions.scale(d)))"
    elseif d isa Gamma
        return "$(dist_name)($(Distributions.shape(d)), $(Distributions.scale(d)))"
    elseif d isa Uniform
        return "$(dist_name)($(minimum(d)), $(maximum(d)))"
    elseif d isa TDist
        return "$(dist_name)($(d.df))"
    elseif d isa Dirac
        return "$(dist_name)($(d.value))"
    elseif d isa Categorical
        return "$(dist_name)($(d.p))" # Probabilities are fixed, no need for {T}
    elseif d isa LKJCholesky
        # Essential for multivariate models (e.g., mixed effects)
        return "$(dist_name)($(d.d), $(d.eta))"
    elseif d isa MvNormal
        # Handle common cases for MvNormal, especially with UniformScaling
        mean_str = string(mean(d))
        cov_str = if d.Σ isa UniformScaling
            "$(d.Σ.λ) * I"
        else
            string(d.Σ) # Fallback for other matrix types
        end
        return "$(dist_name)($(mean_str), $(cov_str))"
    elseif d isa Truncated
        inner_dist_str = _distribution_to_string(d.untruncated)
        lower_val = isnothing(d.lower) ? -Inf : d.lower
        upper_val = isnothing(d.upper) ? Inf : d.upper
        lower_str = isinf(lower_val) ? (lower_val < 0 ? "-Inf" : "Inf") : "$(lower_val)"
        upper_str = isinf(upper_val) ? (upper_val < 0 ? "-Inf" : "Inf") : "$(upper_val)"
        return "truncated($(inner_dist_str), $(lower_str), $(upper_str))"
    elseif d isa Product
        if hasproperty(d, :v) && d.v isa Fill
            inner_dist_str = _distribution_to_string(d.v.value)
            n = length(d.v)
            return "filldist($(inner_dist_str), $(n))"
        elseif hasproperty(d, :v) && all(dist -> dist == d.v[1], d.v)
             inner_dist_str = _distribution_to_string(d.v[1])
             n = length(d.v)
             return "filldist($(inner_dist_str), $(n))"
        else
            inner_strs = [_distribution_to_string(dist) for dist in d.v]
            return "Product([$(join(inner_strs, ", "))])"
        end
    elseif d isa Fill
        inner_dist_str = _distribution_to_string(d.value)
        n = length(d)
        return "filldist($(inner_dist_str), $(n))"
    else
        @warn "String conversion for distribution type `$(typeof(d))` is not explicitly handled. The generated code may not be type-stable."
        return string(d)
    end
end



"""

    _compute_block_init_epsilon(dim::Int, key::Union{Symbol, String, Nothing}, init_ϵ,

      min_ϵ::Real, max_ϵ::Real)::Float64

Calculates or retrieves a well-conditioned initial proposal step size (ϵ) for an HMC/NUTS
  parameter block of dimension `dim`.
Uses optimal Roberts & Rosenthal (2001) dimensional scaling ϵ ~ O(dim^(-1/4)) clamped within
  [min_ϵ, max_ϵ].
"""

function _compute_block_init_epsilon(
    dim::Int,
    key::Union{Symbol, String, Nothing},
    init_ϵ::Union{Real, Dict, Symbol, Nothing},
    min_ϵ::Real,
    max_ϵ::Real
)::Float64
    if init_ϵ === 0.0 || init_ϵ === :turing
        return 0.0 # Tells Turing to run default unconstrained search
    elseif init_ϵ isa Real
        return Float64(clamp(init_ϵ, min_ϵ, max_ϵ))
    elseif init_ϵ isa Dict
        k_sym = !isnothing(key) ? Symbol(key) : :default
        if haskey(init_ϵ, k_sym)
            val = init_ϵ[k_sym]
            return val == 0.0 ? 0.0 : Float64(clamp(val, min_ϵ, max_ϵ))
        elseif haskey(init_ϵ, :default)
            val = init_ϵ[:default]
            return val == 0.0 ? 0.0 : Float64(clamp(val, min_ϵ, max_ϵ))
        end
    end



    # Default :auto / precomputed proposal heuristic:

    # 0.2 * dim^(-1/4), clamped to [min_ϵ, min(max_ϵ, 0.25)]

    # Prevents large initial leapfrog overshoots on low-dimensional GLM blocks (intercept/slopes)

    scale_factor = 0.2 * (max(dim, 1)^(-0.25))

    effective_max = min(max_ϵ, 0.25)

    return Float64(clamp(scale_factor, min_ϵ, effective_max))

end



"""

    _compute_block_max_depth(key::Union{Symbol, String, Nothing}, max_depth::Union{Int,

      Dict, Symbol, Nothing}, hyper=nothing)::Int

Determines maximum tree depth for NUTS trajectory doubling in block `key`.
If hyper contains spectral Laplacian eigenvalues (hyper.L), uses theoretical condition
  number bound L* = ceil(log2(π * sqrt(κ))).
"""

function _compute_block_max_depth(
    key::Union{Symbol, String, Nothing},
    max_depth::Union{Int, Dict, Symbol, Nothing},
    hyper::Union{NamedTuple, Dict, Nothing}=nothing
)::Int
    if max_depth isa Int
        return max(1, max_depth)
    elseif max_depth isa Dict
        k_sym = !isnothing(key) ? Symbol(key) : :default
        if haskey(max_depth, k_sym)
            val = max_depth[k_sym]
            return val isa Int ? max(1, val) : 10
        elseif haskey(max_depth, :default)
            val = max_depth[:default]
            return val isa Int ? max(1, val) : 10
        end
    end


    # Automatic spectral condition number estimation:
    # L_steps ≈ π * sqrt(κ), where κ = λ_max / λ_min
    if !isnothing(hyper) && hasproperty(hyper, :L)
        L_eig = hyper.L
        pos_L = filter(x -> x > 1e-6, L_eig)
        if length(pos_L) >= 2
            anisotropy = maximum(pos_L) / minimum(pos_L)
            theoretical_depth = ceil(Int, log2(pi * sqrt(anisotropy)))
            return clamp(theoretical_depth, 4, 10)
        end
    end



    return 10

end



"""

    _compute_block_target_acceptance(key::Union{Symbol, String, Nothing}, target_acceptance,

      hyper=nothing)::Float64

Determines optimal target acceptance rate (δ) for HMC/NUTS block `key`.
Uses δ = 0.65 for isotropic/well-conditioned blocks and δ = 0.80 - 0.90 for high condition
  number GMRF/hierarchical manifolds.
"""

function _compute_block_target_acceptance(
    key::Union{Symbol, String, Nothing},
    target_acceptance::Union{Real, Dict, Symbol, Nothing},
    hyper::Union{NamedTuple, Dict, Nothing}=nothing
)::Float64
    if target_acceptance isa Real
        return Float64(clamp(target_acceptance, 0.5, 0.99))
    elseif target_acceptance isa Dict
        k_sym = !isnothing(key) ? Symbol(key) : :default
        if haskey(target_acceptance, k_sym)
            return Float64(clamp(target_acceptance[k_sym], 0.5, 0.99))
        elseif haskey(target_acceptance, :default)
            return Float64(clamp(target_acceptance[:default], 0.5, 0.99))
        end
    end

    # Condition number adaptation:
    if !isnothing(hyper) && hasproperty(hyper, :L)
        L_eig = hyper.L
        pos_L = filter(x -> x > 1e-6, L_eig)
        if length(pos_L) >= 2
            anisotropy = maximum(pos_L) / minimum(pos_L)
            return anisotropy > 1000.0 ? 0.90 : (anisotropy > 100.0 ? 0.80 : 0.65)
        end
    end



    return 0.80

end



"""

    _compute_block_adaptation_steps(key, adaptation_steps, dim, hyper=nothing,
      n_samples=nothing; use_dense_metric=false)::Int
Calculates the mathematically principled number of warmup/adaptation steps for an MCMC block.
Accounts for:
1. Metric estimation dimension: Diagonal metric requires O(sqrt(D)) variance exploration;
  Dense metric requires O(D) covariance samples.
2. Dual averaging convergence: Step-size dual averaging requires ~150-300 iterations to
  stabilize within ±5% error.
3. Condition number (κ): High condition numbers (> 1000) require extended settling.
4. Total sample ceiling (if n_samples provided).
"""
function _compute_block_adaptation_steps(
    key::Union{Symbol, String, Nothing},
    adaptation_steps::Union{Int, Dict, Symbol, Nothing},
    dim::Int,
    hyper::Union{NamedTuple, Dict, Nothing}=nothing,
    n_samples::Union{Int, Nothing}=nothing;
    use_dense_metric::Bool=false
)::Int
    if adaptation_steps isa Int
        return max(5, adaptation_steps)
    elseif adaptation_steps isa Dict
        k_sym = !isnothing(key) ? Symbol(key) : :default
        if haskey(adaptation_steps, k_sym)
            val = adaptation_steps[k_sym]
            return val isa Int ? max(5, val) : 300
        elseif haskey(adaptation_steps, :default)
            val = adaptation_steps[:default]
            return val isa Int ? max(5, val) : 300
        end
    end
    # Automatic principled calculation:
    base_steps = if use_dense_metric && dim > 1
        150 + 4 * dim
    else
        150 + ceil(Int, 25.0 * sqrt(max(1, dim)))
    end
    if !isnothing(hyper) && hasproperty(hyper, :L)
        L_eig = hyper.L
        pos_L = filter(x -> x > 1e-6, L_eig)
        if length(pos_L) >= 2
            anisotropy = maximum(pos_L) / minimum(pos_L)
            if anisotropy > 1000.0
                base_steps = ceil(Int, base_steps * 1.3)
            end
        end
    end
    calculated = clamp(base_steps, 100, 1000)
    if !isnothing(n_samples) && n_samples > 0
        if n_samples <= 200
            return max(5, min(calculated, ceil(Int, 0.5 * n_samples)))
        else
            return min(calculated, ceil(Int, 0.5 * n_samples))
        end
    end
    return calculated
end
"""
    _resolve_adtype(adtype::Union{Symbol, ADTypes.AbstractADType}, num_params::Int;
      param_threshold::Int=100)::ADTypes.AbstractADType
Resolves the automatic differentiation backend for MCMC samplers.
Supports Symbol specifications (`:forwarddiff`, `:reversediff`, `:enzyme`, `:tracker`)
as well as instantiated `ADTypes.AbstractADType` objects.
Automatically switches to reverse-mode differentiation when `num_params > param_threshold`.
"""
function _resolve_adtype(
    adtype::Union{Symbol, ADTypes.AbstractADType},
    num_params::Int;
    param_threshold::Int=100
)::ADTypes.AbstractADType
    resolved = if adtype isa Symbol
        s = Symbol(lowercase(string(adtype)))
        if s == :forwarddiff
            ADTypes.AutoForwardDiff()
        elseif s == :reversediff
            ADTypes.AutoReverseDiff(compile=false)
        elseif s == :enzyme
            Sys.iswindows() ? ADTypes.AutoReverseDiff(compile=false) : ADTypes.AutoEnzyme()
        elseif s == :tracker
            ADTypes.AutoTracker()
        else
            @warn "Unknown AD backend symbol :$(adtype); defaulting to AutoForwardDiff()."
            ADTypes.AutoForwardDiff()
        end
    else
        adtype
    end
    if (resolved isa ADTypes.AutoForwardDiff && num_params > param_threshold) ||
       resolved isa ADTypes.AutoEnzyme
        if Sys.iswindows()
            if resolved isa ADTypes.AutoEnzyme
                @warn "Enzyme is currently not stable on Windows. Switching to ReverseDiff backend."
            end
            return ADTypes.AutoReverseDiff(compile=false)
        else
            @info "Model has > $(param_threshold) parameters. Using Enzyme for reverse-mode performance."
            return ADTypes.AutoEnzyme()
        end
    end
    return resolved
end
"""
    _resolve_metric_type(use_dense_metric::Bool, dim::Int)
Selects between `DenseEuclideanMetric` and `DiagEuclideanMetric` for HMC/NUTS samplers.
Dense metrics are enabled only when `use_dense_metric` is true and dimension is moderate
(2 <= dim <= 50) to prevent ill-conditioning during sample covariance estimation.
"""
function _resolve_metric_type(use_dense_metric::Bool, dim::Int)
    if use_dense_metric && dim > 1 && dim <= 50
        return Turing.Inference.AdvancedHMC.DenseEuclideanMetric
    else
        return Turing.Inference.AdvancedHMC.DiagEuclideanMetric
    end
end
"""
    _extract_all_components(M)
Recursively extracts all model components, including those from top-level `M.components`
and nested multi-fidelity components from `M.nested_components`.
Returns a Vector of NamedTuples: `(key = Symbol, spec = spec, hyper = hyper)`.
"""
function _extract_all_components(M)
    comp_list = Vector{NamedTuple{(:key, :spec, :hyper), Tuple{Symbol, Any, Any}}}()
    if hasproperty(M, :components) && !isempty(M.components)
        for s in M.components
            k_sym = Symbol(s.key)
            h = hasproperty(s, :hyper) ? s.hyper : nothing
            push!(comp_list, (key=k_sym, spec=s, hyper=h))
        end
    end
    if hasproperty(M, :nested_components) && !isempty(M.nested_components)
        for (sub_k, sub_M) in pairs(M.nested_components)
            if hasproperty(sub_M, :components) && !isempty(sub_M.components)
                for s in sub_M.components
                    nested_k = Symbol("$(sub_k)_$(s.key)")
                    h = hasproperty(s, :hyper) ? s.hyper : nothing
                    push!(comp_list, (key=nested_k, spec=s, hyper=h))
                end
            end
        end
    end
    return comp_list
end
"""
    _normalize_sampler_map(sampler_map)
Normalizes user-supplied `sampler_map` into a `Dict{Symbol, AbstractMCMC.AbstractSampler}`.
Accepts `AbstractDict`, `NamedTuple`, or `nothing`.
"""
function _normalize_sampler_map(sampler_map)
    res = Dict{Symbol, AbstractMCMC.AbstractSampler}()
    if isnothing(sampler_map)
        return res
    elseif sampler_map isa NamedTuple
        for (k, v) in pairs(sampler_map)
            if v isa AbstractMCMC.AbstractSampler
                res[Symbol(k)] = v
            end
        end
    elseif sampler_map isa AbstractDict
        for (k, v) in sampler_map
            if v isa AbstractMCMC.AbstractSampler
                res[Symbol(k)] = v
            end
        end
    end
    return res
end
"""
    _classify_parameter_support(vn, vi, param_reg)
Classifies a model parameter into `:discrete`, `:bounded`, or `:continuous` based on
registered prior distributions, active VarInfo values, and Bijector transforms.
"""
function _classify_parameter_support(vn, vi, param_reg)
    vn_sym = _get_varname_symbol(vn)
    if !isnothing(param_reg) && haskey(param_reg.descriptors, vn_sym)
        desc = param_reg.descriptors[vn_sym]
        if !isnothing(desc.prior)
            if desc.prior isa Distributions.DiscreteDistribution
                return :discrete
            elseif desc.prior isa Distributions.Truncated ||
                   desc.prior isa Distributions.Beta ||
                   desc.prior isa Distributions.Uniform
                return :bounded
            end
        end
    end
    val = try
        vi[vn]
    catch
        try
            hasproperty(vi, vn_sym) ? getproperty(vi, vn_sym) : nothing
        catch
            nothing
        end
    end
    if !isnothing(val) && (val isa Integer || (val isa AbstractArray && eltype(val) <: Integer))
        return :discrete
    end
    if hasproperty(vi, :transform_strategy) &&
       (vi.transform_strategy isa AbstractDict) &&
       haskey(vi.transform_strategy, vn)
        transform = vi.transform_strategy[vn]
        if transform isa Bijectors.Exp || transform isa Bijectors.Logistic
            return :bounded
        end
    end
    return :continuous
end
"""
    precompute_step_sizes(model_obj::DynamicPPL.Model; min_ϵ::Real=1e-4, max_ϵ::Real=1.0,
      max_depth=10, target_acceptance=0.8, adaptation_steps=:auto, init_ϵ=:auto,
      n_samples=nothing)
Pre-computes and summarizes the initial proposal step sizes (ϵ), maximum tree depths,
recommended target acceptance rates, condition numbers, and principled adaptation steps
for all parameter blocks in a model based on Roberts & Rosenthal (2001) dimensional
curvature scaling and spectral graph Laplacian condition numbers.

# Arguments
- `model_obj::DynamicPPL.Model`: The instantiated Turing model.
- `min_ϵ::Real`: Lower bound on initial step size (default `1e-4`).
- `max_ϵ::Real`: Upper bound on initial step size (default `1.0`).
- `max_depth`: Maximum NUTS tree depth (default `10`).
- `target_acceptance`: Target acceptance rate (default `0.80`).
- `adaptation_steps`: Warmup iteration count (default `:auto`).
- `init_ϵ`: Initial step size heuristic or mapping (default `:auto`).
- `n_samples`: Total posterior sample count for adaptation budget ceiling.

# Returns
- A `Dict{Symbol, NamedTuple}` detailing each block's dimension, step size, max depth,
  target acceptance, adaptation steps, condition number, and variable names.
"""
function precompute_step_sizes(
    model_obj::DynamicPPL.Model;
    min_ϵ::Real=1e-4,
    max_ϵ::Real=1.0,
    max_depth::Union{Int, Dict{Symbol, Int}, Symbol, Nothing}=10,
    target_acceptance::Union{Real, Dict{Symbol, <:Real}, Symbol, Nothing}=0.8,
    adaptation_steps::Union{Int, Dict{Symbol, Int}, Symbol, Nothing}=:auto,
    init_ϵ::Union{Real, Dict{Symbol, <:Real}, Symbol, Nothing}=:auto,
    n_samples::Union{Int, Nothing}=nothing
)
    vi = DynamicPPL.VarInfo(model_obj)
    vns = keys(vi)
    result = Dict{Symbol, NamedTuple{(:dim, :init_ϵ, :max_depth, :target_acceptance,
        :adaptation_steps, :condition_number, :variables), Tuple{Int, Float64, Int, Float64,
        Int, Float64, Vector{Symbol}}}}()
    all_comps = hasproperty(model_obj.args, :M) ? _extract_all_components(model_obj.args.M) : []
    processed = Set{VarName}()
    if !isempty(all_comps)
        for comp in all_comps
            k_sym = comp.key
            k_str = string(k_sym)
            comp_vns = filter(
                vn -> occursin(Regex("_$(k_str)(_\\d+)?\$"), string(vn)) ||
                      startswith(string(vn), "$(k_str)_"),
                vns
            )
            if !isempty(comp_vns)
                dim = length(comp_vns)
                comp_hyper = comp.hyper
                range_val = 1.0
                if !isnothing(comp_hyper) && hasproperty(comp_hyper, :L)
                    pos_L = filter(x -> x > 1e-6, comp_hyper.L)
                    if length(pos_L) >= 2
                        range_val = Float64(maximum(pos_L) / minimum(pos_L))
                    end
                end
                eps_val = _compute_block_init_epsilon(dim, k_sym, init_ϵ, min_ϵ, max_ϵ)
                depth_val = _compute_block_max_depth(k_sym, max_depth, comp_hyper)
                acc_val = _compute_block_target_acceptance(k_sym, target_acceptance, comp_hyper)
                adapt_val = _compute_block_adaptation_steps(k_sym, adaptation_steps, dim,
                    comp_hyper, n_samples)
                result[k_sym] = (
                    dim = dim,
                    init_ϵ = eps_val,
                    max_depth = depth_val,
                    target_acceptance = acc_val,
                    adaptation_steps = adapt_val,
                    condition_number = range_val,
                    variables = [_get_varname_symbol(v) for v in comp_vns]
                )
                union!(processed, comp_vns)
            end
        end
    end
    # Check for fixed effects block
    fixed_vns = filter(
        vn -> vn in setdiff(vns, processed) &&
              (occursin(r"^(intercept|beta(_flat)?)(_\d+)?$", string(vn)) ||
               startswith(string(vn), "beta_") || startswith(string(vn), "intercept_")),
        vns
    )
    if !isempty(fixed_vns)
        dim = length(fixed_vns)
        eps_val = _compute_block_init_epsilon(dim, :fixed, init_ϵ, min_ϵ, max_ϵ)
        depth_val = _compute_block_max_depth(:fixed, max_depth, nothing)
        acc_val = _compute_block_target_acceptance(:fixed, target_acceptance, nothing)
        adapt_val = _compute_block_adaptation_steps(:fixed, adaptation_steps, dim,
            nothing, n_samples)
        result[:fixed] = (
            dim = dim,
            init_ϵ = eps_val,
            max_depth = depth_val,
            target_acceptance = acc_val,
            adaptation_steps = adapt_val,
            condition_number = 1.0,
            variables = [_get_varname_symbol(v) for v in fixed_vns]
        )
        union!(processed, fixed_vns)
    end
    remaining = setdiff(vns, processed)
    if !isempty(remaining)
        dim = length(remaining)
        eps_val = _compute_block_init_epsilon(dim, :remaining, init_ϵ, min_ϵ, max_ϵ)
        depth_val = _compute_block_max_depth(:remaining, max_depth, nothing)
        acc_val = _compute_block_target_acceptance(:remaining, target_acceptance, nothing)
        adapt_val = _compute_block_adaptation_steps(:remaining, adaptation_steps, dim,
            nothing, n_samples)
        result[:remaining] = (
            dim = dim,
            init_ϵ = eps_val,
            max_depth = depth_val,
            target_acceptance = acc_val,
            adaptation_steps = adapt_val,
            condition_number = 1.0,
            variables = [_get_varname_symbol(v) for v in remaining]
        )
    end
    return result
end
"""
    get_optimal_sampler(model_obj::DynamicPPL.Model; ...)
Constructs a mathematically optimal composite MCMC sampler for a `TethysBayes` model by
assigning specialized samplers to parameter blocks with Roberts & Rosenthal (2001)
dimensional step-size scaling (ϵ ~ O(D^(-1/4))), dual-averaging warmup lengths,
spectral condition-number depth bounds, and adaptive Euclidean metric conditioning.
# Mathematical Principles
1. **Dimensional Step Scaling**: Initial proposal step sizes scale as ϵ ~ 0.2 * D^(-1/4),
   mitigating leapfrog energy drift in high-dimensional latent random fields while
   preventing initial trajectory overshoots.
2. **Spectral Manifold Tuning**: In spatial GMRF models with graph Laplacian eigenvalue
   spectrum λ, trajectory length scales as L* ≈ π * sqrt(κ) where κ = λ_max / λ_min,
   bounding tree depth to prevent trajectory loops and U-turns.
3. **Metric Selection**: `DenseEuclideanMetric` is applied for moderate parameter dimensions
   (2 <= D <= 50) when `use_dense_metric_for_components=true` to capture cross-parameter
   covariance without incurring cubic numerical instability.
4. **Support-Aware Partitioning**: Discrete parameters are isolated into Particle Gibbs (PG),
   bounded parameters into Slice sampling, and continuous parameters into condition-adapted NUTS.
# Arguments
- `model_obj::DynamicPPL.Model`: The instantiated Turing model.
- `sampler_choice`: A specific sampler instance, or a Symbol (`:auto`, `:nuts`, `:hmc`,
  `:mh`, `:slice`, `:pg`, `:gibbs`). Defaults to `:auto`.
- `sampler_map`: A `Dict` or `NamedTuple` mapping parameter `Symbol`s to specific samplers.
- `adtype`: The automatic differentiation backend (`AutoForwardDiff()`, `:reversediff`,
  `:enzyme`, etc.).
- `group_components::Bool`: Whether to group component parameters into joint Gibbs blocks.
  Defaults to `true`.
- `use_dense_metric_for_components::Bool`: Whether to use `DenseEuclideanMetric` for
  correlated component blocks with dimension <= 50. Defaults to `true`.
- `adaptation_steps`: Adaptation warmup steps (Int, Dict, or `:auto`).
- `target_acceptance`: Target acceptance rate for NUTS (Real, Dict, or `:auto`). Defaults to `0.80`.
- `init_ϵ`: Initial step size for NUTS (Real, Dict, or `:auto`).
- `init_epsilon`, `step_size`: Keyword aliases for `init_ϵ`.
- `max_depth`: Maximum tree depth for NUTS (Int or Dict). Defaults to `10`.
- `min_ϵ::Real`: Minimum allowed initial step size. Defaults to `1e-4`.
- `max_ϵ::Real`: Maximum allowed initial step size. Defaults to `1.0`.
- `n_samples`: Total sampling count per chain, used as an adaptation ceiling.
- `n_particles::Int`: Number of particles for `PG` sampler (default `20`).
- `n_chains::Int`: Number of chains to be run (default `1`).
# Returns
- An `AbstractMCMC.AbstractSampler` (e.g. `Turing.Gibbs` or single `Turing.NUTS`).
"""
function get_optimal_sampler(
    model_obj::DynamicPPL.Model;
    sampler_choice::Union{Symbol, AbstractMCMC.AbstractSampler}=:auto,
    sampler_map::Union{AbstractDict, NamedTuple, Nothing}=nothing,
    adtype::Union{Symbol, ADTypes.AbstractADType}=ADTypes.AutoForwardDiff(),
    group_components::Bool=true,
    use_dense_metric_for_components::Bool=true,
    adaptation_steps::Union{Int, Dict{Symbol, Int}, Symbol, Nothing}=:auto,
    target_acceptance::Union{Real, Dict{Symbol, <:Real}, Symbol, Nothing}=0.8,
    init_ϵ::Union{Real, Dict{Symbol, <:Real}, Symbol, Nothing}=:auto,
    init_epsilon::Union{Real, Dict{Symbol, <:Real}, Symbol, Nothing}=nothing,
    step_size::Union{Real, Dict{Symbol, <:Real}, Symbol, Nothing}=nothing,
    max_depth::Union{Int, Dict{Symbol, Int}, Symbol, Nothing}=10,
    min_ϵ::Real=1e-4,
    max_ϵ::Real=1.0,
    n_samples::Union{Int, Nothing}=nothing,
    n_particles::Int=20,
    n_chains::Int=1
)
    # Resolve step size aliases
    effective_init_ϵ = !isnothing(step_size) ? step_size : (
        !isnothing(init_epsilon) ? init_epsilon : init_ϵ
    )
    # Stage 0: Direct sampler instance pass-through
    if sampler_choice isa AbstractMCMC.AbstractSampler
        return sampler_choice
    end
    # Normalize sampler map
    norm_sampler_map = _normalize_sampler_map(sampler_map)
    # Inspect VarInfo and extract variables using invokelatest to guard world age
    vi = Base.invokelatest(DynamicPPL.VarInfo, model_obj)
    vns = keys(vi)
    num_params = length(vns)
    param_reg = if hasproperty(model_obj.args, :spec_registry) &&
                   haskey(model_obj.args.spec_registry, :parameters)
        model_obj.args.spec_registry[:parameters]
    elseif hasproperty(model_obj.args, :M)
        build_param_registry(model_obj.args.M)
    else
        build_param_registry(model_obj)
    end
    # Resolve global AD backend
    adtype_to_use = _resolve_adtype(adtype, num_params; param_threshold=100)
    # Classify support for each variable
    var_supports = Dict(vn => _classify_parameter_support(vn, vi, param_reg) for vn in vns)
    discrete_all = filter(vn -> var_supports[vn] == :discrete, collect(vns))
    continuous_all = filter(vn -> var_supports[vn] != :discrete, collect(vns))
    # Stage 0b: Handle non-auto global sampler requests
    if sampler_choice isa Symbol
        choice_sym = Symbol(lowercase(string(sampler_choice)))
        if choice_sym == :nuts
            if isempty(continuous_all)
                return PG(n_particles)
            end
            dim_c = length(continuous_all)
            metric_type = _resolve_metric_type(use_dense_metric_for_components, dim_c)
            global_eps = _compute_block_init_epsilon(dim_c, :global, effective_init_ϵ, min_ϵ, max_ϵ)
            global_depth = _compute_block_max_depth(:global, max_depth, nothing)
            global_acc = _compute_block_target_acceptance(:global, target_acceptance, nothing)
            global_adapt = _compute_block_adaptation_steps(:global, adaptation_steps, dim_c,
                nothing, n_samples; use_dense_metric=use_dense_metric_for_components)
            nuts_sampler = if global_eps > 0.0
                NUTS(global_adapt, global_acc; max_depth=global_depth, init_ϵ=global_eps,
                    metricT=metric_type, adtype=adtype_to_use)
            else
                NUTS(global_adapt, global_acc; max_depth=global_depth, metricT=metric_type,
                    adtype=adtype_to_use)
            end
            if isempty(discrete_all)
                return nuts_sampler
            else
                @info "Model contains $(length(discrete_all)) discrete and $(dim_c) continuous parameters. Partitioning into Gibbs(PG, NUTS)."
                return Gibbs(Tuple(discrete_all) => PG(n_particles), Tuple(continuous_all) => nuts_sampler)
            end
        elseif choice_sym == :hmc
            dim_c = length(continuous_all)
            global_eps = _compute_block_init_epsilon(dim_c, :global, effective_init_ϵ, min_ϵ, max_ϵ)
            global_depth = _compute_block_max_depth(:global, max_depth, nothing)
            hmc_step = global_eps > 0.0 ? global_eps : 0.05
            hmc_steps = max(1, min(global_depth, 10))
            hmc_sampler = HMC(hmc_step, hmc_steps; adtype=adtype_to_use)
            if isempty(discrete_all)
                return hmc_sampler
            else
                return Gibbs(Tuple(discrete_all) => PG(n_particles), Tuple(continuous_all) => hmc_sampler)
            end
        elseif choice_sym == :mh
            return MH()
        elseif choice_sym in (:slice, :ess)
            return ESS()
        elseif choice_sym == :pg
            return PG(n_particles)
        elseif !(choice_sym in (:auto, :gibbs))
            @warn "Unrecognized sampler_choice :$(sampler_choice); defaulting to :auto composite Gibbs partitioning."
        end
    end
    sampler_assignments = []
    all_processed_vns = Set{VarName}()
    # Stage 1: User-provided sampler map (highest precedence)
    for (param_sym, sampler) in norm_sampler_map
        sym_vns = filter(vns) do vn
            vn_sym = _get_varname_symbol(vn)
            if vn_sym == param_sym || Symbol(string(vn)) == param_sym
                return true
            end
            if !isnothing(param_reg) && haskey(param_reg.descriptors, vn_sym)
                desc = param_reg.descriptors[vn_sym]
                if desc.component_key == param_sym
                    return true
                end
            end
            k_str = string(param_sym)
            vn_str = string(vn)
            return occursin(Regex("_$(k_str)(_\\d+)?\$"), vn_str) ||
                   startswith(vn_str, "$(k_str)_")
        end
        if !isempty(sym_vns)
            push!(sampler_assignments, Tuple(sym_vns) => sampler)
            union!(all_processed_vns, sym_vns)
        else
            @warn "Parameter or component :$(param_sym) in sampler_map not found in model."
        end
    end
    # Stage 2: Group components
    if group_components && hasproperty(model_obj.args, :M)
        all_comps = _extract_all_components(model_obj.args.M)
        sort!(all_comps, by=c -> length(string(c.key)), rev=true)
        component_groups = Dict{Symbol, Set{VarName}}()
        vns_to_check = setdiff(vns, all_processed_vns)
        for vn in vns_to_check
            vn_str = string(vn)
            vn_sym = _get_varname_symbol(vn)
            found_key = nothing
            if !isnothing(param_reg) && haskey(param_reg.descriptors, vn_sym)
                desc_key = param_reg.descriptors[vn_sym].component_key
                if desc_key in [c.key for c in all_comps]
                    found_key = desc_key
                end
            end
            if isnothing(found_key)
                for comp in all_comps
                    k_str = string(comp.key)
                    if occursin(Regex("_$(k_str)(_\\d+)?\$"), vn_str) ||
                       startswith(vn_str, "$(k_str)_")
                        found_key = comp.key
                        break
                    end
                end
            end
            if !isnothing(found_key)
                if !haskey(component_groups, found_key)
                    component_groups[found_key] = Set{VarName}()
                end
                push!(component_groups[found_key], vn)
            end
        end
        for comp in all_comps
            key = comp.key
            if !haskey(component_groups, key)
                continue
            end
            params_vns = component_groups[key]
            params_to_process = setdiff(params_vns, all_processed_vns)
            if isempty(params_to_process)
                continue
            end
            # Separate any discrete variables in this component
            comp_discrete = filter(v -> var_supports[v] == :discrete, collect(params_to_process))
            comp_continuous = filter(v -> var_supports[v] != :discrete, collect(params_to_process))
            if !isempty(comp_discrete)
                push!(sampler_assignments, Tuple(comp_discrete) => PG(n_particles))
                union!(all_processed_vns, comp_discrete)
            end
            if !isempty(comp_continuous)
                dim = length(comp_continuous)
                block_adtype = dim <= 10 ? ADTypes.AutoForwardDiff() : adtype_to_use
                block_metric = _resolve_metric_type(use_dense_metric_for_components, dim)
                comp_hyper = comp.hyper
                block_init_ϵ = _compute_block_init_epsilon(dim, key, effective_init_ϵ, min_ϵ, max_ϵ)
                block_max_depth = _compute_block_max_depth(key, max_depth, comp_hyper)
                block_target_acc = _compute_block_target_acceptance(key, target_acceptance, comp_hyper)
                block_adapt = _compute_block_adaptation_steps(key, adaptation_steps, dim,
                    comp_hyper, n_samples; use_dense_metric=use_dense_metric_for_components)
                sampler = if block_init_ϵ > 0.0
                    NUTS(block_adapt, block_target_acc; max_depth=block_max_depth,
                        init_ϵ=block_init_ϵ, metricT=block_metric, adtype=block_adtype)
                else
                    NUTS(block_adapt, block_target_acc; max_depth=block_max_depth,
                        metricT=block_metric, adtype=block_adtype)
                end
                push!(sampler_assignments, Tuple(comp_continuous) => sampler)
                union!(all_processed_vns, comp_continuous)
            end
        end
    end
    # Stage 2b: Group fixed effects and regression slopes into a dedicated block
    if group_components
        unprocessed = setdiff(vns, all_processed_vns)
        fixed_candidates = filter(
            vn -> begin
                vn_sym = _get_varname_symbol(vn)
                is_fixed = false
                if !isnothing(param_reg) && haskey(param_reg.descriptors, vn_sym)
                    d = param_reg.descriptors[vn_sym]
                    is_fixed = (d.component_key == :fixed || d.role in (:fixed_coef, :eiv_innovations))
                end
                if !is_fixed
                    vn_str = string(vn)
                    is_fixed = occursin(r"^(intercept|beta(_flat)?)(_\d+)?$", vn_str) ||
                               startswith(vn_str, "beta_") || startswith(vn_str, "intercept_")
                end
                is_fixed
            end,
            collect(unprocessed)
        )
        if !isempty(fixed_candidates)
            fixed_discrete = filter(v -> var_supports[v] == :discrete, fixed_candidates)
            fixed_continuous = filter(v -> var_supports[v] != :discrete, fixed_candidates)
            if !isempty(fixed_discrete)
                push!(sampler_assignments, Tuple(fixed_discrete) => PG(n_particles))
                union!(all_processed_vns, fixed_discrete)
            end
            if !isempty(fixed_continuous)
                dim = length(fixed_continuous)
                block_adtype = dim <= 10 ? ADTypes.AutoForwardDiff() : adtype_to_use
                block_metric = _resolve_metric_type(use_dense_metric_for_components, dim)
                block_init_ϵ = _compute_block_init_epsilon(dim, :fixed, effective_init_ϵ, min_ϵ, max_ϵ)
                block_max_depth = _compute_block_max_depth(:fixed, max_depth, nothing)
                block_target_acc = _compute_block_target_acceptance(:fixed, target_acceptance, nothing)
                block_adapt = _compute_block_adaptation_steps(:fixed, adaptation_steps, dim,
                    nothing, n_samples; use_dense_metric=use_dense_metric_for_components)
                sampler = if block_init_ϵ > 0.0
                    NUTS(block_adapt, block_target_acc; max_depth=block_max_depth,
                        init_ϵ=block_init_ϵ, metricT=block_metric, adtype=block_adtype)
                else
                    NUTS(block_adapt, block_target_acc; max_depth=block_max_depth,
                        metricT=block_metric, adtype=block_adtype)
                end
                push!(sampler_assignments, Tuple(fixed_continuous) => sampler)
                union!(all_processed_vns, fixed_continuous)
            end
        end
    end
    # Stage 3: Assign samplers to remaining parameters based on their support
    remaining_vns = setdiff(vns, all_processed_vns)
    if !isempty(remaining_vns)
        param_groups = Dict(
            :discrete => Set{VarName}(), 
            :bounded => Set{VarName}(), 
            :other_continuous => Set{VarName}()
        )
        for vn in remaining_vns
            supp = get(var_supports, vn, :continuous)
            if supp == :discrete
                push!(param_groups[:discrete], vn)
            elseif supp == :bounded
                push!(param_groups[:bounded], vn)
            else
                push!(param_groups[:other_continuous], vn)
            end
        end
        # Assign PG sampler to discrete parameters
        if !isempty(param_groups[:discrete])
            push!(sampler_assignments, Tuple(param_groups[:discrete]) => PG(n_particles))
        end
        # Assign tuned NUTS sampler to remaining continuous parameters (Turing Bijectors transforms bounded supports)
        all_remaining_cont = union(param_groups[:bounded], param_groups[:other_continuous])
        if !isempty(all_remaining_cont)
            params = Tuple(all_remaining_cont)
            dim = length(params)
            block_adtype = dim <= 10 ? ADTypes.AutoForwardDiff() : adtype_to_use
            block_metric = _resolve_metric_type(use_dense_metric_for_components, dim)
            block_init_ϵ = _compute_block_init_epsilon(dim, :other_continuous,
                effective_init_ϵ, min_ϵ, max_ϵ)
            block_max_depth = _compute_block_max_depth(:other_continuous, max_depth, nothing)
            block_target_acc = _compute_block_target_acceptance(:other_continuous,
                target_acceptance, nothing)
            block_adapt = _compute_block_adaptation_steps(:other_continuous, adaptation_steps,
                dim, nothing, n_samples; use_dense_metric=false)
            sampler = if block_init_ϵ > 0.0
                NUTS(block_adapt, block_target_acc; max_depth=block_max_depth,
                    init_ϵ=block_init_ϵ, metricT=block_metric, adtype=block_adtype)
            else
                NUTS(block_adapt, block_target_acc; max_depth=block_max_depth,
                    metricT=block_metric, adtype=block_adtype)
            end
            push!(sampler_assignments, params => sampler)
        end
    end
    # Stage 4: Construct and return the final composite sampler
    if isempty(sampler_assignments)
        @warn "Could not identify any parameters to sample. Defaulting to a single NUTS sampler."
        total_init_ϵ = _compute_block_init_epsilon(num_params, :global, effective_init_ϵ,
            min_ϵ, max_ϵ)
        total_max_depth = _compute_block_max_depth(:global, max_depth, nothing)
        total_target_acc = _compute_block_target_acceptance(:global, target_acceptance, nothing)
        total_adapt = _compute_block_adaptation_steps(:global, adaptation_steps, num_params,
            nothing, n_samples; use_dense_metric=false)
        total_metric = _resolve_metric_type(use_dense_metric_for_components, num_params)
        return if total_init_ϵ > 0.0
            NUTS(total_adapt, total_target_acc; max_depth=total_max_depth, init_ϵ=total_init_ϵ,
                metricT=total_metric, adtype=adtype_to_use)
        else
            NUTS(total_adapt, total_target_acc; max_depth=total_max_depth, metricT=total_metric,
                adtype=adtype_to_use)
        end
    elseif length(sampler_assignments) == 1
        return sampler_assignments[1][2]
    else
        return Gibbs(sampler_assignments...)
    end
end
"""
    extract_param_matrix(chain, var_id; expected_dim=nothing)::Matrix{Float64}
Canonical, zero-fail parameter extraction engine for TethysBayes.
Extracts samples for `var_id` (Symbol, String, or DynamicPPL.VarName) across all chain backends
(`VNChain`, `FlexiChain`, `MCMCChains.Chains`, `DataFrame`, `Dict`, `NamedTuple`) and returns a
standardized 2D `Matrix{Float64}` of size `(n_samples, dim)`.
"""
function extract_param_matrix(chain, var_id::Union{Symbol, AbstractString, DynamicPPL.VarName};
    expected_dim::Union{Int, Nothing}=nothing)::Matrix{Float64}
    # 1. Normalize identifier to Symbol and clean base string
    raw_str = string(var_id)
    clean_str = replace(raw_str, r"^Parameter\((.*)\)$" => s"\1", r"^parameters\." => "",
        r"^:+" => "")
    base_name = first(Base.split(clean_str, '['))
    base_sym = Symbol(base_name)
    clean_sym = Symbol(clean_str)
    n_chains_val = 1
    try
        if occursin("FlexiChain", string(typeof(chain))) || occursin("VNChain",
            string(typeof(chain)))
            n_chains_val = FlexiChains.nchains(chain)
        elseif hasproperty(chain, :chains)
            n_chains_val = length(chain.chains)
        elseif hasproperty(chain, :value) && ndims(chain.value) == 3
            n_chains_val = size(chain.value, 3)
        elseif hasproperty(chain, :info) && haskey(chain.info, :n_chains)
            n_chains_val = Int(chain.info.n_chains)
        end
    catch
    end
    # 2. Extract column/variable data robustly across chain formats
    col_data = try
        chain[base_sym]
    catch
        try
            chain[clean_sym]
        catch
            try
                chain[DynamicPPL.VarName(base_sym)]
            catch
                try
                    chain[DynamicPPL.VarName(clean_sym)]
                catch
                    try
                        chain[clean_str]
                    catch
                        try
                            chain[raw_str]
                        catch
                            try
                                chain[Symbol(raw_str)]
                            catch
                                try
                                    df = DataFrame(chain)
                                    df_cols = names(df)
                                    # Strategy A: Exact or cleaned column match
                                    idx = findfirst(df_cols) do n
                                        n_str = string(n)
                                        n_clean = replace(n_str,
                                            r"^Parameter\((.*)\)$" => s"\1",
                                            r"^parameters\." => "", r"^:+" => "")
                                        n_clean == base_name || n_clean == clean_str||
                                            n_str == raw_str || n_str == base_name
                                    end
                                    if !isnothing(idx)
                                        df[!, df_cols[idx]]
                                    else
                                        # Strategy B: Bracketed indices (e.g. 1D var[1], or 2D/tensor var[1, 1], ...)
                                        re_bracket = Regex("^" * escape_string(base_name) * "\\[([\\d,\\s]+)\\]")
                                        matched_cols = filter(n -> occursin(re_bracket, string(n)), df_cols)
                                        if !isempty(matched_cols)
                                            sort!(matched_cols, by = n -> begin
                                                m = match(re_bracket, string(n))
                                                if isnothing(m)
                                                    return (0,)
                                                end
                                                idx_strs = Base.split(m.captures[1], ',')
                                                Tuple(parse(Int, strip(s)) for s in idx_strs)
                                            end)
                                            Matrix(df[!, matched_cols])
                                        else
                                            error("Parameter '$var_id' not found in chain.")
                                        end
                                    end
                                catch err
                                    error("Parameter '$var_id' not found in the MCMC chain: $err")
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    param_data_array = Array(col_data)
    # A `Cholesky` parameter (e.g. `correlation_cholesky`, which every multivariate model
    # carries) is stored as an object per draw and cannot be broadcast to `Float64`, so the
    # generic numeric paths below all fail on it. Flatten each factor to its entries, giving
    # (n_samples, p^2), which is what callers requesting `get_params_matrix(chain, name, K^2)`
    # expect -- they `reshape` the row straight back to a K x K factor.
    if !isempty(param_data_array) && param_data_array[1] isa LinearAlgebra.Cholesky
        n_chol = length(param_data_array)
        p = size(param_data_array[1].L, 1)
        chol_mat = Matrix{Float64}(undef, n_chol, p * p)
        for i in 1:n_chol
            chol_mat[i, :] = vec(Matrix(param_data_array[i].L))
        end
        param_data_array = chol_mat
    end
    # 3. Standardize dimensions to Matrix{Float64} of shape (total_samples, dim)
    
    # Case A: Elements are AbstractArrays (e.g. Vector{Vector{Float64}} from
    #   VNChain/FlexiChain vector params)
    if eltype(param_data_array) <: AbstractArray
        all_elements = vec(param_data_array)
        total_samples = length(all_elements)
        if isempty(all_elements)
            return zeros(Float64, 0, 1)
        end
        first_elem = first(all_elements)
        param_dim = length(first_elem)
        mat = Matrix{Float64}(undef, total_samples, param_dim)
        for i in 1:total_samples
            mat[i, :] = vec(Float64.(all_elements[i]))
        end
        if !isnothing(expected_dim) && param_dim != expected_dim&&
            (total_samples * param_dim) % expected_dim == 0
            return reshape(mat, :, expected_dim)
        end
        return mat
    end
    # Case B: 1D vector of numbers (e.g. from DataFrame column or single chain scalar)
    if ndims(param_data_array) == 1
        n_elem = length(param_data_array)
        if !isnothing(expected_dim) && expected_dim > 1 && n_elem % expected_dim == 0
            return reshape(Float64.(param_data_array), :, expected_dim)
        end
        return reshape(Float64.(param_data_array), :, 1)
    end
    # Case C: 2D array of numbers
    if ndims(param_data_array) == 2
        n_rows, n_cols = size(param_data_array)
        # If columns match the number of chains (and > 1) and expected_dim is 1 or nothing:
        # col_data is [n_iterations, n_chains] representing scalar samples across chains!
        # Collapse all chains into total_samples = n_rows * n_cols
        if n_chains_val > 1 && n_cols == n_chains_val && (isnothing(expected_dim)||
            expected_dim == 1)
            return reshape(Float64.(vec(param_data_array)), n_rows * n_cols, 1)
        elseif !isnothing(expected_dim) && n_cols == expected_dim
            return Float64.(param_data_array)
        elseif !isnothing(expected_dim) && n_rows == expected_dim && n_cols != expected_dim
            # Transposed (dim, total_samples) -> transpose to (total_samples, dim)
            return permutedims(Float64.(param_data_array), (2, 1))
        elseif n_cols == 1
            return Float64.(param_data_array)
        elseif !isnothing(expected_dim) && (n_rows * n_cols) % expected_dim == 0
            return reshape(Float64.(vec(param_data_array)), :, expected_dim)
        else
            return Float64.(param_data_array)
        end
    end
    # Case D: 3D+ array from FlexiChain/MCMCChains: [iterations, dim1, ..., chains]
    num_iterations = size(param_data_array, 1)
    num_chains = size(param_data_array, ndims(param_data_array))
    total_samples = num_iterations * num_chains
    perm_dims = (1, ndims(param_data_array), 2:(ndims(param_data_array)-1)...)
    permuted_data = permutedims(param_data_array, perm_dims)
    param_dims = size(permuted_data)[3:end]
    if isempty(param_dims) # Scalar parameter
        return reshape(Float64.(permuted_data), total_samples, 1)
    else
        dim_prod = prod(param_dims)
        mat_reshaped = reshape(Float64.(permuted_data), total_samples, dim_prod)
        if !isnothing(expected_dim) && dim_prod != expected_dim&&
            (total_samples * dim_prod) % expected_dim == 0
            return reshape(mat_reshaped, :, expected_dim)
        end
        return mat_reshaped
    end
end
"""
    extract_param_vector(chain, var_id)::Vector{Float64}
Extracts a scalar parameter as a 1D `Vector{Float64}` of length `n_samples`.
"""
function extract_param_vector(chain, var_id::Union{Symbol, AbstractString,
    DynamicPPL.VarName})::Vector{Float64}
    mat = extract_param_matrix(chain, var_id; expected_dim=1)
    return vec(mat[:, 1])
end
# get_params_vector is for scalar or fixed-length parameters (expected_len >= 1)
function get_params_vector(chain, param_name::Union{Symbol, AbstractString, DynamicPPL.VarName},
    expected_len::Int=1)
    samples = extract_param_matrix(chain, param_name; expected_dim=expected_len)
    if size(samples, 2) != expected_len
        if prod(size(samples)) == size(samples, 1) * expected_len
            return reshape(samples, size(samples, 1), expected_len)
        elseif size(samples, 1) % expected_len == 0 && size(samples, 2) == 1
            return reshape(samples, :, expected_len)
        else
            error("Parameter '$param_name' has dimension $(size(samples, 2)), but expected $expected_len.")
        end
    end
    return samples # (total_samples, expected_len)
end
# get_params_matrix is for vector/matrix parameters (expected_len >= 1)
function get_params_matrix(chain, param_name::Union{Symbol, AbstractString, DynamicPPL.VarName},
    expected_len::Int)
    return get_params_vector(chain, param_name, expected_len)
end
"""
    convert_to_chains(res, model=nothing, n_samples::Int=100; kwargs...)
Converts optimization results (`maximum_likelihood`, `maximum_a_posteriori`) or
variational inference results (`vi`) from Turing into a sample dictionary format
directly consumable by TethysBayes reconstruction, diagnostics, and plotting routines.
# Mathematical Formulation
For maximum likelihood (ML) or maximum a posteriori (MAP) estimation, the optimizer
yields a mode point estimate \$\\hat{\\theta}\$:
\$\$\\hat{\\theta} = \\arg\\max_\\theta p(\\theta \\mid y)\$\$
Downstream analytical post-processing evaluates expectations and empirical summaries
across draws \$s = 1, \\dots, S\$ (where \$S = n\\_samples\$). Mode estimates are
expanded across synthetic replicate draws:
\$\$\\theta^{(s)} = \\hat{\\theta}, \\quad \\forall s \\in \\{1, \\dots, S\\}\$\$
For variational inference (VI) with approximating distribution \$q_\\phi(\\theta)\$,
Monte Carlo draws are generated via the approximate posterior:
\$\$\\theta^{(s)} \\sim q_\\phi(\\theta), \\quad s = 1, \\dots, S\$\$
# Arguments
- `res`: Optimization result (`Turing.Inference.ModeResult` from `maximum_likelihood`
  or `maximum_a_posteriori`), variational result (`VIResult` / `VariationalPosterior`
  from `vi`), or an existing chain object.
- `model`: Optional `DynamicPPL.Model`.
- `n_samples::Int`: Number of samples or replicates to generate (default: 100).
- `kwargs...`: Additional keyword arguments passed through.
# Returns
- A `Dict{Symbol, Any}` mapping parameter symbols to sample collections of length
  `n_samples` (or the original `res` if it is already a recognized MCMC chain).
"""
function convert_to_chains(res, model=nothing, n_samples::Int=100; kwargs...)
    # 0. Pass through if already a chain or table
    if res isa DataFrame || res isa AbstractDict ||
       occursin("FlexiChain", string(typeof(res))) ||
       occursin("VNChain", string(typeof(res))) ||
       occursin("Chains", string(typeof(res)))
        return res
    end
    n_samples = max(1, n_samples)
    d = Dict{Symbol, Any}()
    # 1. ModeResult (ML / MAP)
    if (hasproperty(res, :params) && hasproperty(res, :optim_result)) ||
       occursin("ModeResult", string(typeof(res)))
        params = hasproperty(res, :params) ? res.params : res
        for vn in keys(params)
            sym = _get_varname_symbol(vn)
            val = params[vn]
            if val isa Number
                s_val = Float64(val)
                vec_s = fill(s_val, n_samples)
                d[sym] = vec_s
                d[Symbol("$(sym)_1")] = vec_s
            elseif val isa AbstractArray
                arr_val = Float64.(copy(val))
                d[sym] = [copy(arr_val) for _ in 1:n_samples]
                for j in 1:length(arr_val)
                    d[Symbol("$(sym)_$j")] = fill(arr_val[j], n_samples)
                end
            else
                d[sym] = fill(val, n_samples)
            end
        end
        return d
    # 2. Variational Inference (VIResult or VariationalPosterior)
    elseif hasproperty(res, :q) ||
           occursin("VIResult", string(typeof(res))) ||
           occursin("VariationalPosterior", string(typeof(res)))
        samples = rand(res, n_samples)
        if !isempty(samples)
            first_s = first(samples)
            for vn in keys(first_s)
                sym = _get_varname_symbol(vn)
                first_val = first_s[vn]
                if first_val isa Number
                    vec_vals = [Float64(s[vn]) for s in samples]
                    d[sym] = vec_vals
                    d[Symbol("$(sym)_1")] = vec_vals
                elseif first_val isa AbstractArray
                    len = length(first_val)
                    arr_samples = [Float64.(s[vn]) for s in samples]
                    d[sym] = arr_samples
                    for j in 1:len
                        d[Symbol("$(sym)_$j")] = [Float64(s[vn][j]) for s in samples]
                    end
                else
                    d[sym] = [s[vn] for s in samples]
                end
            end
        end
        return d
    end
    return res
end
"""
    TethysBayes_sample(model::DynamicPPL.Model, n_samples::Int; kwargs...)
A convenience wrapper for `TethysBayes_sample` that automatically selects an optimal sampler
if one is not provided.
# Arguments
- `model`: The Turing model object.
- `n_samples`: The number of samples to draw per chain.
- `kwargs...`: Additional keyword arguments passed to `get_optimal_sampler` and `Turing.sample`.
# Returns
- The MCMC chain object returned by `Turing.sample`.
"""
function TethysBayes_sample(model::DynamicPPL.Model, n_samples::Int; kwargs...)
    # Separate kwargs for get_optimal_sampler and the main sample call.
    sampler_kwargs = Dict{Symbol, Any}()
    sample_kwargs = Dict{Symbol, Any}()
    # List of keywords known to get_optimal_sampler
    known_sampler_keys = [
        :sampler_choice, :sampler_map, :adtype, :group_components,
        :use_dense_metric_for_components, :adaptation_steps, :target_acceptance,
        :n_particles, :n_chains, :init_ϵ, :init_epsilon, :step_size,
        :max_depth, :min_ϵ, :max_ϵ
    ]
    for (key, value) in kwargs
        if key in known_sampler_keys
            sampler_kwargs[key] = value
        else
            sample_kwargs[key] = value
        end
    end
    n_chains = get(kwargs, :n_chains, 1)
    # Pass n_samples to get_optimal_sampler to enable principled adaptation scaling
    if !haskey(sampler_kwargs, :n_samples)
        sampler_kwargs[:n_samples] = n_samples
    end
    @info "Sampler not provided. Automatically selecting an optimal sampler with principled warmup scaling..."
    sampler = get_optimal_sampler(model; sampler_kwargs...)
    return TethysBayes_sample(model, sampler, n_samples; n_chains=n_chains, sample_kwargs...)
end

 

"""
    TethysBayes_sample(model, sampler, n_samples; n_chains=1, kwargs...)
A wrapper around `Turing.sample` that defaults to the `MCMCThreads` backend for
both single and multi-chain sampling, ensuring consistent output dimensionality.
# Arguments
- `model`: The Turing model object.
- `sampler`: The MCMC sampler to use.
- `n_samples`: The number of samples to draw per chain.
- `n_chains::Int`: The number of MCMC chains to run. Default: `1`.
- `kwargs...`: Additional keyword arguments passed directly to `Turing.sample`.
# Returns
- A 3-dimensional MCMC chain object.
"""
function TethysBayes_sample(model, sampler, n_samples; n_chains::Int=1, kwargs...)
    local chain
    @info "Running $(n_chains) chain(s) using MCMCThreads() backend."
    chain = Base.invokelatest(sample, model, sampler, MCMCThreads(), n_samples, n_chains; kwargs...)
    return chain
end
# Forward single-chain syntax: sample(model, sampler, N; kwargs...) 
# directly to the multi-chain MCMCThreads() signature.
function AbstractMCMC.sample(
    model::AbstractMCMC.AbstractModel,
    sampler::AbstractMCMC.AbstractSampler,
    N::Integer;
    kwargs...
)
    # Default to 1 chain run via MCMCThreads() if n_chains isn't specified
    return AbstractMCMC.sample(model, sampler, MCMCThreads(), N, 1; kwargs...)
end
# Forward multi-chain syntax without ensemble strategy: sample(model, sampler, N, n_chains;
#   kwargs...)
function AbstractMCMC.sample(
    model::AbstractMCMC.AbstractModel,
    sampler::AbstractMCMC.AbstractSampler,
    N::Integer,
    n_chains::Integer;
    kwargs...
)
    return AbstractMCMC.sample(model, sampler, MCMCThreads(), N, n_chains; kwargs...)
end
"""
    _generate_likelihood_section(M::NamedTuple, is_multivariate::Bool)
Generates Turing code for all likelihood-specific priors.
# Arguments
- `M`: The main model configuration `NamedTuple`.
- `is_multivariate`: A boolean indicating if the model is multivariate.
# Returns
- A `String` containing the generated Turing code for the likelihood priors.
"""
function _generate_likelihood_section(
    M::NamedTuple, is_multivariate::Bool; prefix::String = ""
)
    families = [string(get(spec, :family, "gaussian")) for spec in M.likelihood_specs]
    prior_blocks = String[]
    r_nb_sym = !isempty(prefix) ? "r_nb_$(prefix)" : "r_nb"
    phi_hurdle_sym = !isempty(prefix) ? "lik_phi_hurdle_$(prefix)" : "lik_phi_hurdle"
    phi_zi_sym = !isempty(prefix) ? "lik_phi_zi_$(prefix)" : "lik_phi_zi"
    nu_sym = !isempty(prefix) ? "lik_nu_student_t_$(prefix)" : "lik_nu_student_t"
    y_sigma_sym = !isempty(prefix) ? "y_sigma_$(prefix)" : "y_sigma"
    extra_sym = !isempty(prefix) ? "lik_extra_params_$(prefix)" : "lik_extra_params"
    L_corr_sym = !isempty(prefix) ? "L_corr_$(prefix)" : "correlation_cholesky"
    dirichlet_phi_sym = !isempty(prefix) ? "dirichlet_phi_$(prefix)" : "dirichlet_phi"
    # Prior for Negative Binomial dispersion
    if any(f -> f == "negbin", families)
        push!(prior_blocks, "$(r_nb_sym) ~ Exponential(1.0)")
    end
    # Prior for Zero-Inflation or Hurdle probability
    if get(M, :user_provided_hurdle, false)
        push!(prior_blocks, "$(phi_hurdle_sym) ~ Beta(1,1)")
    elseif get(M, :use_zi, false)
        push!(prior_blocks, "$(phi_zi_sym) ~ Beta(1,1)")
    end
    # Prior for Student's T degrees of freedom
    if any(f -> f == "student_t", families)
        push!(prior_blocks,
            "$(nu_sym) ~ Exponential(1.0)")
    end
    # Prior for observation standard deviation (for Gaussian-like families)
    # "mvnormal"/"multivariate_normal" are included: the joint multivariate Gaussian path
    # uses `y_sigma` as the diagonal of D, so omitting it left `y_sigma` undeclared for a
    # model that explicitly asked for `family=mvnormal`.
    if any(f -> f in ["gaussian", "lognormal", "student_t", "laplace", "half_normal",
        "half_student_t", "mvnormal", "multivariate_normal"], families)
        y_sigma_prior_str = _distribution_to_string(Exponential(1.0))
        if is_multivariate
            push!(prior_blocks,
                "$(y_sigma_sym) ~ filldist($(y_sigma_prior_str), K)")
        else
            push!(prior_blocks, "$(y_sigma_sym) ~ $(y_sigma_prior_str)")
        end
    end
    # Prior for extra parameters (e.g., Gamma shape, Beta precision)
    if any(f -> f in ["gamma", "beta", "inverse_gaussian", "pareto", "half_student_t"], families)
        push!(prior_blocks,
            "$(extra_sym) ~ Exponential(1.0)")
    end
    # Prior for multivariate correlation matrix (skipped for multinomial models)
    if is_multivariate && !get(M, :is_multinomial, false)
        # NOTE: `eta = 1` makes the LKJ density uniform on the correlation (for K=2 it is
        # exactly 1 on (-1, 1)), so it places no penalty as rho -> +-1 and the posterior
        # is poorly conditioned -- HMC still reports 36-39 divergent transitions. `eta = 2`
        # (Stan's default) removes the boundary mass and is worth considering, but it is a
        # prior change for every multivariate model and was NOT what fixed the crash: see
        # the non-finite-covariance fallback in `get_dist_ref` (`MvNormalFamily`).
        push!(prior_blocks, "$(L_corr_sym) ~ LKJCholesky(K, 1.0)")
    end
    # Prior for Dirichlet dispersion / precision parameter
    if get(M, :is_multinomial, false) && string(get(M, :multinomial_family, "")) in ["dirichlet_multinomial", "dirichlet"]
        push!(prior_blocks, "$(dirichlet_phi_sym) ~ Exponential(1.0)")
    end
    # --- Priors for Ordinal Model ---
    ordinal_spec_idx = findfirst(s -> string(get(s, :family, "")) == "ordinal", M.likelihood_specs)
    if !isnothing(ordinal_spec_idx)
        spec = M.likelihood_specs[ordinal_spec_idx]
        K = get(spec, :K, 0)
        latent_dist = get(spec, :latent_dist, :logistic)
        alpha_1_sym = !isempty(prefix) ? "ordinal_alpha_unscaled_1_$(prefix)" : "ordinal_alpha_unscaled_1"
        diffs_sym = !isempty(prefix) ? "ordinal_alpha_diffs_$(prefix)" : "ordinal_alpha_diffs"
        df_sym = !isempty(prefix) ? "ordinal_df_$(prefix)" : "ordinal_df"
        if K > 2
            # Prior for the first cut-point and the positive differences for subsequent cut-points.
            push!(prior_blocks, "$(alpha_1_sym) ~ Normal(0, 5)")
            push!(prior_blocks, "$(diffs_sym) ~ filldist(Exponential(1.0), $(K - 2))")
        elseif K == 2
            # For a binary ordinal model, only one cut-point is needed.
            push!(prior_blocks, "$(alpha_1_sym) ~ Normal(0, 5)")
        end
        # Prior for the degrees of freedom if using a Student's T latent distribution.
        if latent_dist == :student_t
            push!(prior_blocks, "$(df_sym) ~ Exponential(1.0)")
        end
    end
    return join(prior_blocks, "\n    ")
end
function _generate_univariate_likelihood_block(M::NamedTuple; prefix::String = "")
    family = string(M.likelihood_specs[1][:family])
    family_symbol = QuoteNode(Symbol(family))
    y_sigma_name = !isempty(prefix) ? "y_sigma_$(prefix)" : "y_sigma"
    r_nb_name = !isempty(prefix) ? "r_nb_$(prefix)" : "r_nb"
    phi_zi_name = !isempty(prefix) ? "lik_phi_zi_$(prefix)" : "lik_phi_zi"
    phi_hurdle_name = !isempty(prefix) ? "lik_phi_hurdle_$(prefix)" : "lik_phi_hurdle"
    nu_name = !isempty(prefix) ? "lik_nu_student_t_$(prefix)" : "lik_nu_student_t"
    extra_name = !isempty(prefix) ? "lik_extra_params_$(prefix)" : "lik_extra_params"
    # Determine which kwargs are needed based on the family
    needs_sigma = family in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"]
    needs_rnb = family == "negbin"
    needs_nu = family == "student_t"
    needs_extra = family in ["gamma", "beta", "inverse_gaussian", "pareto", "half_student_t"]
    kwargs_parts = String[]
    extra_param_logic = ""
    if needs_sigma
        push!(kwargs_parts, "sigma_y=$(y_sigma_name)")
    end
    if needs_rnb
        push!(kwargs_parts, "r_nb=$(r_nb_name)")
    end
    if get(M, :use_zi, false)
        push!(kwargs_parts, "phi_zi=$(phi_zi_name)")
    end
    if get(M, :user_provided_hurdle, false)
        push!(kwargs_parts, "phi_hurdle=$(phi_hurdle_name)")
    end
    if needs_nu || needs_extra
        extra_param_logic = if needs_nu; "extra_p = $(nu_name)"
        else; "extra_p = $(extra_name)"; end
        push!(kwargs_parts, "extra_params=extra_p")
    end
    if get(M, :user_provided_trials, false)
        push!(kwargs_parts, "trial=M.trials[:, 1]")
    end
    if get(M, :user_provided_weights, false)
        push!(kwargs_parts, "weight=M.weights[:, 1]")
    end
    if get(M, :user_provided_censor_lower, false)
        push!(kwargs_parts, "censor_lower=M.censor_lower[:, 1]")
    end
    if get(M, :user_provided_censor_upper, false)
        push!(kwargs_parts, "censor_upper=M.censor_upper[:, 1]")
    end
    if get(M, :user_provided_hurdle, false)
        push!(kwargs_parts, "hurdle=M.hurdle[:, 1]")
    end
    has_obs_vec_kw = get(M, :user_provided_trials, false) ||
                     get(M, :user_provided_weights, false) ||
                     get(M, :user_provided_censor_lower, false) ||
                     get(M, :user_provided_censor_upper, false) ||
                     get(M, :user_provided_hurdle, false)
    block_content = if has_obs_vec_kw
        kwargs_parts_i = String[]
        for p in kwargs_parts
            if startswith(p, "trial=")
                push!(kwargs_parts_i, "trial=M.trials[i, 1]")
            elseif startswith(p, "weight=")
                push!(kwargs_parts_i, "weight=M.weights[i, 1]")
            elseif startswith(p, "censor_lower=")
                push!(kwargs_parts_i, "censor_lower=M.censor_lower[i, 1]")
            elseif startswith(p, "censor_upper=")
                push!(kwargs_parts_i, "censor_upper=M.censor_upper[i, 1]")
            elseif startswith(p, "hurdle=")
                push!(kwargs_parts_i, "hurdle=M.hurdle[i, 1]")
            else
                push!(kwargs_parts_i, p)
            end
        end
        kwargs_i_str = join(kwargs_parts_i, ", ")
        """
        $(extra_param_logic)
        log_lik_sum = sum(i -> Distributions.logpdf(TethysBayes_Likelihood($(family_symbol), eta[i]; $(kwargs_i_str)), M.y_obs[i]), 1:length(eta))
        """
    else
        kwargs_str = join(kwargs_parts, ", ")
        """
        $(extra_param_logic)
        d_lik_vec = TethysBayes_Likelihood.($(family_symbol), eta; $(kwargs_str))
        log_lik_sum = sum(Distributions.logpdf.(d_lik_vec, M.y_obs))
        """
    end
    return """
    let
        local log_lik_sum
        $(block_content)
        Turing.@addlogprob! log_lik_sum
    end
    """
end
function _generate_multivariate_likelihood_block(M::NamedTuple; prefix::String = "")
    y_sigma_name = !isempty(prefix) ? "y_sigma_$(prefix)" : "y_sigma"
    r_nb_name = !isempty(prefix) ? "r_nb_$(prefix)" : "r_nb"
    phi_zi_name = !isempty(prefix) ? "lik_phi_zi_$(prefix)" : "lik_phi_zi"
    phi_hurdle_name = !isempty(prefix) ? "lik_phi_hurdle_$(prefix)" : "lik_phi_hurdle"
    nu_name = !isempty(prefix) ? "lik_nu_student_t_$(prefix)" : "lik_nu_student_t"
    extra_name = !isempty(prefix) ? "lik_extra_params_$(prefix)" : "lik_extra_params"
    L_corr_name = !isempty(prefix) ? "L_corr_$(prefix)" : "correlation_cholesky"
    dirichlet_phi_name = !isempty(prefix) ? "dirichlet_phi_$(prefix)" : "dirichlet_phi"
    K = M.outcomes_N

    # --- Joint Gaussian --------------------------------------------------------
    # A multivariate Gaussian is y_i ~ MVN(eta_i, D R D): the correlation belongs in the
    # RESIDUAL COVARIANCE, not applied to the linear predictor. The marginal path below
    # emitted `eta_latent * L` and then a sum of per-outcome marginals, which means the
    # correlation never entered the density at all and `correlation_cholesky` was
    # unidentifiable whenever eta was constant across observations.
    #
    # `MvNormalFamily` already implements this: get_dist_ref builds
    # `Diagonal(s) * (L * L') * Diagonal(s)` from `extra_params[:correlation_cholesky]`.
    # It needs one likelihood per OBSERVATION holding the K-vector of etas, which is why
    # this is a separate branch rather than a tweak of the outcome loop below.
    gaus_families = ("gaussian", "mvnormal", "multivariate_normal")
    nspec = length(M.likelihood_specs)
    # Guard on `is_multinomial` and on the spec actually existing: a multinomial /
    # Dirichlet model has several outcomes but a SINGLE likelihood spec (the whole
    # composition is one draw), so iterating `outcomes_N` over `likelihood_specs` would
    # index past the end.
    joint_ok = !get(M, :is_multinomial, false) && K > 1 && nspec >= K &&
               all(string(get(M.likelihood_specs[k], :family, "")) in gaus_families
                   for k in 1:K)
    if joint_ok
        return """
        let
            local log_lik_sum
            d_lik_vec = [TethysBayes_Likelihood(:"mvnormal", collect(view(eta_latent, i, :));
                sigma_y = $(y_sigma_name),
                extra_params = Dict(:correlation_cholesky => $(L_corr_name)))
                for i in 1:N]
            log_lik_sum = sum(Distributions.logpdf.(d_lik_vec, eachrow(M.y_obs)))
            Turing.@addlogprob! log_lik_sum
        end
        """
    end

    if get(M, :is_multinomial, false)
        fam = Symbol(get(M, :multinomial_family, :multinomial))
        trials_code = get(M, :user_provided_trials, false) ? "M.trials[:, 1]" : "vec(sum(M.y_obs, dims=2))"
        if fam == :multinomial
            return """
            let
                local log_lik_sum
                eta_full = hcat(fill(zero(T), N, 1), eta_latent)
                probs_mat = LogExpFunctions.softmax(eta_full; dims=2)
                trials_vec = $(trials_code)
                log_lik_sum = sum(i -> Distributions.logpdf(Distributions.Multinomial(Int(trials_vec[i]), collect(probs_mat[i, :])), M.y_obs[i, :]), 1:N)
                Turing.@addlogprob! log_lik_sum
            end
            """
        elseif fam == :categorical
            return """
            let
                local log_lik_sum
                eta_full = hcat(fill(zero(T), N, 1), eta_latent)
                probs_mat = LogExpFunctions.softmax(eta_full; dims=2)
                log_lik_sum = sum(i -> Distributions.logpdf(Distributions.Categorical(collect(probs_mat[i, :])), Int(M.y_obs[i])), 1:N)
                Turing.@addlogprob! log_lik_sum
            end
            """
        elseif fam == :dirichlet_multinomial
            return """
            let
                local log_lik_sum
                eta_full = hcat(fill(zero(T), N, 1), eta_latent)
                probs_mat = LogExpFunctions.softmax(eta_full; dims=2)
                trials_vec = $(trials_code)
                log_lik_sum = sum(i -> Distributions.logpdf(Distributions.DirichletMultinomial(Int(trials_vec[i]), max.($(dirichlet_phi_name) .* collect(probs_mat[i, :]), 1e-6)), M.y_obs[i, :]), 1:N)
                Turing.@addlogprob! log_lik_sum
            end
            """
        elseif fam == :dirichlet
            return """
            let
                local log_lik_sum
                eta_full = hcat(fill(zero(T), N, 1), eta_latent)
                probs_mat = LogExpFunctions.softmax(eta_full; dims=2)
                log_lik_sum = sum(i -> Distributions.logpdf(Distributions.Dirichlet(max.($(dirichlet_phi_name) .* collect(probs_mat[i, :]), 1e-6)), M.y_obs[i, :]), 1:N)
                Turing.@addlogprob! log_lik_sum
            end
            """
        end
    end
    loop_body_parts = String[]
    for k in 1:M.outcomes_N
        family = string(M.likelihood_specs[k][:family])
        family_symbol = QuoteNode(Symbol(family))
        needs_sigma = family in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"]
        needs_rnb = family == "negbin"
        needs_nu = family == "student_t"
        needs_extra = family in ["gamma", "beta", "inverse_gaussian", "pareto", "half_student_t"]
        kwargs_parts = String[]
        extra_param_logic = ""
        if needs_sigma
            push!(kwargs_parts, "sigma_y=$(y_sigma_name)[$k]")
        end
        if needs_rnb
            push!(kwargs_parts, "r_nb=$(r_nb_name)")
        end
        if get(M, :use_zi, false)
            push!(kwargs_parts, "phi_zi=$(phi_zi_name)")
        end
        if get(M, :user_provided_hurdle, false)
            push!(kwargs_parts, "phi_hurdle=$(phi_hurdle_name)")
        end
        if needs_nu || needs_extra
            extra_param_logic = if needs_nu; "extra_p = $(nu_name)"
            else; "extra_p = $(extra_name)"; end
            push!(kwargs_parts, "extra_params=extra_p")
        end
        if get(M, :user_provided_trials, false)
            push!(kwargs_parts, "trial=M.trials[:, $k]")
        end
        if get(M, :user_provided_weights, false)
            push!(kwargs_parts, "weight=M.weights[:, $k]")
        end
        if get(M, :user_provided_censor_lower, false)
            push!(kwargs_parts, "censor_lower=M.censor_lower[:, $k]")
        end
        if get(M, :user_provided_censor_upper, false)
            push!(kwargs_parts, "censor_upper=M.censor_upper[:, $k]")
        end
        if get(M, :user_provided_hurdle, false)
            push!(kwargs_parts, "hurdle=M.hurdle[:, $k]")
        end
        kwargs_str = join(kwargs_parts, ", ")
        block_content = """
            $(extra_param_logic)
            d_lik_vec_k = TethysBayes_Likelihood.($(family_symbol), view(eta_correlated, :, $k);
              $(kwargs_str))
            Turing.@addlogprob! sum(Distributions.logpdf.(d_lik_vec_k, view(M.y_obs, :, $k)))
        """
        outcome_block = """
        # Likelihood for outcome $(k)
        let
            $(block_content)
    end
    """
        push!(loop_body_parts, outcome_block)
    end
    loop_body = join(loop_body_parts, "\n\n")
    return """
    eta_correlated = eta_latent * $(L_corr_name).L
    $(loop_body)
    """
end
"""
    _generate_ordinal_likelihood_block(M::NamedTuple; prefix::String = "")
Generates the Turing code block for an ordinal regression likelihood. This version
is CPU-only.
"""
function _generate_ordinal_likelihood_block(M::NamedTuple; prefix::String = "")
    spec = M.likelihood_specs[1]
    K = get(spec, :K, 0)
    if K < 2
        return ""
    end
    latent_dist_val = get(spec, :latent_dist, :logistic)
    non_prop_terms = get(M, :non_proportional_effects, Symbol[])
    is_npo = !isempty(non_prop_terms)
    npo_indices = findall(x -> x in non_prop_terms, M.Xfixed_names)
    n_npo_vars = length(npo_indices)
    alpha_1_sym = !isempty(prefix) ? "ordinal_alpha_unscaled_1_$(prefix)" : "ordinal_alpha_unscaled_1"
    diffs_sym = !isempty(prefix) ? "ordinal_alpha_diffs_$(prefix)" : "ordinal_alpha_diffs"
    df_sym = !isempty(prefix) ? "ordinal_df_$(prefix)" : "ordinal_df"
    beta_npo_name = !isempty(prefix) ? "beta_npo_$(prefix)" : "beta_npo"
    npo_update_block = ""
    if is_npo && n_npo_vars > 0
        npo_update_block = """
        # Non-proportional effects calculation
        X_npo = M.Xfixed[:, $(npo_indices)]
        beta_npo_matrix = reshape($(beta_npo_name), $(n_npo_vars), $(K-1))
        eta_npo = X_npo * beta_npo_matrix
        """
    end
    return """
    # Ordinal Likelihood Block
    let
        # Reconstruct the ordered cut-points from their unscaled parameters.
        alphas_computed = if $(K > 2)
            cumsum([$(alpha_1_sym); $(diffs_sym)])
        else
            [$(alpha_1_sym)]
        end
        latent_dist_symbol = :$(latent_dist_val)
        $(npo_update_block)
        # Proportional effect for all observations
        eta_prop = eta
        # Calculate cumulative probabilities for all observations in a vectorized manner
        eta_matrix = if $(is_npo && n_npo_vars > 0)
            eta_prop .+ eta_npo
        else
            # Broadcast the proportional effect across all cut-points
            eta_prop .* ones(T, 1, $(K-1))
        end
        # linear_predictor_vec is now a matrix of size [N_obs, K-1]
        linear_predictor_matrix = alphas_computed' .- eta_matrix
        cumulative_probs_matrix = if latent_dist_symbol == :normal
            Distributions.cdf.(Normal(), linear_predictor_matrix)
        elseif latent_dist_symbol == :logistic
            LogExpFunctions.logistic.(linear_predictor_matrix)
        elseif latent_dist_symbol == :student_t
            Distributions.cdf.(TDist($(df_sym)), linear_predictor_matrix)
        else
            error("Unsupported latent distribution ':\$(latent_dist_symbol)' for ordinal model.")
        end
        # Calculate probabilities for each category for all observations
        probs_matrix = Array{T}(undef, M.y_N, $(K))
        if $(K > 1)
            probs_matrix[:, 1] = cumulative_probs_matrix[:, 1]
            for j in 2:($(K-1))
                probs_matrix[:, j] = max.(0.0, cumulative_probs_matrix[:, j] .-
                  cumulative_probs_matrix[:, j-1])
            end
            probs_matrix[:, $(K)] = max.(0.0, 1.0 .- cumulative_probs_matrix[:, $(K-1)])
        else
            probs_matrix[:, 1] .= 1.0
        end
        # Normalize probabilities row-wise
        probs_matrix ./= (sum(probs_matrix, dims=2) .+ 1e-9)
        # Use broadcasting to apply logpdf to each observation
        log_likelihoods = logpdf.(Categorical.(eachrow(probs_matrix)), M.y_obs)
        Turing.@addlogprob! sum(log_likelihoods)
    end
    """
end
"""
    _generate_final_likelihood_block(M::NamedTuple, is_multivariate::Bool; prefix::String = "")
Generates the final likelihood block for the Turing model, dispatching to the
appropriate helper based on the model architecture and handling special cases.
# Arguments
- `M`: The main model configuration `NamedTuple`.
- `is_multivariate`: A boolean indicating if the model is multivariate.
- `prefix`: An optional prefix string for sub-model likelihood parameters.
# Returns
- A `String` containing the generated Turing code for the likelihood block.
"""
function _generate_final_likelihood_block(
    M::NamedTuple, is_multivariate::Bool; prefix::String = ""
)
    # Check if a component like a PointProcess or a marginalized GMRF handles its own likelihood.
    has_custom_likelihood_from_component = any(
        spec -> (spec.component_obj isa PointProcess || (hasproperty(spec.component_obj,
            :method) && spec.component_obj.method == :marginalized)),
        M.components
    )
    if has_custom_likelihood_from_component
        return "" # The component's `get_updates` method will add the log-probability.
    end
    if is_multivariate
        return _generate_multivariate_likelihood_block(M; prefix = prefix)
    else
        # For univariate models, check for special families like ordinal.
        family = string(M.likelihood_specs[1][:family])
        if family == "ordinal"
            return _generate_ordinal_likelihood_block(M; prefix = prefix)
        else
            return _generate_univariate_likelihood_block(M; prefix = prefix)
        end
    end
end
"""
    _generate_intercept_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)
Generates the Turing code for the global intercept's prior.
# Arguments
- `M`: The main model configuration `NamedTuple`.
- `is_multivariate`: A boolean indicating if the model is multivariate.
- `eta_name`: The name of the linear predictor variable (unused, but kept for signature
  consistency).
# Returns
- A tuple `(priors_code::String, updates_code::String)`, where `updates_code` is always empty.
"""
function _generate_intercept_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)
    if !get(M, :add_intercept, false)
        return "", ""
    end
    intercept_prior_obj = get(M, :intercept_prior, Normal(0, 5))
    dist_str = if is_multivariate
        "filldist($(_distribution_to_string(intercept_prior_obj)), K)"
    else
        _distribution_to_string(intercept_prior_obj)
    end
    prior_code = "intercept ~ $(dist_str)"
    # The update code is intentionally empty. The intercept is added during the
    # initialization of the `eta` vector in the model assembler to ensure AD type stability.
    update_code = ""
    return prior_code, update_code
end
"""
    _generate_offset_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)
Generates the Turing code for adding log-offsets to the linear predictor.
"""
function _generate_offset_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)
    # Check if offsets are provided and are non-trivial.
    if !haskey(M, :log_offsets) || all(iszero, M.log_offsets)
        return ""
    end
    if is_multivariate
        # For multivariate models, eta_name is `eta_latent` (an N x K matrix),
        # and M.log_offsets is also an N x K matrix.
        return "$(eta_name) .+= M.log_offsets"
    else
        # For univariate models, eta_name is `eta` (an N-element vector).
        # M.log_offsets is an N x 1 matrix, so we must select the first column.
        return "$(eta_name) = $(eta_name) .+ M.log_offsets[:, 1]"
    end
end
"""
    _generate_fixed_effects_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)
Generates the Turing code for the priors and linear predictor updates for all
fixed effects. This version is CPU-only.
"""
function _generate_fixed_effects_block(
    M::NamedTuple, is_multivariate::Bool, eta_name::String; prefix::String = ""
)
    if get(M, :Xfixed_N, 0) == 0
        return "", ""
    end
    priors_vec = get(M, :Xfixed_priors_vec, [Normal(0, 5) for _ in 1:M.Xfixed_N])
    is_ordinal = any(spec -> string(get(spec, :family, "")) == "ordinal", M.likelihood_specs)
    non_prop_terms = is_ordinal ? get(M, :non_proportional_effects, Symbol[]) : Symbol[]
    prop_indices = collect(1:M.Xfixed_N)
    npo_indices = Int[]
    if is_ordinal && !isempty(non_prop_terms)
        npo_indices = findall(x -> x in non_prop_terms, M.Xfixed_names)
        prop_indices = setdiff(prop_indices, npo_indices)
    end
    n_prop = length(prop_indices)
    n_npo = length(npo_indices)
    K_ordinal = is_ordinal ? get(M.likelihood_specs[1], :K, 0) : 0
    eiv_map = hasproperty(M, :Xfixed_eiv_map) ? M.Xfixed_eiv_map : Dict{Symbol, Vector{Float64}}()
    prior_parts = String[]
    update_parts = String[]
    M_ref = !isempty(prefix) ? "sub_M_$(prefix)" : "M"
    # --- Proportional Effects ---
    if n_prop > 0
        priors_prop = priors_vec[prop_indices]
        all_same_prop = !isempty(priors_prop) && all(p -> p == priors_prop[1], priors_prop)
        beta_prop_name = !isempty(prefix) ? (is_multivariate ? "beta_flat_$(prefix)" : "beta_$(prefix)") : (is_multivariate ? "beta_flat" : "beta")
        n_params_prop = is_multivariate ? n_prop * M.outcomes_N : n_prop
        prior_label = Symbol(beta_prop_name)
        # Generate prior string for coefficients
        if all_same_prop
            prior_str = _distribution_to_string(priors_prop[1])
            push!(prior_parts, "$(beta_prop_name) ~ filldist($(prior_str), $(n_params_prop))")
        else
            priors_to_use = is_multivariate ? vcat([priors_prop for _ in 1:M.outcomes_N]...) : priors_prop
            priors_str_list = [_distribution_to_string(p) for p in priors_to_use]
            push!(prior_parts,
                "$(beta_prop_name) ~ Product([$(join(priors_str_list, ", "))])")
        end
        # Identify EIV and non-EIV columns
        eiv_prop_indices = [j for j in prop_indices if haskey(eiv_map, M.Xfixed_names[j])]
        std_prop_indices = [j for j in prop_indices if !haskey(eiv_map, M.Xfixed_names[j])]
        # Generate latent innovation priors for EIV covariates
        for j in eiv_prop_indices
            col_name = M.Xfixed_names[j]
            eiv_sym = !isempty(prefix) ? "innovations_eiv_$(prefix)_$(col_name)" :
                "innovations_eiv_$(col_name)"
            push!(prior_parts, "$(eiv_sym) ~ " *
                "filldist(Normal(0, 1), N)")
        end
        # Standard (non-EIV) linear update
        if !isempty(std_prop_indices)
            update_code = if is_multivariate
                if length(std_prop_indices) == M.Xfixed_N
                    "LinearAlgebra.mul!($(eta_name), $(M_ref).Xfixed, " *
                        "reshape($(beta_prop_name), $(n_prop), " *
                        "$(M_ref).outcomes_N), 1.0, 1.0)"
                else
                    "LinearAlgebra.mul!($(eta_name), " *
                        "view($(M_ref).Xfixed, :, $(std_prop_indices)), " *
                        "view(reshape($(beta_prop_name), $(n_prop), " *
                        "$(M_ref).outcomes_N), $(std_prop_indices), :), " *
                        "1.0, 1.0)"
                end
            else
                "$(eta_name) = $(eta_name) .+ $(M_ref).Xfixed[:, " *
                    "$(std_prop_indices)] * " *
                    "$(beta_prop_name)[$(std_prop_indices)]"
            end
            push!(update_parts, update_code)
        end
        # EIV latent linear update
        for j in eiv_prop_indices
            col_name = M.Xfixed_names[j]
            eiv_sym = !isempty(prefix) ? "innovations_eiv_$(prefix)_$(col_name)" : "innovations_eiv_$(col_name)"
            if is_multivariate
                eiv_code = """
                let
                    lat_noise = $(M_ref).Xfixed_eiv_map[$(QuoteNode(col_name))] .* $(eiv_sym)
                    X_lat_$(col_name) = $(M_ref).Xfixed[:, $(j)] .+ lat_noise
                    beta_j = reshape($(beta_prop_name), $(n_prop), $(M_ref).outcomes_N)[$(j), :]
                    $(eta_name) .+= X_lat_$(col_name) * beta_j'
                end"""
                push!(update_parts, eiv_code)
            else
                eiv_code = """
                let
                    lat_noise = $(M_ref).Xfixed_eiv_map[$(QuoteNode(col_name))] .* $(eiv_sym)
                    X_lat_$(col_name) = $(M_ref).Xfixed[:, $(j)] .+ lat_noise
                    $(eta_name) = $(eta_name) .+ X_lat_$(col_name) .* $(beta_prop_name)[$(j)]
                end"""
                push!(update_parts, eiv_code)
            end
        end
    end
    # --- Non-Proportional Effects (for Ordinal Models) ---
    if n_npo > 0 && K_ordinal > 1
        priors_npo = priors_vec[npo_indices]
        all_same_npo = !isempty(priors_npo) && all(p -> p == priors_npo[1], priors_npo)
        beta_npo_name = !isempty(prefix) ? "beta_npo_$(prefix)" : "beta_npo"
        n_npo_params = n_npo * (K_ordinal - 1)
        prior_npo_label = Symbol(beta_npo_name)
        if all_same_npo
            prior_str = _distribution_to_string(priors_npo[1])
            push!(prior_parts, "$(beta_npo_name) ~ filldist($(prior_str), $(n_npo_params))")
        else
            full_priors_list = vcat([priors_npo for _ in 1:(K_ordinal-1)]...)
            priors_str_list = [_distribution_to_string(p) for p in full_priors_list]
            push!(prior_parts,
                "$(beta_npo_name) ~ Product([$(join(priors_str_list, ",
                "))])")
        end
    end
    priors_code = join(prior_parts, "\n    ")
    updates_code = join(update_parts, "\n    ")
    return priors_code, updates_code
end
"""
    process_smooth_module!(opt_dict::Dict, mod_data::Dict, registries::Dict, hyperpriors::Dict)
Processes a module with `structure=:smooth` (invoked via `random(var, structure=:smooth,
  model=...)`).
Constructs basis matrices for static smoothers and sets up coordinate data for continuous
  and dynamic kernel-based models.
# Arguments
- `opt_dict`: The main model configuration dictionary (`M`).
- `mod_data`: The parsed data for the `random(structure=:smooth)` module.
- `registries`, `hyperpriors`: Additional configuration dictionaries.
# Returns
- `true` to indicate that a component object should be created.
"""
function process_smooth_module!(
    opt_dict::Dict, mod_data::Dict, registries::Dict, hyperpriors::Dict
)
    basis_registry = opt_dict[:basis_matrices]
    data = opt_dict[:data]
    params = mod_data[:params]
    model_param = get(params, :model, "pspline")
    original_nbins_param = get(params, :nbins, 20)
    variables = mod_data[:variables]
    n_vars = length(variables)
    # Resolve the nbins parameter at the beginning to avoid recursion.
    local nbins_resolved
    if original_nbins_param isa Int || original_nbins_param isa Vector{Int}
        nbins_resolved = original_nbins_param
    else
        calling_mod = get(opt_dict, :calling_module, Main)
        try
            nbins_resolved = Core.eval(calling_mod, original_nbins_param)
        catch e
            error("Could not evaluate `nbins` parameter `$(original_nbins_param)`. Error: $e")
        end
    end
    nbins_per_dim_vec = Int[]
    total_bins_for_component_obj = 0
    if n_vars > 0
        if nbins_resolved isa Int
            nbins_per_dim_vec = fill(nbins_resolved, n_vars)
            total_bins_for_component_obj = nbins_resolved^n_vars
        elseif nbins_resolved isa Vector{Int}
            if length(nbins_resolved) != n_vars
                error("`nbins` vector length must match number of variables for smooth. Got $(length(nbins_resolved)) for $n_vars variables.")
            end
            nbins_per_dim_vec = nbins_resolved
            total_bins_for_component_obj = prod(nbins_resolved)
        else
            error("Resolved `nbins` parameter is not an Int or Vector{Int}. Got type $(typeof(nbins_resolved))")
        end
        mod_data[:params][:nbins] = total_bins_for_component_obj
        if !isempty(nbins_per_dim_vec)
            mod_data[:params][:nbins_per_dim] = nbins_per_dim_vec
        end
    end
    # Categorize models to determine processing path
    basis_models = ["pspline", "bspline", "tps", "moran", "gp", "barycentric", "linear", "invdist"]
    dynamic_basis_models = ["wavelet", "fft"]
    continuous_kernel_models = ["gp", "fitc", "svgp", "nystrom", "warp", "spde", "exponentialdecay", "rff", "kriging"]
    gmrfs_on_bins_models = ["rw1", "rw2", "ar1", "icar", "besag", "cyclic"]
    model_str = string(model_param)
    # --- Path 1: Dynamic Basis Models (e.g., wavelet, fft) ---
    if model_str in dynamic_basis_models
        if all(v -> hasproperty(data, Symbol(v)), mod_data[:variables])
            coords = Matrix{Float64}(data[!, Symbol.(mod_data[:variables])])
            mod_data[:params][:coords] = coords
        else
            error("Coordinate variables for smooth model not found in data: $(mod_data[:variables])")
        end
        return true
    end
    # --- Path 2: Static Basis Models (e.g., pspline, tps) ---
    if model_str in basis_models
        if !isempty(mod_data[:variables]) 
            reg_key = Symbol(join(mod_data[:variables], "_"))
            if all(hasproperty(data, Symbol(v)) for v in mod_data[:variables])
                local_kwargs = Dict(params)
                delete!(local_kwargs, :nbins)
                B_smooth_matrix, actual_nbins_for_component = if n_vars == 1
                    v_vec = data[!, Symbol(mod_data[:variables][1])]
                    TethysBayes_smooth_basis_1D(model_str, v_vec, nbins_per_dim_vec[1], get(params,
                        :degree, 3); local_kwargs...)
                else
                    c_mat = Matrix{Float64}(data[!, Symbol.(mod_data[:variables])])
                    B_matrix = if n_vars == 2
                        TethysBayes_smooth_basis_2D(model_str, c_mat, nbins_per_dim_vec; local_kwargs...)
                    elseif n_vars == 3
                        TethysBayes_smooth_basis_3D(model_str, c_mat, nbins_per_dim_vec; local_kwargs...)
                    elseif n_vars == 4
                        TethysBayes_smooth_basis_4D(model_str, c_mat, nbins_per_dim_vec; local_kwargs...)
                    else
                        error("Smoothers with more than 4 dimensions are not supported for this basis type.")
                    end
                    (B_matrix, size(B_matrix, 2)) # Return matrix and its column count
                end
                basis_registry[reg_key] = B_smooth_matrix
                mod_data[:params][:nbins] = actual_nbins_for_component # Update nbins with the actual count
                if n_vars == 1
                    v_sym = Symbol(mod_data[:variables][1])
                    if !haskey(opt_dict, :domain_values)
                        opt_dict[:domain_values] = Dict{Symbol, Any}()
                    end
                    v_unique = sort(unique(data[!, v_sym]))
                    opt_dict[:domain_values][v_sym] = v_unique
                    opt_dict[Symbol("$(v_sym)_values")] = v_unique
                end
            end
        end
    # --- Path 3: Continuous Kernel Models (e.g., gp, fitc) ---
    elseif model_str in continuous_kernel_models
        if all(v -> hasproperty(data, Symbol(v)), mod_data[:variables])
            coords = Matrix{Float64}(data[!, Symbol.(mod_data[:variables])])
            mod_data[:params][:coords] = coords
            if model_str in ["fitc", "svgp", "nystrom", "gp"]
                n_inducing_default = min(100, size(coords, 1))
                n_inducing = get(mod_data[:params], :n_inducing, n_inducing_default)
                Z_inducing = generate_inducing_points(coords, n_inducing; seed=42, method="kmeans")
                mod_data[:params][:Z_inducing] = Z_inducing
            end
        else
            @warn "Continuous kernel smooth specified, but coordinate variables not found in data. Component may be misspecified."
        end
    # --- Path 4: GMRFs on Binned Covariates (e.g., rw2 on age) ---
    elseif model_str in gmrfs_on_bins_models
        vars = mod_data[:variables]
        if length(vars) != 1
            @warn "GMRF smooth on $(join(vars, ",")) requires exactly 1 variable. Skipping."; return true
        end
        var_sym = Symbol(vars[1])
        nbins = get(mod_data[:params], :nbins, 20)
        _, indices = apply_discretization_logic(data[!, var_sym], nbins)
        index_key = Symbol("mixed_idx_$(string(vars[1]))")
        opt_dict[index_key] = indices
        mod_data[:params][:indices] = indices
        mod_data[:params][:n_cat] = length(unique(indices))
        mod_data[:type] = :mixed # Re-tag for the mixed effect processor
    end    
    mod_data[:params][:model] = model_param
    return true
end
"""
    process_eigen_module!(opt_dict, mod_data, registries, hyperpriors)
Processes the `eigen()` module, which performs Bayesian Principal Component Analysis
(PCA) for dimensionality reduction.
# Version
v1.0.0
# Mathematical Summary
The `eigen()` component models a set of \$P\$ observed variables \$\\mathbf{Y}\$ (an \$N
  \\times P\$
matrix) as a linear combination of \$K\$ latent factors (principal components)
\$\\mathbf{F}\$ (an \$N \\times K\$ matrix) plus residual noise:
\$\\mathbf{Y} = \\mathbf{F} \\mathbf{L}^T + \\mathbf{E}\$
where:
- \$\\mathbf{L}\$ is the \$P \\times K\$ matrix of factor loadings (eigenvectors).
- \$\\mathbf{E}\$ is the residual noise matrix.
This processor prepares the data for the model by:
1.  Extracting the specified variables from the main data frame.
2.  Centering the data matrix by subtracting the column means, a standard
    pre-processing step for PCA.
3.  Validating that the number of requested factors is less than the number of
    input variables.
4.  Pre-calculating indices needed for the Householder transformation, which is used
    to construct the orthonormal loadings matrix \$\\mathbf{L}\$ in a numerically stable way.
# Inputs (from `mod_data`)
- `variables`: A `Vector` of `Symbol`s specifying the columns in the data to be
  used for PCA.
- `params[:n_factors]`: `Int`, the number of latent factors to extract.
# Outputs (mutates `mod_data[:params]`)
- `eigen_data::Matrix`: The centered \$N \\times P\$ data matrix.
- `n_vars::Int`: The number of input variables, \$P\$.
- `n_factors::Int`: The number of latent factors, \$K\$.
- `ltri_indices::Vector{Int}`: Indices for parameterizing the Householder reflectors.
"""
function process_eigen_module!(opt_dict, mod_data, registries, hyperpriors)
    params = mod_data[:params]
    vars_str = mod_data[:variables]
    vars_sym = Symbol.(vars_str)
    if isempty(vars_sym)
        error("The `eigen()` module was called without any variables specified.")
    end
    data = opt_dict[:data]
    if !all(hasproperty(data, v) for v in vars_sym)
        missing_vars = filter(v -> !hasproperty(data, v), vars_sym)
        error("Eigen module variables not found in data: $(missing_vars)")
    end
    # Check for missing values in the specified columns.
    if any(col -> any(ismissing, data[!, col]), vars_sym)
        error("Columns for eigen() module contain missing values. Please handle them before calling TethysBayes().")
    end
    # Extract the data and center it (a standard assumption for PCA).
    eigen_data_matrix = Matrix(data[!, vars_sym])
    eigen_data_matrix .-= mean(eigen_data_matrix, dims=1)
    # Store the data matrix in the module's parameters for the builder to access.
    mod_data[:params][:eigen_data] = eigen_data_matrix
    n_vars = length(vars_sym)
    n_factors = get(params, :n_factors, 1)
    if n_factors >= n_vars
        @warn "Number of factors ($n_factors) for eigen() module should be less than the number of variables ($n_vars). Setting to $(n_vars - 1)."
        n_factors = n_vars - 1
    end
    # Pre-calculate indices for the lower-triangular part of the Householder matrix.
    # This parameterizes the reflector vectors for constructing the orthonormal loadings matrix.
    ltri_mask = [r >= c for r in 1:n_vars, c in 1:n_factors]
    ltri_indices = findall(vec(ltri_mask))
    mod_data[:params][:ltri_indices] = ltri_indices
    mod_data[:params][:n_factors] = n_factors
    mod_data[:params][:n_vars] = n_vars
    return true # Proceed with component creation.
end
"""
    process_mixed_module!(opt_dict, mod_data, registries, hyperpriors)
Processes the `mixed()` module for random effects.
# Arguments
- `opt_dict`: The main model configuration dictionary.
- `mod_data`: The parsed data for the `mixed()` module.
- `registries`, `hyperpriors`: Additional configuration dictionaries.
# Returns
- `true` to indicate that a `Mixed` component object should be created.
"""
function process_mixed_module!(opt_dict, mod_data, registries, hyperpriors)
    data = opt_dict[:data]
    vars = mod_data[:variables]
    response_var = Symbol(opt_dict[:outcomes][1])
    effect_expr, group_var_str = if !isempty(vars) && vars[1] isa Expr && vars[1].head == :call&&
        vars[1].args[1] == :|
        # Handles the `effect | group` syntax.
        (vars[1].args[2], string(vars[1].args[3]))
    elseif length(vars) >= 2
        # Handles the `effect, group` syntax.
        (vars[1], string(vars[2]))
    else
        @warn "The mixed() module requires syntax `mixed(effect | group)` or `mixed(effect, group)`. Skipping."
        return false
    end
    # Replace TethysBayes-specific modules like `intercept()` with `1` for StatsModels.jl.
    effect_expr_mod = _replace_TethysBayes_modules_in_expr(effect_expr)
    schema = StatsModels.schema(data)
    terms = if effect_expr_mod isa Number
        StatsModels.term(effect_expr_mod)
    else
        calling_mod = get(opt_dict, :calling_module, Main)
        form = Core.eval(calling_mod, :(@formula($response_var ~ $(effect_expr_mod))))
        applied_form = StatsModels.apply_schema(form, schema)
        applied_form.rhs
    end
    # Decompose the parsed formula into a vector of individual term strings.
    # This correctly handles multi-term effects like `(1 + cov1 | group)`.
    term_vec = if terms isa StatsModels.TupleTerm
        terms.terms
    elseif terms isa StatsModels.AbstractTerm
        (terms,) # Wrap single term in a tuple for consistent iteration.
    elseif terms isa Tuple
        collect(terms)
    else
        [terms]
    end
    effect_names = String[]
    for term in term_vec
        if term isa StatsModels.InterceptTerm{true}
            push!(effect_names, "1")
        elseif term isa StatsModels.InterceptTerm{false}
            continue # Skip if intercept is explicitly removed with `0`.
        else
            push!(effect_names, _canonical_term_string(term))
        end
    end
    group_var_sym = Symbol(group_var_str)
    if !hasproperty(data, group_var_sym)
        error("Grouping variable ':$group_var_sym' for mixed() module not found in dataset.")
    end
    # Create integer indices for the grouping variable.
    group_data = data[!, group_var_sym]
    unique_levels = unique(group_data)
    n_levels = length(unique_levels)
    n_obs = nrow(data)
    # Add a warning if the number of levels is suspiciously high.
    if n_levels > n_obs / 2 && n_levels > 50 # Heuristic threshold
        @warn "Grouping variable ':$group_var_sym' has a large number of unique levels ($n_levels for $n_obs observations). If this is a continuous variable, the model may be very large and slow. Consider binning the variable or ensuring it is a categorical factor."
    end
    group_map = Dict(v => i for (i, v) in enumerate(unique_levels))
    indices = [group_map[v] for v in group_data]
    # Store the generated indices and parameters in the configuration dictionaries
    # for the code generator to use.
    index_key = Symbol("mixed_idx_$(group_var_str)")
    opt_dict[index_key] = indices
    mod_data[:params][:indices] = indices
    mod_data[:params][:n_cat] = n_levels
    mod_data[:params][:lhs] = effect_names
    mod_data[:params][:grouping_covariate] = group_var_sym
    mod_data[:variables] = [group_var_str]
    return true
end
"""
    process_fixed_module!(opt_dict, mod_data, registries, hyperpriors)
Processes the `fixed()` module, gathering information about fixed effects, custom
contrasts, and priors.
# Arguments
- `opt_dict`: The main model configuration dictionary.
- `mod_data`: The parsed data for the `fixed()` module.
- `registries`, `hyperpriors`: Additional configuration dictionaries.
# Returns
- `false`, as the `fixed()` module itself does not create a `Component` object.
"""
function process_fixed_module!(opt_dict, mod_data, registries, hyperpriors)
    # Ensure necessary keys exist in the configuration dictionary.
    get!(opt_dict, :fixed_effects_from_modules, String[])
    get!(opt_dict, :contrasts, Dict{Symbol, Any}())
    get!(opt_dict, :fixed_effects_priors, Dict{Symbol, Any}())
    get!(opt_dict, :vars_to_categorize, Set{Symbol}())
    get!(opt_dict, :fixed_effects_eiv, Dict{Symbol, Any}())
    params = mod_data[:params]
    vars = mod_data[:variables]
    # Collect all variables specified in this fixed() call.
    for var in vars
        push!(opt_dict[:fixed_effects_from_modules], string(var))
    end
    # Handle Errors-in-Variables (EIV) error standard deviations.
    if haskey(params, :error_sd) || haskey(params, :sd_error) || haskey(params, :se)
        eiv_spec = get(params, :error_sd, get(params, :sd_error, get(params, :se, nothing)))
        for var in vars
            opt_dict[:fixed_effects_eiv][Symbol(var)] = eiv_spec
        end
    end
    # Handle custom contrast coding.
    if haskey(params, :contrast)
        if !isempty(vars)
            contrast_sym = params[:contrast]
            contrast_obj = if haskey(STATSMODELS_CONTRASTS, contrast_sym)
                STATSMODELS_CONTRASTS[contrast_sym]
            else
                @warn "Unknown contrast coding ':$contrast_sym'. Using default (DummyCoding)."
                StatsModels.DummyCoding()
            end
            # Apply the contrast to all variables in this fixed() call.
            for var in vars
                opt_dict[:contrasts][Symbol(var)] = contrast_obj
            end
        else
            @warn "A 'contrast' was specified in a fixed() module with no variable. Ignoring."
        end
    end
    # Handle custom priors.
    if haskey(params, :prior)
        for var in vars
            opt_dict[:fixed_effects_priors][Symbol(var)] = params[:prior]
        end
    end
    # Handle explicit categorization.
    if get(params, :model, nothing) == :categorical || haskey(params, :contrast)
        for var in vars
            push!(opt_dict[:vars_to_categorize], Symbol(var))
        end
    end
    # The fixed() module only configures the model; it does not create a component itself.
    return false
end
"""
    process_custom_module!(opt_dict, mod_data, registries, hyperpriors)
Processes the `custom()` module.
# Arguments
- `opt_dict`: The main model configuration dictionary.
- `mod_data`: The parsed data for the `custom()` module.
- `registries`, `hyperpriors`: Additional configuration dictionaries (not used here).
# Returns
- `true` to indicate that a `Custom` component object should be created.
"""
function process_custom_module!(opt_dict, mod_data, registries, hyperpriors)
    # Validate that the essential `code_fragment` parameter is present.
    if !haskey(mod_data[:params], :code_fragment)
        error("The `custom()` module requires a `code_fragment` argument containing the user-defined Turing code string.")
    end
    # This module is a placeholder for user-defined code.
    # It performs no data processing itself but signals that a component
    # object should be created to hold the user's code fragment.
    return true
end
"""
    process_localadaptive_module!(opt_dict, mod_data, registries, hyperpriors)

Processes the `localadaptive` model, ensuring centroids and cluster assignments are
correctly computed for all spatial units.

# Arguments
- `opt_dict`: The main model configuration dictionary.
- `mod_data`: The parsed data for the `random(model=:localadaptive)` module.
- `registries`, `hyperpriors`: Additional configuration dictionaries (not used here).

# Returns
- `true` to indicate that a `LocalAdaptive` component object should be created.
"""
function process_localadaptive_module!(opt_dict, mod_data, registries, hyperpriors)
    s_N = get(opt_dict, :s_N, 0)
    if s_N == 0
        error("The `localadaptive` model requires a spatial context (`s_N`), but it has not been established. Ensure a spatial variable and adjacency matrix `W` are provided.")
    end
    
    data = opt_dict[:data]

    # The localadaptive model requires centroids for clustering.
    if !haskey(opt_dict, :centroids)
        coord_cols = if haskey(mod_data, :variables) && length(mod_data[:variables]) >= 2
            (Symbol(mod_data[:variables][1]), Symbol(mod_data[:variables][2]))
        else
            try
                _detect_xy_columns(data)
            catch
                nothing
            end
        end

        s_idx_col = if hasproperty(data, :s_idx)
            :s_idx
        elseif haskey(opt_dict, :s_idx_var) && hasproperty(data, Symbol(opt_dict[:s_idx_var]))
            Symbol(opt_dict[:s_idx_var])
        else
            nothing
        end

        if coord_cols !== nothing && s_idx_col !== nothing &&
           hasproperty(data, coord_cols[1]) && hasproperty(data, coord_cols[2])
            cx, cy = coord_cols
            gdf = groupby(data, s_idx_col)
            unique_coords_df = combine(gdf, [cx, cy] .=> first, renamecols=false)
            coord_map = Dict(row[s_idx_col] => Point2D(row[cx],
                row[cy]) for row in eachrow(unique_coords_df))

            if length(coord_map) < s_N
                error("The `localadaptive` model requires coordinates for all $(s_N) spatial units, but only found coordinates for $(length(coord_map)) unique units in the data. Please provide a complete `centroids` vector as a keyword argument.")
            end

            centroids = Vector{Point2D}(undef, s_N)
            for i in 1:s_N
                if !haskey(coord_map, i)
                     error("Coordinate for spatial unit `$s_idx_col = $i` not found in data. The `localadaptive` model requires complete coordinate information.")
                end
                centroids[i] = coord_map[i]
            end
            opt_dict[:centroids] = centroids
        else
            error("The `localadaptive()` model requires centroids for clustering. Provide them via `centroids` or ensure spatial coordinates and indices (s_idx) are in the data frame.")
        end

    end

    centroids = opt_dict[:centroids]
    if length(centroids) != s_N
        error("The number of provided centroids ($(length(centroids))) does not match the number of spatial units s_N ($(s_N)).")
    end
    params = mod_data[:params]
    n_clusters = get(params, :n_clusters, 5)
    
    if length(centroids) < n_clusters
        @warn "Number of spatial units ($(length(centroids))) is less than the requested number of clusters ($n_clusters). Adjusting n_clusters to $(length(centroids))."
        n_clusters = length(centroids)
    end
    
    # Clustering.jl expects a [dims x n_points] matrix
    centroids_matrix = hcat([c.x for c in centroids], [c.y for c in centroids])'
    
    # Perform k-means clustering on the centroids
    kmeans_result = kmeans(centroids_matrix, n_clusters; maxiter=200, display=:none)
    
    # The assignments map each of the s_N centroids to a cluster. This vector now has the
    #   correct length.
    opt_dict[:cluster_assignments] = assignments(kmeans_result)
    opt_dict[:n_clusters] = nclusters(kmeans_result)
    
    return true
end
"""
    process_nested_module!(opt_dict, mod_data, registries, hyperpriors)
Processes the `nested()` module for multi-fidelity or joint models. This function
recursively calls `TethysBayes_config` to create a complete and independent configuration
for the sub-model defined within the `nested()` call.
# Arguments
- `opt_dict`: The main model configuration dictionary.
- `mod_data`: The parsed data for the `nested()` module.
- `registries`, `hyperpriors`: Additional configuration dictionaries.
# Returns
- `false`, as the `nested()` module itself does not create a `Component` object.
"""
function process_nested_module!(opt_dict, mod_data, registries, hyperpriors)
    if !haskey(opt_dict, :nested_components)
        opt_dict[:nested_components] = Dict{Symbol, Any}()
    end
    var = Symbol(mod_data[:variables][1])
    params = mod_data[:params]
    # Check if sub-model was pre-configured (e.g. via paired multi-equation syntax or submodels dict)
    if haskey(opt_dict[:nested_components], var)
        sub_cfg = opt_dict[:nested_components][var]
        if haskey(params, :prior) || haskey(params, :coupling_prior)
            c_prior = get(params, :prior, get(params, :coupling_prior, Normal(1.0, 0.5)))
            sub_cfg = merge(sub_cfg, (coupling_prior = c_prior,))
        end
        if haskey(params, :fixed) || haskey(params, :fixed_coupling)
            f_coupling = get(params, :fixed, get(params, :fixed_coupling, false))
            sub_cfg = merge(sub_cfg, (fixed_coupling = f_coupling,))
        end
        opt_dict[:nested_components][var] = sub_cfg
        return false
    end
    sub_formula_raw = get(params, :formula, "")
    data_source_sym = get(params, :data_source, :data)
    if !haskey(opt_dict, data_source_sym)
        @warn "Data source ':$data_source_sym' for nested module on '$var' not found. Skipping."
        return false
    end
    sub_data = opt_dict[data_source_sym]
    # Prepare keyword arguments for the recursive TethysBayes_config call.
    # Exclude internal state keys specific to the parent model.
    sub_config_kwargs = copy(opt_dict)
    keys_to_exclude = [
        :data, :formula, data_source_sym,
        :y_obs, :y_N, :y_cols, :outcomes, :likelihood_specs,
        :components, :nested_components, :basis_matrices,
        :Xfixed, :Xfixed_N, :Xfixed_names, :Xfixed_priors_vec,
        :add_intercept, :intercept_prior, :model_arch, :outcomes_N,
        :is_multivariate_dynamics, :multivariate_dynamics_key, :model_st,
        :st_interaction_sigma_prior, :sigma_st_interaction_prior,
        :Xfixed_eiv_map, :trials, :weights, :censor_lower, :censor_upper,
        :hurdle, :log_offsets, :user_provided_trials, :user_provided_weights,
        :user_provided_censor_lower, :user_provided_censor_upper,
        :user_provided_hurdle, :user_provided_log_offsets,
        :fixed_effects_from_modules, :generated_model_code
    ]
    for k in keys_to_exclude
        delete!(sub_config_kwargs, k)
    end
    # Robustly handle the formula argument, which can be a String, Symbol, or Expr.
    sub_formula_str::String = if sub_formula_raw isa String
        sub_formula_raw
    elseif sub_formula_raw isa Symbol
        calling_mod = get(opt_dict, :calling_module, Main)
        try
            Core.eval(calling_mod, sub_formula_raw)
        catch e
            error("Could not evaluate the `formula` variable `$(sub_formula_raw)` for the nested module. Ensure it is defined in the calling scope. Error: $e")
        end
    elseif sub_formula_raw isa Expr
        string(sub_formula_raw)
    else
        error("Unsupported type for `formula` argument in nested module: $(typeof(sub_formula_raw))")
    end
    if !(sub_formula_str isa String)
        error("The `formula` argument for the nested module must resolve to a String. Got type: $(typeof(sub_formula_str))")
    end
    # Validate observation count alignment or mapping
    parent_N = size(opt_dict[:data], 1)
    sub_N = size(sub_data, 1)
    # Recursively call TethysBayes_config to create a full configuration for the sub-model.
    calling_mod = get(opt_dict, :calling_module, Main)
    sub_config = TethysBayes_config(
        sub_formula_str, sub_data;
        calling_module = calling_mod,
        sub_config_kwargs...
    )
    # Resolve strata, coupling modes, and mappings via _process_nested_link
    sub_config = _process_nested_link(
        params, sub_config, opt_dict[:data], :main, var, calling_mod;
        parent_scope = opt_dict
    )
    # Store the complete sub-model configuration.
    opt_dict[:nested_components][var] = sub_config
    # The nested module itself does not create a component in the main model loop.
    return false
end



"""

    process_random_module!(opt_dict, mod_data, registries, hyperpriors)


Processes a `random()` module call from the formula. This function is a central
part of the model configuration pipeline. Its primary responsibility is to set up
the structural context (e.g., :spatial, :temporal) for a random effect.
It infers the structure, validates and processes index variables (like `s_idx` or
`t_idx`), and populates the main configuration dictionary (`opt_dict`) with shared
information like the number of spatial units (`s_N`) or the adjacency matrix (`W`).
This processor fully handles the creation and registration of the component and
returns `false` to signal to the main configuration loop that no further processing
is needed for this component.
"""

function process_random_module!(
    opt_dict::Dict, mod_data::Dict, registries::Dict, hyperpriors::Dict
)
    model_name = get(mod_data[:params], :model, :iid)
    model_sym = model_name isa Symbol ? model_name : Symbol(model_name)
    data = opt_dict[:data]
    # If variables omitted, attempt auto-detection by model family
    if isempty(mod_data[:variables])
        if model_sym in [:cyclic, :harmonic]
            det_u = _detect_seasonal_column(data; allow_nothing=true)
            if !isnothing(det_u)
                mod_data[:variables] = [det_u]
            end
        elseif model_sym in [:icar, :besag, :bym2, :leroux, :localadaptive, :spde]
            det_s = _detect_spatial_unit_column(data; allow_nothing=true)
            if !isnothing(det_s)
                mod_data[:variables] = [det_s]
            end
        elseif model_sym in [:ar1, :ar2, :rw1, :rw2]
            det_t = _detect_time_column(data; allow_nothing=true)
            if !isnothing(det_t)
                mod_data[:variables] = [det_t]
            end
        elseif model_sym in [:rff, :gp, :sparsegp, :svgp, :fitc, :tps, :wavelet,
                             :waveletgp, :nystrom, :spectralgp, :barycentric]
            det_xy = _detect_xy_columns(data; allow_nothing=true)
            if !isnothing(det_xy)
                mod_data[:variables] = [det_xy[1], det_xy[2]]
            end
        elseif model_sym == :iid
            det_g = _detect_group_column(data; allow_nothing=true)
            if !isnothing(det_g)
                mod_data[:variables] = [det_g]
            end
        end
        if !isempty(mod_data[:variables])
            mod_data[:params][:positional_args] = mod_data[:variables]
        end
    end


    # Ensure key is present
    if !haskey(mod_data, :key) || mod_data[:key] == Symbol("") ||
       isempty(string(mod_data[:key]))
        if !isempty(mod_data[:variables])
            mod_data[:key] = Symbol(join(string.(mod_data[:variables]), "_"))
        else
            mod_data[:key] = model_sym
        end
    end



    # 1. Infer the structure (:spatial, :temporal, :smooth, etc.)
    structure = _infer_structure_from_args(mod_data[:variables], mod_data[:params])
    mod_data[:params][:structure] = structure
    # 2. Process based on the inferred structure, setting up shared indices in opt_dict
    if structure == :spatial
        variables = mod_data[:variables]
        s_var_sym = if !isempty(variables)
            Symbol(variables[1])
        else
            det_s = _detect_spatial_unit_column(data; allow_nothing=false)
            det_s
        end
        if mod_data[:key] == Symbol("") || isempty(string(mod_data[:key]))
            mod_data[:key] = s_var_sym
        end
        if !hasproperty(data, s_var_sym)
            error("Spatial index variable ':$s_var_sym' not found in data.")
        end
        raw_s = data[!, s_var_sym]
        if any(ismissing, raw_s)
            error("Spatial index variable ':$s_var_sym' contains missing values.")
        end
        unique_units = if applicable(levels, raw_s) && !isempty(levels(raw_s))
            filter(in(unique(raw_s)), levels(raw_s))
        else
            sort(unique(raw_s))
        end
        s_N = length(unique_units)
        # Map to 1-based integer indices 1:s_N for latent spatial parameter indexing
        s_idx_vec = if eltype(raw_s) <: Integer && minimum(raw_s) == 1 &&
                       maximum(raw_s) == s_N
            Int.(raw_s)
        else
            unit_to_idx = Dict(u => i for (i, u) in enumerate(unique_units))
            [unit_to_idx[u] for u in raw_s]
        end
        if !haskey(opt_dict, :s_idx)
            opt_dict[:s_idx] = s_idx_vec
            opt_dict[:s_N] = s_N
            opt_dict[:s_idx_var] = s_var_sym
            opt_dict[:s_values] = unique_units
            opt_dict[Symbol("$(s_var_sym)_values")] = unique_units
            if !haskey(opt_dict, :domain_values)
                opt_dict[:domain_values] = Dict{Symbol, Any}()
            end
            opt_dict[:domain_values][s_var_sym] = unique_units
        end
        if !hasproperty(opt_dict[:data], :s_idx)
            opt_dict[:data][!, :s_idx] = s_idx_vec
        end
    elseif structure == :temporal
        variables = mod_data[:variables]
        t_var_sym = if !isempty(variables)
            Symbol(variables[1])
        else
            det_t = _detect_time_column(data; allow_nothing=false)
            det_t
        end
        if mod_data[:key] == Symbol("") || isempty(string(mod_data[:key]))
            mod_data[:key] = t_var_sym
        end
        if !hasproperty(data, t_var_sym)
            error("Temporal index variable ':$t_var_sym' not found in data.")
        end
        if any(ismissing, data[!, t_var_sym])
            error("Temporal index variable ':$t_var_sym' contains missing values.")
        end
        if !haskey(opt_dict, :t_idx)
            time_opts = Dict(
                :time_method => get(mod_data[:params], :time_method, "regular")
            )
            tu_meta = assign_time_units(data[!, t_var_sym]; time_opts...)
            opt_dict[:t_idx] = tu_meta.idx
            opt_dict[:t_N] = tu_meta.N_cat
            opt_dict[:t_idx_var] = t_var_sym
            opt_dict[:t_values] = tu_meta.mids
            opt_dict[Symbol("$(t_var_sym)_values")] = tu_meta.mids
            if !haskey(opt_dict, :domain_values)
                opt_dict[:domain_values] = Dict{Symbol, Any}()
            end
            opt_dict[:domain_values][t_var_sym] = tu_meta.mids
        end
        if !hasproperty(opt_dict[:data], :t_idx)
            opt_dict[:data][!, :t_idx] = opt_dict[:t_idx]
        end
    elseif structure == :spacetime
        variables = mod_data[:variables]
        s_var_sym = if length(variables) >= 1
            Symbol(variables[1])
        else
            det_s = _detect_spatial_unit_column(data; allow_nothing=false)
            det_s
        end
        t_var_sym = if length(variables) >= 2
            Symbol(variables[2])
        else
            det_t = _detect_time_column(data; allow_nothing=false)
            det_t
        end
        if mod_data[:key] == Symbol("") || isempty(string(mod_data[:key]))
            mod_data[:key] = Symbol("$(s_var_sym)_$(t_var_sym)")
        end
        if !haskey(opt_dict, :s_idx)
            if !hasproperty(data, s_var_sym)
                error("Spatial index variable ':$s_var_sym' not found in data.")
            end
            raw_s = data[!, s_var_sym]
            if any(ismissing, raw_s)
                error("Spatial index variable ':$s_var_sym' contains missing values.")
            end
            unique_units = if applicable(levels, raw_s) && !isempty(levels(raw_s))
                filter(in(unique(raw_s)), levels(raw_s))
            else
                sort(unique(raw_s))
            end
            s_N = length(unique_units)
            s_idx_vec = if eltype(raw_s) <: Integer && minimum(raw_s) == 1 &&
                           maximum(raw_s) == s_N
                Int.(raw_s)
            else
                unit_to_idx = Dict(u => i for (i, u) in enumerate(unique_units))
                [unit_to_idx[u] for u in raw_s]
            end
            opt_dict[:s_idx] = s_idx_vec
            opt_dict[:s_N] = s_N
            opt_dict[:s_idx_var] = s_var_sym
            opt_dict[:s_values] = unique_units
            opt_dict[Symbol("$(s_var_sym)_values")] = unique_units
            if !haskey(opt_dict, :domain_values)
                opt_dict[:domain_values] = Dict{Symbol, Any}()
            end
            opt_dict[:domain_values][s_var_sym] = unique_units
        end
        if !hasproperty(opt_dict[:data], :s_idx)
            opt_dict[:data][!, :s_idx] = opt_dict[:s_idx]
        end
        if !haskey(opt_dict, :t_idx)
            if !hasproperty(data, t_var_sym)
                error("Temporal index variable ':$t_var_sym' not found in data.")
            end
            if any(ismissing, data[!, t_var_sym])
                error("Temporal index variable ':$t_var_sym' contains missing values.")
            end
            time_opts = Dict(
                :time_method => get(mod_data[:params], :time_method, "regular")
            )
            tu_meta = assign_time_units(data[!, t_var_sym]; time_opts...)
            opt_dict[:t_idx] = tu_meta.idx
            opt_dict[:t_N] = tu_meta.N_cat
            opt_dict[:t_idx_var] = t_var_sym
            opt_dict[:t_values] = tu_meta.mids
            opt_dict[Symbol("$(t_var_sym)_values")] = tu_meta.mids
            if !haskey(opt_dict, :domain_values)
                opt_dict[:domain_values] = Dict{Symbol, Any}()
            end
            opt_dict[:domain_values][t_var_sym] = tu_meta.mids
        end
        if !hasproperty(opt_dict[:data], :t_idx)
            opt_dict[:data][!, :t_idx] = opt_dict[:t_idx]
        end
        if !haskey(opt_dict, :st_idx)
            opt_dict[:st_idx] = (opt_dict[:t_idx] .- 1) .* opt_dict[:s_N] .+
                                opt_dict[:s_idx]
        end
        if !hasproperty(opt_dict[:data], :st_idx)
            opt_dict[:data][!, :st_idx] = opt_dict[:st_idx]
        end
    elseif structure == :seasonal || model_sym in [:cyclic, :harmonic]
        variables = mod_data[:variables]
        u_var_sym = if !isempty(variables)
            Symbol(variables[1])
        else
            det_u = _detect_seasonal_column(data; allow_nothing=false)
            det_u
        end
        if mod_data[:key] == Symbol("") || isempty(string(mod_data[:key]))
            mod_data[:key] = u_var_sym
        end
        if !hasproperty(data, u_var_sym)
            error("Seasonal index variable ':$u_var_sym' not found in data.")
        end
        raw_u = data[!, u_var_sym]
        if any(ismissing, raw_u)
            error("Seasonal index variable ':$u_var_sym' contains missing values.")
        end
        unique_units = if applicable(levels, raw_u) && !isempty(levels(raw_u))
            filter(in(unique(raw_u)), levels(raw_u))
        else
            sort(unique(raw_u))
        end
        u_N = length(unique_units)
        # Map to 1-based integer indices 1:u_N for latent seasonal parameter indexing
        u_idx_vec = if eltype(raw_u) <: Integer && minimum(raw_u) == 1 &&
                       maximum(raw_u) == u_N
            Int.(raw_u)
        else
            unit_to_idx = Dict(u => i for (i, u) in enumerate(unique_units))
            [unit_to_idx[u] for u in raw_u]
        end
        if !haskey(opt_dict, :u_idx)
            opt_dict[:u_idx] = u_idx_vec
            opt_dict[:u_N] = u_N
            opt_dict[:u_idx_var] = u_var_sym
            opt_dict[:u_values] = unique_units
            opt_dict[Symbol("$(u_var_sym)_values")] = unique_units
            if !haskey(opt_dict, :domain_values)
                opt_dict[:domain_values] = Dict{Symbol, Any}()
            end
            opt_dict[:domain_values][u_var_sym] = unique_units
        end
        if !hasproperty(opt_dict[:data], :u_idx)
            opt_dict[:data][!, :u_idx] = u_idx_vec
        end
    elseif structure == :smooth
        variables = mod_data[:variables]
        if isempty(variables)
            det_xy = _detect_xy_columns(data; allow_nothing=true)
            if !isnothing(det_xy)
                mod_data[:variables] = [det_xy[1], det_xy[2]]
                mod_data[:params][:positional_args] = mod_data[:variables]
                if mod_data[:key] == Symbol("") || isempty(string(mod_data[:key]))
                    mod_data[:key] = Symbol("$(det_xy[1])_$(det_xy[2])")
                end
            end
        end
    end



    calling_mod = get(opt_dict, :calling_module, Main)

    # Register W from component params if present
    if !haskey(opt_dict, :W) && haskey(mod_data[:params], :W)
        W_arg = mod_data[:params][:W]
        if W_arg isa Symbol || W_arg isa Expr
            try
                opt_dict[:W] = Core.eval(calling_mod, W_arg)
            catch e
                error("Could not evaluate `W` argument `$(W_arg)` for component '$(mod_data[:key])'. Error: $e")
            end
        else
            opt_dict[:W] = W_arg
        end
    end

    # Register Q from component params if present
    if !haskey(opt_dict, :Q) && haskey(mod_data[:params], :Q)
        Q_arg = mod_data[:params][:Q]
        if Q_arg isa Symbol || Q_arg isa Expr
            try
                opt_dict[:Q] = Core.eval(calling_mod, Q_arg)
            catch e
                error("Could not evaluate `Q` argument `$(Q_arg)` for component '$(mod_data[:key])'. Error: $e")
            end
        else
            opt_dict[:Q] = Q_arg
        end
    end

    # Validate spatial dimension compatibility between data indices and W
    if haskey(opt_dict, :s_N) && haskey(opt_dict, :W) && opt_dict[:W] isa AbstractMatrix
        if opt_dict[:s_N] != size(opt_dict[:W], 1)
            error("Number of unique spatial indices ($(opt_dict[:s_N])) in data does not match the dimension of adjacency matrix W ($(size(opt_dict[:W], 1))).")
        end
    end

    # 3. Create the component object
    component_obj = resolve_technical_primitive(
        mod_data, NamedTuple(opt_dict), hyperpriors, opt_dict[:prior_scheme]
    )
    mod_data[:component_obj] = component_obj

    # 4. Get precomputes
    M_nt = NamedTuple(opt_dict)
    precomputes = get_precomputes(component_obj, M_nt, mod_data)

    # 5. Create the final specification object
    spec = (
        key=Symbol(mod_data[:key]), 
        structure=mod_data[:params][:structure], 
        var=join(string.(mod_data[:variables]), "_"), 
        component_obj=component_obj, 
        params=mod_data[:params], 
        hyper=precomputes
    )
    
    # 6. Add the final spec to the main components list
    push!(registries[:components], spec)

    # 7. Return false to signal to TethysBayes_config that this component is fully processed
    return false
end
"""
    adjacency_to_bipartite(W::AbstractMatrix; force_bipartite::Bool=true)
Converts a unipartite square adjacency matrix `W` into a bipartite graph representation by
  finding a 2-coloring.
"""
function adjacency_to_bipartite(W::AbstractMatrix; force_bipartite::Bool=true)
    rows, cols = size(W)
    if rows != cols
        error("Input matrix must be square to represent a unipartite adjacency structure.")
    end
    n = rows
    g = SimpleGraph(W)
    # # Coloring Algorithm: Attempt to find a natural 2-coloring (bipartition)
    # # nodes are assigned to set 0 or set 1
    colors = fill(-1, n)
    is_bipartite = true
    for start_node in 1:n
        if colors[start_node] != -1
            continue
        end
        colors[start_node] = 0
        queue = [start_node]
        while !isempty(queue)
            u = popfirst!(queue)
            for v in Graphs.neighbors(g, u)
                if colors[v] == -1
                    colors[v] = 1 - colors[u]
                    push!(queue, v)
                elseif colors[v] == colors[u]
                    is_bipartite = false
                    if !force_bipartite
                        error("Graph is not bipartite and force_bipartite is false.")
                    end
                end
            end
        end
    end
    # # Fallback: If not bipartite, use a greedy degree-based partition to maximize cut
    if !is_bipartite
        @warn "Graph is not naturally bipartite. Applying greedy partitioning to maximize inter-set edges."
        colors = fill(0, n)
        node_degrees = degree(g)
        sorted_nodes = sortperm(node_degrees, rev=true)
        for u in sorted_nodes
            # # Count neighbors already in set 0 and set 1
            n0 = 0
            n1 = 0
            for v in Graphs.neighbors(g, u)
                if colors[v] == 0
                    n0 += 1
                else
                    n1 += 1
                end
            end
            # # Assign to the set that maximizes connections to the other set
            colors[u] = n0 >= n1 ? 1 : 0
        end
    end
    # # Extraction: Construct the bipartite matrix B
    set1_indices = findall(==(0), colors)
    set2_indices = findall(==(1), colors)
    n1 = length(set1_indices)
    n2 = length(set2_indices)
    if n1 == 0 || n2 == 0
        error("Partitioning failed to create two non-empty sets. Check graph connectivity.")
    end
    # # B is n1 x n2 matrix representing connections from Set 1 to Set 2
    B = spzeros(Float64, n1, n2)
    for (i, u) in enumerate(set1_indices)
        for (j, v) in enumerate(set2_indices)
            if W[u, v] > 0
                B[i, j] = Float64(W[u, v])
            end
        end
    end
    return (
        bipartite_adj = B,
        set1 = set1_indices,
        set2 = set2_indices,
        is_natural = is_bipartite
    )
end



"""

    process_interact_module!(opt_dict, mod_data, registries, hyperpriors)


Processes interaction modules created by operators like `|>`, `∘`, and `⊗`.

# Arguments
- `opt_dict`: The main model configuration dictionary.
- `mod_data`: The parsed data for the interaction module.
- `registries`, `hyperpriors`: Additional configuration dictionaries.

# Returns
- `true` if a component object should be created for this interaction.
- `false` if the interaction is handled globally (e.g., Kronecker product).
"""
function process_interact_module!(opt_dict, mod_data, registries, hyperpriors)
    op = get(mod_data, :operator, get(mod_data[:params], :operator, nothing))
    components = get(mod_data, :components, get(mod_data[:params], :components, []))
    if isnothing(op) || isempty(components)
        return false
    end
    
    if op == :composition && length(components) == 2
        outer_node, inner_node = components[1], components[2]
        
        if outer_node.module_type == :pointprocess
            return true
        end

        is_nonstationary_variance = outer_node.module_type == :random && get(outer_node.args,
            :structure,
            :none) == :spatial && inner_node.module_type == :random && get(inner_node.args,
            :structure, :none) == :smooth
        if is_nonstationary_variance
            modifier_vars = get(inner_node.args, :positional_args, [])
            if isempty(modifier_vars)
                @warn "The modifier component (smooth) of a composition operator is missing variables. Skipping."; return false
            end
            
            # Corrected call: Use process_random_module! for the smooth component.
            smooth_mod_data = Dict(:type => :random, :variables => modifier_vars,
                :params => inner_node.args)
            process_random_module!(opt_dict, smooth_mod_data, registries, hyperpriors)
            
            mod_data[:type] = :nonstationary_variance
            mod_data[:params][:base_node] = outer_node
            mod_data[:params][:modifier_node] = inner_node
            mod_data[:params][:modifier_basis_key] = Symbol(join(modifier_vars, "_"))
            
            return true 
        end
    end

    if op == :pipe && length(components) == 2
        node1, node2 = components[1], components[2]
        
        if node1.module_type == :random && !haskey(node1.args, :structure)
            args1 = copy(node1.args); args1[:vars] = get(node1.args, :positional_args, [])
            node1.args[:structure] = _infer_structure_from_args(args1)
        end

        if node2.module_type == :random && !haskey(node2.args, :structure)
            args2 = copy(node2.args); args2[:vars] = get(node2.args, :positional_args, [])
            node2.args[:structure] = _infer_structure_from_args(args2)
        end
        is_spatially_varying_curve = node1.module_type == :random && get(node1.args,
            :structure, :none) == :smooth &&
                                     node2.module_type == :random && get(node2.args,
                                         :structure, :none) == :spatial

        is_svc = node1.module_type == :fixed && node2.module_type == :random && (
            get(node2.args, :structure, :none) == :spatial ||
            (get(node2.args, :structure, :none) == :smooth &&
             length(get(node2.args, :positional_args, [])) >= 2)
        )
        is_tvc = node1.module_type == :fixed && node2.module_type == :random && get(node2.args,
            :structure, :none) == :temporal
        is_svar = node1.module_type == :random && get(node1.args, :structure,
            :none) == :temporal && node2.module_type == :random && get(node2.args, :structure,
            :none) == :spatial

        if is_spatially_varying_curve
            dynamic_node = node1
            dynamic_vars = get(dynamic_node.args, :positional_args, [])
            if isempty(dynamic_vars)
                error("The dynamic part of a pipe operator (e.g., a smoother) must have a variable.")
            end
            
            # Corrected call: Use process_random_module! for the smooth component.
            smooth_mod_data = Dict(:type => :random, :variables => dynamic_vars,
                :params => dynamic_node.args)
            process_random_module!(opt_dict, smooth_mod_data, registries, hyperpriors)
            
            state_node = node2
            # Corrected call: Use process_random_module! for the spatial component.
            spatial_mod_data = Dict(:type => :random, :variables => get(state_node.args,
                :positional_args, []), :params => state_node.args)
            process_random_module!(opt_dict, spatial_mod_data, registries, hyperpriors)
            
            mod_data[:params][:dynamic_component_node] = dynamic_node
            mod_data[:params][:state_component_node] = state_node
            
            return true

        elseif is_svc
            covariate_node = node1
            spatial_node = node2
            cov_args = get(covariate_node.args, :positional_args, [])
            if isempty(cov_args)
                @warn "SVC model is missing a covariate. Skipping."; return false
            end
            covariate_name = Symbol(cov_args[1])
            spatial_vars = get(spatial_node.args, :positional_args, [])
            
            process_random_module!(opt_dict, Dict(:type => :spatial,
                :params => spatial_node.args, :variables => spatial_vars), registries,
                hyperpriors)
            
            mod_data[:type] = :svc
            mod_data[:variables] = [covariate_name, spatial_vars...]
            mod_data[:params][:covariate] = covariate_name
            mod_data[:params][:positional_args] = spatial_vars
            mod_data[:params][:spatial_model_spec] = spatial_node
            
            inner_comp_type = Symbol(get(spatial_node.args, :model, :icar))
            inner_obj = resolve_technical_primitive(
                Dict(:key => "inner", :type => inner_comp_type, :params => spatial_node.args,
                    :variables => spatial_vars),
                opt_dict, hyperpriors, opt_dict[:prior_scheme]
            )
            mod_data[:params][:inner_model_obj] = inner_obj
            return true

        elseif is_tvc
            covariate_node = node1
            temporal_node = node2
            cov_args = get(covariate_node.args, :positional_args, [])
            if isempty(cov_args)
                @warn "TVC model is missing a covariate. Skipping."; return false
            end
            covariate_name = Symbol(cov_args[1])
            temporal_vars = get(temporal_node.args, :positional_args, [])

            process_random_module!(opt_dict, Dict(:type => :temporal,
                :params => temporal_node.args, :variables => temporal_vars), registries,
                hyperpriors)

            mod_data[:type] = :tvc
            mod_data[:variables] = [covariate_name, temporal_vars...]
            mod_data[:params][:covariate] = covariate_name
            mod_data[:params][:temporal_model_spec] = temporal_node
            
            inner_comp_type = Symbol(get(temporal_node.args, :model, :ar1))
            inner_obj = resolve_technical_primitive(
                Dict(:key => "inner", :type => inner_comp_type, :params => temporal_node.args,
                    :variables => temporal_vars),
                opt_dict, hyperpriors, opt_dict[:prior_scheme]
            )
            mod_data[:params][:inner_model_obj] = inner_obj
            return true
            
        elseif is_svar
            temporal_node = node1
            spatial_node = node2
            
            process_random_module!(opt_dict, Dict(:type => :temporal,
                :params => temporal_node.args, :variables => get(temporal_node.args,
                :positional_args, [])), registries, hyperpriors)

            process_random_module!(opt_dict, Dict(:type => :spatial,
                :params => spatial_node.args, :variables => get(spatial_node.args,
                :positional_args, [])), registries, hyperpriors)
            mod_data[:type] = :svar
            mod_data[:params][:rho_spatial_node] = spatial_node
            mod_data[:params][:base_temporal_node] = temporal_node
            return true
        end
    end



    if op == :kronecker_product

        if haskey(mod_data[:params], :sigma)

            prior_val = mod_data[:params][:sigma]

            calling_mod = get(opt_dict, :calling_module, Main)
            if prior_val isa Tuple
                opt_dict[:sigma_st_interaction_prior] = create_pc_prior(:sigma, prior_val)
                opt_dict[:st_interaction_sigma_prior] = opt_dict[:sigma_st_interaction_prior]
            elseif prior_val isa Expr
                opt_dict[:sigma_st_interaction_prior] = Core.eval(calling_mod, prior_val)
                opt_dict[:st_interaction_sigma_prior] = opt_dict[:sigma_st_interaction_prior]
            else
                opt_dict[:sigma_st_interaction_prior] = prior_val
                opt_dict[:st_interaction_sigma_prior] = prior_val
            end

        end

        if length(components) == 2
            for comp in components
                if comp.module_type == :random && !haskey(comp.args, :structure)
                    args_c = copy(comp.args); args_c[:vars] = get(comp.args, :positional_args, [])
                    comp.args[:structure] = _infer_structure_from_args(args_c)
                end
            end
            c1_type = get(components[1], :module_type, :unknown); c2_type = get(components[2],
                :module_type, :unknown)
            spatial_node = c1_type == :random && get(components[1].args, :structure,
                :none) == :spatial ? components[1] : (c2_type == :random && get(components[2].args,
                :structure, :none) == :spatial ? components[2] : nothing)
            temporal_node = c1_type == :random && get(components[1].args, :structure, :none) == :temporal ? components[1] : (c2_type == :random && get(components[2].args, :structure, :none) == :temporal ? components[2] : nothing)
            if !isnothing(spatial_node) && !isnothing(temporal_node)
                spatial_vars = get(spatial_node.args, :positional_args, [])
                temporal_vars = get(temporal_node.args, :positional_args, [])
                process_random_module!(opt_dict, Dict(:type => :spatial,
                    :params => spatial_node.args, :variables => spatial_vars), registries,
                    hyperpriors)
                process_random_module!(opt_dict, Dict(:type => :temporal,
                    :params => temporal_node.args, :variables => temporal_vars), registries,
                    hyperpriors)
                spatial_model_str = string(get(spatial_node.args, :model,
                    :iid)); temporal_model_str = string(get(temporal_node.args, :model, :iid))
                has_structured_space = spatial_model_str != "iid"; has_structured_time = temporal_model_str != "iid"
                if has_structured_space && has_structured_time
                    opt_dict[:model_st] = "IV"
                elseif !has_structured_space && has_structured_time
                    opt_dict[:model_st] = "II"
                elseif has_structured_space && !has_structured_time
                    opt_dict[:model_st] = "III"
                else opt_dict[:model_st] = "I"; end
            else
                throw(ArgumentError(
                    "Invalid Kronecker product (⊗) interaction: components must specify one " *
                    "spatial process and one temporal process (e.g., random(s, model=:bym2) ⊗ " *
                    "random(t, model=:ar1)). Received components of types ($(c1_type), $(c2_type))."
                ))
            end
            s_idx = get(opt_dict, :s_idx, nothing); t_idx = get(opt_dict, :t_idx,
                nothing); s_N = get(opt_dict, :s_N, nothing)
            if !isnothing(s_idx) && !isnothing(t_idx) && !isnothing(s_N)
                mod_data[:params][:indices] = [(t - 1) * s_N + s for (s, t) in zip(s_idx, t_idx)]
            end
            return true
        else
            @warn "Kronecker product with more than 2 components is not yet supported in process_interact_module!."
        end

    end

    
    return true
end




"""

    _process_mosaic_grouping!(opt_dict::Dict, mod_data::Dict)
Partitions the spatial domain into mosaic regions using k-means clustering or predefined
  cluster indices.
Returns a `NamedTuple` containing the grouping column name and number of regions.
"""

function _process_mosaic_grouping!(opt_dict, mod_data)
    s_N = get(opt_dict, :s_N, 0)
    if s_N == 0
        error("Mosaic models require a spatial context (`s_N`) to be established first. Ensure a spatial variable and adjacency matrix `W` are provided.")
    end

    data = opt_dict[:data]
    params = mod_data[:params]
    mosaic_param = get(params, :mosaic, :none)
    cluster_assignments::Vector{Int}
    n_regions::Int
    group_col_name = Symbol("mosaic_group_for_", mod_data[:key])
    if mosaic_param == :kmeans
        # --- K-Means Clustering Logic ---
        # This path uses spatial coordinates to perform clustering.
        if !haskey(opt_dict, :centroids)
            coord_cols = try
                _detect_xy_columns(data)
            catch
                nothing
            end
            s_idx_col = if hasproperty(data, :s_idx)
                :s_idx
            elseif haskey(opt_dict, :s_idx_var) && hasproperty(data, Symbol(opt_dict[:s_idx_var]))
                Symbol(opt_dict[:s_idx_var])
            else
                nothing
            end
            if coord_cols !== nothing && (s_idx_col !== nothing || haskey(opt_dict, :s_idx))
                cx, cy = coord_cols
                coord_map = Dict{Int, Point2D}()
                s_idx_vec = haskey(opt_dict, :s_idx) ? opt_dict[:s_idx] :
                            (s_idx_col !== nothing ? data[!, s_idx_col] : 1:nrow(data))
                for i in 1:nrow(data)
                    idx = s_idx_vec[i]
                    if !haskey(coord_map, idx)
                        coord_map[idx] = Point2D(data[i, cx], data[i, cy])
                    end
                end
                if length(coord_map) < s_N
                    error("Mosaic k-means requires coordinates for all $(s_N) spatial units, but only found $(length(coord_map)).")
                end
                opt_dict[:centroids] = [coord_map[i] for i in 1:s_N]
            else
                error("Mosaic k-means requires centroids or spatial coordinates in the data.")
            end
        end
        centroids = opt_dict[:centroids]
        if length(centroids) != s_N
            error("Number of centroids ($(length(centroids))) does not match s_N ($(s_N)).")
        end
        # Resolve n_regions, allowing it to be a symbol pointing to a variable.
        n_regions_raw = get(params, :n_regions, 5)
        n_regions_req::Int = if n_regions_raw isa Symbol
            calling_mod = get(opt_dict, :calling_module, Main)
            try
                Core.eval(calling_mod, n_regions_raw)
            catch e
                error("Could not evaluate `n_regions` variable `:$(n_regions_raw)`. Ensure it is defined. Error: $e")
            end
        elseif n_regions_raw isa Int
            n_regions_raw
        else
            error("`n_regions` must be an Integer or a Symbol pointing to an Integer. Got: $(typeof(n_regions_raw))")
        end
        if length(centroids) < n_regions_req
            @warn "Number of centroids ($(length(centroids))) is less than the requested number of regions ($n_regions_req). Adjusting n_regions to $(length(centroids))."
            n_regions_req = length(centroids)
        end
        centroids_matrix = hcat([c.x for c in centroids], [c.y for c in centroids])'
        kmeans_result = kmeans(centroids_matrix, n_regions_req; maxiter=200, display=:none)
        cluster_assignments = assignments(kmeans_result)
        n_regions = nclusters(kmeans_result)
    elseif mosaic_param isa Symbol
        # --- Pre-defined Grouping Column Logic (Enhanced Robustness) ---
        # This path uses a column in the DataFrame to define the spatial groups.
        if !hasproperty(data, mosaic_param)
            error("The specified mosaic grouping column `:$(mosaic_param)` was not found in the data.")
        end
        # Create a mapping from s_idx to the group value.
        s_idx_vec = haskey(opt_dict, :s_idx) ? opt_dict[:s_idx] :
                    (hasproperty(data, :s_idx) ? data.s_idx : 1:nrow(data))
        group_vals = data[!, mosaic_param]
        s_idx_to_group = Dict{Int, Any}()
        for i in 1:nrow(data)
            s_id = s_idx_vec[i]
            g_val = group_vals[i]
            if haskey(s_idx_to_group, s_id) && s_idx_to_group[s_id] != g_val
                error("Spatial unit `s_idx = $s_id` has multiple conflicting values in the grouping column `:$(mosaic_param)`. Each spatial unit must belong to exactly one group.")
            end
            s_idx_to_group[s_id] = g_val
        end
        # Map unique group levels to integers 1:n_regions.
        unique_group_levels = unique(group_vals)
        n_regions = length(unique_group_levels)
        level_to_int_map = Dict(level => i for (i, level) in enumerate(unique_group_levels))
        # Build the final cluster_assignments vector for all s_N units.
        cluster_assignments = Vector{Int}(undef, s_N)
        for i in 1:s_N
            if !haskey(s_idx_to_group, i)
                error("Spatial unit `s_idx = $i` is missing from the grouping column `:$(mosaic_param)`. All spatial units from 1 to `s_N` must have a defined group.")
            end
            cluster_assignments[i] = level_to_int_map[s_idx_to_group[i]]
        end
    else
        error("Invalid `mosaic` parameter. Must be `:kmeans` or a Symbol pointing to a grouping column.")
    end



    # Create the observation-level grouping column needed by the `mixed` processor.

    s_idx_lookup = haskey(opt_dict, :s_idx) ? opt_dict[:s_idx] :

                   (hasproperty(opt_dict[:data], :s_idx) ? opt_dict[:data].s_idx : 1:nrow(data))
    opt_dict[:data][!, group_col_name] = cluster_assignments[s_idx_lookup]
    

    return (group_col_name=group_col_name, n_regions=n_regions)

end



"""

    process_sciml_module!(opt_dict::Dict, mod_data::Dict, registries::Dict, hyperpriors::Dict)



Processes the `sciml()` module call, validating arguments and setting up temporal context.
This version is CPU-only.
"""
function process_sciml_module!(
    opt_dict::Dict, mod_data::Dict, registries::Dict, hyperpriors::Dict
)
    data = opt_dict[:data]
    params = mod_data[:params]

    variables = mod_data[:variables]

    calling_mod = get(opt_dict, :calling_module, Main)
    # 1. Set up temporal context from the time index variable.
    time_var_sym = if !isempty(variables)
        Symbol(variables[1])
    else
        det_t = _detect_time_column(data; allow_nothing=false)
        det_t
    end

    if !hasproperty(data, time_var_sym)
        error("Time index variable ':$time_var_sym' for sciml() module not found in data.")
    end

    time_opts = Dict(:time_method => get(params, :time_method, "regular"))
    tu_meta = assign_time_units(data[!, time_var_sym]; time_opts...)
    opt_dict[:t_idx] = tu_meta.idx
    opt_dict[:t_N] = tu_meta.N_cat
    opt_dict[:t_idx_var] = time_var_sym
    opt_dict[:t_coords] = data[!, time_var_sym] # Store original time coordinates for interpolation.
    # 2. Validate and evaluate all required SciML parameters.
    required_args = [:model_function, :initial_state_prior, :parameter_priors, :tspan, :solver]
    for arg in required_args
        if !haskey(params, arg)
            error("The `sciml()` module is missing the required keyword argument `:$arg`.")
        end
        raw_val = params[arg]
        try
            evaluated_val = Core.eval(calling_mod, raw_val)
            params[arg] = evaluated_val
        catch e
            error("Could not evaluate the `$(arg)` argument `$(raw_val)` for the sciml() module. Ensure it is defined in the calling scope. Error: $e")
        end
    end

    # 3. Evaluate optional SciML keyword arguments.
    optional_args = [:saveat, :solver_kwargs]
    for arg in optional_args
        if haskey(params, arg)
            raw_val = params[arg]
            if raw_val isa Expr || raw_val isa Symbol
                try
                    params[arg] = Core.eval(calling_mod, raw_val)
                catch e
                    @warn "Could not evaluate `$(arg)` argument for sciml() module. Using default. Error: $e"
                    if arg == :solver_kwargs
                        params[arg] = Dict()
                    end
                end
            end
        end
    end

    # 4. Store necessary evaluated parameters in the main model configuration.
    opt_dict[:sciml_solver] = params[:solver]
    opt_dict[:sciml_tspan] = params[:tspan]
    opt_dict[:sciml_saveat] = get(params, :saveat, 0.1) # Default saveat
    # 5. Create and store a problem template.
    initial_state_prior = params[:initial_state_prior]
    parameter_priors = params[:parameter_priors]
    u0_mean = mean(initial_state_prior)
    u0_placeholder = u0_mean isa Number ? [u0_mean] : vec(u0_mean)
    p_placeholder = [mean(p) for p in parameter_priors]
    prob_func = getfield(calling_mod, params[:model_function])
    solver_kwargs = get(params, :solver_kwargs, Dict())
    prob_template = ODEProblem(prob_func, u0_placeholder, params[:tspan], p_placeholder;
        solver_kwargs...)
    if !haskey(opt_dict, :sciml_problem_templates)
        opt_dict[:sciml_problem_templates] = Dict{Symbol, Any}()
    end

    opt_dict[:sciml_problem_templates][Symbol(mod_data[:key])] = prob_template



    return true
end



"""
    process_dynamics_module!(opt_dict::Dict, mod_data::Dict, registries::Dict, hyperpriors::Dict)
Processes the `dynamics()` module, ensuring spatial and temporal contexts are established.
  self-contained and consistent with the refactored
architecture. It no longer calls deprecated processors. Instead, it directly
handles the setup of spatial and temporal indices from its own arguments, resolves
the adjacency matrix `W`, and validates all necessary parameters for the specified
mechanistic model.
# Arguments
- `opt_dict`: The main model configuration dictionary (`M`).
- `mod_data`: The parsed data for the `dynamics()` module.
- `registries`, `hyperpriors`: Additional configuration dictionaries.
# Returns
- `true` to indicate that a `Dynamics` component object should be created.
"""
function process_dynamics_module!(
    opt_dict::Dict, mod_data::Dict, registries::Dict, hyperpriors::Dict
)
    params = mod_data[:params]
    data = opt_dict[:data]
    variables = mod_data[:variables] # Positional arguments from the formula, e.g., s_idx, year
    # 1. Validate and set up spatial and temporal indices from formula arguments.
    if length(variables) < 2
        s_var = _detect_spatial_unit_column(data; allow_nothing=true)
        t_var = _detect_time_column(data; allow_nothing=true)
        if !isnothing(s_var) && !isnothing(t_var)
            spatial_idx_var = s_var
            temporal_idx_var = t_var
        else
            error(
                "The `dynamics()` module requires a spatial and temporal index " *
                "(e.g., `dynamics(district, year, ...)`), or detectable columns in data."
            )
        end
    else
        spatial_idx_var = Symbol(variables[1])
        temporal_idx_var = Symbol(variables[2])
    end
    # Resolve spatial index and map to 1-based integer indices 1:s_N
    if !hasproperty(data, spatial_idx_var)
        error("Spatial index variable ':$spatial_idx_var' for dynamics module not found in data.")
    end
    raw_s = data[!, spatial_idx_var]
    if any(ismissing, raw_s)
        error("Spatial index variable ':$spatial_idx_var' contains missing values.")
    end
    unique_units = if applicable(levels, raw_s) && !isempty(levels(raw_s))
        filter(in(unique(raw_s)), levels(raw_s))
    else
        sort(unique(raw_s))
    end
    s_N = length(unique_units)
    s_idx_vec = if eltype(raw_s) <: Integer && minimum(raw_s) == 1 &&
                   maximum(raw_s) == s_N
        Int.(raw_s)
    else
        unit_to_idx = Dict(u => i for (i, u) in enumerate(unique_units))
        [unit_to_idx[u] for u in raw_s]
    end
    opt_dict[:s_idx] = s_idx_vec
    opt_dict[:s_N] = s_N
    opt_dict[:s_idx_var] = spatial_idx_var
    opt_dict[:s_values] = unique_units
    if !hasproperty(opt_dict[:data], :s_idx)
        opt_dict[:data][!, :s_idx] = s_idx_vec
    end
    # Resolve temporal index and its number of levels
    if !hasproperty(data, temporal_idx_var)
        error("Temporal index variable ':$temporal_idx_var' for dynamics module not found in data.")
    end
    if any(ismissing, data[!, temporal_idx_var])
        error("Temporal index variable ':$temporal_idx_var' contains missing values.")
    end
    time_opts = Dict(:time_method => get(params, :time_method, "regular"))
    tu_meta = assign_time_units(data[!, temporal_idx_var]; time_opts...)
    opt_dict[:t_idx] = tu_meta.idx
    opt_dict[:t_N] = tu_meta.N_cat
    opt_dict[:t_idx_var] = temporal_idx_var
    opt_dict[:t_values] = tu_meta.mids
    if !hasproperty(opt_dict[:data], :t_idx)
        opt_dict[:data][!, :t_idx] = tu_meta.idx
    end
    # 2. Resolve adjacency matrix `W`.
    # Prioritize `W` from the module's parameters, then fallback to global opt_dict.
    if haskey(params, :W)
        w_val = params[:W]
        if w_val isa Expr || w_val isa Symbol
            calling_mod = get(opt_dict, :calling_module, Main)
            try
                opt_dict[:W] = Core.eval(calling_mod, w_val)
            catch e
                error("Could not evaluate `W` argument `$(w_val)` for dynamics module. Error: $e")
            end
        else
            opt_dict[:W] = w_val
        end
    end
    if !haskey(opt_dict, :W)
        error("Dynamics models require an adjacency matrix `W`, passed either to the module (e.g., `dynamics(..., W=my_W)`) or as a keyword argument to the main `@tbm` call (e.g., `@tbm(..., W=my_W)`).")
    end
    if opt_dict[:s_N] != size(opt_dict[:W], 1)
        error("Number of unique spatial indices ($(opt_dict[:s_N])) does not match the dimension of the provided adjacency matrix W ($(size(opt_dict[:W], 1))).")
    end
    # 3. Model Type Verification
    model_type = string(get(params, :model, "none"))
    if model_type == "none"
        error("Dynamics module requires a 'model' parameter (e.g., model='advection').")
    end
    # 4. Covariate and Parameter Validation
    if model_type in ["advection", "advection_diffusion"]
        if !haskey(params, :velocity_prior) && !haskey(opt_dict[:hyperpriors], "velocity")
            @warn "Advection model specified without explicit velocity priors. Using system defaults."
        end
    end
    if model_type in ["diffusion", "advection_diffusion"]
        if !haskey(params, :diffusion_prior) && !haskey(opt_dict[:hyperpriors], "diffusion")
            @warn "Diffusion model specified without explicit diffusion priors. Using system defaults."
        end
    end
    # 5. Mapping Spatiotemporal State
    # We pre-calculate the spatiotemporal flat index (st_idx) to allow the
    # code generator to map the [s_N, t_N] state matrix to the observation vector N.
    s_idx = opt_dict[:s_idx]
    t_idx = opt_dict[:t_idx]
    s_N = opt_dict[:s_N]
    opt_dict[:st_idx] = [(t_val - 1) * s_N + s_val for (s_val, t_val) in zip(s_idx, t_idx)]
    return true
end
const MODULE_PROCESSORS = Dict{Symbol, Function}(
    :random => process_random_module!,
    :fixed => process_fixed_module!,
    :mixed => process_mixed_module!,
    :nested => process_nested_module!,
    :transfer => process_nested_module!,
    :fidelity => process_nested_module!,
    :eigen => process_eigen_module!,
    :dynamics => process_dynamics_module!,
    :interact => process_interact_module!,
    :custom => process_custom_module!,
    :sciml => process_sciml_module!
)
