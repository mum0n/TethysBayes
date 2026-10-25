# ==============================================================================
# TethysBayes HIERARCHICAL WORKFLOW & PROVENANCE ENGINE
# ==============================================================================
# Generic Infrastructure for Multi-Tier Hierarchical Spatiotemporal Pipelines:
# - Data Trait Fingerprinting & Change-Detection
# - Relational Provenance Manifest (DuckDB)
# - Smart Caching & Upstream Dependency Invalidation
# - Pipeline Status Inspection & Restart Recommendations
# - Zero-Copy Relational I/O & Geometric Mapping
# ==============================================================================

"""
    compute_data_traits(df::DataFrame; temporal_col=nothing, spatial_cols=(:s_x, :s_y)) -> Dict{Symbol, Any}

Computes a deterministic metadata and statistical fingerprint for a dataset,
capturing row/column dimensions, spatial bounding box envelopes, temporal horizons,
and content summary hashes.

# Mathematical Formulation
The bounding box is evaluated as:

\$\$
\\mathcal{B}(D) = \\left[ \\min(x), \\max(x) \\right] \\times \\left[ \\min(y), \\max(y) \\right]
\$\$

The content hash \$h(D)\$ combines the schema dimensions and summary statistics
of all column vectors to detect mutations in raw survey observations.

# Arguments
- `df::DataFrame`: Input dataset.
- `temporal_col::Union{Symbol, Nothing}`: Column name for temporal indexing.
- `spatial_cols::Tuple{Symbol, Symbol}`: Coordinates for spatial bounding box.

# Returns
- `Dict{Symbol, Any}`: Dictionary containing dimensions, bounding box, temporal range, and hash.
"""
function compute_data_traits(
    df::DataFrame;
    temporal_col::Union{Symbol, AbstractString, Nothing} = nothing,
    spatial_cols::Union{Tuple{Symbol, Symbol}, Vector{Symbol}, Nothing} = nothing
)
    traits = Dict{Symbol, Any}(
        :rows => nrow(df),
        :cols => ncol(df)
    )
    resolved_spatial = if !isnothing(spatial_cols)
        (Symbol(spatial_cols[1]), Symbol(spatial_cols[2]))
    else
        try
            _detect_xy_columns(df)
        catch
            nothing
        end
    end

    if resolved_spatial !== nothing && hasproperty(df, resolved_spatial[1]) &&
       hasproperty(df, resolved_spatial[2])
        traits[:min_x] = round(minimum(df[!, resolved_spatial[1]]), digits=4)
        traits[:max_x] = round(maximum(df[!, resolved_spatial[1]]), digits=4)
        traits[:min_y] = round(minimum(df[!, resolved_spatial[2]]), digits=4)
        traits[:max_y] = round(maximum(df[!, resolved_spatial[2]]), digits=4)
    end

    resolved_time = if !isnothing(temporal_col)
        Symbol(temporal_col)
    else
        try
            _detect_time_column(df)
        catch
            nothing
        end
    end

    if resolved_time !== nothing && hasproperty(df, resolved_time)
        traits[:min_time] = minimum(df[!, resolved_time])
        traits[:max_time] = maximum(df[!, resolved_time])
        traits[:n_times] = length(unique(df[!, resolved_time]))
    end

    
    # Deterministic hash based on dimensions and column summaries
    h_val = hash(nrow(df))
    for col in names(df)
        h_val = hash(col, hash(summary(df[!, col]), h_val))
    end
    traits[:hash] = string(h_val, base=16)
    return traits
end

"""
    map_to_units(xs::AbstractVector, ys::AbstractVector, centroids::Vector) -> Vector{Int}

Maps arbitrary 2D observation coordinates \$(x_i, y_i)\$ to their nearest discrete
areal spatial unit centroid via Euclidean distance minimization:

\$\$
u(x_i, y_i) = \\arg\\min_{s \\in \\{1, \\dots, S\\}} \\| (x_i, y_i) - \\mathbf{c}_s \\|_2
\$\$

# Arguments
- `xs::AbstractVector`: Vector of X coordinates.
- `ys::AbstractVector`: Vector of Y coordinates.
- `centroids::Vector`: Vector of unit centroid tuples or vectors `(c_x, c_y)`.

# Returns
- `Vector{Int}`: Indices of the nearest spatial units (1-indexed).
"""
function map_to_units(xs::AbstractVector, ys::AbstractVector, centroids::Vector)
    return [argmin([hypot(x - c[1], y - c[2]) for c in centroids]) for (x, y) in zip(xs, ys)]
end

"""
    init_pipeline_manifest!(db_path::AbstractString)

Initializes the relational provenance manifest table `pipeline_manifest` in DuckDB
if it does not already exist.

# Schema
- `tier_id` (`VARCHAR PRIMARY KEY`): Unique tier identifier (e.g. `"tier1_depth"`).
- `tier_name` (`VARCHAR`): Human-readable tier title.
- `status` (`VARCHAR`): Execution status (`"COMPLETED"`, `"STALE"`, `"FAILED"`).
- `update_frequency` (`VARCHAR`): Schedule (e.g. `"Decadal"`, `"Annual"`).
- `last_run` (`VARCHAR`): UTC ISO timestamp of execution.
- `data_hash` (`VARCHAR`): Fingerprint hash of input data.
- `data_rows` (`BIGINT`): Input record count.
- `data_traits` (`VARCHAR`): Serialized traits dictionary.
- `upstream_deps` (`VARCHAR`): Comma-separated list of upstream prerequisite tiers.
- `bundle_path` (`VARCHAR`): File path to JLD2 model bundle.
- `table_name` (`VARCHAR`): Table name of predictions in DuckDB.
"""
function _with_duckdb(f::Function, db_path::AbstractString)
    db = DuckDB.DB(db_path)
    con = DuckDB.connect(db)
    try
        return f(con)
    finally
        try DuckDB.disconnect(con) catch end
        try DuckDB.close(db) catch end
        GC.gc()
    end
end

"""
    init_pipeline_manifest!(db_path::AbstractString)

Initializes the metadata and provenance manifest table in DuckDB if not present.
"""
function init_pipeline_manifest!(db_path::AbstractString)
    _with_duckdb(db_path) do con
        DuckDB.query(con, """
            CREATE TABLE IF NOT EXISTS pipeline_manifest (
                tier_id VARCHAR PRIMARY KEY,
                tier_name VARCHAR,
                status VARCHAR,
                update_frequency VARCHAR,
                last_run VARCHAR,
                data_hash VARCHAR,
                data_rows BIGINT,
                data_traits VARCHAR,
                upstream_deps VARCHAR,
                bundle_path VARCHAR,
                table_name VARCHAR
            )
        """)
    end
end

"""
    write_tier_table!(db_path::AbstractString, df::DataFrame, tbl_name::AbstractString)

Writes or replaces a DataFrame into the relational DuckDB project database with
automatic connection cleanup and garbage collection.
"""
function write_tier_table!(db_path::AbstractString, df::DataFrame, tbl_name::AbstractString)
    _with_duckdb(db_path) do con
        _write_df_to_duckdb(con, df, tbl_name, true)
    end
end

"""
    read_tier_table(db_path::AbstractString, tbl_name::AbstractString) -> DataFrame

Reads a table from the relational DuckDB project database into a Julia `DataFrame`.
"""
function read_tier_table(db_path::AbstractString, tbl_name::AbstractString)
    return _with_duckdb(db_path) do con
        res = DuckDB.query(con, "SELECT * FROM $tbl_name")
        DataFrame(res)
    end
end

"""
    has_tier_table(db_path::AbstractString, tbl_name::AbstractString) -> Bool

Checks whether a table exists in the DuckDB project database.
"""
function has_tier_table(db_path::AbstractString, tbl_name::AbstractString)
    if !isfile(db_path)
        return false
    end
    return _with_duckdb(db_path) do con
        df = DataFrame(DuckDB.query(con, 
            "SELECT table_name FROM information_schema.tables WHERE table_name = '$tbl_name'"))   
        nrow(df) > 0
    end
end

"""
    get_manifest_entry(db_path::AbstractString, tier_id::AbstractString) -> Union{DataFrameRow, Nothing}

Retrieves the manifest record for a given tier from the DuckDB project database.
"""
function get_manifest_entry(db_path::AbstractString, tier_id::AbstractString)
    if !isfile(db_path)
        return nothing
    end
    init_pipeline_manifest!(db_path)
    return _with_duckdb(db_path) do con
        df = DataFrame(DuckDB.query(con, 
            "SELECT * FROM pipeline_manifest WHERE tier_id = '$tier_id'"))    
        nrow(df) > 0 ? df[1, :] : nothing
    end
end

"""
    update_manifest_entry!(db_path::AbstractString; tier_id, tier_name, status, update_frequency, traits, upstream_deps, bundle_path, table_name)

Updates or registers a tier's provenance and execution record into the DuckDB manifest.
"""
function update_manifest_entry!(
    db_path::AbstractString;
    tier_id::AbstractString,
    tier_name::AbstractString,
    status::AbstractString,
    update_frequency::AbstractString,
    traits::Dict{Symbol, Any},
    upstream_deps::Vector{String},
    bundle_path::AbstractString,
    table_name::AbstractString
)
    init_pipeline_manifest!(db_path)
    now_str = Dates.format(Dates.now(Dates.UTC), "yyyy-mm-dd HH:mm:ss") * " UTC"
    traits_str = repr(traits)
    traits_escaped = replace(traits_str, "'" => "''")
    deps_str = join(upstream_deps, ",")
    data_hash = get(traits, :hash, "")
    data_rows = Int64(get(traits, :rows, 0))

    _with_duckdb(db_path) do con
        DuckDB.query(con, "DELETE FROM pipeline_manifest WHERE tier_id = '$tier_id'")
        DuckDB.query(con, """
            INSERT INTO pipeline_manifest VALUES (
                '$tier_id', '$tier_name', '$status', '$update_frequency', '$now_str', 
                '$data_hash', $data_rows, '$traits_escaped', '$deps_str', '$bundle_path', '$table_name'
            )
        """)
    end
end



"""
    is_tier_up_to_date(db_path::AbstractString, tier_id::AbstractString, current_traits::Dict{Symbol, Any}, upstream_deps::Vector{String}) -> Bool

Validates whether a tier's serialized model and DuckDB results are valid and up-to-date
by checking file presence, data trait hash consistency, and upstream execution timestamps.
"""
function is_tier_up_to_date(
    db_path::AbstractString,
    tier_id::AbstractString,
    current_traits::Dict{Symbol, Any},
    upstream_deps::Vector{String}
)
    entry = get_manifest_entry(db_path, tier_id)
    if entry === nothing || entry.status != "COMPLETED"
        return false
    end
    
    # Verify bundle file exists on disk
    bundle_jld2 = entry.bundle_path * ".jld2"
    if !isfile(bundle_jld2)
        return false
    end

    # Verify input data hash matches
    if entry.data_hash != get(current_traits, :hash, "")
        return false
    end

    # Verify upstream dependencies were not modified after this tier's run
    for dep in upstream_deps
        dep_entry = get_manifest_entry(db_path, dep)
        if dep_entry === nothing || dep_entry.status != "COMPLETED"
            return false
        end
        if dep_entry.last_run > entry.last_run
            return false
        end
    end

    return true
end

"""
    check_pipeline_status(db_path::AbstractString, tiers_spec::Vector{<:NamedTuple}; verbose=true) -> DataFrame

Inspects the entire multi-tier hierarchy against the relational manifest,
evaluates timestamps and data trait fingerprints, and generates actionable restart recommendations.

# Arguments
- `db_path::AbstractString`: Path to DuckDB database.
- `tiers_spec::Vector`: List of NamedTuples `(id, name, freq, traits, deps)`.
- `verbose::Bool`: Whether to print an ASCII summary table.

# Returns
- `DataFrame`: Table of tier statuses and action recommendations.
"""
function check_pipeline_status(db_path::AbstractString, tiers_spec::Vector{<:NamedTuple}; verbose=true)
    init_pipeline_manifest!(db_path)
    
    records = DataFrame(
        tier_id = String[],
        tier_name = String[],
        update_frequency = String[],
        status = String[],
        last_run = String[],
        data_rows = Int64[],
        recommendation = String[]
    )

    for t in tiers_spec
        up_to_date = is_tier_up_to_date(db_path, t.id, t.traits, t.deps)
        entry = get_manifest_entry(db_path, t.id)
        last_run_str = entry !== nothing ? entry.last_run : "Never"
        rows_cnt = Int64(get(t.traits, :rows, 0))
        
        status_str = up_to_date ? "CACHED" : (entry !== nothing ? "STALE" : "NOT_RUN")
        rec_str = up_to_date ? "[SKIP] Up-to-date (cached)" : (entry !== nothing ? "[RUN] Data/Dep Changed" : "[RUN] Initial Execution Required")
        
        push!(records, (
            tier_id = t.id,
            tier_name = t.name,
            update_frequency = t.freq,
            status = status_str,
            last_run = last_run_str,
            data_rows = rows_cnt,
            recommendation = rec_str
        ))
    end

    if verbose
        println("\n" * "="^95)
        println("TethysBayes HIERARCHICAL WORKFLOW: STATUS & RESTART RECOMMENDATIONS")
        println("="^95)
        @printf("%-28s %-22s %-10s %-22s %-20s\n", "Tier / Segment", "Update Schedule", "Status", "Last Run (UTC)", "Action Recommendation")
        println("-"^95)
        for row in eachrow(records)
            @printf("%-28s %-22s %-10s %-22s %-20s\n", row.tier_name, row.update_frequency, row.status, row.last_run, row.recommendation)
        end
        println("="^95 * "\n")
    end

    return records
end
