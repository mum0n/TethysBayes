#!/usr/bin/env julia
# ==============================================================================
# TethysBayes Command-Line Interface (CLI) Runner: scripts/TethysBayesr.jl
#
# Parses declarative text configurations (TOML) to instantiate, sample, and
# persist Bayesian Spatio-Temporal Models (@tbm) to JLD2 or DuckDB.
#
# Usage:
#   julia --project=. scripts/TethysBayesr.jl <config.toml> [options]
#   julia --project=. scripts/TethysBayesr.jl --config=<config.toml> [options]
#   julia --project=. scripts/TethysBayesr.jl --help
# ==============================================================================

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))

using TethysBayes
using TOML
using DataFrames
using Dates
using Printf
using Turing
using DynamicPPL
using DuckDB
using JLD2

# Load optional RData support if available
try
    using RData
catch
end

"""
    print_cli_banner()

Prints formatted terminal header with version and usage information.
"""
function print_cli_banner()
    println("="^80)
    println("  TethysBayes CLI Model Runner: TethysBayesr")
    println("  Bayesian Spatio-Temporal Modeling & Analytical Persistence Engine")
    println("="^80)
end

"""
    print_cli_help()

Displays command-line usage syntax, configuration options, and examples.
"""
function print_cli_help()
    print_cli_banner()
    println("""
Usage:
  julia --project=. scripts/TethysBayesr.jl <config.toml> [options]
  julia --project=. scripts/TethysBayesr.jl --config=<config.toml> [options]

Options:
  -c, --config=<file>     Path to TOML configuration file
  -o, --output=<path>     Override output file or directory path (.jld2 or .duckdb)
  -f, --format=<fmt>      Override output format: 'jld2', 'duckdb', or 'both'
  -s, --samples=<N>       Override number of posterior MCMC samples
  -w, --warmup=<N>        Override number of warmup / adaptation steps
  --sampler=<name>        Override sampler: 'optimal', 'nuts', 'gibbs', 'mh'
  --no-sample             Instantiate and validate model without running MCMC
  --dry-run               Validate configuration and model without saving
  -v, --verbose           Enable detailed runtime logging
  -h, --help              Display this help guide and exit

Example:
  julia --project=. scripts/TethysBayesr.jl scripts/config_bathymetry_rff.toml
  julia --project=. scripts/TethysBayesr.jl --config=model.toml --format=duckdb
""")
end

"""
    parse_cli_args(args::Vector{String})::Dict{Symbol, Any}

Parses command-line arguments into a typed dictionary of flags and parameters.
"""
function parse_cli_args(args::Vector{String})::Dict{Symbol, Any}
    parsed = Dict{Symbol, Any}(
        :config_file => nothing,
        :output_path => nothing,
        :format => nothing,
        :n_samples => nothing,
        :n_warmup => nothing,
        :sampler => nothing,
        :sample => nothing,
        :dry_run => false,
        :verbose => nothing,
        :help => false
    )

    for arg in args
        if arg in ["-h", "--help"]
            parsed[:help] = true
            return parsed
        elseif startswith(arg, "--config=")
            parsed[:config_file] = String(Base.split(arg, "=", limit=2)[2])
        elseif startswith(arg, "-c=")
            parsed[:config_file] = String(Base.split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--output=")
            parsed[:output_path] = String(Base.split(arg, "=", limit=2)[2])
        elseif startswith(arg, "-o=")
            parsed[:output_path] = String(Base.split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--format=")
            parsed[:format] = String(Base.split(arg, "=", limit=2)[2])
        elseif startswith(arg, "-f=")
            parsed[:format] = String(Base.split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--samples=")
            parsed[:n_samples] = parse(Int, Base.split(arg, "=", limit=2)[2])
        elseif startswith(arg, "-s=")
            parsed[:n_samples] = parse(Int, Base.split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--warmup=")
            parsed[:n_warmup] = parse(Int, Base.split(arg, "=", limit=2)[2])
        elseif startswith(arg, "-w=")
            parsed[:n_warmup] = parse(Int, Base.split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--sampler=")
            parsed[:sampler] = String(Base.split(arg, "=", limit=2)[2])
        elseif arg == "--no-sample"
            parsed[:sample] = false
        elseif arg == "--dry-run"
            parsed[:dry_run] = true
        elseif arg in ["-v", "--verbose"]
            parsed[:verbose] = true
        elseif !startswith(arg, "-") && isnothing(parsed[:config_file])
            parsed[:config_file] = arg
        end
    end

    return parsed
end

"""
    resolve_file_path(path::AbstractString, base_dir::AbstractString)::String

Resolves a file path relative to current working directory or configuration directory.
"""
function resolve_file_path(path::AbstractString, base_dir::AbstractString)::String
    if isabspath(path)
        return path
    end
    if isfile(path)
        return abspath(path)
    end
    alt_path = joinpath(base_dir, path)
    if isfile(alt_path)
        return abspath(alt_path)
    end
    return abspath(path)
end

"""
    load_user_libraries(packages::Vector{String})

Dynamically loads user-specified Julia packages required for data reading,
transformation, or model preparation. If a package is not yet installed in
the current environment, an informative error is raised.
"""
function load_user_libraries(packages::Vector{String})
    for pkg_name in packages
        pkg_trimmed = strip(pkg_name)
        isempty(pkg_trimmed) && continue
        pkg_sym = Symbol(pkg_trimmed)
        println("[Environment] Loading user library: ", pkg_sym)
        try
            @eval Main using $(pkg_sym)
        catch err
            error("Failed to load user library '$(pkg_name)': $(err)\n" *
                  "Ensure '$(pkg_name)' is installed in the active environment via `Pkg.add(\"$(pkg_name)\")`.")
        end
    end
end

"""
    load_input_dataset(cfg::Dict{String, Any}, config_dir::AbstractString)::DataFrame

Loads a tabular dataset from various formats (.rda, .jld2, .csv, .duckdb).
"""
function load_input_dataset(cfg::Dict{String, Any}, config_dir::AbstractString)::DataFrame
    data_cfg = get(cfg, "data", Dict{String, Any}())
    data_path_raw = get(data_cfg, "path", get(data_cfg, "file", nothing))

    if isnothing(data_path_raw)
        error("Missing required configuration parameter: `[data].path`")
    end

    data_path = resolve_file_path(String(data_path_raw), config_dir)
    if !isfile(data_path)
        error("Input data file not found at: $(data_path)")
    end

    key = get(data_cfg, "key", nothing)
    explicit_fmt = lowercase(get(data_cfg, "format", "auto"))

    fmt = if explicit_fmt != "auto"
        explicit_fmt
    elseif endswith(lowercase(data_path), ".rda") || endswith(lowercase(data_path), ".rdata")
        "rda"
    elseif endswith(lowercase(data_path), ".jld2")
        "jld2"
    elseif endswith(lowercase(data_path), ".csv")
        "csv"
    elseif endswith(lowercase(data_path), ".duckdb") || endswith(lowercase(data_path), ".db")
        "duckdb"
    else
        "unknown"
    end

    println("[Data] Loading dataset from: ", data_path, " (format: ", fmt, ")")

    if fmt == "rda"
        try
            # Ensure RData is accessible in Main if requested
            if !isdefined(Main, :RData)
                @eval Main using RData
            end
            loaded = Base.invokelatest(Main.RData.load, data_path)
            target_key = !isnothing(key) ? String(key) : first(keys(loaded))
            if !haskey(loaded, target_key)
                error("Key '$(target_key)' not found in RDA file. Available: $(collect(keys(loaded)))")
            end
            df = DataFrame(loaded[target_key])
            println("[Data] Extracted object '", target_key, "' with size: ", size(df))
            return df
        catch err
            error("Failed to load RDA data file: $(err)")
        end

    elseif fmt == "jld2"
        try
            loaded = JLD2.load(data_path)
            target_key = !isnothing(key) ? String(key) : first(keys(loaded))
            if !haskey(loaded, target_key)
                error("Key '$(target_key)' not found in JLD2. Available: $(collect(keys(loaded)))")
            end
            obj = loaded[target_key]
            df = obj isa DataFrame ? obj : DataFrame(obj)
            println("[Data] Extracted JLD2 object '", target_key, "' with size: ", size(df))
            return df
        catch err
            error("Failed to load JLD2 data file: $(err)")
        end

    elseif fmt == "duckdb"
        try
            table_name = get(data_cfg, "table", "data")
            sql_query = get(data_cfg, "query", "SELECT * FROM $(table_name)")
            df = TethysBayes.query_duckdb(data_path, sql_query)
            println("[Data] Queried DuckDB table with size: ", size(df))
            return df
        catch err
            error("Failed to query DuckDB file: $(err)")
        end

    elseif fmt == "csv"
        try
            # Fast zero-dependency CSV read via DuckDB's native parser
            db = DuckDB.DB()
            con = DuckDB.connect(db)
            escaped_path = replace(data_path, "\\" => "/")
            query = "SELECT * FROM read_csv_auto('$(escaped_path)')"
            df = DataFrame(DuckDB.query(con, query))
            DuckDB.disconnect(con)
            DuckDB.close(db)
            println("[Data] Loaded CSV with size: ", size(df))
            return df
        catch err
            error("Failed to parse CSV file: $(err)")
        end

    else
        error("Unsupported dataset format: '$(fmt)'. Expected .rda, .jld2, .csv, or .duckdb.")
    end
end

"""
    apply_preprocessing(df::DataFrame, cfg::Dict{String, Any}, config_dir::AbstractString)::DataFrame

Executes user-defined preprocessing operations, scripts, or inline code to transform
the input DataFrame prior to model compilation and sampling.
"""
function apply_preprocessing(df::DataFrame, cfg::Dict{String, Any}, config_dir::AbstractString)::DataFrame
    prep_cfg = get(cfg, "preprocessing", Dict{String, Any}())
    if isempty(prep_cfg)
        return df
    end

    processed_df = copy(df)

    # 1. Execute external user-defined preprocessing script if provided
    if haskey(prep_cfg, "script")
        script_path_raw = String(prep_cfg["script"])
        script_file = resolve_file_path(script_path_raw, config_dir)
        if !isfile(script_file)
            error("Preprocessing script file not found at: $(script_file)")
        end
        println("[Preprocessing] Running user script: ", script_file)
        # Expose df to Main before running script
        @eval Main df_input = $(processed_df)
        Base.include(Main, script_file)
        if isdefined(Main, :df_output)
            processed_df = Main.df_output
        elseif isdefined(Main, :df_input)
            processed_df = Main.df_input
        end
    end

    # 2. Execute custom transform function if named
    if haskey(prep_cfg, "transform")
        transform_name = Symbol(prep_cfg["transform"])
        println("[Preprocessing] Applying transform function: ", transform_name)
        if isdefined(Main, transform_name)
            func = getfield(Main, transform_name)
            processed_df = Base.invokelatest(func, processed_df)
        else
            error("Preprocessing transform function '$(transform_name)' is not defined in Main.")
        end
    end

    # 3. Execute inline code if provided
    if haskey(prep_cfg, "code")
        code_str = String(prep_cfg["code"])
        println("[Preprocessing] Evaluating inline preprocessing code...")
        @eval Main df = $(processed_df)
        res_code = Base.include_string(Main, code_str)
        if res_code isa DataFrame
            processed_df = res_code
        elseif isdefined(Main, :df) && Main.df isa DataFrame
            processed_df = Main.df
        end
    end

    println("[Preprocessing] Completed preprocessing. Final data dimensions: ", size(processed_df))
    return processed_df
end

"""
    main()

Entry point for the TethysBayes command-line interface.
"""
function main()
    cli_args = parse_cli_args(ARGS)

    if cli_args[:help] || (isnothing(cli_args[:config_file]) && isempty(ARGS))
        print_cli_help()
        return 0
    end

    config_file = cli_args[:config_file]
    if isnothing(config_file) || !isfile(config_file)
        print_cli_banner()
        println("ERROR: Configuration file not found: $(config_file)")
        println("Run with --help for usage instructions.")
        return 1
    end

    print_cli_banner()
    config_dir = dirname(abspath(config_file))
    println("[Config] Reading configuration from: ", abspath(config_file))
    cfg = TOML.parsefile(config_file)

    # 1. Extract metadata and model configuration
    meta_cfg = get(cfg, "metadata", Dict{String, Any}())
    model_name = get(meta_cfg, "name", "TethysBayes_model")
    description = get(meta_cfg, "description", "TethysBayes Model Execution")
    println("[Meta] Model: ", model_name, " | ", description)

    model_cfg = get(cfg, "model", Dict{String, Any}())
    formula_str = get(model_cfg, "formula", nothing)
    if isnothing(formula_str)
        error("Missing required parameter: `[model].formula`")
    end
    println("[Model] Formula: ", formula_str)

    verbose_raw = cli_args[:verbose] !== nothing ? cli_args[:verbose] : get(model_cfg, "verbose", false)
    verbose_opt = verbose_raw isa Bool ? verbose_raw : parse(Bool, string(verbose_raw))
    use_gpu_raw = get(model_cfg, "use_gpu", false)
    use_gpu_opt = use_gpu_raw isa Bool ? use_gpu_raw : parse(Bool, string(use_gpu_raw))
    prior_scheme_raw = get(model_cfg, "prior_scheme", "pcpriors")
    prior_scheme_sym = Symbol(prior_scheme_raw)

    # 2. Load user-specified libraries (environment configuration)
    env_cfg = get(cfg, "environment", Dict{String, Any}())
    prep_cfg = get(cfg, "preprocessing", Dict{String, Any}())
    user_pkgs = String[]
    if haskey(env_cfg, "packages")
        append!(user_pkgs, [String(p) for p in env_cfg["packages"]])
    end
    if haskey(prep_cfg, "packages")
        append!(user_pkgs, [String(p) for p in prep_cfg["packages"]])
    end
    if !isempty(user_pkgs)
        load_user_libraries(unique(user_pkgs))
    end

    # 3. Load dataset
    raw_df = load_input_dataset(cfg, config_dir)

    # 4. User data preprocessing & transformation
    df = apply_preprocessing(raw_df, cfg, config_dir)

    # 5. Optional areal units or spatial graph matrix
    au_obj = nothing
    data_cfg = get(cfg, "data", Dict{String, Any}())
    if haskey(data_cfg, "au_path")
        au_file = resolve_file_path(String(data_cfg["au_path"]), config_dir)
        if isfile(au_file)
            au_loaded = JLD2.load(au_file)
            au_obj = haskey(au_loaded, "au") ? au_loaded["au"] : first(values(au_loaded))
            println("[Data] Loaded spatial areal units from: ", au_file)
        end
    end

    # 6. Model instantiation
    println("[Model] Instantiating TethysBayes dynamic model...")
    t_start_inst = time()
    model = TethysBayes.TethysBayes_core(
        formula_str,
        df;
        verbose = verbose_opt,
        use_gpu = use_gpu_opt,
        prior_scheme = prior_scheme_sym
    )
    t_inst = time() - t_start_inst
    @printf("[Model] Instantiation completed in %.2f seconds.\n", t_inst)

    if cli_args[:dry_run]
        println("[DryRun] Configuration validated successfully. Exiting.")
        return 0
    end

    # 7. MCMC Sampling Execution
    sampling_cfg = get(cfg, "sampling", Dict{String, Any}())
    sample_override = cli_args[:sample]
    do_sample = sample_override !== nothing ? sample_override : get(sampling_cfg, "sample", true)

    chain = nothing
    res = nothing

    if do_sample
        n_samples = cli_args[:n_samples] !== nothing ? cli_args[:n_samples] : get(sampling_cfg, "n_samples", 100)
        n_warmup = cli_args[:n_warmup] !== nothing ? cli_args[:n_warmup] : get(sampling_cfg, "n_warmup", 50)
        n_chains = get(sampling_cfg, "n_chains", 1)
        target_accept = get(sampling_cfg, "target_accept", 0.8)
        sampler_name = lowercase(cli_args[:sampler] !== nothing ? cli_args[:sampler] : get(sampling_cfg, "sampler", "optimal"))
        progress_bar = get(sampling_cfg, "progress", true)

        println("\n[Sampling] Initializing MCMC execution:")
        println("  - Sampler strategy: ", sampler_name)
        println("  - Posterior draws:  ", n_samples)
        println("  - Adaptation steps: ", n_warmup)
        println("  - Parallel chains:  ", n_chains)

        t_start_sample = time()

        sampler_obj = if sampler_name in ["optimal", "auto", "gibbs"]
            TethysBayes.get_optimal_sampler(model; adaptation_steps=n_warmup)
        elseif sampler_name == "nuts"
            Turing.NUTS(n_warmup, target_accept)
        elseif sampler_name == "mh"
            Turing.MH()
        elseif sampler_name == "hmc"
            Turing.HMC(0.05, 10)
        else
            @warn "Unrecognized sampler '$(sampler_name)'. Defaulting to optimal sampler."
            TethysBayes.get_optimal_sampler(model; adaptation_steps=n_warmup)
        end

        chain = TethysBayes.TethysBayes_sample(
            model,
            sampler_obj,
            n_samples;
            progress = progress_bar
        )
        t_sample = time() - t_start_sample
        @printf("[Sampling] MCMC sampling completed in %.2f seconds.\n", t_sample)

        println("[Analysis] Computing comprehensive model results...")
        res = TethysBayes.model_results_comprehensive(model, chain; data=df, au=au_obj)

        if haskey(res, :metrics)
            m = res.metrics
            println("\n" * "-"^50)
            println("  Summary Performance Metrics:")
            @printf("  - RMSE:          %.4f\n", get(m, :rmse, NaN))
            @printf("  - Pearson r:     %.4f\n", get(m, :r_pearson, NaN))
            @printf("  - WAIC:          %.2f\n", get(m, :waic, NaN))
            @printf("  - Mean Rhat:     %.4f\n", get(m, :rhat, NaN))
            @printf("  - Minimum ESS:   %.1f\n", get(m, :ess, NaN))
            println("-"^50 * "\n")
        end
    end

    # 8. Analytical Output Persistence
    out_cfg = get(cfg, "output", Dict{String, Any}())
    out_path_raw = cli_args[:output_path] !== nothing ? cli_args[:output_path] : get(out_cfg, "path", "output/$(model_name).jld2")
    out_format_raw = lowercase(cli_args[:format] !== nothing ? cli_args[:format] : get(out_cfg, "format", "auto"))
    compress_opt = get(out_cfg, "compress", true)
    table_prefix = get(out_cfg, "table_prefix", "")

    # Automatically determine format from file extension if auto
    out_fmt = if out_format_raw != "auto"
        out_format_raw
    elseif endswith(lowercase(out_path_raw), ".duckdb") || endswith(lowercase(out_path_raw), ".db")
        "duckdb"
    elseif endswith(lowercase(out_path_raw), ".jld2")
        "jld2"
    else
        "jld2"
    end

    resolved_out_path = resolve_file_path(out_path_raw, config_dir)
    out_dir = dirname(resolved_out_path)
    if !isempty(out_dir) && !isdir(out_dir)
        mkpath(out_dir)
    end

    println("[Output] Persisting model results to: ", resolved_out_path, " (format: ", out_fmt, ")")

    if out_fmt == "jld2"
        saved_file = TethysBayes.save_TethysBayes_model(
            resolved_out_path,
            model;
            chain = chain,
            au = au_obj,
            metadata = meta_cfg,
            compress = compress_opt
        )
        println("[Output] Successfully saved JLD2 model file: ", saved_file)

    elseif out_fmt == "duckdb"
        if isnothing(res)
            error("Cannot save DuckDB results without posterior samples. Enable `[sampling].sample = true`.")
        end
        saved_file = TethysBayes.save_TethysBayes_results(
            resolved_out_path,
            res;
            model = model,
            chain = chain,
            au = au_obj,
            table_prefix = table_prefix,
            overwrite = true
        )
        println("[Output] Successfully saved DuckDB database: ", saved_file)

    elseif out_fmt == "both"
        if isnothing(res)
            error("Cannot save complete bundle without posterior samples. Enable `[sampling].sample = true`.")
        end
        base_name = replace(resolved_out_path, r"\.(jld2|duckdb|db)$"i => "")
        saved_bundle = TethysBayes.save_TethysBayes_bundle(
            base_name,
            model,
            chain,
            res;
            au = au_obj,
            metadata = meta_cfg,
            compress = compress_opt
        )
        println("[Output] Successfully saved complete TethysBayes bundle:")
        println("  - Model state:  ", saved_bundle.model_file)
        println("  - Results data: ", saved_bundle.duckdb_file)
    else
        error("Unrecognized output format: '$(out_fmt)'. Must be 'jld2', 'duckdb', or 'both'.")
    end

    println("\n[Complete] TethysBayes CLI execution finished successfully.")
    return 0
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    exit(main())
end
