module TethysBayesPlotsExt

using TethysBayes
using Plots
using StatsPlots
using ColorSchemes
using DataFrames
using Statistics
using StatsBase
using LinearAlgebra
using Dates
using OrderedCollections
using Random
using DynamicPPL
using Turing
using Distributions
using NamedArrays
using Graphs

# Required functions for the extension to overload
import TethysBayes: TethysBayes_plots, _TethysBayes_plots_impl, model_results_plots,
    par_credible_interval_plot, par_forest_plot, plot_ppc,
    plot_prior_vs_posterior, plot_marginal_effects, plot_spatial_residuals,
    plot_spatiotemporal_facets

# Internal functions from TethysBayes that plotting.jl needs
import TethysBayes: _detect_xy_columns, _detect_time_column, _detect_response_column,
    _detect_spatial_unit_column
import TethysBayes: _generate_conditional_predictions
import TethysBayes: AbstractModelArchitecture, UnivariateArchitecture,
    MultivariateArchitecture, MultifidelityArchitecture
import TethysBayes: Mixed, RFF, SpectralGP, WaveletGP, SPDE, TPS, GP, Nystrom, SVC

# Include plotting implementation
include("../src/plotting.jl")

end # module TethysBayesPlotsExt
