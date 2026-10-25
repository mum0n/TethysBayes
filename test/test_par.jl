# ==============================================================================
# TethysBayes Test Suite: Population Attributable Risk (PAR / PAF) Engine
# ==============================================================================

if !@isdefined(TethysBayes_Likelihood)
    include(joinpath(@__DIR__, "test_helpers.jl"))
end

@testset "Population Attributable Risk (PAR/PAF) Engine" begin
    # 1. Test Levin formula
    chain_mock = DataFrame(smoking = fill(log(2.0), 100))  # beta = log(2.0) -> RR = 2.0
    res_levin = par_from_posterior(
        chain_mock, "smoking";
        family = "poisson",
        exposure_prevalence = 0.4,
        method = :levin
    )
    expected_levin = (0.4 * (2.0 - 1.0)) / (1.0 + 0.4 * (2.0 - 1.0))
    @test isapprox(res_levin.paf_mean, expected_levin; atol=1e-6)
    @test isapprox(res_levin.rr_mean, 2.0; atol=1e-6)

    # 2. Test Miettinen formula
    res_miettinen = par_from_posterior(
        chain_mock, "smoking";
        family = "poisson",
        exposure_prevalence = 0.4,
        method = :miettinen
    )
    expected_miettinen = 0.4 * (2.0 - 1.0) / 2.0
    @test isapprox(res_miettinen.paf_mean, expected_miettinen; atol=1e-6)

    # 3. Test Binomial exact RR conversion with baseline risk
    res_binom = par_from_posterior(
        chain_mock, "smoking";
        family = "binomial",
        baseline_risk = 0.2,
        exposure_prevalence = 0.4
    )
    expected_rr_binom = 0.3333333333333333 / 0.2
    @test isapprox(res_binom.rr_mean, expected_rr_binom; atol=1e-5)

    # 4. Test Prevented Fraction for protective exposure
    chain_mock_prot = DataFrame(diet = fill(log(0.5), 100))  # beta = log(0.5) -> RR = 0.5
    res_prot = par_from_posterior(
        chain_mock_prot, "diet";
        family = "poisson",
        exposure_prevalence = 0.5
    )
    expected_pf = (0.5 * (1.0 - 0.5)) / (0.5 * (1.0 - 0.5) + 0.5)
    @test isapprox(res_prot.prevented_fraction_mean, expected_pf; atol=1e-5)
    @test res_prot.paf_mean < 0.0

    # 5. Test Continuous exposure validation and threshold dichotomization
    chain_mock[!, :pollution] = fill(log(1.5), 100)
    df_cont = DataFrame(
        pollution = [12.0, 15.0, 22.0, 28.0, 35.0],
        cases = [1, 2, 4, 5, 8]
    )
    # Continuous without threshold should throw ArgumentError (no silent fallback or clamping)
    @test_throws ArgumentError par_from_posterior(
        chain_mock, "pollution";
        family = "poisson",
        exposure_var = "pollution",
        data = df_cont
    )

    # Continuous with threshold=20.0 (3 of 5 exposed -> prevalence = 0.6)
    res_cont = par_from_posterior(
        chain_mock, "pollution";
        family = "poisson",
        exposure_var = "pollution",
        threshold = 20.0,
        data = df_cont,
        outcome_var = "cases"
    )
    @test isapprox(res_cont.exposure_prevalence, 0.6; atol=1e-6)
    @test res_cont.attributable_cases !== nothing

    # 6. Test summarize_par_effects and export_par_to_table
    chain_multi = DataFrame(
        smoking = fill(log(1.8), 50),
        pollution = fill(log(1.5), 50)
    )
    multi_res = summarize_par_effects(
        chain_multi;
        covariates = ["smoking", "pollution"],
        families = Dict("smoking" => "binomial", "pollution" => "poisson"),
        baseline_risks = Dict("smoking" => 0.1),
        exposure_prevalences = Dict("smoking" => 0.25, "pollution" => 0.40)
    )
    tbl = export_par_to_table(multi_res)
    @test nrow(tbl) == 2
    @test hasproperty(tbl, :paf_mean)
    @test hasproperty(tbl, :method)

    # 7. Test single result export_par_to_table
      tbl_single = export_par_to_table(res_levin)
      @test nrow(tbl_single) == 1
  end

  @testset "L3: par_* is the absolute rate difference, not the PAF fraction" begin
      using Random
      # `par_*` used to be a bare alias of `paf_*`, i.e. a dimensionless fraction, which
      # contradicts the module's own definition `PAR = I_pop - I_0 = PAF x I_pop`.
      Random.seed!(42)
      n = 400
      # Poisson log-link: log(RR) = beta, so RR = exp(beta) > 1.
      ch = DataFrame(beta_exposure = exp.(0.30 .+ 0.05 .* randn(n)))
      df = DataFrame(exposed = repeat([0.0, 1.0], inner=2), y = [1.0, 3, 1, 3])

      # Poisson infers no baseline risk, so PAR is undefined and must be `nothing`
      # rather than a crash on `* nothing`, or silently the fraction it used to alias.
      r_pois = par_from_posterior(ch, "exposure"; family="poisson",
                                  exposure_var="exposed", exposure_prevalence=0.5, data=df)
      @test isnothing(r_pois.par_mean)

      r = par_from_posterior(ch, "exposure"; family="binomial", exposure_var="exposed",
                             exposure_prevalence=0.5, baseline_risk=0.10, data=df,
                             method=:levin)

      # Levin: q = p_pop (RR - 1);  PAR = I_0 q. PAR is LINEAR in q, so the mean-RR
      # reference is exact here (unlike PAF, which is nonlinear in RR and therefore
      # satisfies mean(PAF) != PAF(mean RR) by Jensen -- do not "fix" that gap).
      I0 = r.baseline_risk
      @test isapprox(r.par_mean, I0 * 0.5 * (r.rr_mean - 1.0); rtol=1e-5)

      # The two are genuinely different quantities, and PAR is on the risk scale.
      @test !isapprox(r.par_mean, r.paf_mean; rtol=1e-3)
      @test 0 < r.par_mean < 1

      # Summaries must stay ordered, and the population-mean form must agree to within
      # the Jensen gap.
      @test r.par_lower <= r.par_median <= r.par_upper
      @test isapprox(I0 * r.paf_mean / (1 - r.paf_mean), r.par_mean; rtol=5e-2)
  end

      @testset "excess_cases needs a real observed total" begin
          # `excess_cases = PAF * total_observed`. The old code fell back to
          # `y_total = 1.0` when the outcome column was absent, which made
          # `excess_cases_mean` numerically IDENTICAL to `paf_mean` -- a dimensionless
          # fraction reported under a name that says "cases".
          n = 200
          dfe = DataFrame(
              exposed = repeat([0.0, 1.0], inner = n ÷ 2),
              y = repeat([2.0, 5.0], inner = n ÷ 2),
          )
          me = TethysBayes.TethysBayes_core("likelihood(y, family=poisson) ~ intercept() + exposed",
                              dfe; verbose=false)
          che = sample(MersenneTwister(3), me, TethysBayes.Turing.MH(), 40;
                       progress=false, check_model=false)

          # With the outcome present it is a genuine count: PAF x total.
          r = TethysBayes.par_counterfactual(me, che; exposure_var="exposed", data=dfe)
          ytot = sum(dfe.y)
          @test isapprox(r.excess_cases_mean, r.paf_mean * ytot; rtol=1e-12)
          @test !isapprox(r.excess_cases_mean, r.paf_mean; rtol=1e-6)
          @test r.excess_cases_ci_lower <= r.excess_cases_mean <= r.excess_cases_ci_upper

          # Without it, the count is undefined and must be `nothing` -- the same
          # convention `par_mean` already uses, not a number that looks like a count.
          r2 = TethysBayes.par_counterfactual(me, che; exposure_var="exposed",
                                       data=select(dfe, :exposed))
          @test isnothing(r2.excess_cases_mean)
          @test isnothing(r2.excess_cases_ci_lower)
          @test isnothing(r2.excess_cases_ci_upper)
          # The estimable quantity survives: PAF needs no observed total.
          @test !isnothing(r2.paf_mean)
          @test isfinite(r2.paf_mean)
          @test length(r2.raw_paf_samples) == 40
          @test r2.par_mean === nothing   # unchanged, also unavailable here
      end

      @testset "Marginalization over the random effect (population_average=true)" begin
      using Random, Statistics, LogExpFunctions
      _logistic = LogExpFunctions.logistic

      beta, eta0, sigma = 0.4, -0.7, 0.9
      s2 = sigma^2

      # 1. The Gauss-Hermite table must integrate the weight's moments exactly. These
      #    are the exact values of int x^{2k} exp(-x^2) dx for k = 0,1,2,3, so a wrong
      #    table (wrong convention, e.g. probabilists' instead of physicists') fails here.
      @test isapprox(sum(TethysBayes._GH_WEIGHTS), sqrt(pi); rtol=1e-12)
      @test isapprox(sum(TethysBayes._GH_WEIGHTS .* TethysBayes._GH_NODES.^2), sqrt(pi)/2; rtol=1e-12)
      @test isapprox(sum(TethysBayes._GH_WEIGHTS .* TethysBayes._GH_NODES.^4), 3sqrt(pi)/4; rtol=1e-12)
      @test isapprox(sum(TethysBayes._GH_WEIGHTS .* TethysBayes._GH_NODES.^6), 15sqrt(pi)/8; rtol=1e-12)

      # 2. Quadrature must agree with brute-force Monte Carlo over a SHARED field.
      #    Sharing is the case the conditional value gets wrong.
      Random.seed!(99)
      us = randn(MersenneTwister(99), 2_000_000) .* sigma
      mc_p0 = mean(_logistic.(eta0 .+ us))
      @test isapprox(TethysBayes._logistic_normal_mean(eta0, sigma), mc_p0; rtol=3e-3)

      mc_rr = mean(_logistic.(eta0 .+ beta .+ us)) / mean(_logistic.(eta0 .+ us))
      marg_rr = TethysBayes._marginal_logit_rr(beta, eta0, s2)
      cond_rr = _logistic(eta0 + beta) / _logistic(eta0)
      @test isapprox(marg_rr, mc_rr; rtol=3e-3)
      # The conditional value must be measurably wrong, otherwise this test proves nothing.
      @test !isapprox(cond_rr, mc_rr; rtol=2e-2)
      @test cond_rr > mc_rr   # overstated, not understated

      # 3. sigma -> 0 must recover the conditional value EXACTLY, so the correction is a
      #    strict generalisation and cannot perturb a random-effect-free model.
      @test isapprox(TethysBayes._marginal_logit_rr(beta, eta0, 0.0), cond_rr; rtol=1e-12)
      @test isapprox(TethysBayes._logistic_normal_mean(eta0, 0.0), _logistic(eta0); rtol=1e-12)

      # 4. Log link: closed form exp(eta0 + s2/2), and the population mean is exactly
      #    exp(s2/2) times the reference individual -- a RATIO, not a one-sided percentage.
      pop = TethysBayes._population_baseline_risk([eta0], "poisson", [s2])[1]
      @test isapprox(pop, exp(eta0 + s2/2); rtol=1e-12)
      @test isapprox(pop / exp(eta0), exp(s2/2); rtol=1e-12)
      @test isapprox(pop, mean(exp.(eta0 .+ us)); rtol=3e-3)
      @test isapprox(exp(s2/2), 1.4993; rtol=1e-3)   # 50% higher, or 33% lower

      # 5. `nothing` variance must NOT be treated as zero variance. This is the
      #    unavailable-vs-absent distinction; conflating them would silently apply a
      #    zero correction to a model whose field was never identified.
      @test isnothing(TethysBayes._population_baseline_risk([eta0], "poisson", nothing))
      @test isnothing(TethysBayes._population_baseline_risk([eta0], "poisson", Float64[]))
      @test isnothing(TethysBayes._population_baseline_risk([eta0], "poisson", [NaN]))
      @test isnothing(TethysBayes._random_effect_variance(DataFrame(x=fill(1.0, 3))))

      # 6. `y_sigma` is observation noise and must be EXCLUDED from the field variance;
      #    folding it in would double-count residual variation.
      n = 200
      ch_noise = Dict(
          :intercept => reshape(fill(eta0, n), 1, n),
          :sigma_region => reshape(fill(sigma, n), 1, n),
          :y_sigma => reshape(fill(3.0, n), 1, n),
      )
      @test isapprox(TethysBayes._random_effect_variance(ch_noise)[1], s2; rtol=1e-12)
      ch_no_field = Dict(
          :intercept => reshape(fill(eta0, n), 1, n),
          :beta_exposure => reshape(fill(beta, n), 1, n),
          :y_sigma => reshape(fill(3.0, n), 1, n),
      )
      @test isnothing(TethysBayes._random_effect_variance(ch_no_field))

      # 7. End-to-end: default is unchanged, population_average=true marginalizes.
      ch = Dict(
          :intercept => reshape(fill(eta0, n), 1, n),
          :beta_exposure => reshape(fill(beta, n), 1, n),
          :sigma_region => reshape(fill(sigma, n), 1, n),
      )
      df = DataFrame(exposed = repeat([0.0, 1.0], inner=2), y = [1, 1, 0, 0])
      r_cond = par_from_posterior(ch, "exposure"; family="binomial",
                                   exposure_var="exposed", exposure_prevalence=0.5, data=df)
      r_pop = par_from_posterior(ch, "exposure"; family="binomial", exposure_var="exposed",
                                 exposure_prevalence=0.5, data=df, population_average=true)

      # Default must be untouched: this is the guarantee that existing reported numbers
      # do not change without an explicit opt-in.
      @test isapprox(r_cond.baseline_risk, _logistic(eta0); rtol=1e-9)
      @test isapprox(r_cond.rr_mean, cond_rr; rtol=1e-9)

      @test isapprox(r_pop.baseline_risk, TethysBayes._logistic_normal_mean(eta0, sigma); rtol=1e-9)
      @test isapprox(r_pop.rr_mean, marg_rr; rtol=1e-9)
      @test r_pop.baseline_risk > r_cond.baseline_risk     # baseline goes up
      @test r_pop.rr_mean < r_cond.rr_mean                 # RR goes down

      # 8. PAR must remain consistent with the identity PAR = I_0 * PAF/(1-PAF) under BOTH
      #    paths, including the new one, or the marginalization has broken the invariant.
      for r in (r_cond, r_pop)
          I0 = r.baseline_risk
          @test isapprox(r.par_mean, I0 * r.paf_mean / (1 - r.paf_mean); rtol=1e-6)
      end

      # 9. With no field in the chain, population_average=true must fall back to the
      #    conditional value rather than erroring or applying a zero-variance correction.
      r_fallback = par_from_posterior(ch_no_field, "beta_exposure"; family="binomial",
                                      baseline_risk=0.2, exposure_prevalence=0.5,
                                      population_average=true)
      @test r_fallback.paf_mean > 0
      @test isfinite(r_fallback.paf_mean)
  end
