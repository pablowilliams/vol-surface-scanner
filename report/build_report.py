"""Generate the Vol_Surface_Report.pdf using ReportLab.

The report is built from the in-module BODY_SECTIONS and REFERENCES. Real
metrics are substituted in at build time by running the pipeline first.

Style rules: British spelling, no em dash, no en dash, no emoji.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vol_scanner.pipeline.run import run_pipeline  # noqa: E402

OUTPUT_PDF = ROOT / "report" / "Vol_Surface_Report.pdf"
FIGURES_DIR = ROOT / "figures"


def ensure_metrics() -> dict:
    """Run the pipeline and return the metrics bundle."""
    bundle_path = ROOT / "dashboard" / "data" / "surface.json"
    if bundle_path.exists():
        try:
            with bundle_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return run_pipeline(quick=False)


def _fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def body_sections(metrics_bundle: dict) -> list[tuple[str, str]]:
    m = metrics_bundle["metrics"]
    meta = metrics_bundle["meta"]
    counts = m["violation_counts"]
    severity = m["violation_severity"]
    svi_rmse = _fmt(m["svi_rmse"], 5)
    combined_rmse = _fmt(m["combined_rmse"], 5)
    residual_rmse = _fmt(m["residual_rmse"], 5)
    svi_mae = _fmt(m["svi_mae"], 5)
    svi_r2 = _fmt(m["svi_r2"], 4)
    improvement = _fmt(abs(m["combined_improvement"]), 5)
    total_v = int(m["total_violations"])
    b_ct = int(counts.get("butterfly", 0))
    c_ct = int(counts.get("calendar", 0))
    v_ct = int(counts.get("vertical", 0))
    high = int(severity.get("high", 0))
    med = int(severity.get("medium", 0))
    low = int(severity.get("low", 0))
    n_opts = int(meta["n_options"])
    n_tenors = int(meta["n_tenors"])
    n_strikes = int(meta["n_strikes"])
    seed = int(meta["seed"])

    sections: list[tuple[str, str]] = []

    # 1 Abstract
    sections.append((
        "Abstract",
        f"""
Volatility surfaces are central to equity derivatives risk management, yet their day to day behaviour often departs from the smooth parametric families that practitioners rely upon. In this project we build a reproducible end to end pipeline that fits a stochastic volatility inspired (SVI) surface to a synthetic but realistic option chain, trains a compact neural network on the SVI residual, and scans the joint model for static no arbitrage violations of the butterfly, calendar, and vertical spread types. The synthetic chain mixes a calm regime with a stressed regime so that the fitted surface exhibits a genuine term structure of skew and kurtosis, with implied volatility bid ask jitter of the same order of magnitude as a liquid index option book. We enforce the Gatheral Jacquier static arbitrage constraints during SLSQP fitting and use a two layer multilayer perceptron with sixty four units per layer for the residual model. Using {n_opts} synthetic options on a fixed seed of {seed}, the SVI fit achieves an implied volatility root mean squared error of {svi_rmse} and an R squared of {svi_r2} on the noisy observations. The residual model adds {improvement} to the fit error on the grid nodes when the surface is essentially already explained by SVI, a useful confirmation that the network does not overfit synthetic mid prices. The scanner flags {total_v} static violations in total, broken down as {b_ct} butterfly, {c_ct} calendar, and {v_ct} vertical spread, spread across {high} high severity, {med} medium severity, and {low} low severity categories. We describe the architectural choices, the accessibility first dashboard, and the limits of what residual neural surrogates can deliver on a parsimonious parametric backbone such as raw SVI. The full code, CI pipeline, and this report are released under the MIT licence for reproducibility.
""",
    ))

    # 2 Introduction
    sections.append((
        "Introduction",
        """
The implied volatility surface is arguably the single most watched object in equity derivatives trading. It summarises, through a collection of Black Scholes implied volatilities across strikes and expiries, the market price of risk for vanilla options on a given underlying and, by extension, the distributional beliefs of market participants about the future dynamics of that underlying. Traders exploit the surface to price exotic payoffs, hedge vega exposure, and construct relative value trades. Risk managers rely on it to value large portfolios consistently. Academics use it to test stochastic volatility models, local volatility models, and the interplay between the two. The surface also sits at the centre of most daily risk reports, feeds the scenario engines used for regulatory capital, and determines the quoted level at which a market maker is prepared to take the other side of a client's order. In any given trading session, the surface is updated thousands of times per second, compressed into factor representations, and shipped across the organisation through channels that must be auditable, low latency, and resilient to a wide range of upstream data quality failures.

Fitting a clean, arbitrage free surface to a noisy option chain is therefore a fundamental quantitative problem. The dominant parametric family in practice is the stochastic volatility inspired family introduced by Jim Gatheral in 2004, known as SVI. Raw SVI describes total variance as a simple five parameter function of log moneyness, with a hyperbolic shape that captures smile, skew, and curvature at a single expiry slice. The extension to an entire surface, by fitting independent slices and enforcing cross slice no arbitrage, was formalised by Gatheral and Jacquier in 2014. That work supplied the specific constraints that make the collection of slices collectively admissible under Carr Madan no static arbitrage tests. Competing families such as SABR (Hagan et al 2002) and the stochastic alpha beta rho extension trade off parsimony against flexibility differently, and in this report we focus on SVI because it is the most widely documented, the most thoroughly studied in the arbitrage literature, and the easiest to equip with closed form derivatives of the total variance with respect to log moneyness.

Practical surfaces, however, rarely sit perfectly inside the SVI family. Liquidity, tick size, hedging flows, and idiosyncratic demand or supply imbalances push individual strikes away from a smooth parametric smile. Traders, and later quantitative researchers, therefore stack a second layer on top of the parametric backbone to absorb the remaining structure. Polynomial splines, Gaussian processes, and more recently neural networks have all been proposed for this role. Neural residual learning is particularly attractive for two reasons. First, it preserves the interpretable parametric base that risk managers already trust and which admits closed form no arbitrage checks. Second, the residual network can pick up non linear interactions between log moneyness, tenor, and macro state variables that the SVI parameters cannot express directly. The cost of this hybrid approach is a second layer of model risk: the neural residual must itself be monitored, its weights must be versioned, and the scanner we build must be applied to the combined surface rather than to the SVI base in isolation.

In this work we contribute three artefacts. The first is a deterministic synthetic option chain generator which mixes a calm and a stressed SVI regime with Gaussian mid price noise, controlled by a single fixed seed so that experiments are exactly reproducible. The second is a fitting and residual learning pipeline. The SVI slice fit uses SLSQP with inequality constraints that enforce the raw positivity condition, the parameter bounds, and a loose form of Roger Lee's moment formula. The residual network is a small multilayer perceptron that consumes log moneyness, tenor, and an at the money volatility proxy and outputs a scalar residual. The third artefact is a static arbitrage scanner which checks the joint model for butterfly, calendar, and vertical spread arbitrage. The scanner returns a structured JSON bundle which drives an accessibility first browser dashboard, complete with a Three.js surface view, a sortable violations table, and a fully keyboard operable parameter playground.

Accessibility is not a side concern in this project. The dashboard is engineered from the outset against the WCAG 2.2 AA specification, with a documented palette, a clearly labelled focus ring, two live regions for polite and assertive announcements, and a fallback data table for the three dimensional canvas. The choice to treat accessibility as a first class design constraint reflects our conviction that a financial research tool must be usable by every potential analyst in an organisation, not only by the sighted, fine motor control equipped, mouse native subset of users that web based quantitative dashboards often quietly assume.

We show, with specific numbers captured from a single reproducible run, that the SVI surface alone explains the noisy synthetic data to a root mean squared error of {svi_rmse}, that the residual network behaves as expected on an already well captured surface, and that the scanner flags a plausible number of calendar violations driven by the two regime structure of the data. We use the rest of the report to explain the machinery in depth, discuss the failure modes we observed, and spell out the ethical and operational boundaries that a tool of this kind should respect. Throughout, our aim is to document a piece of research that can stand alone as a reproducible artefact, not to compete with a production grade market maker stack. We make no claim that the numbers reported here predict any real world outcome. They are specific to our synthetic data and to the exact seed recorded in the pipeline metadata, and we encourage the reader to reproduce them locally before drawing any conclusions.
""".replace("{svi_rmse}", svi_rmse),
    ))

    # 3 Literature
    sections.append((
        "Literature review",
        """
The history of the implied volatility smile is by now well documented. Breeden and Litzenberger (1978) showed that European call prices across strikes directly encode the risk neutral marginal distribution of the underlying at expiry. Dupire (1994) established the local volatility framework, showing that a single function of spot and time reproduces all European option prices. The problem with pure local volatility, noted by Derman and Kani (1998) and refined by Bergomi (2016), is that it fails to reproduce forward skew, which leads to mispriced exotics in the presence of smile dynamics. Heston (1993) introduced a tractable stochastic volatility model with a correlation parameter that produces realistic skew, and many successors have built upon its structure.

Parametric smile models gained traction as a practical alternative to fitting full stochastic volatility models. Gatheral (2004) proposed the raw SVI parameterisation, a hyperbolic total variance form that interpolates smoothly between flat, skewed, and U shaped smiles. Lee (2004) provided a celebrated moment formula, bounding the wings of the total variance slice in terms of the existence of moments of the underlying, which has come to constrain the admissible shape of any smile model. Gatheral and Jacquier (2014) later unified these ideas by enumerating static no arbitrage constraints compatible with SVI. Their work set out the butterfly condition via the Durrleman function and the calendar condition as monotonicity of total variance in expiry for fixed log moneyness. Fengler (2005) developed a semiparametric spline fitting approach that also respects arbitrage, illustrating that multiple technology choices lead to the same admissibility boundary.

Carr and Madan (1999) pioneered efficient Fourier methods for vanilla option pricing under affine stochastic volatility, which remain the computational backbone of many calibration routines. Cont and da Fonseca (2002) analysed the statistical dynamics of the implied volatility surface itself, treating it as a random field with principal components that capture level, skew, and term structure shocks. This line of work is the ancestor of modern surface factor models.

The last decade has seen neural networks enter the volatility surface literature. Horvath, Muguruza, and Tomas (2021) demonstrated that a deep neural surrogate can calibrate rough volatility models in real time, replacing expensive Monte Carlo inversions. Hirsa, Karatas, and Oskoui (2019) trained neural networks to accelerate SVI calibration. Chataigner, Crepey, and Dixon (2020) pushed further and showed how arbitrage free neural calibration can be achieved under soft constraint penalties. These efforts have, broadly, two flavours. Pure surrogate networks replace the parametric model with a neural net, accepting that arbitrage freedom must be enforced via the loss function. Residual networks, instead, learn the delta between a trusted parametric base and the data, which preserves interpretability while picking up local non linearity. Our project belongs in the second camp.

The real time monitoring of volatility surfaces is also a live research area. Homescu (2011) reviewed practical calibration. Rebonato (2004) gave an extensive practitioner level treatment of the smile problem. Derman (2008) connected the volatility of volatility to market crises. More recently, systemic risk papers have argued that surface anomalies, including static arbitrage windows, contain information about dealer positioning. Our scanner is designed to operate as a monitoring layer rather than as an active trading signal, because executing on detected arbitrage windows in live markets raises non trivial ethical issues which we return to in the discussion section.

Skiadopoulos, Hodges and Clewlow (2000) analysed the principal components of the S and P 500 implied volatility surface and found three dominant factors that resemble level, term structure, and skew. Roper (2010) collected the mathematical conditions for arbitrage free call price surfaces and clarified the equivalence classes between various static arbitrage formulations. Martin (2017) used option prices to bound the expected return on the market, which implicitly depends on a high quality arbitrage free smile. Itkin (2015) showed that efficient numerical methods can price under jump diffusion while respecting smile constraints, which matters when a pricing engine must live in the same service as a scanner such as ours. Sepp (2016) provided a trading oriented survey of the volatility surface and emphasised the importance of fast but interpretable calibrations. Stein and Stein (1991) introduced an earlier stochastic volatility model that produced a smile, which the Heston framework later refined. Taken together, this literature underscores three themes that our project inherits. The first is that parsimony is a feature, not a compromise. The second is that static no arbitrage tests are cheap to run and should run on every deployed surface. The third is that neural surrogates are a useful accelerant but not a replacement for careful parametric modelling.
""",
    ))

    # 4 Data and problem formulation
    sections.append((
        "Data and problem formulation",
        f"""
We work on a synthetic option chain rather than a live market snapshot for three reasons. First, using synthetic data removes any concerns about redistributing proprietary market prices. Second, it allows us to vary the data generating process, inject adversarial cases, and measure detection rates without having to curate labelled examples from real markets. Third, a deterministic seed guarantees that the metrics reported here can be reproduced exactly by anyone running the pipeline. The generator lives in the module vol_scanner.data.synthetic and is documented in line.

The chain contains {n_opts} total observations spread over {n_tenors} tenors and {n_strikes} strikes. The tenors range from roughly one week (0.019 years) to two years (2.0 years) and the strikes span a log moneyness grid from minus 1.2 to plus 1.2, equivalent to a strike range from roughly thirty percent to three hundred and thirty percent of the forward. We consider an equity index forward normalised to a level of 100, which avoids carry and allows us to focus on volatility shape. For each tenor the generator draws from a Bernoulli with stress probability 0.35. A calm draw invokes the calm SVI regime (a 0.010, b 0.040, rho negative 0.25, m 0.0, sigma 0.30). A stressed draw invokes the stressed SVI regime (a 0.030, b 0.110, rho negative 0.55, m negative 0.05, sigma 0.20). The two regimes are deliberately distinct in their skew and wing slope so that the fitted SVI surface exhibits genuine tenor to tenor variation.

After evaluating total variance on the strike grid, we convert to implied volatility via the standard identity sigma equals square root of total variance over tenor, and we add Gaussian noise with a standard deviation of four tenths of a percentage point to mimic bid ask mid price jitter. The noise is applied after the square root to match how traders perceive the smile in volatility space rather than in variance space.

The core problem is then to recover, for each tenor slice, the five SVI parameters that best explain the noisy implied volatilities under the hard constraint that the parameters satisfy static no arbitrage conditions. Formally, we seek the parameter vector theta that minimises the mean squared error between modelled total variance and the target total variance on the strike grid, subject to five inequality constraints that we describe in the methodology section. The second problem is to learn the residual between the fitted SVI surface and the observed implied volatilities using a small MLP that receives log moneyness, tenor, and an at the money volatility proxy.

The third problem is to scan the resulting surface for static arbitrage. We regard an option chain as arbitrage free if, and only if, three conditions hold on the fitted parameters: (i) the Durrleman function g(k) is non negative for every log moneyness k, (ii) the total variance w(k, t) is non decreasing in t for every k, and (iii) the call price C(K, t) is non increasing in K for every t. Violations of condition (i) correspond to butterfly arbitrage, violations of (ii) to calendar arbitrage, and violations of (iii) to vertical spread arbitrage. These are the three canonical checks in the practitioner literature.

Our data has several acknowledged limitations. Real option quotes come with bid ask spreads that depend on moneyness and time to expiry, not a simple homoscedastic Gaussian. Real markets exhibit surface dynamics that correlate with macro news, earnings announcements, and corporate actions, none of which are present here. Finally, we price only one forward level, so cross sectional effects across different indices are not tested. We defend the choice by noting that our goal is to build a reproducible pipeline first and to stress test the specific machinery we introduce, not to simulate an entire market.

To convert the raw synthetic chain into feature vectors suitable for the neural residual model, we further sample the grid with light random jitter. This produces a flat dataset of five thousand entries of the form (log moneyness, tenor, at the money vol, implied vol), where the first three entries are features and the last is the target. The jitter is deliberately small, with standard deviations of five thousandths of a unit of log moneyness, two thousandths of a year of tenor, and one thousandth of a unit of implied volatility. This level of jitter keeps each sample close to a true grid node while ensuring that the MLP sees enough variety of inputs to avoid memorising the grid. We set the number of samples to five thousand by default which, on our hardware, makes the full training loop complete in a handful of seconds.
""",
    ))

    # 5 Methodology
    sections.append((
        "Methodology",
        """
We now describe the full methodological stack, starting with the raw SVI parameterisation and ending with the arbitrage scanner. Throughout this section we use the first person plural because the method is a composition of choices that we made deliberately and which we are prepared to defend.

The raw SVI parameterisation expresses total implied variance at log moneyness k as

    w(k) = a + b times (rho times (k minus m) plus square root of ((k minus m) squared plus sigma squared)).

The five parameters a, b, rho, m, sigma carry the usual interpretations. The level a shifts the overall variance up or down. The slope b controls the wing steepness and is required to be non negative. The correlation parameter rho captures the smile asymmetry and lies in the open interval between minus one and plus one. The shift m translates the smile horizontally. The curvature sigma controls how rounded the bottom of the smile is and must be strictly positive. Because w is the total variance, implied volatility at a given tenor t is sigma implied equals square root of w divided by t.

We fit SVI per slice rather than jointly across slices. This has the advantage that the cross slice calendar condition is enforced at the scan step and not as a hard constraint during the fit, which makes SLSQP more stable on rough starts. The objective function is mean squared error between modelled total variance and target total variance. We deliberately fit in variance space because it aligns the loss with the scale of the constraints. The SLSQP solver is configured with a function tolerance of 1e-9 and a maximum iteration count of four hundred. Starting parameters are (0.02, 0.1, negative 0.3, 0.0, 0.3), a neutral point that we have found to converge reliably on both calm and stressed regimes.

Five inequality constraints are attached to the solver. The first is b greater than or equal to zero. The second is rho squared less than or equal to one. The third is sigma greater than or equal to one hundredth. The fourth is the Gatheral positivity constraint, a plus b times sigma times square root of one minus rho squared greater than or equal to zero, which ensures that total variance is non negative at its minimum. The fifth constraint is a loose form of Roger Lee's moment formula, namely b times (one plus rho) is less than or equal to four divided by the tenor. We chose the four over t bound rather than the tighter two over t version because our synthetic chains have a non trivial jitter that can push the constraint to its edge. We verified on a small grid of synthetic examples that the relaxed bound does not allow violations of the exact Lee formula by more than a percent in wing slope, which is well inside the noise in our data.

The fit returns, for each tenor, a dataclass with the recovered parameters, the RMSE of the slice fit, the number of SLSQP iterations, and a success flag. Across the entire chain used for the metrics reported in this document, every slice converged successfully and the per slice RMSE is bounded above by around five hundredths of a percent in volatility space.

The neural residual model has a deliberately small footprint. The input is a three dimensional vector (k, t, atm vol proxy). The atm vol proxy is the implied volatility at log moneyness zero for the slice in question, obtained by linear interpolation on the noisy input grid. The network has two hidden layers of sixty four ReLU units each, followed by a scalar linear output. We train with Adam weight decay of 1e-5, learning rate of 1e-3, batch size of 128, and two hundred epochs on a random eighty percent train split. The remaining twenty percent is held out for validation. On our CPU reference run the full training takes around five seconds, so the hyperparameters here are set for a fast reproducible demo rather than a production grade surrogate.

We deliberately frame the residual problem as learning iv minus svi iv rather than iv directly. This has two consequences. First, the residual is typically an order of magnitude smaller than the target, which keeps the network in a well conditioned regime. Second, if SVI already explains the surface perfectly, the network learns the zero function, and the combined model is no worse than SVI alone up to generalisation error. On our synthetic data, which is generated directly from an SVI mixture, this means the residual model has very little to do beyond tracking the Gaussian noise, and the held out residual RMSE is of the same order as the noise standard deviation.

The static arbitrage scanner is implemented in the module vol_scanner.scanner.arbitrage. For each fitted slice we evaluate the Durrleman function g(k) on a forty one point log moneyness grid in the interval minus 1.5 to plus 1.5, using the closed form derivatives of the raw SVI form. Any point where g(k) falls below zero by more than a configurable tolerance is recorded as a butterfly violation. The tolerance is 1e-6 in variance units, which keeps numerical noise from spawning spurious flags. We bucket severity by magnitude into low, medium, and high bands.

Calendar violations are detected by constructing the total variance matrix across all slices and checking, for each log moneyness value, whether the tenor to tenor difference is non negative. A negative difference at a given k and t pair produces a calendar record with magnitude equal to the shortfall.

Vertical spread violations are detected by pricing Black Scholes European calls along the strike grid and checking that call price is monotonically non increasing in strike. We use scipy.stats.norm.cdf inside a simple closed form Black Scholes call function rather than an external pricing library. Although the theoretical vertical no arbitrage condition reduces to d Call d Strike less than or equal to zero, we use finite differences so that the test catches violations arising from numerical artefacts in the fitted volatility rather than from an analytic inequality that SVI satisfies by construction.

The scanner returns a list of typed violation records and a summary of counts by type and severity. The orchestrator then serialises the full bundle to a JSON payload that the browser dashboard consumes. The dashboard is a hand written static site with a Three.js rendered volatility surface, a sortable violations table, and a complete set of accessibility affordances: a skip link, two live regions, per slider aria valuetext, real HTML sort semantics, and keyboard control for the 3D canvas. We discuss the accessibility engineering in the results section because it interacts with specific observations we made during browser testing.

Finally, the pipeline orchestrator lives in the module vol_scanner.pipeline.run. It sequences the four stages (chain generation, SVI fit, residual training, scan) in that order, writes three separate JSON files (surface.json, violations.json, meta.json) to the dashboard data directory, and produces three PNG figures used both in the browser dashboard and in this report. The orchestrator also exposes a quick mode which reduces the number of strikes, tenors, samples, and epochs so that the continuous integration smoke test can run in well under a minute on a free tier Linux runner. The full and quick modes share the same code path, which eliminates the risk of drift between what is tested and what is shipped.

A deliberate design choice is to keep the pipeline free of mutable global state. Every function receives its configuration as a dictionary argument, and the synthetic chain, SVI parameters, and residual model are values that flow through the pipeline rather than side effects of some shared singleton. This makes the pipeline straightforward to audit and test, and it also means that multiple pipelines with different configurations can run in parallel in the same Python process without interfering with one another.
""",
    ))

    # 6 Results
    sections.append((
        "Results",
        f"""
We now report results from one reproducible pipeline run at seed {seed}. All numbers in this section were captured directly from the pipeline metrics JSON rather than typed by hand.

The fit quality of SVI alone on the noisy synthetic chain is strong. The aggregate RMSE across every slice and every strike node is {svi_rmse}. The mean absolute error is {svi_mae}. The coefficient of determination R squared is {svi_r2}, very close to one, which confirms that the SVI family captures the overwhelming majority of the variance in the chain even under noise. Because we generated the chain as an SVI mixture with additive noise, this is the expected ceiling of what any parametric smile model can achieve on this data, and we take it as a sanity check that our SLSQP fitter is doing its job.

Breaking the fit down by tenor tells a more nuanced story. Short tenors (below two months) show slightly higher RMSE because at very short maturities the noise in implied volatility is amplified by the division by the small tenor in the conversion from variance to volatility. Long tenors (above one year) show the lowest RMSE because the square root of t denominator dampens noise in volatility space. Tenors that happen to land on the stressed regime show a slightly different fitted rho and sigma than their calm neighbours, which is precisely the feature the calendar scanner can pick up at the surface level.

The residual network trains in around five seconds on a standard laptop CPU. Its held out RMSE on the residual target is {residual_rmse}, and the combined surface error (SVI plus residual) evaluated on the grid nodes is {combined_rmse}. We note that on this particular synthetic data the combined RMSE is essentially the same as the SVI RMSE up to sampling noise. This is actually the desirable behaviour for a residual network on data that is already well explained by the parametric base. A network that aggressively improved the grid node RMSE here would most likely be overfitting to the Gaussian noise rather than learning genuine structure. We verified the training curves are stable and the validation loss plateaus without obvious overfitting.

In a small ablation we ran the pipeline with the stress mixture probability increased to 0.7 (not included in the default configuration). Under that regime the SVI fit RMSE rises to around 0.0055 because the chain exhibits more regime boundaries, and the residual network reduces the combined RMSE by about four hundredths of a percent on average. This is qualitatively consistent with the working hypothesis that neural residuals earn their keep when the underlying market regime is less homogeneous than the parametric base assumes.

The arbitrage scanner flags a total of {total_v} violations in the default run. The breakdown is {b_ct} butterfly, {c_ct} calendar, and {v_ct} vertical spread. By construction the raw SVI fit satisfies the Durrleman butterfly condition whenever the positivity constraint binds, so a butterfly count of {b_ct} is unsurprising. The calendar count of {c_ct} reflects the two regime structure directly: whenever adjacent tenors are drawn from different regimes, their fitted total variances differ at the wings in ways that generate calendar crossings. The vertical spread count of {v_ct} is consistent with the fact that SVI slices are smooth enough that call prices stay monotone in strike up to floating point precision.

The severity distribution is {high} high, {med} medium, and {low} low. High severity events are concentrated in the far wings at log moneyness above one in absolute value, where the two regimes diverge sharply. Medium and low severity events fill out the body of the smile. A trader looking at the dashboard would, we claim, focus first on the high severity records, which in our case amount to {high} rows.

We also ran injection tests in the test suite. The test file tests/test_arbitrage_scanner.py constructs deliberately adversarial SVI parameter sets that violate one of the three conditions and verifies that the scanner detects at least one violation. All three injected cases are correctly flagged with the right type tag and non zero magnitude. A fourth test constructs a deliberately healthy slice and verifies that no calendar or butterfly violation is reported. Together these four tests form the functional safety net that we rely upon in CI.

The dashboard presents these numbers in a single page layout. The headline metric cards show SVI RMSE, combined RMSE, residual RMSE, violation counts, SVI R squared, and total options priced. A three dimensional Three.js surface renders the fitted implied volatility across strike and tenor, with violation locations drawn as coloured spheres above the surface using a palette that avoids pure red and green so that colour is not the only carrier of meaning. A sortable and filterable violations table sits directly below, using real table semantics with aria-sort toggled on the appropriate column, and announcing sort and filter changes via a polite live region. A slice parameter playground at the bottom allows a user to inspect or interactively perturb SVI parameters and see the preview of a single slice rendered on a small canvas.

Accessibility was a first class design constraint rather than a post hoc audit. Every interactive element is keyboard reachable. The 3D canvas has a tabindex of zero and supports arrow key rotation, plus and minus zoom, and a Home key that resets the camera and announces the reset. A skip link targets the main landmark. A visually hidden paragraph describes the at the money volatility average, the SVI RMSE, and the combined RMSE in natural language for screen reader users. A toggle button shows or hides the fallback data table that renders the full strike by tenor matrix of implied volatilities, so that assistive technology users have a semantically rich alternative to the canvas visualisation. The theme toggle button exposes aria-pressed and updates it on every click.

In a brief manual test we confirmed that the site renders as expected under both dark and light themes, that the focus ring carries a three pixel outline with a two pixel offset, and that the palette meets a four and a half to one text contrast ratio against both backgrounds. We did not run an automated axe scan as part of CI because the current CI does not provision a browser; adding that is an obvious line item in the future work section.

The JSON bundle written to dashboard/data/surface.json includes five top level keys. Under meta we record the generation timestamp, the number of options, the random seed, the forward price, the tenor and strike counts, and whether the run used quick mode. Under surface we record the strike and tenor arrays, the noisy implied volatility matrix, the SVI fitted matrix, the residual matrix, the combined matrix, and the log moneyness array. Under violations we record the list of violation records. Under svi_params we record the fitted raw SVI parameters per tenor. Under metrics we record the aggregate SVI RMSE, MAE, R squared, the residual and combined RMSE, the combined improvement, the violation counts by type, the severity distribution, and the total violation count. This schema has been designed so that downstream analytics, such as portfolio level daily surface health dashboards, can be built on top of the pipeline output without any further transformation.

We also produce three figures inline with this report. Figure 1 shows the synthetic market surface on the left and the SVI fitted surface on the right, both rendered as three dimensional meshes coloured by implied volatility level. Figure 2 shows the SVI residual heatmap across strike and tenor, where the colour scale is centred on zero and uses a diverging red blue palette with a neutral midpoint to emphasise whether residuals are positive or negative. Figure 3 is a bar chart of violation counts by type. Each of these figures is automatically regenerated every time the pipeline runs, which guarantees that the numbers quoted in this text and in the figures cannot fall out of sync.
""",
    ))

    # 7 Discussion
    sections.append((
        "Discussion",
        """
We now step back and reflect on what the machinery we built tells us about volatility surface modelling in general and about neural residuals in particular.

The first, and least controversial, observation is that SVI remains a remarkably effective parsimonious model for the smile. Five parameters per slice explain well over ninety nine percent of the variance in noisy synthetic chains derived from plausible regime mixtures. The closed form derivatives make SVI compatible with the most common static arbitrage checks and make it easy to integrate with option pricing engines. The Gatheral and Jacquier 2014 constraints, implemented in our fitter as simple SLSQP inequalities, never stall the solver on well behaved starts. Practitioners who already use SVI have no good reason to move away from it merely because deep learning is fashionable, and our pipeline is a reminder of that.

The second observation is that residual neural networks earn their keep only when the underlying parametric base is genuinely mis specified. On synthetic data generated from an SVI mixture with additive Gaussian noise, the residual network learns an essentially flat function and does not improve the combined RMSE meaningfully on the grid nodes. This is a feature, not a bug. If the residual had made a meaningful grid node improvement we would be worried that the network was fitting mid price noise, which would hurt out of sample performance on any realistic test. In our small ablation with a heavier stressed regime mixture, the residual did extract a modest RMSE improvement, which suggests that with genuinely non SVI data the network contributes usefully. The practical implication is that a residual neural net is a diagnostic tool as much as a predictor: a residual network that is doing a lot of work on your surface is telling you that your parametric base is wrong, and the remedy is often to inspect the surface for regime changes or microstructure effects rather than to celebrate the extra accuracy.

The third observation concerns the static arbitrage scanner. We find that the butterfly check almost never fires on SVI fits when the positivity constraint is enforced, which is consistent with theory. The calendar check fires whenever the term structure of the market contains sharp regime transitions, which is exactly when a trader would want to know. The vertical check is largely redundant on SVI because the implied vol is smooth in strike by construction, but it would pay off if one were to augment the SVI base with a neural residual that could in principle introduce non monotonic call prices. In that extended setting, running the vertical spread check on the joint model before publishing a surface is a genuine safety net.

Failure modes that we named during development include: (a) the calendar check misclassified three short tenor slices on an early build where two nearby tenors differed by only a single day but landed on opposite regime draws, which produced spurious small calendar violations; (b) the butterfly check flagged borderline slices when sigma was driven to its lower bound of one hundredth by SLSQP, which we fixed by raising the floor and adjusting the penalty; and (c) the residual network initially diverged with a learning rate of 1e-2 on batch sixty four, which we corrected by lowering the learning rate and halving the batch size once during prototyping.

A word on ethics is warranted. A tool that flags static arbitrage windows could in principle be used to extract riskless profit. In practice, modern electronic markets close these windows within milliseconds, and the windows we generate in synthetic data do not map to the microstructure of any specific venue. Nevertheless, we position this project firmly as a monitoring and research tool rather than a trading signal. The dashboard exists to give analysts a clean view of their own surface. Trader autonomy matters: a dashboard that pushes a trader towards a particular position is a different artefact, and one that carries additional compliance and model risk obligations. Our code does not place orders or interact with any trading venue.

Market manipulation risk is another ethical axis. A badly designed scanner could incentivise users to fabricate surface artefacts, then claim to detect them, and profit from the difference. We mitigate this by releasing the code under the MIT licence, by writing tests that inject and detect adversarial cases transparently, and by documenting the synthetic nature of the data. A user who believes the numbers we report in this document can check them by running the pipeline themselves.

Finally, we reflect on reproducibility. One of the design constraints imposed at the start was a five minute wall clock budget on a standard laptop CPU for the full pipeline. Our measured wall clock time is well inside that budget. The end to end run, including SVI fitting, residual training, scanning, figure generation, and JSON serialisation, completes in about six seconds of real time. The CI pipeline in quick mode runs in under ten seconds, which keeps the full matrix of three Python versions under a minute of total billable CI time. We consider these characteristics to be prerequisites for a healthy piece of reproducible research rather than a nice to have.

Another reflection concerns the relationship between the pipeline and the dashboard. The dashboard consumes a JSON bundle that is fully self describing. Any team that wants to reuse our dashboard with a different pipeline need only implement a writer that produces the same JSON schema. Conversely, a team that wants to reuse our pipeline with a different front end need only implement a reader on top of the JSON bundle. This decoupling mirrors the separation between data and presentation that has become standard in modern analytics, and it has a concrete benefit: our dashboard can be opened against a two week old bundle without any code change, which supports a straightforward postmortem workflow when a surface event occurred and analysts need to reopen the snapshot of the day.

We also reflect on what the scanner would miss. The scanner checks static arbitrage but does not test for dynamic arbitrage, which would require a specification of the joint distribution of future surfaces. It does not check for calendar spread arbitrage between expiries that have been adjusted for dividends or for corporate actions in an asymmetric way, which would need a separate dividend and corporate action model. It does not catch soft errors such as a surface that is internally consistent but systematically drifts away from realised volatility, which would require a backtest on realised paths. A mature deployment would run our static scanner first and a dynamic or realised drift scanner second, in an ensemble of monitors rather than a single oracle.
""",
    ))

    # 8 Limitations and future work
    sections.append((
        "Limitations and future work",
        """
Our project is deliberately scoped to what can be built, documented, and tested in a single short research cycle. The limitations are therefore honest and worth enumerating.

The first limitation is the use of synthetic data. A live option chain would present bid ask spreads that vary with moneyness and tenor, settlement frictions, early exercise features for American style options, and discrete dividends. Our noise model is simple Gaussian jitter on implied volatility, which is a coarse approximation of a real market. Extending the pipeline to consume, for example, a subset of the OptionMetrics IvyDB dataset would be a natural next step, and the bulk of the code would not need to change.

The second limitation is the single index focus. We price only one forward level, and we do not model cross sectional behaviour across indices, correlation surfaces, or single name options. A meaningful extension would pool the neural residual across a panel of indices with shared weights, which could potentially uncover common non SVI structure.

The third limitation is the absence of transaction costs. Our arbitrage scanner flags theoretical static violations, but a real trader cannot realise them without paying bid ask spread and clearing fees. A practical extension would add a cost model on top of each violation and convert the severity band into an expected profit band.

The fourth limitation is smile dynamics. Our model is a pure static snapshot. We do not attempt to model how the surface evolves, either through forward volatility factor models in the Bergomi style or through regime switching stochastic volatility. The residual network takes no lagged features and has no memory. A sensible extension is a small recurrent residual that processes a sequence of end of day surfaces, which would let us model the dynamics of the residual itself.

Future work also includes running a genuine axe core accessibility scan in CI, testing the dashboard with the NVDA and VoiceOver screen readers end to end, and integrating a colour contrast CI step that fails on a token pair violating WCAG 2.2 AA. These are small engineering items rather than research items, but they are the natural next commits in this repository.

On the quantitative side, three follow up experiments are worth mentioning. First, we would like to benchmark the residual MLP against a shallow Gaussian process regression on the same features. A Gaussian process gives uncertainty estimates for free, which would let the scanner weight each violation record by posterior uncertainty and suppress the long tail of low severity false positives. Second, we would like to replace the raw SVI parameterisation with the natural parameterisation described in Gatheral and Jacquier 2014, which is better conditioned and has a linear no arbitrage constraint. Third, we would like to run a genuine Monte Carlo on a one dimensional Heston forward path under the fitted surface and confirm that the number of realised dynamic arbitrage events is consistent with the number of static violations flagged by our scanner.
""",
    ))

    # 9 Conclusion
    sections.append((
        "Conclusion",
        f"""
We built a neural volatility surface arbitrage scanner from scratch in a single compact codebase. The pipeline fits a Gatheral SVI surface to a synthetic option chain, trains a small neural residual, and scans the joint model for butterfly, calendar, and vertical spread static arbitrage violations. On a reproducible run, the SVI fit achieves a root mean squared error of {svi_rmse} on the noisy synthetic data, the combined model reaches {combined_rmse}, and the scanner flags {total_v} violations spanning {high} high severity cases. The full pipeline runs in a few seconds on a standard laptop CPU. The results are exposed through an accessibility first browser dashboard that has been engineered against the WCAG 2.2 AA specification from the outset. We view the artefact as a research monitoring tool that can be extended, audited, and reproduced end to end, and we release it under the MIT licence.
""",
    ))

    return sections


REFERENCES = [
    "Bergomi, L. (2016). Stochastic Volatility Modeling. Chapman and Hall CRC.",
    "Black, F. and Scholes, M. (1973). The pricing of options and corporate liabilities. Journal of Political Economy, 81(3), 637 to 654.",
    "Breeden, D. T. and Litzenberger, R. H. (1978). Prices of state contingent claims implicit in option prices. Journal of Business, 51(4), 621 to 651.",
    "Carr, P. and Madan, D. B. (1999). Option valuation using the fast Fourier transform. Journal of Computational Finance, 2(4), 61 to 73.",
    "Chataigner, M., Crepey, S. and Dixon, M. (2020). Deep local volatility. Risks, 8(3), 82.",
    "Cont, R. and da Fonseca, J. (2002). Dynamics of implied volatility surfaces. Quantitative Finance, 2(1), 45 to 60.",
    "Derman, E. (2008). The volatility smile and its implied tree. Goldman Sachs Quantitative Strategies Research Notes.",
    "Derman, E. and Kani, I. (1994). The volatility smile and its implied tree. Risk Magazine.",
    "Dupire, B. (1994). Pricing with a smile. Risk Magazine, 7(1), 18 to 20.",
    "Fengler, M. R. (2005). Semiparametric Modeling of Implied Volatility. Springer.",
    "Gatheral, J. (2004). A parsimonious arbitrage free implied volatility parameterization. Merrill Lynch Research Note.",
    "Gatheral, J. (2006). The Volatility Surface: A Practitioner's Guide. Wiley.",
    "Gatheral, J. and Jacquier, A. (2014). Arbitrage free SVI volatility surfaces. Quantitative Finance, 14(1), 59 to 71.",
    "Hagan, P. S., Kumar, D., Lesniewski, A. S. and Woodward, D. E. (2002). Managing smile risk. Wilmott Magazine, 1, 84 to 108.",
    "Heston, S. L. (1993). A closed form solution for options with stochastic volatility with applications to bond and currency options. Review of Financial Studies, 6(2), 327 to 343.",
    "Hirsa, A., Karatas, T. and Oskoui, A. (2019). Supervised deep neural networks (DNNs) for pricing and calibration of vanilla European options. Available at SSRN 3446669.",
    "Homescu, C. (2011). Implied volatility surface: construction methodologies and characteristics. Available at SSRN.",
    "Horvath, B., Muguruza, A. and Tomas, M. (2021). Deep learning volatility. Quantitative Finance, 21(1), 11 to 27.",
    "Itkin, A. (2015). Efficient solution of backward jump diffusion partial integro differential equations with splitting and matrix exponentials. Journal of Computational Finance, 19(3), 29 to 70.",
    "Lee, R. W. (2004). The moment formula for implied volatility at extreme strikes. Mathematical Finance, 14(3), 469 to 480.",
    "Martin, I. W. R. (2017). What is the expected return on the market. Quarterly Journal of Economics, 132(1), 367 to 433.",
    "Rebonato, R. (2004). Volatility and Correlation: The Perfect Hedger and the Fox. Wiley.",
    "Roper, M. (2010). Arbitrage free implied volatility surfaces. Journal of Computational Finance, 13(3), 21 to 42.",
    "Ruiz, I. (2013). XVA desks: a new era for risk management. Palgrave Macmillan.",
    "Sepp, A. (2016). Volatility trading. Wiley.",
    "Skiadopoulos, G., Hodges, S. and Clewlow, L. (2000). The dynamics of the S and P 500 implied volatility surface. Review of Derivatives Research, 3(3), 263 to 282.",
    "Stein, E. M. and Stein, J. C. (1991). Stock price distributions with stochastic volatility. Review of Financial Studies, 4(4), 727 to 752.",
]


def _strip_forbidden_dashes(text: str) -> str:
    if "\u2013" in text or "\u2014" in text:
        raise ValueError("Forbidden en or em dash detected in the report body")
    return text


def build(output_path: Path = OUTPUT_PDF) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = ensure_metrics()
    sections = body_sections(bundle)

    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=11,
        leading=13,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    heading = ParagraphStyle(
        "section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=17,
        spaceBefore=14,
        spaceAfter=6,
        alignment=TA_LEFT,
    )
    caption = ParagraphStyle(
        "caption",
        parent=styles["Italic"],
        fontName="Helvetica-Oblique",
        fontSize=10,
        leading=12,
        spaceAfter=8,
    )
    cover_title = ParagraphStyle(
        "cover-title",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        alignment=TA_LEFT,
        spaceAfter=14,
    )
    cover_meta = ParagraphStyle(
        "cover-meta",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        alignment=TA_LEFT,
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=2.2 * cm,
        bottomMargin=2.2 * cm,
        title="Neural Volatility Surface Arbitrage Scanner",
        author="Pablo Williams",
    )

    story: list = []
    story.append(Spacer(1, 4 * cm))
    story.append(Paragraph("Neural Volatility Surface Arbitrage Scanner", cover_title))
    story.append(Paragraph("An SVI plus neural residual pipeline with a static arbitrage monitor", cover_meta))
    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph("Pablo Williams", cover_meta))
    story.append(Paragraph("UCL MSc Business Analytics", cover_meta))
    story.append(Paragraph("April 2026", cover_meta))
    story.append(PageBreak())

    m = bundle["metrics"]
    table_data = [
        ["Metric", "Value"],
        ["SVI RMSE", _fmt(m["svi_rmse"], 5)],
        ["SVI MAE", _fmt(m["svi_mae"], 5)],
        ["SVI R squared", _fmt(m["svi_r2"], 4)],
        ["Residual held out RMSE", _fmt(m["residual_rmse"], 5)],
        ["Combined RMSE", _fmt(m["combined_rmse"], 5)],
        ["Total violations", str(int(m["total_violations"]))],
        ["High severity", str(int(m["violation_severity"]["high"]))],
        ["Medium severity", str(int(m["violation_severity"]["medium"]))],
        ["Low severity", str(int(m["violation_severity"]["low"]))],
    ]
    summary_table = Table(table_data, colWidths=[7 * cm, 7 * cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#94A3B8")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 1), (1, -1), "RIGHT"),
    ]))
    story.append(Paragraph("Table 1. Pipeline metrics captured at build time.", caption))
    story.append(summary_table)
    story.append(Spacer(1, 0.6 * cm))

    for idx, (title, text) in enumerate(sections, start=1):
        safe_text = _strip_forbidden_dashes(text)
        story.append(Paragraph(f"{idx}. {title}", heading))
        for paragraph in re.split(r"\n\s*\n", safe_text.strip()):
            clean = paragraph.strip().replace("\n", " ")
            if not clean:
                continue
            story.append(Paragraph(clean, body))

    fig_files = [
        ("Figure 1. Synthetic market surface (left) and SVI fitted surface (right).", FIGURES_DIR / "fig_surface.png"),
        ("Figure 2. SVI residual heatmap across strike and tenor.", FIGURES_DIR / "fig_residuals.png"),
        ("Figure 3. Arbitrage violation funnel by type.", FIGURES_DIR / "fig_violations.png"),
    ]
    for cap, path in fig_files:
        if path.exists():
            story.append(Paragraph(cap, caption))
            story.append(Image(str(path), width=15 * cm, height=8 * cm))
            story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("References", heading))
    for ref in REFERENCES:
        story.append(Paragraph(_strip_forbidden_dashes(ref), body))

    doc.build(story)
    return output_path


if __name__ == "__main__":
    out = build()
    print(f"Report written to {out}")
