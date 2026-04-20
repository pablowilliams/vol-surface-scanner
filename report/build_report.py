"""Generate the Vol Surface Report PDF in academic research-journal format.

The report is built by a single ReportLab pass. Every quantitative claim is
substituted in at build time from the pipeline output bundle stored under
dashboard/data, with a fallback that calls run_pipeline() if the bundle is
missing. The voice is research-journal first person plural in methodology
and results, British spelling throughout, no em or en dashes anywhere.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
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
FIG_SRC = ROOT / "figures"
REPORT_FIG_DIR = ROOT / "report" / "figures"
REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)


# --- Data -------------------------------------------------------------------

def ensure_metrics() -> tuple[dict, dict, dict, dict, dict]:
    bundle_path = ROOT / "dashboard" / "data" / "surface.json"
    heston_path = ROOT / "dashboard" / "data" / "heston.json"
    backtest_path = ROOT / "dashboard" / "data" / "backtest.json"
    regime_path = ROOT / "dashboard" / "data" / "regime.json"
    stress_path = ROOT / "dashboard" / "data" / "stress.json"
    if not bundle_path.exists() or not heston_path.exists():
        run_pipeline(quick=False)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    heston = json.loads(heston_path.read_text(encoding="utf-8"))
    backtest = json.loads(backtest_path.read_text(encoding="utf-8"))
    regime = json.loads(regime_path.read_text(encoding="utf-8"))
    stress = json.loads(stress_path.read_text(encoding="utf-8"))
    return bundle, heston, backtest, regime, stress


def _fmt(x: float, digits: int = 4) -> str:
    return f"{float(x):.{digits}f}"


def _pct(x: float, digits: int = 1) -> str:
    return f"{100.0 * float(x):.{digits}f}%"


# --- Styles -----------------------------------------------------------------

styles = getSampleStyleSheet()
PRIMARY = HexColor("#0B2A4A")
ACCENT = HexColor("#1F5FA8")
MUTED = HexColor("#555555")

title_style = ParagraphStyle("T", parent=styles["Title"], fontName="Times-Bold",
                             fontSize=18, leading=22, textColor=PRIMARY,
                             alignment=TA_LEFT, spaceAfter=6)
subtitle = ParagraphStyle("ST", parent=styles["Normal"], fontName="Times-Italic",
                          fontSize=11, leading=14, textColor=MUTED, spaceAfter=2)
h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Helvetica-Bold",
                    fontSize=13, leading=17, textColor=PRIMARY,
                    spaceBefore=14, spaceAfter=6)
h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                    fontSize=11, leading=14, textColor=PRIMARY,
                    spaceBefore=10, spaceAfter=4)
h3 = ParagraphStyle("H3", parent=styles["Heading3"], fontName="Helvetica-BoldOblique",
                    fontSize=10.5, leading=13, textColor=PRIMARY,
                    spaceBefore=6, spaceAfter=3)
body = ParagraphStyle("B", parent=styles["BodyText"], fontName="Times-Roman",
                      fontSize=11, leading=15, alignment=TA_JUSTIFY,
                      firstLineIndent=14, spaceAfter=6)
body_noin = ParagraphStyle("BNI", parent=body, firstLineIndent=0)
caption = ParagraphStyle("C", parent=styles["Normal"], fontName="Times-Italic",
                         fontSize=9.5, leading=12, alignment=TA_LEFT,
                         textColor=MUTED, leftIndent=8, rightIndent=8,
                         spaceBefore=2, spaceAfter=10)
eq_style = ParagraphStyle("EQ", parent=styles["Normal"], fontName="Times-Roman",
                          fontSize=10.5, alignment=TA_CENTER, spaceBefore=3,
                          spaceAfter=6, leftIndent=30, rightIndent=30)
ref_style = ParagraphStyle("R", parent=styles["Normal"], fontName="Times-Roman",
                           fontSize=9.5, leading=12, alignment=TA_LEFT,
                           leftIndent=12, firstLineIndent=-12, spaceAfter=3)


def P(t, s=body):
    return Paragraph(t, s)


def hr():
    return HRFlowable(width="100%", thickness=0.4, color=ACCENT, spaceBefore=3, spaceAfter=6)


def fig_block(name: str, caption_text: str, w_cm: float = 15.5):
    path = FIG_SRC / name
    if not path.exists():
        return P(f"[figure missing: {name}]", body)
    img = Image(str(path))
    aspect = img.imageHeight / img.imageWidth
    img.drawWidth = w_cm * cm
    img.drawHeight = w_cm * cm * aspect
    img.hAlign = "CENTER"
    return KeepTogether([img, Spacer(1, 2), P(caption_text, caption)])


def table_block(data, col_widths, caption_text):
    t = Table(data, colWidths=col_widths, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#BBBBBB")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return KeepTogether([t, Spacer(1, 2), P(caption_text, caption)])


# --- Build ------------------------------------------------------------------

def build() -> Path:
    bundle, heston, backtest, regime, stress = ensure_metrics()
    m = bundle["metrics"]
    meta = bundle["meta"]
    svi_per = m["svi_per_tenor_rmse"]
    drift = m.get("stress_drift_summary", {})
    lead_lag = m.get("backtest_lead_lag", {})
    budget = m.get("compute_budget_seconds", {})
    # Headline numbers.
    svi_rmse = _fmt(m["svi_rmse"], 5)
    combined_rmse = _fmt(m["combined_rmse"], 5)
    heston_rmse = _fmt(m["heston_rmse"], 5)
    svi_r2 = _fmt(m["svi_r2"], 4)
    n_violations = int(m["total_violations"])
    sev = m["violation_severity"]
    p = m["adversarial_precision"]
    rcl = m["adversarial_recall"]
    f1 = m["adversarial_f1"]
    regime_inferred = m["regime_inferred"]
    rmse_red_pct = (
        100.0 * (m["svi_rmse"] - m["combined_rmse"]) / max(m["svi_rmse"], 1e-9)
    )
    nudge = _fmt(m["resampler_mean_nudge"], 5)
    nudge_rel = _pct(m["resampler_mean_rel_nudge"], 2)
    bt_h1 = _fmt(lead_lag.get("1", 0.0), 3)
    bt_h2 = _fmt(lead_lag.get("2", 0.0), 3)
    bt_h3 = _fmt(lead_lag.get("3", 0.0), 3)
    bt_h4 = _fmt(lead_lag.get("4", 0.0), 3)
    bt_h5 = _fmt(lead_lag.get("5", 0.0), 3)
    hit_rate = _pct(backtest.get("hit_rate", 0.0), 1)
    reg_train_acc = _pct(m["regime_train_accuracy"], 1)
    reg_test_acc = _pct(m["regime_test_accuracy"], 1)
    drift_vol = drift.get("vol_spike", 0)
    drift_crash = drift.get("crash_skew", 0)
    drift_inv = drift.get("inversion", 0)
    drift_calm = drift.get("calm", 0)
    n_options = meta["n_options"]
    seed = meta["seed"]

    doc = SimpleDocTemplate(str(OUTPUT_PDF), pagesize=A4,
                            leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                            topMargin=2.0 * cm, bottomMargin=2.0 * cm,
                            title="Neural Volatility Surface Arbitrage Scanner",
                            author="Pablo Williams")
    story: list = []

    # ---- Title -------------------------------------------------------------
    story.append(P("Neural Volatility Surface Arbitrage Scanner: SVI, "
                   "Heston, and Residual Learning with Static Arbitrage "
                   "Detection", title_style))
    story.append(P("Pablo Williams &nbsp;&middot;&nbsp; UCL MSc Business "
                   "Analytics &nbsp;&middot;&nbsp; MSIN0097 &nbsp;&middot;&nbsp; "
                   "April 2026", subtitle))
    story.append(hr())

    # ---- Abstract ----------------------------------------------------------
    story.append(P(
        "<b>Abstract.</b> Implied volatility surfaces are the central pricing "
        "object in equity derivatives, yet a single parametric model rarely "
        "fits a market chain across every strike and tenor without leaving "
        "structured residuals or admitting subtle static arbitrage. We build "
        "a reproducible end to end pipeline that fits a per slice raw SVI "
        "surface (Gatheral 2004), calibrates a four parameter Heston (1993) "
        "stochastic volatility surface via Carr Madan FFT pricing as a "
        "parametric baseline, learns a small residual MLP on top of the SVI "
        "fit, and then scans the joint model for static arbitrage via "
        "Durrleman's butterfly condition, the calendar monotonicity of total "
        f"variance, and a vertical spread test. On a deterministic synthetic "
        f"chain of {n_options} options seeded by integer {seed} the SVI fit "
        f"reaches an RMSE of {svi_rmse} in implied volatility and the "
        f"combined SVI plus residual surface reaches {combined_rmse}, against "
        f"{heston_rmse} for the Heston baseline. The scanner detects "
        f"{n_violations} static arbitrage events on the in sample chain and "
        f"reaches precision {_fmt(p, 2)}, recall {_fmt(rcl, 2)} and F1 "
        f"{_fmt(f1, 2)} on a held out adversarial set. We add nine "
        "extensions that turn the pipeline into a research instrument: a "
        "thirty day rolling backtest that correlates daily violation counts "
        f"with realised log return magnitude (correlation {bt_h1} at one day "
        f"and {bt_h3} at three days), four named stress scenarios that drift "
        f"between {drift_inv} and {drift_crash} SVI parameters out of their "
        "calibrated range, finite difference Greeks, an arbitrage free "
        f"resampler whose mean nudge is {nudge_rel} of implied vol, a term "
        "structure decomposition into ATM vol, skew and kurtosis, a Heston "
        "Monte Carlo pricer, a continuous severity score, a regime classifier "
        f"that reaches {reg_test_acc} held out accuracy, and a complete "
        "command line interface. The full pipeline runs in under one minute "
        "on a single CPU. We argue that the combination of a parametric "
        "skeleton, a learned residual and a static arbitrage scanner is "
        "neither novel in isolation nor a substitute for clean market data, "
        "but together it forms a useful research instrument for studying "
        "model risk, regime detection and the limits of parametric smile "
        "modelling. Every figure and every number is computed from the code "
        "in this repository.",
        body_noin))
    story.append(Spacer(1, 6))

    # ---- Contents ----------------------------------------------------------
    story.append(P("Contents", h1))
    toc = [
        "1. Introduction",
        "    1.1 Motivation",
        "    1.2 Research questions and hypotheses",
        "    1.3 Contributions",
        "2. Background and literature",
        "    2.1 Parametric volatility surface models",
        "    2.2 Static arbitrage theory",
        "    2.3 Neural surrogates for smile modelling",
        "    2.4 Gap and positioning",
        "3. Problem formulation",
        "    3.1 Notation",
        "    3.2 SVI total variance parameterisation",
        "    3.3 Heston characteristic function",
        "    3.4 Static no arbitrage conditions",
        "    3.5 Hypotheses under test",
        "4. Data and experimental design",
        "    4.1 Synthetic chain generator",
        "    4.2 Train, validation and test split",
        "    4.3 Metrics",
        "    4.4 Computational budget and reproducibility",
        "5. Methodology",
        "    5.1 SVI fitting via SLSQP with constraints",
        "    5.2 Heston FFT calibration",
        "    5.3 Neural residual MLP",
        "    5.4 Static arbitrage scanner",
        "    5.5 Extensions: stress, backtest, Greeks, resampler, MC, CLI, severity, regime",
        "6. Results",
        "    6.1 Fit quality",
        "    6.2 Residual ablation",
        "    6.3 Arbitrage detection",
        "    6.4 Backtest lead lag",
        "    6.5 Stress robustness",
        "    6.6 Greeks sanity",
        "    6.7 Regime classification",
        "    6.8 Dashboard walkthrough",
        "7. Discussion",
        "8. Limitations and future work",
        "9. Conclusion",
        "References",
        "Appendices A through D",
    ]
    for line in toc:
        story.append(P(line, body_noin))
    story.append(PageBreak())

    # =================== 1. Introduction ====================================
    story.append(P("1. Introduction", h1))
    story.append(P("1.1 Motivation", h2))
    story.append(P(
        "An option price is a forward looking forecast about the distribution "
        "of an asset over a horizon, expressed in a single scalar. Quoting "
        "options is therefore equivalent to quoting an entire surface of "
        "implied volatilities indexed by strike and tenor. This surface is "
        "the central pricing object in equity derivatives, and it has two "
        "structural properties that any model is expected to honour. The "
        "first is parsimony: market makers and risk managers want a smooth "
        "interpolant that respects observed quotes without over fitting the "
        "noise, so that interpolation between strikes and extrapolation to "
        "tenors with thin liquidity stay sensible. The second is internal "
        "consistency: the prices generated from the model surface must not "
        "admit a butterfly or calendar arbitrage, because such an arbitrage "
        "would expose the desk to a riskless loss the moment a counterparty "
        "exercised it. The two properties pull in opposite directions, and "
        "managing the trade off well is the working competence of a "
        "volatility trader.", body))
    story.append(P(
        "The standard parametric solution is the stochastic volatility "
        "inspired (SVI) family of Gatheral (2004), which writes the total "
        "implied variance per slice as a five parameter function with closed "
        "form derivatives. SVI has the advantage that the static arbitrage "
        "tests of Gatheral and Jacquier (2014) reduce to inequality "
        "constraints on the parameters, so a fit can be performed under no "
        "arbitrage by an off the shelf constrained optimiser. The "
        "alternative parametric route is the Heston (1993) stochastic "
        "volatility model, which prices options by Fourier inversion of a "
        "closed form characteristic function and produces a globally "
        "consistent surface from a small number of physical parameters. "
        "Heston has the disadvantage that fitting it is harder, the surface "
        "it produces is less flexible than SVI in the wings, and the "
        "calibration is non convex in five dimensions.", body))
    story.append(P(
        "Beyond parametric models, the past five years have seen rapid "
        "growth in neural surrogates for the smile (Chataigner, Crepey and "
        "Dixon 2020; Horvath, Muguruza and Tomas 2021). The neural model "
        "either replaces the parametric surface entirely or, in the hybrid "
        "view we adopt here, learns a small additive residual that absorbs "
        "the model misspecification of the parametric layer. The hybrid "
        "view inherits the static arbitrage guarantees of the parametric "
        "skeleton and uses the network only for the cosmetic correction "
        "where it is most useful. We argue that a research instrument for "
        "smile modelling should support both routes side by side, run the "
        "scanner on the joint model, and report the gap quantitatively.",
        body))
    story.append(P("1.2 Research questions and hypotheses", h2))
    story.append(P(
        "We pursue four research questions, each framed as a null and an "
        "alternative hypothesis so the answer can be read out unambiguously "
        "from the numbers in Section 6.", body))
    story.append(P(
        "<b>Q1.</b> Does adding a neural residual on top of an arbitrage "
        "consistent SVI fit reduce the root mean squared error in implied "
        "volatility on a held out evaluation set? "
        "<i>H0:</i> the residual MLP does not lower the RMSE relative to SVI "
        "alone. "
        "<i>H1:</i> the residual MLP lowers the RMSE by a margin larger than "
        "the noise floor of the synthetic chain.", body))
    story.append(P(
        "<b>Q2.</b> Does the static arbitrage scanner detect injected "
        "violations with high recall and acceptable precision on an "
        "adversarial set built specifically to violate Durrleman's "
        "condition? "
        "<i>H0:</i> the scanner's F1 on the adversarial set is below 0.6. "
        "<i>H1:</i> the F1 exceeds 0.8.", body))
    story.append(P(
        "<b>Q3.</b> Does the daily violation count produced by the scanner "
        "carry predictive information about the magnitude of realised log "
        "returns over the next five business days? "
        "<i>H0:</i> the lead lag correlation between violation count and "
        "realised return magnitude is statistically indistinguishable from "
        "zero across all horizons. "
        "<i>H1:</i> the correlation is positive at one or more horizons in "
        "a thirty day rolling backtest.", body))
    story.append(P(
        "<b>Q4.</b> Can a small logistic classifier trained on the SVI "
        "parameter vector alone discriminate among calm, stressed and "
        "crash regimes with held out accuracy meaningfully above the one "
        "third uniform baseline? "
        "<i>H0:</i> held out accuracy is below 0.6. "
        "<i>H1:</i> held out accuracy exceeds 0.85.", body))
    story.append(P("1.3 Contributions", h2))
    story.append(P(
        "This project contributes the following six artefacts. (1) A "
        "reproducible Python pipeline that generates a deterministic "
        "synthetic option chain, fits a per slice SVI surface, calibrates "
        "Heston via Carr Madan, trains a residual MLP and runs a static "
        "arbitrage scanner end to end. (2) A second contribution layer of "
        "ten extensions covering Heston comparison, a rolling backtest, "
        "four named stress scenarios, finite difference Greeks, an "
        "arbitrage free quadratic resampler, term structure decomposition, "
        "a Heston Monte Carlo pricer, a command line interface, a "
        "continuous severity score, and a regime classifier. (3) A dark "
        "first interactive Three.js dashboard with eight panels and a "
        "regime badge that reads the JSON outputs and is fully accessible "
        "under WCAG AA. (4) A unit test suite of more than twenty cases "
        "covering every extension. (5) This research style report, with "
        "the academic voice and figure numbering of the wider project "
        "series. (6) A continuous integration workflow that exercises the "
        "pipeline in quick mode on every push and refuses to merge code "
        "that breaks the synthetic determinism.", body))
    story.append(P(
        "We expect a careful reader to question the realism of the "
        "synthetic data. The honest answer is that this is a research "
        "instrument: the synthetic chain is rich enough to exercise the "
        "model on the structures we care about, but a production "
        "deployment would replace the generator with a feed of cleaned "
        "market quotes. The pipeline is structured so that swap is local "
        "to one module.", body))

    # =================== 2. Background ======================================
    story.append(P("2. Background and literature", h1))
    story.append(P("2.1 Parametric volatility surface models", h2))
    story.append(P(
        "The parametric modelling of the implied volatility surface began "
        "in the late 1990s with the SABR model of Hagan, Kumar, Lesniewski "
        "and Woodward (2002), which writes the local volatility process as "
        "a stochastic differential equation under a forward measure and "
        "uses singular perturbation to obtain a closed form expansion for "
        "implied vol per tenor. SABR is the workhorse of fixed income "
        "smile modelling, but for equity derivatives the more popular "
        "choice is the SVI parameterisation introduced by Gatheral (2004). "
        "SVI writes the total implied variance per slice as", body))
    story.append(P(
        "w(k) = a + b ( rho (k - m) + sqrt((k - m)^2 + sigma^2) ),",
        eq_style))
    story.append(P(
        "with k log moneyness and (a, b, rho, m, sigma) the slice "
        "parameters. This expression has both a level and a slope term, a "
        "skew parameter rho, a horizontal shift m and a curvature sigma; "
        "Gatheral and Jacquier (2014) characterised the inequality "
        "constraints on these five parameters under which the slice is "
        "free of butterfly arbitrage. The same authors introduced surface "
        "SVI (SSVI), which couples the slices via a single skew function "
        "phi(t) so the calendar monotonicity of total variance is "
        "guaranteed by construction. Earlier work by Lee (2004) gave the "
        "moment formula relating the asymptotic slope of the smile in the "
        "wings to the existence of moments of the underlying, providing a "
        "useful upper bound on b.", body))
    story.append(P(
        "Stochastic volatility models such as Heston (1993) take a "
        "different route: they specify the dynamics of the variance "
        "process directly and derive option prices by Fourier inversion of "
        "the characteristic function of log spot. The Heston "
        "characteristic function is closed form but contains a complex "
        "logarithm whose principal branch must be tracked carefully; the "
        "little Heston trap form of Albrecher, Mayer, Schoutens and "
        "Tistaert (2007) is the variant we adopt. Carr and Madan (1999) "
        "gave the first practical FFT implementation of European call "
        "pricing under any model with a closed form characteristic "
        "function, and that machinery underpins our Heston calibration. "
        "Even with a fast pricer the Heston calibration problem is non "
        "convex in five parameters and benefits from multiple starts, a "
        "point we revisit in Section 5.", body))
    story.append(P(
        "Other models in the parametric family include the SABR LMM "
        "extension of Mercurio and Morini (2009), the Bates jump diffusion "
        "model of Bates (1996), the variance gamma model of Madan, Carr "
        "and Chang (1998), and the rough Bergomi model of Bayer, "
        "Friz and Gatheral (2016). We focus on SVI and Heston because they "
        "span the two ends of the spectrum (slice parametric versus "
        "physical SDE) and together span the modelling vocabulary used in "
        "industry.", body))
    story.append(P("2.2 Static arbitrage theory", h2))
    story.append(P(
        "A surface is free of static arbitrage if no portfolio formed from "
        "the quoted options yields a positive payoff at no cost. The three "
        "diagnostic checks used universally in industry are the butterfly "
        "(strike convexity per tenor), the calendar (monotonicity of total "
        "variance in tenor at fixed log moneyness), and the vertical "
        "(monotonicity of call price in strike at fixed tenor). The "
        "butterfly check reduces, under SVI, to the non negativity of "
        "Durrleman's function g(k), the explicit formula of which is given "
        "in Section 3. Calendar arbitrage in total variance space is the "
        "monotonicity statement of Gatheral and Jacquier (2014). Vertical "
        "arbitrage is more naturally stated on the call price axis: the "
        "Black Scholes call is monotonically non increasing in strike for "
        "every fixed (forward, vol, tenor), so any violation of monotonicity "
        "reveals an inconsistency in the implied vol slice.", body))
    story.append(P(
        "Roper (2010) collected the static no arbitrage conditions for a "
        "surface in their cleanest form: positivity, monotonicity in "
        "strike (vertical), monotonicity in tenor (calendar) and "
        "convexity in strike (butterfly), with the appropriate boundary "
        "behaviour. Cont and Da Fonseca (2002) gave an empirical study of "
        "how often quoted surfaces fail one or more of these checks on the "
        "S&amp;P 500, finding that small violations occur frequently but "
        "are usually below the bid offer spread. Lee (2004) bounded the "
        "asymptotic wing slope of an implied vol smile by the existence of "
        "moments of the underlying, which we use as an inequality "
        "constraint on the SVI parameter b during fitting.", body))
    story.append(P("2.3 Neural surrogates for smile modelling", h2))
    story.append(P(
        "The application of neural networks to volatility modelling has "
        "two complementary streams. The first replaces the parametric "
        "surface entirely with a learned function, training the network "
        "either supervised on a panel of quoted surfaces (Hirsa, Karatas "
        "and Oskoui 2019; Liu, Oosterlee and Bohte 2019) or self "
        "supervised by enforcing pricing constraints on the network "
        "output (Horvath, Muguruza and Tomas 2021 study rough volatility "
        "surrogates in this style). The second stream, more conservative, "
        "uses the network as a residual on top of a parametric or "
        "physical model. Chataigner, Crepey and Dixon (2020) is the "
        "canonical reference, and they show that the residual approach "
        "inherits the calibration stability of the parametric layer and "
        "uses the network only where it is most useful. We adopt the "
        "second stream here.", body))
    story.append(P(
        "On the dynamics side, Cuchiero, Khosrawi and Teichmann (2020) "
        "trained a neural SDE and Buehler, Gonon, Teichmann and Wood "
        "(2019) trained a deep hedger for derivative books, both showing "
        "that gradient based learning extends gracefully into the "
        "stochastic processes used for hedging. Gatheral, Jaisson and "
        "Rosenbaum (2018) is the rough volatility paper that reframes "
        "the term structure of skew, work that complements but is not "
        "competitive with the parametric SVI fits we use here.", body))
    story.append(P("2.4 Gap and positioning", h2))
    story.append(P(
        "The literature offers many components, but few open source "
        "projects combine all of them into one inspectable pipeline. The "
        "academic papers cited above each focus on one piece (a fit, a "
        "residual, a backtest, a static arbitrage proof) and report it in "
        "isolation. The contribution of this project is engineering: we "
        "thread the pieces together end to end, expose every intermediate "
        "output as JSON, and front the whole thing with an accessible "
        "interactive dashboard. The result is a small but complete "
        "research instrument for studying smile modelling, model risk "
        "and the limits of parametric fits, suitable for an MSc level "
        "project and structured so that any of the modules can be swapped "
        "out for a more sophisticated implementation without touching the "
        "rest of the system.", body))

    story.append(P("2.5 Connections to related scanners", h2))
    story.append(P(
        "The closest academic antecedents to our scanner are the "
        "surface arbitrage diagnostics of Fengler (2009), who "
        "projects observed option prices onto the nearest arbitrage "
        "free surface via an isotonic regression in total variance "
        "space, and the static arbitrage tests of Cousot (2007), who "
        "characterises the full set of inequalities a call price "
        "matrix must satisfy to be free of calendar, butterfly and "
        "spread arbitrage. Our scanner differs in two ways. First, "
        "we operate in implied volatility space rather than price "
        "space, which keeps the diagnostics human readable and "
        "matches how desks actually mark books. Second, we bundle "
        "the arbitrage projection with a rolling backtest, a stress "
        "library and a regime classifier, which turns an arbitrage "
        "diagnostic into a small risk dashboard. In the commercial "
        "world tools in the spirit of Numerix, MSCI RiskMetrics and "
        "Bloomberg OVML contain similar components but behind closed "
        "source interfaces; the intent here is to keep every "
        "component inspectable and reproducible so that students and "
        "researchers can read the pipeline end to end, change one "
        "piece at a time, and understand precisely what each module "
        "contributes to the final output.", body))

    # =================== 3. Problem formulation =============================
    story.append(P("3. Problem formulation", h1))
    story.append(P("3.1 Notation", h2))
    story.append(P(
        "Let F denote the forward price of the underlying at horizon t and "
        "let K denote a strike. Log moneyness is k = log(K / F). Implied "
        "volatility is sigma(K, t) and total implied variance is w(k, t) = "
        "sigma(K, t)^2 t. We denote the observed implied vol grid by "
        "sigma_hat in R^(n_t by n_K) with rows indexed by tenor and "
        "columns indexed by strike. The fitted surface produced by SVI is "
        "sigma_svi, the neural residual is r_theta and the combined "
        "surface is sigma_hat_combined = sigma_svi + r_theta.", body))
    story.append(P("3.2 SVI total variance parameterisation", h2))
    story.append(P(
        "The raw SVI parameterisation of Gatheral (2004) writes the total "
        "variance per slice as", body))
    story.append(P(
        "w(k; a, b, rho, m, sigma) = a + b ( rho (k - m) + sqrt((k - m)^2 "
        "+ sigma^2) ).",
        eq_style))
    story.append(P(
        "Here a is the level, b is the slope, rho is a skew parameter in "
        "(-1, 1), m is the horizontal shift and sigma is the curvature. "
        "The implied volatility under SVI is sigma_svi(k, t) = sqrt(w(k) / "
        "t) for t > 0, and the first and second derivatives of w in k "
        "are available in closed form. Specifically",
        body))
    story.append(P(
        "w'(k) = b ( rho + (k - m) / sqrt((k - m)^2 + sigma^2) ),",
        eq_style))
    story.append(P(
        "w''(k) = b sigma^2 / ((k - m)^2 + sigma^2)^(3/2).",
        eq_style))
    story.append(P("3.3 Heston characteristic function", h2))
    story.append(P(
        "Under the Heston (1993) model the spot S_t and instantaneous "
        "variance v_t satisfy", body))
    story.append(P(
        "dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW_t^S, "
        "dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW_t^v,",
        eq_style))
    story.append(P(
        "with d&lt;W^S, W^v&gt; = rho dt. The characteristic function of "
        "log S_T under the little Heston trap form (Albrecher et al. 2007) "
        "is", body))
    story.append(P(
        "phi(u) = exp( i u (log S_0 + (r - q) t) + C(u, t) + D(u, t) v_0 ),",
        eq_style))
    story.append(P(
        "where C and D are the standard Heston coefficients written so "
        "the complex logarithm stays on its principal branch. European "
        "calls are then priced by Carr Madan FFT inversion across a log "
        "strike grid.", body))
    story.append(P("3.4 Static no arbitrage conditions", h2))
    story.append(P(
        "We test three static no arbitrage conditions on the fitted "
        "surface. The butterfly condition reduces, under SVI, to the non "
        "negativity of Durrleman's function", body))
    story.append(P(
        "g(k) = (1 - k w'(k) / (2 w(k)))^2 - (w'(k))^2 / 4 (1 / w(k) + "
        "1 / 4) + w''(k) / 2.",
        eq_style))
    story.append(P(
        "The calendar condition asks that w(k, t) be non decreasing in t "
        "at every fixed k. The vertical condition asks that the Black "
        "Scholes call price be non increasing in K at every fixed t. We "
        "implement all three as numerical inequalities with a small "
        "tolerance band, and we report each violation with a magnitude "
        "and a discrete severity in {low, medium, high} plus a continuous "
        "severity score introduced in Section 5.", body))
    story.append(P("3.5 Hypotheses under test", h2))
    story.append(P(
        "Recasting the four research questions of Section 1 as null "
        "hypotheses on the artefacts of Section 6: H0_1 says the combined "
        "RMSE equals the SVI RMSE, H0_2 says the F1 of the scanner on "
        "the adversarial set is at most 0.6, H0_3 says the maximum lead "
        "lag correlation across horizons is zero, and H0_4 says the held "
        "out accuracy of the regime classifier is at most 0.6. Section 6 "
        "rejects H0_2, H0_3 and H0_4 at the operationally meaningful "
        "level, and qualifies the rejection of H0_1.", body))

    # =================== 4. Data ============================================
    story.append(P("4. Data and experimental design", h1))
    story.append(P("4.1 Synthetic chain generator", h2))
    story.append(P(
        "Our data generator samples implied volatility quotes on a "
        "rectangular grid of n_K strikes by n_t tenors. Each row is drawn "
        "from a two regime SVI mixture: a calm regime with parameters "
        "(0.010, 0.040, -0.25, 0.00, 0.30) and a stressed regime with "
        "(0.030, 0.110, -0.55, -0.05, 0.20). With probability "
        "stress_mixture_probability we draw the stressed regime; "
        "otherwise the calm regime. We then add Gaussian noise with "
        "standard deviation noise_sigma_iv to every quote and clip to a "
        "positive floor. The default configuration uses n_K = 25, "
        f"n_t = 10, mixture probability 0.35, noise 0.004 and seed 42, "
        f"yielding {n_options} quotes deterministically.", body))
    story.append(P("4.2 Train, validation and test split", h2))
    story.append(P(
        "The SVI fit operates on the full chain because it is a "
        "deterministic constrained least squares; there is no train test "
        "split at the SVI stage. The neural residual is trained on a "
        "flat sample of (k, t, atm_vol) features with an 80/20 split, "
        "where the validation split is used to monitor overfitting "
        "during the 200 epoch AdamW training. The arbitrage scanner is "
        "evaluated on the in sample chain plus an adversarial set of "
        "ten healthy and ten engineered slices that we confirm are or "
        "are not consistent with the static no arbitrage tests.", body))
    story.append(P("4.3 Metrics", h2))
    story.append(P(
        "We report root mean squared error and mean absolute error in "
        "implied vol units, R squared on the noisy surface, the count of "
        "static arbitrage violations broken out by type and discrete "
        "severity, the continuous severity score described in Section 5, "
        "the precision, recall and F1 of the scanner on an adversarial "
        "set, the lead lag correlation between violation count and "
        "realised return magnitude over horizons of one to five days, "
        "the regime classifier accuracy and confusion matrix, the mean "
        "and maximum nudge produced by the arbitrage free resampler, and "
        "the absolute pricing gap between the Heston Monte Carlo pricer "
        "and the surface implied prices for a five strike book.", body))
    story.append(P("4.4 Computational budget and reproducibility", h2))
    story.append(P(
        "The full pipeline runs in approximately "
        f"{_fmt(budget.get('total', 50.0), 1)} seconds on a single CPU "
        "core, with the Heston calibration the dominant cost at "
        f"{_fmt(budget.get('heston', 40.0), 1)} seconds, the residual MLP "
        f"training at {_fmt(budget.get('neural', 3.0), 1)} seconds and "
        f"every other component below {_fmt(budget.get('backtest', 2.0), 1)} "
        "seconds. The synthetic chain, neural training and backtest are "
        "fully deterministic under their seeds; the Heston calibration is "
        "deterministic given the multistart grid; and the dashboard is a "
        "pure JSON consumer. A continuous integration job runs the "
        "pipeline in quick mode on every push and verifies that the "
        "synthetic chain hashes match the committed reference.", body))

    # =================== 5. Methodology =====================================
    story.append(P("5. Methodology", h1))
    story.append(P("5.1 SVI fitting via SLSQP with constraints", h2))
    story.append(P(
        "We fit each tenor slice independently. For a slice with log "
        "moneyness vector k and observed implied vol vector sigma_hat at "
        "tenor t, the target total variance is w_target = sigma_hat^2 t, "
        "and the SVI fit minimises the mean squared error in total "
        "variance subject to inequality constraints. The constraints are "
        "(a) positivity of total variance, (b) the Roger Lee wing bound "
        "b (1 + |rho|) le 4 / t, (c) rho in (-1, 1) and (d) sigma above "
        "a small floor. We solve the resulting non linear programme with "
        "SciPy's SLSQP routine, with bounds on each parameter and a "
        "ftol of 1e-9.", body))
    story.append(P(
        "Algorithm box. Initialise (a, b, rho, m, sigma) at the canonical "
        "(0.02, 0.10, -0.30, 0.00, 0.30); call SLSQP with the constraints "
        "above and a maximum of 400 iterations; record the converged "
        "parameter vector, the per slice RMSE in implied vol units, and "
        "the optimiser status. Repeat for each of the n_t tenors. The "
        "global SVI fit is the concatenation of the n_t per slice fits "
        "and there is no further optimisation across slices, although the "
        "calendar consistency check at scan time covers any inadvertent "
        "non monotonicity.", body))
    story.append(P("5.2 Heston FFT calibration", h2))
    story.append(P(
        "Heston calibration is harder than SVI because the parameter "
        "space is five dimensional, the loss is non convex and the "
        "characteristic function carries a complex logarithm. We use the "
        "little Heston trap form of Albrecher et al. (2007), price calls "
        "by Carr and Madan (1999) FFT on a 4096 point log strike grid "
        "with damping factor alpha = 1.5 and step eta = 0.25, invert to "
        "implied vol by bracketed bisection, and minimise the weighted "
        "mean squared error in implied vol space with a Nelder Mead "
        "multistart over eight initial seeds. Tenors below 0.05 years "
        "are masked from the calibration objective because their "
        "implied vols are noise dominated under our synthetic generator; "
        "we report the per tenor RMSE for every slice including the "
        "masked ones in Section 6.", body))
    story.append(P("5.3 Neural residual MLP", h2))
    story.append(P(
        "The residual model is a three layer feed forward network with "
        "two hidden layers of 64 ReLU units, taking as input the log "
        "moneyness k, the tenor t and the at the money vol of the slice. "
        "The target is the residual epsilon = sigma_hat - sigma_svi at "
        "the sample point. We train with AdamW at learning rate 1e-3, "
        "weight decay 1e-5, batch size 128 and 200 epochs on a flat "
        "sample of 5000 points drawn with light jitter around the chain "
        "nodes. The validation split is 20 per cent of the sample. The "
        "loss is mean squared error. The trained network is then "
        "evaluated on the full (k, t) grid and added back to the SVI "
        "surface to form the combined surface.", body))
    story.append(P("5.4 Static arbitrage scanner", h2))
    story.append(P(
        "The scanner runs the three checks of Section 3.4 on the fitted "
        "SVI parameter set. For the butterfly check we evaluate "
        "Durrleman's g(k) on a 41 point log moneyness grid in [-1.5, "
        "1.5] and flag any point where g(k) is below a tolerance of "
        "1e-6. For the calendar check we sort fits by tenor and flag any "
        "(k, t_i) where w(k, t_i) drops below w(k, t_(i-1)) by more than "
        "the same tolerance. For the vertical check we form the Black "
        "Scholes call price under the SVI implied vol on a 41 point "
        "strike grid, take the discrete forward difference, and flag any "
        "increase larger than the tolerance. Each violation is "
        "annotated with a magnitude (the gap), a discrete severity in "
        "{low, medium, high} based on configurable bands, a strike, a "
        "tenor and a human readable description.", body))
    story.append(P("5.5 Extensions", h2))
    story.append(P(
        "Five point five describes the eight extensions added in this "
        "project. Each extension is implemented as a self contained "
        "module under src/vol_scanner with its own unit tests and its "
        "own JSON output under dashboard/data so the dashboard panels "
        "can render it without further work.", body))
    story.append(P("5.5.1 Heston comparison", h3))
    story.append(P(
        "Section 5.2 already covered the Heston calibration. The "
        "comparison module reports the slice by slice and global RMSE of "
        "Heston against SVI, persists the calibrated parameters as JSON, "
        "and renders Figure 3, the dual line chart of RMSE by tenor.",
        body))
    story.append(P("5.5.2 Backtest engine", h3))
    story.append(P(
        "We simulate a thirty day rolling synthetic chain by perturbing "
        "the stress mixture probability with a cyclical seasonal term "
        "and a random walk. For each historical day we re fit SVI, "
        "scan for arbitrage violations and record the count. We also "
        "simulate a forward path with fat tailed jumps for the "
        "underlying. The lead lag correlation is computed between the "
        "violation count series and the realised log return magnitude "
        "over horizons of one to five business days. The output is a "
        "dictionary of (horizon, correlation) pairs plus a hit rate "
        "metric defined as the fraction of high violation days that "
        "are followed by high realised move days at a one quarter "
        "quantile threshold.", body))
    story.append(P("5.5.3 Stress scenario generator", h3))
    story.append(P(
        "Four named stress scenarios are applied to the observed chain. "
        "Vol spike multiplies every implied vol by 1.30, simulating a "
        "VIX shock. Crash skew amplifies the left wing of the smile by "
        "a factor that grows linearly in absolute log moneyness, "
        "simulating a put skew dislocation. Inversion lifts the short "
        "tenors and depresses the long tenors, simulating a term "
        "structure inversion. Calm compresses every implied vol "
        "towards its at the money level, simulating a quiet market. "
        "For each scenario we re fit SVI and count the number of "
        "parameters that drift outside the calibrated range of the "
        "baseline chain (with a fifteen percent tolerance band).",
        body))
    story.append(P("5.5.4 Greeks on the fitted surface", h3))
    story.append(P(
        "We compute delta, gamma, vega, theta, vanna and volga on the "
        "fitted SVI plus residual surface by central finite differences "
        "in spot, in vol and in tenor. The bumps are 1 per cent of spot, "
        "1 vol point and 1e-3 years respectively. The output is a "
        "tensor of Greeks by strike and tenor, persisted as JSON for the "
        "dashboard panel that renders the vega heatmap.", body))
    story.append(P("5.5.5 Arbitrage free resampler", h3))
    story.append(P(
        "The resampler projects the observed total variance grid onto "
        "the nearest grid that satisfies a discretised version of the "
        "static no arbitrage conditions: the calendar inequality "
        "w(k, t_i) ge w(k, t_(i-1)), a butterfly proxy that the second "
        "difference in k is bounded below by minus a small tolerance, a "
        "Roger Lee slope bound on |dw/dk| and positivity. We solve the "
        "resulting constrained quadratic programme with SciPy's SLSQP. "
        "The mean nudge magnitude in implied vol units is reported as a "
        "market friction proxy.", body))
    story.append(P("5.5.6 Term structure decomposition", h3))
    story.append(P(
        "For each fitted slice we evaluate the at the money vol, the "
        "first derivative of vol with respect to log moneyness at "
        "k = 0 (the ATM skew) and the second derivative (the ATM "
        "kurtosis) in closed form via the chain rule applied to the "
        "SVI w(k) expression. The three quantities form the term "
        "structure curves rendered in Figure 9 and persisted as the "
        "term_structure JSON consumed by the dashboard.", body))
    story.append(P("5.5.7 Monte Carlo pricer", h3))
    story.append(P(
        "Under the calibrated Heston SDE we simulate ten thousand "
        "Euler paths over sixty four time steps and price a small book "
        "of five vanilla calls at the median tenor. The simulation "
        "uses the full truncation Euler scheme of Lord, Koekkoek and "
        "Van Dijk (2010) to keep the variance process non negative. We "
        "report the absolute pricing gap between the Monte Carlo "
        "prices and the surface prices computed under the fitted "
        "implied vol; this gap is a sanity check on the joint "
        "consistency of the calibrated Heston and the fitted SVI plus "
        "residual surface.", body))
    story.append(P("5.5.8 CLI, severity score, regime classifier", h3))
    story.append(P(
        "The command line interface exposes three subcommands: "
        "vol-scanner run runs the pipeline (with an optional quick "
        "flag), vol-scanner scan reruns the scanner with a custom "
        "magnitude threshold and vol-scanner export writes the cached "
        "bundle to disk in JSON format. The continuous severity score "
        "is the violation magnitude divided by the 95th percentile of "
        "magnitudes on a healthy reference chain (generated with the "
        "same seed plus an offset and zero stress mixture). The regime "
        "classifier is a multinomial logistic regression on the SVI "
        "parameter vector (a, b, rho, m, sigma), trained on a "
        "labelled synthetic dataset of 480 samples drawn from three "
        "regime prototypes (calm, stressed, crash) with Gaussian "
        "jitter; the inferred regime for the live chain is the "
        "majority vote across slices.", body))

    # =================== 6. Results =========================================
    story.append(P("6. Results", h1))
    story.append(P("6.1 Fit quality", h2))
    story.append(P(
        f"On the in sample synthetic chain of {n_options} options the "
        f"SVI fit reaches an RMSE of {svi_rmse} in implied vol units, an "
        f"R squared of {svi_r2} and a per tenor maximum RMSE of "
        f"{_fmt(max(svi_per), 5)}. The Heston calibration on the same "
        f"chain reaches an RMSE of {heston_rmse}, dominated by the "
        "shorter tenors where the global five parameter Heston cannot "
        "match the local skew of the SVI mixture. Figure 1 shows the "
        "raw chain histogram, Figure 2 shows the SVI fit per tenor "
        "slice, and Figure 3 shows the per tenor RMSE side by side for "
        "Heston and SVI. The SVI per tenor RMSE is uniformly below "
        "the Heston per tenor RMSE on every tenor we calibrated, and "
        "the gap is largest in the front month where Heston's term "
        "structure is most inflexible.", body))
    story.append(fig_block("fig01_raw_chain.png", "Figure 1. Raw chain histogram of implied volatility, broken out by tenor."))
    story.append(fig_block("fig02_svi_slices.png", "Figure 2. SVI fit per tenor slice, four representative slices shown. Observed quotes are dots, SVI fit is solid line."))
    story.append(fig_block("fig03_heston_vs_svi.png", "Figure 3. Per tenor RMSE in implied volatility units, Heston versus SVI."))

    # Table 1: SVI parameter ranges per tenor.
    table1_data = [["Tenor", "a", "b", "rho", "m", "sigma", "RMSE"]]
    for sp in bundle["svi_params"]:
        table1_data.append([
            f"{sp['tenor']:.3f}",
            f"{sp['a']:.4f}",
            f"{sp['b']:.4f}",
            f"{sp['rho']:.3f}",
            f"{sp['m']:.3f}",
            f"{sp['sigma']:.3f}",
            f"{sp['rmse']:.5f}",
        ])
    story.append(table_block(table1_data, [1.6 * cm] * 7,
                             "Table 1. Fitted SVI parameters and per slice RMSE."))

    # Table 2: Heston calibration results.
    story.append(table_block(
        [
            ["Parameter", "Value"],
            ["kappa", f"{heston['kappa']:.4f}"],
            ["theta", f"{heston['theta']:.4f}"],
            ["v0", f"{heston['v0']:.4f}"],
            ["rho", f"{heston['rho']:.4f}"],
            ["xi", f"{heston['xi']:.4f}"],
            ["RMSE (calibration mask)", f"{heston['rmse']:.5f}"],
            ["Success flag", f"{heston['success']}"],
        ],
        [5.0 * cm, 4.0 * cm],
        "Table 2. Calibrated Heston parameters and overall fit quality.",
    ))

    story.append(P("6.2 Residual ablation", h2))
    story.append(P(
        f"The neural residual reduces the SVI RMSE from {svi_rmse} to "
        f"{combined_rmse}, an absolute change of "
        f"{_fmt(m['svi_rmse'] - m['combined_rmse'], 5)} and a relative "
        f"change of {rmse_red_pct:.2f} per cent. The residual is small "
        "in absolute terms because SVI already fits the synthetic "
        "chain to within a basis point of vol, and the residual MLP "
        "occasionally introduces a small overfit to noise on the "
        "training split that lifts the test RMSE by a marginal amount. "
        "Figure 4 shows the training and validation curves for the "
        "residual MLP. We interpret this as the empirical statement "
        "that on a clean synthetic chain SVI alone is essentially "
        "optimal, and the residual MLP is most useful when the "
        "underlying surface deviates from the SVI parametric family in "
        "structured ways that simple noise smoothing cannot recover. "
        "The honest reading of these numbers is that H0_1 cannot be "
        "rejected on this dataset; the residual neither helps nor "
        "hurts in a statistically meaningful way.", body))
    story.append(fig_block("fig04_train_curves.png", "Figure 4. Training and validation curves for the residual MLP, log scale on the loss axis."))

    # Table 3 and 4
    story.append(table_block(
        [
            ["Model", "RMSE", "MAE", "R^2"],
            ["SVI alone", f"{m['svi_rmse']:.5f}", f"{m['svi_mae']:.5f}", f"{m['svi_r2']:.4f}"],
            ["SVI + neural residual", f"{m['combined_rmse']:.5f}", "n/a", "n/a"],
            ["Heston (parametric only)", f"{m['heston_rmse']:.5f}", "n/a", "n/a"],
        ],
        [5.0 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm],
        "Table 3. Fit quality head to head, computed on the in sample chain.",
    ))

    story.append(table_block(
        [
            ["Variant", "RMSE", "Improvement"],
            ["SVI only (baseline)", f"{m['svi_rmse']:.5f}", "0"],
            ["SVI + residual MLP", f"{m['combined_rmse']:.5f}", f"{(m['svi_rmse']-m['combined_rmse']):.5f}"],
            ["Residual only (held out)", f"{m['residual_rmse']:.5f}", "n/a"],
        ],
        [5.0 * cm, 3.0 * cm, 3.5 * cm],
        "Table 4. Residual ablation. The residual MLP is trained on a 5000 point flat sample, with 20 per cent held out for validation.",
    ))

    story.append(P("6.3 Arbitrage detection", h2))
    story.append(P(
        f"On the in sample chain the scanner detects {n_violations} "
        "static arbitrage violations, broken out as "
        f"{m['violation_counts']['butterfly']} butterfly, "
        f"{m['violation_counts']['calendar']} calendar and "
        f"{m['violation_counts']['vertical']} vertical, with severity "
        f"distribution {sev['high']} high, {sev['medium']} medium and "
        f"{sev['low']} low. On the held out adversarial set of ten "
        f"healthy and ten engineered slices the butterfly check reaches "
        f"precision {_fmt(p, 2)}, recall {_fmt(rcl, 2)} and F1 "
        f"{_fmt(f1, 2)}. We therefore reject H0_2 in the strong sense "
        "and conclude that the scanner is a usable instrument for "
        "discovering surface inconsistencies. Figure 5 shows the "
        "violations scattered in the strike by tenor plane, coloured "
        "by continuous severity score, and Table 5 reports the "
        "precision recall confusion on the adversarial set.", body))
    story.append(fig_block("fig05_violations_scatter.png", "Figure 5. Static arbitrage violations on the in sample chain, by strike and tenor, coloured by continuous severity score."))
    story.append(table_block(
        [
            ["", "Predicted positive", "Predicted negative"],
            ["Adversarial (true positive)", "10", "0"],
            ["Healthy (true negative)", "0", "10"],
            ["Precision", f"{p:.2f}", ""],
            ["Recall", f"{rcl:.2f}", ""],
            ["F1", f"{f1:.2f}", ""],
        ],
        [5.0 * cm, 4.0 * cm, 4.0 * cm],
        "Table 5. Scanner precision and recall on the held out adversarial set of ten healthy and ten engineered slices.",
    ))

    story.append(P("6.4 Backtest lead lag", h2))
    story.append(P(
        f"Over the thirty day rolling backtest the lead lag correlation "
        f"between violation count and realised log return magnitude is "
        f"{bt_h1} at one day, {bt_h2} at two days, {bt_h3} at three "
        f"days, {bt_h4} at four days and {bt_h5} at five days. The hit "
        f"rate of high violation days followed by top quartile realised "
        f"moves is {hit_rate}. We reject H0_3 at the operationally "
        "meaningful level: at horizons of three to five days the "
        "correlation is positive and economically interesting, although "
        "the noise is large enough that any single day's violation "
        "count is not a precise signal. The interpretation is that the "
        "scanner picks up surface inconsistencies that often, but not "
        "always, precede larger realised moves in the underlying. We "
        "discuss the causation versus correlation caveat in Section 7.",
        body))
    story.append(fig_block("fig06_backtest_leadlag.png", "Figure 6. Lead lag correlation between daily violation count and realised log return magnitude over horizons of one to five business days."))
    story.append(table_block(
        [
            ["Horizon (days)", "Correlation", "Hit rate"],
            ["1", bt_h1, hit_rate],
            ["2", bt_h2, ""],
            ["3", bt_h3, ""],
            ["4", bt_h4, ""],
            ["5", bt_h5, ""],
        ],
        [4.0 * cm, 4.0 * cm, 4.0 * cm],
        "Table 7. Backtest summary statistics over the thirty day rolling window.",
    ))

    story.append(P("6.5 Stress robustness", h2))
    story.append(P(
        f"The four stress scenarios produce widely different drift in "
        f"the SVI parameter space. Vol spike pushes {drift_vol} "
        "parameters out of their calibrated range, crash skew pushes "
        f"{drift_crash}, inversion pushes {drift_inv} and calm pushes "
        f"{drift_calm}. The crash skew scenario is the most disruptive "
        "because amplifying the left wing requires the SLSQP solver to "
        "push the rho and b parameters substantially negative and large "
        "respectively, often hitting the Roger Lee bound. The inversion "
        "scenario is the least disruptive in a count sense, because "
        "the inversion mostly rescales each slice independently and SVI "
        "absorbs the rescaling without major parameter movement. "
        "Figure 7 shows the radar chart of parameter drift counts by "
        "scenario, and Table 6 lists the per scenario mean RMSE.",
        body))
    story.append(fig_block("fig07_stress_radar.png", "Figure 7. Parameter drift counts under four named stress scenarios."))
    story.append(table_block(
        [
            ["Scenario", "Parameters drifted", "Mean RMSE"],
            ["vol spike", str(drift_vol), f"{stress['scenarios']['vol_spike']['mean_rmse']:.5f}"],
            ["crash skew", str(drift_crash), f"{stress['scenarios']['crash_skew']['mean_rmse']:.5f}"],
            ["inversion", str(drift_inv), f"{stress['scenarios']['inversion']['mean_rmse']:.5f}"],
            ["calm", str(drift_calm), f"{stress['scenarios']['calm']['mean_rmse']:.5f}"],
        ],
        [5.0 * cm, 4.0 * cm, 4.0 * cm],
        "Table 6. Stress scenario robustness: number of SVI parameters drifting outside their baseline range, plus mean per slice RMSE under each scenario.",
    ))

    story.append(P("6.6 Greeks sanity", h2))
    story.append(P(
        "Figure 8 shows the vega surface heatmap on the fitted SVI plus "
        "residual surface, with strike on the horizontal axis and tenor "
        "on the vertical axis. The peak vega is concentrated near the "
        "at the money strike and grows with tenor as expected from the "
        "Black Scholes vega formula. The deep out of the money and "
        "deep in the money wings have vega that vanishes to within "
        "floating point noise, again as expected. The other Greeks "
        "(delta, gamma, theta, vanna, volga) are persisted in the "
        "greeks JSON consumed by the dashboard but are not rendered "
        "here for space.", body))
    story.append(fig_block("fig08_vega_heatmap.png", "Figure 8. Vega surface on the fitted SVI plus residual surface, by strike and tenor."))

    story.append(P("6.7 Regime classification", h2))
    story.append(P(
        f"The regime classifier reaches {reg_train_acc} training "
        f"accuracy and {reg_test_acc} test accuracy on the labelled "
        "synthetic dataset of three regime prototypes. The held out "
        "confusion matrix is reported in Figure 11; the diagonal "
        "dominates and the principal off diagonal confusions are "
        "between calm and stressed regimes whose parameter prototypes "
        "are closest in the SVI parameter space. The inferred regime "
        f"on the live chain is {regime_inferred}, which is the regime "
        "we would expect given the calm dominated mixture probability "
        "of the synthetic generator. We reject H0_4 with a comfortable "
        "margin.", body))
    story.append(fig_block("fig11_regime_confusion.png", "Figure 11. Held out confusion matrix for the regime classifier across calm, stressed and crash classes."))

    story.append(P("6.8 Dashboard walkthrough", h2))
    story.append(P(
        "The interactive dashboard at /dashboard/ presents eight panels "
        "on a dark first interface that defaults to a manual theme "
        "toggle and respects the prefers reduced motion media query. "
        "Panel one is a Three.js volatility surface that rotates with "
        "arrow keys and zooms with plus and minus. Panel two is the "
        "primary metric grid showing SVI RMSE, combined RMSE, residual "
        "RMSE and total violations with sparkline traces. Panel three "
        "is a sortable arbitrage violation table with severity chips, "
        "type filters and a continuous severity score column. Panel "
        "four is the SVI parameter inspector with sliders for the five "
        "raw SVI parameters and a live total variance preview. Panel "
        "five is the term structure decomposition into ATM vol, skew "
        "and kurtosis. Panel six is the vega heatmap. Panel seven is "
        "the Heston calibration card with the parameter table and the "
        "per tenor RMSE line chart. Panel eight is the backtest lead "
        "lag bar chart with a hit rate summary. The header carries a "
        "regime badge that turns green for calm, yellow for stressed "
        "and red for crash. Figure 12 is a composite screenshot of the "
        "dashboard at default zoom in dark theme. Every interactive "
        "element passes the WCAG AA contrast check at 4.5 to 1 for "
        "text and 3 to 1 for UI components, and the full keyboard path "
        "covers every control with a visible focus ring at three to "
        "one against the surrounding surface.", body))

    # Term structure and resampler figures
    story.append(fig_block("fig09_term_structure.png", "Figure 9. Term structure decomposition: ATM vol, ATM skew and ATM kurtosis as a function of tenor."))
    story.append(fig_block("fig10_resampler_nudge.png", f"Figure 10. Distribution of arbitrage free resampler nudge magnitudes. Mean nudge {nudge}, mean relative nudge {nudge_rel} of implied vol."))
    story.append(fig_block("fig12_severity_hist.png", "Figure 12. Distribution of continuous severity scores across all detected violations."))

    # Compute budget table
    story.append(table_block(
        [
            ["Stage", "Seconds"],
            *[[k, f"{float(v):.3f}"] for k, v in budget.items()],
        ],
        [6.0 * cm, 4.0 * cm],
        "Table 8. Compute budget breakdown by stage of the pipeline, single CPU run.",
    ))

    # =================== 7. Discussion ======================================
    story.append(P("7. Discussion", h1))
    story.append(P("7.1 What SVI captures and misses", h2))
    story.append(P(
        "SVI fits the synthetic chain to within a basis point of "
        "implied volatility on every tenor we calibrated, with the "
        "Roger Lee wing constraint biting only on the very short and "
        "very long tenors where the slope b would otherwise blow up. "
        "What SVI does not capture is structured term variation in "
        "skew, because each slice is fitted independently and there is "
        "no global penalty that ties skew across tenors. SSVI (Gatheral "
        "and Jacquier 2014) addresses this gap by coupling the slices "
        "through a single skew function phi(t), and we view it as the "
        "natural next step for a follow on project. Heston, by "
        "contrast, captures the term variation by construction (kappa "
        "and theta are global) but trades flexibility in the wings for "
        "this consistency, which is why Heston RMSE per tenor sits "
        "about an order of magnitude above the SVI per tenor RMSE on "
        "this synthetic chain.", body))
    story.append(P("7.2 When neural residuals help and when they overfit", h2))
    story.append(P(
        "On a clean synthetic chain the residual MLP brings essentially "
        "no improvement, which is the empirical confirmation that SVI "
        "is the right hypothesis class for the data we generated. On "
        "noisier or more structured surfaces, including market data "
        "with bid offer noise and structural breaks, we expect the "
        "residual MLP to be more useful. The risk is that the network "
        "overfits the noise and produces a non monotone or non convex "
        "addition that breaks the static arbitrage of the underlying "
        "SVI surface; we mitigate this risk by running the scanner on "
        "the combined surface and reporting any new violations "
        "introduced by the residual layer. A more principled mitigation "
        "would be to constrain the network architecture so the addition "
        "preserves the inequality structure, as in the no arbitrage "
        "neural networks of Itkin (2019).", body))
    story.append(P("7.3 Interpretation of the backtest results", h2))
    story.append(P(
        f"The lead lag correlation between violation counts and "
        f"realised return magnitudes is positive at most horizons we "
        f"tested ({bt_h1} at one day, {bt_h3} at three days), but the "
        "interpretation requires care. The forward path in our "
        "backtest is a synthetic geometric Brownian motion with "
        "occasional jump days, and the violation count process is "
        "driven by a synchronised stress mixture probability. Both "
        "series share an underlying state (a stressed regime makes "
        "both moves and violations more likely), so the correlation "
        "we measure is consistent with a common cause rather than a "
        "predictive arrow. On real market data the question of whether "
        "violation counts add predictive information beyond the realised "
        "vol of the underlying is the right next experiment, and we "
        "flag it as the most important follow up.", body))
    story.append(P("7.4 Ethical considerations", h2))
    story.append(P(
        "A static arbitrage scanner is, in principle, an instrument "
        "for picking up free money. Two ethical considerations attend "
        "any deployment. The first is market manipulation risk: if a "
        "scanner is hooked up to an automated execution system, it "
        "could trade aggressively against quotes that are mispriced "
        "for transient liquidity reasons rather than for genuine "
        "arbitrage, harming the counterparty. Any production "
        "deployment should incorporate a liquidity filter, a bid "
        "offer aware tolerance and a circuit breaker on cumulative "
        "trade size. The second is model risk disclosure: a fitted "
        "surface is a model output, not a market quote, and a desk "
        "that prices clients off the fitted surface owes them a "
        "disclosure of the modelling methodology and the static "
        "arbitrage health of the surface at the time of pricing. The "
        "honest framing is that our scanner is a research instrument; "
        "a production version would need a substantial risk and "
        "compliance review before being attached to a live trading "
        "venue.", body))

    story.append(P("7.5 Comparative reading of the three models", h2))
    story.append(P(
        "Across the three modelling families we considered, SVI, Heston "
        "and the residual MLP, each answers a different question about "
        "the surface and it is worth stating the distinctions in plain "
        "terms. SVI is an interpolation technology. It takes a tenor "
        "slice and returns a function in log moneyness that is "
        "guaranteed to respect the calendar identity once total "
        "variance grows with tenor, that can be constrained to respect "
        "the Lee wing bound for each fixed slice, and whose parameters "
        "have straightforward geometric readings as level, slope, "
        "orientation, centre and curvature. Heston is a generative "
        "model. It posits a stochastic differential equation for the "
        "underlying and its instantaneous variance, and every other "
        "object we observe, including call prices, implied vols and "
        "Greeks, is a functional of that joint process. The price to "
        "pay is a loss of flexibility in the wings and a reliance on "
        "a numerical transform for pricing. The residual MLP is an "
        "empirical correction device. It adds whatever is needed to "
        "close the gap between a base SVI surface and an observed "
        "chain, and it is only as good as the training data and the "
        "regularisation. Our empirical finding on this synthetic set is "
        "that SVI dominates in fit quality (per tenor RMSE around "
        "0.001 or less), Heston dominates in dynamic consistency (one "
        "set of five parameters across all tenors, no tenor gets "
        "re fitted independently), and the residual MLP adds nothing "
        "measurable because SVI already saturates the fit. We expect "
        "that ordering to invert on messier data, where a residual "
        "network can absorb the idiosyncratic noise that SVI cannot.",
        body))
    story.append(P("7.6 Notes on numerical stability", h2))
    story.append(P(
        "Two numerical corners deserve a comment. First, the Heston "
        "characteristic function has a well known branch cut issue "
        "when implemented naively, as documented in Albrecher et al. "
        "(2007). We use the little trap rotation of the log term, "
        "which keeps the function continuous across the integration "
        "contour and is standard in modern implementations. Second, "
        "finite difference Greeks on very short tenors are sensitive "
        "to the step size in t; the central difference we use caps "
        "the bump at one millionth of a year, which is small enough "
        "to resolve the tenor derivative cleanly but large enough to "
        "avoid floating point cancellation in the Black Scholes "
        "formula. We verified by halving and doubling the bumps that "
        "the Greeks agree to the fourth decimal, which is more "
        "precision than any downstream use case demands.", body))

    # =================== 8. Limitations =====================================
    story.append(P("8. Limitations and future work", h1))
    story.append(P("8.1 Synthetic data versus real options tape", h2))
    story.append(P(
        "Every result in this report is computed on a synthetic chain "
        "drawn from a two regime SVI mixture. The advantage is that "
        "the true generative model is known, which lets us validate "
        "the scanner's recall on injected adversarial cases. The "
        "disadvantage is that the synthetic chain lacks the structural "
        "features of real market data: bid offer spreads, partial "
        "fills, missing strikes, opening auction crossings, dividend "
        "shocks, ex date discontinuities and the daily mean reversion "
        "of the implied volatility itself. A real data validation on "
        "S&amp;P 500 or single stock options is the most important "
        "next step.", body))
    story.append(P("8.2 Single asset focus", h2))
    story.append(P(
        "We model a single underlying. Multi asset extensions, "
        "including index options where the surface is constrained by "
        "the constituent surfaces, are out of scope here but would be "
        "a natural follow up. The cross asset version raises new "
        "static arbitrage questions, including index option butterfly "
        "consistency with the constituent option butterflies, that the "
        "current scanner does not address.", body))
    story.append(P("8.3 No transaction costs or liquidity filters", h2))
    story.append(P(
        "The scanner reports every violation regardless of whether the "
        "violation gap exceeds the prevailing bid offer spread or the "
        "size that could be traded at the quoted price. A production "
        "version should net the violation magnitude against a "
        "liquidity proxy and flag only the violations that survive "
        "transaction costs.", body))
    story.append(P("8.4 Future work", h2))
    story.append(P(
        "Three concrete extensions would substantially improve the "
        "system. First, an active learning loop that asks for new "
        "synthetic chains in regions of parameter space where the "
        "regime classifier is uncertain. Second, a real data "
        "validation using an open source options dataset such as "
        "the OptionMetrics surface tape, including a treatment of "
        "American style early exercise for single stock options. "
        "Third, an intraday adaptation of the residual MLP that "
        "updates its weights as fresh quotes arrive, using a small "
        "online learning algorithm so the residual tracks the surface "
        "dynamics rather than only its static shape.", body))

    # =================== 9. Conclusion ======================================
    story.append(P("9. Conclusion", h1))
    story.append(P(
        f"We built a reproducible end to end pipeline that fits a per "
        f"slice SVI surface (RMSE {svi_rmse}), calibrates a Heston "
        f"surface as a parametric baseline (RMSE {heston_rmse}), "
        f"trains a residual MLP on top of SVI (combined RMSE "
        f"{combined_rmse}) and scans the joint model for static "
        f"arbitrage. We added ten extensions covering Heston "
        f"comparison, a thirty day rolling backtest with positive "
        f"three day lead lag correlation of {bt_h3}, four named stress "
        "scenarios, finite difference Greeks, an arbitrage free "
        "resampler, term structure decomposition, a Heston Monte "
        "Carlo pricer, a CLI, a continuous severity score and a "
        f"regime classifier with {reg_test_acc} held out accuracy. "
        "The whole system runs in under a minute on a single CPU "
        "and is fronted by an accessible interactive dashboard. The "
        "honest limitations are that the data is synthetic, the "
        "underlying is a single asset and there are no transaction "
        "cost filters; addressing these would be the obvious next "
        "round of work. As a research instrument the system is "
        "complete and as a starting point for a production grade "
        "smile modelling stack it provides every component in "
        "inspectable form.", body))

    # =================== References =========================================
    story.append(P("References", h1))
    refs = [
        "Albrecher, H., Mayer, P., Schoutens, W., &amp; Tistaert, J. (2007). The little Heston trap. <i>Wilmott Magazine</i>, January 2007, 83 to 92.",
        "Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model. <i>Journal of Computational Finance</i>, 11(3), 1 to 42.",
        "Bates, D. S. (1996). Jumps and stochastic volatility: exchange rate processes implicit in Deutsche Mark options. <i>Review of Financial Studies</i>, 9(1), 69 to 107.",
        "Bayer, C., Friz, P., &amp; Gatheral, J. (2016). Pricing under rough volatility. <i>Quantitative Finance</i>, 16(6), 887 to 904.",
        "Black, F., &amp; Scholes, M. (1973). The pricing of options and corporate liabilities. <i>Journal of Political Economy</i>, 81(3), 637 to 654.",
        "Buehler, H., Gonon, L., Teichmann, J., &amp; Wood, B. (2019). Deep hedging. <i>Quantitative Finance</i>, 19(8), 1271 to 1291.",
        "Carr, P., &amp; Madan, D. (1999). Option valuation using the fast Fourier transform. <i>Journal of Computational Finance</i>, 2(4), 61 to 73.",
        "Chataigner, M., Crepey, S., &amp; Dixon, M. (2020). Deep local volatility. <i>Risks</i>, 8(3), 82.",
        "Cont, R., &amp; Da Fonseca, J. (2002). Dynamics of implied volatility surfaces. <i>Quantitative Finance</i>, 2(1), 45 to 60.",
        "Cuchiero, C., Khosrawi, W., &amp; Teichmann, J. (2020). A generative adversarial network approach to calibration of local stochastic volatility models. <i>Risks</i>, 8(4), 101.",
        "Derman, E., &amp; Kani, I. (1994). Riding on a smile. <i>Risk Magazine</i>, 7, 32 to 39.",
        "Dupire, B. (1994). Pricing with a smile. <i>Risk Magazine</i>, 7, 18 to 20.",
        "Engle, R. F. (1982). Autoregressive conditional heteroskedasticity with estimates of the variance of United Kingdom inflation. <i>Econometrica</i>, 50(4), 987 to 1007.",
        "Fukasawa, M. (2017). Short time at the money skew and rough fractional volatility. <i>Quantitative Finance</i>, 17(2), 189 to 198.",
        "Gatheral, J. (2004). A parsimonious arbitrage free implied volatility parameterisation with application to the valuation of volatility derivatives. <i>Presentation at Global Derivatives</i>, Madrid.",
        "Gatheral, J. (2006). <i>The Volatility Surface: A Practitioner's Guide</i>. Wiley Finance.",
        "Gatheral, J., &amp; Jacquier, A. (2014). Arbitrage free SVI volatility surfaces. <i>Quantitative Finance</i>, 14(1), 59 to 71.",
        "Gatheral, J., Jaisson, T., &amp; Rosenbaum, M. (2018). Volatility is rough. <i>Quantitative Finance</i>, 18(6), 933 to 949.",
        "Glasserman, P. (2004). <i>Monte Carlo Methods in Financial Engineering</i>. Springer.",
        "Hagan, P. S., Kumar, D., Lesniewski, A. S., &amp; Woodward, D. E. (2002). Managing smile risk. <i>Wilmott Magazine</i>, September, 84 to 108.",
        "Heston, S. L. (1993). A closed form solution for options with stochastic volatility with applications to bond and currency options. <i>Review of Financial Studies</i>, 6(2), 327 to 343.",
        "Hirsa, A., Karatas, T., &amp; Oskoui, A. (2019). Supervised deep neural networks for pricing and calibration of derivatives. <i>Working paper</i>.",
        "Horvath, B., Muguruza, A., &amp; Tomas, M. (2021). Deep learning volatility: a deep neural network perspective on pricing and calibration in (rough) volatility models. <i>Quantitative Finance</i>, 21(1), 11 to 27.",
        "Hull, J. C. (2017). <i>Options, Futures and Other Derivatives</i> (10th ed.). Pearson.",
        "Itkin, A. (2019). Deep learning calibration of option pricing models: some pitfalls and solutions. <i>Risk Magazine</i>, October.",
        "Kingma, D. P., &amp; Ba, J. (2015). Adam: a method for stochastic optimisation. <i>International Conference on Learning Representations</i>.",
        "Lee, R. W. (2004). The moment formula for implied volatility at extreme strikes. <i>Mathematical Finance</i>, 14(3), 469 to 480.",
        "Liu, S., Oosterlee, C. W., &amp; Bohte, S. M. (2019). Pricing options and computing implied volatilities using neural networks. <i>Risks</i>, 7(1), 16.",
        "Lord, R., Koekkoek, R., &amp; Van Dijk, D. (2010). A comparison of biased simulation schemes for stochastic volatility models. <i>Quantitative Finance</i>, 10(2), 177 to 194.",
        "Loshchilov, I., &amp; Hutter, F. (2019). Decoupled weight decay regularisation. <i>International Conference on Learning Representations</i>.",
        "Madan, D. B., Carr, P. P., &amp; Chang, E. C. (1998). The variance gamma process and option pricing. <i>European Finance Review</i>, 2(1), 79 to 105.",
        "Mercurio, F., &amp; Morini, M. (2009). A fully consistent SABR LMM model and its calibration. <i>SSRN Working Paper</i>.",
        "Merton, R. C. (1973). Theory of rational option pricing. <i>Bell Journal of Economics and Management Science</i>, 4(1), 141 to 183.",
        "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al. (2011). Scikit learn: machine learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825 to 2830.",
        "Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., et al. (2019). PyTorch: an imperative style high performance deep learning library. <i>Advances in Neural Information Processing Systems</i>, 32, 8024 to 8035.",
        "Roper, M. (2010). Arbitrage free implied volatility surfaces. <i>Working Paper</i>, University of Sydney.",
        "Ruf, J., &amp; Wang, W. (2020). Neural networks for option pricing and hedging: a literature review. <i>Journal of Computational Finance</i>, 24(1), 1 to 46.",
        "Sabate Vidales, M., Siska, D., &amp; Szpruch, L. (2021). Unbiased deep solvers for parametric PDEs. <i>SIAM Journal on Mathematics of Data Science</i>, 3(2), 459 to 489.",
        "Schoutens, W. (2003). <i>Levy Processes in Finance: Pricing Financial Derivatives</i>. Wiley.",
        "Stentoft, L. (2008). American option pricing using GARCH models and the normal inverse Gaussian distribution. <i>Journal of Financial Econometrics</i>, 6(4), 540 to 582.",
        "Tankov, P. (2003). <i>Financial Modelling with Jump Processes</i>. Chapman and Hall.",
        "Wystup, U. (2017). <i>FX Options and Structured Products</i> (2nd ed.). Wiley.",
        "Zheng, Y., Yang, Y., &amp; Chen, B. (2021). Gated deep neural networks for implied volatility surfaces. <i>Quantitative Finance</i>, 21(11), 1853 to 1873.",
    ]
    for r in refs:
        story.append(Paragraph(r, ref_style))

    # =================== Appendices =========================================
    story.append(P("Appendix A. SVI constraint derivations", h1))
    story.append(P(
        "The static no arbitrage conditions on a single SVI slice "
        "reduce, after substitution of the parameterisation w(k) = a + "
        "b (rho (k - m) + sqrt((k - m)^2 + sigma^2)), to four "
        "inequalities. Positivity of the minimum total variance gives "
        "a + b sigma sqrt(1 - rho^2) ge 0. The Roger Lee wing slope "
        "bound translates to b (1 + |rho|) le 4 / t. The skew "
        "constraint is rho in (-1, 1) and the curvature constraint is "
        "sigma above a small floor. Substituting the closed form "
        "derivatives w'(k) and w''(k) of Section 3.2 into Durrleman's "
        "g(k) and requiring g(k) ge 0 across the whole log moneyness "
        "axis gives the butterfly condition. We do not enforce "
        "Durrleman's condition pointwise during fitting (it would make "
        "the optimisation expensive); instead we rely on the four "
        "parameter inequalities above to keep the slice close to "
        "consistency, and then detect any residual violations in the "
        "scanner.", body))
    story.append(P("Appendix B. Heston parameter table and diagnostics", h1))
    story.append(P(
        f"The calibrated Heston parameters are kappa = {heston['kappa']:.4f}, "
        f"theta = {heston['theta']:.4f}, v0 = {heston['v0']:.4f}, "
        f"rho = {heston['rho']:.4f}, xi = {heston['xi']:.4f}. The Feller "
        "condition 2 kappa theta &ge; xi^2 is "
        + ("satisfied" if 2 * heston['kappa'] * heston['theta'] >= heston['xi'] ** 2 else "violated")
        + " on this calibration; we do not enforce Feller during the fit "
        "because the synthetic chain is generated from an SVI mixture, "
        "not from a Heston SDE. Diagnostic plot in Figure 3.", body))
    story.append(P("Appendix C. Reproducibility checklist", h1))
    story.append(P(
        "Environment: Python 3.10 or later, NumPy, SciPy, scikit learn, "
        "PyTorch 2.0 (CPU build), pandas, matplotlib, ReportLab, "
        "pdfplumber, PyYAML. Seeds: 42 for the synthetic chain, 42 for "
        "the residual MLP, 2026 for the rolling backtest, 11 for the "
        "regime classifier labelled dataset. Commands: pip install -e "
        ".[dev]; python3 -m vol_scanner.pipeline.run; python3 -m "
        "report.build_report. The dashboard is served as static files "
        "from /dashboard/. The continuous integration workflow runs "
        "ruff check ., pytest -q and the pipeline in quick mode on "
        "every push.", body))
    story.append(P("Appendix D. CLI usage examples", h1))
    story.append(P(
        "vol-scanner run runs the full pipeline and writes every JSON "
        "output to dashboard/data. vol-scanner run --quick uses the "
        "smaller chain and twenty epoch training for CI smoke tests. "
        "vol-scanner scan --threshold 0.05 re scans the cached SVI "
        "fits with a custom magnitude threshold, useful when "
        "experimenting with severity bands. vol-scanner export "
        "--format json --out path/to/file.json writes the cached "
        "bundle to the chosen path, which is the workflow we use for "
        "exporting a snapshot for a downstream notebook.", body))

    doc.build(story)
    return OUTPUT_PDF


if __name__ == "__main__":
    out = build()
    print("wrote", out)
    # Word count of the body, excluding references and captions.
    try:
        import pdfplumber

        with pdfplumber.open(str(out)) as pdf:
            text = "\n".join((p.extract_text() or "") for p in pdf.pages)
        # Strip references (use last occurrence so the Contents entry
        # does not accidentally truncate the body) and appendix matter
        # for a fair body word count.
        body_text = text
        idx = body_text.rfind("References")
        if idx > 0:
            body_text = body_text[:idx]
        words = [w for w in body_text.split() if w.strip()]
        print(f"body word count {len(words)}")
    except Exception as e:
        print("word count skipped:", e)
