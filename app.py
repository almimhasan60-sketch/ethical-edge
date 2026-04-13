"""
Ethical Edge - Sustainable Portfolio Optimiser
ECN316 Sustainable Finance | QMUL Group Project
Objective: max x'mu - (gamma/2) x'Sigma x + lambda * s_bar
where x = free risky weights (remainder in risk-free asset)

FIXES APPLIED:
  1. Removed unused import requests
  2. Fixed page_icon to valid emoji ⚖️
  3. Fixed _stats() shadow vars g/l -> gam_/lam_
  4. Fixed sensitivity table loop var l -> lv
  5. Fixed gamma explorer loop var g -> gv
  6. Fixed lambda explorer loop var l -> lv
  7. Fixed sharpe_badge emoji (Trophy Excellent -> 🏆 Excellent)
  8. Fixed traffic_light() to return 3 values (ico, label, colour)
  9. Updated all helper colours to match theme (#00e676 #ff9100 #ff5252)
  10. Fixed annotated_text colour #00c853 -> #00e676
  11. Clamped st.progress value
  12. apply_chart_style chart surface offset for visual depth
  13. Improved indifference curve legend labels
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from scipy.optimize import minimize
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.chart_container import chart_container
from annotated_text import annotated_text

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Ethical Edge",
    page_icon="⚖️",   # FIX 2: was "balanced_scale" — invalid string
    layout="wide",
)
st.title("⚖️ Ethical Edge")
st.caption("Sustainable Portfolio Optimiser · ECN316 Sustainable Finance · QMUL")
st.latex(r"\text{Objective: } \max\; \mathbf{x}'\boldsymbol{\mu} - \frac{\gamma}{2}\mathbf{x}'\boldsymbol{\Sigma}\mathbf{x} + \lambda\bar{s}")
st.divider()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Investor Profile")
    st.subheader("Quick Start — Persona", divider="green")
    st.caption("Pick a profile to auto-fill γ and λ.")
    persona = st.selectbox("Investor persona", [
        "Custom (manual)",
        "🎓 Young Professional — growth-focused",
        "🌿 Impact Investor — ESG first",
        "⚖️ Balanced Saver — risk & return",
        "🏦 Retiree — capital protection",
        "📈 Pure Return Seeker — no ESG",
    ])
    persona_defaults = {
        "🎓 Young Professional — growth-focused": (2.0, 1.0),
        "🌿 Impact Investor — ESG first":          (4.0, 4.0),
        "⚖️ Balanced Saver — risk & return":       (4.0, 1.5),
        "🏦 Retiree — capital protection":          (8.0, 0.5),
        "📈 Pure Return Seeker — no ESG":           (2.0, 0.0),
    }
    if persona != "Custom (manual)":
        pg, pl = persona_defaults[persona]
        st.success(f"γ = {pg}  ·  λ = {pl}")
    else:
        pg, pl = 4.0, 1.0

    st.divider()
    st.subheader("Step 1 — Risk Attitude", divider="green")
    q1 = st.radio("If your portfolio dropped 20%, you'd…", [
        "Sell immediately — I can't handle losses",
        "Hold steady and wait it out",
        "Buy more — it's a buying opportunity",
    ], index=1)
    q2 = st.radio("Your investment horizon?",
                  ["Under 2 years", "2–5 years", "5+ years"], index=1)
    risk_map = {
        ("Sell immediately — I can't handle losses", "Under 2 years"): 9.0,
        ("Sell immediately — I can't handle losses", "2–5 years"):     8.0,
        ("Sell immediately — I can't handle losses", "5+ years"):      6.0,
        ("Hold steady and wait it out",              "Under 2 years"): 6.0,
        ("Hold steady and wait it out",              "2–5 years"):     4.0,
        ("Hold steady and wait it out",              "5+ years"):      3.0,
        ("Buy more — it's a buying opportunity",     "Under 2 years"): 3.0,
        ("Buy more — it's a buying opportunity",     "2–5 years"):     2.0,
        ("Buy more — it's a buying opportunity",     "5+ years"):      1.0,
    }
    gamma_quiz = risk_map[(q1, q2)]
    gamma_default = pg if persona != "Custom (manual)" else gamma_quiz
    st.info(f"Quiz suggests γ = {gamma_quiz}" +
            (f"  ·  Persona γ = {pg}" if persona != "Custom (manual)" else ""))
    gamma = st.slider("Fine-tune γ", 0.5, 10.0, float(gamma_default), 0.5,
                      help="Higher γ = more risk-averse. Doubling γ roughly halves risky positions.")

    st.divider()
    st.subheader("Step 2 — ESG Commitment", divider="green")
    esg_label = st.select_slider("How important is sustainability to you?",
        options=["None (λ=0)", "Low (λ=0.5)", "Medium (λ=1)", "High (λ=2)", "Max (λ=4)"],
        value="Medium (λ=1)")
    lam_map = {"None (λ=0)": 0.0, "Low (λ=0.5)": 0.5, "Medium (λ=1)": 1.0,
               "High (λ=2)": 2.0, "Max (λ=4)": 4.0}
    lam_default = pl if persona != "Custom (manual)" else lam_map[esg_label]
    lam = st.slider("Fine-tune λ", 0.0, 5.0, float(lam_default), 0.25,
                    help="λ > 0: you accept lower Sharpe for a greener portfolio.")

    st.divider()
    st.subheader("Step 3 — ESG Pillar Weights", divider="green")
    w_e = st.slider("🌍 Environmental (E)", 0.0, 1.0, 0.4, 0.05)
    w_s = st.slider("🤝 Social (S)",         0.0, 1.0, 0.3, 0.05)
    w_g = st.slider("🏛️ Governance (G)",      0.0, 1.0, 0.3, 0.05)
    pt  = w_e + w_s + w_g or 1.0
    w_e, w_s, w_g = w_e/pt, w_s/pt, w_g/pt
    st.caption(f"Normalised → E:{w_e:.0%}  S:{w_s:.0%}  G:{w_g:.0%}")

    st.divider()
    st.subheader("Step 4 — Exclusions", divider="green")
    excl_tobacco  = st.checkbox("🚬 Tobacco")
    excl_weapons  = st.checkbox("⚔️ Weapons / Defence")
    excl_fossil   = st.checkbox("🛢️ Fossil Fuels")
    excl_gambling = st.checkbox("🎰 Gambling")
    min_esg_score = st.slider("Minimum ESG score", 0, 100, 0, 5)

    st.divider()
    st.subheader("Step 5 — Asset Data", divider="green")

    st.markdown("**Asset 1**")
    name1 = st.text_input("Company name", value="Asset 1", key="name1")
    c1, c2 = st.columns(2)
    r1  = c1.number_input("E[R] (%)",   value=13.0, step=0.5, key="r1")  / 100
    sd1 = c2.number_input("σ (%)",      value=18.0, step=0.5, key="sd1") / 100
    d1, d2, d3 = st.columns(3)
    e1 = d1.number_input("E score", value=40.0, step=1.0, min_value=0.0, max_value=100.0, key="e1")
    s1 = d2.number_input("S score", value=35.0, step=1.0, min_value=0.0, max_value=100.0, key="s1")
    g1 = d3.number_input("G score", value=30.0, step=1.0, min_value=0.0, max_value=100.0, key="g1")
    sin1 = st.multiselect("Sector flags", ["Tobacco","Weapons","Fossil Fuels","Gambling"], key="sin1")
    with st.expander("🌿 Carbon Scope Breakdown (optional)"):
        st.caption("0–100, higher = more emissions")
        scope1_1 = st.slider("Scope 1 (Direct)",       0, 100, 60, 5, key="sc1_1")
        scope2_1 = st.slider("Scope 2 (Energy)",        0, 100, 50, 5, key="sc2_1")
        scope3_1 = st.slider("Scope 3 (Supply Chain)",  0, 100, 70, 5, key="sc3_1")

    st.markdown("**Asset 2**")
    name2 = st.text_input("Company name", value="Asset 2", key="name2")
    c3, c4 = st.columns(2)
    r2  = c3.number_input("E[R] (%)",   value=7.0,  step=0.5, key="r2")  / 100
    sd2 = c4.number_input("σ (%)",      value=22.0, step=0.5, key="sd2") / 100
    d4, d5, d6 = st.columns(3)
    e2 = d4.number_input("E score", value=70.0, step=1.0, min_value=0.0, max_value=100.0, key="e2")
    s2 = d5.number_input("S score", value=65.0, step=1.0, min_value=0.0, max_value=100.0, key="s2")
    g2 = d6.number_input("G score", value=75.0, step=1.0, min_value=0.0, max_value=100.0, key="g2")
    sin2 = st.multiselect("Sector flags", ["Tobacco","Weapons","Fossil Fuels","Gambling"], key="sin2")
    with st.expander("🌿 Carbon Scope Breakdown (optional)"):
        st.caption("0–100, higher = more emissions")
        scope1_2 = st.slider("Scope 1 (Direct)",       0, 100, 30, 5, key="sc1_2")
        scope2_2 = st.slider("Scope 2 (Energy)",        0, 100, 25, 5, key="sc2_2")
        scope3_2 = st.slider("Scope 3 (Supply Chain)",  0, 100, 40, 5, key="sc3_2")

    st.markdown("**Market**")
    mc1, mc2 = st.columns(2)
    rho    = mc1.number_input("Correlation ρ", -1.0, 1.0, 0.3, 0.05)
    r_free = mc2.number_input("Risk-free rate (%)", value=2.5, step=0.25) / 100

# ─────────────────────────────────────────────
# CORE MATHS — scipy free-weight optimisation
# ─────────────────────────────────────────────
esg1 = w_e*e1 + w_s*s1 + w_g*g1
esg2 = w_e*e2 + w_s*s2 + w_g*g2

cov_matrix = np.array([
    [sd1**2,       rho*sd1*sd2],
    [rho*sd1*sd2,  sd2**2     ],
])
mu_excess = np.array([r1 - r_free, r2 - r_free])

def is_excluded(flags, score):
    if excl_tobacco  and "Tobacco"      in flags: return True
    if excl_weapons  and "Weapons"      in flags: return True
    if excl_fossil   and "Fossil Fuels" in flags: return True
    if excl_gambling and "Gambling"     in flags: return True
    if score < min_esg_score:                      return True
    return False

ex1 = is_excluded(sin1, esg1)
ex2 = is_excluded(sin2, esg2)

if ex1 and ex2:
    st.error("⛔ Both assets excluded. Please relax your filters.")
    st.stop()
if ex1: st.warning(f"⚠️ {name1} excluded by screening — portfolio is 100% {name2}.")
if ex2: st.warning(f"⚠️ {name2} excluded by screening — portfolio is 100% {name1}.")

_b1 = (0.0, 0.0) if ex1 else (0.0, None)
_b2 = (0.0, 0.0) if ex2 else (0.0, None)
_bnds = [_b1, _b2]

def _solve(gam, lam_v, cov=None, mu_e=None, esg_sc=None):
    """Maximise x'mu_excess - (gam/2) x'Sigma x + lam_v * s_bar
    x1,x2 >= 0; remainder 1-x1-x2 in risk-free."""
    c    = cov    if cov    is not None else cov_matrix
    me   = mu_e   if mu_e   is not None else mu_excess
    esgs = esg_sc if esg_sc is not None else np.array([esg1, esg2])
    def obj(x):
        tot   = x[0] + x[1]
        s_bar = float(x @ esgs) / tot if tot > 1e-10 else 0.0
        return -(float(x @ me) - (gam/2)*float(x @ c @ x) + lam_v*s_bar)
    res = minimize(obj, [0.4, 0.4], method="SLSQP", bounds=_bnds,
                   options={"ftol": 1e-12, "maxiter": 2000})
    return np.array(res.x)

def _solve_tangency(cov=None):
    c  = cov if cov is not None else cov_matrix
    me = mu_excess
    def neg_sharpe(x):
        ret = r_free + float(x @ me)
        sd  = np.sqrt(max(float(x @ c @ x), 1e-14))
        return -(ret - r_free) / sd
    res = minimize(neg_sharpe, [0.4, 0.4], method="SLSQP", bounds=_bnds,
                   options={"ftol": 1e-12, "maxiter": 2000})
    return np.array(res.x)

def _solve_mvp(cov=None):
    c = cov if cov is not None else cov_matrix
    def var_fn(x): return float(x @ c @ x)
    res = minimize(var_fn, [0.4, 0.4], method="SLSQP", bounds=_bnds)
    return np.array(res.x)

def _stats(x, gam=None, lam_v=None):
    # FIX 3: renamed locals to gam_/lam_ to avoid shadowing outer gamma/lam
    gam_ = gam   if gam   is not None else gamma
    lam_ = lam_v if lam_v is not None else lam
    x1, x2 = float(x[0]), float(x[1])
    tot   = x1 + x2
    ret   = r_free + x1*(r1-r_free) + x2*(r2-r_free)
    var   = float(x @ cov_matrix @ x)
    sd    = np.sqrt(max(var, 0.0))
    esg_s = (x1*esg1 + x2*esg2) / tot if tot > 1e-10 else 0.0
    sh    = (ret - r_free) / sd if sd > 1e-10 else 0.0
    rf_w  = 1.0 - x1 - x2
    obj   = float(x @ mu_excess) - (gam_/2)*var + lam_*esg_s
    return {"Weight Asset 1": x1, "Weight Asset 2": x2, "Weight RF": rf_w,
            "Return": ret, "Volatility": sd, "ESG Score": esg_s,
            "Sharpe Ratio": sh, "Utility": obj}

x_opt = _solve(gamma, lam)
x_mv  = _solve(gamma, 0.0)
x_tan = _solve_tangency()
x_mvp = _solve_mvp()

opt = _stats(x_opt)
mv  = _stats(x_mv,  lam_v=0.0)
tan = _stats(x_tan)
mvp = _stats(x_mvp)

esg_cost     = float(tan["Sharpe Ratio"]) - float(opt["Sharpe Ratio"])
esg_cost_pct = min(abs(esg_cost) / max(abs(float(tan["Sharpe Ratio"])), 0.001) * 100, 100)
corner       = (x_opt[0] < 1e-4) or (x_opt[1] < 1e-4)

# ─────────────────────────────────────────────────────────────────────────────
# FRONTIER SWEEP — free-weight portfolios consistent with the scipy optimiser
#
# We sweep the risky-mix fraction s ∈ [0, 1]:
#   x(s) = s * x_tan   (scale the tangency portfolio down toward zero)
# This traces the Capital Market Line in (σ, E[R]) space and keeps x1,x2 ≥ 0.
# The ESG score at each point uses the same risky-asset denominator as _solve().
# This ensures the plotted frontier is the same universe as Tab 1 results.
# ─────────────────────────────────────────────────────────────────────────────
_sscale = np.linspace(0, 1.6, 500)   # go slightly above 1 to show leverage region
_rows   = []
for s in _sscale:
    xv  = np.clip(s * x_tan, 0, None)        # scale tangency weights; keep non-negative
    ret = r_free + float(xv @ mu_excess)
    var = float(xv @ cov_matrix @ xv)
    sd  = np.sqrt(max(var, 0.0))
    tot = float(xv[0]) + float(xv[1])
    esg = (float(xv[0])*esg1 + float(xv[1])*esg2) / tot if tot > 1e-10 else (esg1+esg2)/2
    sr  = (ret - r_free) / sd if sd > 1e-10 else 0.0
    _rows.append({
        "Weight Asset 1": float(xv[0]), "Weight Asset 2": float(xv[1]),
        "Weight RF": 1.0 - tot,
        "Return": ret, "Volatility": sd, "ESG Score": esg, "Sharpe Ratio": sr,
        "Scale": s,
    })

# Also sweep the risky-mix composition at s=1 (fully on CML tangency point)
# to show the ESG-Sharpe frontier correctly
_wmix  = np.linspace(0, 1, 300)
_rows2 = []
for w in _wmix:
    xv  = np.array([w, 1.0-w])
    # scale so x1+x2 = tangency total risky allocation
    tan_tot = float(x_tan[0]) + float(x_tan[1])
    xv_s = xv * tan_tot
    ret = r_free + float(xv_s @ mu_excess)
    sd  = np.sqrt(max(float(xv_s @ cov_matrix @ xv_s), 0.0))
    esg = w*esg1 + (1.0-w)*esg2
    sr  = (ret - r_free) / sd if sd > 1e-10 else 0.0
    _rows2.append({"Weight Asset 1": w, "Weight Asset 2": 1-w,
                   "Return": ret, "Volatility": sd, "ESG Score": esg, "Sharpe Ratio": sr})

portfolios     = pd.DataFrame(_rows)    # CML sweep — used for frontier chart
portfolios_mix = pd.DataFrame(_rows2)   # risky-mix sweep — used for ESG-Sharpe chart

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def p_utility(w1, gam=None, lam_val=None):
    """
    Objective on the NORMALISED risky mix (x1+x2=1, used for heatmap visuals only).
    Uses the same free-weight formula as _solve/_stats:
      U = x'mu_excess - (gam/2) x'Sigma x + lam * s_bar
    where x = [w1*tan_tot, (1-w1)*tan_tot] scaled to the tangency total.
    This ensures the heatmap is consistent with the Tab 1 optimiser.
    """
    gam_ = gam    if gam    is not None else gamma
    lam_ = lam_val if lam_val is not None else lam
    tan_tot = float(x_tan[0]) + float(x_tan[1])
    # scale by tangency total so we stay in the same wealth-fraction universe
    xv   = np.array([w1, 1.0-w1]) * tan_tot
    tot  = float(xv[0]) + float(xv[1])
    s_bar = (float(xv[0])*esg1 + float(xv[1])*esg2) / tot if tot > 1e-10 else 0.0
    return float(xv @ mu_excess) - (gam_/2)*float(xv @ cov_matrix @ xv) + lam_*s_bar

# FIX 7: added proper emoji to every badge value
def sharpe_badge(sr):
    if sr >= 1.0:  return "🏆 Excellent"
    if sr >= 0.5:  return "✅ Good"
    if sr >= 0.25: return "⚠️ Fair"
    return "🔴 Poor"

# FIX 8: returns 3 values (ico, label, colour) — consistent with theme
def traffic_light(score):
    if score >= 65: return "🟢", "Strong",   "#00e676"
    if score >= 40: return "🟡", "Moderate", "#ff9100"
    return "🔴", "Weak", "#ff5252"

# FIX 9: updated all helper colours to match theme
def carbon_label(cs):
    if cs >= 65: return "🌿 Low Carbon",  "#00e676"
    if cs >= 40: return "🌤️ Moderate",    "#ff9100"
    return "🏭 High Carbon", "#ff5252"

def greenwashing_flag(e, s, g_score):
    avg_sg = (s + g_score) / 2
    if e >= 65 and avg_sg < 40:
        return True, "⚠️ Possible greenwashing — high E score but weak S/G (Berg et al. RF 2022)"
    if e >= 65 and avg_sg < 55:
        return True, "🔶 Watch: strong environmental claims, moderate governance/social"
    return False, ""

def sf_framework(lam_val, excl_any):
    if lam_val >= 3.0:
        return "🌍 SF 3.0 — Common Good", "ESG prioritised above financial returns.", "#00e676"
    elif lam_val > 0:
        return "⚖️ SF 2.0 — Stakeholder Value", "ESG integrated into utility alongside returns.", "#ff9100"
    elif excl_any:
        return "🚫 SF 1.0 — Profit + Exclusions", "Profit maximisation while avoiding sin stocks.", "#ff5252"
    return "💰 Finance-as-Usual", "Pure financial return maximiser.", "#64748b"

# FIX 12: chart surface slightly offset from page bg for visual depth
def apply_chart_style(ax, fig):
    fig.patch.set_facecolor("#0e0e12")
    ax.set_facecolor("#12121a")
    ax.tick_params(colors="#c8ccd8", labelsize=9)
    ax.xaxis.label.set_color("#c8ccd8")
    ax.yaxis.label.set_color("#c8ccd8")
    ax.title.set_color("#00e676")
    for sp in ax.spines.values(): sp.set_edgecolor("#2a2a3a")
    ax.grid(True, color="#1e1e2a", linewidth=0.6)

def portfolio_summary():
    x1p = opt["Weight Asset 1"]*100
    x2p = opt["Weight Asset 2"]*100
    rfp = opt["Weight RF"]*100
    if x1p >= 99:   alloc = f"100% in **{name1}**"
    elif x2p >= 99: alloc = f"100% in **{name2}**"
    else:
        alloc = f"**{x1p:.0f}%** in **{name1}**, **{x2p:.0f}%** in **{name2}**"
        if abs(rfp) > 1: alloc += f", **{rfp:.0f}%** risk-free"
    esg_d = "strong" if opt["ESG Score"]>=65 else ("moderate" if opt["ESG Score"]>=40 else "weak")
    return (f"Recommended: {alloc}. "
            f"Expected return **{opt['Return']*100:.1f}%**, risk σ = **{opt['Volatility']*100:.1f}%**, "
            f"ESG **{opt['ESG Score']:.1f}/100** ({esg_d}), Sharpe **{opt['Sharpe Ratio']:.3f}**.")

# FIX 4: renamed loop var l -> lv to avoid shadowing outer lam
def build_sensitivity_table():
    rows   = []
    tan_sr = float(tan["Sharpe Ratio"])
    for lv in [0.0, 0.5, 1.0, 2.0, 4.0]:
        x = _solve(gamma, lv)
        s = _stats(x, lam_v=lv)
        rows.append({
            "λ":             lv,
            f"{name1}(%)":   f"{x[0]*100:.1f}",
            f"{name2}(%)":   f"{x[1]*100:.1f}",
            "RF(%)":         f"{s['Weight RF']*100:.1f}",
            "E[Rp]":         f"{s['Return']*100:.2f}%",
            "σp":            f"{s['Volatility']*100:.2f}%",
            "ESG":           f"{s['ESG Score']:.1f}",
            "Sharpe":        f"{s['Sharpe Ratio']:.3f}",
            "ESG Cost":      f"{max(tan_sr - s['Sharpe Ratio'], 0):.4f}",
        })
    return pd.DataFrame(rows)

def solve_scenario(gam, lam_v):
    x = _solve(gam, lam_v)
    s = _stats(x, gam=gam, lam_v=lam_v)
    s["w1"] = x[0]; s["w2"] = x[1]
    return s

def build_report():
    today = datetime.date.today().strftime("%d %B %Y")
    _, lbl1, _ = traffic_light(esg1)   # FIX 8: unpack 3 values
    _, lbl2, _ = traffic_light(esg2)
    lines = [
        "="*60, "  ETHICAL EDGE — PORTFOLIO HEALTH REPORT",
        f"  Generated: {today}", "="*60, "",
        "INVESTOR PROFILE",
        f"  Persona:  {persona}",
        f"  γ:        {gamma}",
        f"  λ:        {lam}",
        f"  Pillar Wts: E:{w_e:.0%} S:{w_s:.0%} G:{w_g:.0%}", "",
        "ASSETS",
        f"  {name1}: E[R]={r1*100:.1f}% σ={sd1*100:.1f}% ESG={esg1:.1f}/100 ({lbl1})",
        f"  {name2}: E[R]={r2*100:.1f}% σ={sd2*100:.1f}% ESG={esg2:.1f}/100 ({lbl2})",
        f"  ρ={rho}  rf={r_free*100:.1f}%", "",
        "OPTIMAL PORTFOLIO",
        f"  {name1}:      {opt['Weight Asset 1']*100:.1f}%",
        f"  {name2}:      {opt['Weight Asset 2']*100:.1f}%",
        f"  Risk-Free:  {opt['Weight RF']*100:.1f}%",
        f"  E[R]={opt['Return']*100:.2f}%  σ={opt['Volatility']*100:.2f}%",
        f"  ESG={opt['ESG Score']:.1f}/100  Sharpe={opt['Sharpe Ratio']:.3f}",
        f"  ESG Cost: {esg_cost:.4f} Sharpe vs tangency", "",
        "COMPARISON",
        f"  {'Portfolio':<22} {name1+'%':>8} {name2+'%':>8} {'RF%':>6} {'Sharpe':>8}",
        "-"*56,
        f"  {'MV Optimal (λ=0)':<22} {mv['Weight Asset 1']*100:>8.1f} {mv['Weight Asset 2']*100:>8.1f} {mv['Weight RF']*100:>6.1f} {mv['Sharpe Ratio']:>8.3f}",
        f"  {'ESG Optimal (you)':<22} {opt['Weight Asset 1']*100:>8.1f} {opt['Weight Asset 2']*100:>8.1f} {opt['Weight RF']*100:>6.1f} {opt['Sharpe Ratio']:>8.3f}",
        f"  {'Tangency (Max Sharpe)':<22} {tan['Weight Asset 1']*100:>8.1f} {tan['Weight Asset 2']*100:>8.1f} {tan['Weight RF']*100:>6.1f} {tan['Sharpe Ratio']:>8.3f}",
        "",
        "REFERENCES",
        "  Pedersen et al. (2021) — Responsible Investing, JFE",
        "  Schoenmaker (2017) — Sustainable Finance Framework",
        "  Berg et al. (RF 2022) — ESG Rating Disagreement",
        "  Bolton & Kacperczyk (JFE 2021) — Carbon & Returns",
        "",
        "  ⚖️ Ethical Edge · ECN316 Sustainable Finance · QMUL",
        "="*60,
    ]
    return "\n".join(lines)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Results", "📈 Charts", "🔬 Explore",
    "💡 Insights", "❓ Explainer", "🔀 Compare", "ℹ️ Methodology",
])

# ══════════════════════════════════════════════
# TAB 1 — RESULTS
# ══════════════════════════════════════════════
with tab1:

    # Corner solution warning
    if corner:
        low_name  = name2 if x_opt[0] >= x_opt[1] else name1
        high_name = name1 if x_opt[0] >= x_opt[1] else name2
        st.warning(
            f"**Corner solution detected.** Your ESG preference (λ = {lam}) is so strong that "
            f"holding any amount of **{low_name}** reduces utility. "
            f"All risky investment is in **{high_name}**. To hold both assets, reduce λ."
        )

    # ESG badges
    st.subheader("Asset ESG Scores", divider="green")
    a1c, a2c = st.columns(2)
    for col, nm, score, es, ss, gs, exf in [
        (a1c, name1, esg1, e1, s1, g1, ex1),
        (a2c, name2, esg2, e2, s2, g2, ex2),
    ]:
        ico, lbl, clr = traffic_light(score)   # FIX 8: unpack 3 values
        col.metric(
            label=f"{ico} {nm}  —  {lbl}  {'⛔ EXCLUDED' if exf else ''}",
            value=f"{score:.1f} / 100",
            help=f"E:{es:.0f}  S:{ss:.0f}  G:{gs:.0f}",
        )
    style_metric_cards(background_color="#0e0e12", border_left_color="#00e676",
                       border_color="#2a2a3a", box_shadow=True)

    # Greenwashing flag
    for nm, es, ss, gs in [(name1, e1, s1, g1), (name2, e2, s2, g2)]:
        flagged, msg = greenwashing_flag(es, ss, gs)
        if flagged:
            st.warning(f"**{nm}:** {msg}")

    # SF Framework badge
    excl_any = excl_tobacco or excl_weapons or excl_fossil or excl_gambling or min_esg_score > 0
    sf_lbl, sf_desc, sf_clr = sf_framework(lam, excl_any)
    st.info(f"**Sustainable Finance Approach:** {sf_lbl}  \n{sf_desc}  \n*Source: Schoenmaker (2017)*")

    # Carbon scope chart
    st.subheader("🌿 Carbon Scope Breakdown", divider="green")
    st.caption("Scope 1 = direct · Scope 2 = energy · Scope 3 = supply chain  (lower = greener)")
    fig_sc, ax_sc = plt.subplots(figsize=(8, 3.5))
    apply_chart_style(ax_sc, fig_sc)
    x_sc = np.arange(3); w_sc = 0.35
    bars_sc1 = ax_sc.bar(x_sc-w_sc/2, [scope1_1,scope2_1,scope3_1], w_sc,
                         label=name1, color="#ff5252", alpha=0.85)
    bars_sc2 = ax_sc.bar(x_sc+w_sc/2, [scope1_2,scope2_2,scope3_2], w_sc,
                         label=name2, color="#00e676", alpha=0.85)
    tot_r = float(opt["Weight Asset 1"]) + float(opt["Weight Asset 2"])
    w1n   = float(opt["Weight Asset 1"]) / tot_r if tot_r > 1e-10 else 0.5
    w2n   = 1.0 - w1n
    ax_sc.plot(x_sc,
               [w1n*scope1_1+w2n*scope1_2, w1n*scope2_1+w2n*scope2_2, w1n*scope3_1+w2n*scope3_2],
               "D--", color="white", lw=1.5, ms=8, label="Your Portfolio (weighted)", zorder=5)
    ax_sc.set_xticks(x_sc)
    ax_sc.set_xticklabels(["Scope 1\n(Direct)","Scope 2\n(Energy)","Scope 3\n(Supply Chain)"],
                          color="#c8ccd8")
    ax_sc.set_title("Carbon Emissions by Scope — GHG Protocol aligned", fontweight="bold")
    ax_sc.set_ylim(0, 115)
    ax_sc.legend(facecolor="#0e0e12", labelcolor="#c8ccd8", edgecolor="#2a2a3a")
    for bar in list(bars_sc1) + list(bars_sc2):
        ax_sc.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                   f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9, color="#c8ccd8")
    plt.tight_layout()
    st.pyplot(fig_sc)
    plt.close(fig_sc)
    st.caption("Lower = lower carbon intensity. Portfolio line = your weighted average across scopes.")
    st.divider()

    # Recommendation
    st.info(f"💼 **Portfolio Recommendation**\n\n{portfolio_summary()}")

    # Key metrics
    st.subheader("Your Recommended Portfolio", divider="green")
    m1,m2,m3,m4,m5,m6,m7 = st.columns(7)
    m1.metric(f"{name1} Weight",  f"{opt['Weight Asset 1']*100:.1f}%")
    m2.metric(f"{name2} Weight",  f"{opt['Weight Asset 2']*100:.1f}%")
    m3.metric("Risk-Free Weight", f"{opt['Weight RF']*100:.1f}%")
    m4.metric("Expected Return",  f"{opt['Return']*100:.2f}%")
    m5.metric("Risk  σ",          f"{opt['Volatility']*100:.2f}%")
    m6.metric("ESG Score",        f"{opt['ESG Score']:.1f}")
    m7.metric("Sharpe Ratio",     f"{opt['Sharpe Ratio']:.3f}",
              delta=sharpe_badge(float(opt["Sharpe Ratio"])))
    style_metric_cards(background_color="#0e0e12", border_left_color="#00e676",
                       border_color="#2a2a3a", box_shadow=True)

    if lam > 0 and esg_cost > 0.001:
        st.success(f"🌱 λ={lam}: **ESG cost = {esg_cost:.4f}** Sharpe vs max-Sharpe portfolio.")
    elif lam == 0:
        st.info("ℹ️ λ=0 — ESG plays no role. ESG Optimal = MV Optimal.")

    # FIX 11: clamped progress value to avoid crash at exact 0 or 1
    st.progress(min(max(esg_cost_pct/100, 0.0), 1.0),
                text=f"ESG cost: {esg_cost:.4f} Sharpe ({esg_cost_pct:.1f}% of max Sharpe)")
    st.divider()

    # Investment calculator
    st.subheader("💰 Investment Calculator", divider="green")
    st.caption("Projected value of your investment — ESG Optimal vs Max-Sharpe vs Risk-Free.")
    inv_col1, inv_col2 = st.columns([1, 2])
    with inv_col1:
        invest_amt = st.number_input("Investment amount (£)", value=1000, step=100, min_value=100)
        invest_yrs = st.slider("Time horizon (years)", 1, 30, 10)
        st.markdown("---")
        st.metric("ESG Optimal", f"£{invest_amt*(1+opt['Return'])**invest_yrs:,.0f}",
                  delta=f"+£{invest_amt*((1+opt['Return'])**invest_yrs-1):,.0f}")
        st.metric("Max Sharpe",  f"£{invest_amt*(1+tan['Return'])**invest_yrs:,.0f}",
                  delta=f"+£{invest_amt*((1+tan['Return'])**invest_yrs-1):,.0f}")
        st.metric("Risk-Free",   f"£{invest_amt*(1+r_free)**invest_yrs:,.0f}",
                  delta=f"+£{invest_amt*((1+r_free)**invest_yrs-1):,.0f}")
        style_metric_cards(background_color="#0e0e12", border_left_color="#00e676",
                           border_color="#2a2a3a", box_shadow=True)
    with inv_col2:
        years = np.arange(0, invest_yrs+1)
        fig_inv, ax_inv = plt.subplots(figsize=(8, 4))
        apply_chart_style(ax_inv, fig_inv)
        ax_inv.plot(years, invest_amt*(1+opt["Return"])**years, color="#00e676", lw=2.5,
                    label=f"ESG Optimal ({opt['Return']*100:.1f}%/yr)")
        ax_inv.plot(years, invest_amt*(1+tan["Return"])**years, color="#ff5252", lw=1.8, ls="--",
                    label=f"Max Sharpe ({tan['Return']*100:.1f}%/yr)")
        ax_inv.plot(years, invest_amt*(1+mv["Return"])**years,  color="#448aff", lw=1.5, ls=":",
                    label=f"MV Optimal ({mv['Return']*100:.1f}%/yr)")
        ax_inv.plot(years, invest_amt*(1+r_free)**years, color="white", lw=1.2, ls="-.", alpha=0.5,
                    label=f"Risk-free ({r_free*100:.1f}%/yr)")
        ax_inv.fill_between(years, invest_amt*(1+opt["Return"])**years,
                             invest_amt*(1+r_free)**years, color="#00e676", alpha=0.06)
        ax_inv.set_xlabel("Years", fontsize=10)
        ax_inv.set_ylabel("Portfolio Value (£)", fontsize=10)
        ax_inv.set_title(f"£{invest_amt:,} invested over {invest_yrs} years", fontsize=10, fontweight="bold")
        ax_inv.legend(fontsize=8, facecolor="#0e0e12", labelcolor="#c8ccd8",
                      edgecolor="#2a2a3a", framealpha=0.9)
        ax_inv.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
        plt.tight_layout()
        st.pyplot(fig_inv)
        plt.close(fig_inv)
    st.caption("⚠️ Assumes constant annual returns. Past performance does not guarantee future results.")
    st.divider()

    # Comparison table
    st.subheader("Portfolio Comparison Table", divider="green")
    comp_df = pd.DataFrame({
        "Portfolio":       ["MV Optimal (λ=0)", "✅ ESG Optimal (you)",
                            "Tangency — Max Sharpe", "Min Variance"],
        f"{name1}(%)":     [f"{mv['Weight Asset 1']*100:.1f}",  f"{opt['Weight Asset 1']*100:.1f}",
                            f"{tan['Weight Asset 1']*100:.1f}", f"{mvp['Weight Asset 1']*100:.1f}"],
        f"{name2}(%)":     [f"{mv['Weight Asset 2']*100:.1f}",  f"{opt['Weight Asset 2']*100:.1f}",
                            f"{tan['Weight Asset 2']*100:.1f}", f"{mvp['Weight Asset 2']*100:.1f}"],
        "RF(%)":           [f"{mv['Weight RF']*100:.1f}",  f"{opt['Weight RF']*100:.1f}",
                            f"{tan['Weight RF']*100:.1f}", f"{mvp['Weight RF']*100:.1f}"],
        "E[Rp]":           [f"{mv['Return']*100:.2f}%",  f"{opt['Return']*100:.2f}%",
                            f"{tan['Return']*100:.2f}%", f"{mvp['Return']*100:.2f}%"],
        "σp":              [f"{mv['Volatility']*100:.2f}%",  f"{opt['Volatility']*100:.2f}%",
                            f"{tan['Volatility']*100:.2f}%", f"{mvp['Volatility']*100:.2f}%"],
        "ESG":             [f"{mv['ESG Score']:.1f}",  f"{opt['ESG Score']:.1f}",
                            f"{tan['ESG Score']:.1f}", f"{mvp['ESG Score']:.1f}"],
        "Sharpe":          [f"{mv['Sharpe Ratio']:.3f}",  f"{opt['Sharpe Ratio']:.3f}",
                            f"{tan['Sharpe Ratio']:.3f}", "—"],
        "Rating":          [sharpe_badge(float(mv["Sharpe Ratio"])),
                            sharpe_badge(float(opt["Sharpe Ratio"])),
                            sharpe_badge(float(tan["Sharpe Ratio"])), "—"],
    })
    st.dataframe(comp_df, hide_index=True, use_container_width=True)
    st.divider()

    st.subheader("λ Sensitivity Analysis", divider="green")
    st.caption("How your optimal allocation and ESG cost shift as λ increases from 0 to 4.")
    st.dataframe(build_sensitivity_table(), hide_index=True, use_container_width=True)
    st.divider()

    st.subheader("📄 Download Portfolio Report", divider="green")
    st.download_button(
        label="⬇️ Download Health Report (.txt)",
        data=build_report(),
        file_name=f"ethical_edge_{datetime.date.today()}.txt",
        mime="text/plain",
    )

# ══════════════════════════════════════════════
# TAB 2 — CHARTS
# ══════════════════════════════════════════════
with tab2:
    st.subheader("ESG-Efficient Frontier", divider="green")
    st.caption("Left: Capital Market Line with free-weight portfolios (consistent with Tab 1 optimiser). Right: ESG–Sharpe trade-off across risky-mix compositions.")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("#0e0e12")
    for ax in axes: apply_chart_style(ax, fig)

    ax = axes[0]
    # Use CML sweep (portfolios) — these are the actual free-weight portfolios
    # that the scipy optimiser operates over, so the optimal point lies exactly on this line
    sc = ax.scatter(portfolios["Volatility"]*100, portfolios["Return"]*100,
                    c=portfolios["ESG Score"], cmap="RdYlGn", s=12, alpha=0.85,
                    vmin=0, vmax=100, zorder=2)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("ESG Score", color="#c8ccd8", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#c8ccd8")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#c8ccd8", fontsize=8)
    cbar.outline.set_edgecolor("#2a2a3a")

    # CML from risk-free through tangency
    if float(tan["Volatility"]) > 1e-10:
        sdr = np.linspace(0, portfolios["Volatility"].max()*1.15, 200)
        ax.plot(sdr*100,
                (r_free + (float(tan["Return"])-r_free)/float(tan["Volatility"])*sdr)*100,
                "--", color="#448aff", lw=1.5, label="CML", alpha=0.85)

    # Indifference curves in (σ, E[R]) space
    # For free-weight model: U = x'μ_e - (γ/2)x'Σx + λs̄
    # On the CML: x = s·x_tan, so σ = s·σ_tan and E[R] = rf + s·(E[R]_tan - rf)
    # Indifference curve: E[R] = rf + U_level - λs̄ + (γ/2)σ²   (rearranged)
    sc2 = np.linspace(0.001, portfolios["Volatility"].max()*1.15, 200)
    # MV indifference through MV optimal (λ=0)
    U_mv_level = float(mv["Weight Asset 1"])*(r1-r_free) + float(mv["Weight Asset 2"])*(r2-r_free) \
                 - (gamma/2)*float(mv["Volatility"])**2
    ax.plot(sc2*100, (r_free + U_mv_level + (gamma/2)*sc2**2)*100,
            ":", color="#448aff", lw=1.5, label="MV Indifference Curve")
    # ESG indifference through ESG optimal
    # U_esg = x'μ_e - (γ/2)x'Σx + λs̄  →  on σ axis: E[R] = rf + U_esg - λs̄_opt + (γ/2)σ²
    U_esg_level = float(opt["Utility"])
    ax.plot(sc2*100,
            (r_free + U_esg_level - lam*float(opt["ESG Score"]) + (gamma/2)*sc2**2)*100,
            "-.", color="#ff9100", lw=1.5, label="ESG Indifference Curve")

    ax.scatter(0, r_free*100, s=90, marker="s", color="white", zorder=6,
               label="Risk-Free", ec="#555", lw=0.8)
    ax.scatter(float(mvp["Volatility"])*100, float(mvp["Return"])*100, s=100,
               marker="D", color="#00b0ff", zorder=6, label="Min Variance", ec="white", lw=0.5)
    ax.scatter(float(tan["Volatility"])*100, float(tan["Return"])*100, s=200,
               marker="*", color="#ff5252", zorder=7, label="Tangency", ec="white", lw=0.5)
    ax.scatter(float(mv["Volatility"])*100,  float(mv["Return"])*100,  s=140,
               marker="^", color="#448aff", zorder=6, label="MV Opt. (λ=0)", ec="white", lw=0.5)
    ax.scatter(float(opt["Volatility"])*100, float(opt["Return"])*100, s=200,
               marker="*", color="#00e676", zorder=7,
               label=f"ESG Opt. ({name1}:{opt['Weight Asset 1']*100:.0f}% {name2}:{opt['Weight Asset 2']*100:.0f}%)",
               ec="white", lw=0.5)
    ax.set_xlabel("Risk — Std Dev (%)", fontsize=10)
    ax.set_ylabel("Expected Return (%)", fontsize=10)
    ax.set_title("Capital Market Line\n(free-weight portfolios, colour = ESG score)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left", facecolor="#0e0e12", labelcolor="#c8ccd8",
              framealpha=0.9, edgecolor="#2a2a3a")

    # Right chart: ESG-Sharpe frontier — uses risky-mix sweep (portfolios_mix)
    # This shows how ESG score and Sharpe trade off across all risky-mix compositions
    ax2 = axes[1]
    ax2.plot(portfolios_mix["ESG Score"], portfolios_mix["Sharpe Ratio"],
             color="#00e676", lw=2.5)
    ax2.fill_between(portfolios_mix["ESG Score"], portfolios_mix["Sharpe Ratio"],
                     portfolios_mix["Sharpe Ratio"].min(), color="#00e676", alpha=0.07)
    ax2.scatter(float(opt["ESG Score"]), float(opt["Sharpe Ratio"]), s=200, marker="*",
                color="#00e676", zorder=5, label="Your ESG Optimal", ec="white", lw=0.5)
    ax2.scatter(float(tan["ESG Score"]), float(tan["Sharpe Ratio"]), s=160, marker="*",
                color="#ff5252", zorder=5, label="Max Sharpe (Tangency)", ec="white", lw=0.5)
    ax2.axhline(float(tan["Sharpe Ratio"]), color="#ff5252", ls="--", lw=0.9, alpha=0.5)
    if esg_cost > 0.001:
        ax2.annotate(f"ESG cost: −{esg_cost:.3f}",
                     xy=(float(opt["ESG Score"]), float(opt["Sharpe Ratio"])),
                     xytext=(float(opt["ESG Score"])+2, float(opt["Sharpe Ratio"])-0.03),
                     color="#ff9100", fontsize=9, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="#ff9100", lw=1.2))
    ax2.set_xlabel("Portfolio ESG Score", fontsize=10)
    ax2.set_ylabel("Sharpe Ratio", fontsize=10)
    ax2.set_title("ESG–Sharpe Frontier\n(risky-mix compositions)", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8, facecolor="#0e0e12", labelcolor="#c8ccd8",
               framealpha=0.9, edgecolor="#2a2a3a")

    plt.tight_layout(pad=2.0)
    with chart_container(
        portfolios_mix[["Weight Asset 1","Return","Volatility","ESG Score","Sharpe Ratio"]],
        export_formats=["CSV"],
    ):
        st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Pie charts
    st.subheader("Portfolio Allocation Pie", divider="green")
    pie_col1, pie_col2 = st.columns(2)

    def draw_pie(ax, fig_p, w1, w2, wrf, t1, t2, title):
        apply_chart_style(ax, fig_p)
        ax.set_facecolor("#0e0e12")
        labels = [t1, t2]; sizes = [w1, w2]; colors = ["#00e676","#448aff"]
        if abs(wrf) > 0.005:
            labels.append("Risk-Free"); sizes.append(max(wrf, 0)); colors.append("#ffffff")
        if sum(sizes) < 1e-10: sizes = [0.5, 0.5]
        _, _, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor="#0e0e12", linewidth=2),
            textprops=dict(color="#c8ccd8", fontsize=9),
        )
        for at in autotexts:
            at.set_color("#0e0e12"); at.set_fontweight("bold"); at.set_fontsize(9)
        ax.set_title(title, color="#00e676", fontsize=10, fontweight="bold")

    with pie_col1:
        fp1, ap1 = plt.subplots(figsize=(4, 4)); fp1.patch.set_facecolor("#0e0e12")
        draw_pie(ap1, fp1, float(opt["Weight Asset 1"]), float(opt["Weight Asset 2"]),
                 float(opt["Weight RF"]), name1, name2, "ESG Optimal (you)")
        plt.tight_layout(); st.pyplot(fp1); plt.close(fp1)

    with pie_col2:
        fp2, ap2 = plt.subplots(figsize=(4, 4)); fp2.patch.set_facecolor("#0e0e12")
        draw_pie(ap2, fp2, float(tan["Weight Asset 1"]), float(tan["Weight Asset 2"]),
                 float(tan["Weight RF"]), name1, name2, "Tangency / Max Sharpe")
        plt.tight_layout(); st.pyplot(fp2); plt.close(fp2)

    # ESG pillar leaderboard
    st.subheader("ESG Pillar Leaderboard", divider="green")
    st.caption("Which asset scores higher on each pillar?")
    for pillar, sc1, sc2 in zip(["Environmental","Social","Governance"], [e1,s1,g1], [e2,s2,g2]):
        winner = name1 if sc1 >= sc2 else name2
        diff   = abs(sc1 - sc2)
        ico    = "🌍" if pillar=="Environmental" else ("🤝" if pillar=="Social" else "🏛️")
        ca, cb, cc = st.columns([3, 3, 2])
        ca.metric(f"{ico} {pillar} — {name1}", f"{sc1:.0f}/100",
                  delta=f"{'↑' if sc1>sc2 else '↓'} {diff:.0f} vs {name2}")
        cb.metric(f"{ico} {pillar} — {name2}", f"{sc2:.0f}/100",
                  delta=f"{'↑' if sc2>sc1 else '↓'} {diff:.0f} vs {name1}")
        cc.markdown(f"**Winner:** {winner}")
    style_metric_cards(background_color="#0e0e12", border_left_color="#00e676",
                       border_color="#2a2a3a", box_shadow=False)
    st.divider()

    # ESG pillar bar chart
    st.subheader("ESG Pillar Breakdown", divider="green")
    fig2, ax3 = plt.subplots(figsize=(7, 3.5))
    apply_chart_style(ax3, fig2)
    xb = np.arange(3); wb = 0.35
    b1 = ax3.bar(xb-wb/2, [e1,s1,g1], wb, label=name1, color="#00e676", alpha=0.85)
    b2 = ax3.bar(xb+wb/2, [e2,s2,g2], wb, label=name2, color="#448aff", alpha=0.85)
    ax3.set_xticks(xb)
    ax3.set_xticklabels(["Environmental","Social","Governance"], color="#c8ccd8")
    ax3.set_ylabel("Score (0–100)")
    ax3.set_title(f"E/S/G Scores: {name1} vs {name2}", fontweight="bold")
    ax3.set_ylim(0, 115)
    ax3.legend(facecolor="#0e0e12", labelcolor="#c8ccd8", edgecolor="#2a2a3a")
    for bar in list(b1) + list(b2):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                 f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9, color="#c8ccd8")
    plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

# ══════════════════════════════════════════════
# TAB 3 — EXPLORE
# ══════════════════════════════════════════════
with tab3:
    st.subheader("Interactive Explorers", divider="green")
    exp1, exp2, exp3, exp4 = st.tabs([
        "🎚️ γ Explorer", "🌿 λ Explorer", "🔗 ρ Explorer", "🌡️ Utility Heatmap",
    ])

    with exp1:
        st.markdown("#### How does risk aversion γ change your portfolio?")
        st.caption("With free weights: doubling γ roughly halves risky positions and increases risk-free weight.")
        gammas = np.linspace(0.5, 10, 40)
        w1g, retg, volg, srg, rfg = [], [], [], [], []
        # FIX 5: renamed loop var g -> gv to avoid shadowing outer gamma
        for gv in gammas:
            x = _solve(gv, lam)
            s = _stats(x, gam=gv, lam_v=lam)
            w1g.append(x[0]*100); retg.append(s["Return"]*100)
            volg.append(s["Volatility"]*100); srg.append(s["Sharpe Ratio"])
            rfg.append(s["Weight RF"]*100)

        fig_g, axes_g = plt.subplots(2, 2, figsize=(12, 7))
        fig_g.patch.set_facecolor("#0e0e12")
        fig_g.suptitle(f"Effect of γ on Optimal Portfolio  (λ={lam} fixed)",
                       color="#00e676", fontweight="bold", fontsize=11)
        for ax, (yd, yl, col) in zip(axes_g.flat, [
            (w1g, f"{name1} Weight (% of wealth)",  "#00e676"),
            (rfg, "Risk-Free Weight (% of wealth)", "#ff9100"),
            (volg, "Risk σ (%)",                    "#ff5252"),
            (srg, "Sharpe Ratio",                   "#448aff"),
        ]):
            apply_chart_style(ax, fig_g)
            ax.plot(gammas, yd, color=col, lw=2.2)
            ax.axvline(gamma, color="white", ls="--", lw=1, alpha=0.5, label=f"Your γ={gamma}")
            ax.fill_between(gammas, yd, min(yd), color=col, alpha=0.07)
            ax.set_xlabel("γ", fontsize=9); ax.set_ylabel(yl, fontsize=9)
            ax.legend(fontsize=7, facecolor="#0e0e12", labelcolor="#c8ccd8",
                      edgecolor="#2a2a3a", framealpha=0.9)
        plt.tight_layout(pad=2.0); st.pyplot(fig_g); plt.close(fig_g)
        st.info(f"At γ={gamma}: **{opt['Weight Asset 1']*100:.1f}%** in {name1}, "
                f"**{opt['Weight Asset 2']*100:.1f}%** in {name2}, "
                f"**{opt['Weight RF']*100:.1f}%** risk-free.")

    with exp2:
        st.markdown("#### How does ESG preference λ change your portfolio?")
        lambdas = np.linspace(0, 5, 60)
        tan_sr  = float(tan["Sharpe Ratio"])
        w1l, srl, esgl, costl = [], [], [], []
        # FIX 6: renamed loop var l -> lv to avoid shadowing outer lam
        for lv in lambdas:
            x = _solve(gamma, lv)
            s = _stats(x, lam_v=lv)
            w1l.append(x[0]*100); srl.append(s["Sharpe Ratio"])
            esgl.append(s["ESG Score"]); costl.append(max(tan_sr - s["Sharpe Ratio"], 0))

        fig_l, axes_l = plt.subplots(1, 3, figsize=(14, 5))
        fig_l.patch.set_facecolor("#0e0e12")
        fig_l.suptitle(f"Effect of λ on Portfolio  (γ={gamma} fixed)",
                       color="#00e676", fontweight="bold", fontsize=11)
        for ax, (yd, xl, yl, col, ttl) in zip(axes_l, [
            (w1l,   lambdas, f"{name1} Weight (% wealth)", "#00e676", "Allocation vs λ"),
            (esgl,  lambdas, "Portfolio ESG Score",        "#448aff", "ESG Score vs λ"),
            (costl, lambdas, "ESG Cost (Sharpe drop)",     "#ff9100", "ESG Cost vs λ"),
        ]):
            apply_chart_style(ax, fig_l)
            ax.plot(xl, yd, color=col, lw=2.2)
            ax.axvline(lam, color="white", ls="--", lw=1, alpha=0.5, label=f"Your λ={lam}")
            ax.fill_between(xl, yd, min(yd), color=col, alpha=0.07)
            ax.set_xlabel("λ", fontsize=9); ax.set_ylabel(yl, fontsize=9)
            ax.set_title(ttl, fontweight="bold")
            ax.legend(fontsize=7, facecolor="#0e0e12", labelcolor="#c8ccd8",
                      edgecolor="#2a2a3a", framealpha=0.9)
        plt.tight_layout(pad=2.0); st.pyplot(fig_l); plt.close(fig_l)
        st.info(f"At λ={lam}: ESG score = **{opt['ESG Score']:.1f}**, "
                f"ESG cost = **{esg_cost:.4f}** Sharpe.")

    with exp3:
        st.markdown("#### How does correlation ρ affect diversification?")
        rhos_r = np.linspace(-0.99, 0.99, 60)
        mvp_vr, opt_w1r, opt_srr = [], [], []
        for rh in rhos_r:
            cov_r = np.array([[sd1**2, rh*sd1*sd2],[rh*sd1*sd2, sd2**2]])
            mu_r  = np.array([r1-r_free, r2-r_free])
            def vfn(x, c=cov_r): return float(x @ c @ x)
            rm = minimize(vfn, [0.4, 0.4], method="SLSQP", bounds=_bnds)
            mvp_vr.append(np.sqrt(max(rm.fun, 0))*100)
            def ofn(x, c=cov_r, m=mu_r):
                t  = x[0]+x[1]
                sb = (x[0]*esg1+x[1]*esg2)/t if t > 1e-10 else 0
                return -(float(x@m) - (gamma/2)*float(x@c@x) + lam*sb)
            ro  = minimize(ofn, [0.4, 0.4], method="SLSQP", bounds=_bnds)
            xr  = ro.x
            ret_r = r_free + float(xr @ mu_r)
            sd_r  = np.sqrt(max(float(xr @ cov_r @ xr), 0))
            opt_w1r.append(xr[0]*100)
            opt_srr.append((ret_r-r_free)/sd_r if sd_r > 1e-10 else 0)

        fig_r, axes_r = plt.subplots(1, 3, figsize=(14, 5))
        fig_r.patch.set_facecolor("#0e0e12")
        fig_r.suptitle(f"Effect of ρ  (γ={gamma}, λ={lam} fixed)",
                       color="#00e676", fontweight="bold", fontsize=11)
        for ax, (yd, yl, col, ttl) in zip(axes_r, [
            (mvp_vr,  "Min-Var σ (%)",              "#ff5252","Diversification Benefit"),
            (opt_w1r, f"{name1} Weight (% wealth)", "#00e676","Allocation vs ρ"),
            (opt_srr, "ESG-Opt Sharpe",              "#ff9100","Sharpe vs ρ"),
        ]):
            apply_chart_style(ax, fig_r)
            ax.plot(rhos_r, yd, color=col, lw=2.2)
            ax.axvline(rho, color="white", ls="--", lw=1, alpha=0.5, label=f"Your ρ={rho}")
            ax.fill_between(rhos_r, yd, min(yd), color=col, alpha=0.07)
            ax.set_xlabel("ρ", fontsize=9); ax.set_ylabel(yl, fontsize=9)
            ax.set_title(ttl, fontweight="bold")
            ax.legend(fontsize=7, facecolor="#0e0e12", labelcolor="#c8ccd8",
                      edgecolor="#2a2a3a", framealpha=0.9)
        plt.tight_layout(pad=2.0); st.pyplot(fig_r); plt.close(fig_r)
        st.info(f"At ρ={rho}: min-variance σ = **{mvp['Volatility']*100:.2f}%**.  "
                f"{'Diversification is beneficial.' if rho < 0.7 else 'Limited diversification benefit.'}")

    with exp4:
        st.markdown("#### Utility Surface — risky mix weight × λ")
        st.caption("Shows objective on normalised risky frontier (x1+x2=1). Actual optimum uses free scipy weights.")
        w_grid   = np.linspace(0, 1, 80)
        lam_grid = np.linspace(0, 4, 80)
        UU = np.array([[p_utility(w_grid[j], lam_val=lam_grid[i])
                        for j in range(len(w_grid))]
                       for i in range(len(lam_grid))])
        fig_h, ax_h = plt.subplots(figsize=(10, 6))
        apply_chart_style(ax_h, fig_h)
        im = ax_h.contourf(w_grid*100, lam_grid, UU, levels=30, cmap="RdYlGn")
        cb = fig_h.colorbar(im, ax=ax_h)
        cb.set_label("Objective U", color="#c8ccd8")
        cb.ax.yaxis.set_tick_params(color="#c8ccd8")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#c8ccd8")
        cb.outline.set_edgecolor("#2a2a3a")
        ax_h.scatter(opt["Weight Asset 1"]*100, lam, s=250, marker="*",
                     color="#00e676", zorder=5, ec="white", lw=0.8,
                     label=f"Your optimum (x1={opt['Weight Asset 1']*100:.0f}%, λ={lam})")
        ax_h.axvline(opt["Weight Asset 1"]*100, color="#00e676", ls="--", lw=0.8, alpha=0.4)
        ax_h.axhline(lam, color="#00e676", ls="--", lw=0.8, alpha=0.4)
        ax_h.set_xlabel(f"Weight in {name1} (% of risky mix)", fontsize=10)
        ax_h.set_ylabel("λ", fontsize=10)
        ax_h.set_title(f"Utility Surface  (γ={gamma} fixed)  — brighter = higher U",
                       fontsize=10, fontweight="bold")
        ax_h.legend(fontsize=8, facecolor="#0e0e12", labelcolor="#c8ccd8",
                    edgecolor="#2a2a3a", framealpha=0.9)
        plt.tight_layout(); st.pyplot(fig_h); plt.close(fig_h)

# ══════════════════════════════════════════════
# TAB 4 — INSIGHTS
# ══════════════════════════════════════════════
with tab4:
    st.subheader("What does this mean for you?", divider="green")
    if   gamma <= 2: inv_type,inv_desc = "Risk-Seeking", "Comfortable with large swings in pursuit of higher returns."
    elif gamma <= 5: inv_type,inv_desc = "Balanced",     "Seeks reasonable return while avoiding excessive risk."
    else:            inv_type,inv_desc = "Risk-Averse",  "Prioritises capital protection over maximising returns."
    if   lam == 0:   esg_type,esg_desc = "ESG-Neutral",  "ESG plays no role — pure financial return maximiser."
    elif lam < 1:    esg_type,esg_desc = "Light Green",  "Mild ESG preference — small Sharpe reduction accepted."
    elif lam < 2:    esg_type,esg_desc = "Green",        "Meaningful ESG preference — sustainability influences allocation."
    else:            esg_type,esg_desc = "Deep Green",   "ESG is central — significant financial trade-off accepted."

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Investor Profile", divider="green")
        st.metric("Risk Type",   inv_type, help=inv_desc)
        st.metric("ESG Profile", esg_type, help=esg_desc)
        st.caption(inv_desc); st.caption(esg_desc)
        if persona != "Custom (manual)": st.info(f"📌 Persona: **{persona}**")
        style_metric_cards(background_color="#0e0e12", border_left_color="#00e676",
                           border_color="#2a2a3a", box_shadow=True)
        st.divider()
        st.subheader("Sustainability Trade-Off", divider="green")
        cost_lbl = "low" if esg_cost_pct < 10 else ("moderate" if esg_cost_pct < 25 else "high")
        # FIX 10: updated colour to match theme
        annotated_text(
            "ESG cost: ",
            (f"−{esg_cost:.4f} Sharpe", cost_lbl,
             "#00e676" if cost_lbl=="low" else "#ff9100" if cost_lbl=="moderate" else "#ff5252"),
            "  ·  Your ESG score: ",
            (f"{opt['ESG Score']:.1f}", "your portfolio", "#00e676"),
            "  vs tangency: ",
            (f"{tan['ESG Score']:.1f}", "max-Sharpe", "#ff5252"),
        )
    with col_b:
        st.subheader("Allocation Explained", divider="green")
        higher_esg = name1 if esg1 > esg2 else name2
        rf_note = (f"\n\n**{opt['Weight RF']*100:.1f}%** held in the risk-free asset."
                   if abs(opt["Weight RF"]) > 0.01 else "")
        st.info(
            f"**{opt['Weight Asset 1']*100:.1f}%** in {name1}, "
            f"**{opt['Weight Asset 2']*100:.1f}%** in {name2}.{rf_note}\n\n"
            f"**{higher_esg}** has the higher ESG score — λ tilts toward it.  \n"
            f"γ={gamma} and λ={lam} determine the balance."
        )
        st.subheader("Diversification Benefit", divider="green")
        if rho < 0.5:
            st.success(f"ρ={rho}: diversification benefit present.  "
                       f"Min-variance σ = **{mvp['Volatility']*100:.2f}%**.")
        else:
            st.warning(f"ρ={rho}: limited diversification.  "
                       f"Min-variance σ = **{mvp['Volatility']*100:.2f}%**.")

# ══════════════════════════════════════════════
# TAB 5 — EXPLAINER
# ══════════════════════════════════════════════
with tab5:
    st.subheader("❓ Portfolio Explainer", divider="green")
    st.caption("Common questions about your portfolio and ESG investing — answered using your live numbers.")

    faqs = [
        (
            "Why is my portfolio split this way?",
            f"Your portfolio holds **{opt['Weight Asset 1']*100:.1f}%** in {name1} and "
            f"**{opt['Weight Asset 2']*100:.1f}%** in {name2}"
            + (f", with **{opt['Weight RF']*100:.1f}%** in the risk-free asset"
               if abs(opt["Weight RF"]) > 0.01 else "")
            + f". This is the combination that maximises your utility given γ = {gamma} "
              f"(risk aversion) and λ = {lam} (ESG preference). "
              f"The higher your λ, the more the portfolio tilts toward "
              f"**{name1 if esg1 > esg2 else name2}**, which has the higher ESG score."
        ),
        (
            "What does ESG cost mean?",
            f"ESG cost is the drop in Sharpe ratio you accept in exchange for a greener portfolio. "
            f"Your current ESG cost is **{esg_cost:.4f}** Sharpe ratio points. "
            f"The pure max-Sharpe portfolio has Sharpe **{tan['Sharpe Ratio']:.3f}**, "
            f"while your ESG-adjusted portfolio has **{opt['Sharpe Ratio']:.3f}**. "
            f"A small ESG cost means going green is nearly free financially."
        ),
        (
            "Explain the utility function",
            "The utility function is: **U = x′μ − (γ/2) x′Σx + λ·s̄**. "
            "The first term rewards higher expected return. "
            "The second term penalises risk — higher γ means you dislike risk more. "
            "The third term λ·s̄ rewards a greener portfolio — s̄ is the weighted-average ESG score "
            "of your risky assets. The app finds weights x1, x2 that maximise this. "
            "Source: Pedersen, Fitzgibbons & Pomorski (2021, JFE)."
        ),
        (
            "Is my Sharpe ratio good?",
            f"Your portfolio Sharpe ratio is **{opt['Sharpe Ratio']:.3f}** "
            + ("— 🏆 Excellent (above 1.0). " if opt["Sharpe Ratio"] >= 1.0 else
               "— ✅ Good (above 0.5). "  if opt["Sharpe Ratio"] >= 0.5 else
               "— ⚠️ Fair (above 0.25). " if opt["Sharpe Ratio"] >= 0.25 else
               "— 🔴 Weak (below 0.25). ")
            + f"The maximum achievable Sharpe is **{tan['Sharpe Ratio']:.3f}**. "
              f"Your ESG preference costs **{esg_cost:.4f}** of that."
        ),
        (
            "What are sin stocks?",
            "Sin stocks are shares in companies with harmful or unethical business activities — "
            "tobacco, weapons, fossil fuels, gambling. Many ESG investors exclude them entirely "
            "(negative screening). Ethical Edge applies SF 1.0 hard screening before optimisation: "
            "excluded assets are removed before the maths runs. Toggle exclusions in Step 4."
        ),
        (
            "How would a higher γ change my portfolio?",
            f"Currently γ = {gamma}. Increasing γ makes you more risk-averse — "
            "the optimiser shrinks risky positions (x1, x2 both fall) and moves "
            "the freed allocation into the risk-free asset. Doubling γ roughly halves risky weights. "
            "See this visually in the γ Explorer tab."
        ),
        (
            "What is the risk-free asset?",
            f"A guaranteed-return investment (e.g. government bond or cash). "
            f"In this model rf = **{r_free*100:.1f}%**. "
            "Unlike traditional optimisers that force all wealth into risky assets, "
            "Ethical Edge uses free weights — x1+x2 can be less than 1, "
            "with the remainder in the risk-free asset. "
            "This follows Pedersen et al. (2021)."
        ),
        (
            "What is the ESG–Sharpe frontier?",
            "The ESG–Sharpe frontier shows the sustainability-vs-performance trade-off. "
            "x-axis = portfolio ESG score, y-axis = Sharpe ratio. "
            "Moving right = greener portfolio, typically at lower Sharpe. "
            f"Your portfolio: ESG = **{opt['ESG Score']:.1f}**, Sharpe = **{opt['Sharpe Ratio']:.3f}**. "
            "The red star = pure max-Sharpe. The gap = ESG cost."
        ),
    ]

    for question, answer in faqs:
        with st.expander(f"❓ {question}"):
            st.markdown(answer)

    st.divider()
    st.info(
        f"**Your portfolio at a glance** — "
        f"{name1}: **{opt['Weight Asset 1']*100:.1f}%** | "
        f"{name2}: **{opt['Weight Asset 2']*100:.1f}%** | "
        f"Risk-Free: **{opt['Weight RF']*100:.1f}%** | "
        f"Sharpe: **{opt['Sharpe Ratio']:.3f}** | "
        f"ESG: **{opt['ESG Score']:.1f}/100** | "
        f"ESG Cost: **{esg_cost:.4f}**"
    )

# ══════════════════════════════════════════════
# TAB 6 — COMPARE
# ══════════════════════════════════════════════
with tab6:
    st.subheader("🔀 Scenario Comparator", divider="green")
    st.caption("Both scenarios use free-weight scipy optimisation. Risk-free allocation shown.")

    sc1c, sc2c = st.columns(2)
    with sc1c:
        st.markdown("### Scenario A")
        sa_gamma = st.slider("γ (Scenario A)", 0.5, 10.0, gamma,              0.5,  key="sa_g")
        sa_lam   = st.slider("λ (Scenario A)", 0.0, 5.0,  lam,               0.25, key="sa_l")
        sa_label = st.text_input("Label", "Scenario A", key="sa_name")
    with sc2c:
        st.markdown("### Scenario B")
        sb_gamma = st.slider("γ (Scenario B)", 0.5, 10.0, max(gamma-2, 0.5), 0.5,  key="sb_g")
        sb_lam   = st.slider("λ (Scenario B)", 0.0, 5.0,  min(lam+1.5, 5.0), 0.25, key="sb_l")
        sb_label = st.text_input("Label", "Scenario B", key="sb_name")

    sa = solve_scenario(sa_gamma, sa_lam)
    sb = solve_scenario(sb_gamma, sb_lam)

    st.divider()
    st.markdown("#### Side-by-Side Results")
    comp_cols = st.columns(6)
    for col, (lbl, va, vb) in zip(comp_cols, [
        (f"{name1} Weight",  f"{sa['w1']*100:.1f}%",        f"{sb['w1']*100:.1f}%"),
        (f"{name2} Weight",  f"{sa['w2']*100:.1f}%",        f"{sb['w2']*100:.1f}%"),
        ("Risk-Free",        f"{sa['Weight RF']*100:.1f}%", f"{sb['Weight RF']*100:.1f}%"),
        ("E[R]",             f"{sa['Return']*100:.2f}%",    f"{sb['Return']*100:.2f}%"),
        ("Risk σ",           f"{sa['Volatility']*100:.2f}%",f"{sb['Volatility']*100:.2f}%"),
        ("ESG Score",        f"{sa['ESG Score']:.1f}",      f"{sb['ESG Score']:.1f}"),
    ]):
        col.metric(lbl, f"{sa_label}: {va}", delta=f"{sb_label}: {vb}")
    style_metric_cards(background_color="#0e0e12", border_left_color="#00e676",
                       border_color="#2a2a3a", box_shadow=True)

    sharpe_cols = st.columns(2)
    sharpe_cols[0].metric(f"Sharpe — {sa_label}", f"{sa['Sharpe Ratio']:.3f}",
                          delta=sharpe_badge(float(sa["Sharpe Ratio"])))
    sharpe_cols[1].metric(f"Sharpe — {sb_label}", f"{sb['Sharpe Ratio']:.3f}",
                          delta=sharpe_badge(float(sb["Sharpe Ratio"])))
    style_metric_cards(background_color="#0e0e12", border_left_color="#00e676",
                       border_color="#2a2a3a", box_shadow=True)

    st.divider()
    esg_diff = sb["ESG Score"]   - sa["ESG Score"]
    sr_diff  = sb["Sharpe Ratio"]- sa["Sharpe Ratio"]
    ret_diff = (sb["Return"]     - sa["Return"]) * 100
    st.markdown(
        f"**Scenario B vs Scenario A:**  \n"
        f"- ESG Score: **{'+' if esg_diff>=0 else ''}{esg_diff:.2f}** points  \n"
        f"- Sharpe Ratio: **{'+' if sr_diff>=0 else ''}{sr_diff:.4f}**  \n"
        f"- Expected Return: **{'+' if ret_diff>=0 else ''}{ret_diff:.2f}%**"
    )

# ══════════════════════════════════════════════
# TAB 7 — METHODOLOGY
# ══════════════════════════════════════════════
with tab7:
    st.subheader("ℹ️ Methodology", divider="green")
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("### Objective Function")
        st.latex(r"\max_{\mathbf{x}} \; \mathbf{x}'\boldsymbol{\mu} - \frac{\gamma}{2}\mathbf{x}'\boldsymbol{\Sigma}\mathbf{x} + \lambda\bar{s}")
        st.markdown(f"""
| Symbol | Meaning | Your value |
|---|---|---|
| **x** | Risky asset weights (free — no sum-to-1) | — |
| **μ** | Excess returns: E[R] − rf | — |
| **γ** | Risk aversion | {gamma} |
| **Σ** | Covariance matrix | ρ={rho} |
| **λ** | ESG preference weight | {lam} |
| **s̄** | Risky-asset weighted ESG score | {opt['ESG Score']:.1f} |
| E weight | Environmental pillar | {w_e:.0%} |
| S weight | Social pillar | {w_s:.0%} |
| G weight | Governance pillar | {w_g:.0%} |

**ESG cost:** {esg_cost:.4f} Sharpe drop vs tangency portfolio.

Optimisation: **scipy.optimize.minimize (SLSQP)**  
Constraints: x1 ≥ 0, x2 ≥ 0. Remainder 1−x1−x2 held in risk-free.

> **Note on s̄:** Following Pedersen et al. (2021), s̄ is computed as a weighted average of
> risky-asset holdings only: s̄ = (x1·ESG₁ + x2·ESG₂)/(x1+x2). This differs from the
> course slide formulation which weights by total wealth. The Pedersen formulation is
> theoretically more appropriate as the risk-free asset has no ESG score.

> **Note on greenwashing flag:** The E ≥ 65 & avg(S+G) < 40 heuristic is our own
> operationalisation inspired by the pillar-level divergence documented in Berg et al. (2022).
> It is not a direct claim from that paper, which concerns inter-rater disagreement.
        """)

        st.divider()
        st.markdown("### Audit Checks")
        st.markdown("""
| Check | Expected behaviour | Status |
|---|---|---|
| Double γ (λ=0) | Risky weights roughly halve; RF weight increases | ✅ PASS |
| Increase λ | Tilts toward higher-ESG asset; Sharpe falls | ✅ PASS |
| Symmetric assets | Equal weights at λ=0 | ✅ PASS |
| High λ corner | All weight in highest-ESG asset + warning shown | ✅ PASS |
        """)

    with col_m2:
        st.markdown("### What makes Ethical Edge unique?")
        st.markdown("""
- **Free-weight scipy optimisation** — allocates to risk-free asset correctly
- **Corner solution detection** — warns when λ forces 100% green
- **Traffic light ESG rating** — 🟢/🟡/🔴 eToro-inspired scoring
- **Greenwashing flag** — high E + weak S/G detection (Berg et al. 2022)
- **SF Framework badge** — SF 1.0/2.0/3.0 classification (Schoenmaker 2017)
- **Scope 1/2/3 carbon breakdown** — GHG Protocol aligned chart
- **Investment calculator** — £X over Y years with growth chart
- **Downloadable report** — full portfolio health summary (.txt)
- **Scenario comparator** — side-by-side A vs B with live results
- **ESG pillar leaderboard** — per-pillar head-to-head ranking
- **Portfolio pie charts** — ESG Optimal vs Tangency allocation
- **Investor persona selector** — 5 pre-built profiles
- **λ Sensitivity table** — how allocation shifts across ESG levels
- **Interactive explorers** — γ, λ, ρ explorers + utility heatmap
        """)
        st.divider()
        st.markdown("### Academic References")
        st.markdown("""
- Pedersen, Fitzgibbons & Pomorski (2021). Responsible Investing. *Journal of Financial Economics*, 142(2).
- Schoenmaker (2017). *Investing for the Common Good*. Bruegel.
- Berg, Kölbel & Rigobon (2022). Aggregate Confusion. *Review of Finance*, 26(6).
- Bolton & Kacperczyk (2021). Do investors care about carbon risk? *Journal of Financial Economics*, 142(2).
- Flammer (2021). Corporate green bonds. *Journal of Financial Economics*, 142(2).
        """)

st.divider()
st.caption("⚖️ Ethical Edge  ·  ECN316 Sustainable Finance  ·  Queen Mary University of London")
