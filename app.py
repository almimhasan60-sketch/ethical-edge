"""
Ethical Edge – Sustainable Portfolio Optimiser
ECN316 Sustainable Finance | QMUL Group Project
Utility: U = E[Rp] - (γ/2)·σ²p + λ·s̄  (Lecture 6)

NEW FEATURES (v4):
  1. Traffic light ESG rating (eToro-inspired)
  2. £ Investment calculator with projected value chart
  3. Carbon score visual
  4. Downloadable portfolio health report
  5. Side-by-side scenario comparator (A vs B)
  6. ESG pillar leaderboard
  7. Animated portfolio pie chart
  8. Sustainability story card
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
import io
import datetime

from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stoggle import stoggle
from streamlit_extras.mention import mention
from streamlit_extras.chart_container import chart_container
from annotated_text import annotated_text

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Ethical Edge",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Ethical Edge")
st.caption("Sustainable Portfolio Optimiser  ·  ECN316 Sustainable Finance  ·  QMUL")
st.latex(r"U = E[R_p] - \frac{\gamma}{2}\sigma_p^2 + \lambda\bar{s}")
st.divider()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Investor Profile")

    st.subheader("Quick Start — Persona", divider="green")
    st.caption("Pick a profile to auto-fill γ and λ.")
    persona = st.selectbox(
        "Investor persona",
        ["Custom (manual)",
         "🎓 Young Professional — growth-focused",
         "🌿 Impact Investor — ESG first",
         "⚖️ Balanced Saver — risk & return",
         "🏦 Retiree — capital protection",
         "📈 Pure Return Seeker — no ESG"],
    )
    persona_defaults = {
        "🎓 Young Professional — growth-focused":  (2.0, 1.0),
        "🌿 Impact Investor — ESG first":           (4.0, 4.0),
        "⚖️ Balanced Saver — risk & return":        (4.0, 1.5),
        "🏦 Retiree — capital protection":           (8.0, 0.5),
        "📈 Pure Return Seeker — no ESG":            (2.0, 0.0),
    }
    if persona != "Custom (manual)":
        pg, pl = persona_defaults[persona]
        st.success(f"γ = {pg}  ·  λ = {pl}")
    else:
        pg, pl = 4.0, 1.0

    st.divider()

    st.subheader("Step 1 — Risk Attitude", divider="green")
    q1 = st.radio("If your portfolio dropped 20%, you'd…",
        ["Sell immediately — I can't handle losses",
         "Hold steady and wait it out",
         "Buy more — it's a buying opportunity"], index=1)
    q2 = st.radio("Your investment horizon?",
        ["Under 2 years", "2–5 years", "5+ years"], index=1)
    risk_map = {
        ("Sell immediately — I can't handle losses","Under 2 years"):9.0,
        ("Sell immediately — I can't handle losses","2–5 years"):8.0,
        ("Sell immediately — I can't handle losses","5+ years"):6.0,
        ("Hold steady and wait it out","Under 2 years"):6.0,
        ("Hold steady and wait it out","2–5 years"):4.0,
        ("Hold steady and wait it out","5+ years"):3.0,
        ("Buy more — it's a buying opportunity","Under 2 years"):3.0,
        ("Buy more — it's a buying opportunity","2–5 years"):2.0,
        ("Buy more — it's a buying opportunity","5+ years"):1.0,
    }
    gamma_quiz = risk_map[(q1,q2)]
    gamma_default = pg if persona != "Custom (manual)" else gamma_quiz
    st.info(f"Quiz suggests γ = {gamma_quiz}" + (f"  ·  Persona γ = {pg}" if persona!="Custom (manual)" else ""))
    gamma = st.slider("Fine-tune γ", 0.5, 10.0, float(gamma_default), 0.5,
                      help="Higher γ = more risk-averse.")

    st.divider()
    st.subheader("Step 2 — ESG Commitment", divider="green")
    esg_label = st.select_slider("How important is sustainability to you?",
        options=["None (λ=0)","Low (λ=0.5)","Medium (λ=1)","High (λ=2)","Max (λ=4)"],
        value="Medium (λ=1)")
    lam_map = {"None (λ=0)":0.0,"Low (λ=0.5)":0.5,"Medium (λ=1)":1.0,"High (λ=2)":2.0,"Max (λ=4)":4.0}
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
    c1,c2 = st.columns(2)
    r1  = c1.number_input("E[R] (%)", value=13.0, step=0.5, key="r1") / 100
    sd1 = c2.number_input("σ (%)",    value=18.0, step=0.5, key="sd1") / 100
    d1,d2,d3 = st.columns(3)
    e1 = d1.number_input("E", value=40.0, step=1.0, min_value=0.0, max_value=100.0, key="e1")
    s1 = d2.number_input("S", value=35.0, step=1.0, min_value=0.0, max_value=100.0, key="s1")
    g1 = d3.number_input("G", value=30.0, step=1.0, min_value=0.0, max_value=100.0, key="g1")
    sin1 = st.multiselect("Sector flags", ["Tobacco","Weapons","Fossil Fuels","Gambling"], key="sin1")
    with st.expander("🌿 Carbon Scope Breakdown (optional)"):
        st.caption("Scope 1=direct, 2=energy, 3=supply chain (0–100, higher = more emissions)")
        scope1_1 = st.slider("Scope 1 emissions", 0, 100, 60, 5, key="sc1_1")
        scope2_1 = st.slider("Scope 2 emissions", 0, 100, 50, 5, key="sc2_1")
        scope3_1 = st.slider("Scope 3 emissions", 0, 100, 70, 5, key="sc3_1")

    st.markdown("**Asset 2**")
    name2 = st.text_input("Company name", value="Asset 2", key="name2")
    c3,c4 = st.columns(2)
    r2  = c3.number_input("E[R] (%)", value=7.0,  step=0.5, key="r2") / 100
    sd2 = c4.number_input("σ (%)",    value=22.0, step=0.5, key="sd2") / 100
    d4,d5,d6 = st.columns(3)
    e2 = d4.number_input("E", value=70.0, step=1.0, min_value=0.0, max_value=100.0, key="e2")
    s2 = d5.number_input("S", value=65.0, step=1.0, min_value=0.0, max_value=100.0, key="s2")
    g2 = d6.number_input("G", value=75.0, step=1.0, min_value=0.0, max_value=100.0, key="g2")
    sin2 = st.multiselect("Sector flags", ["Tobacco","Weapons","Fossil Fuels","Gambling"], key="sin2")
    with st.expander("🌿 Carbon Scope Breakdown (optional)"):
        st.caption("Scope 1=direct, 2=energy, 3=supply chain (0–100, higher = more emissions)")
        scope1_2 = st.slider("Scope 1 emissions", 0, 100, 30, 5, key="sc1_2")
        scope2_2 = st.slider("Scope 2 emissions", 0, 100, 25, 5, key="sc2_2")
        scope3_2 = st.slider("Scope 3 emissions", 0, 100, 40, 5, key="sc3_2")

    st.markdown("**Market**")
    mc1,mc2 = st.columns(2)
    rho    = mc1.number_input("Correlation ρ", -1.0, 1.0, 0.3, 0.05)
    r_free = mc2.number_input("Risk-free rate (%)", value=2.5, step=0.25) / 100

# ─────────────────────────────────────────────
# CORE MATHS
# ─────────────────────────────────────────────
esg1 = w_e*e1 + w_s*s1 + w_g*g1
esg2 = w_e*e2 + w_s*s2 + w_g*g2

def is_excluded(flags, score):
    if excl_tobacco  and "Tobacco"      in flags: return True
    if excl_weapons  and "Weapons"      in flags: return True
    if excl_fossil   and "Fossil Fuels" in flags: return True
    if excl_gambling and "Gambling"     in flags: return True
    if score < min_esg_score: return True
    return False

ex1 = is_excluded(sin1, esg1)
ex2 = is_excluded(sin2, esg2)

if ex1 and ex2:
    st.error("⛔ Both assets excluded. Please relax your filters.")
    st.stop()
if ex1: st.warning(f"⚠️ {name1} excluded — portfolio is 100% {name2}.")
if ex2: st.warning(f"⚠️ {name2} excluded — portfolio is 100% {name1}.")

weights = np.array([0.0]) if ex1 else (np.array([1.0]) if ex2 else np.linspace(0,1,1000))

def portfolio_ret(w1):    return w1*r1 + (1-w1)*r2
def portfolio_sd(w1):     return np.sqrt(w1**2*sd1**2+(1-w1)**2*sd2**2+2*rho*w1*(1-w1)*sd1*sd2)
def av_esg(w1):           return w1*esg1 + (1-w1)*esg2
def p_sharpe(w1, rf=None):
    rf_ = rf if rf is not None else r_free
    sd  = portfolio_sd(w1)
    return (portfolio_ret(w1)-rf_)/sd if sd>0 else -np.inf
def p_utility(w1, gam=None, lam_val=None):
    g = gam     if gam     is not None else gamma
    l = lam_val if lam_val is not None else lam
    return portfolio_ret(w1)-(g/2)*portfolio_sd(w1)**2+l*av_esg(w1)

p_ret,p_vol,p_esg,p_sh,p_util,p_wts = [],[],[],[],[],[]
for w in weights:
    p_ret.append(portfolio_ret(w)); p_vol.append(portfolio_sd(w))
    p_esg.append(av_esg(w));        p_sh.append(p_sharpe(w))
    p_util.append(p_utility(w));    p_wts.append(w)

portfolios = pd.DataFrame({
    "Weight Asset 1":p_wts,"Weight Asset 2":[1-w for w in p_wts],
    "Return":p_ret,"Volatility":p_vol,"ESG Score":p_esg,
    "Sharpe Ratio":p_sh,"Utility":p_util,
})
portfolios["MV_Utility"] = portfolios["Return"]-(gamma/2)*portfolios["Volatility"]**2

idx_opt = portfolios["Utility"].idxmax()
idx_mv  = portfolios["MV_Utility"].idxmax()
idx_tan = portfolios["Sharpe Ratio"].idxmax()
idx_mvp = portfolios["Volatility"].idxmin()
opt = portfolios.loc[idx_opt]
mv  = portfolios.loc[idx_mv]
tan = portfolios.loc[idx_tan]
mvp = portfolios.loc[idx_mvp]

esg_cost     = float(tan["Sharpe Ratio"]-opt["Sharpe Ratio"])
esg_cost_pct = min(abs(esg_cost)/max(abs(tan["Sharpe Ratio"]),0.001)*100, 100)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def sharpe_badge(sr):
    if sr>=1.0:  return "🏆 Excellent"
    if sr>=0.5:  return "✅ Good"
    if sr>=0.25: return "⚠️ Fair"
    return "🔴 Poor"

# FEATURE 1 — Traffic light ESG rating
def traffic_light(score):
    if score >= 65: return "🟢", "Strong",   "#16a34a"
    if score >= 40: return "🟡", "Moderate", "#d97706"
    return "🔴", "Weak", "#dc2626"

# FEATURE 3 — Carbon score (proxy from E-pillar, normalised 0–100)
def carbon_score(esg_composite, e_score, e_weight):
    # Higher E score = lower carbon footprint proxy
    raw = e_score * e_weight + esg_composite * (1 - e_weight)
    return min(max(raw, 0), 100)

def carbon_label(cs):
    if cs >= 65: return "🌿 Low Carbon",   "#16a34a"
    if cs >= 40: return "🌤️ Moderate",      "#d97706"
    return "🏭 High Carbon", "#dc2626"

def portfolio_summary():
    w1p = opt["Weight Asset 1"]*100; w2p = opt["Weight Asset 2"]*100
    alloc = (f"100% in **{name1}**" if w1p>=99 else
             f"100% in **{name2}**" if w2p>=99 else
             f"**{w1p:.0f}%** in **{name1}** and **{w2p:.0f}%** in **{name2}**")
    esg_d = "strong" if opt["ESG Score"]>=65 else ("moderate" if opt["ESG Score"]>=40 else "weak")
    sr_d  = sharpe_badge(float(opt["Sharpe Ratio"])).split(" ",1)[1].lower()
    return (f"Based on your profile, we recommend investing {alloc}. "
            f"Expected return **{opt['Return']*100:.1f}%**, risk σ = **{opt['Volatility']*100:.1f}%**, "
            f"ESG score **{opt['ESG Score']:.1f}/100** ({esg_d}), "
            f"Sharpe ratio **{opt['Sharpe Ratio']:.3f}** ({sr_d}).")

def build_sensitivity_table():
    rows = []
    for l in [0.0,0.5,1.0,2.0,4.0]:
        utils = [p_utility(w,lam_val=l) for w in weights]
        bi = int(np.argmax(utils)); w1 = weights[bi]
        rows.append({"λ":l,f"{name1}(%)":f"{w1*100:.1f}",f"{name2}(%)":f"{(1-w1)*100:.1f}",
                     "E[Rp]":f"{portfolio_ret(w1)*100:.2f}%","σp":f"{portfolio_sd(w1)*100:.2f}%",
                     "ESG":f"{av_esg(w1):.1f}","Sharpe":f"{p_sharpe(w1):.3f}",
                     "ESG Cost":f"{max(float(tan['Sharpe Ratio'])-p_sharpe(w1),0):.4f}"})
    return pd.DataFrame(rows)

def apply_chart_style(ax, fig):
    fig.patch.set_facecolor("#0e0e12")
    ax.set_facecolor("#0e0e12")
    ax.tick_params(colors="#c8ccd8",labelsize=9)
    ax.xaxis.label.set_color("#c8ccd8"); ax.yaxis.label.set_color("#c8ccd8")
    ax.title.set_color("#00e676")
    for sp in ax.spines.values(): sp.set_edgecolor("#2a2a3a")
    ax.grid(True,color="#1e1e2a",linewidth=0.6)

# FEATURE: Greenwashing flag (Berg et al. 2022 — ESG rating disagreement)
def greenwashing_flag(e, s, g):
    """
    Flag if E score is high but S or G are low — potential greenwashing.
    High E + low S/G = good environmental claims but poor social/governance.
    """
    avg_sg = (s + g) / 2
    if e >= 65 and avg_sg < 40:
        return True, "⚠️ Possible greenwashing — high E score but weak S/G"
    if e >= 65 and avg_sg < 55:
        return True, "🔶 Watch: strong environmental claims, moderate governance/social"
    return False, ""

# FEATURE: SF Framework badge (Schoenmaker 2017)
def sf_framework(lam_val, excl_any):
    """
    Classify the portfolio approach into Schoenmaker's SF typology.
    SF 1.0: sin stock exclusion only (λ=0 + exclusions)
    SF 2.0: ESG in utility function (λ>0)
    SF 3.0: ESG dominates (λ>=3)
    """
    if lam_val >= 3.0:
        return "🌍 SF 3.0 — Common Good", "ESG and environmental impact prioritised above financial returns.", "#16a34a"
    elif lam_val > 0:
        return "⚖️ SF 2.0 — Stakeholder Value", "ESG factors integrated into the utility function alongside financial return.", "#d97706"
    elif excl_any:
        return "🚫 SF 1.0 — Profit + Exclusions", "Profit maximisation while avoiding sin stocks.", "#dc2626"
    else:
        return "💰 Finance-as-Usual", "Pure financial return maximiser — no ESG considerations.", "#6b7280"

# FEATURE 4 — Build report text
def build_report():
    today = datetime.date.today().strftime("%d %B %Y")
    ico1, lbl1, _ = traffic_light(esg1)
    ico2, lbl2, _ = traffic_light(esg2)
    cs_port = carbon_score(opt["ESG Score"], opt["Weight Asset 1"]*e1+opt["Weight Asset 2"]*e2, w_e)
    cs_lbl, _ = carbon_label(cs_port)
    lines = [
        "=" * 60,
        "  ETHICAL EDGE — PORTFOLIO HEALTH REPORT",
        f"  Generated: {today}",
        "=" * 60,
        "",
        "INVESTOR PROFILE",
        f"  Persona:          {persona}",
        f"  Risk Aversion γ:  {gamma}",
        f"  ESG Preference λ: {lam}",
        f"  ESG Pillar Wts:   E:{w_e:.0%}  S:{w_s:.0%}  G:{w_g:.0%}",
        "",
        "ASSETS",
        f"  {name1}:  E[R]={r1*100:.1f}%  σ={sd1*100:.1f}%  ESG={esg1:.1f}/100 {ico1} {lbl1}",
        f"  {name2}:  E[R]={r2*100:.1f}%  σ={sd2*100:.1f}%  ESG={esg2:.1f}/100 {ico2} {lbl2}",
        f"  Correlation ρ={rho}  Risk-free rf={r_free*100:.1f}%",
        "",
        "OPTIMAL PORTFOLIO (ESG-Adjusted)",
        f"  {name1} weight:    {opt['Weight Asset 1']*100:.1f}%",
        f"  {name2} weight:    {opt['Weight Asset 2']*100:.1f}%",
        f"  Expected return:  {opt['Return']*100:.2f}%",
        f"  Risk (σ):         {opt['Volatility']*100:.2f}%",
        f"  ESG Score:        {opt['ESG Score']:.1f}/100",
        f"  Sharpe Ratio:     {opt['Sharpe Ratio']:.3f}  {sharpe_badge(float(opt['Sharpe Ratio']))}",
        f"  ESG Cost:         {esg_cost:.4f} Sharpe vs tangency portfolio",
        f"  Carbon Score:     {cs_port:.1f}/100  {cs_lbl}",
        "",
        "PORTFOLIO COMPARISON",
        f"  {'Portfolio':<30} {name1+' %':>10} {name2+' %':>10} {'Sharpe':>8}",
        "-" * 62,
        f"  {'MV Optimal (λ=0)':<30} {mv['Weight Asset 1']*100:>10.1f} {mv['Weight Asset 2']*100:>10.1f} {mv['Sharpe Ratio']:>8.3f}",
        f"  {'ESG Optimal (you)':<30} {opt['Weight Asset 1']*100:>10.1f} {opt['Weight Asset 2']*100:>10.1f} {opt['Sharpe Ratio']:>8.3f}",
        f"  {'Tangency (Max Sharpe)':<30} {tan['Weight Asset 1']*100:>10.1f} {tan['Weight Asset 2']*100:>10.1f} {tan['Sharpe Ratio']:>8.3f}",
        f"  {'Min Variance':<30} {mvp['Weight Asset 1']*100:>10.1f} {mvp['Weight Asset 2']*100:>10.1f} {'—':>8}",
        "",
        "UTILITY FUNCTION (ECN316 Lecture 6)",
        "  U = E[Rp] - (γ/2)·σ²p + λ·s̄",
        "",
        "ACADEMIC REFERENCES",
        "  Flammer (JFE 2021) — Corporate Green Bonds",
        "  Bolton & Kacperczyk (JFE 2021) — Carbon & Returns",
        "  Schoenmaker (2017) — Sustainable Finance Framework",
        "  Berg et al. (RF 2022) — ESG Rating Disagreement",
        "",
        "  Ethical Edge · ECN316 Sustainable Finance · QMUL",
        "=" * 60,
    ]
    return "\n".join(lines)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
    "📊 Results","📈 Charts","🔬 Explore",
    "💡 Insights","🤖 AI Explainer","🔀 Compare","ℹ️ Methodology"
])

# ══════════════════════════════════════════════
# TAB 1 — RESULTS
# ══════════════════════════════════════════════
with tab1:

    # FEATURE 1 — Traffic light ESG scores + FEATURE 3 carbon score
    st.subheader("Asset ESG Scores", divider="green")
    a1c,a2c = st.columns(2)
    for col, nm, score, es, ss, gs, exf in [
        (a1c, name1, esg1, e1, s1, g1, ex1),
        (a2c, name2, esg2, e2, s2, g2, ex2),
    ]:
        ico, lbl, clr = traffic_light(score)
        cs = carbon_score(score, es, w_e)
        cs_lbl, cs_clr = carbon_label(cs)
        col.metric(
            label=f"{ico} {nm}  —  {lbl}  {'⛔' if exf else ''}",
            value=f"{score:.1f} / 100",
            help=f"E:{es:.0f}  S:{ss:.0f}  G:{gs:.0f}"
        )
        col.caption(f"Carbon proxy: **{cs:.1f}/100**  {cs_lbl}")
    style_metric_cards(background_color="#0e0e12", border_left_color="#00e676",
                       border_color="#2a2a3a", box_shadow=True)

    # FEATURE: Greenwashing flag
    for nm, es, ss, gs in [(name1,e1,s1,g1),(name2,e2,s2,g2)]:
        flagged, msg = greenwashing_flag(es, ss, gs)
        if flagged:
            st.warning(f"**{nm}:** {msg} — *reference: Berg et al. (RF 2022) on ESG rating disagreement*")

    # FEATURE: SF Framework badge
    excl_any = excl_tobacco or excl_weapons or excl_fossil or excl_gambling or min_esg_score > 0
    sf_lbl, sf_desc, sf_clr = sf_framework(lam, excl_any)
    st.info(f"**Your Sustainable Finance Approach:** {sf_lbl}  \n{sf_desc}  \n*Source: Schoenmaker (2017) SF typology*")

    # FEATURE: Scope 1/2/3 carbon breakdown chart
    st.subheader("🌿 Carbon Scope Breakdown", divider="green")
    st.caption("Scope 1 = direct emissions · Scope 2 = purchased energy · Scope 3 = supply chain  (lower = greener)")
    fig_sc, ax_sc = plt.subplots(figsize=(8, 3.5))
    apply_chart_style(ax_sc, fig_sc)
    scopes = ["Scope 1\n(Direct)", "Scope 2\n(Energy)", "Scope 3\n(Supply Chain)"]
    vals1  = [scope1_1, scope2_1, scope3_1]
    vals2  = [scope1_2, scope2_2, scope3_2]
    x_sc   = np.arange(3); w_sc = 0.35
    bars_sc1 = ax_sc.bar(x_sc - w_sc/2, vals1, w_sc, label=name1, color="#ff5252", alpha=0.85)
    bars_sc2 = ax_sc.bar(x_sc + w_sc/2, vals2, w_sc, label=name2, color="#00e676", alpha=0.85)

    # Portfolio weighted scope scores
    w1_opt = float(opt["Weight Asset 1"]); w2_opt = float(opt["Weight Asset 2"])
    port_scopes = [w1_opt*scope1_1 + w2_opt*scope1_2,
                   w1_opt*scope2_1 + w2_opt*scope2_2,
                   w1_opt*scope3_1 + w2_opt*scope3_2]
    ax_sc.plot(x_sc, port_scopes, "D--", color="white", lw=1.5, ms=8,
               label=f"Your Portfolio (weighted)", zorder=5)

    ax_sc.set_xticks(x_sc); ax_sc.set_xticklabels(scopes, color="#c8ccd8")
    ax_sc.set_ylabel("Emission Intensity (0=low, 100=high)")
    ax_sc.set_title("Carbon Emissions by Scope — aligned with GHG Protocol", fontweight="bold")
    ax_sc.legend(facecolor="#0e0e12", labelcolor="#c8ccd8", edgecolor="#2a2a3a")
    ax_sc.set_ylim(0, 115)
    for bar in list(bars_sc1) + list(bars_sc2):
        ax_sc.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                   f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9, color="#c8ccd8")
    plt.tight_layout()
    st.pyplot(fig_sc)
    plt.close(fig_sc)
    st.caption("Lower scores = lower carbon intensity. Portfolio line shows your weighted average exposure across scopes.")

    st.divider()

    # FEATURE 8 — Sustainability story card
    port_cs = carbon_score(
        opt["ESG Score"],
        opt["Weight Asset 1"]*e1 + opt["Weight Asset 2"]*e2,
        w_e
    )
    cs_lbl_port, cs_clr_port = carbon_label(port_cs)
    higher_esg_name = name1 if esg1 > esg2 else name2
    story = (
        f"Your portfolio directs **{opt['Weight Asset 1']*100:.0f}%** into **{name1}** "
        f"and **{opt['Weight Asset 2']*100:.0f}%** into **{name2}**. "
        f"By prioritising **{higher_esg_name}** — the higher-ESG asset — "
        f"your investment actively supports {'environmental and social responsibility' if lam>1 else 'a mild sustainability tilt'}. "
        f"Your portfolio achieves a carbon score of **{port_cs:.1f}/100** ({cs_lbl_port}), "
        f"and an overall ESG score of **{opt['ESG Score']:.1f}/100**. "
        f"{'Every pound you invest carries a green premium — you are paying ' + f'{esg_cost:.4f}' + ' in Sharpe ratio for a more sustainable world.' if lam>0 and esg_cost>0.001 else 'With λ=0, your allocation is driven entirely by financial return and risk.'}"
    )
    st.success(f"🌱 **Your Sustainability Story**\n\n{story}")

    st.divider()

    # Plain-English recommendation
    st.info(f"💼 **Portfolio Recommendation**\n\n{portfolio_summary()}")

    # Key metrics
    st.subheader("Your Recommended Portfolio", divider="green")
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric(f"{name1} Weight", f"{opt['Weight Asset 1']*100:.1f}%")
    m2.metric(f"{name2} Weight", f"{opt['Weight Asset 2']*100:.1f}%")
    m3.metric("Expected Return",  f"{opt['Return']*100:.2f}%")
    m4.metric("Risk  σ",          f"{opt['Volatility']*100:.2f}%")
    m5.metric("ESG Score",        f"{opt['ESG Score']:.1f}")
    m6.metric("Sharpe Ratio",     f"{opt['Sharpe Ratio']:.3f}",
              delta=sharpe_badge(float(opt["Sharpe Ratio"])))
    style_metric_cards(background_color="#0e0e12", border_left_color="#00e676",
                       border_color="#2a2a3a", box_shadow=True)

    if lam>0 and esg_cost>0.001:
        st.success(f"🌱 λ={lam}: **ESG cost = {esg_cost:.4f}** Sharpe vs max-Sharpe portfolio.")
    elif lam==0:
        st.info("ℹ️ λ=0 — ESG plays no role. ESG Optimal = MV Optimal.")
    st.progress(esg_cost_pct/100,
                text=f"ESG cost: {esg_cost:.4f} Sharpe ({esg_cost_pct:.1f}% of max Sharpe)")

    st.divider()

    # FEATURE 2 — Investment calculator
    st.subheader("💰 Investment Calculator", divider="green")
    st.caption("See what your investment could be worth — comparing ESG Optimal vs pure Max-Sharpe.")
    inv_col1, inv_col2 = st.columns([1,2])
    with inv_col1:
        invest_amt  = st.number_input("Investment amount (£)", value=1000, step=100, min_value=100)
        invest_yrs  = st.slider("Time horizon (years)", 1, 30, 10)
        st.markdown("---")
        esg_final  = invest_amt * (1 + opt["Return"])**invest_yrs
        mv_final   = invest_amt * (1 + mv["Return"])**invest_yrs
        tan_final  = invest_amt * (1 + tan["Return"])**invest_yrs
        rf_final   = invest_amt * (1 + r_free)**invest_yrs

        st.metric("ESG Optimal final value",    f"£{esg_final:,.0f}",
                  delta=f"+£{esg_final-invest_amt:,.0f}")
        st.metric("Max-Sharpe final value",     f"£{tan_final:,.0f}",
                  delta=f"+£{tan_final-invest_amt:,.0f}")
        st.metric("Risk-free final value",      f"£{rf_final:,.0f}",
                  delta=f"+£{rf_final-invest_amt:,.0f}")
        style_metric_cards(background_color="#0e0e12", border_left_color="#00e676",
                           border_color="#2a2a3a", box_shadow=True)

    with inv_col2:
        years = np.arange(0, invest_yrs+1)
        fig_inv, ax_inv = plt.subplots(figsize=(8,4))
        apply_chart_style(ax_inv, fig_inv)
        ax_inv.plot(years, invest_amt*(1+opt["Return"])**years,
                    color="#00e676", lw=2.5, label=f"ESG Optimal ({opt['Return']*100:.1f}%/yr)")
        ax_inv.plot(years, invest_amt*(1+tan["Return"])**years,
                    color="#ff5252", lw=1.8, ls="--", label=f"Max Sharpe ({tan['Return']*100:.1f}%/yr)")
        ax_inv.plot(years, invest_amt*(1+mv["Return"])**years,
                    color="#448aff", lw=1.5, ls=":", label=f"MV Optimal ({mv['Return']*100:.1f}%/yr)")
        ax_inv.plot(years, invest_amt*(1+r_free)**years,
                    color="white", lw=1.2, ls="-.", alpha=0.5, label=f"Risk-free ({r_free*100:.1f}%/yr)")
        ax_inv.fill_between(years, invest_amt*(1+opt["Return"])**years,
                             invest_amt*(1+r_free)**years, color="#00e676", alpha=0.06)
        ax_inv.set_xlabel("Years", fontsize=10)
        ax_inv.set_ylabel("Portfolio Value (£)", fontsize=10)
        ax_inv.set_title(f"£{invest_amt:,} invested over {invest_yrs} years", fontsize=10, fontweight="bold")
        ax_inv.legend(fontsize=8, facecolor="#0e0e12", labelcolor="#c8ccd8",
                      edgecolor="#2a2a3a", framealpha=0.9)
        ax_inv.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"£{x:,.0f}"))
        plt.tight_layout()
        st.pyplot(fig_inv)
        plt.close(fig_inv)
        st.caption("⚠️ Assumes constant annual returns. Past performance does not guarantee future results.")

    st.divider()

    # Comparison table
    st.subheader("Portfolio Comparison Table", divider="green")
    comp_df = pd.DataFrame({
        "Portfolio":       ["MV Optimal (λ=0)","✅ ESG Optimal (you)","Tangency — Max Sharpe","Min Variance"],
        f"{name1}(%)":     [f"{mv['Weight Asset 1']*100:.1f}",f"{opt['Weight Asset 1']*100:.1f}",
                            f"{tan['Weight Asset 1']*100:.1f}",f"{mvp['Weight Asset 1']*100:.1f}"],
        f"{name2}(%)":     [f"{mv['Weight Asset 2']*100:.1f}",f"{opt['Weight Asset 2']*100:.1f}",
                            f"{tan['Weight Asset 2']*100:.1f}",f"{mvp['Weight Asset 2']*100:.1f}"],
        "E[Rp]":           [f"{mv['Return']*100:.2f}%",f"{opt['Return']*100:.2f}%",
                            f"{tan['Return']*100:.2f}%",f"{mvp['Return']*100:.2f}%"],
        "σp":              [f"{mv['Volatility']*100:.2f}%",f"{opt['Volatility']*100:.2f}%",
                            f"{tan['Volatility']*100:.2f}%",f"{mvp['Volatility']*100:.2f}%"],
        "ESG":             [f"{mv['ESG Score']:.1f}",f"{opt['ESG Score']:.1f}",
                            f"{tan['ESG Score']:.1f}",f"{mvp['ESG Score']:.1f}"],
        "Sharpe":          [f"{mv['Sharpe Ratio']:.3f}",f"{opt['Sharpe Ratio']:.3f}",
                            f"{tan['Sharpe Ratio']:.3f}","—"],
        "Rating":          [sharpe_badge(float(mv["Sharpe Ratio"])),
                            sharpe_badge(float(opt["Sharpe Ratio"])),
                            sharpe_badge(float(tan["Sharpe Ratio"])),"—"],
    })
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("λ Sensitivity Analysis", divider="green")
    st.caption("How allocation shifts as ESG preference λ increases.")
    st.dataframe(build_sensitivity_table(), hide_index=True, use_container_width=True)

    st.divider()

    # FEATURE 4 — Download report
    st.subheader("📄 Download Portfolio Report", divider="green")
    report_text = build_report()
    st.download_button(
        label="⬇️ Download Portfolio Health Report (.txt)",
        data=report_text,
        file_name=f"ethical_edge_report_{datetime.date.today()}.txt",
        mime="text/plain",
        help="Download a full summary of your portfolio settings and results.",
    )

# ══════════════════════════════════════════════
# TAB 2 — CHARTS
# ══════════════════════════════════════════════
with tab2:

    st.subheader("ESG-Efficient Frontier", divider="green")

    fig, axes = plt.subplots(1,2,figsize=(13,5.5))
    fig.patch.set_facecolor("#0e0e12")
    for ax in axes: apply_chart_style(ax,fig)

    ax = axes[0]
    sc = ax.scatter(portfolios["Volatility"]*100, portfolios["Return"]*100,
                    c=portfolios["ESG Score"], cmap="RdYlGn", s=12, alpha=0.85,
                    vmin=0, vmax=100, zorder=2)
    cbar = fig.colorbar(sc,ax=ax,pad=0.02)
    cbar.set_label("Portfolio ESG Score",color="#c8ccd8",fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#c8ccd8")
    plt.setp(cbar.ax.yaxis.get_ticklabels(),color="#c8ccd8",fontsize=8)
    cbar.outline.set_edgecolor("#2a2a3a")

    if tan["Volatility"]>0:
        sdr = np.linspace(0,portfolios["Volatility"].max()*1.25,200)
        ax.plot(sdr*100,(r_free+(tan["Return"]-r_free)/tan["Volatility"]*sdr)*100,
                "--",color="#448aff",lw=1.5,label="CML",alpha=0.85)
    sc2 = np.linspace(0.001,portfolios["Volatility"].max()*1.25,200)
    Umv = float(mv["Return"])-(gamma/2)*float(mv["Volatility"])**2
    ax.plot(sc2*100,(Umv+(gamma/2)*sc2**2)*100,":",color="#448aff",lw=1.5,label="MV Indiff.")
    Uesg=float(opt["Utility"])
    ax.plot(sc2*100,((Uesg-lam*float(opt["ESG Score"]))+(gamma/2)*sc2**2)*100,
            "-.",color="#ff9100",lw=1.5,label="ESG Indiff.")

    ax.scatter(0,r_free*100,s=90,marker="s",color="white",zorder=6,label="Risk-Free",ec="#555",lw=0.8)
    ax.scatter(mvp["Volatility"]*100,mvp["Return"]*100,s=100,marker="D",color="#00b0ff",zorder=6,label="Min Var.",ec="white",lw=0.5)
    ax.scatter(tan["Volatility"]*100,tan["Return"]*100,s=200,marker="*",color="#ff5252",zorder=7,label="Tangency",ec="white",lw=0.5)
    ax.scatter(mv["Volatility"]*100,mv["Return"]*100,s=140,marker="^",color="#448aff",zorder=6,label="MV Opt.",ec="white",lw=0.5)
    ax.scatter(opt["Volatility"]*100,opt["Return"]*100,s=200,marker="*",color="#00e676",zorder=7,
               label=f"ESG Opt. ({opt['Weight Asset 1']*100:.0f}%/{opt['Weight Asset 2']*100:.0f}%)",ec="white",lw=0.5)
    ax.set_xlabel("Risk — Std Dev (%)",fontsize=10); ax.set_ylabel("Expected Return (%)",fontsize=10)
    ax.set_title("Mean–Variance Space\n(colour = ESG score)",fontsize=10,fontweight="bold")
    ax.legend(fontsize=7,loc="upper left",facecolor="#0e0e12",labelcolor="#c8ccd8",framealpha=0.9,edgecolor="#2a2a3a")
    ax.set_ylim(min(r_free-0.005,portfolios["Return"].min()-0.005)*100,portfolios["Return"].max()*1.18*100)
    ax.set_xlim(-0.3,portfolios["Volatility"].max()*1.12*100)

    ax2=axes[1]
    ax2.plot(portfolios["ESG Score"],portfolios["Sharpe Ratio"],color="#00e676",lw=2.5)
    ax2.fill_between(portfolios["ESG Score"],portfolios["Sharpe Ratio"],
                     portfolios["Sharpe Ratio"].min(),color="#00e676",alpha=0.07)
    ax2.scatter(opt["ESG Score"],opt["Sharpe Ratio"],s=200,marker="*",color="#00e676",zorder=5,label="Your ESG Optimal",ec="white",lw=0.5)
    ax2.scatter(tan["ESG Score"],tan["Sharpe Ratio"],s=160,marker="*",color="#ff5252",zorder=5,label="Max Sharpe",ec="white",lw=0.5)
    ax2.axhline(float(tan["Sharpe Ratio"]),color="#ff5252",ls="--",lw=0.9,alpha=0.5)
    if esg_cost>0.001:
        ax2.annotate(f"ESG cost: −{esg_cost:.3f}",
                     xy=(float(opt["ESG Score"]),float(opt["Sharpe Ratio"])),
                     xytext=(float(opt["ESG Score"])+2,float(opt["Sharpe Ratio"])-0.03),
                     color="#ff9100",fontsize=9,fontweight="bold",
                     arrowprops=dict(arrowstyle="->",color="#ff9100",lw=1.2))
    ax2.set_xlabel("Portfolio ESG Score",fontsize=10); ax2.set_ylabel("Sharpe Ratio",fontsize=10)
    ax2.set_title("ESG–Sharpe Frontier",fontsize=10,fontweight="bold")
    ax2.legend(fontsize=8,facecolor="#0e0e12",labelcolor="#c8ccd8",framealpha=0.9,edgecolor="#2a2a3a")

    plt.tight_layout(pad=2.0)
    with chart_container(portfolios[["Weight Asset 1","Return","Volatility","ESG Score","Sharpe Ratio","Utility"]],
                         export_formats=["CSV"]):
        st.pyplot(fig,use_container_width=True)
    plt.close(fig)

    # FEATURE 7 — Portfolio pie chart
    st.subheader("Portfolio Allocation Pie", divider="green")
    pie_col1, pie_col2 = st.columns(2)

    def draw_pie(ax, fig_p, w1, w2, t1, t2, title):
        apply_chart_style(ax, fig_p)
        ax.set_facecolor("#0e0e12")
        sizes = [w1, w2] if w1+w2>0 else [0.5, 0.5]
        colors= ["#00e676","#448aff"]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=[t1,t2], colors=colors,
            autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor="#0e0e12",linewidth=2),
            textprops=dict(color="#c8ccd8",fontsize=9),
        )
        for at in autotexts: at.set_color("#0e0e12"); at.set_fontweight("bold"); at.set_fontsize(9)
        ax.set_title(title, color="#00e676", fontsize=10, fontweight="bold")

    with pie_col1:
        fig_p1,ax_p1 = plt.subplots(figsize=(4,4))
        fig_p1.patch.set_facecolor("#0e0e12")
        draw_pie(ax_p1,fig_p1,
                 float(opt["Weight Asset 1"]),float(opt["Weight Asset 2"]),
                 name1,name2,"ESG Optimal (you)")
        plt.tight_layout()
        st.pyplot(fig_p1); plt.close(fig_p1)

    with pie_col2:
        fig_p2,ax_p2 = plt.subplots(figsize=(4,4))
        fig_p2.patch.set_facecolor("#0e0e12")
        draw_pie(ax_p2,fig_p2,
                 float(tan["Weight Asset 1"]),float(tan["Weight Asset 2"]),
                 name1,name2,"Tangency / Max Sharpe")
        plt.tight_layout()
        st.pyplot(fig_p2); plt.close(fig_p2)

    # FEATURE 6 — ESG pillar leaderboard
    st.subheader("ESG Pillar Leaderboard", divider="green")
    st.caption("Which asset scores higher on each pillar?")

    pillars = ["Environmental","Social","Governance"]
    scores1 = [e1,s1,g1]; scores2 = [e2,s2,g2]
    for pillar, sc1, sc2 in zip(pillars, scores1, scores2):
        winner = name1 if sc1 >= sc2 else name2
        diff   = abs(sc1-sc2)
        ico    = "🌍" if pillar=="Environmental" else ("🤝" if pillar=="Social" else "🏛️")
        c_a, c_b, c_c = st.columns([3,3,2])
        c_a.metric(f"{ico} {pillar} — {name1}", f"{sc1:.0f}/100",
                   delta=f"{'↑' if sc1>sc2 else '↓'} {diff:.0f} vs {name2}")
        c_b.metric(f"{ico} {pillar} — {name2}", f"{sc2:.0f}/100",
                   delta=f"{'↑' if sc2>sc1 else '↓'} {diff:.0f} vs {name1}")
        c_c.markdown(f"**Winner:** {winner}")
        style_metric_cards(background_color="#0e0e12",border_left_color="#00e676",
                           border_color="#2a2a3a",box_shadow=False)

    st.divider()
    # ESG pillar bar chart
    st.subheader("ESG Pillar Breakdown", divider="green")
    fig2,ax3 = plt.subplots(figsize=(7,3.5))
    apply_chart_style(ax3,fig2)
    x=np.arange(3); width=0.35
    bars1=ax3.bar(x-width/2,[e1,s1,g1],width,label=name1,color="#00e676",alpha=0.85)
    bars2=ax3.bar(x+width/2,[e2,s2,g2],width,label=name2,color="#448aff",alpha=0.85)
    ax3.set_xticks(x); ax3.set_xticklabels(["Environmental","Social","Governance"],color="#c8ccd8")
    ax3.set_ylabel("Score (0–100)"); ax3.set_title(f"E/S/G: {name1} vs {name2}",fontweight="bold")
    ax3.legend(facecolor="#0e0e12",labelcolor="#c8ccd8",edgecolor="#2a2a3a"); ax3.set_ylim(0,115)
    for bar in list(bars1)+list(bars2):
        ax3.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.5,f"{bar.get_height():.0f}",
                 ha="center",va="bottom",fontsize=9,color="#c8ccd8")
    plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

# ══════════════════════════════════════════════
# TAB 3 — EXPLORE
# ══════════════════════════════════════════════
with tab3:
    st.subheader("Interactive Explorers", divider="green")
    exp1,exp2,exp3,exp4 = st.tabs(["🎚️ γ Explorer","🌿 λ Explorer","🔗 ρ Explorer","🌡️ Utility Heatmap"])

    with exp1:
        st.markdown("#### How does risk aversion γ change your portfolio?")
        gammas=np.linspace(0.5,10,40)
        w1g,retg,volg,srg=[],[],[],[]
        for g in gammas:
            utils=[p_utility(w,gam=g) for w in weights]
            bi=int(np.argmax(utils)); w=weights[bi]
            w1g.append(w*100); retg.append(portfolio_ret(w)*100)
            volg.append(portfolio_sd(w)*100); srg.append(p_sharpe(w))

        fig_g,axes_g=plt.subplots(2,2,figsize=(12,7))
        fig_g.patch.set_facecolor("#0e0e12")
        fig_g.suptitle(f"Effect of γ on Optimal Portfolio (λ={lam} fixed)",color="#00e676",fontweight="bold",fontsize=11)
        for ax,(yd,yl,col) in zip(axes_g.flat,[(w1g,f"{name1} Weight (%)",  "#00e676"),
                                                (volg,"Risk σ (%)",          "#ff5252"),
                                                (retg,"Expected Return (%)","#448aff"),
                                                (srg, "Sharpe Ratio",        "#ff9100")]):
            apply_chart_style(ax,fig_g)
            ax.plot(gammas,yd,color=col,lw=2.2)
            ax.axvline(gamma,color="white",ls="--",lw=1,alpha=0.5,label=f"Your γ={gamma}")
            ax.fill_between(gammas,yd,min(yd),color=col,alpha=0.07)
            ax.set_xlabel("γ",fontsize=9); ax.set_ylabel(yl,fontsize=9)
            ax.legend(fontsize=7,facecolor="#0e0e12",labelcolor="#c8ccd8",edgecolor="#2a2a3a",framealpha=0.9)
        plt.tight_layout(pad=2.0); st.pyplot(fig_g); plt.close(fig_g)
        st.info(f"At γ={gamma}: **{opt['Weight Asset 1']*100:.1f}%** in {name1}, σ=**{opt['Volatility']*100:.2f}%**, Sharpe=**{opt['Sharpe Ratio']:.3f}**")

    with exp2:
        st.markdown("#### How does ESG preference λ change your portfolio?")
        lambdas=np.linspace(0,5,60); tan_sr=float(tan["Sharpe Ratio"])
        w1l,srl,esgl,costl=[],[],[],[]
        for l in lambdas:
            utils=[p_utility(w,lam_val=l) for w in weights]
            bi=int(np.argmax(utils)); w=weights[bi]
            w1l.append(w*100); sr=p_sharpe(w)
            srl.append(sr); esgl.append(av_esg(w)); costl.append(max(tan_sr-sr,0))

        fig_l,axes_l=plt.subplots(1,3,figsize=(14,5))
        fig_l.patch.set_facecolor("#0e0e12")
        fig_l.suptitle(f"Effect of λ (γ={gamma} fixed)",color="#00e676",fontweight="bold",fontsize=11)
        for ax,(yd,xl,yl,col,ttl) in zip(axes_l,[
            (w1l,  lambdas,f"{name1} Weight (%)","#00e676","Allocation vs λ"),
            (esgl, lambdas,"Portfolio ESG Score","#448aff","ESG Score vs λ"),
            (costl,lambdas,"ESG Cost (Sharpe)","#ff9100","ESG Cost vs λ"),
        ]):
            apply_chart_style(ax,fig_l)
            ax.plot(xl,yd,color=col,lw=2.2)
            ax.axvline(lam,color="white",ls="--",lw=1,alpha=0.5,label=f"Your λ={lam}")
            ax.fill_between(xl,yd,min(yd),color=col,alpha=0.07)
            ax.set_xlabel("λ",fontsize=9); ax.set_ylabel(yl,fontsize=9); ax.set_title(ttl,fontweight="bold")
            ax.legend(fontsize=7,facecolor="#0e0e12",labelcolor="#c8ccd8",edgecolor="#2a2a3a",framealpha=0.9)
        plt.tight_layout(pad=2.0); st.pyplot(fig_l); plt.close(fig_l)
        st.info(f"At λ={lam}: ESG score=**{opt['ESG Score']:.1f}**, ESG cost=**{esg_cost:.4f}** Sharpe.")

    with exp3:
        st.markdown("#### How does correlation ρ affect diversification?")
        rhos=np.linspace(-0.99,0.99,60)
        mvp_vol_r,opt_w1_r,opt_sr_r=[],[],[]
        for rh in rhos:
            def sd_r(w1,rh=rh): return np.sqrt(w1**2*sd1**2+(1-w1)**2*sd2**2+2*rh*w1*(1-w1)*sd1*sd2)
            def ut_r(w1,rh=rh): return portfolio_ret(w1)-(gamma/2)*sd_r(w1)**2+lam*av_esg(w1)
            def sh_r(w1,rh=rh): sd=sd_r(w1); return (portfolio_ret(w1)-r_free)/sd if sd>0 else -np.inf
            vols_=[sd_r(w) for w in weights]
            mvp_vol_r.append(vols_[int(np.argmin(vols_))]*100)
            bi=int(np.argmax([ut_r(w) for w in weights]))
            opt_w1_r.append(weights[bi]*100); opt_sr_r.append(sh_r(weights[bi]))

        fig_r,axes_r=plt.subplots(1,3,figsize=(14,5))
        fig_r.patch.set_facecolor("#0e0e12")
        fig_r.suptitle(f"Effect of Correlation ρ (γ={gamma}, λ={lam} fixed)",color="#00e676",fontweight="bold",fontsize=11)
        for ax,(yd,yl,col,ttl) in zip(axes_r,[
            (mvp_vol_r,"Min-Var σ (%)","#ff5252","Diversification Benefit"),
            (opt_w1_r, f"Optimal {name1} (%)","#00e676","Allocation vs ρ"),
            (opt_sr_r, "ESG-Opt Sharpe","#ff9100","Sharpe vs ρ"),
        ]):
            apply_chart_style(ax,fig_r)
            ax.plot(rhos,yd,color=col,lw=2.2)
            ax.axvline(rho,color="white",ls="--",lw=1,alpha=0.5,label=f"Your ρ={rho}")
            ax.fill_between(rhos,yd,min(yd),color=col,alpha=0.07)
            ax.set_xlabel("ρ",fontsize=9); ax.set_ylabel(yl,fontsize=9); ax.set_title(ttl,fontweight="bold")
            ax.legend(fontsize=7,facecolor="#0e0e12",labelcolor="#c8ccd8",edgecolor="#2a2a3a",framealpha=0.9)
        plt.tight_layout(pad=2.0); st.pyplot(fig_r); plt.close(fig_r)
        st.info(f"At ρ={rho}: min-variance σ=**{mvp['Volatility']*100:.2f}%**. {'Diversification is beneficial.' if rho<0.7 else 'Limited diversification benefit.'}")

    with exp4:
        st.markdown("#### Utility Surface — weight × λ")
        w_grid=np.linspace(0,1,80); lam_grid=np.linspace(0,4,80)
        WW,LL=np.meshgrid(w_grid,lam_grid)
        UU=np.array([[p_utility(w_grid[j],lam_val=lam_grid[i]) for j in range(len(w_grid))] for i in range(len(lam_grid))])
        fig_h,ax_h=plt.subplots(figsize=(10,6))
        apply_chart_style(ax_h,fig_h)
        im=ax_h.contourf(WW*100,LL,UU,levels=30,cmap="RdYlGn")
        cb=fig_h.colorbar(im,ax=ax_h)
        cb.set_label("Utility U",color="#c8ccd8"); cb.ax.yaxis.set_tick_params(color="#c8ccd8")
        plt.setp(cb.ax.yaxis.get_ticklabels(),color="#c8ccd8"); cb.outline.set_edgecolor("#2a2a3a")
        ax_h.scatter(opt["Weight Asset 1"]*100,lam,s=250,marker="*",color="#00e676",zorder=5,ec="white",lw=0.8,
                     label=f"Your optimum (w₁={opt['Weight Asset 1']*100:.0f}%, λ={lam})")
        ax_h.axvline(opt["Weight Asset 1"]*100,color="#00e676",ls="--",lw=0.8,alpha=0.4)
        ax_h.axhline(lam,color="#00e676",ls="--",lw=0.8,alpha=0.4)
        ax_h.set_xlabel(f"Weight in {name1} (%)",fontsize=10); ax_h.set_ylabel("λ",fontsize=10)
        ax_h.set_title(f"Utility Surface (γ={gamma} fixed)  — brighter = higher U",fontsize=10,fontweight="bold")
        ax_h.legend(fontsize=8,facecolor="#0e0e12",labelcolor="#c8ccd8",edgecolor="#2a2a3a",framealpha=0.9)
        plt.tight_layout(); st.pyplot(fig_h); plt.close(fig_h)

# ══════════════════════════════════════════════
# TAB 4 — INSIGHTS
# ══════════════════════════════════════════════
with tab4:
    st.subheader("What does this mean for you?", divider="green")
    if   gamma<=2: inv_type,inv_desc="Risk-Seeking","Comfortable with large swings in pursuit of higher returns."
    elif gamma<=5: inv_type,inv_desc="Balanced","Seeks a reasonable return while avoiding excessive risk."
    else:          inv_type,inv_desc="Risk-Averse","Prioritises capital protection over maximising returns."
    if   lam==0:  esg_type,esg_desc="ESG-Neutral","ESG plays no role — pure financial return maximiser."
    elif lam<1:   esg_type,esg_desc="Light Green","Mild ESG preference — accepts a small Sharpe reduction."
    elif lam<2:   esg_type,esg_desc="Green","Meaningful ESG preference — sustainability influences allocation."
    else:         esg_type,esg_desc="Deep Green","ESG is central — willing to accept a significant financial trade-off."

    col_a,col_b=st.columns(2)
    with col_a:
        st.subheader("Investor Profile",divider="green")
        st.metric("Risk Type",inv_type,help=inv_desc); st.metric("ESG Profile",esg_type,help=esg_desc)
        st.caption(inv_desc); st.caption(esg_desc)
        if persona!="Custom (manual)": st.info(f"📌 Persona: **{persona}**")
        style_metric_cards(background_color="#0e0e12",border_left_color="#00e676",border_color="#2a2a3a",box_shadow=True)
        st.divider()
        st.subheader("Sustainability Trade-Off",divider="green")
        cost_lbl="low" if esg_cost_pct<10 else ("moderate" if esg_cost_pct<25 else "high")
        annotated_text(
            "ESG cost: ",
            (f"−{esg_cost:.4f} Sharpe",cost_lbl,
             "#00c853" if cost_lbl=="low" else "#ff9100" if cost_lbl=="moderate" else "#ff5252"),
            "  ·  Your ESG score: ",
            (f"{opt['ESG Score']:.1f}","your portfolio","#00c853"),
            "  vs tangency: ",
            (f"{tan['ESG Score']:.1f}","max-Sharpe","#ff5252"),
        )
    with col_b:
        st.subheader("Allocation Explained",divider="green")
        higher_esg=name1 if esg1>esg2 else name2
        higher_ret=name1 if r1>r2 else name2
        st.info(f"**{opt['Weight Asset 1']*100:.1f}%** in {name1} and **{opt['Weight Asset 2']*100:.1f}%** in {name2}.  \n\n"
                f"**{higher_esg}** has the higher ESG score — λ tilts toward it.  \n"
                f"**{higher_ret}** offers the higher expected return.  \nγ={gamma} and λ={lam} determine the balance.")
        st.subheader("Diversification Benefit",divider="green")
        if rho<0.5:
            st.success(f"ρ={rho}: diversification benefit is present. Min-variance σ=**{mvp['Volatility']*100:.2f}%**, "
                       f"below both {name1} ({sd1*100:.1f}%) and {name2} ({sd2*100:.1f}%).")
        else:
            st.warning(f"ρ={rho}: limited diversification. Min-variance σ=**{mvp['Volatility']*100:.2f}%**.")

# ══════════════════════════════════════════════
# TAB 5 — AI EXPLAINER
# ══════════════════════════════════════════════
with tab5:
    st.subheader("🤖 AI Portfolio Explainer", divider="green")
    st.caption("Ask anything about your portfolio, ESG investing, or the theory. The AI knows your current settings.")

    portfolio_context = f"""
You are an expert in sustainable finance and portfolio theory, helping a retail investor using the Ethical Edge app.

CURRENT PORTFOLIO:
- {name1}: E[R]={r1*100:.1f}%, σ={sd1*100:.1f}%, ESG={esg1:.1f}/100
- {name2}: E[R]={r2*100:.1f}%, σ={sd2*100:.1f}%, ESG={esg2:.1f}/100
- ρ={rho}, rf={r_free*100:.1f}%, γ={gamma}, λ={lam}
- ESG weights: E:{w_e:.0%} S:{w_s:.0%} G:{w_g:.0%}

RESULTS:
- ESG Optimal: {opt['Weight Asset 1']*100:.1f}% {name1} / {opt['Weight Asset 2']*100:.1f}% {name2}
- E[R]={opt['Return']*100:.2f}%, σ={opt['Volatility']*100:.2f}%, ESG={opt['ESG Score']:.1f}, Sharpe={opt['Sharpe Ratio']:.3f}
- ESG Cost={esg_cost:.4f} Sharpe vs tangency
- Persona: {persona}

Answer clearly and concisely in plain English. Reference specific numbers when relevant.
Use the ECN316 framework (Schoenmaker SF typology, Markowitz MV theory, ESG cost concept).
Keep answers under 200 words unless more detail is needed.
"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"]=="assistant" else "👤"):
            st.markdown(msg["content"])

    if not st.session_state.chat_history:
        st.markdown("**💬 Try asking:**")
        cols=st.columns(3)
        suggestions=["Why is my portfolio split this way?","What does the ESG cost mean?",
                     "Explain the utility function simply","Is my Sharpe ratio good?",
                     "What are sin stocks?","How would a higher γ change my portfolio?"]
        for i,sug in enumerate(suggestions):
            with cols[i%3]:
                if st.button(sug,key=f"sug_{i}",use_container_width=True):
                    st.session_state.chat_history.append({"role":"user","content":sug})
                    st.rerun()

    user_input=st.chat_input("Ask about your portfolio or ESG investing…")
    if user_input:
        st.session_state.chat_history.append({"role":"user","content":user_input})
        st.rerun()

    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"]=="user":
        with st.chat_message("assistant",avatar="🤖"):
            with st.spinner("Thinking…"):
                try:
                    resp=requests.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={"Content-Type":"application/json"},
                        json={"model":"claude-sonnet-4-6","max_tokens":1000,
                              "system":portfolio_context,
                              "messages":[{"role":m["role"],"content":m["content"]}
                                          for m in st.session_state.chat_history]},
                        timeout=30,
                    )
                    reply=resp.json()["content"][0]["text"]
                except Exception as e:
                    reply=f"Sorry, couldn't connect to AI. Error: {e}"
            st.markdown(reply)
            st.session_state.chat_history.append({"role":"assistant","content":reply})

    if st.session_state.chat_history:
        if st.button("🗑️ Clear conversation",key="clear_chat"):
            st.session_state.chat_history=[]; st.rerun()

# ══════════════════════════════════════════════
# TAB 6 — SCENARIO COMPARATOR  (FEATURE 5)
# ══════════════════════════════════════════════
with tab6:
    st.subheader("🔀 Scenario Comparator", divider="green")
    st.caption("Build two portfolios side by side and compare them instantly. Great for exploring 'what if' questions.")

    sc1c,sc2c=st.columns(2)

    with sc1c:
        st.markdown("### Scenario A")
        sa_gamma = st.slider("γ (Scenario A)", 0.5, 10.0, gamma, 0.5, key="sa_g")
        sa_lam   = st.slider("λ (Scenario A)", 0.0, 5.0,  lam,   0.25, key="sa_l")
        sa_label = st.text_input("Label", "Scenario A", key="sa_name")

    with sc2c:
        st.markdown("### Scenario B")
        sb_gamma = st.slider("γ (Scenario B)", 0.5, 10.0, max(gamma-2,0.5), 0.5, key="sb_g")
        sb_lam   = st.slider("λ (Scenario B)", 0.0, 5.0,  min(lam+1.5,5.0), 0.25, key="sb_l")
        sb_label = st.text_input("Label", "Scenario B", key="sb_name")

    def solve_scenario(gam, lam_v):
        utils=[p_utility(w,gam=gam,lam_val=lam_v) for w in weights]
        bi=int(np.argmax(utils)); w1=weights[bi]
        return {"w1":w1,"ret":portfolio_ret(w1),"sd":portfolio_sd(w1),
                "esg":av_esg(w1),"sharpe":p_sharpe(w1),
                "utility":p_utility(w1,gam=gam,lam_val=lam_v)}

    sa=solve_scenario(sa_gamma,sa_lam)
    sb=solve_scenario(sb_gamma,sb_lam)

    st.divider()
    st.markdown("#### Side-by-Side Results")

    comp_cols=st.columns(5)
    metrics=[
        (f"{name1} Weight",f"{sa['w1']*100:.1f}%",f"{sb['w1']*100:.1f}%"),
        (f"{name2} Weight",f"{(1-sa['w1'])*100:.1f}%",f"{(1-sb['w1'])*100:.1f}%"),
        ("Expected Return",f"{sa['ret']*100:.2f}%",f"{sb['ret']*100:.2f}%"),
        ("Risk σ",         f"{sa['sd']*100:.2f}%",  f"{sb['sd']*100:.2f}%"),
        ("ESG Score",      f"{sa['esg']:.1f}",       f"{sb['esg']:.1f}"),
    ]
    for col,(lbl,va,vb) in zip(comp_cols,metrics):
        col.metric(lbl,f"{sa_label}: {va}",delta=f"{sb_label}: {vb}")
    style_metric_cards(background_color="#0e0e12",border_left_color="#00e676",
                       border_color="#2a2a3a",box_shadow=True)

    sharpe_comp=st.columns(2)
    sharpe_comp[0].metric(f"Sharpe — {sa_label}",f"{sa['sharpe']:.3f}",
                          delta=sharpe_badge(sa['sharpe']))
    sharpe_comp[1].metric(f"Sharpe — {sb_label}",f"{sb['sharpe']:.3f}",
                          delta=sharpe_badge(sb['sharpe']))
    style_metric_cards(background_color="#0e0e12",border_left_color="#00e676",
                       border_color="#2a2a3a",box_shadow=True)

    # Side-by-side frontier chart
    st.divider()
    fig_sc,axes_sc=plt.subplots(1,2,figsize=(13,5))
    fig_sc.patch.set_facecolor("#0e0e12")
    fig_sc.suptitle("Scenario Comparison — Efficient Frontiers",color="#00e676",fontweight="bold",fontsize=11)
    for ax in axes_sc: apply_chart_style(ax,fig_sc)

    for ax,(scen,col,lbl) in zip(axes_sc,[(sa,"#00e676",sa_label),(sb,"#448aff",sb_label)]):
        ax.scatter(portfolios["Volatility"]*100,portfolios["Return"]*100,
                   c=portfolios["ESG Score"],cmap="RdYlGn",s=10,alpha=0.6,vmin=0,vmax=100,zorder=2)
        ax.scatter(scen["sd"]*100,scen["ret"]*100,s=250,marker="*",color=col,zorder=7,
                   label=f"{lbl}\nW₁={scen['w1']*100:.0f}% | SR={scen['sharpe']:.3f}",
                   ec="white",lw=0.8)
        ax.scatter(float(tan["Volatility"])*100,float(tan["Return"])*100,s=150,marker="*",color="#ff5252",
                   zorder=6,label="Tangency",ec="white",lw=0.5)
        ax.set_xlabel("Risk σ (%)",fontsize=9); ax.set_ylabel("Return (%)",fontsize=9)
        ax.set_title(f"{lbl}  (γ={sa_gamma if col=='#4ade80' else sb_gamma}, λ={sa_lam if col=='#4ade80' else sb_lam})",
                     fontsize=9,fontweight="bold")
        ax.legend(fontsize=8,facecolor="#0e0e12",labelcolor="#c8ccd8",edgecolor="#2a2a3a",framealpha=0.9)
        ax.set_ylim(portfolios["Return"].min()*100*0.95,portfolios["Return"].max()*100*1.15)
        ax.set_xlim(0,portfolios["Volatility"].max()*100*1.1)

    plt.tight_layout(pad=2.0); st.pyplot(fig_sc); plt.close(fig_sc)

    # Difference summary
    st.divider()
    esg_diff  = sb["esg"]   - sa["esg"]
    sr_diff   = sb["sharpe"]- sa["sharpe"]
    ret_diff  = (sb["ret"]  - sa["ret"])*100
    st.markdown(f"""
**Scenario B vs Scenario A:**
- ESG Score: **{'+' if esg_diff>=0 else ''}{esg_diff:.2f}** points
- Sharpe Ratio: **{'+' if sr_diff>=0 else ''}{sr_diff:.4f}**
- Expected Return: **{'+' if ret_diff>=0 else ''}{ret_diff:.2f}%**
""")

# ══════════════════════════════════════════════
# TAB 7 — METHODOLOGY
# ══════════════════════════════════════════════
with tab7:
    st.subheader("Theoretical Framework",divider="green")
    col_m1,col_m2=st.columns(2)
    with col_m1:
        st.markdown("#### Utility Function (ECN316 Lecture 6)")
        st.latex(r"U = E[R_p] - \frac{\gamma}{2}\sigma_p^2 + \lambda\bar{s}")
        st.markdown(f"""
| Symbol | Meaning | Your value |
|---|---|---|
| γ | Absolute risk aversion | {gamma} |
| λ | ESG preference weight | {lam} |
| s̄ | Composite ESG score | Pillar-weighted |
| E weight | Environmental | {w_e:.0%} |
| S weight | Social | {w_s:.0%} |
| G weight | Governance | {w_g:.0%} |
**ESG cost:** {esg_cost:.4f} Sharpe vs tangency portfolio.
        """)
        st.divider()
        stoggle("📐 Why this utility function?",
            f"Standard MV utility penalises risk via variance. We extend with λs̄ (composite ESG). Positive λ = investor values sustainability and accepts lower Sharpe for greener portfolio. When λ=0: pure Markowitz MV optimisation. Your ESG cost: {esg_cost:.4f}.")
        st.space(1)
        stoggle("🚫 Why apply exclusions before optimisation?",
            "Sin stock screening applied BEFORE the utility function — not as an in-model constraint. Follows Sustainable Finance 1.0 (Schoenmaker 2017): remove worst-offending companies, then optimise within remaining universe. Hard screen, not a preference weight.")
        st.space(1)
        stoggle("📊 How is the ESG-efficient frontier built?",
            "Tutorial 5.2 approach: define portfolio_ret(w), portfolio_sd(w), av_esg(w) as functions of w₁. Loop over 1,000 weights ∈[0,1]. Utility = E[R]−(γ/2)σ²+λs̄. ESG-optimal = w₁ maximising U. Tangency = w₁ maximising Sharpe.")
        st.space(1)
        stoggle("🚨 What is greenwashing and how do we detect it?",
            "Greenwashing (Berg et al. RF 2022) occurs when companies claim strong ESG credentials but the evidence is mixed across pillars. Our flag triggers when a company scores highly on Environmental (E≥65) but weakly on Social and Governance (avg S+G <40). This pattern — strong environmental PR, weak internal governance — is a classic greenwashing signal identified in the ESG rating disagreement literature.")
        st.space(1)
        stoggle("🌍 What are the Schoenmaker SF Framework tiers?",
            "Schoenmaker (2017) defines three tiers: SF 1.0 — profit maximisation while avoiding sin stocks (hard exclusion screening). SF 2.0 — stakeholder value, where ESG is integrated into the objective function (our λ term). SF 3.0 — common good, where sustainability and environmental impact are prioritised above financial returns. Ethical Edge supports all three tiers depending on your λ and exclusion settings.")
        st.space(1)
        stoggle("🏭 What are Scope 1, 2 and 3 emissions?",
            "The GHG Protocol defines three scopes: Scope 1 = direct emissions from owned or controlled sources (e.g. company vehicles, boilers). Scope 2 = indirect emissions from purchased electricity or heat. Scope 3 = all other indirect emissions in the value chain, including suppliers and product use. Scope 3 is typically the largest and hardest to measure. Bolton & Kacperczyk (JFE 2021) show that carbon emissions — especially Scope 1 — are priced in stock returns, meaning high-emission firms face a carbon risk premium.")
    with col_m2:
        st.markdown("#### What makes Ethical Edge unique?")
        st.markdown("""
- **Traffic light ESG rating** — 🟢/🟡/🔴 visual rating (eToro-inspired)
- **Greenwashing flag** — detects high E + weak S/G (Berg et al. 2022)
- **SF Framework badge** — classifies approach as SF 1.0/2.0/3.0 (Schoenmaker 2017)
- **Scope 1/2/3 carbon breakdown** — GHG Protocol-aligned emissions chart
- **Carbon score** — proxy environmental footprint of your portfolio
- **Investment calculator** — £X over Y years projected growth chart
- **Downloadable report** — full portfolio health report (.txt)
- **Scenario comparator** — side-by-side A vs B with live frontier charts
- **ESG pillar leaderboard** — which asset wins on E, S, G separately
- **Portfolio pie chart** — weight visualisation for retail investors
- **Sustainability story card** — plain-English narrative of your impact
- **Investor persona selector** — 5 pre-built profiles
- **Named assets** — real company names throughout
- **λ Sensitivity table** — shows trade-off across preference levels
- **Interactive explorers** — γ, λ, ρ, utility heatmap
- **AI Explainer** — Claude-powered Q&A pre-loaded with your portfolio
        """)
        st.divider()
        st.markdown("#### Academic References")
        mention(label="Flammer (JFE 2021) — Corporate Green Bonds",icon="📄",url="https://doi.org/10.1016/j.jfineco.2021.01.010")
        mention(label="Bolton & Kacperczyk (JFE 2021) — Carbon & Returns",icon="📄",url="https://doi.org/10.1016/j.jfineco.2021.05.008")
        mention(label="Schoenmaker (2017) — Sustainable Finance",icon="📄",url="https://ssrn.com/abstract=3066210")
        mention(label="Berg et al. (RF 2022) — ESG Rating Disagreement",icon="📄",url="https://doi.org/10.1093/rof/rfac033")

st.divider()
st.caption("⚖️ Ethical Edge  ·  ECN316 Sustainable Finance  ·  Queen Mary University of London")
