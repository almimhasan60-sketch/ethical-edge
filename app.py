"""
Ethical Edge – Sustainable Portfolio Optimiser
ECN316 Sustainable Finance | QMUL Group Project

Built on Tutorial 5.2 structure.
Utility: U = E[Rp] - (γ/2)·σ²p + λ·s̄
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("⚖️ Ethical Edge")
st.caption("Sustainable Portfolio Optimiser  ·  ECN316 Sustainable Finance  ·  QMUL")
st.latex(r"U = E[R_p] - \frac{\gamma}{2}\sigma_p^2 + \lambda\bar{s}")
st.divider()

# ─────────────────────────────────────────────
# SIDEBAR INPUTS
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Investor Profile")

    # Step 1 — Risk attitude quiz
    st.subheader("Step 1 — Risk Attitude", divider="green")
    st.caption("Two questions to estimate your risk aversion γ.")

    q1 = st.radio(
        "If your portfolio dropped 20%, you'd…",
        ["Sell immediately — I can't handle losses",
         "Hold steady and wait it out",
         "Buy more — it's a buying opportunity"],
        index=1,
    )
    q2 = st.radio(
        "Your investment horizon?",
        ["Under 2 years", "2–5 years", "5+ years"],
        index=1,
    )

    risk_map = {
        ("Sell immediately — I can't handle losses", "Under 2 years"):    9.0,
        ("Sell immediately — I can't handle losses", "2–5 years"):        8.0,
        ("Sell immediately — I can't handle losses", "5+ years"):         6.0,
        ("Hold steady and wait it out",              "Under 2 years"):    6.0,
        ("Hold steady and wait it out",              "2–5 years"):        4.0,
        ("Hold steady and wait it out",              "5+ years"):         3.0,
        ("Buy more — it's a buying opportunity",     "Under 2 years"):    3.0,
        ("Buy more — it's a buying opportunity",     "2–5 years"):        2.0,
        ("Buy more — it's a buying opportunity",     "5+ years"):         1.0,
    }
    gamma_auto = risk_map[(q1, q2)]
    st.info(f"Suggested risk aversion: **γ = {gamma_auto}**")
    gamma = st.slider("Fine-tune γ  (higher = more risk-averse)",
                      min_value=0.5, max_value=10.0,
                      value=gamma_auto, step=0.5)

    # Step 2 — ESG preference
    st.subheader("Step 2 — ESG Commitment", divider="green")
    esg_label = st.select_slider(
        "How important is sustainability to you?",
        options=["None (λ=0)", "Low (λ=0.5)", "Medium (λ=1)",
                 "High (λ=2)", "Max (λ=4)"],
        value="Medium (λ=1)",
    )
    lam_map = {"None (λ=0)": 0.0, "Low (λ=0.5)": 0.5, "Medium (λ=1)": 1.0,
               "High (λ=2)": 2.0, "Max (λ=4)": 4.0}
    lam = st.slider("Fine-tune λ  (ESG preference weight)",
                    min_value=0.0, max_value=5.0,
                    value=lam_map[esg_label], step=0.25)

    # Step 3 — ESG pillar weights
    st.subheader("Step 3 — ESG Pillar Weights", divider="green")
    st.caption("How to combine E, S and G into one score.")
    w_e = st.slider("🌍 Environmental (E)", 0.0, 1.0, 0.4, 0.05)
    w_s = st.slider("🤝 Social (S)",         0.0, 1.0, 0.3, 0.05)
    w_g = st.slider("🏛️ Governance (G)",      0.0, 1.0, 0.3, 0.05)
    pt  = w_e + w_s + w_g or 1.0
    w_e, w_s, w_g = w_e/pt, w_s/pt, w_g/pt
    st.caption(f"Normalised → E: {w_e:.0%}  ·  S: {w_s:.0%}  ·  G: {w_g:.0%}")

    # Step 4 — Exclusions
    st.subheader("Step 4 — Exclusions", divider="green")
    st.caption("Sin stock screening (Sustainable Finance 1.0).")
    excl_tobacco  = st.checkbox("🚬 Tobacco")
    excl_weapons  = st.checkbox("⚔️ Weapons / Defence")
    excl_fossil   = st.checkbox("🛢️ Fossil Fuels")
    excl_gambling = st.checkbox("🎰 Gambling")
    min_esg_score = st.slider("Minimum ESG score", 0, 100, 0, 5)

    # Step 5 — Asset data
    st.subheader("Step 5 — Asset Data", divider="green")

    st.markdown("**Asset 1**")
    c1, c2 = st.columns(2)
    r1  = c1.number_input("E[R] (%)",  value=10.0, step=0.5, key="r1") / 100
    sd1 = c2.number_input("σ (%)",     value=15.0, step=0.5, key="sd1") / 100
    d1, d2, d3 = st.columns(3)
    e1 = d1.number_input("E score", value=70.0, step=1.0,
                         min_value=0.0, max_value=100.0, key="e1")
    s1 = d2.number_input("S score", value=60.0, step=1.0,
                         min_value=0.0, max_value=100.0, key="s1")
    g1 = d3.number_input("G score", value=80.0, step=1.0,
                         min_value=0.0, max_value=100.0, key="g1")
    sin1 = st.multiselect("Sector flags",
                          ["Tobacco","Weapons","Fossil Fuels","Gambling"], key="sin1")

    st.markdown("**Asset 2**")
    c3, c4 = st.columns(2)
    r2  = c3.number_input("E[R] (%)",  value=7.0,  step=0.5, key="r2") / 100
    sd2 = c4.number_input("σ (%)",     value=22.0, step=0.5, key="sd2") / 100
    d4, d5, d6 = st.columns(3)
    e2 = d4.number_input("E score", value=30.0, step=1.0,
                         min_value=0.0, max_value=100.0, key="e2")
    s2 = d5.number_input("S score", value=40.0, step=1.0,
                         min_value=0.0, max_value=100.0, key="s2")
    g2 = d6.number_input("G score", value=20.0, step=1.0,
                         min_value=0.0, max_value=100.0, key="g2")
    sin2 = st.multiselect("Sector flags",
                          ["Tobacco","Weapons","Fossil Fuels","Gambling"], key="sin2")

    st.markdown("**Market**")
    cm1, cm2 = st.columns(2)
    rho    = cm1.number_input("Correlation ρ", value=0.3,
                              step=0.05, min_value=-1.0, max_value=1.0)
    r_free = cm2.number_input("Risk-free rate (%)", value=2.5, step=0.25) / 100

# ─────────────────────────────────────────────
# COMPOSITE ESG SCORES  (pillar-weighted)
# ─────────────────────────────────────────────
esg1 = w_e*e1 + w_s*s1 + w_g*g1
esg2 = w_e*e2 + w_s*s2 + w_g*g2

# ─────────────────────────────────────────────
# EXCLUSION SCREENING  (SF 1.0 — applied first)
# ─────────────────────────────────────────────
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
    st.error("⛔ Both assets are excluded. Please relax your exclusion filters.")
    st.stop()
if ex1:
    st.warning("⚠️ Asset 1 excluded — portfolio is 100% Asset 2.")
if ex2:
    st.warning("⚠️ Asset 2 excluded — portfolio is 100% Asset 1.")

weights = (np.array([0.0]) if ex1
           else np.array([1.0]) if ex2
           else np.linspace(0, 1, 1000))

# ─────────────────────────────────────────────
# STEP 3 — FUNCTIONS  (Tutorial 5.2 structure)
# ─────────────────────────────────────────────
def portfolio_ret(w1):
    return w1*r1 + (1-w1)*r2

def portfolio_sd(w1):
    return np.sqrt(
        w1**2 * sd1**2
        + (1-w1)**2 * sd2**2
        + 2 * rho * w1 * (1-w1) * sd1 * sd2
    )

def av_esg(w1):
    return w1*esg1 + (1-w1)*esg2

def p_sharpe(w1):
    sd = portfolio_sd(w1)
    return (portfolio_ret(w1) - r_free) / sd if sd > 0 else -np.inf

def p_utility(w1):
    return portfolio_ret(w1) - (gamma/2)*portfolio_sd(w1)**2 + lam*av_esg(w1)

# ─────────────────────────────────────────────
# STEP 4 — BUILD PORTFOLIOS  (Tutorial 5.2 structure)
# ─────────────────────────────────────────────
p_ret     = []
p_vol     = []
p_esg     = []
p_sh      = []
p_util    = []
p_weights = []

for w in weights:
    p_ret.append(portfolio_ret(w))
    p_vol.append(portfolio_sd(w))
    p_esg.append(av_esg(w))
    p_sh.append(p_sharpe(w))
    p_util.append(p_utility(w))
    p_weights.append(w)

portfolios = pd.DataFrame({
    "Weight Asset 1": p_weights,
    "Weight Asset 2": [1-w for w in p_weights],
    "Return":         p_ret,
    "Volatility":     p_vol,
    "ESG Score":      p_esg,
    "Sharpe Ratio":   p_sh,
    "Utility":        p_util,
})

# ─────────────────────────────────────────────
# FIND KEY PORTFOLIOS
# ─────────────────────────────────────────────
# ESG-Optimal: maximise ESG utility
idx_opt = portfolios["Utility"].idxmax()
opt      = portfolios.loc[idx_opt]

# MV Optimal (λ=0): maximise standard MV utility
portfolios["MV_Utility"] = portfolios["Return"] - (gamma/2)*portfolios["Volatility"]**2
idx_mv  = portfolios["MV_Utility"].idxmax()
mv      = portfolios.loc[idx_mv]

# Tangency: maximise Sharpe ratio
idx_tan = portfolios["Sharpe Ratio"].idxmax()
tan     = portfolios.loc[idx_tan]

# Min variance
idx_mvp = portfolios["Volatility"].idxmin()
mvp     = portfolios.loc[idx_mvp]

esg_cost     = float(tan["Sharpe Ratio"] - opt["Sharpe Ratio"])
esg_cost_pct = min(abs(esg_cost) / max(abs(tan["Sharpe Ratio"]), 0.001) * 100, 100)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Results", "📈 Charts", "💡 Insights", "ℹ️ Methodology"]
)

# ════════════════════════════════════════════
# TAB 1 — RESULTS
# ════════════════════════════════════════════
with tab1:

    st.subheader("Asset ESG Scores", divider="green")
    a1col, a2col = st.columns(2)

    def esg_badge(score):
        if score >= 65: return "🟢 Strong"
        if score >= 40: return "🟡 Moderate"
        return "🔴 Weak"

    a1col.metric(
        label=f"Asset 1  —  ESG {esg_badge(esg1)}",
        value=f"{esg1:.1f} / 100",
        help=f"E:{e1:.0f}  S:{s1:.0f}  G:{g1:.0f}  |  weights E:{w_e:.0%} S:{w_s:.0%} G:{w_g:.0%}"
    )
    a2col.metric(
        label=f"Asset 2  —  ESG {esg_badge(esg2)}",
        value=f"{esg2:.1f} / 100",
        help=f"E:{e2:.0f}  S:{s2:.0f}  G:{g2:.0f}  |  weights E:{w_e:.0%} S:{w_s:.0%} G:{w_g:.0%}"
    )
    style_metric_cards(
        background_color="#0d1f10",
        border_left_color="#4ade80",
        border_color="#1a3a1a",
        box_shadow=True,
    )

    st.divider()
    st.subheader("Your Recommended Portfolio", divider="green")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Asset 1 Weight",  f"{opt['Weight Asset 1']*100:.1f}%")
    m2.metric("Asset 2 Weight",  f"{opt['Weight Asset 2']*100:.1f}%")
    m3.metric("Expected Return", f"{opt['Return']*100:.2f}%")
    m4.metric("Risk  σ",         f"{opt['Volatility']*100:.2f}%")
    m5.metric("ESG Score",       f"{opt['ESG Score']:.1f}")
    m6.metric("Sharpe Ratio",    f"{opt['Sharpe Ratio']:.3f}")
    style_metric_cards(
        background_color="#0d1f10",
        border_left_color="#4ade80",
        border_color="#1a3a1a",
        box_shadow=True,
    )

    if lam > 0 and esg_cost > 0.001:
        st.success(
            f"🌱 Your ESG preference (λ = {lam}) shifts your portfolio toward greener assets. "
            f"**ESG cost = {esg_cost:.4f}** drop in Sharpe ratio vs the max-Sharpe portfolio."
        )
    elif lam == 0:
        st.info("ℹ️ λ = 0 — ESG plays no role. ESG Optimal and MV Optimal are identical.")

    st.divider()
    st.subheader("Portfolio Comparison Table", divider="green")

    comp_df = pd.DataFrame({
        "Portfolio": [
            "MV Optimal (λ=0)",
            "✅ ESG Optimal (you)",
            "Tangency — Max Sharpe",
            "Min Variance",
        ],
        "Asset 1 (%)": [
            f"{mv['Weight Asset 1']*100:.1f}",
            f"{opt['Weight Asset 1']*100:.1f}",
            f"{tan['Weight Asset 1']*100:.1f}",
            f"{mvp['Weight Asset 1']*100:.1f}",
        ],
        "Asset 2 (%)": [
            f"{mv['Weight Asset 2']*100:.1f}",
            f"{opt['Weight Asset 2']*100:.1f}",
            f"{tan['Weight Asset 2']*100:.1f}",
            f"{mvp['Weight Asset 2']*100:.1f}",
        ],
        "E[Rp]": [
            f"{mv['Return']*100:.2f}%",
            f"{opt['Return']*100:.2f}%",
            f"{tan['Return']*100:.2f}%",
            f"{mvp['Return']*100:.2f}%",
        ],
        "σp": [
            f"{mv['Volatility']*100:.2f}%",
            f"{opt['Volatility']*100:.2f}%",
            f"{tan['Volatility']*100:.2f}%",
            f"{mvp['Volatility']*100:.2f}%",
        ],
        "ESG": [
            f"{mv['ESG Score']:.2f}",
            f"{opt['ESG Score']:.2f}",
            f"{tan['ESG Score']:.2f}",
            f"{mvp['ESG Score']:.2f}",
        ],
        "Sharpe": [
            f"{mv['Sharpe Ratio']:.3f}",
            f"{opt['Sharpe Ratio']:.3f}",
            f"{tan['Sharpe Ratio']:.3f}",
            "—",
        ],
    })
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

# ════════════════════════════════════════════
# TAB 2 — CHARTS  (matplotlib, Tutorial 5.2 style extended)
# ════════════════════════════════════════════
with tab2:

    st.subheader("ESG-Efficient Frontier", divider="green")
    st.caption("Efficient frontier coloured by ESG score · CML · key portfolios marked")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("#060a06")

    for ax in axes:
        ax.set_facecolor("#0d1f10")
        ax.tick_params(colors="#b8debb", labelsize=9)
        ax.xaxis.label.set_color("#b8debb")
        ax.yaxis.label.set_color("#b8debb")
        ax.title.set_color("#4ade80")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1a3a1a")
        ax.grid(True, color="#142814", linewidth=0.6)

    # ── LEFT: efficient frontier coloured by ESG ─────
    ax = axes[0]
    sc = ax.scatter(
        portfolios["Volatility"]*100,
        portfolios["Return"]*100,
        c=portfolios["ESG Score"],
        cmap="RdYlGn", s=12, alpha=0.85,
        vmin=0, vmax=100, zorder=2,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Portfolio ESG Score", color="#b8debb", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#b8debb")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#b8debb", fontsize=8)
    cbar.outline.set_edgecolor("#1a3a1a")

    # CML — dashed green line from risk-free through tangency
    if tan["Volatility"] > 0:
        sd_range = np.linspace(0, portfolios["Volatility"].max()*1.25, 200)
        cml = r_free + (tan["Return"] - r_free) / tan["Volatility"] * sd_range
        ax.plot(sd_range*100, cml*100, "--",
                color="#60a5fa", linewidth=1.5, label="CML", alpha=0.85, zorder=3)

    # Indifference curves
    sc2  = np.linspace(0.001, portfolios["Volatility"].max()*1.25, 200)
    U_mv = float(mv["Return"]) - (gamma/2)*float(mv["Volatility"])**2
    ax.plot(sc2*100, (U_mv + (gamma/2)*sc2**2)*100,
            ":", color="#60a5fa", linewidth=1.5, label="MV Indiff. Curve")
    U_esg = float(opt["Utility"])
    ax.plot(sc2*100, ((U_esg - lam*float(opt["ESG Score"])) + (gamma/2)*sc2**2)*100,
            "-.", color="#f59e0b", linewidth=1.5, label="ESG Indiff. Curve")

    # Key points
    ax.scatter(0, r_free*100,
               s=90, marker="s", color="white", zorder=6,
               label="Risk-Free", edgecolors="#555", linewidths=0.8)
    ax.scatter(mvp["Volatility"]*100, mvp["Return"]*100,
               s=100, marker="D", color="#22d3ee", zorder=6,
               label="Min Variance", edgecolors="white", linewidths=0.5)
    ax.scatter(tan["Volatility"]*100, tan["Return"]*100,
               s=200, marker="*", color="#ef4444", zorder=7,
               label="Tangency (Max Sharpe)", edgecolors="white", linewidths=0.5)
    ax.scatter(mv["Volatility"]*100, mv["Return"]*100,
               s=140, marker="^", color="#60a5fa", zorder=6,
               label="MV Optimal (λ=0)", edgecolors="white", linewidths=0.5)
    ax.scatter(opt["Volatility"]*100, opt["Return"]*100,
               s=200, marker="*", color="#4ade80", zorder=7,
               label=f"ESG Optimal (ESG={opt['ESG Score']:.1f})",
               edgecolors="white", linewidths=0.5)

    ax.set_xlabel("Risk — Standard Deviation (%)", fontsize=10)
    ax.set_ylabel("Expected Return (%)", fontsize=10)
    ax.set_title("Mean–Variance Space\n(colour = ESG score)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left",
              facecolor="#0d1f10", labelcolor="#b8debb",
              framealpha=0.9, edgecolor="#1a3a1a")
    y_lo = min(r_free - 0.005, portfolios["Return"].min() - 0.005)
    ax.set_ylim(y_lo*100, portfolios["Return"].max()*1.18*100)
    ax.set_xlim(-0.3, portfolios["Volatility"].max()*1.12*100)

    # ── RIGHT: ESG–Sharpe frontier ─────────────
    ax2 = axes[1]
    ax2.plot(portfolios["ESG Score"], portfolios["Sharpe Ratio"],
             color="#4ade80", linewidth=2.5, zorder=2)
    ax2.fill_between(portfolios["ESG Score"], portfolios["Sharpe Ratio"],
                     portfolios["Sharpe Ratio"].min(),
                     color="#4ade80", alpha=0.07)
    ax2.scatter(opt["ESG Score"], opt["Sharpe Ratio"],
                s=200, marker="*", color="#4ade80", zorder=5,
                label="Your ESG Optimal", edgecolors="white", linewidths=0.5)
    ax2.scatter(tan["ESG Score"], tan["Sharpe Ratio"],
                s=160, marker="*", color="#ef4444", zorder=5,
                label="Max Sharpe", edgecolors="white", linewidths=0.5)
    ax2.axhline(float(tan["Sharpe Ratio"]),
                color="#ef4444", linestyle="--", linewidth=0.9, alpha=0.5)

    if esg_cost > 0.001:
        ax2.annotate(
            f"ESG cost: −{esg_cost:.3f}",
            xy=(float(opt["ESG Score"]), float(opt["Sharpe Ratio"])),
            xytext=(float(opt["ESG Score"]) + 2, float(opt["Sharpe Ratio"]) - 0.03),
            color="#f59e0b", fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#f59e0b", lw=1.2),
        )

    ax2.set_xlabel("Portfolio ESG Score", fontsize=10)
    ax2.set_ylabel("Sharpe Ratio", fontsize=10)
    ax2.set_title("ESG–Sharpe Frontier\n(sustainability–performance trade-off)",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8, facecolor="#0d1f10",
               labelcolor="#b8debb", framealpha=0.9, edgecolor="#1a3a1a")

    plt.tight_layout(pad=2.0)

    # chart_container wraps the chart and adds a CSV download button
    with chart_container(portfolios[["Weight Asset 1","Return","Volatility",
                                     "ESG Score","Sharpe Ratio","Utility"]],
                         export_formats=["CSV"]):
        st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ════════════════════════════════════════════
# TAB 3 — INSIGHTS
# ════════════════════════════════════════════
with tab3:

    st.subheader("What does this mean for you?", divider="green")

    # Investor type
    if   gamma <= 2: inv_type, inv_desc = "Risk-Seeking",  "Comfortable with large swings in pursuit of higher returns."
    elif gamma <= 5: inv_type, inv_desc = "Balanced",       "Seeks a reasonable return while avoiding excessive risk."
    else:            inv_type, inv_desc = "Risk-Averse",    "Prioritises capital protection over maximising returns."

    if   lam == 0:   esg_type, esg_desc = "ESG-Neutral", "ESG plays no role — pure financial return maximiser."
    elif lam < 1:    esg_type, esg_desc = "Light Green",  "Mild ESG preference — accepts a small Sharpe reduction for greener assets."
    elif lam < 2:    esg_type, esg_desc = "Green",         "Meaningful ESG preference — sustainability influences allocation."
    else:            esg_type, esg_desc = "Deep Green",    "ESG is central — willing to accept a significant financial trade-off."

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Investor Profile", divider="green")
        st.metric("Risk Type", inv_type, help=inv_desc)
        st.metric("ESG Profile", esg_type, help=esg_desc)
        st.caption(inv_desc)
        st.caption(esg_desc)
        style_metric_cards(
            background_color="#0d1f10",
            border_left_color="#4ade80",
            border_color="#1a3a1a",
            box_shadow=True,
        )

        st.divider()
        st.subheader("Sustainability Trade-Off", divider="green")
        cost_label = "low" if esg_cost_pct < 10 else ("moderate" if esg_cost_pct < 25 else "high")
        annotated_text(
            "ESG cost: ",
            (f"−{esg_cost:.4f} Sharpe", cost_label,
             "#166534" if cost_label == "low" else "#854d0e" if cost_label == "moderate" else "#7f1d1d"),
            "  ·  Your ESG score: ",
            (f"{opt['ESG Score']:.1f}", "your portfolio", "#166534"),
            "  vs tangency: ",
            (f"{tan['ESG Score']:.1f}", "max-Sharpe", "#7f1d1d"),
        )

    with col_b:
        st.subheader("Allocation Explained", divider="green")
        higher_esg = "Asset 1" if esg1 > esg2 else "Asset 2"
        higher_ret = "Asset 1" if r1 > r2 else "Asset 2"
        st.info(
            f"Your optimal portfolio holds **{opt['Weight Asset 1']*100:.1f}%** in Asset 1 "
            f"and **{opt['Weight Asset 2']*100:.1f}%** in Asset 2.  \n\n"
            f"{higher_esg} has the higher ESG score — a positive λ tilts allocation toward it.  \n"
            f"{higher_ret} offers the higher expected return.  \n"
            f"Your γ = {gamma} and λ = {lam} together determine the balance."
        )

        st.subheader("Diversification Benefit", divider="green")
        if rho < 0.5:
            st.success(
                f"With ρ = {rho}, your portfolio benefits from diversification. "
                f"The min-variance portfolio has σ = **{mvp['Volatility']*100:.2f}%**, "
                f"below both Asset 1 ({sd1*100:.1f}%) and Asset 2 ({sd2*100:.1f}%)."
            )
        else:
            st.warning(
                f"With ρ = {rho}, diversification benefit is limited due to high positive correlation. "
                f"Min-variance σ = **{mvp['Volatility']*100:.2f}%**."
            )

# ════════════════════════════════════════════
# TAB 4 — METHODOLOGY
# ════════════════════════════════════════════
with tab4:

    st.subheader("Theoretical Framework", divider="green")
    col_m1, col_m2 = st.columns(2)

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

**ESG cost:** {esg_cost:.4f} drop in Sharpe ratio vs the tangency portfolio.
        """)

        st.divider()

        # stoggle — expandable concept cards
        stoggle(
            "📐 Why this utility function?",
            f"""The standard mean-variance utility penalises risk via the variance term.
We extend it with λs̄, where s̄ is the portfolio's composite ESG score.
A positive λ means the investor directly values sustainability and accepts a
lower Sharpe ratio in exchange for a greener portfolio.
When λ = 0, the model reduces to pure MV optimisation (Markowitz, 1952).
Your current ESG cost is {esg_cost:.4f} in Sharpe ratio — the price of your green preferences."""
        )
        st.space(1)
        stoggle(
            "🚫 Why apply exclusions before optimisation?",
            """Sin stock screening (tobacco, weapons, fossil fuels, gambling) is applied
BEFORE the utility function is evaluated — not as an in-model constraint.
This follows Sustainable Finance 1.0 (Schoenmaker, 2017): avoid the
worst-offending companies outright, then optimise within the remaining
investable universe. It is a hard screen, not a preference weight."""
        )
        st.space(1)
        stoggle(
            "📊 How is the ESG-efficient frontier built?",
            """We follow the Tutorial 5.2 approach:
1. Define portfolio_ret(w), portfolio_sd(w), av_esg(w), p_sharpe(w) as functions of w₁.
2. Loop over 1,000 weights w₁ ∈ [0,1] to build the full frontier.
3. Extend each portfolio's utility to U = E[R] − (γ/2)σ² + λs̄.
4. The ESG-optimal portfolio is the weight w₁ that maximises U.
5. The tangency portfolio maximises the Sharpe ratio independently of preferences."""
        )

    with col_m2:
        st.markdown("#### What makes Ethical Edge unique?")
        st.markdown("""
- **Investor profile quiz** — γ derived from behavioural questions
- **Pillar-weighted ESG** — separate E, S, G inputs with user-defined weights  
- **Sin stock exclusion** — SF 1.0 hard screening before optimisation
- **ESG–Sharpe frontier** — visualises the sustainability–performance trade-off
- **Annotated ESG cost** — shows exactly what you sacrifice for being green
- **Investor type classifier** — plain-English risk and ESG profile
        """)

        st.divider()
        st.markdown("#### Academic References")
        mention(label="Flammer (JFE 2021) — Corporate Green Bonds",
                icon="📄", url="https://doi.org/10.1016/j.jfineco.2021.01.010")
        mention(label="Bolton & Kacperczyk (JFE 2021) — Carbon & Returns",
                icon="📄", url="https://doi.org/10.1016/j.jfineco.2021.05.008")
        mention(label="Schoenmaker (2017) — Sustainable Finance",
                icon="📄", url="https://ssrn.com/abstract=3066210")
        mention(label="Berg et al. (RF 2022) — ESG Rating Disagreement",
                icon="📄", url="https://doi.org/10.1093/rof/rfac033")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.caption("⚖️ Ethical Edge  ·  ECN316 Sustainable Finance  ·  Queen Mary University of London")
