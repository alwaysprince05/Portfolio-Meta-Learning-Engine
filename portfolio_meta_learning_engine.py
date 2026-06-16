import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import norm

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Meta-Learning Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL CSS — Premium Dark Quant Aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"], .stApp {
    background-color: #0D0F14 !important;
    color: #C9D1D9 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D0F14 0%, #111520 100%) !important;
    border-right: 1px solid #1E2D40 !important;
}
section[data-testid="stSidebar"] * { color: #C9D1D9 !important; }

/* Hide default header */
header[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #0D1117 0%, #0D2137 40%, #0D1117 100%);
    border: 1px solid #1E3A5F;
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(0,255,198,0.06) 0%, transparent 60%),
                radial-gradient(circle at 80% 20%, rgba(88,166,255,0.05) 0%, transparent 50%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00FFC6, #58A6FF, #FF79C6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 1rem;
    color: #6B7DA8;
    font-weight: 400;
    margin: 0;
}

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #58A6FF;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    border-left: 3px solid #00FFC6;
    padding-left: 12px;
    margin: 32px 0 16px 0;
}

/* KPI Cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 24px;
}
.kpi-card {
    background: linear-gradient(145deg, #111520, #0D1117);
    border: 1px solid #1E2D40;
    border-radius: 12px;
    padding: 20px 20px 16px 20px;
    position: relative;
    overflow: hidden;
}
.kpi-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.kpi-card.green::after  { background: linear-gradient(90deg, #00FFC6, transparent); }
.kpi-card.blue::after   { background: linear-gradient(90deg, #58A6FF, transparent); }
.kpi-card.purple::after { background: linear-gradient(90deg, #FF79C6, transparent); }
.kpi-card.orange::after { background: linear-gradient(90deg, #FFAA00, transparent); }

.kpi-label {
    font-size: 0.72rem;
    color: #6B7DA8;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
    margin-bottom: 8px;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.65rem;
    font-weight: 600;
    line-height: 1;
}
.kpi-value.green  { color: #00FFC6; }
.kpi-value.red    { color: #FF5555; }
.kpi-value.blue   { color: #58A6FF; }
.kpi-value.purple { color: #FF79C6; }
.kpi-value.orange { color: #FFAA00; }

/* Regime pill badges */
.regime-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.5px;
    margin: 0 4px 8px 0;
}
.regime-badge.bull  { background: rgba(0,255,198,0.12); color: #00FFC6; border: 1px solid #00FFC6; }
.regime-badge.bear  { background: rgba(255,85,85,0.12);  color: #FF5555; border: 1px solid #FF5555; }
.regime-badge.trend { background: rgba(88,166,255,0.12); color: #58A6FF; border: 1px solid #58A6FF; }

/* Chart containers */
.chart-card {
    background: #0D1117;
    border: 1px solid #1E2D40;
    border-radius: 12px;
    padding: 4px;
    margin-bottom: 16px;
}

/* Strategy timeline label */
.strat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #00FFC6;
    background: rgba(0,255,198,0.08);
    border: 1px solid rgba(0,255,198,0.2);
    border-radius: 8px;
    padding: 6px 14px;
    display: inline-block;
    margin-right: 8px;
}

/* Sidebar styling */
.sidebar-section {
    background: rgba(30,45,64,0.3);
    border: 1px solid #1E2D40;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 14px;
}

/* Divider */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1E3A5F, transparent);
    margin: 28px 0;
}

/* Footer */
.footer {
    text-align: center;
    color: #3A4560;
    font-size: 0.78rem;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #1A2035;
}

/* Override Streamlit metric */
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #00FFC6 !important;
}
[data-testid="stMetricLabel"] { color: #6B7DA8 !important; }

/* Slider accent */
.stSlider > div > div > div > div { background: #00FFC6 !important; }

/* Multiselect tag */
.stMultiSelect span[data-baseweb="tag"] {
    background: rgba(0,255,198,0.15) !important;
    border: 1px solid #00FFC6 !important;
}

/* Info/error boxes */
.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
        <div style='font-size:2rem;'>⚡</div>
        <div style='font-size:1.1rem; font-weight:700; color:#00FFC6; letter-spacing:1px;'>META-LEARNING</div>
        <div style='font-size:0.72rem; color:#3A4560; letter-spacing:2px; text-transform:uppercase;'>Engine Controls</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.72rem; color:#6B7DA8; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;'>Strategy Universe</div>", unsafe_allow_html=True)
    strategy_set = st.multiselect(
        label="",
        options=["Mean-Variance", "Risk Parity", "Minimum Variance", "Momentum Tilt", "Defensive Allocation"],
        default=["Mean-Variance", "Risk Parity", "Momentum Tilt"],
        label_visibility="collapsed"
    )
    if not strategy_set:
        strategy_set = ["Mean-Variance"]

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.72rem; color:#6B7DA8; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;'>Lookback Window (days)</div>", unsafe_allow_html=True)
    lookback = st.slider("", 30, 120, 60, 10, key="lookback", label_visibility="collapsed")

    st.markdown("<div style='font-size:0.72rem; color:#6B7DA8; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px; margin-top:12px;'>Risk Aversion λ</div>", unsafe_allow_html=True)
    risk_aversion = st.slider("", 0.1, 5.0, 2.0, 0.1, key="risk_av", label_visibility="collapsed")

    st.markdown("<div style='font-size:0.72rem; color:#6B7DA8; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px; margin-top:12px;'>Regime Sensitivity</div>", unsafe_allow_html=True)
    regime_sensitivity = st.slider("", 0.1, 2.0, 1.0, 0.1, key="regime_s", label_visibility="collapsed")

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:rgba(0,255,198,0.06); border:1px solid rgba(0,255,198,0.2); border-radius:8px; padding:12px; font-size:0.78rem;'>
        <span style='color:#00FFC6; font-weight:600;'>● DEMO MODE</span><br>
        <span style='color:#6B7DA8;'>Live data: SPY · TLT · GLD</span><br>
        <span style='color:#6B7DA8;'>Period: 2023 full year</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:24px; font-size:0.72rem; color:#3A4560; text-align:center;'>
        Built by <span style='color:#58A6FF;'>Prince Maurya</span><br>
        <span style='color:#1E2D40;'>github.com/alwaysprince05</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# BACKEND FUNCTIONS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    data = yf.download(["SPY", "TLT", "GLD"], start="2023-01-01", end="2023-12-31", progress=False)
    df = data["Close"].dropna(how="all")
    return df

def compute_features(df, lookback):
    returns = np.log(df / df.shift(1)).dropna()
    vol     = returns.rolling(lookback).std() * np.sqrt(252)
    corr    = returns.rolling(lookback).corr().dropna()
    trend   = returns.rolling(lookback).mean()
    return returns, vol, corr, trend

def detect_regimes(vol, corr, trend, regime_sensitivity):
    scaler     = StandardScaler()
    vol_scaled = scaler.fit_transform(vol.fillna(0))
    gmm        = GaussianMixture(n_components=2, random_state=42)
    vol_regime = gmm.fit_predict(vol_scaled)

    corr_matrix = corr.groupby(corr.index).mean().fillna(0)
    pca         = PCA(n_components=1)
    corr_regime = pca.fit_transform(corr_matrix)
    corr_regime = (corr_regime > np.median(corr_regime)).astype(int)

    trend_strength = (trend.abs().mean(axis=1) > regime_sensitivity * trend.abs().mean().mean()).astype(int)

    n = len(vol.index)
    regime_df = pd.DataFrame({
        "Volatility":   np.array(vol_regime)[-n:],
        "Correlation":  np.array(corr_regime.flatten())[-n:],
        "Trend":        np.array(trend_strength)[-n:]
    }, index=vol.index)
    return regime_df

def mean_variance(returns, risk_aversion):
    mu, cov, n = returns.mean(), returns.cov(), len(returns.columns)
    obj  = lambda w: -np.dot(w, mu) + risk_aversion * np.sqrt(np.dot(w, np.dot(cov, w)))
    res  = minimize(obj, np.ones(n)/n, bounds=[(0,1)]*n,
                    constraints={'type':'eq','fun':lambda w: np.sum(w)-1})
    return res.x

def risk_parity(returns):
    cov, n = returns.cov(), len(returns.columns)
    def obj(w):
        pv = np.dot(w, np.dot(cov, w))
        rc = w * cov.dot(w) / pv
        return np.sum((rc - 1/n)**2)
    res = minimize(obj, np.ones(n)/n, bounds=[(0,1)]*n,
                   constraints={'type':'eq','fun':lambda w: np.sum(w)-1})
    return res.x

def min_variance(returns):
    cov, n = returns.cov(), len(returns.columns)
    res = minimize(lambda w: np.dot(w, np.dot(cov, w)),
                   np.ones(n)/n, bounds=[(0,1)]*n,
                   constraints={'type':'eq','fun':lambda w: np.sum(w)-1})
    return res.x

def momentum_tilt(returns):
    mu  = returns.mean()
    mom = mu / (returns.std() + 1e-9)
    w   = mom / (mom.sum() + 1e-9)
    return np.clip(w, 0, 1)

def defensive_allocation(returns, vol):
    inv_vol = 1 / (vol.iloc[-1] + 1e-6)
    return inv_vol / inv_vol.sum()

def bayesian_strategy_selection(returns, strategies, regime_df, lookback, risk_aversion):
    n = len(returns)
    strategy_timeline, confidence_scores, allocations = [], [], []
    for i in range(lookback, n):
        window     = returns.iloc[i - lookback:i]
        vol_window = window.rolling(lookback).std().iloc[-1]
        scores     = {}
        for strat in strategies:
            if   strat == "Mean-Variance":       w = mean_variance(window, risk_aversion)
            elif strat == "Risk Parity":         w = risk_parity(window)
            elif strat == "Minimum Variance":    w = min_variance(window)
            elif strat == "Momentum Tilt":       w = momentum_tilt(window)
            elif strat == "Defensive Allocation":w = defensive_allocation(window, vol_window)
            else:                                w = np.ones(window.shape[1]) / window.shape[1]
            pr = np.dot(w, window.mean())
            pv = np.sqrt(np.dot(w, np.dot(window.cov(), w)))
            scores[strat] = pr / (pv + 1e-9)
        best   = max(scores, key=scores.get)
        vals   = list(scores.values())
        conf   = norm.cdf(scores[best], np.mean(vals), np.std(vals) + 1e-9)
        strategy_timeline.append(best)
        confidence_scores.append(conf)
        allocations.append(w)
    idx = returns.index[lookback:]
    return (pd.Series(strategy_timeline, index=idx),
            pd.Series(confidence_scores, index=idx),
            pd.DataFrame(allocations, index=idx, columns=returns.columns))

def adaptive_rebalance(strategy_timeline, regime_df):
    pts = [0]
    for i in range(1, len(strategy_timeline)):
        if (strategy_timeline.iloc[i] != strategy_timeline.iloc[i-1] or
                not regime_df.iloc[i].equals(regime_df.iloc[i-1])):
            pts.append(i)
    return pts

def compute_performance(returns, allocations, rebalance_points):
    last_w    = allocations.iloc[0].values
    port_rets = []
    for i in range(len(allocations)):
        if i in rebalance_points:
            last_w = allocations.iloc[i].values
        port_rets.append(np.dot(last_w, returns.iloc[i].values))
    port_rets    = pd.Series(port_rets, index=allocations.index)
    equity_curve = (1 + port_rets).cumprod()
    rolling_sh   = port_rets.rolling(20).mean() / (port_rets.rolling(20).std() + 1e-9) * np.sqrt(252)
    drawdown     = equity_curve / equity_curve.cummax() - 1
    return equity_curve, drawdown, rolling_sh, port_rets

# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────
DARK_BG    = "#0D1117"
GRID_COLOR = "#1A2035"
NEON       = "#00FFC6"
BLUE       = "#58A6FF"
PINK       = "#FF79C6"
ORANGE     = "#FFAA00"
RED        = "#FF5555"

def base_layout(title="", height=380):
    return dict(
        title=dict(text=title, font=dict(color=NEON, size=13, family="Inter"), x=0.01),
        template="plotly_dark",
        plot_bgcolor=DARK_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color="#8B9DC3", family="Inter", size=11),
        height=height,
        margin=dict(l=12, r=12, t=40, b=12),
        xaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, showgrid=True),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, showgrid=True),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID_COLOR,
                    borderwidth=1, font=dict(size=10)),
    )

def kpi_card(label, value, color_class="green", prefix="", suffix=""):
    return f"""
    <div class='kpi-card {color_class}'>
        <div class='kpi-label'>{label}</div>
        <div class='kpi-value {color_class}'>{prefix}{value}{suffix}</div>
    </div>"""

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class='hero-banner'>
    <div class='hero-title'>⚡ Portfolio Meta-Learning Engine</div>
    <div class='hero-sub'>
        Institutional-grade adaptive portfolio construction · Bayesian strategy selection ·
        Spectral market decomposition · Real-time regime detection
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
with st.spinner("🔄 Fetching market data & running meta-learning engine..."):
    try:
        df = load_data()
        if df is None or df.empty:
            raise ValueError("No data returned from yfinance.")

        returns, vol, corr, trend = compute_features(df, lookback)
        regime_df = detect_regimes(vol, corr, trend, regime_sensitivity)
        strategy_timeline, confidence_scores, allocations = bayesian_strategy_selection(
            returns, strategy_set, regime_df, lookback, risk_aversion
        )
        rebalance_pts = adaptive_rebalance(strategy_timeline, regime_df)
        equity_curve, drawdown, rolling_sh, port_rets = compute_performance(
            returns.iloc[lookback:], allocations, rebalance_pts
        )

        # ── KPI Row ──────────────────────────────────────────────────────────
        total_return  = equity_curve.iloc[-1] - 1
        max_dd        = drawdown.min()
        ann_sharpe    = port_rets.mean() / (port_rets.std() + 1e-9) * np.sqrt(252)
        ann_vol       = port_rets.std() * np.sqrt(252)
        n_rebalances  = len(rebalance_pts)
        current_strat = strategy_timeline.iloc[-1]
        avg_conf      = confidence_scores.mean()

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            ret_color = "green" if total_return > 0 else "red"
            st.markdown(f"""
            <div class='kpi-card {ret_color}'>
                <div class='kpi-label'>Total Return</div>
                <div class='kpi-value {ret_color}'>{'+' if total_return>0 else ''}{total_return*100:.2f}%</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class='kpi-card blue'>
                <div class='kpi-label'>Ann. Sharpe Ratio</div>
                <div class='kpi-value blue'>{ann_sharpe:.3f}</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class='kpi-card purple'>
                <div class='kpi-label'>Max Drawdown</div>
                <div class='kpi-value purple'>{max_dd*100:.2f}%</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class='kpi-card orange'>
                <div class='kpi-label'>Ann. Volatility</div>
                <div class='kpi-value orange'>{ann_vol*100:.2f}%</div>
            </div>""", unsafe_allow_html=True)

        # ── Regime + Strategy Info Bar ────────────────────────────────────────
        last_regime = regime_df.iloc[-1]
        vol_state   = "HIGH VOL" if last_regime["Volatility"] == 1 else "LOW VOL"
        corr_state  = "HIGH CORR" if last_regime["Correlation"] == 1 else "LOW CORR"
        trend_state = "TRENDING" if last_regime["Trend"] == 1 else "MEAN-REV"

        col_r, col_s = st.columns([3, 2])
        with col_r:
            st.markdown(f"""
            <div style='background:#0D1117; border:1px solid #1E2D40; border-radius:10px; padding:14px 18px; margin-bottom:16px;'>
                <div style='font-size:0.7rem; color:#6B7DA8; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;'>Current Market Regime</div>
                <span class='regime-badge bull'>{vol_state}</span>
                <span class='regime-badge trend'>{corr_state}</span>
                <span class='regime-badge {"bull" if trend_state=="TRENDING" else "bear"}'>{trend_state}</span>
                <span style='margin-left:12px; font-size:0.8rem; color:#6B7DA8;'>
                    · {len(rebalance_pts)} rebalance events
                </span>
            </div>
            """, unsafe_allow_html=True)
        with col_s:
            st.markdown(f"""
            <div style='background:#0D1117; border:1px solid #1E2D40; border-radius:10px; padding:14px 18px; margin-bottom:16px;'>
                <div style='font-size:0.7rem; color:#6B7DA8; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;'>Active Strategy</div>
                <span class='strat-label'>{current_strat}</span>
                <span style='font-size:0.78rem; color:#6B7DA8;'>conf: {avg_conf:.1%}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

        # ── ROW 1: Equity Curve + Drawdown ───────────────────────────────────
        st.markdown("<div class='section-header'>📈 Performance Analytics</div>", unsafe_allow_html=True)
        col_eq, col_dd = st.columns(2)

        with col_eq:
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=equity_curve.index, y=equity_curve.values,
                mode="lines", name="Equity Curve",
                line=dict(color=NEON, width=2.5),
                fill="tozeroy",
                fillcolor="rgba(0,255,198,0.06)"
            ))
            for rp in rebalance_pts[::max(1, len(rebalance_pts)//8)]:
                if rp < len(equity_curve):
                    fig_eq.add_vline(
                        x=equity_curve.index[rp],
                        line_dash="dot", line_color="rgba(88,166,255,0.35)", line_width=1
                    )
            layout = base_layout("Equity Curve (Cumulative)", height=320)
            layout["yaxis"]["tickformat"] = ".3f"
            fig_eq.update_layout(**layout)
            st.plotly_chart(fig_eq, use_container_width=True)

        with col_dd:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=drawdown.index, y=drawdown.values * 100,
                mode="lines", name="Drawdown",
                line=dict(color=RED, width=2),
                fill="tozeroy",
                fillcolor="rgba(255,85,85,0.08)"
            ))
            layout_dd = base_layout("Drawdown (%)", height=320)
            layout_dd["yaxis"]["ticksuffix"] = "%"
            fig_dd.update_layout(**layout_dd)
            st.plotly_chart(fig_dd, use_container_width=True)

        # ── ROW 2: Rolling Sharpe + Strategy Timeline ─────────────────────────
        col_sh, col_st = st.columns(2)

        with col_sh:
            fig_sh = go.Figure()
            fig_sh.add_hline(y=0, line_color=GRID_COLOR, line_width=1)
            fig_sh.add_hline(y=1, line_color="rgba(0,255,198,0.2)", line_dash="dash", line_width=1)
            fig_sh.add_trace(go.Scatter(
                x=rolling_sh.index, y=rolling_sh.values,
                mode="lines", name="Rolling Sharpe (20d)",
                line=dict(color=BLUE, width=2),
                fill="tozeroy",
                fillcolor="rgba(88,166,255,0.06)"
            ))
            layout_sh = base_layout("Rolling Sharpe Ratio (20-day)", height=320)
            fig_sh.update_layout(**layout_sh)
            st.plotly_chart(fig_sh, use_container_width=True)

        with col_st:
            strat_colors = {
                "Mean-Variance":       NEON,
                "Risk Parity":         BLUE,
                "Minimum Variance":    PINK,
                "Momentum Tilt":       ORANGE,
                "Defensive Allocation":"#A29BFE"
            }
            fig_st = go.Figure()
            for strat in strategy_set:
                mask = strategy_timeline == strat
                fig_st.add_trace(go.Scatter(
                    x=strategy_timeline[mask].index,
                    y=confidence_scores[mask].values,
                    mode="markers",
                    name=strat,
                    marker=dict(
                        color=strat_colors.get(strat, NEON),
                        size=7,
                        opacity=0.85,
                        symbol="circle"
                    )
                ))
            layout_st = base_layout("Strategy Selection & Confidence", height=320)
            layout_st["yaxis"]["tickformat"] = ".0%"
            layout_st["yaxis"]["title"] = "Confidence"
            fig_st.update_layout(**layout_st)
            st.plotly_chart(fig_st, use_container_width=True)

        st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

        # ── ROW 3: Portfolio Allocation Over Time ─────────────────────────────
        st.markdown("<div class='section-header'>⚖️ Portfolio Allocation Evolution</div>", unsafe_allow_html=True)
        asset_colors = [NEON, BLUE, PINK, ORANGE, "#A29BFE"]
        fig_alloc = go.Figure()
        for i, asset in enumerate(allocations.columns):
            fig_alloc.add_trace(go.Scatter(
                x=allocations.index,
                y=allocations[asset].values * 100,
                mode="lines",
                name=asset,
                line=dict(color=asset_colors[i % len(asset_colors)], width=2),
                stackgroup="one",
                fillcolor=f"rgba({int(asset_colors[i%len(asset_colors)][1:3],16)},"
                          f"{int(asset_colors[i%len(asset_colors)][3:5],16)},"
                          f"{int(asset_colors[i%len(asset_colors)][5:7],16)},0.25)"
            ))
        layout_alloc = base_layout("Asset Weight Allocation Over Time (%)", height=300)
        layout_alloc["yaxis"]["ticksuffix"] = "%"
        layout_alloc["yaxis"]["range"]      = [0, 100]
        fig_alloc.update_layout(**layout_alloc)
        st.plotly_chart(fig_alloc, use_container_width=True)

        st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

        # ── ROW 4: Spectral Decomposition ─────────────────────────────────────
        st.markdown("<div class='section-header'>📡 Spectral Market Decomposition</div>", unsafe_allow_html=True)

        # Compute FFT
        port_ret_arr = np.dot(allocations.values, returns.iloc[lookback:].values.T)
        port_ret_arr = np.mean(port_ret_arr, axis=0)
        fft_window, step = 32, 4
        freq_list, amp_list, time_list = [], [], []
        for i in range(0, len(port_ret_arr) - fft_window, step):
            seg   = port_ret_arr[i:i + fft_window]
            fft   = np.fft.rfft(seg)
            freqs = np.fft.rfftfreq(fft_window, d=1)
            amp_list.append(np.abs(fft))
            freq_list.append(freqs)
            time_list.append(i)

        freq_arr = np.array(freq_list)
        amp_arr  = np.array(amp_list)
        time_arr = np.array(time_list)

        col_3d, col_fft = st.columns([3, 2])

        with col_3d:
            fig_surf = go.Figure(data=[go.Surface(
                z=amp_arr.T,
                x=time_arr,
                y=freq_arr[0],
                colorscale=[
                    [0,   "#0D1117"],
                    [0.3, "#0D2137"],
                    [0.6, "#00668A"],
                    [1.0, "#00FFC6"]
                ],
                showscale=True,
                colorbar=dict(
                    title=dict(text="Amplitude", font=dict(color=NEON, size=10)),
                    tickfont=dict(color="#8B9DC3", size=9),
                    bgcolor=DARK_BG, thickness=12
                ),
                lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3),
                opacity=0.95
            )])
            fig_surf.update_layout(
                scene=dict(
                    xaxis=dict(title=dict(text="Time", font=dict(color=NEON, size=10)),
                               backgroundcolor=DARK_BG, gridcolor=GRID_COLOR,
                               tickfont=dict(color="#8B9DC3", size=8)),
                    yaxis=dict(title=dict(text="Frequency", font=dict(color=BLUE, size=10)),
                               backgroundcolor=DARK_BG, gridcolor=GRID_COLOR,
                               tickfont=dict(color="#8B9DC3", size=8)),
                    zaxis=dict(title=dict(text="Amplitude", font=dict(color=ORANGE, size=10)),
                               backgroundcolor=DARK_BG, gridcolor=GRID_COLOR,
                               tickfont=dict(color="#8B9DC3", size=8)),
                    bgcolor=DARK_BG,
                    camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9))
                ),
                template="plotly_dark",
                paper_bgcolor=DARK_BG,
                plot_bgcolor=DARK_BG,
                font=dict(color="#8B9DC3", family="Inter"),
                height=440,
                margin=dict(l=0, r=0, t=30, b=0),
                title=dict(text="Rolling Spectral Surface (FFT)", font=dict(color=NEON, size=12), x=0.01)
            )
            st.plotly_chart(fig_surf, use_container_width=True)

        with col_fft:
            last_amp   = amp_arr[-1]
            last_freqs = freq_arr[-1]
            dom_idx    = np.argmax(last_amp)
            dom_freq   = last_freqs[dom_idx]
            dom_amp    = last_amp[dom_idx]

            # Metric pills
            st.markdown(f"""
            <div style='display:flex; gap:12px; margin-bottom:16px; flex-wrap:wrap;'>
                <div style='background:#0D1117; border:1px solid {NEON}22; border-radius:8px; padding:12px 16px; flex:1;'>
                    <div style='font-size:0.65rem; color:#6B7DA8; text-transform:uppercase; letter-spacing:1px;'>Dom. Frequency</div>
                    <div style='font-family:JetBrains Mono,monospace; font-size:1.3rem; color:{NEON}; font-weight:600;'>{dom_freq:.4f}</div>
                    <div style='font-size:0.65rem; color:#3A4560;'>Hz</div>
                </div>
                <div style='background:#0D1117; border:1px solid {BLUE}22; border-radius:8px; padding:12px 16px; flex:1;'>
                    <div style='font-size:0.65rem; color:#6B7DA8; text-transform:uppercase; letter-spacing:1px;'>Dom. Amplitude</div>
                    <div style='font-family:JetBrains Mono,monospace; font-size:1.3rem; color:{BLUE}; font-weight:600;'>{dom_amp:.4f}</div>
                    <div style='font-size:0.65rem; color:#3A4560;'>units</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            fig_fft = go.Figure()
            fig_fft.add_trace(go.Scatter(
                x=last_freqs, y=last_amp,
                mode="lines", name="Amplitude",
                line=dict(color=NEON, width=2),
                fill="tozeroy", fillcolor="rgba(0,255,198,0.07)"
            ))
            fig_fft.add_trace(go.Scatter(
                x=[dom_freq], y=[dom_amp],
                mode="markers", name="Dominant",
                marker=dict(size=12, color=PINK, symbol="diamond",
                            line=dict(color="#fff", width=1))
            ))
            layout_fft = base_layout("FFT Spectrum (Latest Window)", height=390)
            layout_fft["xaxis"]["title"] = "Frequency (Hz)"
            layout_fft["yaxis"]["title"] = "Amplitude"
            fig_fft.update_layout(**layout_fft)
            st.plotly_chart(fig_fft, use_container_width=True)

        st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

        # ── ROW 5: 3D Allocation + Regime Heatmap ─────────────────────────────
        st.markdown("<div class='section-header'>🔬 Multi-Dimensional Analysis</div>", unsafe_allow_html=True)
        col_a3d, col_rheat = st.columns(2)

        with col_a3d:
            alloc_3d = allocations.tail(40)
            assets   = alloc_3d.columns.tolist()
            dates    = alloc_3d.index.strftime('%m/%d').tolist()
            fig_3d   = go.Figure()
            for i, asset in enumerate(assets):
                c = asset_colors[i % len(asset_colors)]
                r, g, b = int(c[1:3],16), int(c[3:5],16), int(c[5:7],16)
                fig_3d.add_trace(go.Scatter3d(
                    x=list(range(len(dates))), y=[i]*len(dates),
                    z=alloc_3d[asset].values,
                    mode="lines+markers", name=asset,
                    line=dict(color=c, width=5),
                    marker=dict(size=4, color=c, opacity=0.9),
                    text=dates
                ))
            fig_3d.update_layout(
                scene=dict(
                    xaxis=dict(title=dict(text="Time", font=dict(color=NEON, size=10)),
                               tickvals=list(range(0,len(dates),8)),
                               ticktext=dates[::8],
                               backgroundcolor=DARK_BG, gridcolor=GRID_COLOR,
                               tickfont=dict(color="#8B9DC3", size=8)),
                    yaxis=dict(title=dict(text="Asset", font=dict(color=BLUE, size=10)),
                               tickvals=list(range(len(assets))),
                               ticktext=assets,
                               backgroundcolor=DARK_BG, gridcolor=GRID_COLOR,
                               tickfont=dict(color="#8B9DC3", size=8)),
                    zaxis=dict(title=dict(text="Weight", font=dict(color=ORANGE, size=10)),
                               backgroundcolor=DARK_BG, gridcolor=GRID_COLOR,
                               tickfont=dict(color="#8B9DC3", size=8)),
                    bgcolor=DARK_BG,
                    camera=dict(eye=dict(x=2, y=-1.5, z=1))
                ),
                template="plotly_dark",
                paper_bgcolor=DARK_BG,
                font=dict(color="#8B9DC3", family="Inter"),
                height=440,
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                title=dict(text="3D Portfolio Allocation Over Time", font=dict(color=NEON, size=12), x=0.01)
            )
            st.plotly_chart(fig_3d, use_container_width=True)

        with col_rheat:
            regime_plot = regime_df.tail(60).T
            fig_heat = go.Figure(data=go.Heatmap(
                z=regime_plot.values,
                x=[str(d)[:10] for d in regime_plot.columns],
                y=regime_plot.index.tolist(),
                colorscale=[[0, DARK_BG], [0.5, "#1E3A5F"], [1.0, NEON]],
                showscale=False,
                xgap=1, ygap=2,
                hovertemplate="Date: %{x}<br>Regime: %{y}<br>State: %{z}<extra></extra>"
            ))
            layout_heat = base_layout("Market Regime Heatmap (Last 60 Days)", height=440)
            layout_heat["xaxis"]["tickangle"] = -45
            layout_heat["xaxis"]["nticks"]    = 10
            fig_heat.update_layout(**layout_heat)
            st.plotly_chart(fig_heat, use_container_width=True)

        # ── FOOTER ────────────────────────────────────────────────────────────
        st.markdown("""
        <div class='footer'>
            ⚡ <strong style='color:#58A6FF;'>Portfolio Meta-Learning Engine</strong> ·
            Built by <strong style='color:#00FFC6;'>Prince Maurya</strong> ·
            <a href='https://github.com/alwaysprince05/Portfolio-Meta-Learning-Engine'
               style='color:#3A4560; text-decoration:none;'>github.com/alwaysprince05</a>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Dashboard error: {e}")
        st.code(str(e))
