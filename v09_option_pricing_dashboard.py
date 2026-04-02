# Streamlit App: European Option Pricing Dashboard
# Methods: Black-Scholes, Finite Difference, Monte Carlo
# Instruments: Call, Put, Binary Call, Binary Put, Forward
# Run: streamlit run v09_option_pricing_dashboard.py

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from scipy.stats import norm
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

st.set_page_config(layout="wide", page_title="Option Pricing Terminal")

# ─────────────────────────────────────────────
# Custom CSS — theme-neutral (light & dark safe)
# All colors use inherit / currentColor / low-opacity
# neutrals so they adapt to Streamlit's active theme.
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }

    /* ── Instrument radio toggle ── */
    .stRadio > div { flex-direction: row; gap: 0.5rem; flex-wrap: wrap; }
    .stRadio > div > label {
        border: 1.5px solid rgba(100, 100, 100, 0.35);
        border-radius: 6px;
        padding: 5px 18px;
        cursor: pointer;
        font-weight: 600;
        font-size: 0.9rem;
        color: inherit;
        background: transparent;
        transition: border-color 0.15s, background 0.15s;
    }
    .stRadio > div > label:hover {
        border-color: rgba(100, 100, 100, 0.7);
    }

    /* ── Greek metric cards ── */
    .metric-box {
        border: 1px solid rgba(100, 100, 100, 0.25);
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
        background: rgba(128, 128, 128, 0.05);
    }
    .metric-label {
        font-size: 0.70rem;
        opacity: 0.6;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: inherit;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: inherit;
        margin-top: 4px;
    }

    /* ── Section divider ── */
    .section-divider {
        border-top: 1px solid rgba(100, 100, 100, 0.2);
        margin: 1.1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Matplotlib theme — clean white background,
# neutral gray axes; renders well in both modes
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f7f7f7",
    "axes.edgecolor":    "#aaaaaa",
    "axes.labelcolor":   "#444444",
    "xtick.color":       "#555555",
    "ytick.color":       "#555555",
    "text.color":        "#333333",
    "grid.color":        "#dddddd",
    "grid.linewidth":    0.7,
    "legend.framealpha": 0.85,
    "legend.edgecolor":  "#cccccc",
})

# Plot line colors — distinct and readable on white
PRIMARY_COLOR = "#1f77b4"   # mpl default blue
OVERLAY_COLOR = "#d62728"   # mpl default red
SPOT_COLOR    = "#e6820e"   # amber
STRIKE_COLOR  = "#2ca02c"   # green

# ─────────────────────────────────────────────
# Pricing + Greeks Functions
# ─────────────────────────────────────────────

def _d1d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_call(S, K, T, r, sigma):
    d1, d2 = _d1d2(S, K, T, r, sigma)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T)
    theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    rho   = K * T * np.exp(-r * T) * norm.cdf(d2)
    return price, delta, gamma, vega, theta, rho


def bs_put(S, K, T, r, sigma):
    d1, d2 = _d1d2(S, K, T, r, sigma)
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    delta = norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T)
    theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    return price, delta, gamma, vega, theta, rho


def binary_call(S, K, T, r, sigma):
    d1, d2 = _d1d2(S, K, T, r, sigma)
    er = np.exp(-r * T)
    price = er * norm.cdf(d2)
    delta = er * norm.pdf(d2) / (S * sigma * np.sqrt(T))
    gamma = -er * norm.pdf(d2) * d1 / (S ** 2 * sigma ** 2 * T)
    vega  = -er * norm.pdf(d2) * d1 / sigma
    theta = (-r * er * norm.cdf(d2)
             - er * norm.pdf(d2) * ((r - 0.5 * sigma ** 2) / (sigma * np.sqrt(T)) - d2 / (2 * T)))
    rho   = -T * er * norm.cdf(d2) + er * norm.pdf(d2) * np.sqrt(T) / sigma
    return price, delta, gamma, vega, theta, rho


def binary_put(S, K, T, r, sigma):
    d1, d2 = _d1d2(S, K, T, r, sigma)
    er = np.exp(-r * T)
    price = er * norm.cdf(-d2)
    delta = -er * norm.pdf(d2) / (S * sigma * np.sqrt(T))
    gamma = er * norm.pdf(d2) * d1 / (S ** 2 * sigma ** 2 * T)
    vega  = er * norm.pdf(d2) * d1 / sigma
    theta = (-r * er * norm.cdf(-d2)
             + er * norm.pdf(d2) * ((r - 0.5 * sigma ** 2) / (sigma * np.sqrt(T)) - d2 / (2 * T)))
    rho   = T * er * norm.cdf(-d2) - er * norm.pdf(d2) * np.sqrt(T) / sigma
    return price, delta, gamma, vega, theta, rho


def forward_price(S, K, T, r):
    price = S - K * np.exp(-r * T)
    delta = 1.0
    gamma = 0.0
    vega  = 0.0
    theta = r * K * np.exp(-r * T)
    rho   = -K * T * np.exp(-r * T)
    return price, delta, gamma, vega, theta, rho


# ─────────────────────────────────────────────
# Finite Difference (Crank-Nicolson)
# ─────────────────────────────────────────────

def fd_price_grid(S0, K, T, r, sigma, M, N, Smax, option_type="Call"):
    S = np.linspace(0, Smax, M + 1)
    if option_type == "Call":
        V = np.maximum(S - K, 0)
    elif option_type == "Put":
        V = np.maximum(K - S, 0)
    else:
        return S, np.full_like(S, np.nan)

    dt = T / N
    i  = np.arange(1, M)
    a  =  0.25 * dt * (sigma ** 2 * i ** 2 - r * i)
    b  = -0.5  * dt * (sigma ** 2 * i ** 2 + r)
    c  =  0.25 * dt * (sigma ** 2 * i ** 2 + r * i)

    A = diags([-a[1:], 1 - b, -c[:-1]], [-1, 0, 1]).tocsc()
    B = diags([ a[1:], 1 + b,  c[:-1]], [-1, 0, 1]).tocsc()

    for n in range(N):
        rhs = B @ V[1:M]
        if option_type == "Call":
            rhs[-1] += c[-1] * (Smax - K * np.exp(-r * (T - n * dt)))
        V[1:M] = spsolve(A, rhs)

    return S, V


# ─────────────────────────────────────────────
# Monte Carlo
# ─────────────────────────────────────────────

def mc_price(S, K, T, r, sigma, n, option_type="Call"):
    Z  = np.random.randn(n)
    ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    if option_type == "Call":         payoff = np.maximum(ST - K, 0)
    elif option_type == "Put":        payoff = np.maximum(K - ST, 0)
    elif option_type == "Binary Call": payoff = (ST > K).astype(float)
    elif option_type == "Binary Put":  payoff = (ST < K).astype(float)
    elif option_type == "Forward":    payoff = ST - K
    else:                             payoff = np.zeros(n)
    return np.exp(-r * T) * np.mean(payoff)


# ─────────────────────────────────────────────
# Header + Instrument Toggle
# ─────────────────────────────────────────────

st.title("⚡ European Option Pricing Terminal")
st.markdown("*Black–Scholes · Finite Difference · Monte Carlo*")
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

INSTRUMENTS = ["Call", "Put", "Binary Call", "Binary Put", "Forward"]
instrument  = st.radio("Instrument", INSTRUMENTS, horizontal=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

st.sidebar.header("Market Inputs")
S0    = st.sidebar.number_input("S₀  — Spot Price",      min_value=1,    max_value=100000, value=100,  step=1)
K     = st.sidebar.number_input("K   — Strike",          min_value=1,    max_value=150000, value=100,  step=1)
T     = st.sidebar.number_input("T   — Years to Expiry", min_value=0.01, max_value=10.0,   value=1.0,  step=0.01)
r     = st.sidebar.number_input("r   — Risk-Free Rate",  min_value=0.0,  max_value=1.0,    value=0.05, step=0.001, format="%.4f")
sigma = st.sidebar.number_input("σ   — Volatility",      min_value=0.01, max_value=5.0,    value=0.20, step=0.01,  format="%.4f")

st.sidebar.markdown("---")
st.sidebar.header("Numerical Settings")
M     = st.sidebar.slider("FD Grid Size",      100, 1000, 500, step=50)
paths = st.sidebar.slider("Monte Carlo Paths", 10000, 200000, 50000, step=5000)

show_binary_overlay = st.sidebar.checkbox(
    "Overlay companion Greeks on plots",
    value=(instrument in ("Call", "Put", "Binary Call", "Binary Put")),
    help="Overlay the corresponding Binary/Vanilla counterpart Greeks on the same charts"
)

# ─────────────────────────────────────────────
# Dispatch pricing
# ─────────────────────────────────────────────

PRICING_FNS = {
    "Call":        bs_call,
    "Put":         bs_put,
    "Binary Call": binary_call,
    "Binary Put":  binary_put,
}

if instrument == "Forward":
    bs_vals = forward_price(S0, K, T, r)
else:
    bs_vals = PRICING_FNS[instrument](S0, K, T, r, sigma)

fd_supported = instrument in ("Call", "Put")
if fd_supported:
    S_grid, V_grid = fd_price_grid(S0, K, T, r, sigma, M, M, 2 * S0, instrument)
    fd_price_val   = np.interp(S0, S_grid, V_grid)
    dS             = S_grid[1] - S_grid[0]
    delta_fd       = (V_grid[2:] - V_grid[:-2]) / (2 * dS)
    gamma_fd       = (V_grid[2:] - 2 * V_grid[1:-1] + V_grid[:-2]) / (dS ** 2)
    fd_vals        = [
        fd_price_val,
        np.interp(S0, S_grid[1:-1], delta_fd),
        np.interp(S0, S_grid[1:-1], gamma_fd),
        np.nan, np.nan, np.nan,
    ]
else:
    fd_vals = [np.nan] * 6

mc_val  = mc_price(S0, K, T, r, sigma, paths, instrument)
mc_vals = [mc_val] + [np.nan] * 5

# ─────────────────────────────────────────────
# Summary metric cards
# ─────────────────────────────────────────────

price, delta, gamma, vega, theta, rho = bs_vals

m_col   = st.columns(6)
metrics = [
    ("Price",   f"{price:.4f}"),
    ("Delta Δ", f"{delta:.4f}"),
    ("Gamma Γ", f"{gamma:.5f}"),
    ("Vega ν",  f"{vega:.4f}"),
    ("Theta Θ", f"{theta:.4f}"),
    ("Rho ρ",   f"{rho:.4f}"),
]
for col, (label, val) in zip(m_col, metrics):
    with col:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{val}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("")

# ─────────────────────────────────────────────
# Info bar
# ─────────────────────────────────────────────

def moneyness(inst, S0, K):
    if S0 == K:
        return "ATM"
    itm = (inst in ("Call", "Binary Call", "Forward") and S0 > K) or \
          (inst in ("Put",  "Binary Put")              and S0 < K)
    return "ITM" if itm else "OTM"

info_cols = st.columns(4)
info_cols[0].write(f"**Instrument:** {instrument}")
info_cols[1].write(f"**S₀/K:** {S0}/{K}  |  **{moneyness(instrument, S0, K)}**")
info_cols[2].write(f"**σ:** {sigma:.2%}  |  **r:** {r:.2%}  |  **T:** {T:.2f}y")
info_cols[3].write(f"**MC paths:** {paths:,}  |  **FD grid:** {M}×{M}")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Results Table
# ─────────────────────────────────────────────

st.subheader("📊 Pricing Comparison")

labels = ["Price", "Delta", "Gamma", "Vega", "Theta", "Rho"]
df = pd.DataFrame({
    "Metric":            labels,
    "Analytical (B-S)":  [f"{v:.6f}" if not np.isnan(v) else "—" for v in bs_vals],
    "Finite Difference": [f"{v:.6f}" if not np.isnan(v) else "—" for v in fd_vals],
    "Monte Carlo":       [f"{v:.6f}" if not np.isnan(v) else "—" for v in mc_vals],
})
st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Greeks Plots
# ─────────────────────────────────────────────

st.subheader("📈 Greeks vs Spot Price")

S_range = np.linspace(max(1, 0.4 * S0), 1.6 * S0, 300)

def compute_greeks_vec(fn, S_arr, K, T, r, sigma, is_forward=False):
    out = {k: [] for k in ("delta", "gamma", "vega", "theta", "rho")}
    for s in S_arr:
        _, d, g, v, t, rh = forward_price(s, K, T, r) if is_forward else fn(s, K, T, r, sigma)
        out["delta"].append(d); out["gamma"].append(g)
        out["vega"].append(v);  out["theta"].append(t); out["rho"].append(rh)
    return out

is_fwd         = instrument == "Forward"
primary_fn     = PRICING_FNS.get(instrument)
primary_greeks = compute_greeks_vec(primary_fn, S_range, K, T, r, sigma, is_fwd)

BINARY_COMPANION = {
    "Call":        binary_call,
    "Put":         binary_put,
    "Binary Call": bs_call,
    "Binary Put":  bs_put,
}
overlay_greeks = None
overlay_label  = ""
if show_binary_overlay and instrument in BINARY_COMPANION:
    overlay_fn     = BINARY_COMPANION[instrument]
    overlay_greeks = compute_greeks_vec(overlay_fn, S_range, K, T, r, sigma)
    overlay_label  = (f"Binary {instrument}" if instrument in ("Call", "Put")
                      else instrument.replace("Binary ", "Vanilla "))

GREEK_KEYS   = ["delta", "gamma", "vega", "theta", "rho"]
GREEK_TITLES = ["Delta  Δ", "Gamma  Γ", "Vega  ν", "Theta  Θ", "Rho  ρ"]

fig, axes = plt.subplots(2, 3, figsize=(13, 6))
fig.subplots_adjust(hspace=0.44, wspace=0.32)

for idx, (key, title) in enumerate(zip(GREEK_KEYS, GREEK_TITLES)):
    ax = axes[idx // 3][idx % 3]

    ax.plot(S_range, primary_greeks[key],
            color=PRIMARY_COLOR, linewidth=2, label=instrument)

    if overlay_greeks is not None:
        ax.plot(S_range, overlay_greeks[key],
                color=OVERLAY_COLOR, linewidth=1.6,
                linestyle="--", label=overlay_label, alpha=0.9)

    ax.axvline(S0, color=SPOT_COLOR,   linewidth=1.1, linestyle=":", alpha=0.9)
    ax.axvline(K,  color=STRIKE_COLOR, linewidth=1.1, linestyle=":", alpha=0.7)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("Spot Price", fontsize=7.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(True)
    if overlay_greeks is not None:
        ax.legend(fontsize=7, loc="best", handlelength=1.5)

# Last subplot: reference key
ax_last = axes[1][2]
ax_last.axis("off")
legend_lines = [f"—  {instrument}  (primary)"]
if overlay_greeks is not None:
    legend_lines.append(f"--  {overlay_label}  (overlay)")
legend_lines += [
    "",
    f"···  Spot  S₀ = {S0}  (amber)",
    f"···  Strike K = {K}  (green)",
]
ax_last.text(
    0.08, 0.5, "\n".join(legend_lines),
    transform=ax_last.transAxes,
    fontsize=8.5, va="center", fontfamily="monospace",
    bbox=dict(facecolor="white", edgecolor="#cccccc",
              boxstyle="round,pad=0.6", alpha=0.9),
)

st.pyplot(fig)
plt.close(fig)

# ─────────────────────────────────────────────
# Payoff diagram
# ─────────────────────────────────────────────

st.subheader("💰 Payoff at Expiry")

ST_range = np.linspace(max(1, 0.2 * K), 1.8 * K, 400)

def payoff_fn(ST, inst, K):
    if inst == "Call":         return np.maximum(ST - K, 0)
    if inst == "Put":          return np.maximum(K - ST, 0)
    if inst == "Binary Call":  return (ST >= K).astype(float)
    if inst == "Binary Put":   return (ST <= K).astype(float)
    if inst == "Forward":      return ST - K
    return np.zeros_like(ST)

payoffs = payoff_fn(ST_range, instrument, K)

fig2, ax2 = plt.subplots(figsize=(10, 2.8))
ax2.plot(ST_range, payoffs, color=PRIMARY_COLOR, linewidth=2.5, label="Payoff")
ax2.fill_between(ST_range, 0, payoffs, alpha=0.12, color=PRIMARY_COLOR)
ax2.axvline(K,  color=STRIKE_COLOR, linewidth=1.2, linestyle="--", alpha=0.8, label=f"Strike K={K}")
ax2.axvline(S0, color=SPOT_COLOR,   linewidth=1.2, linestyle=":",  alpha=0.9, label=f"Spot S₀={S0}")
ax2.axhline(0,  color="#aaaaaa", linewidth=0.8)
ax2.set_xlabel("S_T  (Spot at Expiry)", fontsize=9)
ax2.set_ylabel("Payoff", fontsize=9)
ax2.set_title(f"{instrument} Payoff Profile", fontsize=10, fontweight="bold")
ax2.legend(fontsize=8)
ax2.grid(True)

st.pyplot(fig2)
plt.close(fig2)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption("Developed by: Al Yazdani  ·  Black–Scholes | Crank–Nicolson FD | Monte Carlo  ·  European options only")
