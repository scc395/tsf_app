import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from scipy.signal import welch
from scipy.stats import entropy

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(page_title="TSF Signal Lab", layout="wide")
st.title("🧪 TSF Signal Synthesis & Analysis Laboratory")
st.markdown("Adjust the physical components using the sliders to observe real-time changes in Trend (T), Seasonality (S), and Forecastability (F).")

# ==========================================
# 2. Control Panel (Optimized for Sensitive Zone)
# ==========================================
st.sidebar.header("🎛️ Signal Injector")
st.sidebar.info("💡 Note: The sliders control the absolute physical amplitude. Due to the variance-based formulas, the TSF ratios saturate quickly. The ranges below are narrowed to the most 'sensitive' mathematical zones.")

# 🌟 优化点：缩小最大值到敏感区，增加滑块步长精度
trend_slope = st.sidebar.slider("📈 Trend Slope", min_value=0.00, max_value=2.00, value=0.50, step=0.05)
season_amp = st.sidebar.slider("🌊 Seasonality Amplitude", min_value=0.00, max_value=2.00, value=0.80, step=0.05)
noise_level = st.sidebar.slider("🎛️ Noise Level", min_value=0.00, max_value=2.00, value=0.50, step=0.05)

# ==========================================
# 3. Signal Synthesis
# ==========================================
n_points = 500
period = 24  
t = np.arange(n_points)

tau_true = trend_slope * (t / n_points) * 10       
s_true = season_amp * np.sin(2 * np.pi * t / period) 
r_true = noise_level * np.random.randn(n_points)   

x = tau_true + s_true + r_true

# ==========================================
# 4. TSF Calculation (Clamped to [0, 1])
# ==========================================
try:
    stl = STL(x, period=period, robust=True).fit()
    var_r = np.var(stl.resid)
    var_sr = np.var(stl.seasonal + stl.resid)
    var_tr = np.var(stl.trend + stl.resid)
    
    raw_S = 1.0 - var_r / var_sr if var_sr > 1e-9 else 0.0
    raw_T = 1.0 - var_r / var_tr if var_tr > 1e-9 else 0.0
    
    S = min(1.0, max(0.0, float(raw_S)))
    T = min(1.0, max(0.0, float(raw_T)))
except Exception:
    S, T = 0.0, 0.0

try:
    x_c = x - np.mean(x)
    f, Pxx = welch(x_c, window='hann', nperseg=min(len(x_c), 256))
    Pxx_norm = Pxx / np.sum(Pxx)
    Pxx_norm = Pxx_norm[Pxx_norm > 0]
    
    H = entropy(Pxx_norm, base=np.e) / np.log(len(Pxx_norm)) if len(Pxx_norm) > 1 else 0.0
    F = min(1.0, max(0.0, float(1.0 - H)))
except Exception:
    F = 0.0

# ==========================================
# 5. Dashboard Rendering
# ==========================================
col1, col2, col3 = st.columns(3)
col1.metric(label="Trend Strength (T)", value=f"{T:.3f}")
col2.metric(label="Seasonality Strength (S)", value=f"{S:.3f}")
col3.metric(label="Forecastability (F)", value=f"{F:.3f}")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(t, x, color='teal', linewidth=1.5, alpha=0.8, label="Synthesized Signal")
ax.set_title("Real-time Time Series Waveform", fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)

with st.expander("🔍 Reveal STL Physical Decomposition"):
    fig2, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t, stl.trend, color='blue')
    axes[0].set_ylabel("Trend")
    axes[1].plot(t, stl.seasonal, color='green')
    axes[1].set_ylabel("Seasonal")
    axes[2].plot(t, stl.resid, color='red', alpha=0.5)
    axes[2].set_ylabel("Residual")
    st.pyplot(fig2)