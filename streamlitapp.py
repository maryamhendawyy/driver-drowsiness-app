import pickle
import streamlit as st
import numpy as np
import pandas as pd
try:
    from scipy.stats import multivariate_normal
except Exception as e:
    import streamlit as _st
    _st.error("Missing required package: scipy. Install it with `python -m pip install scipy` and re-run the app.")
    raise
import time
from PIL import Image
import base64
import io
from streamlit.components.v1 import html as st_html

# -----------------------------------------------------------

# -----------------------------------------------------------
with open("HMM_0.pkl", "rb") as f:
    HMM_0 = pickle.load(f)
with open("HMM_1.pkl", "rb") as f:
    HMM_1 = pickle.load(f)


    

pi = np.load("pi.npy")
A = np.load("A.npy")
means = np.load("means.npy")
covs = np.load("covs.npy")

# Feature names expected by the model (must match what you used during training)
FEATURE_NAMES = [
    'TotalAlpha','TotalBeta','TotalGamma',
    'RelDelta','RelTheta',
    'Theta_Alpha','Beta_Alpha','Gamma_Beta',
    'attention','meditation'
]

# -----------------------------------------------------------
def viterbi(X, pi, A, means, covs):
    T, D = X.shape
    N = len(pi)

    # Make sure the input dimension matches the model's expected dimension
    model_dim = means.shape[1] if means.ndim == 2 else (means.shape[0] if means.ndim == 1 else None)
    if model_dim is not None and D != model_dim:
        raise ValueError(f"Dim mismatch: input has {D} features but model expects {model_dim} features.\nEnsure the uploaded CSV has the same feature columns and order used during training.")

    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)

    # init
    for i in range(N):
        emission = multivariate_normal.pdf(X[0], means[i], covs[i], allow_singular=True)
        delta[0, i] = pi[i] * emission

    # recursion
    for t in range(1, T):
        for j in range(N):
            emission = multivariate_normal.pdf(X[t], means[j], covs[j], allow_singular=True)
            candidates = delta[t-1] * A[:, j]
            delta[t, j] = np.max(candidates) * emission
            psi[t, j] = np.argmax(candidates)

    # backtracking
    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(delta[-1])

    for t in reversed(range(1, T)):
        states[t-1] = psi[t, states[t]]

    return states

# -----------------------------------------------------------
# -----------------------------------------------------------
def _img_to_base64(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

alert_red_b64 = _img_to_base64('alert_red.jpg')
awake_white_b64 = _img_to_base64('awake_white.jpg')
car_gif_b64 = _img_to_base64('car.gif')
# Try to find a road wallpaper image (jpg or png); if present, encode to base64
WALLPAPER_B64 = None
for candidate in ("road_wallpaper.png", "road_wallpaper.jpg"):
    try:
        WALLPAPER_B64 = _img_to_base64(candidate)
        WALLPAPER_EXT = "png" if candidate.endswith(".png") else "jpg"
        break
    except FileNotFoundError:
        WALLPAPER_B64 = None
        WALLPAPER_EXT = None

# -----------------------------------------------------------
# 4) Streamlit UI
# -----------------------------------------------------------
st.set_page_config(page_title="Driver Drowsiness Monitor", layout="wide")
if WALLPAPER_B64:
    bg_css = f"background: url('data:image/{WALLPAPER_EXT};base64,{WALLPAPER_B64}') no-repeat center center fixed; background-size: cover;"
else:
    bg_css = "background: linear-gradient(180deg, #e6eef6 0%, #cfdff0 100%);"

css = """
<style>
.stApp {
""" + bg_css + """
}
.centered {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.alert-icon {
    margin-bottom: 0.1em;
    margin-top: 0.1em;
}
.driver-state {
    font-size: 1.6em;
    font-weight: bold;
    margin-bottom: 0.1em;
    margin-top: 0.05em;
}
.monitor-card {
    background: rgba(255, 255, 255, 0.04);
    border-radius: 10px;
    padding: 6px 12px;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

st.title("🚗 Driver Drowsiness Monitor (HMM + Viterbi)")

# Load the synthetic CSV directly (no upload)
CSV_PATH = "synthetic_states_example_with_state.csv"
df = pd.read_csv(CSV_PATH)

# Sanitize and extract state column if present
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
state_col_name = None
for candidate in ['state', 'classification', 'label', 'class']:
    if candidate in df.columns:
        state_col_name = candidate
        break
if state_col_name:
    states_override = df[state_col_name].astype(int).to_numpy()
else:
    states_override = None
df = df[FEATURE_NAMES]

left_col, center_col, right_col = st.columns([1, 2, 1])
with center_col:
    st.markdown('<div class="centered monitor-card" style="margin-top:12px; width:420px;">', unsafe_allow_html=True)
    # Alert icon placeholder with reduced height to avoid layout shifts
    alert_icon_slot = st.empty()
    st.markdown('<div style="height:56px; width:100%;"></div>', unsafe_allow_html=True)
    # Driver state text placeholder with smaller height
    state_text_slot = st.empty()
    st.markdown('<div style="height:28px; width:100%;"></div>', unsafe_allow_html=True)
    # Car GIF (always animated, only once). Reserve space below for audio controls
    gif_html = ('<div style="height:300px; display:flex; align-items:center; justify-content:center; '
                 'margin-top:2px; margin-bottom:4px;">' +
                f'<img src="data:image/gif;base64,{car_gif_b64}" style="max-width:100%; max-height:300px; display:block;"/>' +
                '</div>')
    st_html(gif_html, height=320)
    # Audio placeholder separate below GIF
    audio_slot = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# Audio and icon sizes
alarm_url = "https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg"
alert_icon_size = 80

# Compute the initial state so UI appears immediately
if len(df) == 0:
    st.warning("No data rows found in CSV.")
    st.stop()

if states_override is not None:
    prev_state = int(states_override[0])
else:
    prev_state = viterbi(df.iloc[0:1].values, pi, A, means, covs)[0]

# Render the initial UI
if prev_state == 1:
    icon_html = f'<img src="data:image/jpeg;base64,{alert_red_b64}" style="width:{alert_icon_size}px; height:{alert_icon_size}px; object-fit:contain; display:block; margin: 0 auto;"/>'
    with alert_icon_slot:
        st.markdown(icon_html, unsafe_allow_html=True)
    with state_text_slot:
        st.markdown('<div style="text-align:center; font-size:28px; color:#d32f2f; font-weight:700; margin-top:8px;">⚠️ Driver is <b>Drowsy</b></div>', unsafe_allow_html=True)
    with audio_slot:
        audio_slot.audio(alarm_url, format="audio/ogg", start_time=0)
else:
    icon_html = f'<img src="data:image/jpeg;base64,{awake_white_b64}" style="width:{alert_icon_size}px; height:{alert_icon_size}px; object-fit:contain; display:block; margin: 0 auto;"/>'
    with alert_icon_slot:
        st.markdown(icon_html, unsafe_allow_html=True)
    with state_text_slot:
        st.markdown('<div style="text-align:center; font-size:28px; color:#388e3c; font-weight:700; margin-top:8px;">✅ Driver is <b>Awake</b></div>', unsafe_allow_html=True)
    with audio_slot:
        audio_slot.empty()
    # Ensure audio player is empty initially
    audio_slot.empty()

# -----------------------------------------------------------
# 5) Real-time loop
# -----------------------------------------------------------

# Audio and icon sizes (already defined above)
for t in range(len(df)):
    row = df.iloc[t:t+1].values
    if states_override is not None:
        state = int(states_override[t])
    else:
        state = viterbi(row, pi, A, means, covs)[0]

    # Only update UI elements when the state changes to reduce layout reflow (prevents shakiness)
    if prev_state != state:
        # Centered alert icon (above car GIF)
        with alert_icon_slot:
            if state == 1:
                icon_html = f'<img src="data:image/jpeg;base64,{alert_red_b64}" style="width:{alert_icon_size}px; height:{alert_icon_size}px; object-fit:contain; display:block; margin: 0 auto;"/>'
            else:
                icon_html = f'<img src="data:image/jpeg;base64,{awake_white_b64}" style="width:{alert_icon_size}px; height:{alert_icon_size}px; object-fit:contain; display:block; margin: 0 auto;"/>'
            st.markdown(icon_html, unsafe_allow_html=True)

        # Centered driver state text (below icon, above GIF)
        with state_text_slot:
            if state == 1:
                st.markdown('<div style="text-align:center; font-size:28px; color:#d32f2f; font-weight:700; margin-top:8px;">⚠️ Driver is <b>Drowsy</b></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="text-align:center; font-size:28px; color:#388e3c; font-weight:700; margin-top:8px;">✅ Driver is <b>Awake</b></div>', unsafe_allow_html=True)

        # Audio: play alarm if drowsy, stop if awake; only re-create when state changes
        with audio_slot:
            if state == 1:
                st.audio(alarm_url, format="audio/ogg", start_time=0)
            else:
                st.empty()
        prev_state = state

    time.sleep(1)
