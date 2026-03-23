import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Ultimate Bioreactor Simulator", layout="wide")

st.title(" Ultimate Bioreactor Simulation Platform")
st.markdown("### AI + Control + Dynamics Integrated System")

#  Sidebar Controls
st.sidebar.header(" Control Panel")

T = st.sidebar.slider("Temperature (°C)", 20, 50, 37)
pH_init = st.sidebar.slider("Initial pH", 4.0, 9.0, 7.0)
X0 = st.sidebar.slider("Initial Biomass", 0.01, 1.0, 0.1)
S0 = st.sidebar.slider("Initial Substrate", 1.0, 20.0, 10.0)

use_pid = st.sidebar.checkbox("Enable pH Control (PID)", True)
run_anim = st.sidebar.checkbox("Run Live Animation", False)

#  Constants
mu_max = 0.6
T_opt = 37
pH_opt = 7
kT = 50
kpH = 1.5
Y = 0.5
k_prod = 0.2

# PID constants
Kp, Ki, Kd = 2.0, 0.1, 0.05

#  Time
t = np.linspace(0, 20, 200)

#  Variables
X = np.zeros(len(t))
S = np.zeros(len(t))
P = np.zeros(len(t))
pH = np.zeros(len(t))

X[0] = X0
S[0] = S0
pH[0] = pH_init

integral = 0
prev_error = 0

#  Simulation
for i in range(1, len(t)):
    mu_T = mu_max * np.exp(-((T - T_opt)**2 / kT))
    mu_pH = np.exp(-((pH[i-1] - pH_opt)**2 / kpH))
    mu = mu_T * mu_pH

    dX = mu * X[i-1]
    dS = -(1/Y) * mu * X[i-1]
    dP = k_prod * X[i-1]
    dpH = -0.05 * X[i-1]

    if use_pid:
        error = pH_opt - pH[i-1]
        integral += error
        derivative = error - prev_error
        control = Kp*error + Ki*integral + Kd*derivative
        dpH += control * 0.01
        prev_error = error

    X[i] = X[i-1] + dX * 0.1
    S[i] = max(S[i-1] + dS * 0.1, 0)
    P[i] = P[i-1] + dP * 0.1
    pH[i] = pH[i-1] + dpH * 0.1

# Layout
col1, col2 = st.columns(2)

# Animation or static plot
if run_anim:
    placeholder = st.empty()
    for i in range(10, len(t)):
        fig, ax = plt.subplots()
        ax.plot(t[:i], X[:i], label="Biomass")
        ax.plot(t[:i], P[:i], label="Product")
        ax.legend()
        placeholder.pyplot(fig)
        time.sleep(0.05)
else:
    with col1:
        st.subheader(" Biomass & Product")
        fig1, ax1 = plt.subplots()
        ax1.plot(t, X, label="Biomass")
        ax1.plot(t, P, label="Product")
        ax1.legend()
        st.pyplot(fig1)

    with col2:
        st.subheader(" Substrate & pH")
        fig2, ax2 = plt.subplots()
        ax2.plot(t, S, label="Substrate")
        ax2.plot(t, pH, label="pH")
        ax2.legend()
        st.pyplot(fig2)

#  AI MODEL
temps = np.linspace(20, 50, 30)
pHs = np.linspace(4, 9, 30)

X_train = []
y_train = []

for temp in temps:
    for ph in pHs:
        mu_T = mu_max * np.exp(-((temp - T_opt)**2 / kT))
        mu_pH = np.exp(-((ph - pH_opt)**2 / kpH))
        growth = mu_T * mu_pH
        X_train.append([temp, ph])
        y_train.append(growth)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict([[T, pH_init]])

#  3D Visualization
fig3d = go.Figure(data=[go.Surface(z=[X[:50], X[:50]])])
fig3d.update_layout(title="3D Reactor Behavior")

# Insights
st.subheader(" Insights Panel")

st.write(f" AI Predicted Growth Score: {pred[0]:.3f}")

if use_pid:
    st.success("PID Control Active → Stable pH")
else:
    st.warning("No Control → pH may drop")

#  Optimizer
if st.button("Find Best Conditions"):
    best_growth = 0
    best_T = 0
    best_pH = 0

    for temp in np.linspace(20, 50, 20):
        for ph in np.linspace(4, 9, 20):
            mu_T = mu_max * np.exp(-((temp - T_opt)**2 / kT))
            mu_pH = np.exp(-((ph - pH_opt)**2 / kpH))
            growth = mu_T * mu_pH

            if growth > best_growth:
                best_growth = growth
                best_T = temp
                best_pH = ph

    st.success(f" Optimal: T={best_T:.2f}°C, pH={best_pH:.2f}")

#  3D plot display
st.plotly_chart(fig3d)

st.markdown("---")
st.write(" Integrated Simulation: Dynamics + Control + AI Optimization")
