# >> streamlit run C:/.../path_to_file/.../dashboard.py (to run the application locally)
import streamlit as st
import numpy as np
from option import EuropeanOption, AmericanOption
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px

# Header
st.title("Binomial Tree Model 🌳🌳")

# Sidebar
st.sidebar.title("Parameters")

# user inputs on sidebar
opt_style = st.sidebar.selectbox('Option Style: ', ('European', 'American'))
opt_type = st.sidebar.selectbox('Option Type: ', ('Call', 'Put'))
S = st.sidebar.number_input('Stock Price (S)', value=500, min_value=0, max_value=5000)
K = st.sidebar.number_input('Strike Price (K)', value=500, min_value=0, max_value=5000)
T = st.sidebar.slider('Maturity (T)', value=1.0, min_value=0.0, max_value=5.0, step=0.1)
r = st.sidebar.slider('Interest Rate (r)', value=0.0, min_value=0.0, max_value=0.2, step=0.01)
sigma = st.sidebar.slider('Volatility (σ)', value=0.2, min_value=0.0, max_value=1.0, step=0.01)
n_steps = st.sidebar.slider('Steps (N)', value=10, min_value=2, max_value=100, step=1)
# Initialize Option
if opt_style == 'European':
    option = EuropeanOption(s=S, k=K, t=T, r=r, sigma=sigma, opt_type=str.lower(opt_type))
else:
    option = AmericanOption(s=S, k=K, t=T, r=r, sigma=sigma, opt_type=str.lower(opt_type))

# Price
asset_prices = option.binomial_asset_prices(n_steps)
f, option_prices_matrix = option.binomial_option_prices(asset_prices)
delta, hedged_positions_matrix = option.binomial_hedged_positions(asset_prices, option_prices_matrix)

st.metric('Option Price', value=f'{f:.4f} EUR', delta=f'{delta:.4f} ∆ (shares)', delta_color="normal", help=None)

# Binomial Model Heatmap
# TODO: find a better way to plot the binomial tree
fig, ax = plt.subplots(figsize=(15, 15))
mask = asset_prices.astype(bool)  # take upper correlation matrix
sns.heatmap(asset_prices, ax=ax, annot=True, mask=~mask)
fig = px.imshow(asset_prices, color_continuous_scale='Greens', origin='upper', width=700, height=700)
st.write(fig)
