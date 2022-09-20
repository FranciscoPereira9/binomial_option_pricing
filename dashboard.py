# >> streamlit run C:/.../path_to_file/.../dashboard.py (to run the application locally)
import streamlit as st
import numpy as np
from option import EuropeanOption, AmericanOption
from bokeh.models import HoverTool, Span, Label
from bokeh.plotting import figure, show
import pandas as pd

# Header
st.title("Binomial Tree Model 🌳")

# Sidebar
st.sidebar.info('This application supports the pricing of European and American Options. For more information on the '
                'theory behind the model please refer to the Theory tab in the menu below.', icon="ℹ️")
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

tab1, tab2 = st.tabs(["Binomial Tree", "Theory"])

with tab1:
    # create a new plot with a title and axis labels
    TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"
    fig = figure(title="Sample Title", x_axis_label='Years', y_axis_label='Position', tools=TOOLS)
    for i in range(n_steps):
        x = [1, 0, 1]
        for j in range(i):
            x.append(0)
            x.append(1)
        x = np.array(x) + i
        a = 0
        b = 0
        y = []
        option = []
        hedge = []
        for x_i in x:
            if b == 2:
                a += 1
                b = 0
            y.append(asset_prices[a, x_i])
            option.append(option_prices_matrix[a, x_i])
            hedge.append(hedged_positions_matrix[a, x_i])
            b += 1
        data = {'x': x, 'price': y, 'option_price': option, 'hedged_position': hedge}
        df = pd.DataFrame(data)
        y = np.array(y)
        # add a line renderer with legend and line thickness to the plot
        fig.line(x='x', y='price', color='#ced4da', line_width=2, source=df)
        fig.circle(x='x', y='price', legend_label="Price", color="#22577a", size=6, source=df)
        hover = fig.select(dict(type=HoverTool))
        hover.tooltips = [("Hedged Position", "@hedged_position"), ("Option Price", "@option_price"),
                          ("Asset Price", "@price")]
    fig.legend.location = "top_left"
    strike = Span(location=K, dimension='width', line_color='#ef233c', line_dash='dashed',
                  line_width=2)
    strike_label = Label(x=0, y=K, text='Strike', border_line_color='#212529', border_line_alpha=0.8,
                        background_fill_color='white', background_fill_alpha=1.0)
    fig.add_layout(strike_label)
    fig.add_layout(strike)
    st.bokeh_chart(fig, use_container_width=True)

with tab2:
    pass