import streamlit as st
import plotly.express as px
import pandas as pd
from scipy.stats import norm
import math
from statsmodels.stats.power import TTestIndPower
import plotly.graph_objects as go


def calculate_sample_size(p0, lift, sig_level, power, alternative):
  """
  Calculate the sample size required for a hypothesis test.
  Parameters:
  p0 (float): The baseline conversion rate.
  lift (float): The minimum detectable effect size.
  sig_level (float): The desired significance level (alpha).
  power (float): The desired statistical power.
  alternative (str): The alternative hypothesis ('two-sided', 'larger', or 'smaller').
  Returns:
  int: The calculated sample size.
  """
  p1 = p0 * (1+lift)
  pooled_p = (p0 + p1) / 2
  effect_size = np.abs(p1 - p0) / np.sqrt(pooled_p * (1 - pooled_p))
  n = TTestIndPower().solve_power(effect_size = effect_size, nobs1 = None, alpha = sig_level, power = power, ratio = 1, alternative = alternative)
  n = round(n)
  return n


def plot_sample_size(lifts, n_values):
  fig = go.Figure(data=go.Scatter(x=lifts, y=n_values))
  fig.update_layout(
    title='Sample size by lift',
    xaxis_title='Lift',
    yaxis_title='Sample size',
    xaxis = dict(tickformat = '.0%'),
    yaxis = dict(tickformat = ',')
  )
  st.plotly_chart(fig)


st.title('Sample size calculator')
p0 = st.number_input('Baseline conversion rate', value = 0.014, step=0.001)
lift = st.number_input('Lift', value = 0.05)
sig_level = st.number_input('Significance level', value = 0.05)
power = st.number_input('Power', value = 0.80)
alternative = st.selectbox('Alternative hypothesis', ['two-sided', 'larger', 'smaller'])


if st.button('Calculate'):
  n = calculate_sample_size(p0, lift, sig_level, power, alternative)


col1, col2 = st.columns(2)
col1.markdown(f'<h3>Lift</h3>', unsafe_allow_html=True)
col1.markdown(f'<h4>{format(lift, ".0%")}</h4>', unsafe_allow_html=True)
col2.markdown(f'<h3>Sample size</h3>', unsafe_allow_html=True)
col2.markdown(f'<h4>{format(n, ",")}</h4>', unsafe_allow_html=True)


lifts = np.arange(0.01, 0.2, 0.01)
n_values = [calculate_sample_size(p0, l, sig_level, power, alternative) for l in lifts]
plot_sample_size(lifts, n_values)
