import streamlit as st
import math
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t
from scipy.stats import norm
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep

st.set_page_config(page_title="Lift & Confidence Interval Calculator", layout="wide")
st.title("📈 Lift and Confidence Interval Calculator")

with st.sidebar:
    metric = st.selectbox("📊 Select the metric:", ["CVR (Conversion Rate)", "Sales per Visitor"])
    conf_level = st.slider("🎯 Confidence Level (2-sided):", min_value=0.60, max_value=0.99, value=0.80, step=0.01)
    alpha = 1 - conf_level
    min_alpha = 0.01  # for fixed axis bounds

if metric == "Sales per Visitor":
    with st.sidebar:
        sales_visitor_control = st.number_input("Control Sales/Visitor", value=2.25)
        lift = st.number_input("Lift (%)", value=5.4) / 100
        std_control = st.number_input("Control Std Dev", value=19.5)
        std_variant = st.number_input("Variant Std Dev", value=19.5)
        n_control = st.number_input("Control Sample Size", value=115000)
        n_variant = st.number_input("Variant Sample Size", value=115000)

    sales_visitor_variant = sales_visitor_control * (1 + lift)

    # CI calculation
    se = math.sqrt((std_variant**2 / n_variant) + (std_control**2 / n_control))
    numerator = (std_variant**2 / n_variant + std_control**2 / n_control) ** 2
    denominator = (
        ((std_variant**2 / n_variant) ** 2) / (n_variant - 1) +
        ((std_control**2 / n_control) ** 2) / (n_control - 1)
    )
    dof = numerator / denominator
    t_crit = t.ppf(1 - alpha / 2, df=dof)
    ci_margin = t_crit * se

    abs_diff = sales_visitor_variant - sales_visitor_control
    ci_low = (abs_diff - ci_margin) / sales_visitor_control
    ci_upp = (abs_diff + ci_margin) / sales_visitor_control

    t_stat = abs_diff / se
    p_value = 2 * (1 - t.cdf(abs(t_stat), df=dof))  # two-sided p-value

    # Display nicely formatted results
    st.subheader("📌 Summary Results")
    st.markdown(f"**Estimated Lift:** `{lift * 100:.2f}%`")
    st.markdown(f"**{(1 - alpha) * 100:.0f}% Confidence Interval (2-sided):** `[ {ci_low * 100:.2f}%, {ci_upp * 100:.2f}% ]`")
    #st.markdown(f"**Confidence Level (variant > control):** `{(1-p_value)*100:.2f}%`")
    st.markdown(f"**P-value (2-sided):** `{(p_value)*100:.2f}%`")

    # Compute maximum possible CI range (for min alpha)
    t_max = t.ppf(1 - min_alpha / 2, df=dof)
    max_margin = t_max * se
    max_ci_low = (abs_diff - max_margin) / sales_visitor_control
    max_ci_upp = (abs_diff + max_margin) / sales_visitor_control
    x_range = [min(-0.05, max_ci_low * 1.1), max(0.05, max_ci_upp * 1.1)]

    # Plotly visualization
    y_val = "Sales/Visitor"
    x_val = (sales_visitor_variant / sales_visitor_control - 1)
    xerr = [[x_val - ci_low], [ci_upp - x_val]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[x_val],
        y=[y_val],
        error_x=dict(type='data', symmetric=False, array=[xerr[1][0]], arrayminus=[xerr[0][0]]),
        mode='markers',
        marker=dict(size=10, color='blue'),
        hovertemplate=f"Metric: {y_val}<br>Estimated Lift: {x_val:.2%}<br>CI Range: [{ci_low:.2%}, {ci_upp:.2%}]"
    ))
    fig.add_shape(type='line', x0=0, x1=0, y0=-1, y1=2, line=dict(dash='dash', color='gray'))
    fig.update_layout(title="Lift with Confidence Interval",
                      xaxis_title="% Lift",
                      yaxis_title="",
                      xaxis_tickformat=".0%",
                      xaxis_range=x_range,
                      showlegend=False,
                      height=300)
    st.plotly_chart(fig)

elif metric == "CVR (Conversion Rate)":
    with st.sidebar:
        control_cvr = st.number_input("Control CVR (%)", value=1.52) / 100
        lift = st.number_input("Lift (%)", value=5.2) / 100
        n_control = st.number_input("Control Sample Size", value=115000)
        n_variant = st.number_input("Variant Sample Size", value=115000)

    variant_cvr = control_cvr * (1 + lift)
    control_conversions = int(round(control_cvr * n_control))
    variant_conversions = int(round(variant_cvr * n_variant))

    ci_low_abs, ci_upp_abs = confint_proportions_2indep(
        count1=variant_conversions,
        nobs1=n_variant,
        count2=control_conversions,
        nobs2=n_control,
        method='wald',
        compare='diff',
        alpha=alpha
    )

    abs_diff = variant_cvr - control_cvr
    ci_low_rel = ci_low_abs / control_cvr
    ci_upp_rel = ci_upp_abs / control_cvr

    z_stat, p_value_cvr = proportions_ztest(
        count=[variant_conversions, control_conversions],
        nobs=[n_variant, n_control],
        alternative='two-sided'  # two-sided p-value; use 'larger' for 1-sided p-value
    )

    st.subheader("📌 Summary Results")
    st.markdown(f"**Estimated Lift:** `{lift * 100:.2f}%`")
    st.markdown(f"**{(1 - alpha) * 100:.0f}% Confidence Interval (2-sided):** `[ {ci_low_rel * 100:.2f}%, {ci_upp_rel * 100:.2f}% ]`")
    #st.markdown(f"**Confidence Level (variant > control):** `{(1-p_value_cvr)*100:.2f}%`")
    st.markdown(f"**P-value (2-sided):** `{(p_value_cvr)*100:.2f}%`")

    # Max range for axis (fixed x-range based on min alpha)
    ci_low_abs_max, ci_upp_abs_max = confint_proportions_2indep(
        count1=variant_conversions,
        nobs1=n_variant,
        count2=control_conversions,
        nobs2=n_control,
        method='wald',
        compare='diff',
        alpha=min_alpha
    )
    max_ci_low_rel = ci_low_abs_max / control_cvr
    max_ci_upp_rel = ci_upp_abs_max / control_cvr
    x_range = [min(-0.05, max_ci_low_rel * 1.1), max(0.05, max_ci_upp_rel * 1.1)]

    # Plotly visualization
    y_val = "CVR"
    x_val = (variant_cvr / control_cvr - 1)
    xerr = [[x_val - ci_low_rel], [ci_upp_rel - x_val]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[x_val],
        y=[y_val],
        error_x=dict(type='data', symmetric=False, array=[xerr[1][0]], arrayminus=[xerr[0][0]]),
        mode='markers',
        marker=dict(size=10, color='green'),
        hovertemplate=f"Metric: {y_val}<br>Estimated Lift: {x_val:.2%}<br>CI Range: [{ci_low_rel:.2%}, {ci_upp_rel:.2%}]"
    ))
    fig.add_shape(type='line', x0=0, x1=0, y0=-1, y1=2, line=dict(dash='dash', color='gray'))
    fig.update_layout(title="Lift with Confidence Interval",
                      xaxis_title="% Lift",
                      yaxis_title="",
                      xaxis_tickformat=".0%",
                      xaxis_range=x_range,
                      showlegend=False,
                      height=300)
    st.plotly_chart(fig)
