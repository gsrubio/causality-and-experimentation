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
    conf_level = st.slider("🎯 Confidence Level:", min_value=0.50, max_value=0.99, value=0.80, step=0.01)
    test_type = st.radio("Test Type:", options=["Two-sided", "One-sided"], index=0)
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

    t_stat = (sales_visitor_variant - sales_visitor_control) / se

    if test_type == "Two-sided":
        t_crit = t.ppf(1 - alpha / 2, df=dof)
        p_value = 2 * (1 - t.cdf(abs(t_stat), df=dof))
        ci_low = (sales_visitor_variant - sales_visitor_control - t_crit * se) / sales_visitor_control
        ci_upp = (sales_visitor_variant - sales_visitor_control + t_crit * se) / sales_visitor_control
    else:
        t_crit = t.ppf(1 - alpha, df=dof)
        p_value = 1 - t.cdf(t_stat, df=dof)
        ci_low = (sales_visitor_variant - sales_visitor_control - t_crit * se) / sales_visitor_control
        ci_upp = float("inf")

    # Display nicely formatted results
    st.subheader("📌 Summary Results")
    st.markdown(f"**Estimated Lift:** `{lift * 100:.2f}%`")
    if test_type == "Two-sided":
        st.markdown(f"**{(1 - alpha) * 100:.0f}% Confidence Interval ({test_type}):** `[ {ci_low * 100:.2f}%, {ci_upp * 100:.2f}% ]`")
    else:
        st.markdown(f"**{(1 - alpha) * 100:.0f}% Confidence Interval ({test_type}):** `[ {ci_low * 100:.2f}%, ∞ )`")
    st.markdown(f"**P-value ({test_type}):** `{(p_value):.4f}`")

    # Max possible CI range (for min alpha, assuming two-sided for bound range)
    t_max = t.ppf(1 - min_alpha / 2, df=dof)
    max_margin = t_max * se
    max_ci_low = (sales_visitor_variant - sales_visitor_control - max_margin) / sales_visitor_control
    max_ci_upp = (sales_visitor_variant - sales_visitor_control + max_margin) / sales_visitor_control
    x_range = [min(-0.05, max_ci_low * 1.1), max(0.05, max_ci_upp * 1.1)]

    # Plot
    y_val = "Sales/Visitor"
    x_val = (sales_visitor_variant / sales_visitor_control - 1)
    xerr = [[x_val - ci_low], [ci_upp - x_val if test_type == "Two-sided" else x_range[1] - x_val]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[x_val],
        y=[y_val],
        error_x=dict(
            type='data', symmetric=False,
            array=[xerr[1][0]], arrayminus=[xerr[0][0]]
        ),
        mode='markers',
        marker=dict(size=10, color='blue'),
        hovertemplate=(
            f"Metric: {y_val}<br>Estimated Lift: {x_val:.2%}<br>"
            f"CI Range: [{ci_low:.2%}, {'∞' if test_type == 'One-sided' else f'{ci_upp:.2%}'}]"
        )
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

    # Z-test alternative
    alt = 'two-sided' if test_type == "Two-sided" else 'larger'

    # Proportions z-test
    z_stat, p_value_cvr = proportions_ztest(
        count=[variant_conversions, control_conversions],
        nobs=[n_variant, n_control],
        alternative=alt
    )

    # CI calculation
    if test_type == "Two-sided":
        ci_low_abs, ci_upp_abs = confint_proportions_2indep(
            count1=variant_conversions,
            nobs1=n_variant,
            count2=control_conversions,
            nobs2=n_control,
            method='wald',
            compare='diff',
            alpha=alpha
        )
        ci_low_rel = ci_low_abs / control_cvr
        ci_upp_rel = ci_upp_abs / control_cvr
    else:
        # One-sided (only lower bound is meaningful for H1: lift > 0)
        ci_low_abs, _ = confint_proportions_2indep(
            count1=variant_conversions,
            nobs1=n_variant,
            count2=control_conversions,
            nobs2=n_control,
            method='wald',
            compare='diff',
            alpha=2 * alpha  # one-sided CI: double the alpha for correct lower bound
        )
        ci_low_rel = ci_low_abs / control_cvr
        ci_upp_rel = float("inf")

    # Display results
    st.subheader("📌 Summary Results")
    st.markdown(f"**Estimated Lift:** `{lift * 100:.2f}%`")
    ci_label = f"{(1 - alpha) * 100:.0f}% Confidence Interval ({test_type}):"
    if test_type == "Two-sided":
        st.markdown(f"**{ci_label}** `[ {ci_low_rel * 100:.2f}%, {ci_upp_rel * 100:.2f}% ]`")
    else:
        st.markdown(f"**{ci_label}** `[ {ci_low_rel * 100:.2f}%, ∞ )`")
    st.markdown(f"**P-value ({test_type}):** `{(p_value_cvr):.4f}`")

    # X range for fixed visual axis
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

    # Plot
    y_val = "CVR"
    x_val = (variant_cvr / control_cvr - 1)

    # Determine a visual upper CI bound when using one-sided test
    visual_ci_upp = max(x_range) if test_type == "One-sided" else ci_upp_rel

    if test_type == "Two-sided":
        xerr = [[x_val - ci_low_rel], [visual_ci_upp - x_val]]
    else:
        xerr = [[x_val - ci_low_rel], [visual_ci_upp - x_val]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[x_val],
        y=[y_val],
        error_x=dict(type='data', symmetric=False, array=[xerr[1][0]], arrayminus=[xerr[0][0]]),
        mode='markers',
        marker=dict(size=10, color='green'),
        hovertemplate=f"Metric: {y_val}<br>Estimated Lift: {x_val:.2%}<br>CI Range: [{ci_low_rel:.2%}, {'∞' if test_type == 'One-sided' else f'{ci_upp_rel:.2%}'}]"
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
