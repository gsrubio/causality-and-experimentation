import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import binomtest

st.title("Coin Toss P-Value Simulation App")
st.write("Simulate coin tosses and visualize the distribution of observed proportions and p-values.")

# Sidebar inputs
st.sidebar.header("Simulation Parameters")
n_tests = st.sidebar.number_input("Number of replications", min_value=100, max_value=50000, value=10000, step=100)
n_obs = st.sidebar.number_input("Number of tosses per replication", min_value=10, max_value=1000000, value=100, step=10)
conv_control = st.sidebar.number_input("Fair Coin Probability (Heads)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
effect = st.sidebar.number_input("Coin Bias (additive)", min_value=-0.5, max_value=0.5, value=0.0, step=0.01)
alpha = st.sidebar.slider("Significance Level (α)", 0.01, 0.5, 0.05, 0.01)

# True probability of heads
p_heads = np.clip(conv_control + effect, 0, 1)

if st.sidebar.button("Run Simulation"):
    np.random.seed(84)
    df = pd.DataFrame(columns=("n", "stat_sig", "obs_prop", "p_value"))
    for _ in range(n_tests):
        # Simulate tosses
        heads = np.random.binomial(n_obs, p_heads)
        obs_prop = heads / n_obs
        # Binomial test for null hypothesis p=conv_control
        result = binomtest(heads, n_obs, conv_control, alternative='two-sided')
        p_value = result.pvalue
        stat_sig = 1 if p_value < alpha else 0
        record = [n_obs, stat_sig, obs_prop, p_value]
        df.loc[len(df)] = record

    # Summary
    summary = {
        "Total Replications": f"{df['n'].count():,}",
        "Significant Results (Biased coin)": f"{int(df['stat_sig'].sum()):,}",
        "False Alarm" if effect == 0 else "Power": f"{df['stat_sig'].mean():.2%}",
        "Avg Observed Proportion": f"{df['obs_prop'].mean():.2%}",
    }
    st.subheader("📊 Summary of Simulated Coin Tosses")
    st.dataframe(pd.DataFrame([summary]))

    # Histogram of observed proportions
    st.subheader("Distribution of Observed Proportions")
    fig1 = px.histogram(
        df,
        x="obs_prop",
        nbins=50,
        template="plotly_dark",
        height=300
    )
    fig1.add_vline(
        x=p_heads,
        line_dash="dash",
        line_color="red",
        annotation_text=f"True p: {p_heads:.2%}",
        annotation_position="top"
    )
    fig1.update_layout(xaxis=dict(tickformat=".0%", title="Observed Proportion (Heads)"))
    st.plotly_chart(fig1)

    # Histogram of p-values
    st.subheader("Distribution of P-values")
    fig2 = px.histogram(
        df,
        x="p_value",
        nbins=50,
        template="plotly_dark",
        height=300
    )
    fig2.add_vline(
        x=alpha,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Alpha: {alpha:.2f}",
        annotation_position="top"
    )
    fig2.update_layout(xaxis=dict(title="P-value"))
    st.plotly_chart(fig2)

    # Scatter plot: Observed Proportion vs P-value
    st.subheader("Observed Proportion vs P-value")
    fig3 = px.scatter(
        df,
        x="obs_prop",
        y="p_value",
        template="plotly_dark",
        height=400,
        opacity=0.7,
        hover_data={"obs_prop": ':.1%', "p_value": ':.2f'}
    )
    fig3.add_hline(
        y=alpha,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Alpha: {alpha:.2f}",
        annotation_position="top left"
    )
    fig3.add_vline(
        x=p_heads,
        line_dash="dash",
        line_color="grey",
        annotation_text=f"True p: {p_heads:.2%}",
        annotation_position="top right"
    )
    fig3.update_layout(
        xaxis=dict(tickformat=".0%", title="Observed Proportion (Heads)"),
        yaxis=dict(title="P-value", range=[0, 1])
    )
    st.plotly_chart(fig3)

    # Scatter plot: Observed Proportion vs Confidence
    st.subheader("Observed Proportion vs 'Confidence'")
    df["confidence"] = 1 - (df["p_value"] / 2)
    confidence_at_alpha = 1 - (alpha / 2)
    fig4 = px.scatter(
        df,
        x="obs_prop",
        y="confidence",
        template="plotly_dark",
        height=400,
        opacity=0.7,
        hover_data={"obs_prop": ':.1%', "confidence": ':.2f'}
    )
    fig4.add_hline(
        y=confidence_at_alpha,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Conf.: {confidence_at_alpha:.2f}",
        annotation_position="top left"
    )
    fig4.add_vline(
        x=p_heads,
        line_dash="dash",
        line_color="grey",
        annotation_text=f"True p: {p_heads:.2%}",
        annotation_position="top right"
    )
    fig4.update_layout(
        xaxis=dict(tickformat=".0%", title="Observed Proportion (Heads)"),
        yaxis=dict(title="'Confidence' (1 - p-value/2)", range=[0, 1.1])
    )
    st.plotly_chart(fig4) 