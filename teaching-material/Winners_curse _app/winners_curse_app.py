import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.stats.proportion import proportions_ztest

# --- Streamlit UI ---
st.title("A/B Test Simulation and Analysis")

# Sidebar parameters
st.sidebar.header("Simulation Parameters")
n_tests = st.sidebar.number_input("Number of A/B Tests", min_value=100, max_value=100000, value=10000, step=1000)
n_obs = st.sidebar.number_input("Sample Size per Variant", min_value=1000, max_value=50000, value=10000, step=1000)
conv_control = st.sidebar.slider("Baseline Conversion Rate", 0.01, 0.5, 0.2, 0.01)
lift = st.sidebar.slider("Expected Lift (%)", 0.01, 0.5, 0.05, 0.01)
alpha = st.sidebar.slider("Significance Level (α)", 0.01, 0.1, 0.05, 0.01)

conv_variant = conv_control * (1 + lift)

# Button to run the simulation
if st.sidebar.button("Run Simulation"):
    st.write("🔄 Running A/B Test Simulations...")

    # Create DataFrame
    df = pd.DataFrame(columns=('n', 'stat_sig', 'obs_effect', 'effect', 'p-value', 'exageration_ratio'))

    # Simulate A/B tests
    for _ in range(n_tests):
        # Sample from control and variant distributions
        success_A = np.random.binomial(n_obs, conv_control)
        success_B = np.random.binomial(n_obs, conv_variant)

        # Perform hypothesis testing
        count = np.array([success_B, success_A])
        nobs = np.array([n_obs, n_obs])
        z_stat, p_value = proportions_ztest(count, nobs, alternative='larger')
        stat_sig = 1 if p_value < alpha else 0
        obs_effect = (success_B / success_A) - 1
        effect = obs_effect if stat_sig == 1 else None
        exageration = effect / ((conv_variant - conv_control) / conv_control) if stat_sig == 1 else None

        # Save results
        record = [n_obs, stat_sig, obs_effect, effect, p_value, exageration]
        df.loc[len(df)] = record

    # Aggregate results
    df_grouped = df.groupby('n').agg({
        'stat_sig': 'mean',
        'p-value': 'median',
        'obs_effect': 'median',
        'effect': 'median',
        'exageration_ratio': 'median'
    }).rename(columns={'stat_sig': 'power', 'obs_effect': 'avg_obs_effect', 'effect': 'avg_obs_effect_stat_sig'}).reset_index()

    # Display summary table
    st.subheader("Summary of Simulated A/B Tests")
    st.write("📊 **Aggregated results from simulated tests:**")
    st.dataframe(df_grouped)

    # --- Visualization ---
    st.subheader("Distribution of Observed Effects")

    # Compute the mean observed effect for stat_sig == 1
    mean_effect_sig = df[df['stat_sig'] == 1]['obs_effect'].mean()

    # Create the histogram figure
    fig = go.Figure()

    # Define colors
    colors = {0: "rgba(0, 100, 255, 0.5)", 1: "rgba(255, 50, 50, 0.6)"}

    # Add histogram for stat_sig == 0 (Not Significant)
    fig.add_trace(go.Histogram(
        x=df[df['stat_sig'] == 0]['obs_effect'],
        name="Not Significant (p ≥ 0.05)",
        marker=dict(color=colors[0], line=dict(color='blue', width=1.2)),
        opacity=0.6,
        hovertemplate="<b>Category:</b> Not Significant<br>"
                      "<b>Bin Range:</b> %{x}<br>"
                      "<b>Count:</b> %{y}<extra></extra>"
    ))

    # Add histogram for stat_sig == 1 (Significant)
    fig.add_trace(go.Histogram(
        x=df[df['stat_sig'] == 1]['obs_effect'],
        name="Significant (p < 0.05)",
        marker=dict(color=colors[1], line=dict(color='red', width=1.2)),
        opacity=0.7,
        hovertemplate="<b>Category:</b> Significant<br>"
                      "<b>Bin Range:</b> %{x}<br>"
                      "<b>Count:</b> %{y}<extra></extra>"
    ))

    # Add vertical line for the mean observed effect (Significant category only)
    fig.add_shape(
        dict(
            type="line",
            x0=mean_effect_sig, x1=mean_effect_sig,
            y0=0, y1=1,  # Scale to full height
            xref='x', yref='paper',
            line=dict(color="black", width=2, dash="dash")
        )
    )

    # Add annotation for the vertical line, including the numeric value
    fig.add_annotation(
        x=mean_effect_sig,
        y=0.95,  # Position at 95% height
        xref="x",
        yref="paper",
        text=f"<b>Avg Observed Effect (Significant):</b><br>{mean_effect_sig:.4f}",
        showarrow=True,
        arrowhead=2,
        ax=40,  # Offset annotation
        ay=-40,
        font=dict(size=12, color="black"),
        bgcolor="rgba(255, 255, 255, 0.8)",  # Light background for readability
        bordercolor="black",
        borderwidth=1
    )

    # Update layout
    fig.update_layout(
        width=900,
        height=550,
        title=dict(text="Histogram of Observed Effects", font=dict(size=20, family="Arial Bold"), x=0.5),
        xaxis=dict(title="Observed Effect", title_font=dict(size=16), tickfont=dict(size=12), showgrid=True),
        yaxis=dict(title="Count", title_font=dict(size=16), tickfont=dict(size=12), showgrid=True),
        plot_bgcolor="white",
        paper_bgcolor="white",
        bargap=0.1,
        barmode="overlay",
        legend=dict(title="Statistical Significance", font=dict(size=14), bgcolor="rgba(240,240,240,0.8)")
    )

    # Display histogram
    st.plotly_chart(fig)
