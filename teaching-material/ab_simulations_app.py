import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.patches as mpatches
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import NormalIndPower
import matplotlib.ticker as mticker  # Import PercentFormatter
import random # Import random module


# Sidebar parameters
st.sidebar.header("Simulation Parameters")
st.sidebar.write("10,000 A/B tests will be simulated with the following parameters:")
n_obs = st.sidebar.number_input("Sample Size per Variant", min_value=100, max_value=1000000, value=10000, step=1000)
conv_control = st.sidebar.number_input("Baseline CVR", min_value=0.01, max_value=0.5, value=0.015, step=0.001, format="%.3f")
lift = st.sidebar.slider("True Lift", 0.01, 0.2, 0.05, 0.01, format="%.2f")
alpha = st.sidebar.slider("One-Sided Significance Level (α)", 0.01, 0.1, 0.05, 0.01)
#power = st.sidebar.slider("Power", 0.8, 1.0, 0.8, 0.01)

n_tests = 10000
conv_variant = conv_control * (1 + lift)
#effect_size = proportion_effectsize(expected_conv_variant, conv_control)  # Calculate effect size for variant population with convertion rate of 21% and control population with convertion rate of 20%
#n_obs = NormalIndPower().solve_power(effect_size=effect_size, alpha=alpha, nobs1=None, power=power, ratio=1.0, alternative='larger')
#n_obs = int(np.ceil(n_obs))  # Round up to nearest whole number

# Button to run the simulation
if st.sidebar.button("Run Simulation"):
    #st.write("🔄 Running A/B Test Simulations...")

    # Create DataFrame
    df = pd.DataFrame(columns=('n', 'obs_effect', 'conf_level', 'stat_sig', 'obs_control', 'obs_variant'))

    # Simulate A/B tests
    for _ in range(n_tests):
        # Sample from control and variant distributions
        success_A = np.random.binomial(n_obs, conv_control)
        success_B = np.random.binomial(n_obs, conv_variant)
        obs_control = success_A / n_obs
        obs_variant = success_B / n_obs

        # Perform hypothesis testing
        count = np.array([success_B, success_A])
        nobs = np.array([n_obs, n_obs])
        z_stat, p_value = proportions_ztest(count, nobs, alternative='larger')
        conf_level = 1-p_value
        stat_sig = 1 if p_value < alpha else 0
        obs_effect = (success_B / success_A) - 1

        # Save results
        record = [n_obs, obs_effect, conf_level, stat_sig, obs_control, obs_variant]
        df.loc[len(df)] = record

    # Compute summary statistics
    summary = { 
        "Total Tests": f"{df['n'].count():,}",  # No decimal places
        "Significant Tests": f"{int(df['stat_sig'].sum()):,}",  # No decimal places, thousand separator
        "Observed Power": f"{df['stat_sig'].mean():.1%}",  # Percent format
        "True Lift": f"{lift:.1%}",  
    }

    summary_df = pd.DataFrame([summary])

    # Display summary table
    #st.subheader("Summary of Simulated A/B Tests")
    st.subheader("📊 **Aggregated results from simulated tests:**")
    #st.write(f"\nThe simulations used a 'Real' Control CVR = {conv_control:.2%} and 'Real' Variant CVR = {conv_variant:.2%} ({lift:.0%} 'Real' Lift).")
    st.dataframe(summary)

    # --- Histogram Visualization ---
    st.subheader("Distribution of Observed Lifts")

    # Compute the mean of obs_effect
    mean_obs_effect = df["obs_effect"].mean()

    # Define color mapping: stat_sig=1 (Red), stat_sig=0 (Blue)
    color_map = {0: "red", 1: "blue"}

    # Create histogram
    fig = px.histogram(
        df, 
        x="obs_effect", 
        #marginal="rug",
        hover_data=df.columns,
        color='stat_sig',
        color_discrete_map=color_map, 
        template='plotly_dark',
        height=300
    )

    # Add vertical line for the mean
    fig.add_vline(
        x=mean_obs_effect, 
        line_dash="dash", 
        line_color="grey", 
        annotation_text=f"Mean: {mean_obs_effect:.2%}",
        annotation_position="top"
    )

    # Format x-axis as percentage
    fig.update_layout(
        xaxis=dict(
            tickformat=".0%",  # 🔹 Display values as percentages
            title="Observed Lift (%)"  # Update x-axis title
        )
    )

    # Display the Seaborn plot in Streamlit
    st.plotly_chart(fig)

    # --- Histogram Visualization of p-values ---
    st.subheader("Distribution of Confidence Levels")

    # Define color mapping: stat_sig=1 (Red), stat_sig=0 (Blue)
    color_map = {0: "red", 1: "blue"}

    fig = px.histogram(
    df, 
    x="conf_level", 
    hover_data=df.columns,
    color='stat_sig',
    color_discrete_map=color_map,  # 🔹 Ensure color consistency
    nbins=10,  # Ensures 10 bins
    height=300
    )

    # Adjust bins to ensure 1 is included
    fig.update_traces(
        xbins=dict(start=0, end=1.001, size=(1.0000001-0)/10)  # Slightly extend end to include 1
    )

     # Display the Seaborn plot in Streamlit
    st.plotly_chart(fig)