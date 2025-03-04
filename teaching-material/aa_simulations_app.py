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


# --- Streamlit UI ---
st.title("A/A Tests Simulations")
st.write("A/A tests are used to validate the statistical power of the test setup by comparing two identical groups.")

# Sidebar parameters
st.sidebar.header("Simulation Parameters")
#st.sidebar.write("10,000 A/A tests will be simulated with the following parameters:")
n_tests = st.sidebar.number_input("Number of simulated A/A tests", min_value=10, max_value=10000, value=50, step=10)
n_obs = st.sidebar.number_input("Sample Size per Variant", min_value=100, max_value=1000000, value=10000, step=1000)
conv_control = st.sidebar.number_input("Baseline CVR", min_value=0.01, max_value=0.2, value=0.015, step=0.005, format="%.3f")
#mde = st.sidebar.slider("MDE (%)", 0.01, 0.2, 0.05, 0.01)
alpha = st.sidebar.slider("Two-sided Significance Level (α)", 0.01, 0.2, 0.05, 0.01)
#power = st.sidebar.slider("Power", 0.8, 1.0, 0.8, 0.01)

lift = 0.0
conv_variant = conv_control * (1 + lift)


# Button to run the simulation
if st.sidebar.button("Run Simulation"):
    #st.write("🔄 Running A/A Test Simulations...")

    # Create DataFrame
    df = pd.DataFrame(columns=('n', 'stat_sig', 'obs_effect', 'p-conf_level'))

    # Simulate A/A tests
    for k in range(n_tests):

        # sample 1 time from the control binomial distribution
        n, p = n_obs, conv_control
        tests = 1
        samples = np.random.binomial(n, p, tests)
        success_A = samples[0]
        trials_A = n

        # sample 1 time from the treatment variant binomial distribution
        n, p = n_obs, conv_variant
        tests = 1
        samples = np.random.binomial(n, p, tests)
        success_B = samples[0]
        trials_B = n

        # perform hypothesis testing
        count = np.array([success_B, success_A])
        nobs = np.array([trials_B, trials_A])
        z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')
        conf_level = 1-p_value
        stat_sig = 1 if p_value < alpha else 0
        obs_effect = (success_B / success_A) -1

        # save results into dataframe
        record = [n_obs, stat_sig, obs_effect, conf_level]
        df.loc[len(df)] = record


    # Compute summary statistics
    summary = {
        "Total Tests": f"{df['n'].count():,}",  # No decimal places
        "Significant Tests": f"{df['stat_sig'].sum():,}",  # No decimal places, thousand separator
        "False positives": f"{df['stat_sig'].mean():.2%}",  # Percent format
        "Avg Observed Lift": f"{df['obs_effect'].mean():.2%}",  # Percent format
    }

    # Convert to DataFrame for better display
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
    color_map = {1: "red", 0: "blue"}

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
    color_map = {1: "red", 0: "blue"}

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