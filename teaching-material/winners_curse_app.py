import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.patches as mpatches
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import NormalIndPower
import matplotlib.ticker as mticker  # Import PercentFormatter
import random # Import random module


# --- Streamlit UI ---
st.title("Winner's Curse - A/B Test Simulations")
st.write("Winner's curse is a phenomenon where 'winning variants' in underpowered experiments tend to show inflated lifts.")

# Sidebar parameters
st.sidebar.header("Simulation Parameters")
st.sidebar.write("10,000 A/B tests will be simulated with the following parameters:")
#n_obs = st.sidebar.number_input("Sample Size per Variant", min_value=100, max_value=1000000, value=10000, step=1000)
#conv_control = st.sidebar.slider("Baseline CVR", 0.01, 0.5, 0.2, 0.01)
mde = st.sidebar.slider("MDE (%)", 0.01, 0.2, 0.05, 0.01)
#alpha = st.sidebar.slider("Significance Level (α)", 0.01, 0.1, 0.05, 0.01)
power = st.sidebar.slider("Power", 0.8, 1.0, 0.8, 0.01)

n_tests = 10000
conv_control = 0.02
lift = mde/random.uniform(1.5, 3)
conv_variant = conv_control * (1 + lift)
expected_conv_variant = conv_control * (1 + mde)
alpha = 0.1
effect_size = proportion_effectsize(expected_conv_variant, conv_control)  # Calculate effect size for variant population with convertion rate of 21% and control population with convertion rate of 20%
n_obs = NormalIndPower().solve_power(effect_size=effect_size, alpha=alpha, nobs1=None, power=power, ratio=1.0, alternative='larger')
n_obs = int(np.ceil(n_obs))  # Round up to nearest whole number

# Button to run the simulation
if st.sidebar.button("Run Simulation"):
    #st.write("🔄 Running A/B Test Simulations...")

    # Create DataFrame
    df = pd.DataFrame(columns=('n', 'obs_effect', 'effect', 'p-value', 'stat_sig', 'exageration_ratio', 'obs_control', 'obs_variant'))

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
        stat_sig = 1 if p_value < alpha else 0
        obs_effect = (success_B / success_A) - 1
        effect = obs_effect if stat_sig == 1 else None
        exageration = effect / ((conv_variant - conv_control) / conv_control) if stat_sig == 1 else None

        # Save results
        record = [n_obs, obs_effect, effect, p_value, stat_sig, exageration, obs_control, obs_variant]
        df.loc[len(df)] = record

    # Compute summary statistics
    summary = { 
        "Total Tests": f"{df['n'].count():,}",  # No decimal places
        "Significant Tests": f"{int(df['stat_sig'].sum()):,}",  # No decimal places, thousand separator
        "Observed Power": f"{df['stat_sig'].mean():.1%}",  # Percent format
        "True Lift": f"{lift:.1%}",  
        "Avg Observed Lift (Signif. Exp.)": f"{df.loc[df['stat_sig'] == 1, 'effect'].mean():.1%}",  # Percent format
        "Avg Exaggeration Ratio": f"{df['exageration_ratio'].mean():.2f}"  # Two decimal places
    }
    summary_df = pd.DataFrame([summary])

    # Display summary table
    #st.subheader("Summary of Simulated A/B Tests")
    st.subheader("📊 **Aggregated results from simulated tests:**")
    #st.write(f"\nThe simulations used a 'Real' Control CVR = {conv_control:.2%} and 'Real' Variant CVR = {conv_variant:.2%} ({lift:.0%} 'Real' Lift).")
    st.dataframe(summary)

    # --- Histogram Visualization ---
    st.subheader("Distribution of Observed Lifts")

    # Compute the mean observed effect for stat_sig == 1
    mean_effect_sig = df[df['stat_sig'] == 1]['obs_effect'].mean()

    # Define colors
    colors = {0: "#FF3232", 1: "#0064FF"}  # Red for Not Significant, Blue for Significant

    # Create Seaborn figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot a single histogram with segmentation by `stat_sig`
    sns.histplot(df, x="obs_effect", bins=50, hue="stat_sig", multiple="stack", 
                 palette=colors, edgecolor='black', alpha=0.7, ax=ax)

    # Add vertical line for Mean Observed Effect (Significant)
    ax.axvline(mean_effect_sig, color="black", linestyle="dashed", linewidth=2)

    # Add vertical line for Real Lift
    ax.axvline(lift, color="black", linestyle="dotted", linewidth=2)

    # Add annotation for Real Lift
    ax.annotate(f"True Lift: {lift:.1%}", xy=(lift, ax.get_ylim()[1] * 0.9), 
                xytext=(-50, 10), textcoords="offset points",
                fontsize=12, color="black", bbox=dict(facecolor="lightgray", edgecolor="black", boxstyle="round,pad=0.3"),
                arrowprops=dict(arrowstyle="->", color="black"))

    # Add annotation for Mean Observed Effect
    ax.annotate(f"Avg Obs. Lift\n(Signif.): {mean_effect_sig:.1%}", xy=(mean_effect_sig, ax.get_ylim()[1] * 0.9), 
                xytext=(50, 10), textcoords="offset points",
                fontsize=12, color="black", bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
                arrowprops=dict(arrowstyle="->", color="black"))

    # Customize aesthetics
    #ax.set_title("Histogram of Observed Lifts", fontsize=18, fontweight="bold", loc="left")
    ax.set_xlabel("Observed Lift", fontsize=14)
    ax.set_ylabel("Exp. count", fontsize=14)

    # Format x-axis as percentage
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    # Create a custom legend with only the segments
    legend_patches = [
        mpatches.Patch(color=colors[1], label="Significant"),
        mpatches.Patch(color=colors[0], label="Not Significant")
    ]
    ax.legend(handles=legend_patches, title="Statistical Significance", fontsize=12, title_fontsize=14, loc="upper center", bbox_to_anchor=(0.8, 0.8), ncol=1)

    # Remove unnecessary spines and gridlines
    sns.despine()
    ax.grid(False)

    # Display the Seaborn plot in Streamlit
    st.pyplot(fig)

# --- Insert Scatter Plot ---
    st.subheader("Data Check - First 100 Experiment Results")
    st.write("Hover over each dot to see more details of each simulated experiment (first 100 only).")

    # Define colors for stat_sig categories
    color_map = {1: "blue", 0: "red"}

    # Subset first 50 experiments
    df_subset = df.head(50)

    # Create scatter plot traces for each category
    trace_0 = go.Scatter(
        x=df_subset[df_subset["stat_sig"] == 0]["obs_effect"],  # X is now Observed Lift
        y=df_subset[df_subset["stat_sig"] == 0].index,          # Y is now Experiment #
        mode="markers",
        marker=dict(color=color_map[0], size=8),
        name="Not Significant",
        customdata=df_subset[df_subset["stat_sig"] == 0][["obs_control", "obs_variant"]],
        hovertemplate="<b>Observed Lift</b>: %{x:.2%}<br>" +          # X is Observed Lift
                    "<b>Experiment #</b>: %{y}<br>" +               # Y is Experiment #
                    "<b>CVR Control</b>: %{customdata[0]:.2%}<br>" +  
                    "<b>CVR Variant</b>: %{customdata[1]:.2%}<br>" +
                    "<b>Stat Sig</b>: Not Significant<extra></extra>"
    )

    trace_1 = go.Scatter(
        x=df_subset[df_subset["stat_sig"] == 1]["obs_effect"],
        y=df_subset[df_subset["stat_sig"] == 1].index,
        mode="markers",
        marker=dict(color=color_map[1], size=8),
        name="Significant",
        customdata=df_subset[df_subset["stat_sig"] == 1][["obs_control", "obs_variant"]],
        hovertemplate="<b>Observed Lift</b>: %{x:.2%}<br>" +
                    "<b>Experiment #</b>: %{y}<br>" +
                    "<b>CVR Control</b>: %{customdata[0]:.2%}<br>" +  
                    "<b>CVR Variant</b>: %{customdata[1]:.2%}<br>" +
                    "<b>Stat Sig</b>: Significant<extra></extra>"
    )

    # Create the figure
    fig = go.Figure([trace_0, trace_1])

    # Update layout
    fig.update_layout(
        xaxis_title="Observed Lift (%)",    # X-axis is now Observed Lift
        yaxis_title="Experiment #",         # Y-axis is now Experiment #
        template="plotly_dark",
        legend_title="Stat Sig"
    )

    # Display Plotly scatter plot in Streamlit
    st.plotly_chart(fig)
