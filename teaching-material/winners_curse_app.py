import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from statsmodels.stats.proportion import proportions_ztest

# --- Streamlit UI ---
st.title("A/B Test Simulation and Analysis")

# Sidebar parameters
st.sidebar.header("Simulation Parameters")
n_tests = st.sidebar.number_input("Number of A/B Tests", min_value=1, max_value=10000, value=10000, step=1000)
n_obs = st.sidebar.number_input("Sample Size per Variant", min_value=100, max_value=1000000, value=10000, step=1000)
conv_control = st.sidebar.slider("Baseline Conversion Rate", 0.01, 0.5, 0.2, 0.01)
alpha = st.sidebar.slider("Significance Level (α)", 0.01, 0.1, 0.05, 0.01)
lift = st.sidebar.slider("Real Lift (%)", 0.01, 0.5, 0.05, 0.01)

conv_variant = conv_control * (1 + lift)

# Button to run the simulation
if st.sidebar.button("Run Simulation"):
    st.write("🔄 Running A/B Test Simulations...")

    # Create DataFrame
    df = pd.DataFrame(columns=('n', 'obs_effect', 'effect', 'p-value', 'stat_sig', 'exageration_ratio'))

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
        record = [n_obs, obs_effect, effect, p_value, stat_sig, exageration]
        df.loc[len(df)] = record

    # Compute summary statistics
    summary = {
        "Total Tests": f"{df['n'].count():,}",  # No decimal places
        "Significant Tests": f"{df['stat_sig'].sum():,}",  # No decimal places, thousand separator
        "Observed Power": f"{df['stat_sig'].mean():.2%}",  # Percent format
        "Avg Observed Lift (Signif. Tests)": f"{df.loc[df['stat_sig'] == 1, 'effect'].mean():.2%}",  # Percent format
        "Avg Exaggeration Ratio": f"{df['exageration_ratio'].mean():.2f}"  # Two decimal places
    }
    summary_df = pd.DataFrame([summary])

    # Display summary table
    st.subheader("Summary of Simulated A/B Tests")
    st.write("📊 **Aggregated results from simulated tests:**")
    st.dataframe(summary)

    # --- Insert Scatter Plot ---
    st.subheader("Observed Lift by Experiment")
    
    fig = px.scatter(
        df.head(100),
        x=df.index[:100],
        y='obs_effect',
        color=df['stat_sig'][:100].astype(str),  # Ensure categorical mapping
        color_discrete_map={"0": "blue", "1": "red"},  # Inverted colors
        category_orders={"stat_sig": ["0", "1"]},  # Ensure order
        labels={"stat_sig": "Stat Sig", "x": "Experiment #", "obs_effect": "Lift"},  # Rename labels
        template='plotly_dark'
    )
    # Add title
    fig.update_layout(title_text="First 100 Experiment Results", title_x=0.5)  # Center-align title

    st.plotly_chart(fig)  # Display Plotly scatter plot in Streamlit

    # --- Histogram Visualization ---
    st.subheader("Distribution of Observed Effects")

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
    ax.annotate(f"Real Lift: {lift:.4f}", xy=(lift, ax.get_ylim()[1] * 0.9), 
                xytext=(-50, 10), textcoords="offset points",
                fontsize=12, color="black", bbox=dict(facecolor="lightgray", edgecolor="black", boxstyle="round,pad=0.3"),
                arrowprops=dict(arrowstyle="->", color="black"))

    # Add annotation for Mean Observed Effect
    ax.annotate(f"Avg Obs. Effect (Signif.): {mean_effect_sig:.4f}", xy=(mean_effect_sig, ax.get_ylim()[1] * 0.9), 
                xytext=(50, 10), textcoords="offset points",
                fontsize=12, color="black", bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
                arrowprops=dict(arrowstyle="->", color="black"))

    # Customize aesthetics
    ax.set_title("Histogram of Observed Lifts", fontsize=18, fontweight="bold", loc="left")
    ax.set_xlabel("Observed Lift", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)

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
