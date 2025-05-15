import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import norm
from statsmodels.stats.proportion import proportions_ztest

st.set_page_config(page_title="FPR and Impact simulations") #, layout="wide")
st.title("🔬 FPR and Impact simulations")

# --- Sidebar Inputs for Sample Size Calculation ---
st.sidebar.header("🔢 Power Calculation Parameters")
test_type = st.sidebar.radio("Test Type", ["Two-sided", "One-sided"])
confidence_level = st.sidebar.slider("Confidence Level (1 - α)", 0.80, 0.99, 0.95, 0.01)
power = st.sidebar.slider("Power (1 - β)", 0.5, 0.99, 0.8, 0.01)
#daily_traffic = st.sidebar.number_input("Daily Traffic (Total)", min_value=100, value=10000, step=100)
#baseline_cvr = st.sidebar.number_input("Baseline CVR (%)", min_value=0.1, max_value=100.0, value=2.0, step=0.1) / 100
daily_traffic = 100000
baseline_cvr = 0.02
mde = st.sidebar.number_input("Minimum Detectable Effect (%)", min_value=0.1, max_value=100.0, value=5.0, step=0.1) / 100

# --- Z-values ---
alpha = 1 - confidence_level
z_alpha = norm.ppf(1 - alpha / (2 if test_type == "Two-sided" else 1))
z_beta = norm.ppf(power)

# --- Sample Size Formula ---
p1 = baseline_cvr
p2 = baseline_cvr * (1 + mde)
pooled_var = p1 * (1 - p1) + p2 * (1 - p2)
delta = abs(p2 - p1)
sample_size_per_variant = int(((z_alpha + z_beta)**2 * pooled_var) / (delta**2))
runtime_days = int(np.ceil((2 * sample_size_per_variant) / daily_traffic))

# --- Sample Size Output ---
#st.markdown("#### 📊 Sample Size & Runtime Estimate")
#sample_summary = {
#    "🧪 Sample Size per Variant": f"{sample_size_per_variant:,}",
#    "📆 Runtime (Days)": f"{runtime_days:,}"
#}
#st.dataframe(pd.DataFrame.from_dict(sample_summary, orient="index").T, use_container_width=True, hide_index=True)


# --- Enhanced Simulation: True Lift and A/B Test Outcome Breakdown ---
st.sidebar.header("🧪 False Positive/Negative Simulation")
lift_mode = st.sidebar.selectbox("True Lift Distribution", ["Distribution", "Naive"])

# --- Calculations ---
sim_sample_size = sample_size_per_variant
n_simulations = 10000
alt = 'two-sided' if test_type == "Two-sided" else 'larger'

if lift_mode == "Naive":
    null_threshold = 0
    win_rate = st.sidebar.slider("Win Rate (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0) / 100
    true_lift_input = st.sidebar.number_input("True Lift (%)", min_value=0.0, max_value=100.0, value=2.0, step=0.1) / 100
elif lift_mode == "Distribution":
    null_threshold = st.sidebar.slider("Null Threshold (%)", min_value=0.0, max_value=5.0, value=0.5, step=0.1) / 100
    dist_mean = st.sidebar.number_input("Mean (%)", min_value=-100.0, max_value=100.0, value=0.0, step=0.1) / 100
    dist_std = st.sidebar.number_input("Standard Deviation (%)", min_value=0.0, max_value=100.0, value=2.0, step=0.1) / 100


if st.sidebar.button("Run Hypothesis Test Risk Simulation"):

    sim_results = []

    for _ in range(n_simulations):
        # --- Generate true lift based on selected mode ---
        if lift_mode == "Naive":
            sim_lift = true_lift_input if np.random.rand() < win_rate else 0.0
        else:  # Distribution mode
            sim_lift = np.random.normal(loc=dist_mean, scale=dist_std)

        # --- Apply true lift to variant B ---
        p_variant_sim = baseline_cvr * (1 + sim_lift)

        # --- Simulate conversion ---
        conv_a = np.random.binomial(sim_sample_size, baseline_cvr)
        conv_b = np.random.binomial(sim_sample_size, p_variant_sim)
        rate_a = conv_a / sim_sample_size
        rate_b = conv_b / sim_sample_size
        obs_lift = (rate_b / rate_a) - 1 if rate_a > 0 else 0

        # --- Z-test ---
        z_stat, p_val = proportions_ztest(
            [conv_b, conv_a],
            [sim_sample_size, sim_sample_size],
            alternative=alt
        )
        stat_sig = int(p_val < alpha)

        sim_results.append((sim_lift, obs_lift, stat_sig))

    sim_df = pd.DataFrame(sim_results, columns=["true_lift", "obs_lift", "stat_sig"])


    # --- True Lift Summary ---
    total_sims = len(sim_df)

    true_lift_small_1 = sim_df[(sim_df["true_lift"] > -0.005) & (sim_df["true_lift"] <= 0.005)]
    true_lift_small_2 = sim_df[(sim_df["true_lift"] > -0.015) & (sim_df["true_lift"] <= 0.015)]
    true_lift_medium_1 = sim_df[(sim_df["true_lift"] > -0.03) & (sim_df["true_lift"] <= 0.03)]
    true_lift_medium_2 = sim_df[(sim_df["true_lift"] > -0.05) & (sim_df["true_lift"] <= 0.05)]
    true_lift_large = sim_df[(sim_df["true_lift"] > -0.1) & (sim_df["true_lift"] <= 0.1)]

    true_lift_positive = sim_df[sim_df["true_lift"] > 0]
    true_lift_above_1 = sim_df[sim_df["true_lift"] >= 0.01]
    true_lift_above_3 = sim_df[sim_df["true_lift"] >= 0.03]
    true_lift_above_5 = sim_df[sim_df["true_lift"] >= 0.05]
    true_lift_above_10 = sim_df[sim_df["true_lift"] >= 0.10]
    true_lift_pos_01 = true_lift_positive[true_lift_positive["true_lift"] <= 0.01]
    avg_true_lift_gt0 = true_lift_positive["true_lift"].mean()
    median_true_lift_gt0 = true_lift_positive["true_lift"].median()

    true_lift_summary = {
        "🔁 Total Simulations": f"{total_sims:,}",
       # "🔹 -0.5% < Lift ≤ 0.5%": f"{len(true_lift_small_1) / total_sims:.2%}",
       # "🔹 -1% < Lift ≤ 1%": f"{len(true_lift_small_2) / total_sims:.2%}",
       # "🔸 -3% < Lift ≤ 3%": f"{len(true_lift_medium_1) / total_sims:.2%}",
       # "🔸 -5% < Lift ≤ 5%": f"{len(true_lift_medium_2) / total_sims:.2%}",
       # "🔶 -10% < Lift ≤ 10%": f"{len(true_lift_large) / total_sims:.2%}",
        "✅ True Lift > 0": f"{len(true_lift_positive) / total_sims:.2%}",
       # "🧪 True Lift (0 < lift ≤ 1%)": f"{len(true_lift_pos_01) / total_sims:.2%}",
        "🎯 True Lift ≥ +1%": f"{len(true_lift_above_1) / total_sims:.2%}",
        "🎯 True Lift ≥ +3%": f"{len(true_lift_above_3) / total_sims:.2%}",
        "🚀 True Lift ≥ +5%": f"{len(true_lift_above_5) / total_sims:.2%}",
       # "🔥 True Lift ≥ 10%": f"{len(true_lift_above_10) / total_sims:.2%}",
        "📈 Avg True Lift (when > 0)": f"{avg_true_lift_gt0:.2%}",
        "📈 Median True Lift (when > 0)": f"{median_true_lift_gt0:.2%}"
    }

    st.markdown("#### 📊 True Lift Distribution Summary (% of Total Simulations)")
    st.dataframe(
        pd.DataFrame.from_dict(true_lift_summary, orient="index", columns=["% of Simulations"]),
        use_container_width=True
    )

    # st.markdown("#### 📉 Distribution of True Lift Across Defined Bands")

    # Compute percentages
    # pct_small_1 = len(true_lift_small_1) / total_sims
    # pct_small_2 = len(true_lift_small_2) / total_sims
    # pct_medium_1 = len(true_lift_medium_1) / total_sims
    # pct_medium_2 = len(true_lift_medium_2) / total_sims
    # pct_large = len(true_lift_large) / total_sims

    #fig = px.histogram(
    #    sim_df,
    #    x="true_lift",
    #    nbins=50,
    #    template="plotly_white",
    #    title="True Lift Distribution",
    #    labels={"true_lift": "True Lift"},
    #    opacity=0.8
    #)

    # Add shaded regions with percentage annotations
    # fig.add_vrect(
    #     x0=-0.005, x1=0.005,
    #     line_width=0, fillcolor="lightblue", opacity=0.3,
    #     annotation_text=f"{pct_small_1:.1%}", annotation_position="top left"
    # )
    # fig.add_vrect(
    #     x0=-0.01, x1=0.01,
    #     line_width=0, fillcolor="lightgreen", opacity=0.2,
    #     annotation_text=f"{pct_small_2:.1%}", annotation_position="top left"
    # )
    # fig.add_vrect(
    #     x0=-0.03, x1=0.03,
    #     line_width=0, fillcolor="gold", opacity=0.15,
    #     annotation_text=f"{pct_medium_1:.1%}", annotation_position="top left"
    # )
    # fig.add_vrect(
    #     x0=-0.05, x1=0.05,
    #     line_width=0, fillcolor="orange", opacity=0.1,
    #     annotation_text=f"{pct_medium_2:.1%}", annotation_position="top left"
    # )
    # fig.add_vrect(
    #     x0=-0.10, x1=0.10,
    #     line_width=0, fillcolor="red", opacity=0.05,
    #     annotation_text=f"{pct_large:.1%}", annotation_position="top left"
    # )

   # fig.update_layout(
    #    xaxis_tickformat=".0%",
    #    xaxis_title="True Lift",
    #    yaxis_title="Count",
    #    hovermode="x unified"
    #)

    #st.plotly_chart(fig, use_container_width=True)


    # --- A/B Test Outcome Summary with Percentages ---
    null_thresh = null_threshold
    total_sims = len(sim_df)

    # Masks
    is_statsig = sim_df["stat_sig"] == 1
    is_notsig = sim_df["stat_sig"] == 0
    true_lift_abs = sim_df["true_lift"].abs()
    obs_lift_abs = sim_df["obs_lift"].abs()
    obs_lift_pos = sim_df["obs_lift"] > 0
    true_lift_pos = sim_df["true_lift"] > 0
    obs_lift_neg = sim_df["obs_lift"] < 0
    true_lift_neg = sim_df["true_lift"] < 0


    #true_lift_pos = (sim_df["true_lift"] > 0).sum()
    true_lift_null = (true_lift_abs > null_thresh).sum()
    true_lift_null_pos = ((true_lift_abs > null_thresh) & true_lift_pos).sum()

    # Redefined categories
    statsig_results = is_statsig.sum()
    false_positives = ((true_lift_abs <= null_thresh) & is_statsig).sum()
    true_positives = ((true_lift_abs > null_thresh) & is_statsig).sum()
    false_negatives = ((true_lift_abs > null_thresh) & is_notsig).sum()
    true_negatives = ((true_lift_abs <= null_thresh) & is_notsig).sum()
    fpr = false_positives / statsig_results 
    power = true_positives / true_lift_null
    
    #avg_lift_tp = sim_df.loc[(true_lift_abs > null_thresh) & is_statsig, "obs_lift"].mean()
    avg_lift = sim_df.loc[is_statsig, "true_lift"].mean()
    avg_obs_lift = sim_df.loc[is_statsig, "obs_lift"].mean()
    sum_lift = sim_df.loc[is_statsig, "true_lift"].sum()

    # -- Winners: observed lift >0 --
    statsig_winners = (is_statsig & obs_lift_pos).sum()
    true_positive_winners = (((true_lift_abs > null_thresh)) & true_lift_pos & is_statsig & obs_lift_pos).sum()
    false_positive_winners = (((true_lift_abs <= null_thresh) | true_lift_neg)  & is_statsig & obs_lift_pos).sum()
    #false_negative_losers = (((true_lift_abs > null_thresh) & true_lift_pos) & (is_notsig | obs_lift_neg)).sum()
    #true_negative_losers = (((true_lift_abs <= null_thresh) | true_lift_neg) & (is_notsig | obs_lift_neg)).sum()
    fpr_winners = false_positive_winners / statsig_winners if statsig_winners > 0 else 0.0
    power_winners = true_positive_winners / true_lift_null_pos


     # -- Losers: observed lift <0 --
    statsig_losers = (is_statsig & obs_lift_neg).sum()
    true_positive_losers = (((true_lift_abs > null_thresh)) & true_lift_pos & is_statsig & obs_lift_neg).sum()
    false_positive_losers = (((true_lift_abs <= null_thresh) | true_lift_neg)  & is_statsig & obs_lift_neg).sum()
    #false_negative_losers = (((true_lift_abs > null_thresh) & true_lift_pos) & (is_notsig | obs_lift_neg)).sum()
    #true_negative_losers = (((true_lift_abs <= null_thresh) | true_lift_neg) & (is_notsig | obs_lift_neg)).sum()
    fpr_winners = false_positive_winners / statsig_winners if statsig_winners > 0 else 0.0
    power_winners = true_positive_winners / true_lift_null_pos

    
    # --- Table with Winners and Losers ---
    # --- Compute base masks ---
    total_sims = len(sim_df)
    null_thresh = null_threshold
    true_lift_abs = sim_df["true_lift"].abs()
    true_lift_pos = sim_df["true_lift"] > 0
    true_lift_neg = sim_df["true_lift"] < 0
    obs_lift_pos = sim_df["obs_lift"] > 0
    obs_lift_neg = sim_df["obs_lift"] < 0
    is_statsig = sim_df["stat_sig"] == 1
    is_notsig = sim_df["stat_sig"] == 0

    # --- True lifts that "should" be detected ---
    should_detect = true_lift_abs > null_thresh 
    total_should_detect = should_detect.sum()
    total_should_detect_w = (should_detect & (sim_df["true_lift"] > 0)).sum()
    total_should_detect_l = (should_detect & (sim_df["true_lift"] < 0)).sum()

    # --- Not significant lifts that "should not" be detected ---
    should_not_detect = true_lift_abs <= null_thresh 
    total_should_not_detect = should_not_detect.sum()
    total_should_not_detect_w = (should_not_detect | (sim_df["true_lift"] < 0)).sum()
    total_should_not_detect_l = (should_not_detect | (sim_df["true_lift"] > 0)).sum()

    # --- Masks per category ---
    overall_tp = (should_detect & is_statsig)
    overall_fp = ((~should_detect) & is_statsig)
    overall_tn = ((~should_detect) & is_notsig)
    overall_fn = (should_detect & is_notsig)

    winner_mask = obs_lift_pos & is_statsig
    loser_mask = obs_lift_neg & is_statsig

    tp_winners = should_detect & true_lift_pos & winner_mask
    fp_winners = (~should_detect | true_lift_neg) & winner_mask
    fn_winners = (should_detect & (sim_df["true_lift"] > 0)) & is_notsig


    tp_losers = should_detect & true_lift_neg & loser_mask
    fp_losers = (~should_detect | true_lift_pos) & loser_mask
    fn_losers = (should_detect & (sim_df["true_lift"] < 0)) & is_notsig    


    # --- Counts ---
    statsig_overall = is_statsig.sum()
    statsig_winners = winner_mask.sum()
    statsig_losers = loser_mask.sum()

    tp_overall = overall_tp.sum()
    fp_overall = overall_fp.sum()
    fn_overall = overall_fn.sum()

    tp_w = tp_winners.sum()
    fp_w = fp_winners.sum()
    fn_w = fn_winners.sum()

    tp_l = tp_losers.sum()
    fp_l = fp_losers.sum()
    fn_l = fn_losers.sum()

    # --- FPRs ---
    fpr_overall = fp_overall / statsig_overall if statsig_overall else 0
    fpr_w = fp_w / statsig_winners if statsig_winners else 0
    fpr_l = fp_l / statsig_losers if statsig_losers else 0

    # --- FPRs 2 ---
    fpr_overall_2 = fp_overall / total_should_not_detect if total_should_not_detect else 0
    fpr_w_2 = fp_w / total_should_not_detect_w if total_should_not_detect_w else 0
    fpr_l_2 = fp_l / total_should_not_detect_l if total_should_not_detect_l else 0

    # --- Power ---
    power_overall = tp_overall / total_should_detect if total_should_detect else 0
    power_w = tp_w / total_should_detect_w if total_should_detect else 0
    power_l = tp_l / total_should_detect_l if total_should_detect else 0

    # --- Build summary table ---
    summary_matrix = pd.DataFrame({
        "Overall": {
            "✅ Statsig Results": statsig_overall / total_sims,
            "✅ True Positives (% total)": tp_overall / total_sims,
            "🚨 False Positives (% total)": fp_overall / total_sims,
            "⚠️ FPR (FP / Statsig)": fpr_overall,
            "⚡ Power (TP / total true effects)": power_overall,
            #"🚨 False Negatives (% total)": fn_overall / total_sims,
            "🚨 False Positives (FP / total true effects)": fpr_overall_2
        },
        "🏆 Winners": {
            "✅ Statsig Results": statsig_winners / total_sims,
            "✅ True Positives (% total)": tp_w / total_sims,
            "🚨 False Positives (% total)": fp_w / total_sims,
            "⚠️ FPR (FP / Statsig)": fpr_w,
            "⚡ Power (TP / total true effects)": power_w,
            #"🚨 False Negatives": fn_w / total_sims,
            "🚨 False Positives (FP / total true effects)": fpr_w_2
        },
    #    "📉 Losers": {
    #        "✅ Statsig Results": statsig_losers / total_sims,
    #        "✅ True Positives": tp_l / total_sims,
    #        "🚨 False Positives": fp_l / total_sims,
    #        "⚠️ FPR (FP / Statsig)": fpr_l,
    #        "⚡ Power (TP / total true effects)": power_l,
    #        #"🚨 False Negatives (% total)": fn_l / total_sims,            
    #        "🚨 False Positives (FP / total true effects)": fpr_l_2
    #    }
    })

    # Format table
    summary_formatted = summary_matrix.applymap(lambda x: f"{x:.0%}" if isinstance(x, float) else x)

    st.markdown(f"#### 📊 A/B Test Simulations Summary")
    st.dataframe(summary_formatted, use_container_width=True)


    avg_lift = sim_df.loc[is_statsig, "true_lift"].mean()
    avg_obs_lift = sim_df.loc[is_statsig, "obs_lift"].mean()
    sum_lift = sim_df.loc[is_statsig, "true_lift"].sum()


    # --- Compute impact metrics ---
    avg_obs_lift = sim_df.loc[is_statsig, "obs_lift"].mean()
    avg_lift = sim_df.loc[is_statsig, "true_lift"].mean()
    sum_lift = sim_df.loc[is_statsig, "true_lift"].sum()

    # --- Build impact summary table with correct order ---
    impact_data = {
        "Tested Experiments": f"{(total_sims / 100):.0f}",
        "Rolled out exp.": f"{(statsig_overall / 100):.0f}", 
        "True Winners": f"{(tp_overall / 100):.0f}",
        "Avg obs. lift of rolled out": f"{avg_obs_lift:.2%}", 
        "Avg true lift of rolled out": f"{avg_lift:.2%}", 
        "Exageration ration (obs/true)": f"{(avg_obs_lift/avg_lift):.2f}",
        "Sum of true lift of rolled out":  f"{(sum_lift / 100):.2%}",
    }

    impact_df = pd.DataFrame.from_dict(impact_data, orient="index", columns=["Value"])

    # --- Display Impact Assessment Table ---
    st.markdown("#### 📊 Impact Assessment")
    st.dataframe(impact_df, use_container_width=True)
