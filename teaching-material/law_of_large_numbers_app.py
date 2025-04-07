import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random

# Title
st.title("A/B Test Monitoring Dashboard Simulation")

# Sidebar controls
st.sidebar.header("Test Parameters")

#n_obs = st.sidebar.number_input("Total Observations", value=26000)
#variants = st.sidebar.selectbox("Number of Variants", options=[2, 3], index=1)
n_obs_group = st.sidebar.number_input("Visitors per day per variant", value=26000)
cvr_control = st.sidebar.number_input("Baseline CVR (Control)", value=0.015, step=0.001, format="%.4f")
lift = st.sidebar.slider("True Lift (Variant vs Control)", min_value=0.0, max_value=0.5, value=0.09, step=0.01)
days = st.sidebar.slider("Duration (Days)", min_value=10, max_value=100, value=40)

# Derived params
#n_obs_group = n_obs / variants
cvr_variant = cvr_control * (1 + lift)

# Seed option
#if st.sidebar.checkbox("Set random seed", value=False):
#    np.random.seed(42)

if st.sidebar.button("Run Simulation"):
    rdm_seed = random.randint(1, 1000)
    np.random.seed(rdm_seed)

    # Display random seed
    st.sidebar.write(f"Random Seed: {rdm_seed}")
    
    # Initialize DataFrame
    df = pd.DataFrame(columns=('day','n_control', 'conv_control', 'obs_cvr_control',
                            'n_variant', 'conv_variant', 'obs_cvr_variant',  'obs_lift'))

    # Sample day 1
    success_A = np.random.binomial(n_obs_group, cvr_control)
    success_B = np.random.binomial(n_obs_group, cvr_variant)
    trials_A = trials_B = n_obs_group
    cvr_A = success_A / trials_A
    cvr_B = success_B / trials_B
    obs_lift = (cvr_B - cvr_A) / cvr_A

    df.loc[len(df)] = [1, trials_A, success_A, cvr_A, trials_B, success_B, cvr_B, obs_lift]

    # Loop for subsequent days
    for j in range(1, days):
        success_A = np.random.binomial(n_obs_group, cvr_control)
        success_B = np.random.binomial(n_obs_group, cvr_variant)
        trials_A = trials_B = n_obs_group

        suc_A_day_ant = df.loc[df.day == j, 'conv_control'].iloc[0]
        suc_B_day_ant = df.loc[df.day == j, 'conv_variant'].iloc[0]
        trials_ant_A = df.loc[df.day == j, 'n_control'].iloc[0]
        trials_ant_B = df.loc[df.day == j, 'n_variant'].iloc[0]

        success_A += suc_A_day_ant
        success_B += suc_B_day_ant
        trials_A += trials_ant_A
        trials_B += trials_ant_B
        cvr_A = success_A / trials_A
        cvr_B = success_B / trials_B
        obs_lift = (cvr_B - cvr_A) / cvr_A

        df.loc[len(df)] = [j+1, trials_A, success_A, cvr_A, trials_B, success_B, cvr_B, obs_lift]


    # === Chart: CVR over time ===
    fig_cvr = go.Figure()

    fig_cvr.add_trace(go.Scatter(
        x=df['day'],
        y=df['obs_cvr_control'],
        mode='lines+markers',
        name='CVR - Control',
        line=dict(shape='linear', color='#916190'),
        hovertemplate="Control<br>Day: %{x}<br>CVR: %{y:.2%}<extra></extra>"
    ))

    fig_cvr.add_trace(go.Scatter(
        x=df['day'],    
        y=df['obs_cvr_variant'],
        mode='lines+markers',
        name='CVR - Variant',
        line=dict(shape='linear'),
        hovertemplate="Variant<br>Day: %{x}<br>CVR: %{y:.2%}<extra></extra>"
    ))

    # Add true CVR reference lines
    fig_cvr.add_hline(
        y=cvr_control,
        line=dict(color='grey', dash='dot'),
        annotation_text=f'True CVR Control = {cvr_control:.2%}',
        annotation_position='bottom right'
    )

    fig_cvr.add_hline(
        y=cvr_variant,
        line=dict(color='grey', dash='dot'),
        annotation_text=f'True CVR Variant = {cvr_variant:.2%}',
        annotation_position='bottom right'
    )

    # Vertical markers
   # fig_cvr.add_vline(x=7, line=dict(color='grey', dash='dot'),
   #                 annotation_text='End of First Week',
   #                 annotation_position='top right')

   # fig_cvr.add_vline(x=17, line=dict(color='grey', dash='dot'),
   #                 annotation_text='End of Experiment',
   #                 annotation_position='top right')

    fig_cvr.update_layout(
        title='Observed CVR Over Time by variant',
        xaxis_title='Day',
        yaxis_title='Conversion Rate (CVR)',
        yaxis_tickformat='.1%',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(x=1.05, y=1),
        margin=dict(r=100),
    )

    st.plotly_chart(fig_cvr, use_container_width=True)

    # === Chart: Observed Lift ===
    fig_lift = go.Figure()

    fig_lift.add_trace(go.Scatter(
        x=df['day'],
        y=df['obs_lift'],
        mode='lines+markers',
        name='Observed Lift',
        line=dict(shape='linear', color='purple')
    ))

    fig_lift.add_hline(
        y=lift,
        line=dict(color='gray', dash='dash'),
        annotation_text=f'True Lift = {lift:.1%}',
        annotation_position='top right'
    )

    #fig_lift.add_vline(x=7, line=dict(color='grey', dash='dot'),
    #                annotation_text='End of First Week',
    #                annotation_position='top right')

    #fig_lift.add_vline(x=17, line=dict(color='grey', dash='dot'),
    #                annotation_text='End of Experiment',
    #                annotation_position='top right')

    fig_lift.update_layout(
        title='Observed Lift Over Time',
        xaxis_title='Day',
        yaxis_title='Observed Lift',
        yaxis_tickformat='.1%',
        plot_bgcolor='rgba(0,0,0,0)',   
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(x=1.05, y=1),
        margin=dict(r=100)
    )

    st.plotly_chart(fig_lift, use_container_width=True)
