import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

st.title("Coin Toss Simulator")

# User input for number of tosses
num_tosses = st.number_input("Number of tosses", min_value=1, value=100, step=1)

if st.button("Simulate Tosses"):
    # Simulate coin tosses (1 = heads, 0 = tails)
    tosses = np.random.randint(0, 2, size=num_tosses)
    num_heads = np.sum(tosses)
    pct_heads = num_heads / num_tosses * 100

    # Cumulative percentage of heads
    cumulative_heads = np.cumsum(tosses)
    toss_numbers = np.arange(1, num_tosses + 1)
    cumulative_pct_heads = cumulative_heads / toss_numbers * 100

    # Display results side by side
    col1, col2 = st.columns(2)
    col1.metric("Number of Heads", num_heads)
    col2.metric("% of Heads", f"{pct_heads:.2f}%")

    # Prepare DataFrame for plotting
    df = pd.DataFrame({
        "Toss #": toss_numbers,
        "% Heads": cumulative_pct_heads
    })

    # Plot line chart
    line = alt.Chart(df).mark_line().encode(
        x="Toss #",
        y=alt.Y("% Heads", scale=alt.Scale(domain=[0, 100]), title="% Heads")
    )
    fifty_line = alt.Chart(pd.DataFrame({"y": [50]})).mark_rule(
        strokeDash=[5,5], color="red"
    ).encode(y="y")

    st.altair_chart(line + fifty_line, use_container_width=True)
