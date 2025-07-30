# AttentionSpanFinalProject/screen_time_app_pages/intro_screen_time.py

import streamlit as st

def show():
    st.subheader("Unveiling Attention Spans in the Digital Age")
    st.markdown("""
    This project is an attempt towards explaining the phenomenon of shortening attention spans in both children and adults,
    using insights derived from 2 comprehensive survey datasets found on Kaggle. In an increasingly digital world, understanding
    how screen time habits, device usage, and various activities impact our ability to focus is crucial.

    Through a series of interactive visualizations, we explore correlations between screen time, age groups,
    productivity, and attention metrics. Our goal is to shed light on potential trends and provide data-driven
    perspectives on this contemporary challenge.
    """)
    st.info("💡 **Play with the interactive elements and navigate through the pages in the sidebar for the best experience!**")

if __name__ == "__main__":
    show()