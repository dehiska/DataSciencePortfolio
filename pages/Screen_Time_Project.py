# pages/Screen_Time_Project.py

import streamlit as st
from AttentionSpanFinalProject.screen_time_app_pages import (
    intro_screen_time,
    data_preview_screen_time, 
    feature_engineering_screen_time, 
    eda_screen_time,
    data_storytelling_screen_time
)

def show():
    st.title("⏰ Screen Time Habits Analysis")
    st.markdown("---") # Separator

    # --- Custom Sidebar Navigation for Screen Time Project ---
    st.sidebar.title("Screen Time Navigation")
    screen_time_pages = {
        "📘 Introduction": intro_screen_time,
        "📊 Data Preview": data_preview_screen_time,
        "🧪 Feature Engineering": feature_engineering_screen_time,
        "📈 EDA": eda_screen_time,
        "📖 Data Storytelling": data_storytelling_screen_time
    }

    selection = st.sidebar.radio("Go to", list(screen_time_pages.keys()))

    selected_page_module = screen_time_pages[selection]

    if hasattr(selected_page_module, 'show'):
        selected_page_module.show()
    else:
        st.error(f"Error: Page '{selection}' does not have a 'show' function.")

if __name__ == "__main__":
    show()