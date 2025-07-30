# pages/Statathon_Project.py

import streamlit as st
from statathon.statathon_app_pages import data_preview, feature_engineering, eda, intro, statathon, data_storytelling

def show():
    st.title("Fraud Detection Project (Statathon)")
    st.markdown("---") # Separator for visual clarity

    # --- Custom Sidebar Navigation for Statathon Project ---
    st.sidebar.title("Statathon Navigation")
    statathon_pages = {
        "📘 Introduction": intro, 
        "📊 Data Preview": data_preview,
        "🧪 Feature Engineering": feature_engineering,
        "📈 EDA": eda,
        "📖 Data Storytelling": data_storytelling
    }

    selection = st.sidebar.radio("Go to", list(statathon_pages.keys()))

    selected_page_module = statathon_pages[selection]

    # Call the appropriate show function
    if hasattr(selected_page_module, 'show_about_me'): # For about_me.py
        selected_page_module.show_about_me()
    elif hasattr(selected_page_module, 'show'): # For all other pages
        selected_page_module.show()
    else:
        st.error(f"Error: Page '{selection}' does not have a 'show' or 'show_about_me' function.")

if __name__ == "__main__":
    show()