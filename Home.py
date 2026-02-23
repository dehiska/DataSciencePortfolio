# Home.py — navigation entry point for Denis Soulima's Portfolio

import streamlit as st

st.set_page_config(
    page_title="Denis Soulima's Data Science Portfolio",
    page_icon="👨‍💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation(
    {
        "": [
            st.Page("pages/Home_Page.py",               title="Home",                      default=True),
            st.Page("pages/About_Me.py",                title="About Me"),
        ],
        "Projects": [
            st.Page("pages/Craigslist_Car_Prices.py",   title="Craigslist Car Prices"),
            st.Page("pages/Data_Center_Analysis.py",    title="Data Center Analysis"),
            st.Page("pages/Data_Scientist_Assistant.py",title="Data Scientist Assistant"),
            st.Page("pages/Gmail_Janitor.py",           title="Gmail Janitor"),
            st.Page("pages/Screen_Time_Project.py",     title="Screen Time Project"),
            st.Page("pages/Seattle_House_Prices_Mapped.py", title="Seattle House Prices Mapped"),
            st.Page("pages/Statathon_Project.py",       title="Statathon Project"),
        ],
        "Legal": [
            st.Page("pages/Privacy_Policy.py",   title="Privacy Policy",   url_path="Privacy_Policy"),
            st.Page("pages/Terms_of_Service.py", title="Terms of Service", url_path="Terms_of_Service"),
        ],
    }
)

pg.run()
