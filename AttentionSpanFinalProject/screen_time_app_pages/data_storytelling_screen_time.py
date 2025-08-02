# AttentionSpanFinalProject/screen_time_app_pages/data_storytelling_screen_time.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# Import Graph modules as needed
from AttentionSpanFinalProject.graph_modules import Graph1, Graph2, Graph3, Graph4, Graph5
# Import the new common data loader
from AttentionSpanFinalProject.graph_modules.data_loader import load_and_preprocess_data

# Remove old data URLs if they are not used elsewhere in this file directly
# url1 = 'AttentionSpanFinalProject/data/data.csv'
# url2 = 'AttentionSpanFinalProject/data/screen_time.csv'

# Remove this cached function entirely, as it's replaced by data_loader.py
# @st.cache_data
# def load_and_prep_data_for_graphs():
#     df1_raw = pd.read_csv(url1)
#     df2_raw = pd.read_csv(url2)
#     screen_time_order = ['Less than 2', '2–4', '4–6', '6–8', '8-10', 'More than 10']
#     age_group_order = ['Below 18', '18–24', '25–34', '35–44', '45 and above']
#     df1_raw['Average Screen Time'] = pd.Categorical(df1_raw['Average Screen Time'], categories=screen_time_order, ordered=True)
#     df1_raw['Age Group'] = pd.Categorical(df1_raw['Age Group'], categories=age_group_order, ordered=True)
#     return df1_raw, df2_raw


def show():
    st.title("📖 Data Storytelling: Breaking Down Screen Time Habits")
    st.markdown("""
    This section presents the core findings of my Screen Time project through a series of interactive visualizations.
    We explore how screen time habits vary across demographics, activities, and devices, and investigate their potential
    impact on attention span and productivity.
    """)

    tab1_label, tab2_label, tab3_label, tab4_label, tab5_label = st.tabs([
        "Screentime for School",
        "Age vs. Screentime",
        "Kids vs. Adults in Attention Span",
        "Who most distracted?",
        "Are productivity apps helpful..."
    ])

    with tab1_label:
        st.subheader('Screen Time by Purpose and Day Type')
        st.markdown("""
        Explore how average screen time varies for educational vs. recreational purposes on weekdays, weekends, or both.
        Select your preferences below to see the corresponding visualization.
        """)

        purpose = st.radio('Select Purpose', ('Educational', 'Recreational', "Both"), key="ds_purpose_radio")
        time = st.radio("Select which days of the week", ("Weekdays", "Weekends", "Both"), key="ds_time_radio")

        if st.button("Generate Chart", key="ds_graph1_button"):
            if purpose == 'Educational' and time == 'Weekdays':
                st.pyplot(Graph1.makeGraph1_1())
            elif purpose == 'Educational' and time == 'Weekends':
                st.pyplot(Graph1.makeGraph1_2())
            elif purpose == 'Educational' and time == 'Both':
                st.pyplot(Graph1.makeGraph1_3())
            elif purpose == 'Recreational' and time == 'Weekdays':
                st.pyplot(Graph1.makeGraph1_4())
            elif purpose == 'Recreational' and time == 'Weekends':
                st.pyplot(Graph1.makeGraph1_5())
            elif purpose == 'Recreational' and time == 'Both':
                st.pyplot(Graph1.makeGraph1_6())
            elif purpose == 'Both' and time == 'Both':
                st.pyplot(Graph1.makeGraph1_7())
            elif purpose == 'Both' and time == 'Weekdays':
                st.pyplot(Graph1.makeGraph1_8())
            elif purpose == 'Both' and time == 'Weekends':
                st.pyplot(Graph1.makeGraph1_9())

            st.markdown("---")
            st.markdown("*Notice that the older the child is, the more hours they need to do schoolwork on screens over the weekend. The older the child is, the more they rely on the weekend to catch up to their schoolwork. Also 15 year old spend on average 2.27 hours on screens for recreational use in ADDITION to the average of 1.7 hours for educational use every day. *")


    with tab2_label: # Content for "Age vs. Screentime" (Graph2)
       st.subheader("Age Group vs. Average Screen Time (Heatmap)")
       st.markdown("""
       This heatmap visualizes the distribution of respondents across different age groups and their reported average screen time.
       It helps us identify which age demographics spend how much time on screens.
       """)
       fig2 = Graph2.makeGraph2()
       st.plotly_chart(fig2, use_container_width=True)
       st.markdown("---")
       st.markdown("*Notice that 34 of the 18-24 year old respondents spend 6-8 hours on screens per day!😲*")
       st.markdown("*18-24 year olds have the highest screen time usage, probably because they were the first to grow up with it? However, it is still interesting to see in data.*")


    with tab3_label: # Content for "Kids vs. Adults in Attention Span" (Graph3)
       st.subheader("Attention Span: Children vs. Adults")
       st.markdown("""
       Here, we compare the average attention span reported by individuals in the 'Below 18' age group (children) versus
       all adult age groups. This chart offers a direct comparison to see if there's a notable difference.
       """)
       fig3 = Graph3.makeGraph3()
       st.pyplot(fig3)
       st.markdown("---")
       st.markdown("*This survey shows that adults DO NOT have a higher attention span compared to children, which is the opposite from what is to be expected. Note the study sample was 200 in total.*")


    with tab4_label: # Content for "Who most distracted?" (Graph4)
       st.subheader("Most Distracted Age Groups")
       st.markdown("""
       This visualization explores the spread of screen time hours across various age groups,
       highlighting potential differences in screen usage patterns that might relate to distraction levels.
       """)
       fig4 = Graph4.makeGraph4() # Call the function here
       st.pyplot(fig4)
       st.markdown("---")
       st.markdown("*All age groups have a median screen time range between 6-8 hours a day. Young adults (25-34) have the highest IQR.*")


    with tab5_label: # Content for "Are productivity apps helpful?" (Graph5)
       st.subheader("Are Productivity Apps Helpful? Usage & Attention Impact")
       st.markdown("""
       This section investigates the relationship between the usage of productivity apps and reported attention spans.
       The first chart shows the overall distribution of productivity app usage, and the second compares attention spans
       between users and non-users across age groups.
       """)
       fig5_1 = Graph5.makeGraph5_1() # Call the function here
       st.pyplot(fig5_1)
       st.markdown("---")
       fig5_2 = Graph5.makeGraph5_2() # Call the function here
       st.pyplot(fig5_2)
       st.markdown("---")
       st.markdown("Note that those who do use productivity apps have a lower screen time. This added screen time could be used as internet content distraction. This means that productivity apps do in fact make people more focused.")

if __name__ == "__main__":
    show()