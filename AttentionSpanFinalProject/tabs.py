import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import Graph1, Graph2, Graph3, Graph4, Graph5
url1 = 'data.csv'
df1 = pd.read_csv(url1)
url2 = 'screen_time.csv'
df2 = pd.read_csv(url2)


#df['tip_pct'] = df['tip'] / df['total_bill'] * 100
#st.title('Restaurant spending')

#fig1 = plt.figure()
#ax1 = fig1.add_subplot()
#ax1.set_xlabel('Bill ($)')
#ax1.set_title('Bill amount distribution')
#ax1.hist(df['total_bill'], bins = 20, color='red', alpha=0.5)


#Graph 2 Tab 2
fig2 = Graph2.makeGraph2()

#Graph 3 Tab 3
fig3 = Graph3.makeGraph3()

#Graph 4 Tab 4
fig4 = Graph4.makeGraph4()

#Graph 5 Tab 5
fig5_1 = Graph5.makeGraph5_1()
fig5_2 = Graph5.makeGraph5_2()



tab1, tab2, tab3, tab4, tab5 = st.tabs(["Screentime for School", "Age vs. Screentime", "Kids vs. Adults in Attention Span", "Who most distracted?", "Are productivity apps helpful?"])

with tab1:
    st.subheader('Which age group gets most distracted from notifications?')

    # Collect user input first
    purpose = st.radio('Select Purpose', ('Educational', 'Recreational', "Both"))
    time = st.radio("Select which days of the week", ("Weekdays", "Weekends", "Both"))

    # Then show submit button at the bottom
    if st.button("Submit"):
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
        
        st.text("Depends on what variables you choose from, the graph changes. From ages 5 to 15, both educational and recreational screen time increase—but recreational use grows much faster. By age 15, teens average over 5 hours of recreational screen time compared to just 2 for education. This widening gap suggests that as kids grow, screen time becomes more about entertainment than learning, potentially impacting focus and productivity.")


with tab2:
    st.subheader("Graph 2: Screen Time by Age Group")
    fig = Graph2.makeGraph2()
    st.pyplot(fig)
    st.text("This graph shows that young adults (18–24) have the highest screen time, with their usage peaking at 6–8 hours per day. In contrast, other age groups have significantly lower screen time, especially those 35 and older. This suggests that young adults are the most digitally engaged, likely due to a mix of school, work, and social media usage.")

with tab3:
    st.pyplot(fig3)
    st.text("Children under 18 actually have a longer average attention span (around 47 minutes) than adults, who average about 35 minutes. This may be due to adults being more mentally overstimulated or more easily distracted by digital notifications and multitasking. It challenges the assumption that maturity always brings better focus.It could also point to differences in learning environments, screen habits, or lifestyle distractions between the two groups.")

with tab4:
    st.pyplot(fig4)
    st.text("This box plot shows that the 18–24 age group not only has the highest median screen time but also the widest range, meaning they vary greatly in how distracted they get by notifications. Younger groups (Below 18) and older adults (45+) show more consistent but lower distraction levels. This reinforces the idea that digital distraction hits young adults the hardest, likely due to constant app engagement.")


with tab5:
    st.pyplot(fig5_1)

    st.text("A majority—61.5%—of respondents don’t use productivity apps, while only 38.5% do. This suggests that most people either aren’t aware of or don’t feel the need for digital tools to manage time and tasks, despite high screen usage and attention issues. It raises questions about whether people are choosing convenience over intentional focus.")
    st.pyplot(fig5_2)

    st.text("Productivity app usage varies widely by age. Those aged 25–34 and below 18 show higher usage, while older age groups, especially 45 and above, are the least engaged. Even among young people, non-use often outweighs use, indicating that productivity tools are still not the norm across any age group.")
