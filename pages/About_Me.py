import streamlit as st
from PIL import Image

def show():
    # --- PAGE CONFIG ---
    st.set_page_config(page_title="About Me", page_icon="👋", layout="wide")

    # --- HERO SECTION ---
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("Denis Soulima")
        st.subheader("Data Scientist with a Foundation in Computer Science")
        st.write("""
        Driven by a passion for uncovering insights from complex data, I am a Master's student in Data Science at the University of Connecticut. With a background in Computer Science and professional experience at companies like IBM, I bring a unique blend of software engineering discipline and advanced statistical knowledge to every project. I thrive on solving real-world challenges, from detecting financial fraud to developing robust data pipelines.
        """)
        # Download Resume Button
        with open("assets/resume2026.docx", "rb") as file:
             st.download_button(
                 label="📄 Download My Resume",
                 data=file,
                 file_name="DenisSoulima_Resume2026.docx",
                 mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
             )

    with col2:
        profile_pic = Image.open("assets/profile_picture_fixed.jpg")
        st.image(profile_pic, width=230, caption="Denis Soulima")


    st.markdown("---")

    # --- AWARDS & RECOGNITION ---
    st.subheader("🏆 Awards & Recognition")
    st.markdown("""
    - **1st Place Winner, Yale Statathon 2024:** Led a team in a competitive challenge hosted by Yale University and Travelers Insurance to predict car insurance fraud. This project involved deep feature engineering, model development, and addressing severe class imbalance to maximize the F1 score.
    """)

    st.markdown("---")

    # --- SKILLS ---
    st.subheader("🛠️ Technical Skills")
    st.markdown("""
    - **Programming:** Python (Pandas, NumPy, Scikit-learn), R, SQL, Java, JavaScript, Haskell
    - **Big Data & Databases:** Databricks, Hadoop, Hive, PostgreSQL, NoSQL
    - **Machine Learning:** Predictive Modeling, Classification, Regression, Clustering, Ensemble Methods, Fraud Detection
    - **Cloud & DevOps:** AWS (S3), Azure, Google Cloud, Docker, CI/CD
    - **Languages:** Fluent in English, Spanish, Ukrainian, Russian, and Polish.
    """)

    st.markdown("---")

    # --- EDUCATION ---
    st.subheader("🎓 Education")
    st.markdown("""
    **University of Connecticut** | Master of Science in Data Science (Expected May 2026)
    - *Relevant Coursework:* Machine Learning, Advanced Statistical Methods, Big Data Systems, Data Visualization, Statistical Computing.

    **Southern Connecticut State University** | Bachelor of Science in Computer Science
    - *Minors:* Mathematics, Mobile Programming, Data Science
    """)

    st.markdown("---")

    # --- PROFESSIONAL EXPERIENCE ---
    st.subheader("💼 Professional Experience")
    st.markdown("""
    **Data Engineer Intern | VictoryWaves**
    - Performed ETL (Extract, Transform, Load) on large datasets using Databricks on Azure, NoSQL Workbench, and AWS S3.
    - Contributed to developing and maintaining data pipelines, ensuring data quality and accessibility.

    **Advanced Geeksquad Technician | Bestbuy**
    - Diagnosed and repaired complex hardware and software issues for clients.
    - Performed data recovery, OS installations, and provided technical training.

    **Back-End Software Engineer Intern | IBM Cloud**
    - Gained hands-on experience in cloud infrastructure and the software development lifecycle.
    """)

# This allows the page to be run directly for testing
if __name__ == "__main__":
    show()
