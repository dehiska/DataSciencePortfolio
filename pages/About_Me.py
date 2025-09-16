# statathon/statathon_app_pages/about_me.py (This file's correct content)

import streamlit as st
import pandas as pd
from PIL import Image, ImageOps

def show_about_me():
    # Title & Introduction
    st.title("👋 Hello, I'm Denis Soulima")
    st.subheader("A Data Scientist | Future Data Science Graduate | Computer Science Graduate")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("""
        Hi! I'm Denis — a passionate **data scientist** and **software developer** with hands-on experience in solving real-world problems through code and data.
                    
        - M.S. in **Data Science** from University of Connecticut
        - B.S. in **Computer Science** from Southern Connecticut State University
        - Minoring in **Mathematics** and **Data Science**
        - Currently preparing for my **Master's in Data Science at UConn Storrs**

        With professional experience at **IBM**, **VictoryWaves**, and **Geek Squad**, I've built robust systems, automated processes, and developed tools to improve efficiency and user experience.

        I'm excited to apply my skills in machine learning, statistical analysis, and full-stack development to create value-driven solutions.
        """)

    with col2:
        # Ensure this path is correct relative to the project root
        img = Image.open("assets/profile_picture.jpg")
        img = ImageOps.exif_transpose(img)
        st.image(img, width=150)

    # Resume Section
    st.markdown("---")
    st.markdown("## 📄 Resume Highlights")

    st.markdown("### Education")
    st.markdown("- **University of Connecticut** - *M.S. in Data Science (Finishing Spring 2026)*")
    st.markdown("- **Southern Connecticut State University** – *B.S. in Computer Science*")
    st.markdown("- **Norwalk Community College** – *A.S. in Mobile Programming*")
    st.markdown("- **Ukrainian Catholic Diocese, Stamford** – *Diploma in Ukrainian Studies*")

    st.markdown("### Experience")
    st.markdown("- **IBM Intern** – Cloud DevOps, CI/CD pipelines, cloud monitoring & security")
    st.markdown("- **Geek Squad IT Specialist** – Technical support, hardware/software troubleshooting, customer training")

    st.markdown("### Skills")
    st.markdown("- **Languages**: Python, Java, JavaScript, SQL, HTML/CSS, Haskell")
    st.markdown("- **Tools**: Git, GitHub, Docker, AWS S3, GCP, Streamlit, Azure Databricks, Google Collab, NoSQL Workbench")
    st.markdown("- **Data Science**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn")
    st.markdown("- **Cloud**: IBM Cloud, CI/CD, DevOps best practices")
    st.markdown("- **Soft Skills**: Multilingual communication (Ukrainian, Russian, Polish, Spanish), problem-solving, teamwork")

    st.markdown("### Languages")
    st.markdown("- Fluent in **English, Ukrainian, Polish, Russian, and Spanish**")

    # Download Resume Button
    # Ensure this path is correct relative to the project root
    with open("assets/resume.docx", "rb") as file:
        st.download_button(
            label="📄 Download Full Resume",
            data=file,
            file_name="resume.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key="download_resume"
        )


if __name__ == "__main__":
    show_about_me()