import os
import streamlit as st
import PyPDF2
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import google.generativeai as genai

from ai_helper import extract_skills_from_jd, generate_resume_suggestions
from utils import (
    clean_text,
    calculate_similarity,
    find_missing_skills,
    calculate_final_score,
    generate_pdf_report
)

# Setup -------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(
    page_title="AI Resume Analyzer (ATS)",
    page_icon="logo.png",
    layout="centered"
)

st.title("AI Resume Analyzer — ATS Mode")
st.caption("Accurate ATS-style scoring with AI-powered skill extraction and suggestions.")


# PDF Extract -------------
def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text
    except Exception as e:
        st.error(f"❌ PDF read error: {e}")
        return ""


# ------------------------- Analyze Button -------------------------
resume_file = st.file_uploader("📄 Upload resume (PDF)", type=["pdf"])
job_desc = st.text_area("📝 Paste Job Description")


if st.button("🔍 Analyze Match"):
    if not resume_file or not job_desc.strip():
        st.warning("⚠ Please upload a resume and provide a Job Description.")
        st.stop()

    with st.spinner("Processing..."):

        # Extract & clean text ----------
        raw_resume = extract_text_from_pdf(resume_file)
        resume_text = clean_text(raw_resume)
        jd_text = clean_text(job_desc)

        # AI skill extraction (IMPORTANT: use RAW JD) ----------
        extracted_skills = extract_skills_from_jd(job_desc)

        if not extracted_skills:
            st.error("❌ Could not extract skills from the Job Description.")
            st.stop()

        # Semantic similarity ----------
        semantic_score = calculate_similarity(resume_text, jd_text)

        # Skill analysis ----------
        missing_skills, matched_skills = find_missing_skills(
            resume_text,
            jd_text,
            extracted_skills
        )

        # Final ATS Score ----------
        final_score = calculate_final_score(
            semantic_score,
            matched_skills,
            missing_skills,
            resume_text,
            jd_text
        )

    # Results -------------
    st.subheader(f"📊 ATS Match Score: **{final_score}%**")

    if final_score >= 80:
        st.success("Excellent match — strong alignment with job requirements.")
    elif final_score >= 60:
        st.info("Good match — consider small improvements.")
    elif final_score >= 40:
        st.warning("Partial match — add or highlight missing skills.")
    else:
        st.error("Low match — resume needs major alignment with the JD.")

    st.markdown(f"### 🔍 Semantic Similarity Score: **{semantic_score}%**")

    st.markdown("### 🧠 Extracted Skills (from JD)")
    st.write(", ".join(extracted_skills))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ✅ Matched Skills")
        st.write(", ".join(matched_skills) if matched_skills else "None")

    with col2:
        st.markdown("### ❌ Missing Skills")
        st.write(", ".join(missing_skills) if missing_skills else "None")

    # Skill chart
    st.markdown("### 📊 Skill Match Overview")
    fig, ax = plt.subplots()
    ax.bar(["Matched", "Missing"], [len(matched_skills), len(missing_skills)],
           color=["#2ecc71", "#e74c3c"])
    ax.set_ylabel("Skill Count")
    ax.set_title("Resume vs JD Skill Comparison")
    st.pyplot(fig)

    # AI suggestions
    st.markdown("### 💡 AI Suggestions to Improve Resume")
    suggestions = generate_resume_suggestions(resume_text, job_desc)
    st.write(suggestions)

    # PDF Export
    report = generate_pdf_report(final_score, matched_skills, missing_skills, suggestions)
    if report:
        st.download_button(
            "📥 Download Full ATS Report (PDF)",
            data=report,
            file_name="ATS_Report.pdf",
            mime="application/pdf"
        )
    else:
        st.error("❌ Failed to generate PDF report.")
