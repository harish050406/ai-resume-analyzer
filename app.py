import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="AI Resume Analyzer", page_icon="💼", layout="centered")

# ---------------------------
# CUSTOM CSS (UI IMPROVEMENT)
# ---------------------------
st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
    color: white;
}

h1 {
    text-align: center;
    color: #38bdf8;
}

h2 {
    text-align: center;
}

h3 {
    color: #22c55e;
}

[data-testid="stFileUploader"] {
    border: 2px dashed #38bdf8;
    padding: 20px;
    border-radius: 10px;
}

textarea {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 10px !important;
}

.stButton>button {
    background-color: #38bdf8;
    color: black;
    border-radius: 10px;
    padding: 10px;
    font-weight: bold;
}

[data-testid="stMetric"] {
    background-color: #1e293b;
    padding: 15px;
    border-radius: 10px;
}

hr {
    border: 1px solid #334155;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# SKILLS LIST
# ---------------------------
skills_list = [
    "python", "java", "sql", "machine learning",
    "data analysis", "html", "css", "javascript",
    "react", "node.js", "aws", "docker"
]

# ---------------------------
# FUNCTIONS
# ---------------------------
def extract_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text.lower()

def extract_skills(text):
    found_skills = []
    for skill in skills_list:
        if skill in text:
            found_skills.append(skill)
    return found_skills

def match_skills(resume_skills, job_skills):
    matched = list(set(resume_skills) & set(job_skills))
    missing = list(set(job_skills) - set(resume_skills))
    return matched, missing

def calculate_similarity(resume_text, job_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_text])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    return similarity[0][0] * 100

def resume_feedback(score, missing_skills):
    feedback = []

    if score < 40:
        feedback.append("⚠️ Your resume is not well aligned with this job.")
    elif score < 70:
        feedback.append("🟡 Your resume is moderately aligned. Improve key skills.")
    else:
        feedback.append("🟢 Great match! You are a strong candidate.")

    if missing_skills:
        feedback.append("📌 Focus on learning these skills:")
        for skill in missing_skills:
            feedback.append(f"- {skill}")

    return feedback

# ---------------------------
# HEADER
# ---------------------------
st.markdown("""
<h1> AI Resume Analyzer</h1>
<p style='text-align:center; font-size:18px;'>
Match your resume with job descriptions instantly!
</p>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------
# INPUT SECTION
# ---------------------------
uploaded_file = st.file_uploader("📄 Upload Resume (PDF)")
job_desc = st.text_area("📝 Paste Job Description")

# ---------------------------
# PROCESS
# ---------------------------
if uploaded_file and job_desc:
    resume_text = extract_text(uploaded_file)

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_desc.lower())

    matched, missing = match_skills(resume_skills, job_skills)
    score = calculate_similarity(resume_text, job_desc)

    # ---------------------------
    # RESULT CARD
    # ---------------------------
    st.markdown("""
    <div style='background-color:#1e293b; padding:20px; border-radius:10px;'>
    <h3>📊 Match Result</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <h2 style='color:#22c55e;'>
    Match Score: {score:.2f}%
    </h2>
    """, unsafe_allow_html=True)

    st.progress(int(score))

    col1, col2 = st.columns(2)

    with col1:
        st.success("Matched Skills")
        if matched:
            for skill in matched:
                st.markdown(f"{skill}")
        else:
            st.write("No matching skills found")

    with col2:
        st.error("Missing Skills")
        if missing:
            for skill in missing:
                st.markdown(f"{skill}")
        else:
            st.write("No missing skills")

    # ---------------------------
    # SUGGESTIONS
    # ---------------------------
    st.divider()
    st.subheader("💡 Suggestions")

    if missing:
        for skill in missing:
            st.markdown(f"👉 Learn **{skill}** to improve your chances")
    else:
        st.markdown(" Your resume matches well!")

    # ---------------------------
    # FEEDBACK
    # ---------------------------
    st.divider()
    st.subheader("Resume Feedback")

    feedback = resume_feedback(score, missing)

    for item in feedback:
        st.write(item)