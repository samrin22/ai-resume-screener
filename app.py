import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🤖 AI Resume Screener")
st.markdown("### 💼 Smart Resume Analyzer using AI")
st.set_page_config(page_title="AI Resume Screener", page_icon="🤖")

resume = st.text_area("Paste Resume Text", "Python developer with ML experience")

job_desc = st.text_area("Paste Job Description", "Looking for machine learning engineer with Python skills")

if st.button("Analyze Match"):
    if resume and job_desc:
        text = [resume, job_desc]

        cv = CountVectorizer(stop_words='english')
        matrix = cv.fit_transform(text)

        similarity = cosine_similarity(matrix)[0][1]
        score = round(similarity * 100, 2)

        st.success(f"Match Score: {score}%")

        # Find missing keywords
        resume_words = set(resume.lower().split())
        job_words = set(job_desc.lower().split())

        missing = job_words - resume_words

        st.subheader("📌 Missing Keywords")
        st.write(list(missing)[:10])

        if score < 50:
            st.warning("Improve your resume with these keywords.")
        else:
            st.success("Good match! Apply now 🚀")