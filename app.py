import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# ── Page Config ──
st.set_page_config(
    page_title="Resume Screener",
    page_icon="📄",
    layout="wide"
)

# ── NLP Setup ──
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

SKILLS = ["python", "machine learning", "deep learning", "sql",
          "nlp", "tensorflow", "pandas", "data analysis",
          "scikit-learn", "tableau", "power bi", "excel",
          "statistics", "numpy", "matplotlib", "seaborn"]

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens
              if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

def extract_skills(text):
    return [skill for skill in SKILLS if skill in text.lower()]

def screen_resumes(jd_text, resumes_dict, threshold):
    cleaned_jd = clean_text(jd_text)
    cleaned_resumes = {k: clean_text(v) for k, v in resumes_dict.items()}

    all_docs = [cleaned_jd] + list(cleaned_resumes.values())
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    scores = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])[0]

    results = pd.DataFrame({
        "Candidate": list(resumes_dict.keys()),
        "Match Score (%)": (scores * 100).round(2),
        "Skills Found": [extract_skills(t) for t in resumes_dict.values()]
    })
    results = results.sort_values("Match Score (%)", ascending=False).reset_index(drop=True)
    results["Rank"] = results.index + 1
    results["Status"] = results["Match Score (%)"].apply(
        lambda x: "✅ Shortlisted" if x >= threshold else "❌ Not Selected"
    )
    return results

# ── UI ──
st.title("📄 Resume Screening & Candidate Scoring System")
st.markdown("Rank candidates against a job description using **NLP + TF-IDF + Cosine Similarity**")
st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Job Description")
    jd_text = st.text_area("Paste the Job Description here", height=200,
        value="""We are looking for a Data Scientist Intern with strong Python skills.
Required: Machine Learning, NLP, Pandas, Scikit-learn, SQL, Data Analysis, Statistics, NumPy.
Experience with data visualization and deep learning is a plus.""")

with col2:
    st.subheader("⚙️ Settings")
    threshold = st.slider("Shortlist Threshold (%)", 10, 80, 30)
    st.info(f"Candidates with score ≥ **{threshold}%** will be shortlisted")

st.divider()
st.subheader("👤 Candidate Resumes")
st.markdown("Enter each candidate's name and resume text below:")

num_candidates = st.number_input("Number of candidates", min_value=2, max_value=10, value=5)

resumes = {}
for i in range(int(num_candidates)):
    with st.expander(f"Candidate {i+1}", expanded=(i < 2)):
        name = st.text_input(f"Name", key=f"name_{i}",
            value=["Alice", "Bob", "Carol", "David", "Eva"][i] if i < 5 else f"Candidate{i+1}")
        resume = st.text_area(f"Resume Text", height=120, key=f"resume_{i}",
            value=[
                "Skills: Python, Machine Learning, Pandas, Scikit-learn, NLP, SQL, NumPy, Statistics. Experience: Data Analyst. Built ML models.",
                "Skills: Java, JavaScript, HTML, CSS, MySQL, Excel. Experience: Web Developer. Built e-commerce websites.",
                "Skills: Python, Deep Learning, TensorFlow, NLP, Pandas, Data Analysis, Matplotlib, Seaborn, Statistics. Experience: ML Intern.",
                "Skills: Excel, Tableau, Power BI, SQL, Statistics, Data Analysis. Experience: Business Analyst.",
                "Skills: Python, Machine Learning, Deep Learning, Scikit-learn, NumPy, Pandas, NLP, TensorFlow, SQL. Experience: AI Research Intern."
            ][i] if i < 5 else "")
        if name and resume:
            resumes[name] = resume

st.divider()

if st.button("🔍 Screen Candidates", type="primary", use_container_width=True):
    if len(resumes) < 2:
        st.error("Please enter at least 2 candidates!")
    elif not jd_text:
        st.error("Please enter a job description!")
    else:
        with st.spinner("Analyzing resumes..."):
            results = screen_resumes(jd_text, resumes, threshold)

        st.success(f"✅ Screening complete! Analyzed {len(resumes)} candidates.")
        st.divider()

        # ── Metrics ──
        shortlisted = results[results["Status"] == "✅ Shortlisted"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Candidates", len(results))
        m2.metric("Shortlisted", len(shortlisted))
        m3.metric("Top Score", f"{results['Match Score (%)'].max()}%")

        st.divider()

        # ── Results Table ──
        st.subheader("📊 Ranking Results")
        display_df = results[["Rank", "Candidate", "Match Score (%)", "Status"]].copy()
        display_df["Skills Found"] = results["Skills Found"].apply(lambda x: ", ".join(x))
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.divider()

        # ── Bar Chart ──
        st.subheader("📈 Match Score Chart")
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#2ecc71" if s >= threshold else "#e74c3c"
                  for s in results["Match Score (%)"]]
        bars = ax.barh(results["Candidate"], results["Match Score (%)"], color=colors)
        for bar, score in zip(bars, results["Match Score (%)"]):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{score}%', va='center', fontsize=10, fontweight='bold')
        ax.axvline(x=threshold, color='gray', linestyle='--', label=f'Threshold ({threshold}%)')
        ax.set_xlabel("Match Score (%)")
        ax.set_title("Candidate Ranking")
        ax.legend()
        st.pyplot(fig)

        st.divider()

        # ── Skills Heatmap ──
        st.subheader("🔥 Skills Heatmap")
        skill_matrix = pd.DataFrame(
            {skill: [1 if skill in extract_skills(text) else 0
                     for text in resumes.values()]
             for skill in SKILLS},
            index=resumes.keys()
        )
        fig2, ax2 = plt.subplots(figsize=(14, 5))
        sns.heatmap(skill_matrix, annot=True, cmap="YlGnBu",
                    linewidths=0.5, cbar=False, ax=ax2, fmt="d")
        ax2.set_title("Skills Presence Heatmap")
        st.pyplot(fig2)

        st.divider()

        # ── Word Cloud ──
        st.subheader("☁️ Resume Keywords Word Cloud")
        all_text = " ".join(resumes.values())
        wc = WordCloud(width=800, height=350, background_color='white',
                       colormap='viridis').generate(all_text)
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        ax3.imshow(wc, interpolation='bilinear')
        ax3.axis('off')
        st.pyplot(fig3)

        # ── Download ──
        st.divider()
        csv = display_df.to_csv(index=False)
        st.download_button("⬇️ Download Results as CSV", csv,
                           "screening_results.csv", "text/csv",
                           use_container_width=True)