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

st.set_page_config(page_title="Resume Screener", page_icon="📄", layout="wide")

# ── NLP Setup ──
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

SKILLS = ["python", "machine learning", "deep learning", "sql",
          "nlp", "tensorflow", "pandas", "data analysis",
          "scikit-learn", "tableau", "power bi", "excel",
          "statistics", "numpy", "matplotlib", "seaborn",
          "keras", "pytorch", "spark", "hadoop", "aws", "docker"]

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

# ── Session State Init ──
if "resumes" not in st.session_state:
    st.session_state.resumes = {
        "Alice": "Skills: Python, Machine Learning, Pandas, Scikit-learn, NLP, SQL, NumPy, Statistics. Experience: Data Analyst at ABC Corp. Built ML models for customer segmentation. Education: B.Tech Computer Science 2024.",
        "Bob": "Skills: Java, JavaScript, HTML, CSS, MySQL, Excel. Experience: Web Developer at XYZ Ltd. Built e-commerce websites. Education: B.Tech Information Technology 2024.",
        "Carol": "Skills: Python, Deep Learning, TensorFlow, NLP, Pandas, Data Analysis, Matplotlib, Seaborn, Statistics. Experience: ML Intern at DataCorp. Built NLP models. Education: B.Tech CS 2025.",
        "David": "Skills: Excel, Tableau, Power BI, SQL, Statistics, Data Analysis. Experience: Business Analyst at FinCorp. Created dashboards. Education: B.Com Finance 2024.",
        "Eva": "Skills: Python, Machine Learning, Deep Learning, Scikit-learn, NumPy, Pandas, NLP, TensorFlow, SQL. Experience: AI Research Intern. Published NLP paper. Education: B.Tech AI 2025."
    }

# ── Header ──
st.title("📄 Resume Screening & Candidate Scoring System")
st.markdown("Rank candidates against a job description using **NLP + TF-IDF + Cosine Similarity**")
st.divider()

# ── JD + Settings ──
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📋 Job Description")
    jd_text = st.text_area("Paste the Job Description here", height=200,
        value="We are looking for a Data Scientist Intern with strong Python skills.\nRequired: Machine Learning, NLP, Pandas, Scikit-learn, SQL, Data Analysis, Statistics, NumPy.\nExperience with data visualization and deep learning is a plus.")

with col2:
    st.subheader("⚙️ Settings")
    threshold = st.slider("Shortlist Threshold (%)", 10, 80, 30)
    st.info(f"Candidates scoring ≥ **{threshold}%** will be shortlisted")
    st.metric("Total Candidates", len(st.session_state.resumes))

st.divider()

# ── Add New Resume ──
st.subheader("➕ Add New Candidate")
with st.expander("Click to add a new candidate", expanded=False):
    col_a, col_b = st.columns([1, 2])
    with col_a:
        new_name = st.text_input("Candidate Name", placeholder="e.g. John Doe")
    with col_b:
        new_resume = st.text_area("Resume Text", height=120,
            placeholder="Paste candidate's resume or skills here...")

    if st.button("➕ Add Candidate", type="primary", use_container_width=True):
        if not new_name.strip():
            st.error("Please enter a candidate name!")
        elif not new_resume.strip():
            st.error("Please enter resume text!")
        elif new_name.strip() in st.session_state.resumes:
            st.error(f"Candidate '{new_name}' already exists! Use a different name.")
        else:
            st.session_state.resumes[new_name.strip()] = new_resume.strip()
            st.success(f"✅ '{new_name}' added successfully!")
            st.rerun()

st.divider()

# ── Manage / Delete Resumes ──
st.subheader("👥 Manage Candidates")

if len(st.session_state.resumes) == 0:
    st.warning("No candidates added yet. Add candidates above!")
else:
    for name in list(st.session_state.resumes.keys()):
        col1, col2, col3 = st.columns([2, 5, 1])
        with col1:
            st.markdown(f"**{name}**")
        with col2:
            preview = st.session_state.resumes[name][:120] + "..." if len(st.session_state.resumes[name]) > 120 else st.session_state.resumes[name]
            st.caption(preview)
        with col3:
            if st.button("🗑️ Delete", key=f"del_{name}", use_container_width=True):
                del st.session_state.resumes[name]
                st.success(f"'{name}' removed!")
                st.rerun()
        st.divider()

# ── Edit Existing Resume ──
st.subheader("✏️ Edit a Candidate's Resume")
with st.expander("Click to edit an existing candidate", expanded=False):
    if len(st.session_state.resumes) == 0:
        st.info("No candidates to edit.")
    else:
        edit_name = st.selectbox("Select Candidate to Edit",
                                  options=list(st.session_state.resumes.keys()))
        edit_text = st.text_area("Edit Resume Text", 
                                  value=st.session_state.resumes.get(edit_name, ""),
                                  height=150)
        if st.button("💾 Save Changes", use_container_width=True):
            st.session_state.resumes[edit_name] = edit_text
            st.success(f"✅ '{edit_name}' updated successfully!")
            st.rerun()

st.divider()

# ── Screen Button ──
if st.button("🔍 Screen All Candidates", type="primary", use_container_width=True):
    if len(st.session_state.resumes) < 2:
        st.error("Please add at least 2 candidates to screen!")
    elif not jd_text.strip():
        st.error("Please enter a job description!")
    else:
        with st.spinner("Analyzing resumes with NLP..."):
            results = screen_resumes(jd_text, st.session_state.resumes, threshold)

        st.success(f"✅ Screening complete! Analyzed {len(st.session_state.resumes)} candidates.")
        st.divider()

        # ── Metrics ──
        shortlisted = results[results["Status"] == "✅ Shortlisted"]
        rejected = results[results["Status"] == "❌ Not Selected"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Candidates", len(results))
        m2.metric("Shortlisted", len(shortlisted))
        m3.metric("Rejected", len(rejected))
        m4.metric("Top Score", f"{results['Match Score (%)'].max()}%")

        st.divider()

        # ── Results Table ──
        st.subheader("📊 Ranking Results")
        display_df = results[["Rank", "Candidate", "Match Score (%)", "Status"]].copy()
        display_df["Skills Found"] = results["Skills Found"].apply(lambda x: ", ".join(x))
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.divider()

        # ── Bar Chart ──
        st.subheader("📈 Match Score Chart")
        fig, ax = plt.subplots(figsize=(10, max(4, len(results) * 0.8)))
        colors = ["#2ecc71" if s >= threshold else "#e74c3c"
                  for s in results["Match Score (%)"]]
        bars = ax.barh(results["Candidate"], results["Match Score (%)"], color=colors)
        for bar, score in zip(bars, results["Match Score (%)"]):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{score}%', va='center', fontsize=10, fontweight='bold')
        ax.axvline(x=threshold, color='gray', linestyle='--',
                   label=f'Threshold ({threshold}%)')
        ax.set_xlabel("Match Score (%)")
        ax.set_title("Candidate Ranking by Job Match Score")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

        st.divider()

        # ── Skills Heatmap ──
        st.subheader("🔥 Skills Presence Heatmap")
        detected_skills = []
        for text in st.session_state.resumes.values():
            detected_skills.extend(extract_skills(text))
        top_skills = list(dict.fromkeys(detected_skills))[:12]

        if top_skills:
            skill_matrix = pd.DataFrame(
                {skill: [1 if skill in extract_skills(text) else 0
                         for text in st.session_state.resumes.values()]
                 for skill in top_skills},
                index=st.session_state.resumes.keys()
            )
            fig2, ax2 = plt.subplots(figsize=(max(10, len(top_skills) * 1.2),
                                              max(4, len(st.session_state.resumes) * 0.8)))
            sns.heatmap(skill_matrix, annot=True, cmap="YlGnBu",
                        linewidths=0.5, cbar=False, ax=ax2, fmt="d")
            ax2.set_title("Skills Presence Heatmap")
            plt.tight_layout()
            st.pyplot(fig2)

        st.divider()

        # ── Word Cloud ──
        st.subheader("☁️ Resume Keywords Word Cloud")
        all_text = " ".join(st.session_state.resumes.values())
        wc = WordCloud(width=800, height=350, background_color='white',
                       colormap='viridis', max_words=60).generate(all_text)
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        ax3.imshow(wc, interpolation='bilinear')
        ax3.axis('off')
        ax3.set_title("Most Common Keywords Across All Resumes")
        plt.tight_layout()
        st.pyplot(fig3)

        st.divider()

        # ── Shortlist Summary ──
        st.subheader("🏆 Final Shortlist Summary")
        col_s, col_r = st.columns(2)
        with col_s:
            st.markdown("### ✅ Shortlisted")
            for _, row in shortlisted.iterrows():
                st.success(f"**#{int(row['Rank'])} {row['Candidate']}** — {row['Match Score (%)']}%")
        with col_r:
            st.markdown("### ❌ Not Selected")
            for _, row in rejected.iterrows():
                st.error(f"**#{int(row['Rank'])} {row['Candidate']}** — {row['Match Score (%)']}%")

        st.divider()

        # ── Download ──
        csv = display_df.to_csv(index=False)
        st.download_button("⬇️ Download Results as CSV", csv,
                           "screening_results.csv", "text/csv",
                           use_container_width=True)
