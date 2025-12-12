import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("online_course_recommendation.xlsx")
    df['text'] = (
        df['course_name'] + " " +
        df['instructor'] + " " +
        df['difficulty_level']
    )
    return df

df = load_data()

# -----------------------------
# CONTENT-BASED MODEL (TF-IDF)
# -----------------------------
@st.cache_resource
def build_tfidf_model():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['text'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = build_tfidf_model()

# -----------------------------
# FUNCTION TO GET RECOMMENDATIONS
# -----------------------------
def recommend(course_name):
    if course_name not in df['course_name'].values:
        return []

    idx = df.index[df['course_name'] == course_name][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # top 5 recommendations
    course_indices = [i[0] for i in sim_scores]
    return df.iloc[course_indices][['course_name', 'instructor', 'rating', 'course_price']]

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🎓 Online Course Recommendation System")

st.write("Choose a course and get 5 similar recommendations")

course_list = df['course_name'].unique()
selected_course = st.selectbox("Select Course", course_list)

if st.button("Recommend"):
    recs = recommend(selected_course)
    st.success("Top Recommendations:")
    st.dataframe(recs)
