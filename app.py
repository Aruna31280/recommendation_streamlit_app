import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load Pickle Files
# -----------------------------
@st.cache_resource
def load_files():
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)
    with open("final_features.pkl", "rb") as f:
        final_features = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return data, final_features, vectorizer

data, final_features, vectorizer = load_files()

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend(query):
    ff = final_features.tocsr()

    # Find courses matching the query
    matches = data[data['course_name'].str.contains(query, case=False, na=False)]
    if matches.empty:
        return None

    # Choose best-rated match
    best_match = matches.loc[matches['rating'].idxmax()]
    idx = best_match.name

    # Cosine similarity
    sim_scores = cosine_similarity(ff[idx], ff).flatten()

    # Top similar courses
    top_idx = sim_scores.argsort()[-500:][::-1]
    similar_courses = data.iloc[top_idx][[
        'course_name', 'difficulty_level', 'course_price', 'instructor', 'rating'
    ]].drop_duplicates(subset=['course_name', 'difficulty_level', 'instructor'], keep='first')

    recommendations = []
    levels = ['Beginner', 'Intermediate', 'Advanced']
    used_instructors = set()

    for level in levels:
        level_courses = similar_courses[
            similar_courses['difficulty_level'].str.contains(level, case=False, na=False)
        ].sort_values(by='rating', ascending=False)

        for _, course in level_courses.iterrows():
            if course['instructor'] not in used_instructors:
                recommendations.append(course)
                used_instructors.add(course['instructor'])
                break

    if not recommendations:
        return None

    return pd.DataFrame(recommendations)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="🎓 Smart Course Recommender", layout="centered")

# Custom CSS (Dark mode with gradient cards)
st.markdown("""
    <style>
        body { background-color: #0e1117; color: #f5f5f5; }
        h1, p { text-align: center; color: #dcdcdc; }
        .stTextInput > div > div > input {
            text-align: center;
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
        }
        .stButton > button {
            background-color: #111827;
            color: white;
            border-radius: 10px;
            padding: 8px 20px;
            border: 1px solid #444;
        }
        .stButton > button:hover {
            background-color: #2563eb;
            color: white;
        }
        .card {
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            color: #111;
            font-size: 16px;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.2);
        }
        .beginner { background: linear-gradient(to right, #e0f7e9, #f7fff8); border-left: 5px solid #22c55e; }
        .intermediate { background: linear-gradient(to right, #e0f2ff, #f5fbff); border-left: 5px solid #3b82f6; }
        .advanced { background: linear-gradient(to right, #ffe0e0, #fff7f7); border-left: 5px solid #ef4444; }
        .course-title { font-weight: bold; font-size: 18px; color: #111; }
        .course-info { color: #333; font-size: 15px; margin-top: 5px; }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>🎓 Smart Course Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p>Find top-rated courses for each level — Beginner, Intermediate, and Advanced (with unique instructors).</p>", unsafe_allow_html=True)

# Search
query = st.text_input(" ", placeholder="Type a topic (e.g., Python, Data, AI)...")

# Centered button
col1, col2, col3 = st.columns([1,2,1])
with col2:
    search = st.button("✨ Recommend")

# -----------------------------
# Results
# -----------------------------
if search:
    if not query.strip():
        st.warning("Please enter a topic to search for courses.")
    else:
        with st.spinner("🔎 Finding top-rated courses..."):
            results = recommend(query)

        if results is None or results.empty:
            st.error(f"No matching courses found for '{query}'. Try another keyword.")
        else:
            st.success(f"✅ Recommended Courses for '{query}'")

            color_map = {
                'Beginner': 'beginner',
                'Intermediate': 'intermediate',
                'Advanced': 'advanced'
            }

            for _, row in results.iterrows():
                level = row['difficulty_level']
                card_color = color_map.get(level, 'beginner')

                st.markdown(f"""
                <div class="card {card_color}">
                    <div class="course-title">📘 {level}</div>
                    <div class="course-info">{row['course_name']}</div>
                    <div class="course-info">⭐ {row['rating']} | 💰 ${row['course_price']} | 👨‍🏫 {row['instructor']}</div>
                </div>
                """, unsafe_allow_html=True)
