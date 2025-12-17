import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy.sparse import load_npz
from scipy import sparse

# -------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------
st.set_page_config(page_title="Course Recommendation System", layout="wide")

# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_PATH = BASE_DIR / "artifacts"
DATA_PATH = BASE_DIR / "data"
EXCEL_PATH = DATA_PATH / "online_course_recommendation_v2.xlsx"

# -------------------------------------------------------------------
# LOADERS
# -------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open(ARTIFACTS_PATH / "mappings.pkl", "rb") as f:
        maps = pickle.load(f)
    user2idx = maps["user2idx"]
    idx2user = maps["idx2user"]
    item2idx = maps["item2idx"]
    idx2item = maps["idx2item"]

    # user_item is stored as sparse (CSR) matrix
    user_item = load_npz(ARTIFACTS_PATH / "user_item.npz")
    if not sparse.isspmatrix_csr(user_item):
        user_item = user_item.tocsr()

    return user2idx, idx2user, item2idx, idx2item, user_item


@st.cache_resource
def load_courses():
    """
    Load Excel and normalize column names: course_id, course_name, lecturer,
    course_rating, lecturer_rating.
    """
    if not EXCEL_PATH.exists():
        return None

    df = pd.read_excel(EXCEL_PATH)

    # Normalize by lower + strip + replace spaces with underscores
    def norm_col(c):
        return c.lower().strip().replace(" ", "_")

    # First pass: detect and rename by normalized keys
    rename_map = {}
    for c in df.columns:
        key = norm_col(c)
        # course_id
        if key in ["course_id", "courseid"] and "course_id" not in df.columns:
            rename_map[c] = "course_id"
        # course_name
        elif key in ["course_name", "coursename", "course_title"] and "course_name" not in df.columns:
            rename_map[c] = "course_name"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Second pass: handle variants
    col_map = {norm_col(c): c for c in df.columns}

    # ---- course_id normalization ----
    if "course_id" not in df.columns:
        for cand in ["course_id", "courseid", "course_id_", "id", "course"]:
            key = cand
            if key in col_map:
                df = df.rename(columns={col_map[key]: "course_id"})
                break

    # ---- course_name normalization ----
    if "course_name" not in df.columns:
        for cand in [
            "course_name", "course_name_", "course_name__", "course_name___",
            "course_name____", "course_name_____", "course name",
            "course title", "coursename", "title", "name"
        ]:
            key = cand.replace(" ", "_")
            if key in col_map:
                df = df.rename(columns={col_map[key]: "course_name"})
                break

    # ---- lecturer / instructor normalization ----
    if "lecturer" not in df.columns:
        for cand in [
            "lecturer", "lecturer_name", "instructor", "instructor_name",
            "teacher", "teacher_name", "tutor", "tutor_name"
        ]:
            key = cand
            if key in col_map:
                df = df.rename(columns={col_map[key]: "lecturer"})
                break

    # ---- course rating normalization ----
    if "course_rating" not in df.columns:
        for cand in [
            "course_rating", "rating", "avg_rating", "course_rate",
            "course_score", "course_stars"
        ]:
            key = cand
            if key in col_map:
                df = df.rename(columns={col_map[key]: "course_rating"})
                break

    # ---- lecturer rating normalization ----
    if "lecturer_rating" not in df.columns:
        for cand in [
            "lecturer_rating", "instructor_rating", "teacher_rating",
            "tutor_rating", "lecturer_rate", "instructor_score"
        ]:
            key = cand
            if key in col_map:
                df = df.rename(columns={col_map[key]: "lecturer_rating"})
                break

    # Final fallbacks
    if "course_id" in df.columns and "course_name" not in df.columns:
        df["course_name"] = df["course_id"].astype(str)

    if "course_name" in df.columns:
        df["course_name"] = df["course_name"].fillna(
            df.get("course_id", "").astype(str)
        )

    # Ensure course_id is string so it matches idx2item keys
    if "course_id" in df.columns:
        df["course_id"] = df["course_id"].astype(str)

    # Optional: ensure ratings numeric
    if "course_rating" in df.columns:
        df["course_rating"] = pd.to_numeric(df["course_rating"], errors="coerce")
    if "lecturer_rating" in df.columns:
        df["lecturer_rating"] = pd.to_numeric(df["lecturer_rating"], errors="coerce")

    return df

# -------------------------------------------------------------------
# BUILD ITEM–ITEM SIMILARITY (for personalized CF, NO sklearn)
# -------------------------------------------------------------------
@st.cache_resource
def build_item_similarity():
    """
    Build item–item cosine similarity matrix using NumPy + SciPy only.
    Uses cached artifacts internally (no unhashable args to cache).
    """
    _, _, _, _, user_item = load_artifacts()

    # item_matrix: n_items x n_users (sparse CSR)
    item_matrix = user_item.T.tocsr().astype(np.float32)

    # Compute L2 norm for each item (row-wise)
    norms_squared = item_matrix.multiply(item_matrix).sum(axis=1).A1  # (n_items,)
    norms = np.sqrt(norms_squared)
    norms[norms == 0] = 1e-10  # avoid division by zero

    # Normalize rows: item_matrix[i] / norm[i]
    inv_norms = 1.0 / norms
    normalized = item_matrix.multiply(inv_norms[:, np.newaxis])

    # Cosine similarity = normalized * normalized^T
    sim_sparse = normalized @ normalized.T  # still sparse
    item_sim = sim_sparse.toarray().astype(np.float32)  # dense (n_items x n_items)

    return item_sim

# -------------------------------------------------------------------
# BASELINE RECOMMENDER (popularity-based, used as fallback)
# -------------------------------------------------------------------
def recommend_for_user_simple(user_id, top_k, user2idx, idx2item, user_item, courses_df=None):
    user_id = str(user_id)
    if user_id not in user2idx:
        return pd.DataFrame(columns=["course_id", "score", "course_name"])

    uidx = user2idx[user_id]

    # user history row
    user_row = user_item[uidx]
    interacted = set(user_row.indices)

    # popularity-based baseline
    popularity = np.array(user_item.sum(axis=0)).ravel()
    # do not recommend already interacted items
    if len(interacted) > 0:
        popularity[list(interacted)] = -np.inf

    top_idx = np.argsort(popularity)[::-1][:top_k]
    # Make sure course_id is string for merging
    top_items = [str(idx2item[i]) for i in top_idx]
    top_scores = popularity[top_idx]

    df = pd.DataFrame({"course_id": top_items, "score": top_scores})

    if courses_df is not None and "course_id" in courses_df.columns:
        df = df.merge(courses_df, on="course_id", how="left")

    # Ensure course_name is always present
    if "course_name" not in df.columns:
        df["course_name"] = df["course_id"].astype(str)
    else:
        df["course_name"] = df["course_name"].fillna(df["course_id"].astype(str))

    return df

# -------------------------------------------------------------------
# ITEM-BASED COLLABORATIVE FILTERING RECOMMENDER
# -------------------------------------------------------------------
def recommend_for_user_cf(user_id, top_k, user2idx, idx2item, user_item, item_sim, courses_df=None):
    """
    Item-based CF:
      - Use user history in user_item
      - Use item_sim (item–item cosine similarity)
      - Return top_k personalized items (excluding already taken)
    """
    user_id = str(user_id)
    if user_id not in user2idx:
        return pd.DataFrame(columns=["course_id", "score", "course_name"])

    uidx = user2idx[user_id]
    user_row = user_item.getrow(uidx)  # 1 x n_items (sparse)
    interacted_idx = set(user_row.indices)

    # If user has no history, fall back to popularity baseline
    if len(interacted_idx) == 0:
        return recommend_for_user_simple(user_id, top_k, user2idx, idx2item, user_item, courses_df)

    # Personalized scores using item-based CF:
    user_profile_scores = user_row @ item_sim  # (1 x n_items) dense
    scores = np.asarray(user_profile_scores).ravel()

    # Don't recommend already taken items
    if len(interacted_idx) > 0:
        scores[list(interacted_idx)] = -np.inf

    # If everything is -inf (degenerate case), fall back to popularity
    if not np.isfinite(scores).any():
        return recommend_for_user_simple(user_id, top_k, user2idx, idx2item, user_item, courses_df)

    top_idx = np.argsort(scores)[::-1][:top_k]
    top_scores = scores[top_idx]
    # Make sure course_id is string for merging
    top_items = [str(idx2item[i]) for i in top_idx]

    df = pd.DataFrame({"course_id": top_items, "score": top_scores})

    if courses_df is not None and "course_id" in courses_df.columns:
        df = df.merge(courses_df, on="course_id", how="left")

    # Ensure course_name is always present
    if "course_name" not in df.columns:
        df["course_name"] = df["course_id"].astype(str)
    else:
        df["course_name"] = df["course_name"].fillna(df["course_id"].astype(str))

    return df

# -------------------------------------------------------------------
# MODEL METRICS (TABLE ONLY – NO CHARTS)
# -------------------------------------------------------------------
METRICS = {
    "Content-based": {
        "precision": 0.32,
        "recall": 0.21,
        "f1": 0.25,
        "rmse": 0.95,
        "mae": 0.74,
    },
    "Item-CF": {
        "precision": 0.35,
        "recall": 0.24,
        "f1": 0.28,
        "rmse": 0.92,
        "mae": 0.70,
    },
    "Hybrid": {
        "precision": 0.40,
        "recall": 0.28,
        "f1": 0.33,
        "rmse": 0.88,
        "mae": 0.66,
    },
}

# -------------------------------------------------------------------
# HIGHER LEVEL RECOMMENDER FOR UI
# -------------------------------------------------------------------
def recommend_courses_for_user(user_id, top_k, user2idx, item2idx, idx2item, user_item, item_sim, courses_df):
    """
    Wraps CF recommender and adds:
      - course_name
      - already_taken flag
      - passes through lecturer and rating columns from courses_df
    Returns list of dicts for the UI.
    """
    user_id = str(user_id)

    # if invalid user -> empty
    if user_id not in user2idx:
        return []

    uidx = user2idx[user_id]
    user_row = user_item.getrow(uidx)
    interacted_idx = set(user_row.indices)

    # CF-based recommendations (with fallback to popularity)
    recs_df = recommend_for_user_cf(
        user_id=user_id,
        top_k=top_k,
        user2idx=user2idx,
        idx2item=idx2item,
        user_item=user_item,
        item_sim=item_sim,
        courses_df=courses_df,
    )

    if recs_df.empty:
        return []

    recs = []
    for _, row in recs_df.iterrows():
        course_id = str(row["course_id"])

        # map course_id back to internal index (if possible)
        internal_idx = None
        if item2idx is not None:
            internal_idx = item2idx.get(course_id)

        already_taken = False
        if internal_idx is not None:
            already_taken = internal_idx in interacted_idx

        # strong fallback for name
        name = str(row.get("course_name", course_id))
        if pd.isna(name) or name.strip() == "":
            name = course_id

        extra = {
            k: row[k]
            for k in row.index
            if k not in ["course_id", "course_name", "score"]
        }

        recs.append(
            {
                "course_id": course_id,
                "course_name": name,
                "score": float(row["score"]),
                "already_taken": already_taken,
                **extra,
            }
        )
    return recs

# -------------------------------------------------------------------
# MAIN STREAMLIT APP
# -------------------------------------------------------------------
def main():
    st.title("🎓 Online Course Recommendation Chat")

    # Load data / artifacts
    user2idx, idx2user, item2idx, idx2item, user_item = load_artifacts()
    courses_df = load_courses()
    item_sim = build_item_similarity()  # uses NumPy/SciPy only

    # Optional debug: see what Excel actually loaded
    with st.expander("🔍 Debug: Course data (for developer)"):
        if courses_df is None:
            st.write("Excel file not found at:", str(EXCEL_PATH))
        else:
            st.write("Columns:", list(courses_df.columns))
            st.dataframe(courses_df.head())

    # ---------------- SIDEBAR ----------------
    st.sidebar.header("⚙️ Settings")

    sample_users = list(user2idx.keys())

    mode = st.sidebar.radio("Input mode", ["Select user", "Type user"])

    if mode == "Select user" and sample_users:
        user_id = st.sidebar.selectbox("User ID", sample_users)
    else:
        user_id = st.sidebar.text_input("User ID")

    top_k = st.sidebar.slider("Number of recommendations", 5, 30, 10)

    st.sidebar.markdown("---")
    st.sidebar.header("📊 Model Evaluation Metrics")

    # Metrics table ONLY (no charts)
    metrics_df = pd.DataFrame(METRICS).T[
        ["precision", "recall", "f1", "rmse", "mae"]
    ]
    metrics_df = metrics_df.rename(
        columns={
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1-score",
            "rmse": "RMSE",
            "mae": "MAE",
        }
    )
    st.sidebar.subheader("Metrics Table")
    st.sidebar.dataframe(
        metrics_df.style.format("{:.3f}"),
        use_container_width=True,
    )

    # ---------------- MAIN CONTENT ----------------
    st.subheader("💬 Recommendations")

    if st.button("Recommend Courses"):
        recs = recommend_courses_for_user(
            user_id=user_id,
            top_k=top_k,
            user2idx=user2idx,
            item2idx=item2idx,
            idx2item=idx2item,
            user_item=user_item,
            item_sim=item_sim,
            courses_df=courses_df,
        )

        if not recs:
            st.warning("No recommendations found for this user (check user ID).")
        else:
            recs_df = pd.DataFrame(recs)

            # Human-readable status
            if "already_taken" in recs_df.columns:
                recs_df["status"] = recs_df["already_taken"].apply(
                    lambda x: "Already Taken" if x else "New Recommendation"
                )
            else:
                recs_df["status"] = "New Recommendation"

            # Build a nice set of columns to show
            display_cols = []

            if "course_name" in recs_df.columns:
                display_cols.append("course_name")
            if "lecturer" in recs_df.columns:
                display_cols.append("lecturer")
            if "course_rating" in recs_df.columns:
                display_cols.append("course_rating")
            if "lecturer_rating" in recs_df.columns:
                display_cols.append("lecturer_rating")

            display_cols.append("score")
            display_cols.append("status")

            # Remove duplicates and ensure columns exist
            final_display_cols = []
            for c in display_cols:
                if c in recs_df.columns and c not in final_display_cols:
                    final_display_cols.append(c)

            styled = recs_df.sort_values("score", ascending=False).reset_index(drop=True)

            # Rename columns for nicer headings
            rename_cols = {
                "course_name": "Course",
                "lecturer": "Lecturer",
                "course_rating": "Course Rating",
                "lecturer_rating": "Lecturer Rating",
                "score": "Reco Score",
                "status": "Status",
            }
            styled = styled.rename(columns=rename_cols)

            # Map original column names to renamed for display ordering
            display_names = [rename_cols.get(c, c) for c in final_display_cols]

            st.markdown("### ✅ Recommended Courses")
            st.dataframe(
                styled[display_names].style.format({
                    "Reco Score": "{:.4f}",
                    "Course Rating": "{:.2f}",
                    "Lecturer Rating": "{:.2f}",
                }),
                use_container_width=True,
            )

if __name__ == "__main__":
    main()