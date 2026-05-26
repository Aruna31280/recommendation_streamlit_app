import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ---------------- Load Model & Preprocessor ---------------- #
def _safe_load_pickle(path, friendly_name):
    if not os.path.exists(path):
        st.error(f"File not found: {path}. Make sure `{friendly_name}` exists in the app folder.")
        st.stop()
    try:
        return pickle.load(open(path, "rb"))
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
        st.stop()

model = _safe_load_pickle("cbf_model_user_aware.pkl", "model")
preprocessor = _safe_load_pickle("preprocessor_user_aware.pkl", "preprocessor")

# ---------------- Load Original Dataset ---------------- #
data_path = "online_course_recommendation_v2.xlsx"
if not os.path.exists(data_path):
    st.error(f"Data file not found: {data_path}. Put the dataset in the app folder.")
    st.stop()

try:
    if data_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)
except Exception as e:
    st.error(f"Failed to read data file {data_path}: {e}")
    st.stop()

st.set_page_config(page_title="Course Recommendation System",
                   page_icon="🎓", layout="wide")

st.title("🎓 Course Recommendation App")
st.write("Enter your **User ID** to get top recommended courses.")

# ---------------- User Input ---------------- #
# Provide both a number input and a sample selector for convenience
user_id = st.number_input("Enter User ID", step=1)

N = st.slider("How many recommendations do you want?", 1,10,5)
agg_method = st.selectbox("Aggregation method (when collapsing per course)", options=["mean","max"], index=0)
# --- Customization filters ---
cert_options = ['Any'] + sorted(df['certification_offered'].dropna().unique().astype(str).tolist())
cert_choice = st.selectbox('Certification offered', options=cert_options, index=0)
difficulty_options = sorted(df['difficulty_level'].dropna().unique().astype(str).tolist())
difficulty_choice = st.multiselect('Difficulty level (leave blank = any)', options=difficulty_options, default=difficulty_options)
min_price = float(df['course_price'].min())
max_price = float(df['course_price'].max())
price_choice = st.slider('Course price range', min_price, max_price, (min_price, max_price))
min_pred = st.slider('Minimum predicted rating to include', 0.0, 5.0, 0.0, step=0.1)
instructor_options = ['Any'] + sorted(df['instructor'].dropna().unique().astype(str).tolist())
instructor_choice = st.selectbox('Instructor (optional)', options=instructor_options, index=0)

if st.button("Recommend"):
    # fetch user data
    user_data = df[df["user_id"] == user_id]

    if user_data.empty:
        st.error("❌ User ID not found in database.")
    else:
        # Prepare all course feature rows for prediction
        test_data = df.copy()   # predict for all courses

        # Validate that the dataset contains the features the preprocessor expects
        feature_names = getattr(preprocessor, "feature_names_in_", None)
        if feature_names is None:
            st.error("The loaded preprocessor does not expose `feature_names_in_`. Ensure the preprocessor is a fitted transformer.")
            st.stop()

        missing = [c for c in feature_names if c not in test_data.columns]
        # If missing features are course-level aggregates, compute them on the fly from the dataset
        if missing:
            # mapping of original columns to the aggregated mean column names expected by the preprocessor
            mean_mapping = {
                'rating': 'mean_rating',
                'course_duration_hours': 'mean_course_duration_hours',
                'course_price': 'mean_course_price',
                'feedback_score': 'mean_feedback_score',
                'time_spent_hours': 'mean_time_spent_hours',
                'previous_courses_taken': 'mean_previous_courses_taken'
            }

            # determine which mean columns we can compute from existing raw columns
            computable = [(orig, mean_col) for orig, mean_col in mean_mapping.items() if mean_col in missing and orig in df.columns]
            if computable:
                st.info("Computing course-level aggregate features from dataset to enable predictions based on `user_id` only.")
                agg_dict = {orig: 'mean' for orig, _ in computable}
                course_agg = df.groupby('course_id').agg(agg_dict).rename(columns={orig: mean_col for orig, mean_col in computable}).reset_index()
                test_data = test_data.merge(course_agg, on='course_id', how='left')

                # recompute missing after merge
                missing = [c for c in feature_names if c not in test_data.columns]

        if missing:
            st.error("The dataset is missing features required by the preprocessor:")
            st.write(missing)
            st.info("Options: (1) provide a dataset that includes these columns, or (2) update the dataset to compute them before launching the app.")
            st.stop()

        X = test_data[feature_names]
        X_trans = preprocessor.transform(X)  # do not fit again!

        preds = model.predict(X_trans)
        test_data["pred_rating"] = preds

        # aggregate predictions per course_id (one row per course)
        if agg_method == 'mean':
            agg_preds = test_data.groupby('course_id', as_index=False).agg({'pred_rating':'mean'})
        else:
            agg_preds = test_data.groupby('course_id', as_index=False).agg({'pred_rating':'max'})

        # attach representative course metadata (name, instructor, and fields for filtering)
        course_meta = df.drop_duplicates('course_id')[['course_id','course_name','instructor','difficulty_level','certification_offered','course_price','enrollment_numbers']]
        agg_preds = agg_preds.merge(course_meta, on='course_id', how='left')

        # filter courses not taken by user
        taken = user_data["course_id"].unique()
        recs = agg_preds[~agg_preds["course_id"].isin(taken)]

        # apply UI filters
        if cert_choice != 'Any':
            recs = recs[recs['certification_offered'] == cert_choice]
        if difficulty_choice:
            recs = recs[recs['difficulty_level'].isin(difficulty_choice)]
        if instructor_choice != 'Any':
            recs = recs[recs['instructor'] == instructor_choice]
        recs = recs[(recs['course_price'] >= price_choice[0]) & (recs['course_price'] <= price_choice[1])]
        recs = recs[recs['pred_rating'] >= min_pred]

        if recs.empty:
            st.warning("No recommendations available — the user may have taken all courses or there are no candidate courses.")
        else:
            top_courses = recs.sort_values(by="pred_rating", ascending=False).head(N)
            top_courses['pred_rating'] = top_courses['pred_rating'].round(3)
            st.subheader(f"⭐ Top {N} Recommended Courses for User {user_id}")
            st.table(top_courses[["course_id","course_name","instructor","pred_rating"]])

            # CSV download
            csv = top_courses.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download recommendations as CSV", data=csv, file_name=f"recommendations_user_{user_id}.csv", mime='text/csv')

            st.success("Recommendation Generated Successfully!")
