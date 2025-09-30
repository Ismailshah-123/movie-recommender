import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import gdown
import os

st.set_page_config(layout="wide")

# -----------------------------
# 0️⃣ Download Large Files from Google Drive
# -----------------------------
def download_from_drive(file_id, output_name):
    if not os.path.exists(output_name):  # only download if file not already present
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_name, quiet=False)

# Download preds_df.pkl
download_from_drive("1PIolsTRggxPWk_8WA3NHbVKG03Ldkfke", "preds_df.pkl")

# Download similarity.pkl
download_from_drive("1-tL3n38J8pRlEmTxYKeneYROlMkASmew", "similarity.pkl")

# -----------------------------
# 1️⃣ Caching functions
# -----------------------------
@st.cache_data
def load_movies(file):
    df = pd.read_csv(file)
    if 'tags' in df.columns:
        df['genres_list'] = df['tags'].apply(lambda x: x.split('|') if pd.notna(x) else [])
    elif 'genres' in df.columns:
        df['genres_list'] = df['genres'].apply(lambda x: x.split('|') if pd.notna(x) else [])
    else:
        df['genres_list'] = [[] for _ in range(len(df))]
    return df

@st.cache_data
def load_preds(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_similarity(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

@st.cache_data
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        data = requests.get(url).json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
    except:
        pass
    return None

@st.cache_data
def fetch_trailer(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        data = requests.get(url).json()
        results = data.get('results', [])
        for video in results:
            if video['type'] == 'Trailer' and video['site'] == 'YouTube':
                return f"https://www.youtube.com/watch?v={video['key']}"
    except:
        pass
    return None

@st.cache_data
def fetch_movie_details(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        data = requests.get(url).json()
        return {
            "overview": data.get('overview', ''),
            "genres": [g['name'] for g in data.get('genres', [])],
            "homepage": data.get('homepage', f"https://www.themoviedb.org/movie/{movie_id}"),
            "release_date": data.get('release_date', ''),
            "rating": data.get('vote_average', 0),
        }
    except:
        return {"overview": "", "genres": [], "homepage": "", "release_date": "", "rating": 0}

# -----------------------------
# 2️⃣ Load data
# -----------------------------
movies = load_movies("movies2.csv")
preds_df = load_preds("preds_df.pkl")
similarity = load_similarity("similarity.pkl")

# -----------------------------
# 3️⃣ UI Header & Filters
# -----------------------------
st.title("🎬 Movie Recommender")
st.write("Hybrid recommender ")

with st.sidebar:
    st.header("Filters")
    user_ids = preds_df.index.tolist()
    selected_user = st.selectbox("Select User:", user_ids)
    selected_movie = st.selectbox("Search Movie:", movies['title'].values)
    all_genres = sorted(set(g for gs in movies['genres_list'] for g in gs))
    selected_genres = st.multiselect("Filter by genres:", all_genres)

# -----------------------------
# 4️⃣ Hybrid Recommendation Function
# -----------------------------
def hybrid_recommend(user_id, movie_title, movies_df, preds_df, similarity, top_n=10, alpha=0.1):
    if selected_genres:
        filtered_movies = movies_df[movies_df['genres_list'].apply(lambda g_list: any(g in g_list for g in selected_genres))]
    else:
        filtered_movies = movies_df

    if movie_title not in filtered_movies['title'].values:
        return pd.DataFrame(columns=['title', 'poster', 'hybrid_score', 'id'])

    cf_scores_user = preds_df.loc[user_id]
    cf_norm = (cf_scores_user - cf_scores_user.min()) / (cf_scores_user.max() - cf_scores_user.min())
    cf_norm = cf_norm.reindex(filtered_movies.index, fill_value=0)

    idx = filtered_movies.index[filtered_movies['title'] == movie_title][0]
    content_norm = similarity[idx]
    content_norm = (content_norm - content_norm.min()) / (content_norm.max() - content_norm.min())
    content_norm = pd.Series(content_norm, index=filtered_movies.index)

    hybrid_scores = alpha * cf_norm + (1 - alpha) * content_norm

    top_idx = hybrid_scores.sort_values(ascending=False).head(top_n).index
    recommendations = filtered_movies.loc[top_idx].copy()
    recommendations['hybrid_score'] = hybrid_scores.loc[top_idx].values
    recommendations['poster'] = recommendations['id'].apply(fetch_poster)
    recommendations['trailer'] = recommendations['id'].apply(fetch_trailer)
    return recommendations[['title', 'poster', 'trailer', 'hybrid_score', 'id']]

# -----------------------------
# 5️⃣ Trending Function
# -----------------------------
def trending_movies(movies_df, top_n=10):
    trending = movies_df.sample(top_n)
    trending['poster'] = trending['id'].apply(fetch_poster)
    trending['trailer'] = trending['id'].apply(fetch_trailer)
    return trending[['title', 'poster', 'trailer', 'id']]

# -----------------------------
# 6️⃣ Display Horizontal Scroll Row
# -----------------------------
def display_row(df):
    cols = st.columns(len(df))
    for i, col in enumerate(cols):
        title = df['title'].iloc[i]
        poster = df['poster'].iloc[i]
        trailer = df['trailer'].iloc[i]
        movie_id = df['id'].iloc[i]
        if poster:
            if trailer:
                col.markdown(f"[**{title}**]({trailer})")
            else:
                col.markdown(f"**{title}**")
            col.image(poster, use_container_width=True)
            details = fetch_movie_details(movie_id)
            if details['overview']:
                col.caption(details['overview'][:120]+"...")

# -----------------------------
# 7️⃣ Show Hybrid Recommendations
# -----------------------------
st.subheader("🔥Recommendations")
if st.button("Show Recommendations"):
    with st.spinner("Fetching recommendations..."):
        recs = hybrid_recommend(selected_user, selected_movie, movies, preds_df, similarity, top_n=10, alpha=0.1)
        if recs.empty:
            st.warning("No recommendations found!")
        else:
            display_row(recs)

# -----------------------------
# 8️⃣ Trending Row
# -----------------------------
st.subheader("🔥 Trending Now")
trending = trending_movies(movies, top_n=10)
display_row(trending)

# -----------------------------
# 9️⃣ Watch History Simulation
# -----------------------------
st.subheader("🎥 Recommended Based on Your Watch History")
watched_movies = movies.sample(10)
watched_movies['poster'] = watched_movies['id'].apply(fetch_poster)
watched_movies['trailer'] = watched_movies['id'].apply(fetch_trailer)
display_row(watched_movies)
