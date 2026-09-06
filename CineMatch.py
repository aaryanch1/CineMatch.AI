"""
AI Recommendation Logic System — Streamlit Dashboard
=====================================================
An advanced, interactive dashboard version of the content-based movie
recommendation engine. Built on the same transparent, explainable core
as the CLI version:

    User Intent -> Preference Representation -> Pattern Matching
    -> Similarity Score -> Ranking -> Recommendation

New in this version (beyond the CLI script):
    - Full interactive UI: multiselect / sliders / live filtering
    - Live-tunable scoring weights (see exactly how changing a weight
      reshapes the ranking — great for demonstrating explainability)
    - Score breakdown charts (rule-based vs. cosine-similarity contribution)
    - A dataset explorer tab with genre/language/rating distributions
    - Downloadable CSV of your personalized recommendations
    - A "How It Works" tab that documents the math live, using your
      current slider settings

This single file is fully self-contained:
    - The dataset is embedded as a string (no CSV file needed).
    - Missing libraries (pandas, numpy, scikit-learn) are auto-installed
      on first run — you only need Python and Streamlit itself.

Run with:
    pip install streamlit
    streamlit run streamlit_app.py
"""

import subprocess
import sys


def _ensure_installed(pip_name: str, import_name: str = None) -> None:
    """
    Make sure a required package is available, installing it silently
    via pip if it's missing. Keeps this file runnable with nothing more
    than `pip install streamlit` beforehand.
    """
    import_name = import_name or pip_name
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", pip_name]
        )


for _pkg, _mod in [("pandas", "pandas"), ("numpy", "numpy"), ("scikit-learn", "sklearn")]:
    _ensure_installed(_pkg, _mod)

from dataclasses import dataclass, field
from io import StringIO
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===========================================================================
# 1. DATASET
# Embedded directly so the app is fully self-contained and needs no
# separate CSV upload to run.
# ===========================================================================

MOVIE_DATA_CSV = """title,genre,language,mood,year,rating,age_group
Interstellar,Sci-Fi,English,Adventure,2014,8.7,13+
The Martian,Sci-Fi,English,Inspiring,2015,8.0,13+
Inception,Sci-Fi,English,Thrilling,2010,8.8,13+
Arrival,Sci-Fi,English,Thought-Provoking,2016,7.9,13+
Interstellar Wars,Sci-Fi,English,Exciting,2021,7.2,13+
Dune,Sci-Fi,English,Epic,2021,8.1,13+
Gravity,Sci-Fi,English,Tense,2013,7.7,13+
Ex Machina,Sci-Fi,English,Thought-Provoking,2014,7.7,18+
The Matrix,Sci-Fi,English,Thrilling,1999,8.7,13+
Guardians of the Galaxy,Sci-Fi,English,Funny,2014,8.0,13+
The Hangover,Comedy,English,Funny,2009,7.7,18+
Superbad,Comedy,English,Funny,2007,7.6,18+
Bridesmaids,Comedy,English,Funny,2011,6.8,18+
The Grand Budapest Hotel,Comedy,English,Quirky,2014,8.1,13+
Dil Chahta Hai,Comedy,Hindi,Feel-Good,2001,8.1,13+
3 Idiots,Comedy,Hindi,Feel-Good,2009,8.4,U
Zindagi Na Milegi Dobara,Comedy,Hindi,Adventure,2011,8.2,13+
Andaz Apna Apna,Comedy,Hindi,Funny,1994,8.1,U
Munna Bhai MBBS,Comedy,Hindi,Funny,2003,8.1,U
The Shawshank Redemption,Drama,English,Emotional,1994,9.3,13+
Forrest Gump,Drama,English,Emotional,1994,8.8,13+
A Beautiful Mind,Drama,English,Emotional,2001,8.2,13+
Pursuit of Happyness,Drama,English,Inspiring,2006,8.0,U
Dangal,Drama,Hindi,Inspiring,2016,8.4,U
Taare Zameen Par,Drama,Hindi,Emotional,2007,8.3,U
Pyaasa,Drama,Urdu,Emotional,1957,8.2,13+
Bol,Drama,Urdu,Emotional,2011,8.0,13+
Khuda Kay Liye,Drama,Urdu,Thought-Provoking,2007,7.9,13+
Ramchand Pakistani,Drama,Urdu,Emotional,2008,7.4,13+
The Notebook,Romance,English,Emotional,2004,7.8,13+
La La Land,Romance,English,Bittersweet,2016,8.0,13+
Jab We Met,Romance,Hindi,Feel-Good,2007,8.1,U
Kabhi Khushi Kabhie Gham,Romance,Hindi,Emotional,2001,7.4,U
John Wick,Action,English,Exciting,2014,7.4,18+
Mad Max: Fury Road,Action,English,Exciting,2015,8.1,18+
The Dark Knight,Action,English,Thrilling,2008,9.0,13+
Extraction,Action,English,Exciting,2020,6.7,18+
War,Action,Hindi,Exciting,2019,6.5,13+
Pathaan,Action,Hindi,Exciting,2023,6.4,13+
The Conjuring,Horror,English,Scary,2013,7.5,18+
Hereditary,Horror,English,Scary,2018,7.3,18+
A Quiet Place,Horror,English,Tense,2018,7.5,13+
Get Out,Horror,English,Thought-Provoking,2017,7.7,18+
Parasite,Drama,English,Thought-Provoking,2019,8.5,18+
Whiplash,Drama,English,Tense,2014,8.5,13+
Coco,Family,English,Feel-Good,2017,8.4,U
Zootopia,Family,English,Funny,2016,8.0,U
Spirited Away,Family,English,Adventure,2001,8.6,U
"""

REQUIRED_COLUMNS = ["title", "genre", "language", "mood", "year", "rating", "age_group"]
TEXT_COLUMNS = ["genre", "language", "mood", "age_group"]

DEFAULT_WEIGHTS = {"genre": 40, "mood": 20, "language": 15, "rating": 15, "year": 10}
DEFAULT_COSINE_BLEND = 0.25
LOW_CONFIDENCE_THRESHOLD = 40.0


# ===========================================================================
# 2. DATA LOADING (cached so the app doesn't re-parse on every interaction)
# ===========================================================================

class DataLoadError(Exception):
    pass


@st.cache_data(show_spinner=False)
def load_data(uploaded_csv_bytes: Optional[bytes] = None) -> pd.DataFrame:
    """
    Load the movie dataset. If the user uploaded their own CSV via the
    sidebar, use that (must match the required schema); otherwise fall
    back to the embedded dataset.
    """
    try:
        if uploaded_csv_bytes:
            df = pd.read_csv(StringIO(uploaded_csv_bytes.decode("utf-8")))
        else:
            df = pd.read_csv(StringIO(MOVIE_DATA_CSV))
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(f"Failed to parse dataset: {exc}")

    if df.empty:
        raise DataLoadError("The dataset contains no rows.")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataLoadError(f"Dataset is missing required columns: {missing_cols}")

    df = df.dropna(subset=["title"]).copy()

    for col in TEXT_COLUMNS:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        df[f"{col}_norm"] = df[col].str.lower()

    df["title"] = df["title"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    return df.reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def build_tfidf_matrix(_df: pd.DataFrame):
    """
    Build the TF-IDF matrix over each movie's genre+mood+language profile.
    Cached as a resource since the vectorizer/matrix are reused across
    every recommendation request as long as the dataset doesn't change.
    """
    combined_text = _df["genre_norm"] + " " + _df["mood_norm"] + " " + _df["language_norm"]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_text)
    return vectorizer, tfidf_matrix


# ===========================================================================
# 3. PREFERENCES + SCORING LOGIC
# ===========================================================================

@dataclass
class UserPreferences:
    genres: List[str] = field(default_factory=list)
    language: Optional[str] = None
    mood: Optional[str] = None
    min_rating: float = 0.0
    year_start: Optional[int] = None
    year_end: Optional[int] = None
    num_recommendations: int = 5


def _safe_lower(value) -> str:
    return str(value).strip().lower() if value is not None else ""


def calculate_match_score(movie_row: pd.Series, prefs: UserPreferences, weights: dict) -> float:
    """Weighted rule-based match score for one movie, using live weights."""
    score = 0.0

    if prefs.genres:
        user_genres = {_safe_lower(g) for g in prefs.genres}
        if _safe_lower(movie_row.get("genre", "")) in user_genres:
            score += weights["genre"]

    if prefs.mood and _safe_lower(movie_row.get("mood", "")) == _safe_lower(prefs.mood):
        score += weights["mood"]

    if prefs.language and _safe_lower(movie_row.get("language", "")) == _safe_lower(prefs.language):
        score += weights["language"]

    if prefs.min_rating is not None:
        movie_rating = movie_row.get("rating", np.nan)
        if pd.notna(movie_rating) and movie_rating >= prefs.min_rating:
            score += weights["rating"]

    if prefs.year_start is not None and prefs.year_end is not None:
        movie_year = movie_row.get("year", np.nan)
        if pd.notna(movie_year) and prefs.year_start <= movie_year <= prefs.year_end:
            score += weights["year"]

    return score


def find_matching_attributes(movie_row: pd.Series, prefs: UserPreferences) -> List[str]:
    matches = []

    if prefs.genres:
        user_genres = {_safe_lower(g) for g in prefs.genres}
        if _safe_lower(movie_row.get("genre", "")) in user_genres:
            matches.append(f"✅ Genre matches **{movie_row['genre']}**")

    if prefs.mood and _safe_lower(movie_row.get("mood", "")) == _safe_lower(prefs.mood):
        matches.append(f"✅ Mood matches **{movie_row['mood']}**")

    if prefs.language and _safe_lower(movie_row.get("language", "")) == _safe_lower(prefs.language):
        matches.append(f"✅ Language matches **{movie_row['language']}**")

    if prefs.min_rating is not None:
        movie_rating = movie_row.get("rating", np.nan)
        if pd.notna(movie_rating) and movie_rating >= prefs.min_rating:
            matches.append(f"✅ Rating **{movie_rating}** ≥ your minimum of **{prefs.min_rating}**")

    if prefs.year_start is not None and prefs.year_end is not None:
        movie_year = movie_row.get("year", np.nan)
        if pd.notna(movie_year) and prefs.year_start <= movie_year <= prefs.year_end:
            matches.append(f"✅ Released **{int(movie_year)}** (within {prefs.year_start}-{prefs.year_end})")

    return matches


def calculate_cosine_scores(df: pd.DataFrame, prefs: UserPreferences,
                             vectorizer: TfidfVectorizer, tfidf_matrix) -> np.ndarray:
    user_terms = " ".join(
        [_safe_lower(g) for g in prefs.genres]
        + [_safe_lower(prefs.mood)]
        + [_safe_lower(prefs.language)]
    ).strip()

    if not user_terms:
        return np.zeros(df.shape[0])

    user_vector = vectorizer.transform([user_terms])
    return cosine_similarity(user_vector, tfidf_matrix).flatten()


def rank_recommendations(df: pd.DataFrame, prefs: UserPreferences,
                          weights: dict, cosine_blend: float) -> pd.DataFrame:
    """Score, blend, and rank every movie against the user's preferences."""
    if df.empty:
        return df

    vectorizer, tfidf_matrix = build_tfidf_matrix(df)
    cosine_scores = calculate_cosine_scores(df, prefs, vectorizer, tfidf_matrix)

    rule_scores = df.apply(lambda row: calculate_match_score(row, prefs, weights), axis=1)
    matched_attrs = df.apply(lambda row: find_matching_attributes(row, prefs), axis=1)

    max_possible = sum(weights.values()) or 1
    rule_percent = (rule_scores / max_possible) * 100
    cosine_percent = cosine_scores * 100

    final_percent = (1 - cosine_blend) * rule_percent + cosine_blend * cosine_percent

    result = df.copy()
    result["rule_percent"] = rule_percent.round(1)
    result["cosine_percent"] = cosine_percent.round(1)
    result["match_percent"] = final_percent.round(1)
    result["matched_attributes"] = matched_attrs

    return result.sort_values(by="match_percent", ascending=False).reset_index(drop=True)


def apply_hard_filters(df: pd.DataFrame, prefs: UserPreferences) -> pd.DataFrame:
    """
    Optional hard cutoffs (used only for the Dataset Explorer's filtered
    view, NOT for the main recommender — the recommender intentionally
    keeps everything in play so low-confidence "closest match" results
    can still surface instead of vanishing entirely).
    """
    filtered = df.copy()
    if prefs.min_rating:
        filtered = filtered[filtered["rating"].fillna(0) >= prefs.min_rating]
    if prefs.year_start is not None and prefs.year_end is not None:
        filtered = filtered[
            filtered["year"].between(prefs.year_start, prefs.year_end, inclusive="both")
            | filtered["year"].isna() == False  # noqa: E712
        ]
    return filtered


# ===========================================================================
# 4. STREAMLIT UI
# ===========================================================================

st.set_page_config(
    page_title="AI Recommendation Logic System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .match-badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-weight: 600; font-size: 0.85rem; color: white;
    }
    .low-conf { background-color: #d97706; }
    .med-conf { background-color: #2563eb; }
    .high-conf { background-color: #16a34a; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎬 AI Recommendation Logic System")
st.caption(
    "A transparent, explainable content-based recommender — weighted rule "
    "matching blended with TF-IDF cosine similarity. No black-box models."
)

# ---------------------------------------------------------------------------
# Sidebar: dataset source + user preferences
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📂 Dataset")
    uploaded = st.file_uploader(
        "Optional: upload your own movies.csv",
        type=["csv"],
        help=f"Must include columns: {', '.join(REQUIRED_COLUMNS)}",
    )

    try:
        df = load_data(uploaded.getvalue() if uploaded else None)
    except DataLoadError as err:
        st.error(f"Could not load dataset: {err}")
        st.stop()

    st.success(f"Loaded {len(df)} titles.")

    st.header("🎯 Your Preferences")

    all_genres = sorted(df["genre"].unique())
    all_languages = sorted(df["language"].unique())
    all_moods = sorted(df["mood"].unique())

    sel_genres = st.multiselect("Favorite genre(s)", all_genres, default=[])
    sel_language = st.selectbox("Preferred language", ["Any"] + all_languages)
    sel_mood = st.selectbox("Preferred mood", ["Any"] + all_moods)
    sel_min_rating = st.slider("Minimum rating", 0.0, 10.0, 7.0, 0.1)

    min_year, max_year = int(df["year"].min()), int(df["year"].max())
    sel_year_range = st.slider("Release year range", min_year, max_year, (min_year, max_year))

    sel_num_recs = st.slider("Number of recommendations", 1, min(15, len(df)), 5)

    with st.expander("⚙️ Advanced: tune the scoring engine"):
        st.caption("See how reweighting each signal reshapes the ranking in real time.")
        w_genre = st.slider("Genre weight", 0, 60, DEFAULT_WEIGHTS["genre"])
        w_mood = st.slider("Mood weight", 0, 60, DEFAULT_WEIGHTS["mood"])
        w_language = st.slider("Language weight", 0, 60, DEFAULT_WEIGHTS["language"])
        w_rating = st.slider("Rating weight", 0, 60, DEFAULT_WEIGHTS["rating"])
        w_year = st.slider("Year weight", 0, 60, DEFAULT_WEIGHTS["year"])
        cosine_blend = st.slider(
            "Cosine-similarity blend", 0.0, 1.0, DEFAULT_COSINE_BLEND, 0.05,
            help="0 = pure rule-based scoring, 1 = pure TF-IDF cosine similarity",
        )
        if st.button("Reset to defaults"):
            st.rerun()

    weights = {"genre": w_genre, "mood": w_mood, "language": w_language,
               "rating": w_rating, "year": w_year}

    generate_clicked = st.button("🔍 Generate Recommendations", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Build UserPreferences from sidebar state
# ---------------------------------------------------------------------------

prefs = UserPreferences(
    genres=sel_genres,
    language=None if sel_language == "Any" else sel_language,
    mood=None if sel_mood == "Any" else sel_mood,
    min_rating=sel_min_rating,
    year_start=sel_year_range[0],
    year_end=sel_year_range[1],
    num_recommendations=sel_num_recs,
)

# Persist results across reruns so switching tabs doesn't lose them
if generate_clicked or "results" not in st.session_state:
    with st.spinner("Scoring and ranking movies..."):
        st.session_state["results"] = rank_recommendations(df, prefs, weights, cosine_blend)
        st.session_state["prefs"] = prefs
        st.session_state["weights"] = weights
        st.session_state["cosine_blend"] = cosine_blend

results = st.session_state["results"]
active_prefs = st.session_state["prefs"]
active_weights = st.session_state["weights"]
active_blend = st.session_state["cosine_blend"]

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_recs, tab_scores, tab_explore, tab_how = st.tabs(
    ["🎯 Recommendations", "📊 Score Analysis", "🗂️ Dataset Explorer", "ℹ️ How It Works"]
)

# --- Tab 1: Recommendations -------------------------------------------------
with tab_recs:
    top_n = results.head(active_prefs.num_recommendations)

    if top_n.empty:
        st.warning("No movies matched, and no close matches could be found either.")
    else:
        if active_prefs.num_recommendations > len(results):
            st.info(f"Only {len(results)} titles exist in the dataset — showing all of them.")

        for i, row in top_n.iterrows():
            score = row["match_percent"]
            if score >= 70:
                badge_class, badge_label = "high-conf", "Strong Match"
            elif score >= LOW_CONFIDENCE_THRESHOLD:
                badge_class, badge_label = "med-conf", "Good Match"
            else:
                badge_class, badge_label = "low-conf", "Closest Available"

            with st.container(border=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.subheader(f"{i + 1}. {row['title']}")
                    st.markdown(
                        f"<span class='match-badge {badge_class}'>{badge_label} — {score}%</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"🎭 {row['genre']}  •  🌐 {row['language']}  •  🎨 {row['mood']}  •  "
                        f"📅 {int(row['year']) if pd.notna(row['year']) else 'N/A'}  •  "
                        f"⭐ {row['rating']}  •  🔞 {row['age_group']}"
                    )
                with col2:
                    st.metric("Match", f"{score}%")

                with st.expander("Why was this recommended?"):
                    if row["matched_attributes"]:
                        for attr in row["matched_attributes"]:
                            st.markdown(f"- {attr}")
                    else:
                        st.markdown("No preferences directly matched — shown as a "
                                    "closest-available suggestion so you still get options.")
                    st.progress(min(int(row["rule_percent"]), 100),
                                text=f"Rule-based score: {row['rule_percent']}%")
                    st.progress(min(int(row["cosine_percent"]), 100),
                                text=f"Cosine similarity score: {row['cosine_percent']}%")

        csv_data = top_n[["title", "genre", "language", "mood", "year", "rating",
                           "age_group", "match_percent"]].to_csv(index=False)
        st.download_button(
            "⬇️ Download these recommendations as CSV",
            data=csv_data,
            file_name="my_movie_recommendations.csv",
            mime="text/csv",
        )

# --- Tab 2: Score Analysis ---------------------------------------------------
with tab_scores:
    top_n = results.head(active_prefs.num_recommendations)

    if top_n.empty:
        st.info("Generate recommendations first to see the score breakdown.")
    else:
        st.subheader("Rule-based vs. Cosine-similarity contribution")
        st.caption(
            "Each bar shows how much of a title's final score came from exact "
            "attribute matching versus TF-IDF cosine similarity."
        )
        chart_df = top_n.set_index("title")[["rule_percent", "cosine_percent"]]
        chart_df.columns = ["Rule-based %", "Cosine similarity %"]
        st.bar_chart(chart_df)

        st.subheader("Final match percentage")
        st.bar_chart(top_n.set_index("title")[["match_percent"]])

        st.subheader("Full scored table")
        st.dataframe(
            top_n[["title", "genre", "language", "mood", "year", "rating",
                   "rule_percent", "cosine_percent", "match_percent"]],
            use_container_width=True,
            hide_index=True,
        )

# --- Tab 3: Dataset Explorer --------------------------------------------------
with tab_explore:
    st.subheader("Browse the full catalog")
    filter_cols = st.columns(3)
    with filter_cols[0]:
        explore_genre = st.multiselect("Filter by genre", all_genres, key="explore_genre")
    with filter_cols[1]:
        explore_lang = st.multiselect("Filter by language", all_languages, key="explore_lang")
    with filter_cols[2]:
        explore_rating = st.slider("Minimum rating", 0.0, 10.0, 0.0, 0.1, key="explore_rating")

    explore_df = df.copy()
    if explore_genre:
        explore_df = explore_df[explore_df["genre"].isin(explore_genre)]
    if explore_lang:
        explore_df = explore_df[explore_df["language"].isin(explore_lang)]
    explore_df = explore_df[explore_df["rating"].fillna(0) >= explore_rating]

    st.dataframe(
        explore_df[["title", "genre", "language", "mood", "year", "rating", "age_group"]],
        use_container_width=True,
        hide_index=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("Titles per genre")
        st.bar_chart(df["genre"].value_counts())
    with col_b:
        st.caption("Titles per language")
        st.bar_chart(df["language"].value_counts())

# --- Tab 4: How It Works ------------------------------------------------------
with tab_how:
    st.subheader("Pipeline")
    st.markdown(
        """
        ```
        User Input
            ↓
        Preference Processing
            ↓
        Feature Matching (weighted rules)
            ↓
        Similarity / Weighted Score (TF-IDF + Cosine, live-blended)
            ↓
        Ranking
            ↓
        Top-N Recommendations
        ```
        """
    )

    st.subheader("Current scoring weights (live — edit them in the sidebar)")
    weight_df = pd.DataFrame(
        {"Attribute": list(active_weights.keys()), "Weight": list(active_weights.values())}
    )
    st.table(weight_df)
    st.caption(
        f"Max possible rule-based score: **{sum(active_weights.values())}**  •  "
        f"Cosine-similarity blend: **{active_blend * 100:.0f}%** of the final score"
    )

    st.subheader("Why content-based filtering, not collaborative filtering?")
    st.markdown(
        """
        - **Content-based filtering** (used here) maps user preferences directly
          to item attributes — it works immediately with no historical
          interaction data, and every result is traceable back to *which*
          preference caused it. This is why it was chosen for this milestone.
        - **Collaborative filtering** instead relies on patterns across many
          users' behavior ("people who liked X also liked Y"). It needs a large
          interaction history and is far less explainable — listed here as a
          possible future extension, not part of this build.
        """
    )

    st.subheader("Why cosine similarity, not Euclidean distance?")
    st.markdown(
        """
        Euclidean distance measures straight-line distance and is sensitive to
        vector *magnitude* — an item with more tags can look artificially
        "farther" even if it shares the same core attributes. Cosine similarity
        instead measures the *angle* between the user's preference vector and
        each movie's TF-IDF vector, so it stays focused on **orientation**
        (shared characteristics) rather than raw size. A score of 1 means
        perfectly aligned preferences; 0 means no shared characteristics.
        """
    )

    st.subheader("Handling edge cases")
    st.markdown(
        """
        - **No preferences given** → every movie is shown, ranked by rating/recency signal only.
        - **No strong matches** → the closest available titles are still shown,
          clearly flagged as *low-confidence* rather than hidden.
        - **Requesting more recommendations than exist** → the app shows the
          full dataset instead of erroring.
        - **Case differences** (`sci-fi` / `Sci-Fi` / `SCI-FI`) → normalized
          internally, so they always match consistently.
        """
    )

st.divider()
st.caption(
    "AI Recommendation Logic System · Content-based filtering with weighted "
    "scoring + TF-IDF cosine similarity · Built for Project 3, Batch 2026"
)
