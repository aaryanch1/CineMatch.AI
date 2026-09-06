# 🎬 AI Recommendation Logic System

A professional, interactive **content-based movie recommendation system** built with Python and Streamlit.

The system demonstrates the fundamental principles of recommendation logic by matching **user preferences with movie attributes** using a combination of:

- Weighted rule-based preference matching
- TF-IDF vectorization
- Cosine similarity
- Score blending
- Recommendation ranking
- Explainable recommendation logic

The project is designed around a simple but powerful concept:

```text
User Intent
     ↓
Preference Representation
     ↓
Pattern Matching
     ↓
Similarity Score
     ↓
Ranking
     ↓
Personalized Recommendations
```

Unlike black-box recommendation systems, this project makes the recommendation process transparent and allows users to understand **why a particular movie was recommended**.

---

# 📌 Project Overview

The **AI Recommendation Logic System** is a content-based recommendation engine that recommends movies according to the user's stated preferences.

The user can select:

- Favorite genre(s)
- Preferred language
- Preferred mood
- Minimum rating
- Release-year range
- Number of recommendations

The system evaluates every movie in the dataset and calculates a recommendation score based on how closely each movie matches the user's preferences.

The final recommendations are ranked from highest to lowest match percentage.

The application also provides an interactive dashboard where users can:

- Generate personalized recommendations
- Analyze recommendation scores
- Explore the complete dataset
- Adjust recommendation weights
- View the recommendation methodology
- Download recommendations as a CSV file
- Upload their own compatible movie dataset

The application is implemented as a single self-contained Streamlit file, with the default movie dataset embedded directly into the application.

---

# 🎯 Project Objective

The main objective of this project is to demonstrate how **user intent can be translated into algorithmic recommendations** without relying on deep learning or complex neural recommendation architectures.

The project focuses on the fundamentals of:

- Logic building
- Pattern matching
- Content-based filtering
- Feature representation
- Similarity measurement
- Weighted scoring
- Ranking algorithms
- Explainable AI concepts

The recommendation engine does not randomly select movies.

Instead, every recommendation is generated from the relationship between:

```text
User Preferences
        +
Movie Attributes
        ↓
Matching Algorithm
        ↓
Recommendation Score
```

---

# 🚀 Key Features

## 1. Interactive User Preference Input

Users can select their preferences directly from the Streamlit sidebar.

Available inputs include:

- Multiple genres
- Preferred language
- Preferred mood
- Minimum movie rating
- Release-year range
- Number of recommendations

The available genres, languages, and moods are dynamically obtained from the loaded dataset.

---

## 2. Weighted Rule-Based Recommendation

The primary recommendation mechanism uses a transparent weighted scoring system.

The default scoring weights are:

| Attribute | Weight |
|---|---:|
| Genre | 40 |
| Mood | 20 |
| Language | 15 |
| Rating | 15 |
| Release Year | 10 |
| **Total** | **100** |

These weights determine how strongly each preference influences the final rule-based score.

For example:

```text
Genre Match      → +40
Mood Match       → +20
Language Match   → +15
Rating Match     → +15
Year Match       → +10
```

A movie matching all five conditions can receive the maximum rule-based score.

---

# 🧠 Recommendation Algorithm

The recommendation engine uses two complementary approaches.

## A. Rule-Based Matching

The system directly compares user preferences with movie attributes.

For example:

```text
User:
Genre = Sci-Fi
Mood = Adventure
Language = English
Minimum Rating = 7.5
Year Range = 2010–2026
```

A movie receives points when its attributes satisfy these conditions.

The implementation performs case-insensitive comparisons so values such as:

```text
Sci-Fi
sci-fi
SCI-FI
```

are treated consistently.

---

# 📐 Rule-Based Score

The raw rule score is normalized into a percentage.

The system calculates:

```text
Rule Percentage =
Rule Score / Maximum Possible Rule Score × 100
```

This produces an easy-to-understand score such as:

```text
Rule-based Score: 85%
```

The maximum possible score is dynamically calculated from the active scoring weights.

---

# 🔍 TF-IDF and Cosine Similarity

In addition to exact attribute matching, the project uses **TF-IDF vectorization** and **cosine similarity**.

The textual movie profile is created from:

- Genre
- Mood
- Language

These attributes are combined into a text representation and converted into TF-IDF vectors.

The user's selected genre, mood, and language are also converted into a text representation.

The system then calculates the cosine similarity between:

```text
User Preference Vector
        ↓
Movie TF-IDF Vector
```

The resulting similarity value is converted into a percentage.

---

# 📊 Why Cosine Similarity?

Cosine similarity measures the angle between two vectors rather than simply measuring their physical distance.

Conceptually:

```text
Cosine Similarity = Similarity of vector direction
```

A value closer to:

```text
1.0 → Highly similar
0.0 → Little or no similarity
```

is used to determine how closely the movie's textual attributes align with the user's preferences.

This makes cosine similarity useful for comparing the preference representation against the movie profiles.

---

# ⚖️ Hybrid Scoring

The final recommendation score combines:

1. Rule-based matching
2. TF-IDF cosine similarity

The default cosine blend is:

```text
25%
```

Therefore, by default:

```text
Final Score =
75% Rule-Based Score
+
25% Cosine Similarity
```

The application calculates:

```text
Final Score =
(1 - cosine_blend) × Rule Score
+
cosine_blend × Cosine Score
```

This creates a hybrid content-based recommendation score.

---

# 🎛️ Live Scoring Controls

One of the major features of the dashboard is the ability to modify the recommendation weights interactively.

Users can change:

- Genre weight
- Mood weight
- Language weight
- Rating weight
- Year weight
- Cosine-similarity contribution

The cosine similarity blend can range from:

```text
0% → Pure rule-based scoring
```

to:

```text
100% → Pure TF-IDF cosine similarity
```

This allows users to observe how changing the importance of different features changes the recommendation ranking.

---

# 🏆 Recommendation Ranking

After calculating scores for all movies, the system:

1. Calculates the rule-based score.
2. Calculates the cosine similarity score.
3. Blends both scores.
4. Creates the final match percentage.
5. Sorts movies in descending order.
6. Returns the highest-ranked movies.

The ranking is therefore based on calculated relevance rather than random selection.

---

# 💡 Explainable Recommendations

The system is designed to make recommendations understandable.

For each recommended movie, users can expand:

```text
Why was this recommended?
```

The application then displays the preferences that matched.

For example:

```text
✓ Genre matches Sci-Fi
✓ Mood matches Adventure
✓ Language matches English
✓ Rating ≥ your minimum rating
✓ Released within your selected year range
```

The system also displays:

```text
Rule-based score
Cosine similarity score
Final match percentage
```

This makes the recommendation process more interpretable and demonstrates the idea of **Explainable AI**.

---

# 🏷️ Recommendation Confidence

Recommendations are categorized based on their final match percentage.

### Strong Match

```text
Score ≥ 70%
```

### Good Match

```text
40%–69.9%
```

### Closest Available

```text
Below 40%
```

This prevents lower-scoring results from being presented as equally strong recommendations.

Instead, weaker results are clearly identified as **closest available suggestions**.

---

# 📂 Dataset

The default dataset is embedded directly inside the Python application.

This makes the project self-contained and means that a separate CSV file is not required for the default setup.

The dataset contains movie attributes including:

```text
title
genre
language
mood
year
rating
age_group
```

The embedded dataset contains movies from multiple categories including:

- Sci-Fi
- Comedy
- Drama
- Romance
- Action
- Horror
- Family

and languages including:

- English
- Hindi
- Urdu

The dataset is designed primarily for demonstrating the recommendation engine.

---

# 📤 Custom Dataset Upload

The application also supports uploading a custom CSV dataset.

Users can upload their own movie dataset from the sidebar.

The uploaded dataset must contain these required columns:

```text
title
genre
language
mood
year
rating
age_group
```

The application validates the dataset schema before using it.

If required columns are missing, the application reports the problem instead of silently processing an incompatible dataset.

---

# 🗂️ Dataset Explorer

The application includes a dedicated **Dataset Explorer** tab.

Users can:

- Filter movies by genre
- Filter movies by language
- Filter movies by minimum rating
- Browse the movie catalog
- View genre distributions
- View language distributions

The dashboard also displays charts showing the number of titles per genre and language.

---

# 📊 Score Analysis

The **Score Analysis** tab provides insight into how recommendations are generated.

It displays:

### Rule-Based Contribution

Shows the percentage produced by exact attribute matching.

### Cosine Similarity Contribution

Shows the percentage contributed by TF-IDF cosine similarity.

### Final Match Percentage

Shows the final blended recommendation score.

A complete scored table is also provided for the recommended movies.

---

# ⬇️ Download Recommendations

Users can download their personalized recommendations as a CSV file.

The exported file contains:

```text
title
genre
language
mood
year
rating
age_group
match_percent
```

The default filename is:

```text
my_movie_recommendations.csv
```

This makes the system useful not only for demonstration but also for exporting recommendation results for further analysis.

---

# ℹ️ How It Works

The application includes a dedicated **How It Works** section explaining the recommendation pipeline.

The complete pipeline is:

```text
User Input
     ↓
Preference Processing
     ↓
Feature Matching
     ↓
Weighted Scoring
     ↓
TF-IDF + Cosine Similarity
     ↓
Score Blending
     ↓
Ranking
     ↓
Top-N Recommendations
```

The dashboard also displays the currently active scoring weights and cosine-similarity blend.

---

# 🤖 Why Content-Based Filtering?

This project uses **content-based filtering** because the recommendation engine directly compares the user's preferences with item attributes.

For example:

```text
User likes:
Sci-Fi + Adventure + English

Movie:
Sci-Fi + Adventure + English
```

The movie becomes a strong candidate because its content aligns with the user's stated interests.

Content-based filtering is particularly appropriate for this project because it does not require a large historical database of user interactions.

---

# 👥 Content-Based vs Collaborative Filtering

| Feature | Content-Based | Collaborative Filtering |
|---|---|---|
| Uses item attributes | Yes | Usually not directly |
| Uses user preferences | Yes | Uses user behavior |
| Requires many users | No | Usually |
| Requires interaction history | No | Yes |
| Explainability | High | Usually lower |
| Used in this project | ✅ | ❌ |
| Future extension | — | ✅ |

Collaborative filtering could be added later when sufficient user-rating or interaction data becomes available. The current implementation intentionally focuses on content-based recommendation logic.

---

# 🧰 Technologies Used

## Programming Language

**Python**

## Framework

**Streamlit**

Used to create the interactive web dashboard.

## Data Processing

**Pandas**

Used for:

- Dataset loading
- Data cleaning
- Filtering
- Data manipulation
- Recommendation result processing

## Numerical Computing

**NumPy**

Used for numerical operations and handling similarity scores.

## Machine Learning

**Scikit-learn**

The project uses:

```text
TfidfVectorizer
cosine_similarity
```

for text representation and similarity calculation.

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/AI-Recommendation-System.git
```

Move into the project directory:

```bash
cd AI-Recommendation-System
```

---

## 2. Install Streamlit

```bash
pip install streamlit
```

The application also checks for missing dependencies and can automatically install:

```text
pandas
numpy
scikit-learn
```

when they are unavailable.

---

# ▶️ Running the Application

Run:

```bash
streamlit run streamlit_app.py
```

After starting the application, Streamlit will provide a local URL that can be opened in a web browser.

---

# 🖥️ Application Interface

The application contains four major sections:

```text
🎯 Recommendations
        ↓
📊 Score Analysis
        ↓
🗂️ Dataset Explorer
        ↓
ℹ️ How It Works
```

### Recommendations

Displays personalized movie recommendations.

### Score Analysis

Explains the contribution of rule-based scoring and cosine similarity.

### Dataset Explorer

Allows users to inspect and filter the movie dataset.

### How It Works

Explains the recommendation pipeline, scoring methodology, and design decisions.

---

# 📁 Project Structure

The current implementation is intentionally self-contained:

```text
AI-Recommendation-System/
│
├── streamlit_app.py
└── README.md
```

The Python application contains:

```text
Dataset
   ↓
Data Loading
   ↓
Data Validation
   ↓
User Preferences
   ↓
Rule-Based Scoring
   ↓
TF-IDF
   ↓
Cosine Similarity
   ↓
Score Blending
   ↓
Ranking
   ↓
Streamlit Dashboard
```

---

# 🧪 Example Recommendation Scenario

Suppose the user selects:

```text
Genre:
Sci-Fi

Language:
English

Mood:
Adventure

Minimum Rating:
7.5

Year:
2010–2026

Recommendations:
5
```

The system evaluates every movie against those preferences.

A movie such as:

```text
Interstellar
```

can receive a strong score because it matches several user-selected attributes.

The result is displayed with:

```text
Movie: Interstellar

Match Score: XX%

Genre: Sci-Fi
Language: English
Mood: Adventure
Year: 2014
Rating: 8.7
```

The user can expand the explanation to see which attributes contributed to the recommendation.

---

# 🛡️ Data Validation

The application performs several data validation operations.

It checks:

- Whether the dataset can be parsed
- Whether the dataset contains rows
- Whether required columns exist
- Whether movie titles are available
- Whether text fields contain valid values
- Whether years can be converted to numeric values
- Whether ratings can be converted to numeric values

Missing text values are replaced with:

```text
Unknown
```

and text fields are normalized for consistent comparison.

---

# ⚡ Performance Optimization

The application uses Streamlit caching to avoid unnecessary repeated processing.

The dataset loading function uses:

```python
@st.cache_data
```

while the TF-IDF matrix is cached using:

```python
@st.cache_resource
```

This allows reusable data and similarity resources to be maintained between interactions rather than rebuilt unnecessarily on every request.

---

# 🧩 Edge Case Handling

The system includes handling for several practical situations.

## No Strong Matches

If no movie achieves a strong score, the system still displays the closest available results instead of simply returning nothing.

These are labeled:

```text
Closest Available
```

## Too Many Requested Recommendations

If the user requests more recommendations than the dataset contains, the application displays the available titles rather than producing an error.

## Case Differences

Preference matching is normalized so that capitalization differences do not prevent matching.

For example:

```text
Sci-Fi
sci-fi
SCI-FI
```

are treated consistently.

---

# 📈 Advantages of This Approach

## Explainable

The recommendation score can be traced back to specific attributes.

## Simple

The algorithm is easier to understand than a neural recommendation model.

## Data Efficient

The system does not require historical user-item interaction data.

## Customizable

Scoring weights can be changed interactively.

## Extensible

The architecture can later be expanded into more advanced recommendation techniques.

## Interactive

The Streamlit interface makes the recommendation process easy to demonstrate.

---

# ⚠️ Current Limitations

This project intentionally focuses on fundamental recommendation logic, so it has several limitations.

### No User History

The current system does not learn from previous user interactions.

### No Collaborative Filtering

Recommendations are not based on what other users liked.

### Demonstration Dataset

The included dataset is designed primarily for demonstration.

### Limited Features

The recommendation engine currently focuses on:

- Genre
- Mood
- Language
- Rating
- Release year

### No Personalized Learning

The system does not continuously update its model based on user feedback.

These limitations provide opportunities for future development.

---

# 🔮 Future Improvements

Possible future versions could include:

## 1. Larger Dataset

Use a significantly larger movie dataset to improve recommendation coverage.

## 2. User Accounts

Allow users to create profiles and save their preferences.

## 3. Rating History

Store movies that users liked, disliked, or rated.

## 4. Collaborative Filtering

Introduce user-item interaction data and recommend movies based on similar users.

## 5. Hybrid Recommendation

Combine:

```text
Content-Based Filtering
+
Collaborative Filtering
```

## 6. Machine Learning Ranking

Train a model to predict recommendation relevance from historical interaction data.

## 7. Feedback Loop

Allow users to provide:

```text
👍 Like
👎 Dislike
⭐ Rating
```

and use the feedback to improve future recommendations.

## 8. More Advanced NLP

Movie descriptions, reviews, keywords, and plot summaries could be incorporated into the TF-IDF representation.

---

# 🎓 Learning Outcomes

By completing this project, the developer demonstrates practical understanding of:

- Python programming
- Pandas data processing
- Dataset validation
- Feature representation
- Content-based recommendation
- Pattern matching
- Weighted scoring
- TF-IDF
- Cosine similarity
- Ranking algorithms
- Explainable recommendation systems
- Streamlit application development
- Interactive data visualization
- CSV data handling

---

# 💼 Portfolio Value

This project demonstrates an important step in understanding recommendation systems.

Rather than relying on a pre-built recommendation API, the project implements the core recommendation logic directly.

It demonstrates the ability to transform:

```text
Raw User Preferences
```

into:

```text
Structured Features
```

then:

```text
Similarity Scores
```

and finally:

```text
Ranked Personalized Recommendations
```

This provides a strong foundation for progressing toward more advanced recommendation architectures.

---

# 🔬 Core Algorithm Summary

The recommendation process can be summarized as:

```text
1. Load movie dataset
        ↓
2. Validate dataset
        ↓
3. Collect user preferences
        ↓
4. Normalize preference values
        ↓
5. Compare preferences with movie attributes
        ↓
6. Calculate weighted rule-based score
        ↓
7. Convert movie profiles into TF-IDF vectors
        ↓
8. Calculate cosine similarity
        ↓
9. Blend rule-based and cosine scores
        ↓
10. Rank movies by final score
        ↓
11. Select Top-N movies
        ↓
12. Explain matching attributes
        ↓
13. Display recommendations
```

---

# 📊 Scoring Formula

The final recommendation score is:

```text
Final Score =
(1 − α) × Rule-Based Score
+
α × Cosine Similarity Score
```

where:

```text
α = Cosine Similarity Blend
```

The default value is:

```text
α = 0.25
```

Therefore:

```text
Final Score =
0.75 × Rule Score
+
0.25 × Cosine Score
```

The rule-based score itself is calculated using the active feature weights:

```text
Genre
+
Mood
+
Language
+
Rating
+
Year
```

This approach keeps the recommendation process both **algorithmic and interpretable**.

---

# 🔐 Design Philosophy

The project follows three major principles:

### 1. Transparency

The user should understand why an item was recommended.

### 2. Relevance

Recommendations should be derived from actual user preferences.

### 3. Simplicity

The system should demonstrate recommendation fundamentals before moving to complex neural recommendation models.

The project therefore intentionally avoids unnecessary black-box modeling.

---

# 👨‍💻 Author

**Aryan Ali**

AI Developer | Python Developer | Machine Learning

Focused on building practical AI and machine-learning solutions using Python, data processing, recommendation logic, and model development concepts.

---

# 📜 License

This project is intended for educational, portfolio, and demonstration purposes.

You may modify and extend the project for learning and development.

---

# ⭐ Project Summary

**AI Recommendation Logic System** is a transparent content-based movie recommendation engine that combines **weighted preference matching with TF-IDF cosine similarity**.

Its core architecture is:

```text
User Intent
     ↓
Preference Processing
     ↓
Weighted Pattern Matching
     ↓
TF-IDF Representation
     ↓
Cosine Similarity
     ↓
Score Blending
     ↓
Recommendation Ranking
     ↓
Explainable Results
```

The project demonstrates the fundamental principle behind recommendation systems:

> **Relevant recommendations come from meaningful alignment between user preferences and item attributes—not random selection.**

This system provides a practical foundation for progressing from basic recommendation logic toward more advanced **content-based, collaborative, hybrid, and machine-learning recommendation systems**.
