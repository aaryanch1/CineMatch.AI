"""
===============================================================================
 DATA CLASSIFICATION USING AI — STREAMLIT DASHBOARD
===============================================================================
Interactive Streamlit front-end for the e-commerce OrderStatus classification
pipeline:

    Upload Data -> Data Understanding -> Data Preprocessing -> Train/Test Split
                -> Model Training -> Prediction -> Model Evaluation -> Download

Run with:
    streamlit run streamlit_app.py

Author  : Aryan (AI/ML)
===============================================================================
"""

import io
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------- #
# PAGE CONFIG
# ------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Order Status Classification",
    page_icon="📦",
    layout="wide",
)

DEFAULT_ID_COLUMNS = [
    "OrderID",
    "CustomerID",
    "TrackingNumber",
    "ShippingAddress",
    "Date",
]
RANDOM_STATE_DEFAULT = 42
TEST_SIZE_DEFAULT = 0.2


# ------------------------------------------------------------------------- #
# HELPERS (cached where it makes sense)
# ------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_dataset(uploaded_file) -> pd.DataFrame:
    """Load a CSV or Excel file uploaded through Streamlit's file_uploader."""
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)


def prepare_features_and_target(df, target_col, id_cols_to_drop, coupon_col):
    df_clean = df.drop(columns=[c for c in id_cols_to_drop if c in df.columns])

    if coupon_col and coupon_col in df_clean.columns:
        df_clean[coupon_col] = df_clean[coupon_col].fillna("NoCoupon")

    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    return X, y, numerical_features, categorical_features


def build_preprocessing_pipeline(numerical_features, categorical_features) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ])


def build_and_train_model(preprocessor, X_train, y_train, max_iter, random_state) -> Pipeline:
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=max_iter, random_state=random_state)),
    ])
    model.fit(X_train, y_train)
    return model


def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def build_results_text(df_shape, train_size, test_size, class_labels,
                        accuracy, precision, recall, f1, report) -> str:
    lines = []
    lines.append("===== CLASSIFICATION MODEL RESULTS =====\n")
    lines.append(f"Dataset Shape:\n{df_shape[0]} rows x {df_shape[1]} columns\n")
    lines.append(f"Training Samples:\n{train_size}\n")
    lines.append(f"Testing Samples:\n{test_size}\n")
    lines.append(f"Target Classes:\n{list(class_labels)}\n")
    lines.append("Model:\nLogistic Regression (multinomial baseline classifier)\n")
    lines.append(f"Accuracy:\n{accuracy:.4f}\n")
    lines.append(f"Precision:\n{precision:.4f} (weighted average)\n")
    lines.append(f"Recall:\n{recall:.4f} (weighted average)\n")
    lines.append(f"F1 Score:\n{f1:.4f} (weighted average)\n")
    lines.append("Classification Report:\n" + report + "\n")
    lines.append("Model Status:\nTraining completed successfully.\n")
    return "\n".join(lines)


# ------------------------------------------------------------------------- #
# SIDEBAR — DATA UPLOAD & CONFIG
# ------------------------------------------------------------------------- #
st.title("📦 Order Status Classification Dashboard")
st.caption("Dataset → Data Understanding → Preprocessing → Train/Test Split → Training → Prediction → Evaluation")

with st.sidebar:
    st.header("1. Dataset")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

    st.header("2. Configuration")
    st.caption("Set these once your file is loaded.")

if uploaded_file is None:
    st.info("👈 Upload a dataset (CSV or Excel) from the sidebar to get started.")
    st.stop()

df = load_dataset(uploaded_file)

with st.sidebar:
    target_col = st.selectbox(
        "Target / class column",
        options=list(df.columns),
        index=(list(df.columns).index("OrderStatus") if "OrderStatus" in df.columns else 0),
    )
    id_cols_to_drop = st.multiselect(
        "Identifier columns to drop",
        options=[c for c in df.columns if c != target_col],
        default=[c for c in DEFAULT_ID_COLUMNS if c in df.columns],
    )
    remaining_object_cols = [
        c for c in df.columns
        if c not in id_cols_to_drop and c != target_col and df[c].dtype == "object"
    ]
    coupon_col = st.selectbox(
        "Column to fill missing values with 'NoCoupon' (optional)",
        options=["(none)"] + remaining_object_cols,
        index=(["(none)"] + remaining_object_cols).index("CouponCode")
        if "CouponCode" in remaining_object_cols else 0,
    )
    coupon_col = None if coupon_col == "(none)" else coupon_col

    test_size = st.slider("Test size", 0.1, 0.5, TEST_SIZE_DEFAULT, 0.05)
    random_state = st.number_input("Random state", value=RANDOM_STATE_DEFAULT, step=1)
    max_iter = st.number_input("Logistic Regression max_iter", value=1000, step=100)

# ------------------------------------------------------------------------- #
# TABS
# ------------------------------------------------------------------------- #
tab_understand, tab_preprocess, tab_train, tab_eval, tab_download = st.tabs(
    ["🔍 Data Understanding", "🧹 Preprocessing", "🏋️ Train/Test & Training",
     "📊 Evaluation", "💾 Download"]
)

# ---- TAB: DATA UNDERSTANDING ---- #
with tab_understand:
    st.subheader("First 5 Rows")
    st.dataframe(df.head())

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Duplicate Rows", int(df.duplicated().sum()))

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Data Types")
        st.dataframe(df.dtypes.astype(str).rename("dtype"))
    with col_b:
        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum().rename("missing_count"))

    st.subheader("Statistical Summary (numeric columns)")
    st.dataframe(df.describe())

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    col_c, col_d = st.columns(2)
    col_c.write("**Categorical Features**")
    col_c.write(categorical_cols)
    col_d.write("**Numerical Features**")
    col_d.write(numerical_cols)

    st.subheader(f"Target Class Distribution — {target_col}")
    class_counts = df[target_col].value_counts()
    st.bar_chart(class_counts)
    st.dataframe(class_counts.rename("count"))

# ---- TAB: PREPROCESSING ---- #
with tab_preprocess:
    X, y, numerical_features, categorical_features = prepare_features_and_target(
        df, target_col, id_cols_to_drop, coupon_col
    )
    st.write(f"**Dropped columns:** {id_cols_to_drop if id_cols_to_drop else 'None'}")
    if coupon_col:
        st.write(f"**Filled missing values in `{coupon_col}` with `'NoCoupon'`.**")
    st.write(f"**Numerical features ({len(numerical_features)}):**", numerical_features)
    st.write(f"**Categorical features ({len(categorical_features)}):**", categorical_features)
    st.write(f"**Feature matrix shape:** {X.shape}")
    st.caption(
        "Scaling and one-hot encoding are fitted only on the training split "
        "inside the pipeline (below), so there is no data leakage from the test set."
    )

# ---- TAB: TRAIN/TEST & TRAINING ---- #
with tab_train:
    st.subheader("Train the model")
    if st.button("🚀 Run train/test split + train model", type="primary"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=int(random_state),
            stratify=y,
        )
        preprocessor = build_preprocessing_pipeline(numerical_features, categorical_features)
        with st.spinner("Training Logistic Regression..."):
            model = build_and_train_model(preprocessor, X_train, y_train, int(max_iter), int(random_state))
        y_pred = model.predict(X_test)

        # Persist everything needed by later tabs in session_state
        st.session_state["model"] = model
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
        st.session_state["y_pred"] = y_pred
        st.session_state["class_labels"] = sorted(y.unique())
        st.session_state["train_size"] = X_train.shape[0]
        st.session_state["test_size_n"] = X_test.shape[0]
        st.session_state["df_shape"] = df.shape

        st.success("Model trained successfully!")
        c1, c2 = st.columns(2)
        c1.metric("Training samples", X_train.shape[0])
        c2.metric("Testing samples", X_test.shape[0])
    else:
        st.info("Configure options in the sidebar, then click the button above to train.")

# ---- TAB: EVALUATION ---- #
with tab_eval:
    if "model" not in st.session_state:
        st.warning("Train the model first in the 'Train/Test & Training' tab.")
    else:
        y_test = st.session_state["y_test"]
        y_pred = st.session_state["y_pred"]
        class_labels = st.session_state["class_labels"]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        report = classification_report(y_test, y_pred, zero_division=0)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{accuracy:.4f}")
        m2.metric("Precision", f"{precision:.4f}")
        m3.metric("Recall", f"{recall:.4f}")
        m4.metric("F1 Score", f"{f1:.4f}")

        st.subheader("Classification Report")
        st.code(report)

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred, labels=class_labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=30)
        plt.title("Confusion Matrix — Logistic Regression")
        plt.tight_layout()
        st.pyplot(fig)

        st.session_state["accuracy"] = accuracy
        st.session_state["precision"] = precision
        st.session_state["recall"] = recall
        st.session_state["f1"] = f1
        st.session_state["report"] = report
        st.session_state["cm_fig"] = fig

# ---- TAB: DOWNLOAD ---- #
with tab_download:
    if "model" not in st.session_state:
        st.warning("Train and evaluate the model first.")
    else:
        model_bytes = pickle.dumps(st.session_state["model"])
        st.download_button(
            "⬇️ Download trained model (.pkl)",
            data=model_bytes,
            file_name="classification_model.pkl",
            mime="application/octet-stream",
        )

        if "report" in st.session_state:
            results_text = build_results_text(
                st.session_state["df_shape"],
                st.session_state["train_size"],
                st.session_state["test_size_n"],
                st.session_state["class_labels"],
                st.session_state["accuracy"],
                st.session_state["precision"],
                st.session_state["recall"],
                st.session_state["f1"],
                st.session_state["report"],
            )
            st.download_button(
                "⬇️ Download evaluation results (.txt)",
                data=results_text,
                file_name="evaluation_results.txt",
                mime="text/plain",
            )
            st.download_button(
                "⬇️ Download confusion matrix (.png)",
                data=fig_to_bytes(st.session_state["cm_fig"]),
                file_name="confusion_matrix.png",
                mime="image/png",
            )
        else:
            st.info("Visit the 'Evaluation' tab first to generate downloadable results.")