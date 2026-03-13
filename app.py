import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Responsible AI Dashboard", layout="wide")

# Now I Load the Dataset

@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"
    ]
    data = pd.read_csv(
        url,
        names=columns,
        na_values=" ?",
        skipinitialspace=True,
    ).dropna()
    return data

# This is Data Preprocess

@st.cache_data
def prepare_data(data: pd.DataFrame):
    encoded = data.copy()
    encoders = {}

    for col in encoded.select_dtypes(include="object").columns:
        le = LabelEncoder()
        encoded[col] = le.fit_transform(encoded[col])
        encoders[col] = le

    X = encoded.drop("income", axis=1)
    y = encoded["income"]

    # I keep original protected attributes for fairness breakdowns
    
    X_raw = data.drop("income", axis=1)

    X_train, X_test, y_train, y_test, X_train_raw, X_test_raw = train_test_split(
        X, y, X_raw,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, X_train_raw, X_test_raw, encoders

# Now I train the Model

@st.cache_resource
def train_model(X_train, y_train):
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model


# Fairness by Group

def group_accuracy(y_true, y_pred, groups: pd.Series):
    """Compute accuracy per group for a protected attribute."""
    results = {}
    fairness_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": groups})

    for group_name, subset in fairness_df.groupby("group"):
        results[group_name] = accuracy_score(subset["y_true"], subset["y_pred"])

    return results

# Run pipeline

raw_data = load_data()
X_train, X_test, y_train, y_test, X_train_raw, X_test_raw, encoders = prepare_data(raw_data)
model = train_model(X_train, y_train)
y_pred = model.predict(X_test)

#Now I evaluate the Model

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# This is Fairness Selection

st.title("Responsible AI Dashboard")
st.caption("Adult Income dataset • Decision Tree Classifier • Performance + fairness overview")

protected_attr = st.selectbox(
    "Choose a protected attribute for fairness comparison:",
    options=["sex", "race"],
    index=0,
)

fairness_results = group_accuracy(y_test, y_pred, X_test_raw[protected_attr])
fairness_df = pd.DataFrame(
    {
        "group": list(fairness_results.keys()),
        "accuracy": list(fairness_results.values()),
    }
).sort_values("accuracy", ascending=False)

# Here I create the Dashboard layout

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy:.3f}")
col2.metric("Precision", f"{precision:.3f}")
col3.metric("Recall", f"{recall:.3f}")

st.subheader("Fairness by Group")
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(fairness_df["group"], fairness_df["accuracy"])
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
ax.set_xlabel(protected_attr.capitalize())
ax.set_title(f"Accuracy by {protected_attr.capitalize()}")
plt.xticks(rotation=20)
plt.tight_layout()
st.pyplot(fig)

with st.expander("Show fairness results dictionary"):
    st.json(fairness_results)

st.subheader("Why this matters for Responsible AI")
st.write(
    """
    This dashboard is useful for inspecting how a machine learning model behaves.
    The performance metrics show whether the classifier works well overall.
    At the same time, the comparison between groups helps us see if the model performs
    similarly for different groups, such as people of different sex or race.
    This supports the Measure function of the NIST AI Risk Management Framework.
    Instead of talking about risks in abstract terms, the dashboard shows concrete metrics
    that can be observed, monitored, and reviewed.
    It also supports transparency, which is an important principle in AI governance.
    In fact, it is possible not only see the final prediction system, but also understand how
    the model behaves for different groups.
    In practice, a dashboard like this does not solve fairness problems. However, it encourages
    some best practices: checking the quality of the model, identifying possible disparities between
    groups, and giving clear evidence to determine if the model is ready to be deployed needs further improvement.
    """
)

st.subheader("Optional preview of the evaluation data")
st.dataframe(X_test_raw.assign(actual=y_test.values, predicted=y_pred).head(20), use_container_width=True)
