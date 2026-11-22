
import os
import joblib
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.dummy import DummyClassifier


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 Titanic Survival Prediction")
st.write(
    "Enter passenger details below to predict the estimated "
    "chance of survival using a Machine Learning model."
)

MODEL_FILE = "model.joblib"

FEATURES = ["pclass", "sex", "age", "fare"]


# --------------------------------
# TRAIN AND SAVE MODEL
# --------------------------------

def train_and_save_model():

    st.info("Training the machine learning model for the first time...")

    url = (
        "https://raw.githubusercontent.com/"
        "datasciencedojo/datasets/master/titanic.csv"
    )

    df = pd.read_csv(url)

    # Select required columns
    df = df[["Survived", "Pclass", "Sex", "Age", "Fare"]].copy()

    # Rename columns
    df.columns = ["survived", "pclass", "sex", "age", "fare"]

    # Encode sex
    df["sex"] = df["sex"].map({
        "female": 1,
        "male": 0
    })

    # Prepare input and output
    X = df[FEATURES]
    y = df["survived"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create machine learning pipeline
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", DecisionTreeClassifier(
            max_depth=4,
            random_state=42
        ))
    ])

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model on unseen test data
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    # Train a simple baseline for comparison
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    baseline_predictions = baseline.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)

    # Save model
    joblib.dump(model, MODEL_FILE)

    st.success("Model trained and saved successfully!")

    return accuracy, baseline_accuracy, matrix


# --------------------------------
# LOAD MODEL WITH CACHING
# --------------------------------

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_FILE):
        train_and_save_model()

    return joblib.load(MODEL_FILE)


# --------------------------------
# LOAD MODEL AND METRICS
# --------------------------------

if not os.path.exists(MODEL_FILE):

    accuracy, baseline_accuracy, matrix = train_and_save_model()

else:

    accuracy = None
    baseline_accuracy = None
    matrix = None


model = load_model()


# --------------------------------
# PASSENGER INPUTS
# --------------------------------

st.subheader("👤 Passenger Details")

pclass = st.selectbox(
    "Ticket Class",
    [1, 2, 3],
    index=2
)

sex = st.radio(
    "Sex",
    ["Female", "Male"],
    horizontal=True
)

age = st.slider(
    "Passenger Age",
    min_value=1,
    max_value=80,
    value=30
)

fare = st.slider(
    "Ticket Fare",
    min_value=0.0,
    max_value=500.0,
    value=30.0,
    step=1.0
)


# --------------------------------
# LIVE PREDICTION
# --------------------------------

st.divider()

st.subheader("🔮 Survival Prediction")

sex_value = 1 if sex == "Female" else 0

input_data = pd.DataFrame(
    [[pclass, sex_value, age, fare]],
    columns=FEATURES
)

# Predict survival probability
probabilities = model.predict_proba(input_data)[0]

survival_probability = probabilities[1]
death_probability = probabilities[0]

prediction = model.predict(input_data)[0]


# --------------------------------
# DISPLAY RESULTS
# --------------------------------

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Estimated Survival Chance",
        f"{survival_probability:.1%}"
    )

with col2:
    st.metric(
        "Estimated Non-Survival Chance",
        f"{death_probability:.1%}"
    )

st.progress(float(survival_probability))

if prediction == 1:

    st.success("🎉 Prediction: Likely to Survive")

else:

    st.error("Prediction: Unlikely to Survive")


# --------------------------------
# MODEL INFORMATION
# --------------------------------

st.divider()

st.subheader("📊 Model Information")

if accuracy is None:

    st.info(
        "The saved model has been loaded. "
        "Restart the app to see the training metrics."
    )

else:

    col1, col2 = st.columns(2)

    col1.metric("Decision Tree Accuracy", f"{accuracy:.2%}")
    col2.metric("Baseline Accuracy", f"{baseline_accuracy:.2%}")

    st.write("Confusion Matrix")
    st.write(matrix)


# --------------------------------
# FOOTER
# --------------------------------

st.divider()

st.caption(
    "Educational project using the historical Titanic dataset. "
    "Predictions are estimates, not historical facts or guarantees."
)
