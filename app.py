from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.npz"
EVALUATION_PATH = PROJECT_ROOT / "artifacts" / "evaluation_results.npz"
SCALER_PATH = PROJECT_ROOT / "data" / "processed" / "scaler.joblib"

CLASS_NAMES = ["Hbb", "Hcc", "QCD"]

FEATURE_NAMES = [
    "Jet_pt",
    "Jet_eta",
    "Jet_phi",
    "Jet_mass",
    "Jet_chHEF",
    "Jet_neHEF",
    "Jet_chEmEF",
    "Jet_neEmEF",
    "Jet_nConstituents",
    "Jet_puId",
]

COLLISION_IMAGE = (
    "https://encrypted-tbn0.gstatic.com/images"
    "?q=tbn:ANd9GcRgQkKV3C8rJvv5dL8NaMljgAJuogbDsk_rNn-dKJ4UE0j8QOAmDUO2MZk&s=10"
)


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Higgs Feature Lab",
    page_icon="⚛️",
    layout="wide",
)


# --------------------------------------------------
# Load saved data
# --------------------------------------------------

@st.cache_data
def load_data():
    dataset = np.load(DATASET_PATH)
    evaluation = np.load(EVALUATION_PATH)

    X_test = dataset["X_test"]
    y_test = evaluation["y_true"]
    y_pred = evaluation["y_pred"]
    y_prob = evaluation["y_prob"]

    if not (
        len(X_test) == len(y_test) == len(y_pred) == len(y_prob)
    ):
        raise ValueError("Test dataset and evaluation results do not match.")

    return X_test, y_test, y_pred, y_prob


@st.cache_resource
def load_scaler():
    import joblib

    return joblib.load(SCALER_PATH)


try:
    X_test, y_test, y_pred, y_prob = load_data()
    scaler = load_scaler()

except FileNotFoundError as error:
    st.error(f"Required project file not found: {error}")
    st.stop()

except (KeyError, ValueError) as error:
    st.error(f"Could not load project data: {error}")
    st.stop()


# --------------------------------------------------
# Event selection
# --------------------------------------------------

def select_random_event():
    st.session_state.event_index = int(
        np.random.randint(0, len(X_test))
    )


if "event_index" not in st.session_state:
    st.session_state.event_index = 0


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("Higgs Feature Lab")

st.markdown(
    """
    Explore how a MiniParT transformer classifies particle collision
    events into **Hbb**, **Hcc**, and **QCD**.

    Investigate jet properties, experiment with input features,
    and understand the model's predictions.
    """
)

st.image(
    COLLISION_IMAGE,
    use_container_width=True,
)


# --------------------------------------------------
# Event selector
# --------------------------------------------------

st.header("Explore a collision event")

col1, col2 = st.columns([3, 1])

with col1:
    event_index = st.number_input(
        "Test event index",
        min_value=0,
        max_value=len(X_test) - 1,
        value=st.session_state.event_index,
        step=1,
        key="event_index",
    )

with col2:
    st.write("")
    st.write("")
    st.button(
        "Random event",
        on_click=select_random_event,
        use_container_width=True,
    )

event_index = int(event_index)


# --------------------------------------------------
# Selected event
# --------------------------------------------------

true_label = CLASS_NAMES[int(y_test[event_index])]
predicted_label = CLASS_NAMES[int(y_pred[event_index])]
probabilities = y_prob[event_index]


# --------------------------------------------------
# Prediction summary
# --------------------------------------------------

st.header("Classification probabilities")

st.caption(
    f"Event {event_index} · "
    f"Actual class: {true_label} · "
    f"Predicted class: {predicted_label}"
)

col1, col2, col3 = st.columns(3)

for column, class_name, probability in zip(
    [col1, col2, col3],
    CLASS_NAMES,
    probabilities,
):
    with column:
        st.metric(
            label=class_name,
            value=f"{probability:.2%}",
        )

probability_df = pd.DataFrame(
    {
        "Class": CLASS_NAMES,
        "Probability": probabilities,
    }
)

st.bar_chart(
    probability_df,
    x="Class",
    y="Probability",
    y_label="Probability",
)


# --------------------------------------------------
# Event features
# --------------------------------------------------

st.header("Jet properties")

# Convert scaled features back to their original values.
raw_features = scaler.inverse_transform(
    X_test[event_index]
)

features_df = pd.DataFrame(
    raw_features,
    columns=FEATURE_NAMES,
    index=["Jet 1", "Jet 2"],
)

st.dataframe(
    features_df,
    use_container_width=True,
)

st.caption(
    "Feature values are displayed in their original, unscaled units."
)
