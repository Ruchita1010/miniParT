import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.npz"
EVALUATION_PATH = PROJECT_ROOT / "artifacts" / "evaluation_results.npz"
SCALER_PATH = PROJECT_ROOT / "data" / "processed" / "scaler.joblib"

CLASS_NAMES = ["Hbb", "Hcc", "QCD"]
CLASS_COLORS = {"Hbb": "#6C8CFF", "Hcc": "#B084FF", "QCD": "#FFB86B"}
CLASS_BLURB = {
    "Hbb": "Higgs boson decaying to a bottom quark pair",
    "Hcc": "Higgs boson decaying to a charm quark pair",
    "QCD": "Background jets from strong-interaction processes",
}
JET_COLORS = ["#4FD1C5", "#FF6B8B"]

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
F = {name: i for i, name in enumerate(FEATURE_NAMES)}

# Illustrative detector geometry (arbitrary units)
BARREL_R = 1.0
HALF_LENGTH = 2.4


# --------------------------------------------------
# Page configuration and styling
# --------------------------------------------------

st.set_page_config(
    page_title="Higgs Feature Lab",
    page_icon="⚛️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(1200px 600px at 10% -10%, #1b2250 0%, transparent 60%),
                    radial-gradient(900px 500px at 100% 0%, #3a1c5c 0%, transparent 55%),
                    #0b0e1a;
    }
    .block-container { padding-top: 2rem; max-width: 1250px; }
    h1, h2, h3 { letter-spacing: -0.02em; }

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(108,140,255,.18), rgba(176,132,255,.12) 55%, rgba(255,184,107,.10));
        border: 1px solid rgba(255,255,255,.08);
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        margin: 0 0 .4rem 0; font-size: 2.6rem;
        background: linear-gradient(90deg, #8fa8ff, #c9a7ff 50%, #ffc98a);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero p { margin: 0; color: #b9c0dd; font-size: 1.05rem; max-width: 780px; }

    .card {
        padding: 1.1rem 1.3rem; border-radius: 16px;
        background: rgba(255,255,255,.04);
        border: 1px solid rgba(255,255,255,.08);
        height: 100%;
    }
    .card .label { color: #98a2c7; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
    .card .value { font-size: 1.9rem; font-weight: 700; margin-top: .15rem; }
    .card .sub { color: #98a2c7; font-size: .85rem; margin-top: .15rem; }

    .pill {
        display: inline-block; padding: .15rem .7rem; border-radius: 999px;
        font-size: .8rem; font-weight: 600; margin-left: .4rem;
    }
    .pill.ok  { background: rgba(72,199,142,.15); color: #48c78e; border: 1px solid rgba(72,199,142,.4); }
    .pill.bad { background: rgba(255,107,139,.15); color: #ff6b8b; border: 1px solid rgba(255,107,139,.4); }

    div[data-baseweb="tab-list"] { gap: .4rem; }
    button[data-baseweb="tab"] { border-radius: 10px 10px 0 0; }
    [data-testid="stSidebar"] { background: rgba(255,255,255,.03); }
    </style>
    """,
    unsafe_allow_html=True,
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

    if not (len(X_test) == len(y_test) == len(y_pred) == len(y_prob)):
        raise ValueError("Test dataset and evaluation results do not match.")

    return X_test, y_test, y_pred, y_prob


@st.cache_resource
def load_scaler():
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

N_EVENTS = len(X_test)


# --------------------------------------------------
# Event selection state
# --------------------------------------------------

if "event_index" not in st.session_state:
    st.session_state.event_index = 0


def set_event(index):
    st.session_state.event_index = int(np.clip(index, 0, N_EVENTS - 1))


def eligible_indices(mode):
    if mode == "Misclassified only":
        mask = y_test != y_pred
    elif mode in CLASS_NAMES:
        mask = y_test == CLASS_NAMES.index(mode)
    else:
        mask = np.ones(N_EVENTS, dtype=bool)
    return np.flatnonzero(mask)


def pick_random(mode):
    pool = eligible_indices(mode)
    if len(pool):
        set_event(np.random.choice(pool))


# --------------------------------------------------
# Collision geometry helpers
# --------------------------------------------------

def unit_vector(eta, phi):
    """Direction of a particle with the given pseudorapidity and azimuth."""
    return np.array(
        [np.cos(phi) / np.cosh(eta), np.sin(phi) / np.cosh(eta), np.tanh(eta)]
    )


def hit_distance(directions):
    """Distance from the origin to the detector surface along each direction."""
    transverse = np.hypot(directions[:, 0], directions[:, 1]) + 1e-9
    s_barrel = BARREL_R / transverse
    s_endcap = HALF_LENGTH / (np.abs(directions[:, 2]) + 1e-9)
    return np.minimum(s_barrel, s_endcap)


def build_constituents(jet_raw, seed):
    """
    Illustrative particle spray for one jet.

    The dataset stores jet-level features, not individual particles, so the
    spray is generated from them: particle count from nConstituents, charged
    share from the charged energy fractions, spread from jet mass / pT.
    """
    rng = np.random.default_rng(seed)

    pt = max(float(jet_raw[F["Jet_pt"]]), 1e-3)
    mass = max(float(jet_raw[F["Jet_mass"]]), 0.0)
    eta = float(jet_raw[F["Jet_eta"]])
    phi = float(jet_raw[F["Jet_phi"]])

    n = int(np.clip(round(jet_raw[F["Jet_nConstituents"]]), 4, 70))

    charged = jet_raw[F["Jet_chHEF"]] + jet_raw[F["Jet_chEmEF"]]
    neutral = jet_raw[F["Jet_neHEF"]] + jet_raw[F["Jet_neEmEF"]]
    charged_share = float(np.clip(charged / (charged + neutral + 1e-9), 0.05, 0.95))

    axis = unit_vector(eta, phi)
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    spread = float(np.clip(2.0 * mass / pt, 0.08, 0.45))
    radius = np.abs(rng.normal(0.0, spread, n))
    angle = rng.uniform(0, 2 * np.pi, n)
    directions = axis + (radius * np.cos(angle))[:, None] * u + (radius * np.sin(angle))[:, None] * v
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    energy = rng.exponential(1.0, n)
    energy = energy / energy.sum() * pt

    return {
        "directions": directions,
        "energy": energy,
        "is_charged": rng.random(n) < charged_share,
        "axis": axis,
        "pt": pt,
    }


def detector_wireframe():
    traces = []
    theta = np.linspace(0, 2 * np.pi, 80)
    for z in np.linspace(-HALF_LENGTH, HALF_LENGTH, 7):
        traces.append(
            go.Scatter3d(
                x=BARREL_R * np.cos(theta), y=BARREL_R * np.sin(theta),
                z=np.full_like(theta, z), mode="lines",
                line=dict(color="rgba(160,175,230,0.22)", width=2),
                hoverinfo="skip", showlegend=False,
            )
        )
    for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        traces.append(
            go.Scatter3d(
                x=[BARREL_R * np.cos(a)] * 2, y=[BARREL_R * np.sin(a)] * 2,
                z=[-HALF_LENGTH, HALF_LENGTH], mode="lines",
                line=dict(color="rgba(160,175,230,0.12)", width=2),
                hoverinfo="skip", showlegend=False,
            )
        )
    # Inner tracker shell
    for z in (-HALF_LENGTH * 0.6, HALF_LENGTH * 0.6):
        traces.append(
            go.Scatter3d(
                x=0.55 * np.cos(theta), y=0.55 * np.sin(theta),
                z=np.full_like(theta, z), mode="lines",
                line=dict(color="rgba(160,175,230,0.14)", width=2),
                hoverinfo="skip", showlegend=False,
            )
        )
    # Beam pipe
    traces.append(
        go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[-HALF_LENGTH * 1.15, HALF_LENGTH * 1.15],
            mode="lines", line=dict(color="rgba(255,255,255,0.55)", width=4),
            name="Beam axis", hoverinfo="skip",
        )
    )
    return traces


def segments(points_start, points_end):
    """Pack many line segments into one Scatter3d (None-separated)."""
    n = len(points_start)
    xs, ys, zs = [], [], []
    for i in range(n):
        xs += [points_start[i, 0], points_end[i, 0], None]
        ys += [points_start[i, 1], points_end[i, 1], None]
        zs += [points_start[i, 2], points_end[i, 2], None]
    return xs, ys, zs


def collision_figure(raw_jets, event_index, show_tracks, show_neutral, show_axes, animate):
    jets = [
        build_constituents(raw_jets[j], seed=event_index * 10 + j)
        for j in range(len(raw_jets))
    ]

    all_dirs = np.vstack([j["directions"] for j in jets])
    all_energy = np.concatenate([j["energy"] for j in jets])
    all_charged = np.concatenate([j["is_charged"] for j in jets])
    all_jet_id = np.concatenate(
        [np.full(len(j["energy"]), idx) for idx, j in enumerate(jets)]
    )
    full_len = hit_distance(all_dirs)
    # Charged particles are tracked to the tracker radius, neutrals to the calorimeter.
    reach = np.where(all_charged, 0.6, 1.0) * full_len
    ends = all_dirs * reach[:, None]

    def dynamic_traces(t):
        data = []
        pos = ends * t
        origin = np.zeros_like(pos)

        if show_tracks:
            m = all_charged
            if m.any():
                x, y, z = segments(origin[m], pos[m])
                data.append(go.Scatter3d(
                    x=x, y=y, z=z, mode="lines", name="Charged tracks",
                    line=dict(color="#7FDBFF", width=3), hoverinfo="skip",
                ))
            else:
                data.append(go.Scatter3d(x=[], y=[], z=[], name="Charged tracks"))
        else:
            data.append(go.Scatter3d(x=[], y=[], z=[], name="Charged tracks"))

        if show_neutral:
            m = ~all_charged
            if m.any():
                x, y, z = segments(origin[m], pos[m])
                data.append(go.Scatter3d(
                    x=x, y=y, z=z, mode="lines", name="Neutral particles",
                    line=dict(color="#FFD166", width=2, dash="dot"), hoverinfo="skip",
                ))
            else:
                data.append(go.Scatter3d(x=[], y=[], z=[], name="Neutral particles"))
        else:
            data.append(go.Scatter3d(x=[], y=[], z=[], name="Neutral particles"))

        # Energy deposits at the head of each particle
        sizes = 3 + 14 * (all_energy / all_energy.max())
        data.append(go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="markers", name="Energy deposits",
            marker=dict(
                size=sizes,
                color=[JET_COLORS[i] for i in all_jet_id],
                opacity=0.85, line=dict(width=0),
            ),
            customdata=np.stack([all_energy, all_jet_id + 1, all_charged], axis=1),
            hovertemplate="Jet %{customdata[1]:.0f}<br>pT share: %{customdata[0]:.1f} GeV<extra></extra>",
        ))
        return data

    fig = go.Figure(data=detector_wireframe() + dynamic_traces(1.0))
    n_static = len(fig.data) - 3

    if show_axes:
        for idx, jet in enumerate(jets):
            length = 0.35 + 0.65 * jet["pt"] / max(j["pt"] for j in jets)
            tip = jet["axis"] * length * 1.6
            fig.add_trace(go.Scatter3d(
                x=[0, tip[0]], y=[0, tip[1]], z=[0, tip[2]],
                mode="lines+text", text=["", f"Jet {idx + 1}"], textposition="top center",
                textfont=dict(color=JET_COLORS[idx], size=14),
                line=dict(color=JET_COLORS[idx], width=9),
                name=f"Jet {idx + 1} axis (pT {jet['pt']:.0f} GeV)",
            ))

    # Collision vertex
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers", name="Collision vertex",
        marker=dict(size=7, color="#ffffff", symbol="diamond"), hoverinfo="skip",
    ))

    if animate:
        ts = np.linspace(0, 1, 18)
        dynamic_idx = list(range(n_static, n_static + 3))
        fig.frames = [
            go.Frame(data=dynamic_traces(t), traces=dynamic_idx, name=f"{t:.2f}")
            for t in ts
        ]
        fig.update_layout(
            updatemenus=[dict(
                type="buttons", direction="left", x=0.02, y=0.02, xanchor="left", yanchor="bottom",
                bgcolor="rgba(30,36,70,.9)", bordercolor="rgba(255,255,255,.2)",
                font=dict(color="#e6e9ff"),
                buttons=[
                    dict(label="▶  Replay collision", method="animate",
                         args=[None, dict(frame=dict(duration=55, redraw=True),
                                          fromcurrent=False, transition=dict(duration=0))]),
                    dict(label="⏸", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                ],
            )],
            sliders=[dict(
                x=0.25, len=0.7, y=0.02, currentvalue=dict(visible=False),
                pad=dict(t=0, b=0),
                steps=[dict(method="animate", label="",
                            args=[[f"{t:.2f}"], dict(mode="immediate",
                                                     frame=dict(duration=0, redraw=True))])
                       for t in ts],
            )],
        )

    fig.update_layout(
        height=640,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(20,24,50,.6)", font=dict(color="#d7dcf5"), x=0.01, y=0.99),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor="rgba(0,0,0,0)", aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.2, z=0.8)),
        ),
    )
    return fig, jets


def eta_phi_figure(jets, raw_jets):
    fig = go.Figure()
    for idx, jet in enumerate(jets):
        d = jet["directions"]
        eta = np.arcsinh(d[:, 2] / (np.hypot(d[:, 0], d[:, 1]) + 1e-9))
        phi = np.arctan2(d[:, 1], d[:, 0])
        fig.add_trace(go.Scatter(
            x=eta, y=phi, mode="markers", name=f"Jet {idx + 1} particles",
            marker=dict(
                size=5 + 28 * np.sqrt(jet["energy"] / jet["energy"].max()),
                color=JET_COLORS[idx], opacity=0.45, line=dict(width=0),
            ),
            hovertemplate="η %{x:.2f}<br>φ %{y:.2f}<extra></extra>",
        ))
        jet_eta, jet_phi = float(raw_jets[idx][F["Jet_eta"]]), float(raw_jets[idx][F["Jet_phi"]])
        fig.add_trace(go.Scatter(
            x=[jet_eta], y=[jet_phi], mode="markers+text", text=[f"Jet {idx + 1}"],
            textposition="top center", name=f"Jet {idx + 1} axis",
            marker=dict(size=14, color=JET_COLORS[idx], symbol="x", line=dict(width=2, color="white")),
        ))
        fig.add_shape(
            type="circle", xref="x", yref="y",
            x0=jet_eta - 0.4, x1=jet_eta + 0.4, y0=jet_phi - 0.4, y1=jet_phi + 0.4,
            line=dict(color=JET_COLORS[idx], dash="dot", width=1.5),
        )
    fig.update_layout(
        height=520, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
        xaxis=dict(title="Pseudorapidity η", gridcolor="rgba(255,255,255,.08)", zeroline=False),
        yaxis=dict(title="Azimuth φ (rad)", gridcolor="rgba(255,255,255,.08)", zeroline=False),
        legend=dict(orientation="h", y=1.08),
        font=dict(color="#d7dcf5"),
    )
    return fig


# --------------------------------------------------
# Sidebar: event navigation
# --------------------------------------------------

with st.sidebar:
    st.markdown("### ⚛️ Event browser")

    mode = st.selectbox(
        "Browse events",
        ["All events", *CLASS_NAMES, "Misclassified only"],
        help="Limits what the Random button picks from.",
    )

    st.number_input(
        "Test event index",
        min_value=0,
        max_value=N_EVENTS - 1,
        step=1,
        key="event_index",
    )

    b1, b2 = st.columns(2)
    b1.button("◀ Prev", on_click=set_event, args=(st.session_state.event_index - 1,), use_container_width=True)
    b2.button("Next ▶", on_click=set_event, args=(st.session_state.event_index + 1,), use_container_width=True)
    st.button("🎲 Random event", on_click=pick_random, args=(mode,), use_container_width=True)

    st.divider()
    st.markdown("### 🎛️ Visualizer")
    animate = st.toggle("Animated collision", value=True)
    show_tracks = st.toggle("Charged tracks", value=True)
    show_neutral = st.toggle("Neutral particles", value=True)
    show_axes = st.toggle("Jet axes", value=True)

    st.caption(f"{N_EVENTS:,} events in the test set")


event_index = int(st.session_state.event_index)

true_label = CLASS_NAMES[int(y_test[event_index])]
predicted_label = CLASS_NAMES[int(y_pred[event_index])]
probabilities = y_prob[event_index]
correct = true_label == predicted_label

raw_features = scaler.inverse_transform(X_test[event_index])
features_df = pd.DataFrame(
    raw_features,
    columns=FEATURE_NAMES,
    index=[f"Jet {i + 1}" for i in range(len(raw_features))],
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>Higgs Feature Lab</h1>
        <p>Explore how a MiniParT transformer classifies particle collision events into
        <b>Hbb</b>, <b>Hcc</b> and <b>QCD</b>. Fly through the collision, inspect jet
        properties and see how confident the model is.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Summary cards
# --------------------------------------------------

confidence = float(probabilities[int(y_pred[event_index])])
status = (
    '<span class="pill ok">Correct</span>'
    if correct
    else '<span class="pill bad">Misclassified</span>'
)

c1, c2, c3, c4 = st.columns(4)
cards = [
    (c1, "Event", f"#{event_index:,}", f"of {N_EVENTS:,} test events"),
    (c2, "Actual class", f'<span style="color:{CLASS_COLORS[true_label]}">{true_label}</span>', CLASS_BLURB[true_label]),
    (c3, "Predicted class",
     f'<span style="color:{CLASS_COLORS[predicted_label]}">{predicted_label}</span>{status}',
     CLASS_BLURB[predicted_label]),
    (c4, "Confidence", f"{confidence:.1%}", "probability of predicted class"),
]
for column, label, value, sub in cards:
    column.markdown(
        f'<div class="card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )

st.write("")


# --------------------------------------------------
# Tabs
# --------------------------------------------------

tab_collision, tab_map, tab_pred, tab_jets, tab_overview = st.tabs(
    ["💥 Collision viewer", "🗺️ η–φ map", "📊 Prediction", "🔬 Jet properties", "📈 Model overview"]
)

with tab_collision:
    fig3d, jets = collision_figure(
        raw_features, event_index, show_tracks, show_neutral, show_axes, animate
    )
    st.plotly_chart(fig3d, use_container_width=True)
    st.caption(
        "Drag to rotate, scroll to zoom, hover deposits for details. Jet axes and kinematics "
        "come from the event's real features. Individual particles are an illustration "
        "generated from jet-level values (constituent count, charged vs. neutral energy "
        "fractions, mass/pT), since the dataset stores no per-particle data."
    )

with tab_map:
    _, jets_for_map = collision_figure(
        raw_features, event_index, False, False, False, False
    )
    st.plotly_chart(eta_phi_figure(jets_for_map, raw_features), use_container_width=True)
    st.caption("Dotted circles mark a ΔR = 0.4 cone around each jet axis. Marker size reflects energy share.")

with tab_pred:
    left, right = st.columns([3, 2])

    with left:
        fig = go.Figure(go.Bar(
            x=probabilities,
            y=CLASS_NAMES,
            orientation="h",
            marker=dict(
                color=[CLASS_COLORS[c] for c in CLASS_NAMES],
                line=dict(
                    color=["white" if c == predicted_label else "rgba(0,0,0,0)" for c in CLASS_NAMES],
                    width=2,
                ),
            ),
            text=[f"{p:.2%}" for p in probabilities],
            textposition="outside",
            cliponaxis=False,
        ))
        fig.update_layout(
            height=320, margin=dict(l=10, r=60, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(range=[0, 1.05], tickformat=".0%", gridcolor="rgba(255,255,255,.08)"),
            yaxis=dict(autorange="reversed"),
            font=dict(color="#d7dcf5", size=14),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number=dict(suffix="%", font=dict(color="#e6e9ff")),
            title=dict(text=f"Confidence in {predicted_label}", font=dict(color="#98a2c7", size=14)),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#98a2c7"),
                bar=dict(color=CLASS_COLORS[predicted_label]),
                bgcolor="rgba(255,255,255,.05)", borderwidth=0,
            ),
        ))
        gauge.update_layout(
            height=320, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(gauge, use_container_width=True)

with tab_jets:
    st.dataframe(
        features_df.style.format("{:.3f}"),
        use_container_width=True,
    )
    st.caption("Feature values are displayed in their original, unscaled units.")

    fraction_cols = ["Jet_chHEF", "Jet_neHEF", "Jet_chEmEF", "Jet_neEmEF"]
    fraction_labels = ["Charged hadron", "Neutral hadron", "Charged EM", "Neutral EM"]
    fraction_colors = ["#4FD1C5", "#FFD166", "#6C8CFF", "#FF6B8B"]

    frac_fig = go.Figure()
    for col_name, label, color in zip(fraction_cols, fraction_labels, fraction_colors):
        frac_fig.add_trace(go.Bar(
            name=label, x=features_df.index, y=features_df[col_name], marker_color=color,
        ))
    frac_fig.update_layout(
        barmode="stack", height=360, title="Jet energy fractions",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="rgba(255,255,255,.08)"), font=dict(color="#d7dcf5"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(frac_fig, use_container_width=True)

with tab_overview:
    accuracy = float((y_test == y_pred).mean())
    n_classes = len(CLASS_NAMES)

    confusion = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(confusion, (y_test.astype(int), y_pred.astype(int)), 1)
    row_sums = confusion.sum(axis=1, keepdims=True).clip(min=1)
    normalized = confusion / row_sums

    o1, o2 = st.columns([1, 2])
    with o1:
        st.markdown(
            f'<div class="card"><div class="label">Overall accuracy</div>'
            f'<div class="value">{accuracy:.2%}</div>'
            f'<div class="sub">{int((y_test == y_pred).sum()):,} of {N_EVENTS:,} events</div></div>',
            unsafe_allow_html=True,
        )
        st.write("")
        counts = pd.Series(y_test.astype(int)).value_counts().sort_index()
        for i, name in enumerate(CLASS_NAMES):
            st.markdown(
                f'<div class="card" style="margin-bottom:.6rem"><div class="label">{name} events</div>'
                f'<div class="value" style="color:{CLASS_COLORS[name]}">{int(counts.get(i, 0)):,}</div></div>',
                unsafe_allow_html=True,
            )

    with o2:
        heat = go.Figure(go.Heatmap(
            z=normalized, x=CLASS_NAMES, y=CLASS_NAMES,
            colorscale=[[0, "#141a3a"], [1, "#6C8CFF"]], zmin=0, zmax=1,
            text=[[f"{normalized[i, j]:.1%}<br>({confusion[i, j]:,})" for j in range(n_classes)]
                  for i in range(n_classes)],
            texttemplate="%{text}", showscale=False,
        ))
        heat.update_layout(
            title="Confusion matrix (row-normalised)", height=460,
            xaxis=dict(title="Predicted"), yaxis=dict(title="Actual", autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d7dcf5"), margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(heat, use_container_width=True)
