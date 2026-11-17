from pathlib import Path

import awkward as ak
import joblib
import numpy as np
import uproot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# We avoid DeepJet/DeepCSV variables as requested.
# Using kinematics + energy fractions + pileup/multiplicity info.
FEATURE_NAMES = [
    'Jet_pt', 'Jet_eta', 'Jet_phi', 'Jet_mass',
    'Jet_chHEF', 'Jet_neHEF', 'Jet_chEmEF', 'Jet_neEmEF',
    'Jet_nConstituents', 'Jet_puId'
]

# Labels: 0 = Hbb, 1 = Hcc, 2 = QCD

# Streaming paths for the three validated CMS Open Data files this lesson
# uses (see "Working in Google Colab" for how these were confirmed).
TTHTOBB_PATH = "root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v2/260000/410F948C-6956-2D45-A170-DE6431E02281.root"

TTHTOCC_PATH = "root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/ttHTocc_M125_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1/50000/5C12D3AA-9311-B840-BB5D-4155D7FF66E4.root"

QCD_BCTOE_PATH = "root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/QCD_Pt_80to170_bcToE_TuneCP5_13TeV_pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v2/270000/A133135A-C83E-D245-846F-210C7AD2D29C.root"

# Extracting features and matching truth labels
def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2*np.pi) - np.pi

def extract_features(filepath, label, is_signal=True, max_events=None):
    tree = uproot.open(filepath)["Events"]
    assert isinstance(tree, uproot.TTree)

    # Load required branches
    branches = FEATURE_NAMES.copy()
    if is_signal:
        branches += [
            "GenPart_pdgId", "GenPart_pt", "GenPart_eta",
            "GenPart_phi", "GenPart_mass", "GenPart_genPartIdxMother"
        ]

    events = tree.arrays(branches, entry_stop=max_events)

    if is_signal:
        # Determine target quark based on label (0: b-quark=5, 1: c-quark=4)
        target_pdg = 5 if label == 0 else 4

        mother_idx = events.GenPart_genPartIdxMother
        valid = mother_idx >= 0
        mother_pdg = ak.where(valid, events.GenPart_pdgId[mother_idx], -999)

        is_higgs_dau = (abs(events.GenPart_pdgId) == target_pdg) & (mother_pdg == 25)
        mask = ak.num(events.GenPart_pt[is_higgs_dau]) == 2
        events = events[mask]
        is_higgs_dau = is_higgs_dau[mask]

        # Build 4-vectors
        jets = ak.zip({
            "pt": events.Jet_pt, "eta": events.Jet_eta,
            "phi": events.Jet_phi, "mass": events.Jet_mass
        }, with_name="Momentum4D")

        dau = ak.zip({
            "pt": events.GenPart_pt[is_higgs_dau], "eta": events.GenPart_eta[is_higgs_dau],
            "phi": events.GenPart_phi[is_higgs_dau], "mass": events.GenPart_mass[is_higgs_dau]
        }, with_name="Momentum4D")

        d1, d2 = dau[:, 0], dau[:, 1]

        # Match using dR < 0.4
        dr1 = np.sqrt((jets.eta - d1.eta[:, None])**2 + delta_phi(jets.phi, d1.phi[:, None])**2)
        dr2 = np.sqrt((jets.eta - d2.eta[:, None])**2 + delta_phi(jets.phi, d2.phi[:, None])**2)
        matched = (dr1 < 0.4) | (dr2 < 0.4)

        matched_events = {f: events[f][matched] for f in FEATURE_NAMES}
        mask_2jets = ak.num(matched_events["Jet_pt"]) == 2
        final_events = {f: v[mask_2jets] for f, v in matched_events.items()}
    else:
        mask_2jets = ak.num(events.Jet_pt) >= 2
        final_events = {f: events[f][mask_2jets][:, :2] for f in FEATURE_NAMES}

    # Stack features into a NumPy array of shape (N_events, 2_jets, N_features)
    feature_list = []
    for feat in FEATURE_NAMES:
        # Fill missing values with 0 (e.g., puId might have NaNs depending on pt)
        arr = ak.fill_none(final_events[feat], 0)
        feature_list.append(ak.to_numpy(arr))

    X = np.stack(feature_list, axis=-1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.full(X.shape[0], label)

    print(f"Loaded label {label}: {X.shape[0]} events")
    return X, y


# Extract features
X_bb, y_bb = extract_features(
    TTHTOBB_PATH, label=0, is_signal=True, max_events=None
)

X_cc, y_cc = extract_features(
    TTHTOCC_PATH, label=1, is_signal=True, max_events=None
)

X_qcd, y_qcd = extract_features(
    QCD_BCTOE_PATH, label=2, is_signal=False, max_events=None
)

# Combine datasets
X = np.concatenate([X_bb, X_cc, X_qcd], axis=0)
y = np.concatenate([y_bb, y_cc, y_qcd], axis=0)

# Check dataset
print("Combined dataset shape:", X.shape)
print("Labels:", np.unique(y, return_counts=True))

# First split: 80% development data, 20% test data
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Second split: 70% training, 10% validation overall
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.125,
    random_state=42,
    stratify=y_temp
)

X_train, X_val, y_train, y_val = (np.asarray(a) for a in (X_train, X_val, y_train, y_val))
X_test, y_test = np.asarray(X_test), np.asarray(y_test)

# Determine actual feature dimension
N_FEATURES = X_train.shape[-1]

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

scaler = StandardScaler()

X_train_scaled = np.asarray(
    scaler.fit_transform(X_train.reshape(-1, N_FEATURES))
).reshape(-1, 2, N_FEATURES)

X_val_scaled = np.asarray(
    scaler.transform(X_val.reshape(-1, N_FEATURES))
).reshape(-1, 2, N_FEATURES)

X_test_scaled = np.asarray(
    scaler.transform(X_test.reshape(-1, N_FEATURES))
).reshape(-1, 2, N_FEATURES)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Save processed datasets
np.savez_compressed(
    PROCESSED_DIR / "dataset.npz",
    X_train=X_train_scaled,
    y_train=y_train,
    X_val=X_val_scaled,
    y_val=y_val,
    X_test=X_test_scaled,
    y_test=y_test,
)

# Save the fitted scaler
joblib.dump(scaler, PROCESSED_DIR / "scaler.joblib")

print("Processed dataset and scaler saved successfully.")

print("\nFinal dataset:")
print("Training:", X_train_scaled.shape)
print("Validation:", X_val_scaled.shape)
print("Testing:", X_test_scaled.shape)
print("Input features per jet:", N_FEATURES)
