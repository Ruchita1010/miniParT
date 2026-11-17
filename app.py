import streamlit as st

st.set_page_config(
    page_title="Particle Jet Classification",
    page_icon='*',
    layout="centered"
)

st.title("Particle Jet Classification")
st.write("The model classifies pairs of jets from CMS Open Data as coming from a Higgs boson decaying to two bottom quarks (Hbb), a Higgs boson decaying to two charm quarks (Hcc), or ordinary QCD background with no Higgs boson involved.")
