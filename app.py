import streamlit as st

st.set_page_config(
    page_title="Test",
    page_icon="✝️",
    layout="wide"
)

st.markdown("""
<style>
body { background: #090b11 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

st.title("Calvary Greetings, Beloved! ✝️")
st.write("If you can see this, the UI is working.")

if st.button("Test Button"):
    st.success("Button works!")
