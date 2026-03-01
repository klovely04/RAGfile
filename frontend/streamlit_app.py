import httpx
import streamlit as st

st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("RAG Knowledge Assistant - Day 1")

api_base = st.text_input("API Base URL", value="http://localhost:8000")

if st.button("Check Health"):
    try:
        resp = httpx.get(f"{api_base}/api/v1/health", timeout=5.0)
        st.success(f"HTTP {resp.status_code}")
        st.json(resp.json())
    except Exception as exc:
        st.error(f"Request failed: {exc}")
