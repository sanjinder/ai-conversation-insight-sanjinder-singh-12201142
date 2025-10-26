import streamlit as st
import importlib

# Import modules but keep UI navigation in this file
import app as live_app
import dataset_analyzer as dataset_app

def main():
    st.set_page_config(page_title="Elevate — Navigation", layout="wide")
    st.sidebar.title("Elevate Navigation")
    choice = st.sidebar.radio("Choose section", ["Live Feedback", "Dataset Analyzer"])

    if choice == "Live Feedback":
        # Use module's main function
        live_app.main()
    else:
        dataset_app.main()

if __name__ == "__main__":
    main()
