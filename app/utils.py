import pandas as pd
import streamlit as st


def load_train_data(label: str) -> tuple[pd.DataFrame, str]:
    uploaded_file = st.file_uploader(label=label, type="csv")
    if not uploaded_file:
        st.stop()

    raw_data = pd.read_csv(uploaded_file)
    if raw_data.empty:
        st.error("Загруженный файл пуст")
        st.stop()
    return raw_data, uploaded_file.name
