import json

import constants
import pandas as pd
import streamlit as st


def _prepare_for_display_model_pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["data_file_name", "learning_metrics", "learning_time", "settings"]]
    df = df.reset_index()

    learning_metrics_df: pd.DataFrame = pd.json_normalize(df["learning_metrics"])
    col_position: int = df.columns.get_loc("learning_metrics")
    df = df.drop(columns=["learning_metrics"])
    df = pd.concat([df.iloc[:, :col_position], learning_metrics_df, df.iloc[:, col_position:]], axis=1)
    # df = pd.concat([df.drop("learning_metrics", axis=1), learning_metrics_df], axis=1)

    df = df.rename(
        columns={
            "index": "Название модели",
            "data_file_name": "Название файла с данными",
            "learning_metrics": "Метрики обучения модели",
            "learning_time": "Время обучения модели",
            "settings": "Настройки модели при обучении",
        }
    )
    return df


def dataset_analyzing_page() -> None:
    st.title("Анализ событий с помощью обученной модели")

    with open(constants.MODEL_PIVOT_TABLE_PATH, "r") as file:
        model_pivot_table_dict: dict[str, dict] = json.load(file)

    model_pivot_table_df: pd.DataFrame = pd.DataFrame.from_dict(model_pivot_table_dict, orient="index")

    st.header("Доступные на данный момент модели")
    st.dataframe(
        _prepare_for_display_model_pivot_table(model_pivot_table_df),
        column_config={"settings": st.column_config.JsonColumn("Настройки модели при обучении", width="large")},
        hide_index=True,
    )

    st.multiselect("Выберите модель для анализа данных", model_pivot_table_df.index, max_selections=1)


dataset_analyzing_page()
