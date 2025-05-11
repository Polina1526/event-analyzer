import json
import time
from io import BytesIO

import constants
import data_processing
import pandas as pd
import pydantic
import streamlit as st
import utils
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder


st.set_page_config(page_title="Анализ данных", page_icon="🔎")


def _prepare_for_display_model_pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df[
        [
            "data_file_name",
            "target_event_name",
            "learning_metrics",
            "learning_time",
            "feature_extraction_time",
            "settings",
        ]
    ]
    df = df.reset_index()

    learning_metrics_df: pd.DataFrame = pd.json_normalize(df["learning_metrics"])
    col_position: int = df.columns.get_loc("learning_metrics")
    df = df.drop(columns=["learning_metrics"])
    df = pd.concat([df.iloc[:, :col_position], learning_metrics_df, df.iloc[:, col_position:]], axis=1)

    df = df.rename(
        columns={
            "index": "Название модели",
            "data_file_name": "Название файла с данными",
            "target_event_name": "Название целевого события",
            "learning_metrics": "Метрики обучения модели",
            "learning_time": "Время обучения модели",
            "feature_extraction_time": "Время извлечения фичей",
            "settings": "Настройки модели при обучении",
        }
    )
    return df


class SidebarSettings(pydantic.BaseModel):
    feature_extruction_chunksize: int


def create_sidebar(threadid_count: int) -> SidebarSettings:
    feature_extruction_chunksize: int = st.sidebar.number_input(
        label="Кол-во цепочек в одном чанке при препроцессинге",
        min_value=1,
        max_value=threadid_count,
        value=min(constants.DEFAULT_CHUNKSIZE, threadid_count),
        step=1,
    )
    return SidebarSettings(feature_extruction_chunksize=feature_extruction_chunksize)


def prepare_for_excel_download(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()


def transform_y(
    y_df: pd.DataFrame, threadid_col_name: str, target_event_name: str, event_chain_id_col_le: LabelEncoder
) -> pd.DataFrame:
    y_df = y_df.reset_index(f"{constants.EVENT_CHAIN_ID_CON_NAME}_encoded")
    y_df[constants.EVENT_CHAIN_ID_CON_NAME] = event_chain_id_col_le.inverse_transform(
        y_df[f"{constants.EVENT_CHAIN_ID_CON_NAME}_encoded"]
    )
    y_df[threadid_col_name] = y_df[constants.EVENT_CHAIN_ID_CON_NAME].str.split("_").str[0]
    y_df = y_df.drop(columns=["y", f"{constants.EVENT_CHAIN_ID_CON_NAME}_encoded", constants.EVENT_CHAIN_ID_CON_NAME])
    y_df = y_df[[threadid_col_name, f"{target_event_name}_occurrence_prob"]]
    return y_df


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

    _model_name: list[str] = st.multiselect(
        "Выберите модель для анализа данных", model_pivot_table_df.index, max_selections=1
    )
    if not _model_name:
        st.stop()
    model_name: str = _model_name[0]
    model: CatBoostClassifier = CatBoostClassifier().load_model(constants.MODEL_SAVING_PATH.format(model_name))
    main_features: list[str] = model_pivot_table_df.loc[
        (model_pivot_table_df.index == model_name), "main_features"
    ].iloc[0]
    target_event_name: str = model_pivot_table_df.loc[
        (model_pivot_table_df.index == model_name), "target_event_name"
    ].iloc[0]
    window_days: int = model_pivot_table_df.loc[(model_pivot_table_df.index == model_name), "window_days"].iloc[0]

    raw_data: pd.DataFrame
    raw_data, _ = utils.load_train_data(label="Загрузите файл с данными для анализа")
    threadid_col_name: str
    raw_data, threadid_col_name = data_processing.column_standardization(raw_data, return_threadid_col_name=True)

    # TODO: просто заплатка для тестирования
    raw_data = raw_data[raw_data[constants.EVENT_TYPE_COL_NAME] != target_event_name]
    # TODO: просто заплатка для тестирования

    processed_data: pd.DataFrame
    y_to_id_mapping: pd.DataFrame
    event_chain_id_col_le: LabelEncoder
    processed_data, y_to_id_mapping, event_chain_id_col_le = data_processing.preprocess_data(
        df=raw_data, target_event=target_event_name, time_window=f"{window_days}D"
    )

    sidebar_settings: SidebarSettings = create_sidebar(
        threadid_count=processed_data[constants.THREADID_COL_NAME].nunique()
    )

    with st.form("analyzing_start_form"):
        result_file_name: str = st.text_input(label="Введите название файла для сохранения результатов анализа")
        form_send: bool = st.form_submit_button("Запустить анализ данных")
    if not form_send:
        st.stop()

    start_feature_extraction_time = time.time()
    features: pd.DataFrame
    y: pd.DataFrame
    features, y = data_processing.feature_extraction(
        df=processed_data, chunksize=sidebar_settings.feature_extruction_chunksize, y_to_id_mapping=y_to_id_mapping
    )
    execution_feature_extraction_time: float = time.time() - start_feature_extraction_time
    st.write(f"Время извлечения фичей: {execution_feature_extraction_time:.4f} сек")

    features = features[main_features]

    with st.spinner("Анализ датасета с помощью обученно модели"):
        y[f"{target_event_name}_occurrence_prob"] = model.predict_proba(features)[:, 1]

    y = transform_y(
        y_df=y,
        threadid_col_name=threadid_col_name,
        target_event_name=target_event_name,
        event_chain_id_col_le=event_chain_id_col_le,
    )
    st.download_button(
        label="Скачать результат",
        data=prepare_for_excel_download(y),
        file_name=f"{result_file_name}.xlsx",
        icon=":material/download:",
    )


dataset_analyzing_page()
