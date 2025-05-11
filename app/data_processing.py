import os
import subprocess

import constants
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from tsfresh.utilities.dataframe_functions import impute


def _aggregate_features_before_target_events(df: pd.DataFrame, target_event: str, time_window: str) -> pd.DataFrame:
    """
    Агрегирует данные для каждого пользователя до каждого целевого события в заданном временном окне.

    Параметры:
    ----------
    df : pd.DataFrame
        Исходный датафрейм с колонками: timestamp, threadid, event_type
    target_event : str
        Название целевого события
    time_window : str
        Временное окно для сбора цепочек (формат pandas timedelta, например "14D")

    Возвращает:
    -----------
    pd.DataFrame
        Датафрейм с агрегированными фичами для каждого целевого события
    """
    df = df.copy()
    df[constants.TIMESTAMP_COL_NAME] = pd.to_datetime(df[constants.TIMESTAMP_COL_NAME])

    targets = df[df[constants.EVENT_TYPE_COL_NAME] == target_event].copy()
    if targets.empty:
        return pd.DataFrame()

    targets["window_start"] = targets[constants.TIMESTAMP_COL_NAME] - pd.to_timedelta(time_window)
    targets = targets[[constants.TIMESTAMP_COL_NAME, constants.THREADID_COL_NAME, "window_start"]].reset_index(
        drop=True
    )
    targets["target_id"] = targets.index

    merged = pd.merge(df, targets, on=constants.THREADID_COL_NAME, how="inner", suffixes=("", "_target"))

    mask = (merged[constants.TIMESTAMP_COL_NAME] >= merged["window_start"]) & (
        merged[constants.TIMESTAMP_COL_NAME] < merged["timestamp_target"]
    )

    return merged[mask].drop(columns=["window_start"])


def _aggregate_features_before_non_target_events(
    df: pd.DataFrame, target_event: str, time_window: str = "14D", min_events: int = 1, random_state: int | None = None
):
    """
    Возвращает цепочки событий для пользователей без целевого события,
    начиная со случайного события и собирая историю назад во времени по длине окна.

    Параметры:
    ----------
    df : pd.DataFrame
        Исходный датафрейм с колонками: timestamp, threadid, event_type
    target_event : str
        Тип целевого события (которого НЕ должно быть у пользователя)
    time_window : str
        Временное окно для сбора цепочек (формат pandas timedelta, например "14D")
    min_events : int
        Минимальное количество событий для включения пользователя в результат
    random_state : int, optional
        Seed для воспроизводимости случайного выбора

    Возвращает:
    -----------
    pd.DataFrame
        Датафрейм с событиями пользователей, у которых не было целевого события,
        с цепочками, идущими назад от случайного события длиной с заданное окно
    """
    df = df.copy()
    df[constants.TIMESTAMP_COL_NAME] = pd.to_datetime(df[constants.TIMESTAMP_COL_NAME])

    users_with_target = df[df[constants.EVENT_TYPE_COL_NAME] == target_event][constants.THREADID_COL_NAME].unique()

    non_target_users = df[~df[constants.THREADID_COL_NAME].isin(users_with_target)]

    if non_target_users.empty:
        return pd.DataFrame()

    np.random.seed(random_state)
    random_events = (
        non_target_users.groupby(constants.THREADID_COL_NAME)
        .apply(lambda x: x.sample(1, random_state=random_state))
        .reset_index(drop=True)
    )

    random_events["window_start"] = random_events[constants.TIMESTAMP_COL_NAME] - pd.to_timedelta(time_window)
    random_events = random_events[[constants.TIMESTAMP_COL_NAME, constants.THREADID_COL_NAME, "window_start"]]
    random_events = random_events.rename(columns={constants.TIMESTAMP_COL_NAME: "random_event_time"})

    merged = pd.merge(
        non_target_users,
        random_events,
        on=constants.THREADID_COL_NAME,
        how="inner",
        suffixes=("", "_target"),
    )

    mask = (merged[constants.TIMESTAMP_COL_NAME] >= merged["window_start"]) & (
        merged[constants.TIMESTAMP_COL_NAME] <= merged["random_event_time"]
    )
    result = merged[mask].drop(columns=["window_start", "random_event_time"])

    user_event_counts = result[constants.THREADID_COL_NAME].value_counts()
    valid_users = user_event_counts[user_event_counts >= min_events].index
    result = result[result[constants.THREADID_COL_NAME].isin(valid_users)]

    return result.reset_index(drop=True)


@st.cache_data(show_spinner="Идёт обработка данных")
def preprocess_data(
    df: pd.DataFrame, target_event: str, time_window: str, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, LabelEncoder]:
    agg_data_target: pd.DataFrame = _aggregate_features_before_target_events(
        df=df, target_event=target_event, time_window=time_window
    )
    if not agg_data_target.empty:
        agg_data_target.target_id = agg_data_target.target_id + 1

    agg_data_non_target: pd.DataFrame = _aggregate_features_before_non_target_events(
        df=df, target_event=target_event, time_window=time_window, min_events=1, random_state=random_state
    )

    non_target_index: pd.DataFrame = (
        pd.Series(agg_data_non_target[constants.THREADID_COL_NAME].unique())
        .to_frame(constants.THREADID_COL_NAME)
        .reset_index()
        .rename(columns={"index": "non_target_id"})
    )
    if not agg_data_target.empty:
        non_target_index["non_target_id"] = non_target_index["non_target_id"] + int(agg_data_target.target_id.max()) + 1
    else:
        non_target_index["non_target_id"] = non_target_index["non_target_id"] + 1
    agg_data_non_target = pd.merge(agg_data_non_target, non_target_index, on=constants.THREADID_COL_NAME)

    all_agg_data = pd.concat([agg_data_target, agg_data_non_target], axis=0)
    if agg_data_target.empty:
        all_agg_data["target_id"] = 0
    all_agg_data.target_id = all_agg_data.target_id.fillna(0).astype(int)
    all_agg_data.non_target_id = all_agg_data.non_target_id.fillna(0).astype(int)

    all_agg_data[constants.EVENT_CHAIN_ID_CON_NAME] = (
        all_agg_data[constants.THREADID_COL_NAME].astype(str)
        + "_"
        + (all_agg_data.target_id + all_agg_data.non_target_id).astype(str)
    )

    event_chain_id_col_le: LabelEncoder = LabelEncoder()
    all_agg_data[f"{constants.EVENT_CHAIN_ID_CON_NAME}_encoded"] = event_chain_id_col_le.fit_transform(
        all_agg_data[constants.EVENT_CHAIN_ID_CON_NAME]
    )
    all_agg_data[f"{constants.EVENT_TYPE_COL_NAME}_encoded"] = LabelEncoder().fit_transform(
        all_agg_data[constants.EVENT_TYPE_COL_NAME]
    )

    y_to_id_mapping: pd.DataFrame = (
        all_agg_data[f"{constants.EVENT_CHAIN_ID_CON_NAME}_encoded"]
        .to_frame(f"{constants.EVENT_CHAIN_ID_CON_NAME}_encoded")
        .assign(y=0)
    )
    y_to_id_mapping.loc[
        y_to_id_mapping[f"{constants.EVENT_CHAIN_ID_CON_NAME}_encoded"].isin(
            all_agg_data[all_agg_data.target_id != 0][f"{constants.EVENT_CHAIN_ID_CON_NAME}_encoded"].unique()
        ),
        "y",
    ] = 1

    return all_agg_data, y_to_id_mapping, event_chain_id_col_le


def column_standardization(
    df: pd.DataFrame, return_threadid_col_name: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, str]:
    with st.container(border=True):
        st.subheader("Информация о датасете")
        col1, col2, col3 = st.columns(3)
        with col1:
            _timestamp_col: list = st.multiselect(
                label="Выберите поле со временем событий", options=df.columns, max_selections=1
            )
        with col2:
            _threadid_col: list = st.multiselect(
                label="Выберите поле содержащее id потоков", options=df.columns, max_selections=1
            )
        with col3:
            _event_col: list = st.multiselect(
                label="Выберите поле содержащее события", options=df.columns, max_selections=1
            )
    if not _timestamp_col or not _threadid_col or not _event_col:
        st.stop()

    timestamp_col: str = _timestamp_col[0]
    threadid_col: str = _threadid_col[0]
    event_col: str = _event_col[0]

    if len(set([timestamp_col, threadid_col, event_col])) < 3:
        st.error("Выбранные поля не могу совпадать")
        st.stop()

    df = df[[timestamp_col, threadid_col, event_col]]
    df = df.rename(
        columns={
            timestamp_col: constants.TIMESTAMP_COL_NAME,
            threadid_col: constants.THREADID_COL_NAME,
            event_col: constants.EVENT_TYPE_COL_NAME,
        }
    )
    df[constants.TIMESTAMP_COL_NAME] = pd.to_datetime(df[constants.TIMESTAMP_COL_NAME], unit="ms")
    df[constants.THREADID_COL_NAME] = df[constants.THREADID_COL_NAME].astype(int)
    df[constants.EVENT_TYPE_COL_NAME] = df[constants.EVENT_TYPE_COL_NAME].astype(str)
    if return_threadid_col_name:
        return df, threadid_col
    return df


@st.cache_data(show_spinner="Извлечение признаков цепочек из данных")
def _feature_extraction(df: pd.DataFrame, chunksize: int) -> pd.DataFrame:
    df.to_parquet(constants.FEATURE_VECTORIZATION_INPUT_PATH)
    worker_path = os.path.abspath(os.path.join(os.path.dirname(__file__), constants.FEATURE_EXTRACTION_SCRIPT_PATH))
    result = subprocess.run(
        [
            "python",
            worker_path,
            constants.FEATURE_VECTORIZATION_INPUT_PATH,
            constants.FEATURE_VECTORIZATION_OUTPUT_PATH,
            str(chunksize),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        st.error("Ошибка при вычислении признаков:")
        st.error(result.stderr)
        st.stop()

    features: pd.DataFrame = pd.read_parquet(constants.FEATURE_VECTORIZATION_OUTPUT_PATH)
    os.remove(constants.FEATURE_VECTORIZATION_INPUT_PATH)
    os.remove(constants.FEATURE_VECTORIZATION_OUTPUT_PATH)
    return features


@st.cache_data(show_spinner="Заполнение пропусков в данных")
def _impute_extracted_features(df: pd.DataFrame) -> pd.DataFrame:
    return impute(df)


def feature_extraction(
    df: pd.DataFrame, chunksize: int, y_to_id_mapping: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    extracted_features = _feature_extraction(df=df, chunksize=chunksize)
    extracted_features = _impute_extracted_features(extracted_features)

    y = (
        y_to_id_mapping[
            y_to_id_mapping[f"{constants.EVENT_CHAIN_ID_CON_NAME}_encoded"].isin(
                df[f"{constants.EVENT_CHAIN_ID_CON_NAME}_encoded"].unique()
            )
        ]
        .sort_values(f"{constants.EVENT_CHAIN_ID_CON_NAME}_encoded")
        .drop_duplicates(f"{constants.EVENT_CHAIN_ID_CON_NAME}_encoded")
        .set_index(f"{constants.EVENT_CHAIN_ID_CON_NAME}_encoded")
    )
    return extracted_features, y
