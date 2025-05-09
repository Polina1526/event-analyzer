import constants
import pandas as pd
import pydantic
import streamlit as st
from data_processing import preprocess_data


def load_train_data() -> pd.DataFrame:
    uploaded_file = st.file_uploader(label="Загрузите файл с данными для обучения и модели", type="csv")
    if not uploaded_file:
        st.write("Ожидается загрузка данных")
        st.stop()

    raw_data = pd.read_csv(uploaded_file)
    if raw_data.empty:
        st.error("Загруженный файл пуст")
        st.stop()

    return raw_data


def column_standardization(df: pd.DataFrame) -> pd.DataFrame:
    with st.container(border=True):
        st.subheader("Информация о датасете")
        col1, col2, col3 = st.columns(3)
        with col1:
            _timestamp_col: list = st.multiselect(
                label="Выберете поле со временем событий", options=df.columns, max_selections=1
            )
        with col2:
            _threadid_col: list = st.multiselect(
                label="Выберете поле содержащее id потоков", options=df.columns, max_selections=1
            )
        with col3:
            _event_col: list = st.multiselect(
                label="Выберете поле содержащее события", options=df.columns, max_selections=1
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

    return df


def get_data_preprocessing_settings(event_options: list[str], max_days_in_data: int) -> tuple[str, int]:
    with st.container(border=True):
        st.subheader("Настройки препроцессинга данных")
        col1, col2 = st.columns(2)
        with col1:
            _target_event_name: list = st.multiselect(
                label="Выберете название целевого события", options=event_options, max_selections=1
            )
        with col2:
            window_days: int = st.number_input(
                label="Временное окно цепочки событий (в днях)",
                min_value=1,
                max_value=max_days_in_data,
                value=min(constants.DEFAULT_WINDOW_DAYS, max_days_in_data),
                step=1,
            )
    if not _target_event_name or not window_days:
        st.stop()

    return _target_event_name[0], window_days


class SidebarSettings(pydantic.BaseModel):
    data_count: int


def create_sidebar(threadid_count: int) -> SidebarSettings:
    st.sidebar.header("Дополнительные настройки")
    data_count: int = st.sidebar.number_input(
        label="Кол-во цепочек, использующихся для обучения и валидации",
        min_value=2,
        max_value=threadid_count,
        value=constants.DEFAULT_DATA_COUNT,
        step=1,
    )
    return SidebarSettings(data_count=data_count)


def limit_data_size(df: pd.DataFrame, data_count: int, random_state: int = 42) -> pd.DataFrame:
    return df[
        df[constants.THREADID_COL_NAME].isin(
            pd.Series(df[constants.THREADID_COL_NAME].unique()).sample(data_count, random_state=random_state)
        )
    ]


def main_app() -> None:
    st.title("Анализатор событий")

    raw_data: pd.DataFrame = load_train_data()
    raw_data = column_standardization(raw_data)

    target_event_name: str
    window_days: int
    target_event_name, window_days = get_data_preprocessing_settings(
        event_options=raw_data[constants.EVENT_TYPE_COL_NAME].unique(),
        max_days_in_data=(
            raw_data[constants.TIMESTAMP_COL_NAME].max() - raw_data[constants.TIMESTAMP_COL_NAME].min()
        ).days,
    )

    processed_data: pd.DataFrame = preprocess_data(
        df=raw_data, target_event=target_event_name, time_window=f"{window_days}D"
    )

    sidebar_settings: SidebarSettings = create_sidebar(threadid_count=raw_data[constants.THREADID_COL_NAME].nunique())

    processed_data = limit_data_size(df=processed_data, data_count=sidebar_settings.data_count)
    # тут можно вставить инфу о том, сколько данных было отфильтровано

    st.write(sidebar_settings)  # TODO

    st.write(raw_data.head(10))  # TODO
    st.write(processed_data.head(10))  # TODO


main_app()
