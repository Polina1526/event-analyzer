import json
import os
import subprocess
import time

import constants
import pandas as pd
import pydantic
import streamlit as st
from catboost import CatBoostClassifier
from data_processing import preprocess_data
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute


st.set_page_config(page_title="Обучение новой модели", page_icon="📖")


def load_train_data() -> tuple[pd.DataFrame, str]:
    uploaded_file = st.file_uploader(label="Загрузите файл с данными для обучения и модели", type="csv")
    if not uploaded_file:
        st.stop()

    raw_data = pd.read_csv(uploaded_file)
    if raw_data.empty:
        st.error("Загруженный файл пуст")
        st.stop()
    return raw_data, uploaded_file.name


def column_standardization(df: pd.DataFrame) -> pd.DataFrame:
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
    return df


def get_data_preprocessing_settings(event_options: list[str], max_days_in_data: int) -> tuple[str, int]:
    with st.container(border=True):
        st.subheader("Настройки препроцессинга данных")
        col1, col2 = st.columns(2)
        with col1:
            _target_event_name: list = st.multiselect(
                label="Выберите название целевого события", options=event_options, max_selections=1
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
    feature_extruction_chunksize: int
    training_validation_size: float
    training_catboost_iterations: int
    training_catboost_depth: int
    training_catboost_learning_rate: float


def create_sidebar(threadid_count: int) -> SidebarSettings:
    data_count: int = st.sidebar.number_input(
        label="Кол-во цепочек, использующихся для обучения и валидации",
        min_value=2,
        max_value=threadid_count,
        value=min(constants.DEFAULT_DATA_COUNT, threadid_count),
        step=1,
    )
    feature_extruction_chunksize: int = st.sidebar.number_input(
        label="Кол-во цепочек в одном чанке при препроцессинге",
        min_value=1,
        max_value=threadid_count,
        value=min(constants.DEFAULT_CHUNKSIZE, threadid_count),
        step=1,
    )
    training_validation_size: float = st.sidebar.number_input(
        label="Доля валидационной выборки",
        min_value=0.0,
        max_value=1.0,
        value=constants.DEFAULT_VALIDATION_SIZE,
        step=0.05,
    )
    training_catboost_iterations: int = st.sidebar.number_input(
        label="Кол-во итераций при обучении",
        min_value=0,
        max_value=100000,
        value=constants.DEFAULT_CATBOOST_ITERATIONS,
        step=1,
    )
    training_catboost_depth: int = st.sidebar.number_input(
        label="Глубина деревьев в бустинге", min_value=0, max_value=100, value=constants.DEFAULT_CATBOOST_DEPTH, step=1
    )
    training_catboost_learning_rate: float = st.sidebar.number_input(
        label="Laerning rate", min_value=0.0, max_value=1.0, value=constants.DEFAULT_CATBOOST_LEARNING_RATE, step=0.001
    )
    return SidebarSettings(
        data_count=data_count,
        feature_extruction_chunksize=feature_extruction_chunksize,
        training_validation_size=training_validation_size,
        training_catboost_iterations=training_catboost_iterations,
        training_catboost_depth=training_catboost_depth,
        training_catboost_learning_rate=training_catboost_learning_rate,
    )


def limit_data_size(df: pd.DataFrame, data_count: int, random_state: int = 42) -> pd.DataFrame:
    return df[
        df[constants.THREADID_COL_NAME].isin(
            pd.Series(df[constants.THREADID_COL_NAME].unique()).sample(data_count, random_state=random_state)
        )
    ]


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
        st.text(result.stderr)
        st.stop()

    features: pd.DataFrame = pd.read_parquet(constants.FEATURE_VECTORIZATION_OUTPUT_PATH)
    os.remove(constants.FEATURE_VECTORIZATION_INPUT_PATH)
    os.remove(constants.FEATURE_VECTORIZATION_OUTPUT_PATH)
    return features


@st.cache_data(show_spinner="Заполнение пропусков в данных")
def _impute_extracted_features(df: pd.DataFrame) -> pd.DataFrame:
    return impute(df)


@st.cache_data(show_spinner="Отбор полезных фичей")
def _select_extracted_features(df: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return select_features(df, y.y)


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

    features_filtered: pd.DataFrame = _select_extracted_features(df=extracted_features, y=y)
    return features_filtered, y


@st.cache_data(show_spinner=False)
def learn_model(
    features: pd.DataFrame,
    y: pd.DataFrame,
    iterations: int,
    depth: int,
    learning_rate: float,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[CatBoostClassifier, pd.DataFrame]:
    X_train, X_test, y_train, y_test = train_test_split(features, y.y, test_size=test_size, random_state=random_state)

    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=False,
        allow_writing_files=False,
    )
    with st.spinner("Обучение модели"):
        model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

    with st.spinner("Подсчёт метрик на обученной модели"):
        probas_X_train = model.predict_proba(X_train)[:, 1]
        probas_X_test = model.predict_proba(X_test)[:, 1]

    metrics: dict[str, list[str]] = {
        "Обучающая выборка": [
            f"{roc_auc_score(y_train, probas_X_train):.4f}",
            f"{average_precision_score(y_train, probas_X_train):.4f}",
        ],
        "Валидационная выборка": [
            f"{roc_auc_score(y_test, probas_X_test):.4f}",
            f"{average_precision_score(y_test, probas_X_test):.4f}",
        ],
    }
    return model, pd.DataFrame.from_dict(metrics, orient="index", columns=["ROC-AUC", "PR-AUC"])


class ModelLearningResults(pydantic.BaseModel):
    learning_time: float
    learning_metrics: dict[str, str]
    data_file_name: str
    settings: dict


def _prepare_learning_metrics_for_saving(metrics_df: pd.DataFrame) -> dict[str, str]:
    metrics_series = metrics_df.stack()
    metrics_series.index = metrics_series.index.map(" ".join)
    return metrics_series.to_dict()


def model_learning_page() -> None:
    st.title("Обучение модели на данных с потоками событий")
    st.sidebar.header("Расширенные настройки")

    raw_data: pd.DataFrame
    file_name: str
    raw_data, file_name = load_train_data()
    raw_data = column_standardization(raw_data)

    target_event_name: str
    window_days: int
    target_event_name, window_days = get_data_preprocessing_settings(
        event_options=raw_data[constants.EVENT_TYPE_COL_NAME].unique(),
        max_days_in_data=(
            raw_data[constants.TIMESTAMP_COL_NAME].max() - raw_data[constants.TIMESTAMP_COL_NAME].min()
        ).days,
    )

    processed_data: pd.DataFrame
    y_to_id_mapping: pd.DataFrame
    processed_data, y_to_id_mapping = preprocess_data(
        df=raw_data, target_event=target_event_name, time_window=f"{window_days}D"
    )

    sidebar_settings: SidebarSettings = create_sidebar(
        threadid_count=processed_data[constants.THREADID_COL_NAME].nunique()
    )

    processed_data = limit_data_size(df=processed_data, data_count=sidebar_settings.data_count)
    # тут можно вставить инфу о том, сколько данных было отфильтровано  # TODO

    with st.form("learning_start_form"):
        model_name: str = st.text_input(label="Введите название модели для сохранения")
        st.form_submit_button("Запустить обучение модели")

    features: pd.DataFrame
    y: pd.DataFrame
    features, y = feature_extraction(
        df=processed_data, chunksize=sidebar_settings.feature_extruction_chunksize, y_to_id_mapping=y_to_id_mapping
    )

    model: CatBoostClassifier
    metrics_df: pd.DataFrame

    start_learning_time = time.time()
    model, metrics_df = learn_model(
        features=features,
        y=y,
        test_size=sidebar_settings.training_validation_size,
        iterations=sidebar_settings.training_catboost_iterations,
        depth=sidebar_settings.training_catboost_depth,
        learning_rate=sidebar_settings.training_catboost_learning_rate,
    )
    execution_learning_time = time.time() - start_learning_time

    st.subheader("Результаты обучения модели")
    st.write(f"Время обучения модели: {execution_learning_time: 0.4f} секунд")
    st.write(metrics_df)

    with st.spinner(f"Сохранение модели {model_name}.cbm"):
        model.save_model(constants.MODEL_SAVING_PATH.format(model_name))

        with open(constants.MODEL_PIVOT_TABLE_PATH, mode="r") as file:
            model_pivot_table_dict: dict[str, dict] = json.load(file)

        model_pivot_table_dict[model_name] = ModelLearningResults(
            learning_time=execution_learning_time,
            learning_metrics=_prepare_learning_metrics_for_saving(metrics_df),
            data_file_name=file_name,
            settings=sidebar_settings.model_dump(),
        ).model_dump()

        with open(constants.MODEL_PIVOT_TABLE_PATH, mode="w") as file:
            json.dump(model_pivot_table_dict, file, indent=4)

    st.success(f"Модель {model_name}.cbm успешно обучена и сохранена")


model_learning_page()
