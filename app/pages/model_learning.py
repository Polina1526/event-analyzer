import json
import time

import constants
import data_processing
import pandas as pd
import pydantic
import streamlit as st
import utils
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tsfresh import select_features


st.set_page_config(page_title="Обучение новой модели", page_icon="📖")


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


@st.cache_data(show_spinner="Отбор полезных фичей")
def _select_features(df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    return select_features(df, y)


@st.cache_data(show_spinner=False)
def learn_model(
    features: pd.DataFrame,
    y: pd.DataFrame,
    iterations: int,
    depth: int,
    learning_rate: float,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[CatBoostClassifier, pd.DataFrame, list[str]]:
    X_train, X_test, y_train, y_test = train_test_split(features, y.y, test_size=test_size, random_state=random_state)

    main_features: list[str] = list(_select_features(df=X_train, y=y_train).columns)
    X_train = X_train[main_features]
    X_test = X_test[main_features]

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
    return model, pd.DataFrame.from_dict(metrics, orient="index", columns=["ROC-AUC", "PR-AUC"]), main_features


class ModelLearningResults(pydantic.BaseModel):
    learning_time: float
    feature_extraction_time: float
    target_event_name: str
    window_days: int
    learning_metrics: dict[str, str]
    data_file_name: str
    settings: dict
    main_features: list[str]


def _prepare_learning_metrics_for_saving(metrics_df: pd.DataFrame) -> dict[str, str]:
    metrics_series = metrics_df.stack()
    metrics_series.index = metrics_series.index.map(" ".join)
    return metrics_series.to_dict()


def model_learning_page() -> None:
    st.title("Обучение модели на данных с потоками событий")
    st.sidebar.header("Расширенные настройки")

    raw_data: pd.DataFrame
    file_name: str
    raw_data, file_name = utils.load_train_data(label="Загрузите файл с данными для обучения и модели")
    raw_data = data_processing.column_standardization(raw_data)

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
    processed_data, y_to_id_mapping, _ = data_processing.preprocess_data(
        df=raw_data, target_event=target_event_name, time_window=f"{window_days}D"
    )

    sidebar_settings: SidebarSettings = create_sidebar(
        threadid_count=processed_data[constants.THREADID_COL_NAME].nunique()
    )

    processed_data = limit_data_size(df=processed_data, data_count=sidebar_settings.data_count)
    # тут можно вставить инфу о том, сколько данных было отфильтровано  # TODO

    with st.form("learning_start_form"):
        model_name: str = st.text_input(label="Введите название модели для сохранения")
        form_send: bool = st.form_submit_button("Запустить обучение модели")
    if not form_send:
        st.stop()

    features: pd.DataFrame
    y: pd.DataFrame
    start_feature_extraction_time = time.time()
    features, y = data_processing.feature_extraction(
        df=processed_data, chunksize=sidebar_settings.feature_extruction_chunksize, y_to_id_mapping=y_to_id_mapping
    )
    execution_feature_extraction_time: float = time.time() - start_feature_extraction_time
    st.write(f"Время извлечения признаков из данных: {execution_feature_extraction_time: 0.4f} секунд")

    model: CatBoostClassifier
    metrics_df: pd.DataFrame
    main_features: list[str]
    start_learning_time = time.time()
    model, metrics_df, main_features = learn_model(
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
            feature_extraction_time=execution_feature_extraction_time,
            target_event_name=target_event_name,
            window_days=window_days,
            learning_metrics=_prepare_learning_metrics_for_saving(metrics_df),
            data_file_name=file_name,
            settings=sidebar_settings.model_dump(),
            main_features=main_features,
        ).model_dump()

        with open(constants.MODEL_PIVOT_TABLE_PATH, mode="w") as file:
            json.dump(model_pivot_table_dict, file, indent=4)

    st.success(f"Модель {model_name}.cbm успешно обучена и сохранена")


model_learning_page()
