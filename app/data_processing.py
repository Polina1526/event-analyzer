import constants
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


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
    random_events.columns = [constants.THREADID_COL_NAME, "random_event_time", "window_start"]

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


def preprocess_data(
    df: pd.DataFrame, target_event: str, time_window: str, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg_data_target: pd.DataFrame = _aggregate_features_before_target_events(
        df=df, target_event=target_event, time_window=time_window
    )
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
    non_target_index["non_target_id"] = non_target_index["non_target_id"] + int(agg_data_target.target_id.max()) + 1
    agg_data_non_target = pd.merge(agg_data_non_target, non_target_index, on=constants.THREADID_COL_NAME)

    all_agg_data = pd.concat([agg_data_target, agg_data_non_target], axis=0)
    all_agg_data.target_id = all_agg_data.target_id.fillna(0).astype(int)
    all_agg_data.non_target_id = all_agg_data.non_target_id.fillna(0).astype(int)

    all_agg_data[constants.EVENT_CHAIN_ID_CON_NAME] = (
        all_agg_data[constants.THREADID_COL_NAME].astype(str)
        + "_"
        + (all_agg_data.target_id + all_agg_data.non_target_id).astype(str)
    )

    all_agg_data[f"{constants.EVENT_CHAIN_ID_CON_NAME}_encoded"] = LabelEncoder().fit_transform(
        all_agg_data[constants.EVENT_CHAIN_ID_CON_NAME]
    )
    all_agg_data[f"{constants.EVENT_TYPE_COL_NAME}_encoded"] = LabelEncoder().fit_transform(
        all_agg_data[constants.EVENT_TYPE_COL_NAME]
    )

    y_to_id_mapping = (
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

    return all_agg_data, y_to_id_mapping
