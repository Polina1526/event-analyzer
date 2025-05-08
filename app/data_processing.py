import constants
import numpy as np
import pandas as pd


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
    targets = targets[[constants.TIMESTAMP_COL_NAME, constants.TREADID_COL_NAME, "window_start"]].reset_index(drop=True)
    targets["target_id"] = targets.index

    merged = pd.merge(df, targets, on=constants.TREADID_COL_NAME, how="inner", suffixes=("", "_target"))

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

    users_with_target = df[df[constants.EVENT_TYPE_COL_NAME] == target_event][constants.TREADID_COL_NAME].unique()

    non_target_users = df[~df[constants.TREADID_COL_NAME].isin(users_with_target)]

    if non_target_users.empty:
        return pd.DataFrame()

    np.random.seed(random_state)
    random_events = (
        non_target_users.groupby(constants.TREADID_COL_NAME)
        .apply(lambda x: x.sample(1, random_state=random_state))
        .reset_index(drop=True)
    )

    random_events["window_start"] = random_events[constants.TIMESTAMP_COL_NAME] - pd.to_timedelta(time_window)
    random_events = random_events[[constants.TIMESTAMP_COL_NAME, constants.TREADID_COL_NAME, "window_start"]]
    random_events.columns = [constants.TREADID_COL_NAME, "random_event_time", "window_start"]

    merged = pd.merge(
        non_target_users,
        random_events,
        on=constants.TREADID_COL_NAME,
        how="inner",
        suffixes=("", "_target"),
    )

    mask = (merged[constants.TIMESTAMP_COL_NAME] >= merged["window_start"]) & (
        merged[constants.TIMESTAMP_COL_NAME] <= merged["random_event_time"]
    )
    result = merged[mask].drop(columns=["window_start", "random_event_time"])

    user_event_counts = result[constants.TREADID_COL_NAME].value_counts()
    valid_users = user_event_counts[user_event_counts >= min_events].index
    result = result[result[constants.TREADID_COL_NAME].isin(valid_users)]

    return result.reset_index(drop=True)
