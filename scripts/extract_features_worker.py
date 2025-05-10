import os
import sys
import typing

import pandas as pd
from tsfresh import extract_features

from app import constants


N_JOBS_AVAILABLE: typing.Final[int] = os.cpu_count()


def _filter_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        [
            constants.TIMESTAMP_COL_NAME,
            f"{constants.EVENT_CHAIN_ID_CON_NAME}_encoded",
            f"{constants.EVENT_TYPE_COL_NAME}_encoded",
        ]
    ]


def _feature_extraction(df: pd.DataFrame, chunksize: int) -> pd.DataFrame:
    return extract_features(
        _filter_unnecessary_columns(
            df.sort_values([f"{constants.EVENT_CHAIN_ID_CON_NAME}_encoded", constants.TIMESTAMP_COL_NAME])
        ),
        column_id=f"{constants.EVENT_CHAIN_ID_CON_NAME}_encoded",
        column_sort=constants.TIMESTAMP_COL_NAME,
        chunksize=chunksize,
        n_jobs=N_JOBS_AVAILABLE - 2,
        disable_progressbar=True,
    )


def main(input_path: str, output_path: str, chunksize: int) -> None:
    df = pd.read_parquet(input_path)
    features: pd.DataFrame = _feature_extraction(df=df, chunksize=chunksize)
    features.to_parquet(output_path)


if __name__ == "__main__":
    input_path: str = sys.argv[1]
    output_path: str = sys.argv[2]
    chunksize: int = int(sys.argv[3])
    main(input_path, output_path, chunksize)
