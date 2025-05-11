import typing


TIMESTAMP_COL_NAME: typing.Final[str] = "timestamp"
THREADID_COL_NAME: typing.Final[str] = "threadid"
EVENT_TYPE_COL_NAME: typing.Final[str] = "event_type"

EVENT_CHAIN_ID_CON_NAME: typing.Final[str] = "event_chain_id"

DEFAULT_WINDOW_DAYS: typing.Final[int] = 14
DEFAULT_DATA_COUNT: typing.Final[int] = 50_000

FEATURE_EXTRACTION_SCRIPT_PATH: typing.Final[str] = "../scripts/extract_features_worker.py"  # ../
FEATURE_VECTORIZATION_INPUT_PATH: typing.Final[str] = "scripts/tmp_data/input.parquet"
FEATURE_VECTORIZATION_OUTPUT_PATH: typing.Final[str] = "scripts/tmp_data/output.parquet"
DEFAULT_CHUNKSIZE: typing.Final[int] = 50

DEFAULT_VALIDATION_SIZE: typing.Final[float] = 0.2
DEFAULT_CATBOOST_ITERATIONS: typing.Final[int] = 300
DEFAULT_CATBOOST_DEPTH: typing.Final[int] = 6
DEFAULT_CATBOOST_LEARNING_RATE: typing.Final[float] = 0.1

MODEL_PIVOT_TABLE_PATH: typing.Final[str] = "models/pivot.json"
MODEL_SAVING_PATH: typing.Final[str] = "models/{}.cbm"
