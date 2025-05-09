import typing


TIMESTAMP_COL_NAME: typing.Final[str] = "timestamp"
THREADID_COL_NAME: typing.Final[str] = "threadid"
EVENT_TYPE_COL_NAME: typing.Final[str] = "event_type"

EVENT_CHAIN_ID_CON_NAME: typing.Final[str] = "event_chain_id"

DEFAULT_WINDOW_DAYS: typing.Final[int] = 14
DEFAULT_DATA_COUNT: typing.Final[int] = 50_000

DEFAULT_CHUNKSIZE: typing.Final[int] = 50

DEFAULT_VALIDATION_SIZE: typing.Final[float] = 0.2
DEFAULT_CATBOOST_ITERATIONS: typing.Final[int] = 300
DEFAULT_CATBOOST_DEPTH: typing.Final[int] = 6
DEFAULT_CATBOOST_LEARNING_RATE: typing.Final[float] = 0.1
