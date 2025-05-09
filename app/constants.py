import typing


TIMESTAMP_COL_NAME: typing.Final[str] = "timestamp"
THREADID_COL_NAME: typing.Final[str] = "threadid"
EVENT_TYPE_COL_NAME: typing.Final[str] = "event_type"

EVENT_CHAIN_ID_CON_NAME: typing.Final[str] = "event_chain_id"

DEFAULT_WINDOW_DAYS: typing.Final[int] = 14
DEFAULT_DATA_COUNT: typing.Final[int] = 50_000

DEFAULT_CHUNKSIZE: typing.Final[int] = 50
