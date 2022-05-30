import datetime


def to_timestamp(iso_time: str):
    return datetime.datetime.fromisoformat(iso_time).timestamp() * 1000
