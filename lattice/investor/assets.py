from dataclasses import dataclass


@dataclass
class Asset:
    long_amt: float
    short_amt: float