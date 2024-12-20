from typing import Protocol

import pandas as pd


class Displayable(Protocol):
    name: str

    def display(self, df: pd.DataFrame) -> None: ...
