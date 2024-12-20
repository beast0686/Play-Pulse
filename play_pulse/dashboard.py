import streamlit as st
from attrs import Factory, define
import pandas as pd

from .types import Displayable


@define
class Dashboard:
    name: str

    components: dict[str, Displayable] = Factory(dict[str, Displayable])

    def display(self, df: pd.DataFrame) -> None:
        """Render the Streamlit dashboard."""
        st.title(self.name)

        selection = st.sidebar.selectbox(
            f"{self.name.title()} components", list(self.components.keys())
        )

        component = self.components[selection]
        component.display(df)

    def add_component(self, component: Displayable) -> None:
        self.components[component.name] = component
