import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from typing import List, Any, Protocol, Dict

class RemoveSeasonality:
    def __init__(self, periods: Dict[str, int]):
        self.periods = periods
        self.seasonal_components = {}  # Dictionary to store seasonal components

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Removes seasonality from specified columns of a DataFrame. - make sure the dataframe is sorted beforehand"""
        for col, period in self.periods.items():
            if period < 2:
                raise ValueError(f"Period must be a positive integer >= 2, got {period} for column '{col}'")

            stl = STL(data[col], period=period, robust=True)
            result = stl.fit()
            seasonal_component = result.seasonal
            data[col] = data[col] - seasonal_component

            # Store the seasonal components for each column
            if col not in self.seasonal_components:
                self.seasonal_components[col] = seasonal_component

        return data

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Re-adds the seasonal components to de-seasonalized data."""
        for col, seasonal_component in self.seasonal_components.items():
            data[col] = data[col] + seasonal_component

        return data
