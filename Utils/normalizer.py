import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List, Any, Protocol, Dict

class Scaling:
    """
    This class provides methods for normalizing and denormalizing data using MinMaxScaler.
    __init__: Initializes the Scaling object with a list of columns to be scaled.
    normalize: Normalizes the specified columns in a DataFrame using MinMaxScaler.
    denormalize: Denormalizes the specified columns in a DataFrame using the inverse transformation of MinMaxScaler."""

    def __init__(self, value_cols: List[str]):
        self.value_cols = value_cols
        self.scaler = MinMaxScaler()

    def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalizes columns specified in value_cols."""
        normalized_data = data.copy()
        normalized_data[self.value_cols] = self.scaler.fit_transform(data[self.value_cols])
        return normalized_data

    def denormalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Denormalizes the specified columns using the scaler."""
        denormalized_data = data.copy()
        denormalized_data[self.value_cols] = self.scaler.inverse_transform(data[self.value_cols])
        return denormalized_data
    
    



