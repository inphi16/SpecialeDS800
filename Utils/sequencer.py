import pandas as pd
import numpy as np
from typing import List

class RollingWindow:
    def __init__(self, seq_number: int, time_col: str, value_cols: List[str]):
        self.seq_number = seq_number
        self.time_col = time_col
        self.value_cols = value_cols

    def sequence(self, data: pd.DataFrame) -> np.array:
        """Extracts sequences from the DataFrame using a rolling window approach."""
        sequences = []
        data_sorted = data.sort_values(by=self.time_col)  # Ensure data is sorted by time column
        for i in range(len(data_sorted) - self.seq_number + 1):
            sequence = data_sorted[self.value_cols].iloc[i:i + self.seq_number].values
            sequences.append(sequence)
        return np.array(sequences)
