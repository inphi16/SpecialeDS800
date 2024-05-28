import pandas as pd
import numpy as np
from typing import List, Any, Protocol, Dict

import os
import sys
# Define the path one level up
parent_directory = os.path.join(os.getcwd(), '../../Workspace/Users/iaaph@energinet.dk/utils')
# Add this path to the sys.path list
sys.path.append(f"{parent_directory}")
import normalizer
import sequencer
import timeseriestransformer


class Normalizer(Protocol):
    value_col: List[str]
    # scaler: MinMaxScaler

    def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalizes all columns in value_col.
        """
        ...

    def denormalize(self, data) -> Any:
        """Denormalizes input using scaler"""
        ...
    


class Sequencer(Protocol):
    time_col: str

    def sequence(self, data: pd.DataFrame) -> np.array:
        """Returns specified sequences on time_col in the form of numpy arrays"""
        ...


class TimeSeriesTransformer(Protocol): 
    '''Generic time series processor. An example could be the removal of seasonality'''
    def transform(self):
        ...

    def inverse_transform(self):
        ...


class Preprocessor:

    def __init__(self, data, normalizer: Normalizer, sequencer: Sequencer, transformers: List[TimeSeriesTransformer] = []) -> None:
        self.data = data
        self.normalizer = normalizer
        self.sequencer = sequencer
        self.transformers = transformers

    def __sort(self) -> None:
        """Sort by sequencer.time_col """
        return self.data.sort_values(by=self.sequencer.time_col).reset_index(drop=True)
    
    def preprocess(self) -> np.ndarray:
        self.__sort()

        self.data = self.normalizer.normalize(self.data)

        for transformer in self.transformers:
            self.data = transformer.transform(self.data)
        
        self.data = self.sequencer.sequence(self.data)

        return self.data
