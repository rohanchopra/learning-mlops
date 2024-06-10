import pandas as pd
import requests
import logging

from io import BytesIO
from typing import List

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader


@data_loader
def load_data(*args, **kwargs) -> pd.DataFrame:
    """
    Template code for loading data from any source.

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    dfs: List[pd.DataFrame] = []

    for year in [2023]:
        for month in [3]:
            url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
            response = requests.get(url)

            if response.status_code != 200:
                raise Exception(response.text)

            df = pd.read_parquet(BytesIO(response.content))
            dfs.append(df)

    return pd.concat(dfs)
