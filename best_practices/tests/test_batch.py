import pandas as pd
from datetime import datetime
import batch


def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)


def test_prepare_data():
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2), dt(1, 10)),
        (1, 2, dt(2, 2), dt(2, 3)),
        (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    processed_data = batch.prepare_data(df, ['PULocationID', 'DOLocationID'])
    expected_data = [
        {'PULocationID': '-1', 'DOLocationID': '-1', 'tpep_pickup_datetime': dt(1, 2), 'tpep_dropoff_datetime': dt(1, 10), 'duration': 8.0},
        {'PULocationID': '1', 'DOLocationID': '-1', 'tpep_pickup_datetime': dt(1, 2), 'tpep_dropoff_datetime': dt(1, 10), 'duration': 8.0},
        {'PULocationID': '1', 'DOLocationID': '2', 'tpep_pickup_datetime': dt(2, 2), 'tpep_dropoff_datetime': dt(2, 3), 'duration': 1.0}
    ]

    assert processed_data.to_dict(orient='records') == expected_data