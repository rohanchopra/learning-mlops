import sys
import pickle
import pandas as pd
import numpy as np

# Use cli parameters for year and month
year = (int(sys.argv[1]) if len(sys.argv) > 1 else 2023)
month = (int(sys.argv[2]) if len(sys.argv) > 2 else 3)


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

print(np.mean(y_pred))

prediction = pd.DataFrame(y_pred)
prediction['ride_id'] = f'{year:04d}/{month:02d}_' + prediction.index.astype('str')

output_file = f"{year:04d}_{month:02d}_predictions.parquet"
prediction.to_parquet(output_file, engine='pyarrow',
                     compression=None, index=False)
