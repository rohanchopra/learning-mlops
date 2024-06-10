import mlflow
import pickle

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("nyc-taxi-hw-3")

@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    dv = data[-1]
    with mlflow.start_run():
        with open('lin_reg_dv.bin', 'wb') as f_out:
            pickle.dump(dv, f_out)

        mlflow.sklearn.log_model(data[0], "lr_model")
        mlflow.log_artifact(local_path="lin_reg_dv.bin", artifact_path="dv_pickle")

    return data[0].intercept_

