from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import sklearn
from pandas import Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def load_lr_model() -> BaseEstimator:

    lr = LinearRegression()

    return lr


def train_model(
    model: BaseEstimator,
    X_train: csr_matrix,
    y_train: Series,
    X_val: Optional[csr_matrix] = None,
    eval_metric: Callable = mean_squared_error,
    fit_params: Optional[Dict] = None,
    y_val: Optional[Series] = None,
    **kwargs,
) -> Tuple[BaseEstimator, Optional[Dict], Optional[np.ndarray]]:
    model.fit(X_train, y_train, **(fit_params or {}))

    metrics = None
    y_pred = None
    if X_val is not None and y_val is not None:
        y_pred = model.predict(X_val)

        rmse = eval_metric(y_val, y_pred, squared=False)
        mse = eval_metric(y_val, y_pred, squared=True)
        metrics = dict(mse=mse, rmse=rmse)

    return model, metrics, y_pred