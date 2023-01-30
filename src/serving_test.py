from datetime import datetime

import mlrun
import numpy as np
import pandas as pd
import plotly.express as px
import requests


@mlrun.handler(
    outputs=[
        "count",
        "error_count",
        "avg_latency",
        "min_latency",
        "max_latency",
        "latency_chart:plot",
    ]
)
def model_server_tester(
    context: mlrun.MLClientCtx,
    dataset: pd.DataFrame,
    endpoint: str,
    label_column: str,
    rows: int = 100,
    max_error: int = 5,
):
    """Test a model server
    :param context:       mlrun context
    :param endpoint:
    :param dataset:       csv/parquet table with test data
    :param label_column:  name of the label column in table
    :param rows:          number of rows to use from test set
    :param max_error:     maximum error for
    """

    if rows and rows < dataset.shape[0]:
        dataset = dataset.sample(rows)
    y_list = dataset.pop(label_column).values.tolist()
    count = err_count = 0
    times = []
    print(endpoint)
    for i, y in zip(range(dataset.shape[0]), y_list):
        if err_count == max_error:
            raise ValueError(f"reached error max limit = {max_error}")
        count += 1
        event_data = dataset.iloc[i].to_dict()['text']
        try:
            start = datetime.now()
            resp = requests.post(f"{endpoint}/predict", json=event_data)
            if not resp.ok:
                context.logger.error(f"bad function resp!!\n{resp.text}")
                err_count += 1
                continue
            times.append((datetime.now() - start).microseconds)

        except OSError as err:
            context.logger.error(f"error in request, data:{event_data}, error: {err}")
            err_count += 1
            continue

    times_arr = np.array(times)
    latency_chart = px.line(
        x=range(2, len(times) + 1),
        y=times_arr[1:],
        title="<i><b>Latency (microsec) X  Invokes</b></i>",
        labels={"y": "latency (microsec)", "x": "invoke number"},
    )

    return (
        count,
        err_count,
        int(np.mean(times_arr)),
        int(np.amin(times_arr)),
        int(np.amax(times_arr)),
        latency_chart,
    )
