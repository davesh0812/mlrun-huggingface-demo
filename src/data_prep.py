from typing import Dict, List, Optional

import mlrun
from datasets import Dataset, load_dataset


def _edit_columns(
    dataset: Dataset,
    drop_columns: List[str] = None,
    rename_columns: Dict[str, str] = None,
):
    if drop_columns:
        dataset = dataset.remove_columns(drop_columns)
    if rename_columns:
        dataset = dataset.rename_columns(rename_columns)
    return dataset


@mlrun.handler(outputs=["train_dataset:dataset", "test_dataset:dataset"])
def prepare_dataset(
    dataset_name: str = "Shayanvsf/US_Airline_Sentiment",
    drop_columns: Optional[List[str]] = [
        "airline_sentiment_confidence",
        "negativereason_confidence",
    ],
    rename_columns: Optional[Dict[str, str]] = {"airline_sentiment": "labels"},
):
    """
    Loading the dataset and editing the columns and logs the
    :param dataset_name:    The name of the dataset to get from the HuggingFace hub
    :param drop_columns:    The columns to drop from the dataset.
    :param rename_columns:  The columns to rename in the dataset.

    """

    # Loading and editing dataset:
    dataset = load_dataset(dataset_name)
    small_train_dataset = dataset["train"].shuffle(seed=42).select(list(range(3000)))
    small_train_dataset = _edit_columns(
        small_train_dataset, drop_columns, rename_columns
    )
    small_test_dataset = dataset["test"].shuffle(seed=42).select(list(range(300)))
    small_test_dataset = _edit_columns(small_test_dataset, drop_columns, rename_columns)

    return small_train_dataset.to_pandas(), small_test_dataset.to_pandas()
