from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import zipfile
import tempfile

from mlrun.frameworks.huggingface import apply_mlrun
from datasets import Dataset, load_dataset, load_metric
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig
from transformers import (
    PreTrainedTokenizer,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import mlrun


def _get_model_dir(model_uri: str):
    model_file, model_artifact, extra_data = mlrun.artifacts.get_model(model_uri)
    model_dir = tempfile.gettempdir()
    # Unzip the Model:
    with zipfile.ZipFile(model_file, "r") as zip_file:
        zip_file.extractall(model_dir)
    
    # Unzip the Tokenizer:
    tokenizer_file = extra_data['tokenizer'].local()
    with zipfile.ZipFile(tokenizer_file, "r") as zip_file:
        zip_file.extractall(model_dir)
        
    return model_dir, model_artifact.extra_data['tokenizer']
    

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


def _compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}


@mlrun.handler(
    outputs=["train_dataset:dataset", "test_dataset:dataset"]
)
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
    small_train_dataset = (
        dataset["train"].shuffle(seed=42).select(list(range(3000)))
    )
    small_train_dataset = _edit_columns(small_train_dataset, drop_columns, rename_columns)
    small_test_dataset = (
        dataset["test"].shuffle(seed=42).select(list(range(300)))
    )
    small_test_dataset = _edit_columns(small_test_dataset, drop_columns, rename_columns)
    
    return small_train_dataset.to_pandas(), small_test_dataset.to_pandas()


@mlrun.handler()
def train(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    pretrained_tokenizer: str = "distilbert-base-uncased",
    pretrained_model: str = "distilbert-base-uncased",
    num_labels: Optional[int] = 2,
    target_dir: str = "finetuning-sentiment-model-3000-samples",
):
    """
    Training and evaluating a pretrained model with a pretrained tokenizer over a dataset.

    :param train_dataset:           The data to train the model on.
    :param test_dataset:            The data to evaluate the model on every epoch.
    :param pretrained_tokenizer:    The name of the pretrained tokenizer from the HuggingFace hub.
    :param pretrained_model:   The name of the pretrained model from the HuggingFace hub.
    :param num_labels:              The number of target labels of the task.
    :param target_dir:              The directory name to save the checkpoint.

    """
    # Creating tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    
    # Convert pd.DataFrame to datasets.Dataset:
    train_dataset = Dataset.from_pandas(train_dataset)
    test_dataset = Dataset.from_pandas(test_dataset)
    
    # Mapping datasets with the tokenizer:
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    # Creating data collator for batching:
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Loading our pretrained model:
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model, num_labels=num_labels
    )

    # Preparing training arguments:
    training_args = TrainingArguments(
        output_dir=target_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        push_to_hub=False,
        evaluation_strategy="epoch",
        eval_steps=1,
        logging_steps=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
    )

    apply_mlrun(trainer, model_name='trained_model')

    # Apply training with evaluation:
    train_output = trainer.train()


def optimize(
    model_path: str,
    target_dir: str = "./optimized"
):
    """
    Optimizing the transformer model using ONNX optimization.
    :param model_path: The path of the model to optimize.
    :param target_dir: The directory to save the ONNX model.
    """
    model_dir, tokenizer = _get_model_dir(model_uri=model_path)
    # Creating configuration for optimization step:
    optimization_config = OptimizationConfig(optimization_level=1)

    # Converting our pretrained model to an ONNX-Runtime model:
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        model_dir, from_transformers=True
    )
    
    # Creating an ONNX-Runtime optimizer from ONNX model:
    optimizer = ORTOptimizer.from_pretrained(ort_model)
    
    apply_mlrun(optimizer, model_name='optimized_model', extra_data={"tokenizer": tokenizer})
    # Optimizing and saving the ONNX model:
    optimizer.optimize(save_dir=target_dir, optimization_config=optimization_config)

