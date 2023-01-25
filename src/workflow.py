import mlrun
from kfp import dsl
from typing import List, Dict, Optional


@dsl.pipeline(name="Sentiment Analysis Pipeline")
def kfpipeline(
    dataset_name: str,
    pretrained_tokenizer: str,
    pretrained_model: str,
):
    # Get our project object:
    project = mlrun.get_current_project()
    
    # Dataset Preparation:
    prepare_dataset_run = mlrun.run_function(
        function="training",
        handler="prepare_dataset",
        name="prepare_data",
        params={"dataset_name": dataset_name},
        outputs=["train_dataset", "test_dataset"],
    )

    # Training:
    training_run = mlrun.run_function(
        function="training",
        handler="train",
        name="training",
        inputs={
            "train_dataset": prepare_dataset_run.outputs["train_dataset"],
            "test_dataset": prepare_dataset_run.outputs["test_dataset"],
        },
        params={
            "pretrained_tokenizer": pretrained_tokenizer,
            "pretrained_model": pretrained_model,
        },
        outputs=["model"],
    )
    
    # Optimization:
    optimization_run = mlrun.run_function(
        function="training",
        handler="optimize",
        name="optimization",
        params={"model_path": training_run.outputs["model"]},
        outputs=["model"],
    )

    # Get the function:
    serving_function = project.get_function("serving")
    graph = serving_function.set_topology("flow", engine="async")
    
    # Build the serving graph:
    graph.to(handler="preprocess", name="preprocess")\
         .to(class_name="ONNXModelServer", name="sentiment-analyzer", model_path=str(optimization_run.outputs["model"]))\
         .to(handler="postprocess", name="postprocess").respond()
    
    # Deploy the serving function:
    mlrun.deploy_function("serving")
