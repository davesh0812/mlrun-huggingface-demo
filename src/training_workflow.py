import mlrun
from kfp import dsl


@dsl.pipeline(name="Sentiment Analysis Pipeline")
def kfpipeline(
    dataset_name: str,
    pretrained_tokenizer: str,
    pretrained_model: str,
    num_epochs: int,
):
    # Get our project object:
    project = mlrun.get_current_project()

    # Dataset Preparation:
    prepare_dataset_run = mlrun.run_function(
        function="data-prep",
        params={"dataset_name": dataset_name},
        outputs=["train_dataset", "test_dataset"],
    )

    # Training:
    training_run = mlrun.run_function(
        function="trainer",
        inputs={
            "train_dataset": prepare_dataset_run.outputs["train_dataset"],
            "test_dataset": prepare_dataset_run.outputs["test_dataset"],
        },
        params={
            "pretrained_tokenizer": pretrained_tokenizer,
            "pretrained_model": pretrained_model,
            "num_epochs": num_epochs,
        },
        outputs=["model"],
    )

    # Optimization:
    optimization_run = mlrun.run_function(
        function="optimizer",
        params={"model_path": training_run.outputs["model"]},
        outputs=["model"],
    )

    # Get the function:
    serving_function = project.get_function("serving")
    serving_function.metadata.tag = "staging"
    serving_function.spec.graph["sentiment-analysis"].class_name = "ONNXModelServer"
    serving_function.spec.graph["sentiment-analysis"].class_args = {
        "model_path": str(optimization_run.outputs["model"])
    }

    # Enable model monitoring
    serving_function.set_tracking()

    # Deploy the serving function:
    deploy_return = mlrun.deploy_function("serving")

    # Model server tester
    mlrun.run_function(
        function="server-tester",
        inputs={"dataset": prepare_dataset_run.outputs["test_dataset"]},
        params={
            "label_column": "labels",
            "endpoint": deploy_return.outputs["endpoint"],
        },
    )
