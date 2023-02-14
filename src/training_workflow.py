import mlrun
from kfp import dsl


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
        function="data-prep",
        name="prepare_data",
        params={"dataset_name": dataset_name},
        outputs=["train_dataset", "test_dataset"],
    )

    # Training:
    training_run = mlrun.run_function(
        function="trainer",
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
        function="optimizer",
        name="optimization",
        params={"model_path": training_run.outputs["model"]},
        outputs=["model"],
    )

    # Get the function:
    serving_function = project.get_function("serving")
    graph = serving_function.spec.graph

    # Build the serving graph:
    graph.to(handler="preprocess_optimize", name="preprocess").to(
        class_name="ONNXModelServer",
        name="sentiment-analysis",
        model_path=optimization_run.outputs["model"],
    ).to(handler="postprocess_optimize", name="postprocess").respond()

    # Deploy the serving function:
    deploy_return = mlrun.deploy_function("serving")

    # Model server tester
    mlrun.run_function(
        function="server_tester",
        name="server_tester",
        inputs={"dataset": prepare_dataset_run.outputs["test_dataset"]},
        params={
            "label_column": "labels",
            "endpoint": deploy_return.outputs["endpoint"],
        },
        auto_build=True,
    ).after(deploy_return)
