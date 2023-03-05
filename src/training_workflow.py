import mlrun
from kfp import dsl


@dsl.pipeline(name="Sentiment Analysis Pipeline")
def kfpipeline(
    dataset_name: str,
    pretrained_tokenizer: str,
    pretrained_model: str,
    TRAIN_output_dir: str,
    TRAIN_evaluation_strategy: str,
    CLASS_num_labels: str,
    TRAIN_num_train_epochs: str,
):
    # Get our project object:
    project = mlrun.get_current_project()

    # Dataset Preparation:
    prepare_dataset_run = mlrun.run_function(
        function="data-prep",
        params={"dataset_name": dataset_name},
        outputs=["train_dataset", "test_dataset", "additional_params"],
    )

    # Training:
    training_run = mlrun.run_function(
        function="hugging_face_classifier_trainer",
        name="trainer",
        inputs={
            "dataset": prepare_dataset_run.outputs["train_dataset"],
            "test_set": prepare_dataset_run.outputs["test_dataset"],
        },
        params={
            "pretrained_tokenizer": pretrained_tokenizer,
            "pretrained_model": pretrained_model,
            "model_class": "transformers.AutoModelForSequenceClassification",
            "label_name": "airline_sentiment",
            "metrics": ["accuracy", "f1"],
            "random_state": 42,
            "TRAIN_output_dir": TRAIN_output_dir,
            "TRAIN_evaluation_strategy": TRAIN_evaluation_strategy,
            "CLASS_num_labels": CLASS_num_labels,
            "TRAIN_num_train_epochs": TRAIN_num_train_epochs,
        },
        handler="train",
        outputs=["model"],
    )

    # Optimization:
    optimization_run = mlrun.run_function(
        function="hugging_face_classifier_trainer",
        name="optimize",
        params={"model_path": training_run.outputs["model"]},
        outputs=["model"],
        handler="optimize",
    )

    # Create serving graph:
    serving_function = project.get_function("serving-trained-onnx")

    # Set the topology and get the graph object:
    graph = serving_function.set_topology("flow", engine="async")
    graph.to(handler="src.serving.preprocess", name="preprocess").to(
        "src.serving.HuggingFaceTokenizerModelServer",
        name="tokenizer",
        task="tokenizer",
        tokenizer_name="distilbert-base-uncased",
        tokenizer_class="AutoTokenizer",
    ).to(
        class_name="mlrun.frameworks.onnx.ONNXModelServer",
        name="sentiment-analysis",
        model_path=str(optimization_run.outputs["model"]),
    ).to(
        handler="src.serving.postprocess", name="postprocess"
    ).respond()

    project.set_function(serving_function, with_repo=True)

    # Enable model monitoring
    serving_function.set_tracking()

    # Deploy the serving function:
    deploy_return = mlrun.deploy_function("serving-trained-onnx")

    # Model server tester
    mlrun.run_function(
        function="server-tester",
        inputs={"dataset": prepare_dataset_run.outputs["test_dataset"]},
        params={
            "label_column": "labels",
            "endpoint": deploy_return.outputs["endpoint"],
        },
    )
