import mlrun


def create_and_set_project(
    name: str = "huggingface",
    git_source: str = "git://github.com/davesh0812/mlrun-huggingface-demo.git#main",
    default_image: str = "davesh0812/mlrun:huggingface-mlrun-demo",
    user_project=False,
    set_serving=True,
):
    # get/create a project and register the data prep and trainer function in it
    project = mlrun.get_or_create_project(
        name=name, context="./", user_project=user_project
    )
    project.set_default_image(default_image)

    project.set_source(git_source, pull_at_runtime=True)

    project.set_function(
        "src/data_prep.py",
        name="data-prep",
        handler="prepare_dataset",
        kind="job",
    )
    project.set_function(
        "hub://hugging_face_classifier_trainer",
        name="hugging_face_classifier_trainer",
        kind="job",
    )
    project.set_function(
        "src/serving_test.py",
        name="server-tester",
        handler="model_server_tester",
        kind="job",
    )

    if set_serving:
        serving_function = mlrun.new_function(
            "serving-pretrained",
            kind="serving",
        )
        project.set_function(serving_function, with_repo=True)

        serving_function_staging = mlrun.new_function(
            "serving-trained-onnx",
            kind="serving",
        )
        project.set_function(serving_function_staging, with_repo=True)

    project.set_workflow("training_workflow", "src/training_workflow.py")
    project.save()

    return project
