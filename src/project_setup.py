import mlrun


def create_and_set_project(
    name: str = "huggingface",
    git_source: str = "git://github.com/davesh0812/mlrun-huggingface-demo.git#main",
    default_image: str = "mlrun/ml-models",
    requirements: str = ["transformers", "datasets", "onnxruntime"],
    user_project=False,
    set_serving=True,
):
    # get/create a project and register the data prep and trainer function in it
    project = mlrun.get_or_create_project(
        name=name, context="./", user_project=user_project
    )

    project.set_source(git_source, pull_at_runtime=True)

    project.set_function(
        "src/data_prep.py",
        name="data-prep",
        image=default_image,
        handler="prepare_dataset",
        kind="job",
    )
    project.set_function(
        "src/function.yaml",
        name="trainer",
        image=default_image,
        kind="job",
        requirements=requirements,
    )
    project.set_function(
        "src/function.yaml",
        name="optimizer",
        image=default_image,
        handler="optimize",
        kind="job",
        requirements=requirements,
    )
    project.set_function(
        "src/serving_test.py",
        name="server-tester",
        image=default_image,
        handler="model_server_tester",
        kind="job",
        requirements=requirements,
    )

    if set_serving:
        serving_function = mlrun.new_function(
            "serving-pretrained",
            kind="serving",
            image="mlrun/ml-models",
            requirements=requirements,
        )
        project.set_function(serving_function)

        serving_function_staging = mlrun.code_to_function(
            filename="src/serving.py",
            name="serving-trained",
            tag="staging",
            kind="serving",
            image=default_image,
            requirements=requirements,
        )
        project.set_function(serving_function_staging, with_repo=True)

    project.set_workflow("training_workflow", "src/training_workflow.py")
    project.save()

    return project
