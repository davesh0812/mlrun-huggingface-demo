import mlrun


def create_and_set_project(
    name: str = "huggingface",
    git_source: str = "git://github.com/davesh0812/mlrun-huggingface-demo.git#main",
    default_image: str = "mlrun/ml-models",
    requirements: str = ['transformers', 'datasets'],
    user_project=False,
    set_serving=True
):
    # get/create a project and register the data prep and trainer function in it
    project = mlrun.get_or_create_project(name=name, context="./", user_project=user_project)

    project.set_source(git_source, pull_at_runtime=True)

    # project.set_function(
    #     name="data-prep",
    #     image=default_image,
    #     handler="src.data_prep.prepare_dataset",
    #     kind="job",
    #     with_repo=True,
    # )
    project.set_function(
        name="trainer",
        image=default_image,
        handler="src.trainer.train",
        kind="job",
        with_repo=True,
        requirements=requirements
    )
    project.set_function(
        name="optimizer",
        image=default_image,
        handler="src.trainer.optimize",
        kind="job",
        with_repo=True,
        requirements=requirements
    )
    project.set_function(
        name="server-tester",
        image="davesh0812/mlrun:mlrun-huggingface-demo-1",
        handler="src.serving_test.model_server_tester",
        kind="job",
        with_repo=True,
        requirements=requirements
    )

    if set_serving:
        serving_function = mlrun.new_function("serving-pretrained", kind="serving", image="mlrun/ml-models",
                                              requirements=requirements)
        project.set_function(serving_function)

        serving_function_staging = mlrun.code_to_function(
            filename="src/serving.py",
            name="serving-trained",
            tag="staging",
            kind="serving",
            image=default_image,
            requirements=requirements
        )
        project.set_function(serving_function_staging)
    
    project.set_workflow("training_workflow", "src/training_workflow.py")
    project.save()

    return project
