import mlrun


def create_and_set_project(
    name: str = "huggingface",
    git_source: str = "git://github.com/davesh0812/mlrun-huggingface-demo.git#main",
    user_project=False,
):
    # get/create a project and register the data prep and trainer function in it
    project = mlrun.get_or_create_project(name=name, context="./", user_project=user_project)

    project.set_source(git_source, pull_at_runtime=True)

    project.set_function(
        name="data-prep",
        image="davesh0812/mlrun:mlrun-huggingface-demo-1",
        handler="src.data_prep.prepare_dataset",
        kind="job",
        with_repo=True,
    )
    project.set_function(
        name="trainer",
        image="davesh0812/mlrun:mlrun-huggingface-demo-1",
        handler="src.trainer.train",
        kind="job",
        with_repo=True,
    )
    project.set_function(
        name="optimizer",
        image="davesh0812/mlrun:mlrun-huggingface-demo-1",
        handler="src.trainer.optimize",
        kind="job",
        with_repo=True,
    )
    project.set_function(
        name="server-tester",
        image="davesh0812/mlrun:mlrun-huggingface-demo-1",
        handler="src.serving_test.model_server_tester",
        kind="job",
        with_repo=True,
    )
    
    project.set_workflow("training_workflow", "src/training_workflow.py")
    project.save()

    return project
