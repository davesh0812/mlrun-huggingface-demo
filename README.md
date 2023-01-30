# NYC Taxi Tutorial

This project demonstrates how to build an ML application and use MLOps to operationalize it.

- [**Notebooks and code**](#notebooks)
- [**Installation (local, GitHub codespaces, Sagemaker)**](#installation)

<a id="notebooks"></a>
## Notebooks and code 

The project contains four notebooks, in the following order:

- [**Exploratory Data Analysis**](./00-exploratory-data-analysis.ipynb)
- [**Data preparation, training and evaluating a model**](./01-dataprep-train-test.ipynb)
- [**Application Serving Pipeline**](./02-serving-pipeline.ipynb)
- [**Pipeline Automation and Model Monitoring**](./03-automation.ipynb)

You can find the python source code under [/src](./src)

<a id="installation"></a>
## Installation

This project can run in different development environments:
1. Local computer (using PyCharm, VSCode, Jupyter, etc.)
2. Inside GitHub Codespaces 
3. Sagemaker studio and Studio Labs (free edition) or other managed Jupyter environments

### Install the code and mlrun client 

To get started, fork this repo into your GitHub account and clone it into your development environment.

To install the package dependencies (not required in GitHub codespaces) use:
 
    make install-requirements
    
If you prefer to use Conda or work in **Sagemaker** use this instead (to create and configure a conda env):

    make conda-env

> Make sure you open the notebooks and select the `mlrun` conda environment 
 
### Install or connect to MLRun service/cluster

The MLRun service and computation can run locally (minimal setup) or over a remote Kubernetes environment.

If your development environment support docker and have enough CPU resources run:

    make mlrun-docker
    
> MLRun UI can be viewed in: http://localhost:8060
    
If your environment is minimal or you are in Sagemaker run mlrun as a process (no UI):

    [conda activate mlrun &&] make mlrun-api
 
For MLRun to run properly you should set your client environment, this is not required when using **codespaces**, the mlrun **conda** environment, or **iguazio** managed notebooks.

Your environment should include `MLRUN_ENV_FILE=<absolute path to the ./mlrun.env file> ` (point to the mlrun .env file in this repo), see [mlrun client setup](https://docs.mlrun.org/en/latest/install/remote.html) instructions for details.  
     
> Note: You can also use a remote MLRun service (over Kubernetes), instead of starting a local mlrun, 
> edit the [mlrun.env](./mlrun.env) and specify its address and credentials  
