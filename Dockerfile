FROM mlrun/ml-models:mlrun-rc31
RUN pip install transformers~=4.26.0 onnx~=1.10.1 onnxruntime~=1.11.1 optimum~=1.6.4 datasets~=2.10.1 scikit-learn~=1.0.2