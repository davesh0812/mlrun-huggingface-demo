# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import tempfile
import zipfile
from typing import Any, Dict, List, Tuple, Union

import mlrun
import numpy as np
import onnxruntime
import transformers
from mlrun.serving.v2_serving import V2ModelServer
from mlrun.frameworks.huggingface import HuggingFaceModelServer
from transformers import AutoTokenizer

LABELS_OPTIMIZE = {0: "NEGATIVE", 1: "POSITIVE"}

LABELS = {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE"}


def preprocess(text: Union[str, bytes]) -> Dict:
    """Converting a simple text into a structured body for the serving function

    :param text: The text to predict
    """
    print(text)
    return {"inputs": [str(text)]}


def postprocess(model_response: Dict) -> List:
    """Transfering the prediction to the gradio interface.

    :param model_response: A dict with the model output
    """

    outputs = model_response["outputs"][0].tolist()
    print(outputs)
    chosen_label = np.argmax(outputs, axis=-1)[0].item()
    score = outputs[0][chosen_label]

    prediction = LABELS.get(chosen_label, None) or LABELS_OPTIMIZE.get(
        chosen_label, None
    )
    return [
        "The sentiment is " + prediction,
        "The prediction score is " + str(score),
    ]


def _get_model_dir(model_uri: str):
    model_file, model_artifact, extra_data = mlrun.artifacts.get_model(model_uri)
    model_dir = tempfile.gettempdir()
    # Unzip the Model:
    with zipfile.ZipFile(model_file, "r") as zip_file:
        zip_file.extractall(model_dir)

    # Unzip the Tokenizer:
    tokenizer_file = extra_data["tokenizer"].local()
    with zipfile.ZipFile(tokenizer_file, "r") as zip_file:
        zip_file.extractall(model_dir)

    return model_dir, model_artifact.extra_data["tokenizer"]


class HuggingFaceTokenizerModelServer(HuggingFaceModelServer):
    def load(self):
        # Loading the pretrained tokenizer:
        if self.tokenizer_class:
            tokenizer_object = getattr(transformers, self.tokenizer_class)
            self._tokenizer = tokenizer_object.from_pretrained(self.tokenizer_name)
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        print(request)
        tokenized_samples: Dict = self._tokenizer(request["inputs"], truncation=True)
        request["inputs"] = [
            val if isinstance(val[0], list) else [val]
            for val in tokenized_samples.values()
        ]
        print(request)
        return request
#
# class ONNXModelServer(V2ModelServer):
#     """
#     ONNX Model serving class, inheriting the V2ModelServer class for being initialized automatically by the model server
#     and be able to run locally as part of a nuclio serverless function, or as part of a real-time pipeline.
#     """
#
#     def __init__(
#         self,
#         context: mlrun.MLClientCtx,
#         name: str,
#         model=None,
#         model_path: str = None,
#         model_name: str = None,
#         execution_providers: List[Union[str, Tuple[str, Dict[str, Any]]]] = None,
#         protocol: str = None,
#         **class_args,
#     ):
#         """
#         Initialize a serving class for an onnx.ModelProto model.
#
#         :param context:             The mlrun context to work with.
#         :param name:                The model name to be served.
#         :param model:               Model to handle or None in case a loading parameters were supplied.
#         :param model_path:          Path to the model's directory to load it from. The onnx file must start with the
#                                     given model name and the directory must contain the onnx file. The model path can be
#                                     also passed as a model object path in the following format:
#                                     'store://models/<PROJECT_NAME>/<MODEL_NAME>:<VERSION>'.
#         :param model_name:          The model name for saving and logging the model:
#                                     * Mandatory for loading the model from a local path.
#                                     * If given a logged model (store model path) it will be read from the artifact.
#                                     * If given a loaded model object and the model name is None, the name will be set to
#                                       the model's object name / class.
#         :param execution_providers: List of the execution providers. The first provider in the list will be the most
#                                     preferred. For example, a CUDA execution provider with configurations and a CPU
#                                     execution provider:
#                                     [
#                                         (
#                                             'CUDAExecutionProvider',
#                                             {
#                                                 'device_id': 0,
#                                                 'arena_extend_strategy': 'kNextPowerOfTwo',
#                                                 'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
#                                                 'cudnn_conv_algo_search': 'EXHAUSTIVE',
#                                                 'do_copy_in_default_stream': True,
#                                             }
#                                         ),
#                                         'CPUExecutionProvider'
#                                     ]
#                                     Defaulted to None - will prefer CUDA Execution Provider over CPU Execution Provider.
#         :param protocol:            -
#         :param class_args:          -
#         """
#         super(ONNXModelServer, self).__init__(
#             context=context,
#             name=name,
#             model_path=model_path,
#             model=model,
#             protocol=protocol,
#             **class_args,
#         )
#
#         # Set the execution providers (default will prefer CUDA Execution Provider over CPU Execution Provider):
#         self._execution_providers = (
#             ["CUDAExecutionProvider", "CPUExecutionProvider"]
#             if execution_providers is None
#             else execution_providers
#         )
#
#         # Prepare inference parameters:
#         self._inference_session = None  # type: onnxruntime.InferenceSession
#         self._input_layers = None  # type: List[str]
#         self._output_layers = None  # type: List[str]
#
#     def load(self):
#         """
#         Use the model handler to get the model file path and initialize an ONNX run time inference session.
#         """
#         model_dir = _get_model_dir(self.model_path)[0]
#         self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
#         # initialize the onnx run time session:
#         self._inference_session = onnxruntime.InferenceSession(
#             os.path.join(model_dir, "model_optimized.onnx"),
#             providers=self._execution_providers,
#         )
#
#         # Get the input layers names:
#         self._input_layers = [
#             input_layer.name for input_layer in self._inference_session.get_inputs()
#         ]
#
#         # Get the outputs layers names:
#         self._output_layers = [
#             output_layer.name for output_layer in self._inference_session.get_outputs()
#         ]
#
#     def predict(self, request: Dict[str, Any]) -> np.ndarray:
#         """
#         Infer the inputs through the model using ONNXRunTime and return its output. The inferred data will be
#         read from the "inputs" key of the request.
#
#         :param request: The request to the model. The input to the model will be read from the "inputs" key.
#
#         :return: The ONNXRunTime session returned output on the given inputs.
#         """
#         # Read the inputs from the request:
#         inputs = request["inputs"]
#
#         # Infer the inputs through the model:
#         return self._inference_session.run(
#             output_names=self._output_layers,
#             input_feed={
#                 input_layer: data
#                 for input_layer, data in zip(self._input_layers, inputs)
#             },
#         )
#
#     def explain(self, request: Dict[str, Any]) -> str:
#         """
#         Return a string explaining what model is being serve in this serving function and the function name.
#
#         :param request: A given request.
#
#         :return: Explanation string.
#         """
#         return f"The '{self.model.name}' model serving function named '{self.name}'"
#
#     def preprocess(self, request: Dict, operation) -> Dict:
#         """preprocess the event body before validate and action"""
#
#         tokenized_samples: Dict = self._tokenizer(request["inputs"], truncation=True)
#         request["inputs"] = [
#             val if isinstance(val[0], list) else [val]
#             for val in tokenized_samples.values()
#         ]
#         return request
#
#     def postprocess(self, request: Dict) -> Dict:
#         outputs = request["outputs"][0].tolist()
#         print(outputs)
#         chosen_label = np.argmax(outputs, axis=-1)[0].item()
#         p = outputs[0][chosen_label]
#         request["outputs"] = [{"label": chosen_label, "score": p}]
#         return request
