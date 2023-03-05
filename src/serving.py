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
import tempfile
import zipfile
from typing import Any, Dict, List, Tuple, Union

import mlrun
import numpy as np
import transformers
from mlrun.frameworks.huggingface import HuggingFaceModelServer
from transformers import AutoTokenizer

LABELS_OPTIMIZE = {0: "NEGATIVE", 1: "POSITIVE"}

LABELS = {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE"}


def preprocess(text: Union[str, bytes]) -> Dict:
    """Converting a simple text into a structured body for the serving function

    :param text: The text to predict
    """
    return {"inputs": [str(text)]}


def postprocess(model_response: Dict) -> List:
    """Transfering the prediction to the gradio interface.

    :param model_response: A dict with the model output
    """

    outputs = model_response["outputs"][0]
    if hasattr(outputs, "tolist"):
        outputs = outputs.tolist()
        chosen_label = np.argmax(outputs, axis=-1)[0].item()
        score = outputs[0][chosen_label]
    elif isinstance(outputs, dict):
        chosen_label = outputs["label"]
        score = outputs["score"]
    else:
        raise mlrun.errors.MLRunRuntimeError(
            f"Got unknown model_response with {type(model_response)} type."
        )

    prediction = LABELS.get(chosen_label, None) or LABELS_OPTIMIZE.get(
        chosen_label, None
    )
    return [
        "The sentiment is " + prediction,
        "The prediction score is " + str(score),
    ]


class HuggingFaceTokenizerModelServer(HuggingFaceModelServer):
    def load(self):
        # Loading the pretrained tokenizer:
        if self.tokenizer_class:
            tokenizer_object = getattr(transformers, self.tokenizer_class)
            self._tokenizer = tokenizer_object.from_pretrained(self.tokenizer_name)
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

    def predict(self, request: Dict[str, Any]) -> str:
        print(request)
        tokenized_samples: Dict = self._tokenizer(request["inputs"], truncation=True)
        request["inputs"] = [
            val if isinstance(val[0], list) else [val]
            for val in tokenized_samples.values()
        ]
        print(request)
        return request

    def postprocess(self, request: Dict) -> Dict:
        print(f"postprocess : {request}")
        request["inputs"] = request["outputs"]["inputs"]
        return request
