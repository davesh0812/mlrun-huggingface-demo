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


from typing import Dict, List, Union

LABELS = {0: "NEGATIVE", 1: "POSITIVE"}


def preprocess(text: Union[str, Dict]) -> Dict:
    """
    Converting a simple text into a structured body for the serving function

    :param text: The text to predict
    """
    return {"inputs": [str(text)]}


def postprocess(model_response: Dict) -> List:
    """
    Transfering the prediction to the gradio interface.

    :param model_response: A dict with the model output
    """
    output = model_response["outputs"][0]
    prediction = LABELS[int(output["label"])]
    return [
        "The sentiment is " + prediction,
        "The prediction score is " + str(output["score"]),
    ]
