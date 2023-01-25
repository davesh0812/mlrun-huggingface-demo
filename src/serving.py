from typing import Dict, List, Union

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
    output = model_response["outputs"][0]
    prediction = LABELS[output["label"]]
    return [
        "The sentiment is " + prediction,
        "The prediction score is " + str(output["score"]),
    ]
