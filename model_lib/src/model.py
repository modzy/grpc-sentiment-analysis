import json
import sys
import string
import numpy as np
from typing import Dict, List
from lime import lime_text

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

"""
The required output structure for a successful inference run for a models is the following JSON:

{
    "data": {
        "result": <inference-result>,
        "explanation": <explanation-data>,
        "drift": <drift-data>,
    }
}

The `data` key is required and stores a dictionary which represents the output for a specific input. The only top-level 
key within these dictionaries that is required is `result`, however, `explanation` and `drift` are additional keys that
may be included if your particular model supports drift detection or explainability. All three of these keys
(`result`, `explanation`, and `drift`) are required to have a particular format in order to provide platform support.
This format type must be specified in the model.yaml file for the version that you are releasing, and the structure for
this format type must be followed. If no formats are specified, it is possible to define your own custom structure on a
per-model basis.

The required output structure for a failed inference run for a models is the following JSON:

{
    "error_message": <error-message>
}

Here, all error information that you can extract can be loaded into a single string and returned. This could be a JSON
string with a structured error log, or a stack trace dumped to a string.

Specifications:
This section details the currently supported specifications for the "result", "explanation", and "drift" fields of each
successful output JSON. These correspond to specifications selected in the `resultsFormat`, `driftFormat`,
`explanationFormat` of the model.yaml file for the particular version of the model.

* `resultsFormat`:

1A) imageClassification

"result": {
    "classPredictions": [
        {"class": <class-1-label>, "score": <class-1-probability>},
        ...,
        {"class": <class-n-label>, "score": <class-n-probability>}
    ]
}

* `driftFormat`

2A) imageRLE

explanation: {
    "maskRLE": <rle-mask>
}

Here, the <rle-mask> is a fortran ordered run-length encoding.

* `explanationFormat`

3A) ResNet50

drift: {
    {
        "layer1": <layer-data>
        "layer2": <layer-data>
        "layer3": <layer-data>
        "layer4": <layer-data>
    }
}

"""


def get_success_json_structure(inference_result, explanation_result, drift_result) -> Dict[str, bytes]:
    output_item_json = {
        "data": {
            "result": inference_result,
            "explanation": explanation_result,
            "drift": drift_result,
        }
    }
    return {"results.json": json.dumps(output_item_json, separators=(",", ":")).encode()}


def get_failure_json_structure(error_message: str) -> Dict[str, bytes]:
    error_json = {"error_message": error_message}

    return {"error": json.dumps(error_json).encode()}


class SentimentAnalysis:
    # Note: Throwing unhandled exceptions that contain lots of information about the issue is expected and encouraged
    # for models when they encounter any issues or internal errors.

    def __init__(self):
        """
        This constructor should perform all initialization for your model. For example, all one-time tasks such as
        loading your model weights into memory should be performed here.

        This corresponds to the Status remote procedure call.
        """
        self.model = SentimentIntensityAnalyzer()
        self.output_classes = ["negative", "neutral", "positive"]

    def predict(self, text):
        '''
        This method replicates self.predict, except it does not include "compound" as an output class
        '''
        all_probs = []
        for item in text:
            results = self.model.polarity_scores(item)
            probs = [v for v in results.values()][:3]
            all_probs.append(probs)

        probs_np = np.array(all_probs)

        return probs_np


    def get_explainability(self, text):
        # set the random seeds for repeatability
        explainer = lime_text.LimeTextExplainer(random_state=0)
        exp = explainer.explain_instance(
            text, 
            self.predict, 
            top_labels=1, 
            num_samples=5000,
            num_features=40
        )

        # filter out any negative explanation scores
        as_map = [tup for tup in exp.as_map()[exp.available_labels()[0]] if (tup[1] >= 0)]
        as_list = [tup for tup in exp.as_list(label=exp.available_labels()[0]) if (tup[1] >= 0)]

        # take only the top 3 highest weighted words if there are at least 3 positive values
        if len(as_map) <3:
            num_samples = len(as_map)
        else:
            num_samples = 3
        
        # convert explanation to index, word, score format to be returned 
        explanation_result = {"wordImportance": [{"index": int(as_map[i][0]),"word": as_list[i][0],"score": as_list[i][1]} for i in range(num_samples)]}
        
        return explanation_result
    
    
    def handle_single_input(self, model_input: Dict[str, bytes], detect_drift: bool, explain: bool) -> Dict[str, bytes]:
        """
        This corresponds to the Run remote procedure call for single inputs.
        """
        # `model_input` will have binary contents for each of the input file types specified in your model.yaml file

        # You are responsible for processing these files in a manner that is specific to your model, and producing
        # inference, drift, and explainability results where appropriate.
        input_txt_contents = model_input["input.txt"].decode()
        probs = self.predict([input_txt_contents])
        probs_list = probs.tolist()[0]
        formatted_results = [{"class": label,"score": score} for label, score in zip(self.output_classes, probs_list)]
        formatted_results.sort(key=lambda x: x["score"],reverse=True)
        top_pred_index = self.output_classes.index(formatted_results[0]["class"])
        inference_result = {"classPredictions": formatted_results}
        drift_result = None
        if not explain:
            explanation_result = None
        else:
            explanation_result = self.get_explainability(input_txt_contents)

        
        # Load the results that your model produced into the standardized output format. If you model ran into
        # an error that you would like to handle internal, you will instead use the `get_failure_json_structure`
        # function in order to produce an error output.
        output_item = get_success_json_structure(inference_result, explanation_result, drift_result)

        return output_item


    def handle_input_batch(self, model_inputs: List[Dict[str, bytes]], detect_drift, explain) -> List[Dict[str, bytes]]:
        """
        This is an optional method that will be attempted to be called when more than one inputs to the model
        are ready to be processed. This enables a user to provide a more efficient means of handling inputs in batch
        that takes advantage of specific properties of their model.

        If you are not implementing custom batch processing, this method should raise a NotImplementedError. If you are
        implementing custom batch processing, then any unhandled exception will be interpreted as a fatal error that
        will result in the entire batch failing. If you would like to allow individual elements of the batch to fail
        without failing the entire batch, then you must handle the exception within this function, and ensure the JSON
        structure for messages with an error has a top level "error" key with a detailed description of the error
        message.

        This corresponds to the Run remote procedure call for batch inputs.

        {
            "error": "your error message here"
        }

        """
        raise NotImplementedError
