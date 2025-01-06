"""GPT4Eval Metric."""

import json
import logging
import os
from concurrent import futures
from typing import Dict, List, Optional, Union

import datasets
import evaluate  # type: ignore
import openai
import pandas as pd
from pydantic import ConfigDict, ValidationError, ValidationInfo, create_model, field_validator


LOG = logging.getLogger(__name__)

_CITATION = """\
@article{fu2023gptscore,
  title={Gptscore: Evaluate as you desire},
  author={Fu, Jinlan and Ng, See-Kiong and Jiang, Zhengbao and Liu, Pengfei},
  journal={arXiv preprint arXiv:2302.04166},
  year={2023}
}
@article{liu2023calibrating,
  title={Calibrating LLM-Based Evaluator},
  author={Liu, Yuxuan and Yang, Tianchi et al.},
  journal={arXiv preprint arXiv:2309.13308},
  year={2023}
}
"""

_DESCRIPTION = """GPT4Eval metric assess whether privatized query documents
preserve the sense of the original query documents.
The metric includes evaluation instructions, rubrics and scoring criteria,
and an output format to compare two text documents.
For a given prompt template T, scoring criteria C,
evaluation aspect a (e.g., coherence, consistency) and a large language model LLM (·),
the sense evaluation score is defined as the LLM (T (d_{qo}, d_{qp}, C, a))
comparing the privatized query document d_{qp} with the original query document d_{qp}."""

_KWARGS_DESCRIPTION = """Args:
    predictions (list of str): Prediction/candidate sentences.
    references (list of str or list of list of str): Reference sentences.
    aspects (list of str or dictionary): Aspect name or dictionary that maps aspect to a description
    openai_api_version (str): OpenAI API version
    openai_api_model_name (str): LLM supported through OpenAI API
    nthreads (int): Number of threads.

Returns:
    score: Aggregated score across different aspects.

Examples:

    >>> predictions = ["hello there", "general kenobi"]
    >>> references = ["hello there", "general kenobi"]
    >>> gpt4evalscore = GPT4EvalScore(criteria_path='./criteria.json')
    >>> results = gpt4evalscore._compute(predictions=predictions, references=references)
    >>> print(results)
    {
    "factuality": 20
    }
    >>> results = gpt4evalscore._compute(predictions=predictions, references=references,\
        aspects=["factuality","relevance"])
    >>> print(results)
    {
    "factuality": 20,
    "relevance": 20
    }
    >>> results = gpt4evalscore._compute(predictions=predictions, references=references,\
        aspects={"factuality": "Does text2 preserve the factual statements of the text1?"})
    >>> print(results)
    {
    "factuality": 20
    }
"""


DEFAULT_CRITERIA = {"factuality": "Does text2 preserve the factual statements of the text1?"}


def check_positive_numeric(cls, v: int, info: ValidationInfo) -> int:
    """Validate that a value is zero or a positive integer."""
    if isinstance(v, int) and v >= 0:
        return v
    else:
        raise ValueError(f"{info.field_name} must be zero or positive integer.")


class GPT4EvalScore(evaluate.Metric):
    """Container for GPT4Eval Metric.

    Attributes:
        criteria_path: custom evaluation criteria

    """

    def __init__(self, criteria_path=None):
        """Initialize the metric with the supported evaluation criteria.

        Args:
            criteria_path: path to a JSON document with the mapping between aspects and descriptions
        """
        if criteria_path and os.path.isfile(criteria_path):
            LOG.info(f"Criteria file is found: {criteria_path}")
            with open(criteria_path) as criteria_file:
                self._CRITERIA = json.load(criteria_file)
        else:
            LOG.warning(f"Criteria file is not found: {criteria_path}. Using default criteria.")
            self._CRITERIA = DEFAULT_CRITERIA
        LOG.debug(f"Defined aspects: {self._CRITERIA.keys()}")

    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="",
            inputs_description=_KWARGS_DESCRIPTION,
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Sequence(datasets.Value("string", id="sequence"), id="references"),
                    }
                ),
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Value("string", id="sequence"),
                    }
                ),
            ],
            codebase_urls=[""],
            reference_urls=[
                "https://arxiv.org/pdf/2302.04166.pdf",
                "https://arxiv.org/pdf/2309.13308.pdf",
            ],
        )

    def _resolve_criteria(self, aspects: Optional[Union[Dict[str, str], List[str]]]) -> Dict[str, str]:
        """Resolve the evaluation criteria to use in the prompt.

        Args:
            aspects: a list of aspects to be evaluated, or a mapping between a aspect to its description

        Returns:
            criteria_: a mapping between a aspect to its description

        Raises:
            ValueError: Validating aspects with the supported list.

        """
        LOG.debug(f"Input Aspects: {aspects}")

        criteria_: Dict[str, str] = dict()
        if aspects is None:
            criteria_ = self._CRITERIA
        elif isinstance(aspects, list):  # aspects = [x, y, z] as a list
            for aspect in aspects:
                if aspect in self._CRITERIA:  # presence in the predfined criteria
                    criteria_[aspect] = self._CRITERIA[aspect]
                else:
                    raise ValueError(
                        f"Aspect {aspect} is not defined. "
                        f"Please pass a path to JSON file or a dictionary with aspect to evaluation criteria."
                        f"E.g., {aspect}: description of the evaluation criteria"
                    )
        elif isinstance(aspects, dict):  # aspect to description mapping as a dict
            LOG.debug("Using custom criteria passed at runtime.")
            criteria_ = aspects
        else:
            raise ValueError(
                "Aspects cannot be empty. "
                "Please provide a aspect name or a mapping of the aspect name to its description."
            )

        return criteria_

    def _compute(
        self,
        predictions,
        references,
        aspects=None,
        openai_api_endpoint=None,
        openai_api_key=None,
        openai_api_version="2023-05-15",
        openai_api_model_name="gpt-4",
        n_threads=1,
    ) -> Dict[str, Union[Dict[str, int], pd.DataFrame, List[str]]]:
        """Compute the GPT4Eval metric.

        Args:
            predictions: a list of privatized query document.
            references: a list of original query documents.
            aspects: Aspect name or dictionary that maps aspect to a description
            openai_api_endpoint: Azure OpenAI API Endpoint
            openai_api_key: Azure OpenAI API Key
            openai_api_version: OpenAI API version
            openai_api_model_name: LLM supported through OpenAI API
            n_threads: Number of threads.

        Returns:
            a dictionary with scores for given aspects across all documents

        """
        if openai_api_endpoint is None:
            openai_api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        if openai_api_key is None:
            openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")

        LOG.debug(f"OPENAI API ENDPOINT: {openai_api_endpoint}")

        self.client = openai.AzureOpenAI(
            api_key=openai_api_key,
            # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
            api_version=openai_api_version,
            # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
            azure_endpoint=openai_api_endpoint,
        )

        criteria = self._resolve_criteria(aspects)
        LOG.debug(f"Resolved Criteria: {criteria}")
        criteria_str = "\n".join(f"{k}: {v}" for k, v in criteria.items())

        validator = create_model(  # type: ignore
            "validator",
            model_config=ConfigDict(extra="forbid"),
            __validators__={
                "positive_numeric": field_validator("*")(check_positive_numeric),
            },
            **{k: (int, ...) for k in criteria.keys()},
        )

        if n_threads > 1:  # threading
            with futures.ThreadPoolExecutor(max_workers=n_threads) as pool:
                futures_list = [
                    pool.submit(self.score, prediction, reference, criteria_str, validator, openai_api_model_name)
                    for prediction, reference in zip(predictions, references)
                ]
                results = [future.result() for future in futures_list]
        else:
            results = [
                self.score(prediction, reference, criteria_str, validator, openai_api_model_name)
                for prediction, reference in zip(predictions, references)
            ]

        score_list, responses = list(zip(*results))
        LOG.debug(f"Score List: {score_list}")
        n_failures = sum(x is None for x in score_list)
        if n_failures:
            LOG.warning(f"Failed evaluations detected: {n_failures} out of {len(score_list)}. See logs for details.")
        scores = pd.DataFrame(score_list)
        summary = dict(scores.mean(axis=0))
        return {"Score": summary, "Scores": scores, "Responses": responses}

    def score(
        self,
        text_privatized,
        text_original,
        criteria,
        validator,
        openai_api_model_name="gpt-4",
    ):
        """Compute the GPT4Eval metric.

        Args:
            text_privatized: privatized query document.
            text_original: original query document.
            criteria: evaluation criteria that describe the aspects
            validator: Pydantic model to validate output
            openai_api_model_name: LLM supported through OpenAI API

        Returns:
            dictionary of scores
            raw text response from API

        """
        scores = {}
        response_str = ""

        rubric = """Please score out of 20 points,
         using only zero or positive integers
         for each different aspect in the evaluation criteria"""
        response_format = "output the scores in a JSON structured response"
        system_prompt = f"""You are a helpful assistant and a judge to evaluate natural language texts.
         You will be given two text segments in the following format: [text1][text2].
         For text2, your task is to provide a score based on the following aspects and the criteria:
         {criteria}
         {rubric} and {response_format}."""
        LOG.debug(f"System Prompt: {system_prompt}")

        gpt_input = "[" + text_original + "] [" + text_privatized + "]"

        try:
            response = self.client.chat.completions.create(
                model=openai_api_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": gpt_input},
                ],
            )

            response_str = response.choices[0].message.content
            LOG.debug(f"OpenAI Response: {response_str}")

            # Parse the OpenAI response to a JSON
            response_dict = None
            try:
                response_dict = json.loads(response_str)
                LOG.debug(f"Parse Response: {response_dict}")
            except json.JSONDecodeError as e:
                LOG.warning(f"Malformed OpenAI response {response_str}. Expected a JSON response: {e}")

            # Validating the JSON response
            if isinstance(response_dict, dict):
                try:
                    validated_response_dict = validator.model_validate(response_dict)
                    scores = validated_response_dict.model_dump(exclude_none=True)
                    LOG.debug(f"Validated Response: {scores}")
                except ValidationError as e:
                    LOG.warning(f"Response Validation Error: {e}")
            else:
                LOG.warning(f"Malformed OpenAI response {response_dict}. Expected a dict.")

        except openai.APIConnectionError as e:
            LOG.warning(f"Failed to connect to OpenAI API: {e}")
        except openai.RateLimitError as e:
            LOG.warning(f"OpenAI API request exceeded rate limit: {e}")
        except openai.APIError as e:
            LOG.warning(f"OpenAI API returned an API Error: {e}")

        return scores, response_str
