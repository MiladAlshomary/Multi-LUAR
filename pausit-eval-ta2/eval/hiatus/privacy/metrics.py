"""Implementation of TA3 metrics."""

import warnings
from typing import Any, Dict, Tuple

import pandas as pd
import torch
from evaluate import load  # type: ignore

from hiatus.privacy.gpt4eval import GPT4EvalScore


def compute_sense_metrics(
    documents,
    gold_referenced_documents,
    documentIDs,
    metric_name="gpt4eval",
    aspects=None,
    criteria_path=None,
    n_threads=1,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]], Dict[str, Any]]:
    """Compute sense metric.

    METEOR is based on unigram matching between the privatized and original texts.
    Original Implementation is at https://huggingface.co/spaces/evaluate-metric/meteor

    GPT4Eval metric assess whether privatized query documents preserve the sense of the original query documents.
    The metric includes evaluation instructions, rubrics and scoring criteria,
    and an output format to compare two text documents.
    For a given prompt template T, scoring criteria C,
    evaluation aspect a (e.g., coherence, consistency) and a large language model LLM (·),
    the sense evaluation score is defined as the LLM (T (d_{qo}, d_{qp}, C, a))
    comparing the privatized query document d_{qp} with the original query document d_{qp}.


    Args:
        documents: privatized texts
        gold_referenced_documents: original texts
        documentIDs: the list of unique document identifiers
        aspects: a list of aspects to be evaluated, or a mapping between a aspect to its description
        metric_name: sense metric name listed in the supported metric list
        criteria_path: a path to a JSON file with aspect to the evaluation criteria mappings
        n_threads: number of concurrent requests to make to the Azure OpenAI service

    Returns:
        instance_records: instance level metric outputs
        summary: summary level metric output
        plot: plot related data

    """
    summary: Dict[str, Any] = {}
    plots: Dict[str, None] = {}
    if metric_name == "meteor":
        warnings.warn("'meteor' is deprecated and will be removed in a future release.", DeprecationWarning)
        meteor = load(metric_name)
        meteor_results = meteor.compute(predictions=documents, references=gold_referenced_documents)
        summary["Sense (METEOR)"] = meteor_results[metric_name]
        instance_records = pd.DataFrame()
    elif metric_name == "gpt4eval":
        gpt4evalscore = GPT4EvalScore(criteria_path=criteria_path)
        results = gpt4evalscore._compute(
            predictions=documents, references=gold_referenced_documents, aspects=aspects, n_threads=n_threads
        )
        instance_records = results["Scores"]
        instance_records.index = documentIDs
        instance_records["OpenAI Responses"] = results["Responses"]
        summary["Sense (GPT4Eval)"] = results["Score"]

    return instance_records, summary, plots


def compute_soundness_metrics(
    documents, gold_referenced_documents, metric_name="perplexity", model_id="gpt2"
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]], Dict[str, Any]]:
    """Compute perplexity.

    Perplexity measures how likely a large language model to generate the input text sequence.
    Original Implementation is at https://huggingface.co/spaces/evaluate-metric/perplexity

    MAUVE summarizes both Type I and Type II errors measured softly using Kullback–Leibler (KL) divergences.
    Original Implementation is at https://huggingface.co/spaces/evaluate-metric/mauve

    Args:
        documents: privatized or original texts (to calculate perplexity) texts
        gold_referenced_documents: original texts
        metric_name: soundness metric name listed in the supported metric list
        model_id: language model name as appeared in the HuggingFace model registry.

    Returns:
        instance_records: instance level metric outputs
        summary: summary level metric output
        plot: plot related data

    """
    instance: Dict[str, Dict[str, int]] = {}
    summary: Dict[str, Dict[str, int]] = {}
    plots: Dict[str, None] = {}
    device_name = "cpu"
    device_id = -1
    if torch.cuda.is_available():
        device_name = "cuda"
        device_id = 0

    if metric_name == "perplexity":
        perplexity = load("perplexity", module_type="metric")
        perplexity_results_privatized = perplexity.compute(
            predictions=documents, model_id=model_id, device=device_name, max_length=1024
        )
        instance["Soundness (Perplexity) (Privatized Text)"] = perplexity_results_privatized["perplexities"]
        summary["Soundness (Perplexity) (Privatized Text)"] = perplexity_results_privatized["mean_perplexity"]

        perplexity_results_original = perplexity.compute(
            predictions=gold_referenced_documents, model_id=model_id, device=device_name, max_length=1024
        )
        instance["Soundness (Perplexity) (Original Text)"] = perplexity_results_original["perplexities"]
        summary["Soundness (Perplexity) (Original Text)"] = perplexity_results_original["mean_perplexity"]

        summary["Soundness (Perplexity) Improvement (%)"] = (
            (perplexity_results_original["mean_perplexity"] - perplexity_results_privatized["mean_perplexity"])
            / perplexity_results_original["mean_perplexity"]
            * 100
        )
    elif metric_name == "mauve":
        warnings.warn("'mauve' is deprecated and will be removed in a future release.", DeprecationWarning)
        mauve = load(metric_name)
        mauve_results = mauve.compute(
            predictions=documents,
            references=gold_referenced_documents,
            featurize_model_name=model_id,
            device_id=device_id,
        )
        summary["Soundness (MAUVE)"] = mauve_results.mauve
    return instance, summary, plots
