"""Experiment implementation for TA3."""

import logging
from pathlib import Path

import pandas as pd

from eval.hiatus import ExperimentBase
from eval.hiatus.attribution.experiment import Experiment as TA2Experiment
from eval.hiatus.featurespace.experiment import Experiment as TA1Experiment
from eval.hiatus.privacy.metrics import compute_sense_metrics, compute_soundness_metrics


_VALID_STYLISTIC_METRICS = ["Mean Reciprocal Rank", "Harmonic Mean of Mean Percentile Rank"]
_VALID_PRIVACY_METRICS = ["Delta Equal Error Rate"]
_VALID_SOUNDNESS_METRICS = ["mauve", "perplexity"]
_VALID_SENSE_METRICS = ["meteor", "gpt4eval"]


LOG = logging.getLogger(__name__)


class Experiment(ExperimentBase):
    """Container for TA3 experiments.

    Attributes:
        config (Dict[str, Any]): Configuration for the experiment.
    """

    def __init__(self, *args, **kwargs):
        """Initialize a new experiment."""
        super().__init__(*args, **kwargs)

        # Load stylistic consistency metrics objects
        self.stylistic_experiments = []
        stylistic_metric_config = self.config["ta3"].get("stylistic_consistency")
        if stylistic_metric_config is None:
            self.stylistic_metric_name = None
        else:
            self.stylistic_metric_name = stylistic_metric_config["metric_name"]

            if self.stylistic_metric_name not in _VALID_STYLISTIC_METRICS:
                raise ValueError(
                    f"Unknown metric {self.stylistic_metric_name}. Valid metrics are {_VALID_STYLISTIC_METRICS}"
                )

            stylistic_metric_config_children = stylistic_metric_config["ta1_system_outputs"]
            feature_ground_truth_path = stylistic_metric_config["ground_truth"]
            for stylistic_metric_config_child in stylistic_metric_config_children:
                stylistic_system_name = stylistic_metric_config_child["ta1_system_name"]

                stylistic_consistency_in_context_flag = stylistic_metric_config_child.get(
                    "in_context_stylistic_consistency",
                )
                if stylistic_consistency_in_context_flag is None:
                    LOG.warning("'in_context_stylistic_consistency' flag not set. Defaulting to True.")
                    stylistic_consistency_in_context_flag = True

                privatized_features_path = stylistic_metric_config_child["privatized_features_dataset"]
                resample_queries_path = stylistic_metric_config_child["resample_queries"]
                resample_candidates_path = stylistic_metric_config_child["resample_candidates"]
                ta1_config = {
                    "name": stylistic_system_name,
                    "ta1": {
                        "features_dataset": Path(privatized_features_path).resolve().as_posix(),
                        "ground_truth": Path(feature_ground_truth_path).resolve().as_posix(),
                        "resample_queries": Path(resample_queries_path).resolve().as_posix(),
                        "resample_candidates": Path(resample_candidates_path).resolve().as_posix(),
                        "compute_metrics_at_distance": False,
                        "top_k": [],
                        "stylistic_consistency": True,
                        "in_context_stylistic_consistency": stylistic_consistency_in_context_flag,
                    },
                }

                stylistic_performer_config_path = stylistic_metric_config_child.get("performer_config_path")
                if stylistic_performer_config_path is not None:
                    ta1_config["ta1"]["performer_config_path"] = stylistic_performer_config_path
                else:
                    LOG.warning("No TA1 performer system config provided; using defaults.")

                LOG.debug(f"TA1 config: {ta1_config}")

                self.stylistic_experiments.append(TA1Experiment(config=ta1_config))

        # Load privacy metrics objects
        self.attribution_experiments = []
        self.deattribution_experiments = []
        privacy_metric_config = self.config["ta3"].get("privacy")
        if privacy_metric_config is None:
            self.privacy_metric_name = None
        else:
            self.privacy_metric_name = privacy_metric_config["metric_name"]

            if self.privacy_metric_name not in _VALID_PRIVACY_METRICS:
                raise ValueError(
                    f"Unknown metric {self.privacy_metric_name}. Valid metrics are {_VALID_PRIVACY_METRICS}"
                )

            self.privacy_metric_name_list = []
            if self.privacy_metric_name == "Delta Equal Error Rate":
                self.privacy_metric_name_list.append("eer")

            _ground_truth_path = privacy_metric_config["ground_truth"]
            _ground_truth_candidate_path = privacy_metric_config["ground_truth_candidate_labels"]
            _ground_truth_query_path = privacy_metric_config["ground_truth_query_labels"]

            privacy_metric_config_children = privacy_metric_config["ta2_system_outputs"]
            assert privacy_metric_config_children, "Must specify at least one TA2 System Output"
            for privacy_metric_config_child in privacy_metric_config_children:
                privacy_system_name = privacy_metric_config_child["ta2_system_name"]
                privacy_in_context_flag = privacy_metric_config_child.get("in_context_privacy")
                if privacy_in_context_flag is None:
                    LOG.warning("Privacy in context flag not set. Defaulting to True.")
                    privacy_in_context_flag = True

                _scores_path = privacy_metric_config_child["original_scores"]
                _scores_candidate_path = privacy_metric_config_child["original_scores_candidate_labels"]
                _scores_query_path = privacy_metric_config_child["original_scores_query_labels"]

                ta2_config = {
                    "name": privacy_system_name,
                    "ta2": {
                        "in_context_privacy": privacy_in_context_flag,
                        "metrics": self.privacy_metric_name_list,
                        "scores": Path(_scores_path).resolve().as_posix(),
                        "scores_candidate_labels": Path(_scores_candidate_path).resolve().as_posix(),
                        "scores_query_labels": Path(_scores_query_path).resolve().as_posix(),
                        "ground_truth": Path(_ground_truth_path).resolve().as_posix(),
                        "ground_truth_candidate_labels": Path(_ground_truth_candidate_path).resolve().as_posix(),
                        "ground_truth_query_labels": Path(_ground_truth_query_path).resolve().as_posix(),
                    },
                }

                privacy_performer_config_path = privacy_metric_config_child.get("performer_config_path")
                if privacy_performer_config_path is not None:
                    ta2_config["ta2"]["performer_config_path"] = privacy_performer_config_path
                else:
                    LOG.warning("No TA2 performer system config provided; using defaults.")

                LOG.debug(f"TA2 config: {ta2_config}")
                self.attribution_experiments.append(TA2Experiment(config=ta2_config))

                _scores_path = privacy_metric_config_child["privatized_scores"]
                _scores_candidate_path = privacy_metric_config_child["privatized_scores_candidate_labels"]
                _scores_query_path = privacy_metric_config_child["privatized_scores_query_labels"]
                ta2_config["ta2"]["scores"] = Path(_scores_path).resolve().as_posix()
                ta2_config["ta2"]["scores_candidate_labels"] = Path(_scores_candidate_path).resolve().as_posix()
                ta2_config["ta2"]["scores_query_labels"] = Path(_scores_query_path).resolve().as_posix()

                LOG.debug(f"Deattribution TA2 config: {ta2_config}")
                self.deattribution_experiments.append(TA2Experiment(config=ta2_config))

        # Load soundness metrics objects
        self.soundness_original_query_texts = None
        self.soundness_privatized_query_texts = None
        self.soundness_query_documentIDs = None
        soundness_metric_config = self.config["ta3"].get("soundness")
        if soundness_metric_config is None:
            self.soundness_metric_name = None
        else:
            # Load soundness metric
            soundness_metric_name = soundness_metric_config["metric_name"]
            if soundness_metric_name not in _VALID_SOUNDNESS_METRICS:
                raise ValueError(
                    f"Unknown metric {soundness_metric_name}. Valid metrics are {_VALID_SOUNDNESS_METRICS}"
                )
            self.soundness_metric_name = soundness_metric_name

            # Load original and privatized documents
            original_doc_path = soundness_metric_config["original_dataset"]
            original_query_dataset = pd.read_json(original_doc_path, lines=True)
            original_query_dataset = original_query_dataset[["documentID", "fullText"]]
            privatized_doc_path = soundness_metric_config["privatized_dataset"]
            privatized_query_dataset = pd.read_json(privatized_doc_path, lines=True)
            privatized_query_dataset = privatized_query_dataset[["documentID", "fullText"]]
            privatized_query_dataset.columns = ["documentID", "fullTextPrivatized"]

            if set(original_query_dataset["documentID"]) != set(privatized_query_dataset["documentID"]):
                raise ValueError("Original and privatized query documents should have the same document identifiers.")

            merge_query_dataset = pd.merge(
                original_query_dataset, privatized_query_dataset, on="documentID", how="inner"
            )
            merge_query_dataset = merge_query_dataset[["documentID", "fullText", "fullTextPrivatized"]]
            self.soundness_original_query_texts = merge_query_dataset["fullText"].to_list()
            self.soundness_privatized_query_texts = merge_query_dataset["fullTextPrivatized"].to_list()
            self.soundness_query_documentIDs = merge_query_dataset["documentID"].to_list()

        # Load sense metrics objects
        self.sense_original_query_texts = None
        self.sense_privatized_query_texts = None
        self.sense_query_documentIDs = None
        sense_metric_config = self.config["ta3"].get("sense")
        if sense_metric_config is None:
            self.sense_metric_name = None
        else:
            sense_metric_config = sense_metric_config.copy()  # don't alter original config
            # Load sense metric
            sense_metric_name = sense_metric_config.pop("metric_name")
            if sense_metric_name not in _VALID_SENSE_METRICS:
                raise ValueError(f"Unknown metric {sense_metric_name}. Valid metrics are {_VALID_SENSE_METRICS}")
            self.sense_metric_name = sense_metric_name

            # Load original and privatized documents
            original_doc_path = sense_metric_config.pop("original_dataset")
            original_query_dataset = pd.read_json(original_doc_path, lines=True)
            original_query_dataset = original_query_dataset[["documentID", "fullText"]]
            privatized_doc_path = sense_metric_config.pop("privatized_dataset")
            privatized_query_dataset = pd.read_json(privatized_doc_path, lines=True)
            privatized_query_dataset = privatized_query_dataset[["documentID", "fullText"]]
            privatized_query_dataset.columns = ["documentID", "fullTextPrivatized"]

            if set(original_query_dataset["documentID"]) != set(privatized_query_dataset["documentID"]):
                raise ValueError("Original and privatized query documents should have the same document identifiers.")

            merge_query_dataset = pd.merge(
                original_query_dataset, privatized_query_dataset, on="documentID", how="inner"
            )
            merge_query_dataset = merge_query_dataset[["documentID", "fullText", "fullTextPrivatized"]]
            self.sense_original_query_texts = merge_query_dataset["fullText"].to_list()
            self.sense_privatized_query_texts = merge_query_dataset["fullTextPrivatized"].to_list()
            self.sense_query_documentIDs = merge_query_dataset["documentID"].to_list()

            # Configure sense metrics with arbitrary options
            self.sense_metric_config = sense_metric_config

        # Check that we have sufficient data
        if self.stylistic_metric_name is not None and not self.stylistic_experiments:
            raise ValueError("You must provide at least one TA1 component to calculate the stylistic consistency.")

        if self.privacy_metric_name is not None and (
            not self.attribution_experiments or not self.deattribution_experiments
        ):
            raise ValueError(
                "You must provide at least one deattribution and attribution system to calculate the privacy metric."
            )

        if (
            self.soundness_metric_name is not None
            and self.sense_metric_name is not None
            and self.soundness_original_query_texts != self.sense_original_query_texts
        ):
            LOG.warning("Using different original texts for soundness and sense metric computation")
        if (
            self.soundness_metric_name is not None
            and self.sense_metric_name is not None
            and self.soundness_privatized_query_texts != self.sense_privatized_query_texts
        ):
            LOG.warning("Using different privatized texts for soundness and sense metric computation")
        if self.soundness_metric_name is not None and len(self.soundness_privatized_query_texts) != len(
            self.soundness_original_query_texts
        ):
            raise ValueError("Length of original and privatized soundness query texts should be same.")
        if self.sense_metric_name is not None and len(self.sense_privatized_query_texts) != len(
            self.sense_original_query_texts
        ):
            raise ValueError("Length of original and privatized sennse query texts should be same.")

    def compute_stylistic_consistency_metrics(self, stylistic_metric_name=None, author_ids=None):
        """Compute stylistic consistency metrics on a subset of privatized query documents."""
        summary = {}

        if stylistic_metric_name is None:
            stylistic_metric_name = self.stylistic_metric_name

        in_context_rank_metrics = []
        out_context_rank_metrics = []
        for stylistic_experiment in self.stylistic_experiments:
            instance_metrics, summary_metrics, plot_data = stylistic_experiment.compute_metrics()
            rank_metric = summary_metrics[stylistic_metric_name]
            if stylistic_experiment.config["ta1"]["in_context_stylistic_consistency"]:
                in_context_rank_metrics.append(rank_metric)
            else:
                out_context_rank_metrics.append(rank_metric)

        if in_context_rank_metrics:
            in_context_stylistic_score = sum(in_context_rank_metrics) / len(in_context_rank_metrics)
            summary["Stylistic Consistency (self)"] = in_context_stylistic_score

        if out_context_rank_metrics:
            out_context_stylistic_score = sum(out_context_rank_metrics) / len(out_context_rank_metrics)
            summary["Stylistic Consistency"] = out_context_stylistic_score

        return pd.DataFrame(), summary, {}

    def compute_privacy_metrics(self, attribution_metric_name=None, author_ids=None):
        """Compute privacy metrics on a subset of privatized query documents."""
        summary = {}

        if attribution_metric_name is None:
            if self.privacy_metric_name == "Delta Equal Error Rate":
                attribution_metric_name = "Equal Error Rate"

        in_context_deattribution_metrics = []
        out_context_deattribution_metrics = []
        for deattribution_experiment in self.deattribution_experiments:
            _, deattribution_summary_metrics, _ = deattribution_experiment.compute_metrics()

            deattribution_metric = deattribution_summary_metrics[attribution_metric_name]

            LOG.info(f"Deattribution metric: {attribution_metric_name} = {deattribution_metric}")

            if deattribution_experiment.config["ta2"]["in_context_privacy"]:
                in_context_deattribution_metrics.append(deattribution_metric)
            else:
                out_context_deattribution_metrics.append(deattribution_metric)

        in_context_attribution_metrics = []
        out_context_attribution_metrics = []
        for attribution_experiment in self.attribution_experiments:
            _, attribution_summary_metrics, _ = attribution_experiment.compute_metrics()

            attribution_metric = attribution_summary_metrics[attribution_metric_name]

            LOG.info(f"Attribution metric: {attribution_metric_name} = {attribution_metric}")

            if attribution_experiment.config["ta2"]["in_context_privacy"]:
                in_context_attribution_metrics.append(attribution_metric)
            else:
                out_context_attribution_metrics.append(attribution_metric)

        if in_context_attribution_metrics:
            in_context_attribution_score = sum(in_context_attribution_metrics) / len(in_context_attribution_metrics)
            in_context_deattribution_score = sum(in_context_deattribution_metrics) / len(
                in_context_deattribution_metrics
            )
            summary[f"Delta {attribution_metric_name} (self)"] = (
                in_context_deattribution_score - in_context_attribution_score
            )

        if out_context_attribution_metrics:
            out_context_attribution_score = sum(out_context_attribution_metrics) / len(out_context_attribution_metrics)
            out_context_deattribution_score = sum(out_context_deattribution_metrics) / len(
                out_context_deattribution_metrics
            )
            summary[f"Delta {attribution_metric_name}"] = (
                out_context_deattribution_score - out_context_attribution_score
            )

        return pd.DataFrame(), summary, {}

    def compute_sense_metrics(self, sense_metric_name=None, max_index=None, **metric_config):
        """Compute TA3 sense metrics on a subset of privatized and original query documents."""
        if sense_metric_name is None:
            sense_metric_name = self.sense_metric_name

        if max_index is not None:
            self.sense_privatized_query_texts = self.sense_privatized_query_texts[:max_index]
            self.sense_original_query_texts = self.sense_original_query_texts[:max_index]

        # Prefer runtime kwargs but default to config defined at initialization
        metric_config_ = self.sense_metric_config.copy()
        metric_config_.update(metric_config)

        instance_metrics, summary_metrics, plot_data = compute_sense_metrics(
            self.sense_privatized_query_texts,
            self.sense_original_query_texts,
            self.sense_query_documentIDs,
            metric_name=sense_metric_name,
            **metric_config_,
        )

        return instance_metrics, summary_metrics, plot_data

    def compute_soundness_metrics(self, soundness_metric_name=None, max_index=None):
        """Compute TA3 soundness metrics on a subset of privatized and original query documents."""
        if soundness_metric_name is None:
            soundness_metric_name = self.soundness_metric_name

        if max_index is not None:
            self.soundness_privatized_query_texts = self.soundness_privatized_query_texts[:max_index]
            self.soundness_original_query_texts = self.soundness_original_query_texts[:max_index]

        instance_metrics, summary_metrics, plot_data = compute_soundness_metrics(
            self.soundness_privatized_query_texts,
            self.soundness_original_query_texts,
            metric_name=soundness_metric_name,
        )

        return instance_metrics, summary_metrics, plot_data

    def compute_metrics(self, author_ids=None, max_index=None):
        """Compute TA3 metrics on a subset of query documents."""
        instance = {}
        summary = {}
        plots = {}

        if self.privacy_metric_name is not None:
            LOG.info("Computing Privacy Metrics.")
            instance_metrics, summary_metrics, plot_data = self.compute_privacy_metrics(author_ids=author_ids)
            assert summary.keys().isdisjoint(summary_metrics.keys()), "Duplicate metric names"
            summary.update(summary_metrics)

        if self.stylistic_metric_name is not None:
            LOG.info("Computing Stylistic Self-Consistency Metrics.")
            instance_metrics, summary_metrics, plot_data = self.compute_stylistic_consistency_metrics(
                author_ids=author_ids
            )
            assert summary.keys().isdisjoint(summary_metrics.keys()), "Duplicate metric names"
            summary.update(summary_metrics)

        if self.sense_metric_name is not None:
            LOG.info("Computing Sense Metrics.")
            instance_metrics, summary_metrics, plot_data = self.compute_sense_metrics(max_index=max_index)
            assert summary.keys().isdisjoint(summary_metrics.keys()), "Duplicate metric names"
            summary.update(summary_metrics)
            assert instance.keys().isdisjoint(instance_metrics.keys()), "Duplicate metric names"
            instance.update(instance_metrics)

        if self.soundness_metric_name is not None:
            LOG.info("Computing Soundness Metrics.")
            instance_metrics, summary_metrics, plot_data = self.compute_soundness_metrics(max_index=max_index)
            assert summary.keys().isdisjoint(summary_metrics.keys()), "Duplicate metric names"
            summary.update(summary_metrics)
            assert instance.keys().isdisjoint(instance_metrics.keys()), "Duplicate metric names"
            instance.update(instance_metrics)

        return pd.DataFrame(instance), pd.Series(summary), plots
