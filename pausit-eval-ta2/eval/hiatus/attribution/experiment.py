"""Experiment implementation for TA2."""

import json
import logging
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
from yaml import Loader, load

from eval.hiatus import ExperimentBase
from eval.hiatus.attribution.metrics import compute_attribution_metrics


LOG = logging.getLogger(__name__)
DEFAULT_THRESHOLD = 0.9


class Experiment(ExperimentBase):
    """Container for TA2 experiments.

    Attributes:
        config (Dict[str, Any]): Configuration for the experiment.
        metrics (Union[str, List[str]]): Which metrics to report.
        threshold (float): Cut off value for true predictions.
        far_target (List[float]): List of FAR values at which to calculate TAR.
        scores (pd.DataFrame): Matrix of attribution scores.
        ground_truth (pd.DataFrame): Matrix of ground truth attributions.
    """

    def __init__(self, *args, **kwargs):
        """Initialize a new experiment."""
        super().__init__(*args, **kwargs)

        self.threshold = None
        performer_config_path = self.config["ta2"].get("performer_config_path")
        if performer_config_path is not None:
            performer_config_path = Path(performer_config_path).resolve()
            if performer_config_path.exists():
                try:
                    performer_config = load(Path(performer_config_path).read_text(), Loader=Loader)
                except Exception as ex:
                    raise Exception("Problem loading TA2 performer system config") from ex
                if "TA2" in performer_config:
                    performer_config = performer_config["TA2"]
                    LOG.debug("TA2 performer config uses optional nested 'TA2' element.")
                self.threshold = performer_config["Threshold"]
                LOG.info(f"Using TA2 prediction threshold {self.threshold}")
        else:
            LOG.warning("No TA2 performer system config provided; using defaults.")
        if self.threshold is None:
            LOG.info(f"Setting threshold to default {DEFAULT_THRESHOLD}")
            self.threshold = DEFAULT_THRESHOLD

        self.far_target = self.config["ta2"].get("far_target")
        self.metrics = self.config["ta2"].get("metrics")

        if self.metrics is None or self.metrics == "" or (isinstance(self.metrics, list) and len(self.metrics) == 0):
            self.metrics = "all"
            LOG.warning(f"'metrics' not provided in config. Defaulting to {self.metrics}.")

        if self.threshold is None:
            self.threshold = 0.9
            LOG.warning(f"'Threshold' not provided in config. Defaulting to {self.threshold}.")

        if self.far_target is None or (isinstance(self.far_target, list) and len(self.far_target) == 0):
            if self.metrics == "all" or "tar_at_far" in self.metrics:
                self.far_target = [0.05]
                LOG.warning(
                    "'Far Target' not provided in config and 'metrics' set to include 'tar_at_far'. "
                    f"Defaulting 'Far Target' to {self.far_target}."
                )

        # Load authorship scores
        self.scores = self.load_dataset(
            score_path=self.config["ta2"]["scores"],
            candidate_labels_path=self.config["ta2"]["scores_candidate_labels"],
            query_labels_path=self.config["ta2"]["scores_query_labels"],
        )
        self.ground_truth = self.load_dataset(
            score_path=self.config["ta2"]["ground_truth"],
            candidate_labels_path=self.config["ta2"]["ground_truth_candidate_labels"],
            query_labels_path=self.config["ta2"]["ground_truth_query_labels"],
        )
        if (
            self.scores.shape != self.ground_truth.shape
            or set(self.scores.index) != set(self.ground_truth.index)
            or set(self.scores.columns) != set(self.ground_truth.columns)
        ):
            raise ValueError("Predictions and Ground Truth indices don't match")
        if len(set(self.ground_truth.index)) != self.ground_truth.shape[0]:
            raise ValueError("Provided non-unique query author index")
        if set(np.unique(self.ground_truth)) != {0, 1}:
            raise ValueError("Ground Truth contains non-boolean values")
        # align scores with ground truth
        if any(self.scores.index != self.ground_truth.index):
            self.scores = self.scores.loc[self.ground_truth.index, :]
        if any(self.scores.columns != self.ground_truth.columns):
            self.scores = self.scores.loc[:, self.ground_truth.columns]
        # make predictions
        self.predictions = (self.scores >= self.threshold).astype(int)

    def compute_metrics(self, author_ids=None, overwrite_config_metrics=None):
        """Compute TA2 metrics on a subset of query documents."""
        if author_ids is None:
            ground_truth = self.ground_truth
            scores = self.scores
            predictions = self.predictions
        else:
            ground_truth = ground_truth.loc[author_ids]
            scores = scores.loc[author_ids]
            predictions = predictions.loc[author_ids]

        if overwrite_config_metrics is not None:
            self.metrics = overwrite_config_metrics

        instance_metrics, summary_metrics, plotting_metrics = compute_attribution_metrics(
            ground_truth=ground_truth,
            scores=scores,
            predictions=predictions,
            far_target=self.far_target,
            which_metrics=self.metrics,
        )

        self.instances = instance_metrics
        self.summary = summary_metrics
        self.plots = plotting_metrics
        return instance_metrics, summary_metrics, plotting_metrics

    def save_to_disk(self, output_directory):
        """Save the experiment to disk."""
        output_path = super().save_to_disk(output_directory)
        name = self.config["name"]
        np.save(output_path / f"scores-{name}.npy", self.scores.to_numpy())
        np.save(output_path / f"predictions-{name}.npy", self.predictions.to_numpy())
        np.save(output_path / f"ground-truth-{name}.npy", self.ground_truth.to_numpy())
        (output_path / f"query-labels-{name}.txt").write_text("\n".join(self.ground_truth.index))
        (output_path / f"candidate-labels-{name}.txt").write_text("\n".join(self.ground_truth.columns))
        self.instances.to_json((output_path / f"instances-metrics-{name}.json"), orient="columns")
        self.summary.to_json((output_path / f"summary-metrics-{name}.json"), orient="columns")
        with (output_path / f"plotting-metrics-{name}.json").open("w") as fp:
            json.dump(self.plots, fp)

    @staticmethod
    def load_dataset(score_path, candidate_labels_path, query_labels_path):
        """Load a TA2 dataset from disk."""
        scores = np.load(score_path)

        candidate_labels = [
            literal_eval(label) for label in Path(candidate_labels_path).read_text().split("\n") if label
        ]

        query_labels = [literal_eval(label) for label in Path(query_labels_path).read_text().split("\n") if label]
        dataset = pd.DataFrame(scores, index=query_labels, columns=candidate_labels)
        return dataset
