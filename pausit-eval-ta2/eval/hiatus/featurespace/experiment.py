"""Experiment implementation for TA1."""

import hashlib
import importlib
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from scipy.stats import rankdata
from sklearn.metrics.pairwise import _VALID_METRICS, pairwise_distances
from yaml import Loader, load

from eval.hiatus import ExperimentBase
from eval.hiatus.featurespace.metrics import compute_rank_metrics


LOG = logging.getLogger(__name__)
DEFAULT_METRIC = "euclidean"


class add_path:
    """Context manager to temporarily add to sys.path.

    https://stackoverflow.com/a/39855753.
    """

    def __init__(self, path):
        """Initialize."""
        self.path = path

    def __enter__(self):
        """Add to path."""
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        """Revert path to previous state."""
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass


def import_module(module_path, reload_module=False):
    """(Re)Import a Python module."""
    module_path = Path(module_path)
    try:
        module_ = sys.modules[module_path.stem]
    except KeyError:
        pass
    else:
        if reload_module:
            return importlib.reload(module_)
        else:
            return module_

    assert module_path.exists(), f"File {module_path} does not exist."
    with add_path(str(module_path.parent.resolve())):
        return importlib.import_module(module_path.stem)


class Experiment(ExperimentBase):
    """Container for TA1 experiments.

    Attributes:
        config (Dict[str, Any]): Configuration for the experiment.
        dataset (datasets.DatasetDict):
        metric (Union[str, Callable]): The metric to use when calculating distance between instances in a
            feature array. If metric is a string, it must be one of the options
            allowed by scikit-learn's pairwise_distances function.
            Alternatively, if metric is a callable function, it is called on each
            pair of instances (rows) and the resulting value recorded. The callable
            should take two arrays from as input and return a value indicating
            the distance between them.
    """

    def __init__(self, *args, **kwargs):
        """Initialize a new experiment."""
        super().__init__(*args, **kwargs)

        rng_seed = self.config["ta1"].get("random_seed", 100)
        self.rng = np.random.default_rng(seed=rng_seed)

        # Load metric from performer config
        self.metric = None
        performer_config_path = self.config["ta1"].get("performer_config_path")
        if performer_config_path is not None:
            performer_config_path = Path(performer_config_path).resolve()
            if performer_config_path.exists():
                try:
                    performer_config = load(performer_config_path.read_text(), Loader=Loader)
                except Exception as ex:
                    raise Exception("Problem loading TA1 performer system config") from ex
                if "TA1" in performer_config:
                    performer_config = performer_config["TA1"]
                    LOG.debug("TA1 performer config uses optional nested 'TA1' element.")
                metric_name = performer_config.get("Metric Name")
                metric_module_path = performer_config.get("Metric Module Path")
                if metric_name is not None:
                    if metric_module_path is not None:
                        raise ValueError(
                            "Problem parsing TA1 performer system config: "
                            f"Both metric_name and metric_module_path in config ({performer_config})"
                        )
                    if metric_name not in _VALID_METRICS:
                        raise ValueError(
                            "Problem parsing TA1 performer system config: "
                            f"Unknown metric {metric_name}. Valid metrics are {_VALID_METRICS}"
                        )
                    self.metric = metric_name
                    LOG.info(f"Using TA1 metric {metric_name}")
                elif metric_module_path is not None:
                    try:
                        metric_module = import_module(performer_config_path.parent / metric_module_path)
                        self.metric = getattr(metric_module, "metric")
                    except Exception as ex:
                        raise ValueError(f"Problem parsing TA1 custom distance function ({performer_config})") from ex
                    LOG.info("Using custom TA1 metric")
                else:
                    raise ValueError(
                        "Problem parsing TA1 performer system config: "
                        f"Neither metric_name nor metric_module_path in config ({performer_config})"
                    )
            else:
                LOG.warning("TA1 performer system config does not exist; using defaults.")
        else:
            LOG.warning("No TA1 performer system config provided; using defaults.")

        if self.metric is None:
            LOG.info(f"Setting metric to default {DEFAULT_METRIC}")
            self.metric = DEFAULT_METRIC

        # Load features
        features_path = self.config["ta1"].get("features_dataset")
        if features_path is None:
            raise ValueError("You must provide a Features Dataset")
        self.dataset = self.load_dataset(features_path)

        # Load labels
        ground_truth_path = self.config["ta1"].get("ground_truth")
        if ground_truth_path is None:
            raise ValueError("You must provide ground truth jsonl with Features Dataset")
        mappings = pd.read_json(ground_truth_path, lines=True, dtype=False)
        assert "authorIDs" in mappings.columns, mappings.columns
        assert "documentID" in mappings.columns, mappings.columns
        mappings["documentID"] = mappings["documentID"].astype(str)
        mapping_dict = mappings.set_index("documentID")["authorIDs"].to_dict()

        features_df = self.dataset.to_pandas()
        features_df["documentID"] = features_df["documentID"].astype(str)

        # Add authorIDs column
        if not features_df["documentID"].isin(mapping_dict).all():
            raise ValueError("Documents in feature set which aren't present in ground truth")
        features_df["authorIDs"] = features_df["documentID"].map(mapping_dict)
        features_df["authorIDs"] = features_df["authorIDs"].astype(str)

        # filter features_dataset based on documentIDs from resample files
        resample_queries_file = self.config["ta1"].get("resample_queries")
        resample_candidates_file = self.config["ta1"].get("resample_candidates")
        q, c = self.resample_dataset(resample_queries_file, resample_candidates_file, features_df)

        # If privatized_features_dataset is passed, replace original queries with new queries
        # Dropping original queries and new candidates
        stylistic_consistency = self.config["ta1"].get("stylistic_consistency", False)
        if stylistic_consistency:
            # REMOVE log message after stylistic consistency rework
            LOG.info("WARNING: Stylistic consistency undergoing rework. Current implementation may fail.")
            c, q = self.build_stylistic_consistency_dataset(c, q, self.rng)

        self.candidate_labels = c["authorIDs"].tolist()
        self.query_labels = q["authorIDs"].tolist()
        self.dataset = DatasetDict({"queries": Dataset.from_pandas(q), "candidates": Dataset.from_pandas(c)})

        self.distance_matrix = None
        self.rank_and_distance = None

    def rank_relevant_documents(self):
        """Compute rank and distance information."""
        if not set(self.query_labels) <= set(self.candidate_labels):
            raise NotImplementedError("Authors in query set which aren't present in candidate set")

        if self.distance_matrix is None:
            LOG.debug("Beginning pairwise distance calculation")
            self.distance_matrix = pairwise_distances(
                self.dataset["queries"]["features"],
                self.dataset["candidates"]["features"],
                metric=self.metric,
                n_jobs=self.config["ta1"].get("n_jobs"),
            )
            self.distance_matrix = self.distance_matrix.round(6)
        else:
            LOG.debug("Using pre-computed distance matrix")

        LOG.debug("Ranking candidate documents based on distance from query documents.")
        ranks = rankdata(self.distance_matrix, method=self.config["ta1"].get("rank_method", "min"), axis=1)
        self.raw_ranks = ranks

        LOG.debug("Finding rank and distance from queries to true candidates.")
        # Indices of candidate documents belonging to each author
        candidates_arr = np.array(self.candidate_labels)
        author_masks = {label: np.flatnonzero(candidates_arr == label) for label in set(self.candidate_labels)}

        all_needle_ranks = []
        all_needle_distances = []
        for i, label in enumerate(self.query_labels):
            instance_needle_ranks = list(ranks[i, author_masks[label]])
            all_needle_ranks.append(instance_needle_ranks)

            instance_needle_distances = list(self.distance_matrix[i, author_masks[label]])
            all_needle_distances.append(instance_needle_distances)

        self.rank_and_distance = pd.DataFrame(
            {
                "Ranks of Relevant Documents": all_needle_ranks,
                "Distance to Relevant Documents": all_needle_distances,
            },
            index=self.query_labels,
        )

    def compute_metrics(self, author_ids=None):
        """Compute TA1 metrics on a subset of query documents."""
        if self.rank_and_distance is None:
            self.rank_relevant_documents()

        if author_ids is None:
            rank_and_distance = self.rank_and_distance
        else:
            rank_and_distance = self.rank_and_distance.loc[author_ids, :]

        rank_and_distance = rank_and_distance.rename(
            columns={
                "Ranks of Relevant Documents": "all_needle_ranks",
                "Distance to Relevant Documents": "all_needle_distances",
            },
            errors="raise",
        )
        idx = rank_and_distance.index
        rank_and_distance = rank_and_distance.to_dict("series")

        instance_metrics, summary_metrics, plot_data = compute_rank_metrics(
            n_candidates=len(self.candidate_labels),
            raw_ranks=self.raw_ranks,
            top_ks=self.config["ta1"].get("top_k"),
            compute_metrics_at_distance=self.config["ta1"].get("compute_metrics_at_distance", False),
            plot_resolution=self.config["ta1"].get("plot_resolution", 100),
            disable_progress_bars=True,
            **rank_and_distance,
        )
        instance_metrics.index = idx
        return instance_metrics, summary_metrics, plot_data

    def save_to_disk(
        self, output_directory: Union[str, os.PathLike], save_dataset: bool = True, save_distance_matrix: bool = True
    ):
        """Save the experiment to disk."""
        output_path = super().save_to_disk(output_directory)

        name = self.config["name"]

        if isinstance(self.metric, str):
            (output_path / f"performer-config-{name}.yaml").write_text(f'"Metric Name": {self.metric}')
            self.config["ta1"]["performer_config_path"] = str(
                (output_path / f"performer-config-{name}.yaml").resolve()
            )
        else:
            LOG.warning("Custom metric will not be saved alongside Experiment.")

        if save_dataset:
            self.dataset.save_to_disk(output_path / "dataset")

        if save_distance_matrix:
            if self.distance_matrix is not None:
                np.save(output_path / f"distance-matrix-{name}.npy", self.distance_matrix)
                (output_path / f"candidate-labels-{name}.txt").write_text("\n".join(self.candidate_labels))
                (output_path / f"query-labels-{name}.txt").write_text("\n".join(self.query_labels))
                super().save_to_disk(output_directory, overwrite_warn=False)
            else:
                LOG.warning(
                    "Attempting to save a distance matrix that hasn't been computed. "
                    "Have you executed the experiment yet?"
                )

        if self.rank_and_distance is not None:
            self.rank_and_distance.to_parquet(output_path / f"rank-and-distance-{name}.parquet")

        super().save_to_disk(output_directory, overwrite_warn=False)

    @staticmethod
    def load_dataset(dataset_path: Union[str, os.PathLike]) -> Dataset:
        """Load a datasets.DatasetDict from disk."""
        dataset = load_from_disk(str(dataset_path))

        if not isinstance(dataset, Dataset):
            raise ValueError("dataset_path should point to a Dataset, not a DatasetDict")
        if "documentID" not in dataset.features.keys():
            raise ValueError("Dataset must contain 'documentID' column")
        if "features" not in dataset.features.keys():
            raise ValueError("Dataset must contain 'features' column")

        return dataset

    @staticmethod
    def resample_dataset(new_queries_file, new_candidates_file, old_features):
        """Sample provided feature space dataset for specific queries/candidates."""
        new_query = pd.read_json(new_queries_file, lines=True)
        new_query_docs = new_query["documentID"].tolist()

        # Look for query features in old_features
        queries_filtered = old_features.loc[old_features["documentID"].isin(new_query_docs)]
        LOG.info("Filtered features dataset based on new queries")

        new_candidates = pd.read_json(new_candidates_file, lines=True)
        new_candidate_docs = new_candidates["documentID"].tolist()

        # Look for candidate features in old_features
        candies_filtered = old_features.loc[old_features["documentID"].isin(new_candidate_docs)]
        LOG.info("Filtered features dataset based on new candidates")

        return queries_filtered, candies_filtered

    @staticmethod
    def build_stylistic_consistency_dataset(orig_candidates, orig_queries, rng):
        """Build new stylistic consistency dataset with original candidates and queries from privatized dataset.

        For every author in the query set from the privatized dataset, one document is kept as a query document.
        The remainder are added to the candidate documents. All authors in both query and candidates get new ids.
        Returns the query set from the privatized dataset and the original
        candidates + the additional candidates.
        Note: not yet compatible with multi-author documents.

        Args:
            orig_candidates (DataFrame): Candidates from original dataset.
            orig_queries (DataFrame): Queries from original dataset.
            rng (numpy.random.Generator): Random number generator.

        Returns:
            orig_and_style_cand (DataFrame): Candidates from new and original datasets.
            unique_style_q (DataFrame): New queries.

        Raises:
            ValueError: Not enough documents in a query set to perform resampling.
        """
        # Give new authors new ids
        new_hash_map = {}
        # authorIDs are str (needs to be updated to be compatible with multi-author docs)
        for i in orig_queries["authorIDs"].unique():
            m = hashlib.md5()
            m.update(i.encode())
            new_hash_map[i] = str(uuid.UUID(m.hexdigest()))
        orig_queries.loc[:, "authorIDs"] = orig_queries["authorIDs"].map(new_hash_map)
        orig_queries.loc[:, "authorIDs"] = orig_queries["authorIDs"].apply(lambda x: str((x,)))

        # Only one document remains as a query; the rest become candidates
        # Only keeping new queries and original candidate sets
        new_query_set = []
        additional_candidates = []
        for i, grp in sorted(orig_queries.groupby("authorIDs")):
            docIDs = grp["documentID"].tolist()
            if len(docIDs) < 2:
                raise ValueError(
                    "Query set with fewer than two documents detected."
                    "This dataset is not compatible with stylistic consistency."
                )
            # Select random doc to remain in query set
            doc_selected = rng.choice(docIDs, 1)[0]
            doc_index = docIDs.index(doc_selected)
            new_query_set.append(doc_selected)
            # Add the rest to candidate set
            add_cands = docIDs[:doc_index] + docIDs[doc_index + 1 :]
            additional_candidates.extend(add_cands)
        new_candidates = pd.concat(
            [orig_candidates, orig_queries.loc[orig_queries["documentID"].isin(additional_candidates)]]
        ).reset_index(drop=True)
        new_queries = orig_queries.loc[orig_queries["documentID"].isin(new_query_set)].reset_index(drop=True)

        LOG.info("Built new features dataset with stylistic consistency queries.")
        return new_candidates, new_queries
