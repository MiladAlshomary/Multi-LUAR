"""Command-line interface for HIATUS metrics."""

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict

import hydra
from omegaconf import DictConfig, OmegaConf

from eval.hiatus.attribution.experiment import Experiment as TA2Experiment
from eval.hiatus.featurespace.experiment import Experiment as TA1Experiment
from eval.hiatus.privacy.experiment import Experiment as TA3Experiment


LOG = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=".", config_name="default")
def entrypoint(config: DictConfig):
    """Define CLI utility."""
    config = OmegaConf.to_object(config)  # type: ignore

    output_path_ = config.get("output_path")
    if output_path_ is None:
        raise ValueError("You must set Output Path' in the config.")

    output_path = Path(output_path_)
    if output_path.is_dir():
        raise ValueError(f"Output path {output_path} should be a file not a directory.")
    if output_path.exists():
        LOG.warning(f"Overwriting existing output at {output_path}")

    results: Dict[str, Any] = {
        "ExperimentConfig": config,
        "datetime": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    if "ta1" in config:
        LOG.info("Beginning TA1 evaluation")
        ta1_experiment = TA1Experiment(config=config)

        instance_metrics, summary_metrics, plot_data = ta1_experiment.compute_metrics()
        results["TA1"] = summary_metrics.to_dict()
        results["TA1"]["plots"] = plot_data
        if config.get("save_instance_metrics", True):
            instance_metrics.index = instance_metrics.index.astype(str)
            results["TA1"]["instance"] = instance_metrics.to_dict()

    if "ta2" in config:
        LOG.info("Beginning TA2 evaluation")
        ta2_experiment = TA2Experiment(config=config)

        instance_metrics, summary_metrics, plot_data = ta2_experiment.compute_metrics()
        results["TA2"] = summary_metrics.to_dict()
        results["TA2"]["plots"] = plot_data
        if config.get("save_instance_metrics", True):
            instance_metrics.index = instance_metrics.index.astype(str)
            results["TA2"]["instance"] = instance_metrics.to_dict()

    if "ta3" in config:
        LOG.info("Beginning TA3 evaluation")
        ta3_experiment = TA3Experiment(config=config)
        instance_metrics, summary_metrics, plot_data = ta3_experiment.compute_metrics()
        results["TA3"] = summary_metrics.to_dict()
        results["TA3"]["plots"] = plot_data
        if config.get("save_instance_metrics", True):
            instance_metrics.index = instance_metrics.index.astype(str)
            results["TA3"]["instance"] = instance_metrics.to_dict()

    output_path.write_text(json.dumps(results, indent=4))


if __name__ == "__main__":
    entrypoint()
