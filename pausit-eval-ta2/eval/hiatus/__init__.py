"""Implementation of HIATUS evaluation metrics.

TA1: stylistic feature spaces in which each author is represented in stable fashion across diverse text types
TA2: algorithms for authorship attribution
TA3: algorithms for author privacy
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

from yaml import Loader, dump, load


__version__ = "2.0.0"


LOG = logging.getLogger(__name__)


class ExperimentBase:
    """Base class for HIATUS experiments.

    Attributes:
        config_path (str): Name for the experiment (used in serialization).
    """

    def __init__(self, *, config_path: Optional[str] = None, config: Optional[Dict] = None):
        """Initialize a new experiment."""
        if config_path is not None:
            if config is not None:
                raise ValueError("Pass exactly one of 'config_path' and 'config'")
            self.config = load(Path(config_path).read_text(), Loader=Loader)
        elif config is not None:
            self.config = config
        else:
            raise ValueError("Pass exactly one of 'config_path' and 'config'")

    def save_to_disk(self, output_directory, overwrite_warn=True):
        """Save the experiment to disk."""
        output_path = Path(output_directory)
        assert not output_path.exists() or output_path.is_dir(), output_path
        output_path.mkdir(exist_ok=True, parents=True)
        if overwrite_warn and len(list(output_path.glob("*"))):
            LOG.warning(
                "Output directory is not empty. Some files may be overwritten. You have five seconds to abort."
            )
            for i in range(5, 0, -1):
                LOG.warning(f"Continuing in {i}...")
                time.sleep(1)

        (output_path / "config.yml").write_text(dump(self.config))

        return output_path
