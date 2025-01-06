"""Preprocess Raw Data CLI."""

import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from eval.hiatus.data.finetune_and_test_authors_sampling import (
    create_finetune_test_partitions,
    save_finetune_documents,
    save_test_documents,
)
from eval.hiatus.data.truncate import truncate_documents


LOG = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="data", config_name="preprocess")
def entrypoint(config: DictConfig):
    """Define data sampling CLI utility."""
    config = OmegaConf.to_object(config)  # type: ignore

    data_dir_raw = config.get("data_dir_raw")
    if data_dir_raw is None:
        raise ValueError("You must set 'data_dir_raw' in the config.")

    data_dir_finetune = config.get("data_dir_finetune")
    if data_dir_finetune is None:
        raise ValueError("You must set 'data_dir_finetune' in the config.")

    data_dir_test = config.get("data_dir_test")
    if data_dir_test is None:
        raise ValueError("You must set 'data_dir_test' in the config.")

    randomSeed = config.get("randomSeed")

    data_files_raw = sorted(Path(data_dir_raw).rglob("*.jsonl"))
    if not data_files_raw:
        raise FileNotFoundError(f"No .jsonl files found in data directory {data_files_raw}")
    else:
        LOG.info(f"Found the following data files to load: {[f.relative_to(data_dir_raw) for f in data_files_raw]}")

    maxTokens = config.get("maxTokens")

    frac_author_finetune_set = config.get("fraction_finetune_authors")

    finetune_sample_name = config.get("finetune_sample_name")
    if not finetune_sample_name:
        raise ValueError("You must specify the sample name to save out the finetune samples")

    df_raw = pd.concat([pd.read_json(fn, lines=True) for fn in data_files_raw], ignore_index=True)

    # truncate the fullText to maxTokens
    df_raw = truncate_documents(df_raw, maxTokens)

    df_raw_foreground = df_raw.loc[df_raw["isForeground"]].copy()
    df_raw_background = df_raw.loc[~df_raw["isForeground"]].copy()

    # get the dataframes representing the finetune and test partitions for foreground and bacground docs
    df_fg_finetune, df_fg_test = create_finetune_test_partitions(
        df_raw_foreground, frac_author_finetune_set, randomSeed
    )
    df_bg_finetune, df_bg_test = create_finetune_test_partitions(
        df_raw_background, frac_author_finetune_set, randomSeed
    )

    # save out the finetune and test partitions
    save_finetune_documents(df_fg_finetune, df_bg_finetune, data_dir_finetune, finetune_sample_name)
    save_test_documents(df_fg_test, df_bg_test, data_dir_test)


if __name__ == "__main__":
    entrypoint()
