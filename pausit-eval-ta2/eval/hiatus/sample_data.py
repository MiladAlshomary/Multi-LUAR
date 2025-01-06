"""Data sampling code CLI."""

import datetime
import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from eval.hiatus.data.combine_cross_genre import combine_crossGenre_samples
from eval.hiatus.data.machine_authors_sampling import combine_human_machine_documents
from eval.hiatus.data.process import TA1_COLUMNS, generateTestSamples, getSilverLabeledForegroundAuthors
from eval.hiatus.data.save import saveCombinedTA1Samples, saveTestSamples


LOG = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="data", config_name="sampling")
def entrypoint(config: DictConfig):
    """Define data sampling CLI utility."""
    config = OmegaConf.to_object(config)  # type: ignore

    timestamp = datetime.datetime.utcnow().isoformat()
    name = config.get("name", timestamp)

    output_path_ = config.get("output_path")
    if output_path_ is None:
        raise ValueError("You must set 'Output Path' in the config.")
    output_path = Path(output_path_)
    if not (output_path.is_dir() and output_path.exists()):
        raise ValueError(f"Output path {output_path.absolute()} should be an existing directory.")
    if list(output_path.glob("*")):
        LOG.warning(f"Output path {output_path} not empty. Samples may be overwritten.")

    data_dir = config.get("data_dir")
    if data_dir is None:
        raise ValueError("You must set 'data_dir' in the config.")
    # Get all data files except the combined TA1 file
    data_files = sorted(Path(data_dir).rglob("*.jsonl"))
    if not data_files:
        raise FileNotFoundError(f"No .jsonl files found in data directory {data_dir}")
    else:
        LOG.info(f"Found the following data files to load: {[f.relative_to(data_dir) for f in data_files]}")

    df_human = pd.concat([pd.read_json(fn, lines=True) for fn in data_files], ignore_index=True)
    silver_foreground = config.get("silver_foreground", False)
    silver_foreground_threshold = config.get("silver_foreground_threshold", 12)

    if silver_foreground:
        df_human, _, _ = getSilverLabeledForegroundAuthors(df_human, threshold=silver_foreground_threshold)

    use_machine_authors = config.get("use_machine_authors")

    metadata = {
        "Name": name,
        "Data Files": [filepath.resolve().as_posix() for filepath in data_files],
        "Silver Foreground": silver_foreground,
        "Timestamp": timestamp,
        "Machine Authors": use_machine_authors,
    }
    if silver_foreground:
        metadata["Silver Foreground Threshold"] = silver_foreground_threshold

    list_df = []

    if use_machine_authors:
        data_dir_machine = config.get("data_dir_machine")
        data_files_machine = sorted(Path(data_dir_machine).rglob("*.jsonl"))
        if not data_files_machine:
            raise FileNotFoundError(f"No .jsonl files found in directory {data_dir_machine}")
        else:
            LOG.info(f"Machine gen. files to load: {[f.relative_to(data_dir_machine) for f in data_files_machine]}")

        list_frac_fg = config.get("fraction_foreground_authors")

        df_machine = pd.concat([pd.read_json(fn, lines=True) for fn in data_files_machine], ignore_index=True)

        save_summary = config.get("save_summary")
        fn_summary = config.get("fn_summary")

        if save_summary and not fn_summary:
            raise ValueError("You must specify the summary filename if you wish to save summary")

        list_df = combine_human_machine_documents(
            df_human, df_machine, list_frac_fg, config["sampling"]["randomSeed"], save_summary, fn_summary, LOG
        )

    else:
        LOG.info("No machine generated files used for this run")
        list_df.append(df_human)
        # Setting the list_frac_fg to an invalid fraction when no machine authors are used
        list_frac_fg = [-1.0]

    for df, frac_fg in zip(list_df, list_frac_fg):
        if use_machine_authors:
            save_path = Path.joinpath(output_path, f"MA_FGSampling-{frac_fg}")
            save_name = name + f"_MA_FGSampling-{frac_fg}"
            # Updating the name in the metadata as well
            metadata["Name"] = save_name
            metadata["Machine Authors Sampling Fraction"] = frac_fg
        else:
            save_path = output_path
            save_name = name

        if config["sampling"]["mode"] == "crossGenre" and config["sampling"]["queryGenre"] is None:
            # A copy of the config is necessary while generating samples in a loop (esp. when sampling machine docs)
            config_copy = config.copy()
            LOG.info("Generating combined crossGenre sample out of multiple simple crossGenre samples")
            genres = df["collectionNum"].unique()
            samples = []
            for genre in sorted(genres):
                config_copy["sampling"]["queryGenre"] = genre
                test_samples = generateTestSamples(df, metadata=metadata, **config_copy["sampling"])
                samples.append(test_samples)
            test_samples = combine_crossGenre_samples(
                samples, config_copy["sampling"]["randomSeed"], config_copy["sampling"]["nMaxDocsPerHaystackSet"]
            )
        elif config["sampling"]["mode"] == "ta1All":
            LOG.info("Generating single TA1 input document file out of all available documents")
            test_sample = df.loc[:, TA1_COLUMNS].copy()
            saveCombinedTA1Samples(save_path, save_name, test_sample)
            return
        else:
            config_copy = config.copy()
            test_samples = generateTestSamples(df, metadata=metadata, **config_copy["sampling"])

        sample_coordination = saveTestSamples(save_path, save_name, test_samples)
        LOG.debug(sample_coordination)


if __name__ == "__main__":
    entrypoint()
