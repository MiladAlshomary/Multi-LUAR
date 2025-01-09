import os
import time
import sys
from glob import glob
from pathlib import Path
import numpy as np
from absl.flags import argparse_flags
from absl import app

from SIVs.utils import dump_ta2_output, get_features, get_file_paths, get_dataset_path
from author_attribution.similarity import Similarity
from author_attribution.longform_attr import apply_srs
from SIVs.siv_baseline_sbert import SIV_Baseline_SBert
from SIVs.siv_baseline_luar_cpy import SIV_Baseline_Luar
from SIVs.siv_multilayer_luar import SIV_Multilayer_Luar
from SIVs.siv_datadreamer_lora import SIV_DataDreamer_LoRA
from SIVs.siv_st import SIV_ST
from eval.hiatus.attribution.experiment import Experiment
from src.arguments import create_argument_parser

def get_dataset_path(input_path):
    dataset_path = glob(os.path.join(input_path, "*queries*"))[0]
    return dataset_path

def run_te_eval(run_id, ground_truth_dir, input_dir, output_dir):
    """Runs T&E eval and returns the summary metrics as an array"""
    dataset_path = get_dataset_path(input_dir)
    HST = os.path.basename(dataset_path).split("_TA2")[0]
    te_config = {"name": "experiment",
                 "ta2": {"scores": os.path.join(output_dir, HST + f"_TA2_query_candidate_attribution_scores_{run_id}.npy"),
                         "scores_candidate_labels": os.path.join(output_dir, HST + f"_TA2_query_candidate_attribution_candidate_labels_{run_id}.txt"),
                         "scores_query_labels": os.path.join(output_dir, HST + f"_TA2_query_candidate_attribution_query_labels_{run_id}.txt"),
                         "ground_truth":  os.path.join(ground_truth_dir , HST + "_TA2_groundtruth.npy"),
                         "ground_truth_candidate_labels":  os.path.join(ground_truth_dir , HST + "_TA2_candidate-labels.txt"),
                         "ground_truth_query_labels":  os.path.join(ground_truth_dir , HST + "_TA2_query-labels.txt")}}

    experiment = Experiment(config=te_config)
    _, summary_metrics, _ = experiment.compute_metrics()
    return summary_metrics

def parse_flags(argv):
    additional_args = {
        ("--input-dir",): {"type": str, "required": True, "help": "Directory where the queries and candidates are stored"},
        ("--output-dir",): {"type": str, "required": True, "help": "Directory where the query and candidate attributions will be stored"},
        ("--ground-truth-dir",): {"type": str, "required": True, "help": "Directory where the ground truth query and candidate attributions are stored"},
        ("--run-id",): {"type": str, "required": True, "help": "Run identifier"},
        ("--query-identifier",): {"type": str, "default": "authorIDs", "help": "Identifier for query embeddings"},
        ("--candidate-identifier",): {"type": str, "default": "authorSetIDs", "help": "Identifier for candidate embeddings"},
        ("--debug",): {"action": "store_true", "help": "Debug mode, each epoch is batch_size size"},
        ("-l", "--language"): {"type": str, "default": "eng", "choices": ["eng", "ru"], "help": "Language"},
        ("-ta1", "--ta1-approach"): {"type": str, "choices": ['datadreamer_lora', 'baseline_sbert', 'baseline_luar', 'multilayer_luar', 'gram2vec', 'russian_st'], "default": "none", "help": "TA1 approach"},
        ("-g", "--generate-features"): {"action": "store_true", "help": "Generate TA1 features"},
<<<<<<< Updated upstream
        ("-ta2", "--ta2-approach"): {"type": str, "choices": ['srs', 'baseline', 'mean_cosine'], "required": True, "help": "TA2 approach"},
=======
        ("-ta2", "--ta2-approach"): {"type": str, "choices": ['srs', 'multilayer_luar_nikhil', 'baseline', 'mean_cosine'], "required": True, "help": "TA2 approach"},
>>>>>>> Stashed changes
        ("-m", "--model-path"): {"type": str, "default": None, "help": "Path to the model for Russian Sentence Transformer"},
        ("-ckpt", "--checkpoint-path"): {"type": str, "default": None, "help": "Path to load LUAR model checkpoint"},
    }

    parser = create_argument_parser(additional_args)
    return parser.parse_args(argv[1:])

def main(args):
    output_dir = os.path.join(args.output_dir, f"TA2_output_{args.ta1_approach}_{args.ta2_approach}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    START = time.time()
    language = args.language
    if args.generate_features:
        if args.ta1_approach == 'baseline_sbert':
            siv = SIV_Baseline_SBert(args.input_dir, args.query_identifier, args.candidate_identifier)
        if args.ta1_approach == 'baseline_luar':
            siv = SIV_Baseline_Luar(args.input_dir, args.query_identifier, args.candidate_identifier, language=language)
        elif args.ta1_approach == 'multilayer_luar':
            siv = SIV_Multilayer_Luar(args.input_dir, args.query_identifier, args.candidate_identifier, args, language=language)
        elif args.ta1_approach == 'datadreamer_lora':
            siv = SIV_DataDreamer_LoRA(args.input_dir, args.query_identifier, args.candidate_identifier, language=language)
        elif args.ta1_approach == 'russian_st':
            siv = SIV_ST(args.input_dir, args.query_identifier, args.candidate_identifier, model_path=args.model_path)
        elif args.ta1_approach == 'none':
            print("cannot use generate features with 'none' as the TA1 approach")
            exit(1)

        if args.ta1_approach == 'multilayer_luar':
            siv.load_model(args)
        else:
            siv.load_model()

        if args.ta2_approach == 'mean_cosine':
            query_path, candidate_path = get_file_paths(args.input_dir)
            scores, query_labels, candidate_labels = siv.get_direct_scores(query_path, candidate_path)
            dump_ta2_output(get_dataset_path(args.input_dir), output_dir, args.run_id, scores, query_labels, candidate_labels)
            te_results = run_te_eval(args.run_id, args.ground_truth_dir, args.input_dir, output_dir)
            print(te_results)
        else:
            siv.generate_sivs(args.input_dir, args.output_dir, args.run_id, args.ta1_approach)
            query_features, candidate_features, query_labels, candidate_labels = get_features(args.output_dir, args.ta1_approach, args.query_identifier, args.candidate_identifier)


        if args.ta2_approach == 'baseline':
            sim = Similarity(query_features, candidate_features, query_labels, candidate_labels, args.input_dir)
            if args.ta1_approach == 'multilayer_luar':
                sim.compute_multilayer_similarities()
            else:
                sim.compute_similarities()
            sim.save_ta2_output(output_dir, args.run_id, args.ta1_approach)
            te_results = run_te_eval(args.run_id, args.ground_truth_dir, args.input_dir, output_dir)
            print(te_results)

<<<<<<< Updated upstream
=======
        if args.ta2_approach == 'multilayer_luar_nikhil':
            sim = MultLuarSimilarity(query_features, candidate_features, query_labels, candidate_labels, args.input_dir)
            sim.compute_similarities()
            sim.save_ta2_output(output_dir, args.run_id, args.ta1_approach)
            te_results = run_te_eval(args.run_id, args.ground_truth_dir, args.input_dir, output_dir)
            print(te_results)
        
>>>>>>> Stashed changes
    else:
        if args.ta2_approach == 'baseline':
            query_features, candidate_features, query_labels, candidate_labels = get_features(args.input_dir, args.ta1_approach, args.query_identifier, args.candidate_identifier)
            sim = Similarity(query_features, candidate_features, query_labels, candidate_labels, args.input_dir)
            sim.compute_similarities()
            sim.save_ta2_output(output_dir, args.run_id, args.ta1_approach)
            te_results = run_te_eval(args.run_id, args.ground_truth_dir, args.input_dir, output_dir)
            print(te_results)
        elif args.ta2_approach == 'srs':
            apply_srs(args.input_dir, output_dir, args.run_id)
            te_results = run_te_eval(args.run_id, args.ground_truth_dir, args.input_dir, output_dir)
            print(te_results)

<<<<<<< Updated upstream
=======
        elif args.ta2_approach == 'multilayer_luar_nikhil':
            query_features, candidate_features, query_labels, candidate_labels = get_features(args.input_dir, args.ta1_approach, args.query_identifier, args.candidate_identifier)
            sim = MultLuarSimilarity(query_features, candidate_features, query_labels, candidate_labels, args.input_dir)
            sim.compute_similarities()
            sim.save_ta2_output(output_dir, args.run_id, args.ta1_approach)
            te_results = run_te_eval(args.run_id, args.ground_truth_dir, args.input_dir, output_dir)
            print(te_results)

>>>>>>> Stashed changes
        else:
            raise ValueError(f"TA2 approach {args.ta2_approach} not recognized")
    print(f"==> Total duration (s): {time.time()-START}\n")

if __name__ == "__main__":
    args = parse_flags(sys.argv)
    print(args)
    # Call the main function directly with parsed arguments
    main(args)
