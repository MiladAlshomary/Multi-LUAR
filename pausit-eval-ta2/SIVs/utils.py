""" General Utilities """

import os
from glob import glob
import yaml
import numpy as np
import torch
from datasets import DatasetDict
from transformers import AutoModel, AutoTokenizer
from absl import logging


SBERT_PATH = "/mnt/swordfish-pool2/nikhil/LUAR/pretrained_weights/paraphrase-distilroberta-base-v1"
LUAR_PATH =  "/mnt/swordfish-pool2/nikhil/LUAR/pretrained_weights/LUAR-MUD"


def get_file_paths(input_path):
    queries_files = glob(os.path.join(input_path, "*queries*"))
    candidates_files = glob(os.path.join(input_path, "*candidates*"))

    queries_fname = queries_files[0]
    candidates_fname = candidates_files[0]
    return queries_fname, candidates_fname


def load_model(model_path, luar=False, load_from_artifacts=False, artifacts_dir=None, language="en"):
    if luar:
        if language == "ru":
            print('Russian')
            model = AutoModel.from_pretrained(os.path.join(model_path, "rrivera1849/LUAR-RU"), trust_remote_code=True)
        else:
            model = AutoModel.from_pretrained(os.path.join(LUAR_PATH), trust_remote_code=True)
    else:
        model = AutoModel.from_pretrained(os.path.join(SBERT_PATH))
        if load_from_artifacts:
            model.load_state_dict(torch.load(glob(os.path.join(artifacts_dir, "*SBERT*"))[0]))
            model.eval()


    return model

def load_tokenizer(language="en", model_path=None):
    if language == "ru":
        print('Russian tokenizer')
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "rrivera1849/LUAR-RU"), trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(SBERT_PATH)
    return tokenizer

def tokenize(text, tokenizer, token_max_length):
    text = text if isinstance(text, list) else [text]

    tokenized_data = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=token_max_length,
        return_tensors="pt"
    )

    return tokenized_data["input_ids"], tokenized_data["attention_mask"]

def mean_pooling(token_embeddings, attention_mask):
    # https://huggingface.co/sentence-transformers/paraphrase-distilroberta-base-v1
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def save_files(ta1_approach, queries, candidates, output_path, run_id):
    dataset_dict = DatasetDict()
    dataset_dict["queries"] = queries
    dataset_dict["candidates"] = candidates

    save_fname = ta1_approach + "_TA1_features_" + run_id
    save_fname = os.path.basename(save_fname)
    dataset_dict.save_to_disk(os.path.join(output_path, save_fname))

    metric = {'Metric Name': 'cosine'}
    save_path = os.path.join(output_path, 'performer-config.yaml')
    with open(save_path, 'w') as outfile:
        yaml.dump(metric, outfile)


def get_features(input_dir, ta1_approach, query_identifier, candidate_identifier):
    logging.info("Loading TA1 features")
    dataset_path = get_dataset_path(input_dir, ta1_approach)
    dataset = DatasetDict.load_from_disk(dataset_path)

    query_features = dataset["queries"]["features"]
    query_labels = dataset["queries"][query_identifier]
    candidate_features = dataset["candidates"]["features"]
    candidate_labels = dataset["candidates"][candidate_identifier]
    return query_features, candidate_features, query_labels, candidate_labels

def get_dataset_path(input_path, ta1_approach):
    try:
        dataset_path = glob(os.path.join(input_path, f"*{ta1_approach}_TA1_features*"))[0]
        print(dataset_path)
    except:
        print("TA1 features for the given approach not found. Use --generate-features to generate the features")
        exit(1)
    return dataset_path

def dump_ta2_output(dataset_path, output_dir, run_id, scores, query_labels, candidate_labels):

    logging.info("Saving similarities and labels")
    HST = os.path.basename(dataset_path).split("_TA2")[0]
    np.save(
        os.path.join(output_dir, HST +
                        f"_TA2_query_candidate_attribution_scores_{run_id}"),
        scores
    )
    fout = open(
        os.path.join(
            output_dir,
            HST +
            f"_TA2_query_candidate_attribution_query_labels_{run_id}.txt"
        ), "w+"
    )
    if query_labels[0][0] == "(":
        tuple_str = True
    else:
        tuple_str = False
    if not tuple_str:
        for label in query_labels:
            fout.write("('"+str(label)+"',)")
            fout.write("\n")
        fout.close()
    else:
        for label in query_labels:
            fout.write(label)
            fout.write("\n")
        fout.close()

    fout = open(
        os.path.join(
            output_dir,
            HST +
            f"_TA2_query_candidate_attribution_candidate_labels_{run_id}.txt"
        ), "w+"
    )

    if not tuple_str:
        for label in candidate_labels:
            fout.write("('"+str(label)+"',)")
            fout.write("\n")
        fout.close()
    else:
        for label in candidate_labels:
            fout.write(label)
            fout.write("\n")
        fout.close()