from glob import glob

from SIVs.siv import SIV
import os
from datasets import DatasetDict, Dataset
from absl import logging
import pandas as pd
import numpy as np
import torch

from SIVs.utils import get_file_paths, load_model, load_tokenizer, tokenize, save_files
from SIVs.datadreamer_lora.luar_utils import load_luar_as_sentence_transformer
from peft import PeftModel

class SIV_DataDreamer_LoRA(SIV):

    def __init__(self, input_dir, query_identifier, candidate_identifier, language="en"):
        super().__init__(input_dir, query_identifier, candidate_identifier, language)
        self.batch_size = 16
        self.author_level = True
        self.text_key = "fullText"
        self.token_max_length = 512
        self.document_batch_size = 32

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_author_level(self, author_level):
        self.author_level = author_level
    
    def set_token_max_length(self, token_max_length):
        self.token_max_length = token_max_length
    
    def set_text_key(self, text_key):
        self.text_key = text_key

    def get_features(self):
        logging.info("Loading TA1 features")
        self.dataset_path = get_dataset_path(self.input_dir)
        dataset = DatasetDict.load_from_disk(self.dataset_path)

        self.query_features = dataset["queries"]["features"]
        self.query_labels = dataset["queries"][self.query_identifier]
        self.candidate_features = dataset["candidates"]["features"]
        self.candidate_labels = dataset["candidates"][self.candidate_identifier]
        return self.query_features, self.candidate_features, self.query_labels, self.candidate_labels

    def load_model(self):
        logging.info("Loading LUAR")
        if self.language == "ru":
            base_model = load_luar_as_sentence_transformer(os.path.join(os.getcwd(), "rrivera1849/LUAR-RU"))
        else:
            base_model = load_luar_as_sentence_transformer(os.path.join(os.getcwd(), "rrivera1849/LUAR-MUD"))
        base_model.max_seq_length = self.token_max_length
        base_model.tokenizer.pad_token = base_model.tokenizer.pad_token or base_model.tokenizer.eos_token

        # Apply LoRA adapter
        if self.language == "ru":
            peft_model_dir = os.path.join(os.getcwd(), "SIVs/datadreamer_lora/russian_checkpoint/")
        else:
            peft_model_dir = os.path.join(os.getcwd(), "SIVs/datadreamer_lora/checkpoint/")
        self.model = PeftModel.from_pretrained(base_model, model_id=peft_model_dir)

        self.tokenizer = load_tokenizer()

        if torch.cuda.is_available():
            logging.info("Using CUDA")
            self.model.half().cuda()

    def extract_embeddings(self, model, tokenizer, data_fname):
        data = pd.read_json(data_fname, lines=True)
        batch_size = self.batch_size

        if self.author_level:
            identifier = "authorIDs" if "queries" in data_fname else "authorSetIDs"
            data[identifier] = data[identifier].apply(lambda x: str(tuple(x)))
            data = data[[identifier, self.text_key]].groupby(identifier).fullText.apply(list).reset_index()

            batch_size = 1
            logging.info("Setting batch size to 1 for author level embeddings with LUAR.")

        else:
            identifier = "documentID"

        all_identifiers, all_outputs = [], []

        for i in range(0, len(data), batch_size):
            if i % 10 == 0:
                print(f"Progress: {int((i/len(data))*100)}%")
            chunk = data.iloc[i:i+batch_size]
            raw_text = list(chunk[self.text_key])[0]
            output = model.encode(raw_text, show_progress_bar=False)
            all_identifiers.extend(chunk[identifier])
            all_outputs.extend([np.mean(output, axis=0).tolist()])

        dataset = Dataset.from_dict({
            identifier: all_identifiers,
            "features": all_outputs,
        })

        return dataset

    def generate_sivs(self, input_dir, output_dir, run_id, ta1_approach):
        queries_fname, candidates_fname = get_file_paths(input_dir)
        logging.info("Extracting Query Embeddings")
        queries = self.extract_embeddings(self.model, self.tokenizer, queries_fname)
        logging.info("Extracting Candidate Embeddings")
        candidates = self.extract_embeddings(self.model, self.tokenizer, candidates_fname)
        self.store_sivs(ta1_approach, queries, candidates, output_dir, run_id)

    def store_sivs(self, queries_fname, queries, candidates, output_dir, run_id):
        logging.info("Saving Dataset and Cosine as Distance Metric")
        save_files(queries_fname, queries, candidates, output_dir, run_id)


def get_dataset_path(input_path):
    dataset_path = glob(os.path.join(input_path, "*TA2_queries*"))[0]
    return dataset_path
