from glob import glob

from SIVs.siv import SIV
import os
from datasets import DatasetDict, Dataset
from absl import logging
import pandas as pd
import torch

from src.models.transformer import Transformer
from SIVs.utils import get_file_paths, load_model, load_tokenizer, tokenize, save_files

CKPT_PATH =  "/home/nv2415/LUAR/src/output/reddit_model/lightning_logs/version_2/checkpoints/epoch=19-step=255100.ckpt"

class SIV_Multilayer_Luar(SIV):
    def __init__(self, input_dir, query_identifier, candidate_identifier, params, language="en"):
        super().__init__(input_dir, query_identifier, candidate_identifier, language)
        self.params = params
        self.batch_size = 16
        self.author_level = True
        self.text_key = "fullText"
        self.token_max_length = self.params.token_max_length
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

    def load_model(self, args):
        print("Loading MultiLayer LUAR")
        
        # Initialize Transformer model
        self.model = Transformer(args)

        # Load the checkpoint
        checkpoint = torch.load(CKPT_PATH, map_location=torch.device("cpu"))
        self.model.load_state_dict(checkpoint["state_dict"], strict=False)

        if torch.cuda.is_available():
            print("Using CUDA")
            self.model.cuda()

        # Load tokenizer
        self.tokenizer = load_tokenizer(self.language, os.path.join(os.getcwd()))

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
            chunk = data.iloc[i : i + batch_size]
            text = [tokenize(t, tokenizer, self.token_max_length) for t in chunk[self.text_key]]
            # print(len(chunk))
            num_samples_per_author = text[0][0].shape[0]

            input_ids = torch.cat([elem[0] for elem in text], dim=0)
            attention_mask = torch.cat([elem[1] for elem in text], dim=0)
            
            if torch.cuda.is_available():
                input_ids = input_ids.to("cuda")
                attention_mask = attention_mask.to("cuda")

            with torch.no_grad():
                input_ids = input_ids.unsqueeze(1)
                attention_mask = attention_mask.unsqueeze(1)
                input_ids = input_ids.reshape((-1, 1, num_samples_per_author, self.token_max_length))
                attention_mask = attention_mask.reshape((-1, 1, num_samples_per_author, self.token_max_length))
                # print(input_ids.shape)
                output, _ = self.model.get_episode_embeddings((input_ids, attention_mask))

            all_identifiers.extend(chunk[identifier])
            all_outputs.append(output.cpu().numpy().tolist())

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