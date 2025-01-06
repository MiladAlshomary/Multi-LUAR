from SIVs.siv import SIV
import os
from datasets import Dataset
from absl import logging
import pandas as pd
import torch
import numpy as np

from SIVs.utils import get_file_paths, load_model, load_tokenizer, tokenize, save_files, mean_pooling


class SIV_Baseline_SBert(SIV):

    def __init__(self, input_dir, query_identifier, candidate_identifier, language="en"):
        super().__init__(input_dir, query_identifier, candidate_identifier, language)
        self.batch_size = 16
        self.author_level = False
        if self.query_identifier == "authorIDs":
            self.author_level = True
        self.text_key = "fullText"
        self.token_max_length = 512
        self.document_batch_size=64
        self.extreme_doc_num=100


    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_author_level(self, author_level):
        self.author_level = author_level
    
    def set_token_max_length(self, token_max_length):
        self.token_max_length = token_max_length
    
    def set_text_key(self, text_key):
        self.text_key = text_key

    def load_model(self):
        logging.info("Loading SBERT")
        path = os.getcwd()

        self.model = load_model(path)

        self.tokenizer = load_tokenizer()

        if torch.cuda.is_available():
            logging.info("Using CUDA")
            self.model.half().cuda()


    def extract_embeddings(self, model, tokenizer, data_fname):
        data = pd.read_json(data_fname, lines=True)
        batch_size = self.batch_size

        if self.author_level:
            identifier = "authorIDs" if "queries" in data_fname else "authorSetIDs"
            data[identifier] = data[identifier].apply(lambda x: x[0])
            data = data[[identifier, self.text_key]].groupby(identifier).fullText.apply(list).reset_index()
        else:
            identifier = "documentID"

        all_identifiers, all_outputs = [], []

        for i in range(0, len(data), batch_size):
            chunk = data.iloc[i:i+batch_size]
            text = [tokenize(t, tokenizer, self.token_max_length) for t in chunk[self.text_key]]

            # variable doc length for sbert
            num_samples_per_author = [t[0].shape[0] for t in text]
            num_samples_per_author = list(np.cumsum(num_samples_per_author))[:-1]

            input_ids = torch.cat([elem[0] for elem in text], dim=0)
            attention_mask = torch.cat([elem[1] for elem in text], dim=0)

            extreme_num_docs = len(input_ids) > self.extreme_doc_num

            if torch.cuda.is_available():
                input_ids = input_ids.to("cuda")
                attention_mask = attention_mask.to("cuda")

            if extreme_num_docs:
                outputs = []
                input_ids_split = torch.split(input_ids, self.document_batch_size, 0)
                attention_mask_split = torch.split(attention_mask, self.document_batch_size, 0) 
                
            with torch.no_grad():
                if extreme_num_docs:
                    for i, a in zip(input_ids_split, attention_mask_split):
                        output = model(input_ids=i, attention_mask=a)[0]
                        outputs.append(output)
                    outputs = torch.cat(outputs, 0)
                    outputs = mean_pooling(outputs, attention_mask)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)[0]
                    outputs = mean_pooling(outputs, attention_mask)
                outputs = torch.tensor_split(outputs, num_samples_per_author) # split into variable docs per author
                outputs = [torch.mean(author, 0) for author in outputs]
                outputs = torch.stack(list(outputs), dim=0)
            
            all_identifiers.extend(chunk[identifier])
            all_outputs.extend(outputs.cpu().numpy().tolist())


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
