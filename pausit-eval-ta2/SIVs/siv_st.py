from SIVs.siv import SIV
import os
from datasets import Dataset
from absl import logging
import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from SIVs.utils import get_file_paths, save_files, mean_pooling


class SIV_ST(SIV):

    def __init__(self, input_dir, query_identifier, candidate_identifier, model_path=None):
        super().__init__(input_dir, query_identifier, candidate_identifier, "ru")
        self.batch_size = 16
        self.author_level = False
        if self.query_identifier == "authorIDs":
            self.author_level = True
        self.text_key = "fullText"
        self.token_max_length = 512
        self.document_batch_size=64
        self.extreme_doc_num=100
        self.model_path = model_path

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_author_level(self, author_level):
        self.author_level = author_level
    
    def set_token_max_length(self, token_max_length):
        self.token_max_length = token_max_length
    
    def set_text_key(self, text_key):
        self.text_key = text_key

    def load_model(self):
        logging.info("Loading Russian Sentence Transformer")
        if not self.model_path:
            path = "Tochka-AI/ruRoPEBert-e5-base-2k"
            # path = os.getcwd()
            # path = os.path.join(path, "SIVs/Russian/checkpoint/")
        else:
            print('model path:', self.model_path)
            path = self.model_path
        self.model = SentenceTransformer(path)

        if torch.cuda.is_available():
            logging.info("Using CUDA")
            self.model.cuda()

    def extract_embeddings(self, data_fname):
        data = pd.read_json(data_fname, lines=True)

        if self.author_level:
            identifier = "authorIDs" if "queries" in data_fname else "authorSetIDs"
            data[identifier] = data[identifier].apply(lambda x: x[0])
            data = data[[identifier, self.text_key]].groupby(identifier).fullText.apply(list).reset_index()
        else:
            identifier = "documentID"

        all_identifiers, all_outputs = [], []
        for i,r in data.iterrows():
            author_texts = r[self.text_key]
   
            with torch.no_grad():
                outputs = self.model.encode(author_texts, convert_to_tensor=True, show_progress_bar=False)
                outputs = outputs.cpu().numpy()
                output = np.mean(outputs, axis=0)
            
            all_identifiers.append(r[identifier])
            all_outputs.append(output)
            # all_identifiers.extend([r[identifier] for _ in range(len(author_texts))])
            # all_outputs.extend(outputs.tolist())


        dataset = Dataset.from_dict({
            identifier: all_identifiers,
            "features": all_outputs,
        })
        return dataset

    def _get_score(self, query_author_embeddings, candidate_author_embeddings):
        all_cosines = cosine_similarity(np.array(query_author_embeddings), np.array(candidate_author_embeddings))
        return np.mean(all_cosines)

    def get_direct_scores(self, queries_fname, candidates_fname):
        ## This is only used for TA2
        queries = pd.read_json(queries_fname, lines=True)
        candidates = pd.read_json(candidates_fname, lines=True)
        query_identifier = "authorIDs"
        candidate_identifier = "authorSetIDs"
        query_embeddings = {}
        candidate_embeddings = {}
        queries[query_identifier] = queries[query_identifier].apply(lambda x: x[0])
        candidates[candidate_identifier] = candidates[candidate_identifier].apply(lambda x: x[0])
        queries = queries[[query_identifier, 'fullText']].groupby(query_identifier).fullText.apply(list).reset_index()
        candidates = candidates[[candidate_identifier, 'fullText']].groupby(candidate_identifier).fullText.apply(list).reset_index()

        print("Extracting Query Embeddings")
        for i,r in queries.iterrows():
            query_embeddings[r[query_identifier]] = []
            with torch.no_grad():
                for j in range(0, len(r["fullText"]), 4):
                    batch_doc = r["fullText"][j:j+4]
                    output = self.model.encode(batch_doc, convert_to_tensor=True, show_progress_bar=False)
                    output = output.cpu().numpy()
                    query_embeddings[r[query_identifier]].extend(output.tolist())
        print("Extracting Candidate Embeddings")
        for i,r in candidates.iterrows():
            candidate_embeddings[r[candidate_identifier]] = []
            with torch.no_grad():
                for j in range(0, len(r["fullText"]), 4):
                    batch_doc = r["fullText"][j:j+4]
                    output = self.model.encode(batch_doc, convert_to_tensor=True, show_progress_bar=False)
                    output = output.cpu().numpy()
                    candidate_embeddings[r[candidate_identifier]].extend(output.tolist())
        candidate_labels = list(candidate_embeddings.keys())
        query_labels = list(query_embeddings.keys())
        print("Calculating Scores")
        scores = [[self._get_score(query_embeddings[q], candidate_embeddings[c]) for c in candidate_labels] for q in query_labels]
        scores = np.array(scores)
        return scores, query_labels, candidate_labels


    def generate_sivs(self, input_dir, output_dir, run_id, ta1_approach):
        queries_fname, candidates_fname = get_file_paths(input_dir)
        logging.info("Extracting Query Embeddings")
        queries = self.extract_embeddings(queries_fname)
        logging.info("Extracting Candidate Embeddings")
        candidates = self.extract_embeddings(candidates_fname)
        self.store_sivs(ta1_approach, queries, candidates, output_dir, run_id)

    def store_sivs(self, queries_fname, queries, candidates, output_dir, run_id):
        logging.info("Saving Dataset and Cosine as Distance Metric")
        save_files(queries_fname, queries, candidates, output_dir, run_id)
