from glob import glob

from SIVs.siv import SIV
import os
from datasets import DatasetDict, Dataset
from absl import logging
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize


from src.models.transformer import Transformer
from SIVs.utils import get_file_paths, load_model, load_tokenizer, tokenize, save_files

CKPT_PATH =  "/mnt/swordfish-pool2/nikhil/LUAR/src/output/reddit_model/lightning_logs/version_2/checkpoints/epoch=19-step=255100.ckpt"

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

    def split_text_to_samples(self, text, tokenizer, min_tokens, max_samples):
        """
        Splits a single text into samples with at least `min_tokens` while keeping sentences intact.
        """
        if isinstance(text, list):
        # Concatenate list of strings into a single string
            text = " ".join(text)

        sentences = sent_tokenize(text)  # Split into sentences
        samples = []
        current_sample = []
        current_token_count = 0

        for sentence in sentences:
            # Tokenize the sentence and count tokens
            tokenized_sentence = tokenizer(sentence, truncation=False)["input_ids"]
            token_count = len(tokenized_sentence)

            # Check if adding this sentence will exceed the minimum token limit
            if current_token_count + token_count >= min_tokens:
                # Finalize the current sample and start a new one
                current_sample.extend(tokenized_sentence)
                samples.append(current_sample)
                current_sample = []
                current_token_count = 0

                # Stop if we reach the max_samples limit
                if len(samples) == max_samples:
                    break
            else:
                # Add sentence to the current sample
                current_sample.extend(tokenized_sentence)
                current_token_count += token_count

        # # Add the last sample if it meets the token limit
        # if current_sample and len(samples) < max_samples:
        #     samples.append(current_sample)

        return samples[:max_samples]  # Ensure we don't exceed the max number of samples

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
            
        length=len(data) if len(data) < 100 else 100
        for i in range(0, len(data), batch_size):
            print(i)
            chunk = data.iloc[i : i + batch_size]

            text_list = chunk[self.text_key]

            # Process each text individually without mixing
            all_samples = []
            for text in text_list.values:
                samples = self.split_text_to_samples(text, tokenizer, min_tokens=self.params.token_max_length, max_samples=16)
                all_samples.extend(samples)

                # Stop if we've reached 16 samples
                # if len(all_samples) >= 16:
                #     break

            # Ensure we have exactly 16 samples by padding with zeros if needed
            # while len(all_samples) < 16:
            #     all_samples.append([0] * self.token_max_length)

            # Prepare inputs for the model
            input_ids = torch.tensor([sample[:self.token_max_length] for sample in all_samples])
            attention_mask = torch.tensor([[1] * len(sample[:self.token_max_length]) for sample in all_samples])

            num_samples_per_author = input_ids.shape[0]
            if torch.cuda.is_available():
                input_ids = input_ids.to("cuda")
                attention_mask = attention_mask.to("cuda")

            with torch.no_grad():
                input_ids = input_ids.unsqueeze(1)
                attention_mask = attention_mask.unsqueeze(1)
                input_ids = input_ids.reshape((-1, 1, num_samples_per_author, self.token_max_length))
                attention_mask = attention_mask.reshape((-1, 1, num_samples_per_author, self.token_max_length))
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
