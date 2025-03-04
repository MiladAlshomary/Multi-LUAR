# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import pandas as pd
import torch
import numpy as np

from src.datasets.retrieval_dataset import RetrievalDataset
from src.utilities.file_utils import Utils as utils

import re

def split_text(text, avg_words=32):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    current_chunk = []
    word_count = 0
    
    for sentence in sentences:
        words = sentence.split()
        if word_count + len(words) > avg_words and current_chunk:
            result.append(" ".join(current_chunk))
            current_chunk = []
            word_count = 0
        
        current_chunk.extend(words)
        word_count += len(words)
    
    if current_chunk:
        result.append(" ".join(current_chunk))
    
    return result


class HRS_Dataset(RetrievalDataset):
    """Torch Dataset object for the PAN Short Stories dataset
    """
    def __init__(
        self, 
        params: argparse.Namespace, 
        split: str, 
        num_sample_per_author: int, 
        is_queries=True
    ):            
        super().__init__(params, split, num_sample_per_author, is_queries)

        self.dataset_path = params.dataset_path
        self.gt_path = params.gt_path
        self.is_queries = is_queries
        
        if is_queries:
            self.file_name = params.queries_file_name
            self.author_clm_name = params.q_author_clm_name

        else:
            self.file_name = params.candidates_file_name
            self.author_clm_name = params.c_author_clm_name


        # we need this because we are returning author-labels as a tensor which should be integer not strings
        self.authorId2Int = {}
        self.load_groundtruth(os.path.join(self.dataset_path, self.gt_path))
        self.load_data(os.path.join(self.dataset_path, self.file_name))
        self.is_test = True

    def load_groundtruth(self, path):
        
        self.ground_truth_assignment = np.load(open(path + '_groundtruth.npy', 'rb'))
        self.candidate_authors = [a[2:-3] for a in  open(path + '_candidate-labels.txt').read().split('\n')][:-1]
        self.query_authors = [a[2:-3] for a in  open(path + '_query-labels.txt').read().split('\n')][:-1]                
        

    def load_data(self, filename: str):
            
        self.data = pd.read_json(filename, lines=True, nrows=self.params.sanity)
        self.data['authorID']  = self.data[self.author_clm_name].apply(lambda x : x[0])
        
        self.data = self.data.groupby('authorID').agg({
            self.text_key: lambda x: list(x)
        }).reset_index()

        # For candidate authors we map them to their corresponding correct query author ids according to the groundtruth matrix
        if not self.is_queries:
            query_authors = []
            self.c_to_q_map = {}
            for j, c_author in enumerate(self.candidate_authors):
                self.c_to_q_map[c_author] = c_author
                for i, q_author in enumerate(self.query_authors):
                    if self.ground_truth_assignment[i,j] == 1:
                        print('map {} --> {}'.format(c_author, q_author))
                        query_authors.append(c_author)
                        self.c_to_q_map[c_author] = q_author

            self.authorId2Int = {self.c_to_q_map[x]:i for i, x in enumerate(self.data['authorID'].unique())}
            self.int2AuthorId = {x[1]:x[0] for x in self.authorId2Int.items()}

            # keep only candidate authors that are the query ones
            self.data = self.data[self.data.authorID.isin(query_authors)]
        else:
            self.authorId2Int = {x:i for i, x in enumerate(self.data['authorID'].unique())}
            self.int2AuthorId = {x[1]:x[0] for x in self.authorId2Int.items()}

        
        self.num_authors = len(self.data)

    def __getitem__(
        self, 
        index: int
    ):
        
        if self.split == "test":
            author_data = self.data.iloc[index].to_dict()
            # Split each text of the author into chunks, each of a round about token_max_length tokens
            author_data[self.text_key] = [chunk for text in author_data[self.text_key] for chunk in split_text(text, avg_words=self.params.token_max_length)]
            #print(len(author_data[self.text_key]))
            #print(author_data[self.text_key][0])
            
            tokenized_episode = self.tokenizer(
                author_data[self.text_key], 
                padding="max_length", 
                truncation=True, 
                max_length=self.params.token_max_length, 
                return_tensors='pt'
            )
            data = self.reformat_tokenized_inputs(tokenized_episode)
            data = [d.reshape(1, -1, self.params.token_max_length) for d in data]

            if self.is_queries:
                author = torch.tensor([self.authorId2Int[author_data['authorID']] for _ in range(self.num_sample_per_author)])
            else:
                # Labels for candidate authors should be mapped to their correspoinding query authors if they have
                author = torch.tensor([self.authorId2Int[self.c_to_q_map[author_data['authorID']]] for _ in range(self.num_sample_per_author)])
        else:
            text = []
            
            for _ in range(self.num_sample_per_author):
                episode = self.sample_random_episode(index, is_test=self.is_test)
                text.extend(episode[self.text_key])
                    
            if self.is_queries:
                author = torch.tensor([self.authorId2Int[author_data['authorID']] for _ in range(self.num_sample_per_author)])
            else:
                author = torch.tensor([self.authorId2Int[self.c_to_q_map[author_data['authorID']]] for _ in range(self.num_sample_per_author)])
            
            data = self.tokenize_text(text)
            if self.params.use_random_windows:
                data = self.sample_random_window(data)
            
            data = [d.reshape(self.num_sample_per_author, -1, self.params.token_max_length) for d in data]

        self.mask_data_bpe(data)
        return data, author
