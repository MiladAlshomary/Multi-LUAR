import os

import numpy as np
from absl import logging
from glob import glob
from sklearn.metrics.pairwise import cosine_similarity
import torch
from sklearn.metrics import pairwise_distances

class Similarity():

    def __init__(self, query_features, candidate_features, query_labels, candidate_labels, input_dir):
        self.query_features = torch.tensor(query_features, dtype=torch.float32)
        self.candidate_features = torch.tensor(candidate_features, dtype=torch.float32)
        self.query_labels = query_labels
        self.candidate_labels = candidate_labels
        self.dataset_path = get_dataset_path(input_dir)

    def compute_similarities(self):
        logging.info("Computing cosine similarities")
        print(len(self.query_features))
        print(len(self.query_features[0]))
        # self.psimilarities = cosine_similarity(
        #     self.query_features, self.candidate_features)

        q_list = torch.cat(
            [e.permute(1, 0, 2) for e in self.query_features], dim=0
        )  # Permute to (batch_size, num_layers, embedding_size) and concatenate
        q_list = q_list.cpu().numpy()  # Convert to numpy

        # Reshape and concatenate target embeddings
        t_list = torch.cat(
            [e.permute(1, 0, 2) for e in self.candidate_features], dim=0
        )  # Permute to (batch_size, num_layers, embedding_size) and concatenate
        t_list = t_list.cpu().numpy()  # Convert to numpy

        num_queries, num_layers, _ = q_list.shape
        num_targets, _, _ = t_list.shape

        print('=====')
        print(q_list.shape)
        print(t_list.shape)
        
        # Initialize a similarity matrix to hold the sum of cosine similarities for each query-target pair
        self.psimilarities = np.zeros((num_queries, num_targets), dtype=np.float32)

        for layer in range(num_layers):
            # Compute cosine similarity for the current layer
            layer_similarities = cosine_similarity(q_list[:, layer, :], Y=t_list[:, layer, :])
            # Add the cosine similarities of this layer to the overall similarity matrix
            self.psimilarities += layer_similarities

        # Compute pairwise distances using the averaged embeddings
        self.psimilarities = cosine_similarity(avg_queries, Y=avg_targets)


    def save_ta2_output(self, output_dir, run_id, ta1_approach):
        logging.info("Saving similarities and labels")
        HST = os.path.basename(self.dataset_path).split("_TA2")[0]

        np.save(
            os.path.join(output_dir, HST +
                         f"_TA2_query_candidate_attribution_scores_{run_id}"),
            self.psimilarities
        )

        fout = open(
            os.path.join(
                output_dir,
                HST +
                f"_TA2_query_candidate_attribution_query_labels_{run_id}.txt"
            ), "w+"
        )
        if self.query_labels[0][0] == "(":
            tuple_str = True
        else:
            tuple_str = False
        if not tuple_str:
            for label in self.query_labels:
                fout.write("('"+str(label)+"',)")
                fout.write("\n")
            fout.close()
        else:
            for label in self.query_labels:
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
            for label in self.candidate_labels:
                fout.write("('"+str(label)+"',)")
                fout.write("\n")
            fout.close()
        else:
            for label in self.candidate_labels:
                fout.write(label)
                fout.write("\n")
            fout.close()


def get_dataset_path(input_path):
    dataset_path = glob(os.path.join(input_path, "*queries*"))[0]
    return dataset_path
