from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset, Dataset
import numpy as np
from einops import rearrange, reduce, repeat
import torch
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.metrics import det_curve, precision_recall_fscore_support, roc_auc_score, roc_curve
from scipy.stats import zscore

from matplotlib import pyplot as plt
import math
import pandas as pd

def get_luar_embeddings(sentences, model, tokenizer, max_length=128, batch_size = 8, is_multi_luar=False):
    max_length = max_length
    episode_length = int(max_length/32)
    num_batches = int(len(sentences)/batch_size)
    sentences_embeddings = []
    with torch.no_grad():
        for batch_texts in np.array_split(sentences, num_batches):
            tokenized_text = tokenizer(
                batch_texts.tolist(), 
                max_length=max_length,
                padding="max_length", 
                truncation=True,
                return_tensors="pt"
            )
            #print(tokenized_text['input_ids'].shape)
            #print(tokenized_text['input_ids'].shape)
            tokenized_text["input_ids"] = tokenized_text["input_ids"].reshape(batch_size, episode_length, -1)
            #print(tokenized_text['input_ids'].shape)
            #print(tokenized_text['input_ids'][0])
            #print(tokenized_text['input_ids'][1])
            tokenized_text["attention_mask"] = tokenized_text["attention_mask"].reshape(batch_size, episode_length, -1)
            #print(tokenized_text["input_ids"].size())       # torch.Size([batch_size, episode_length, max_length])
            #print(tokenized_text["attention_mask"].size())  # torch.Size([batch_size, episode_length, max_length])
        
            out = model(**tokenized_text)
            if is_multi_luar:
                out = rearrange(out, 'l b d -> b l d')
            sentences_embeddings.extend(out)

    return sentences_embeddings


def compute_similarities(x, y, layer=None):
    x = torch.cat([elem.unsqueeze(0) for elem in x])
    y = torch.cat([elem.unsqueeze(0) for elem in y])
        
    if layer == None:
        num_layers = x.shape[1]
        all_layer_similarities = []
        for layer in range(num_layers):
            layer_similarities = cosine_similarity(x[:, layer, :], y[:, layer, :])
            all_layer_similarities.append([layer_similarities])

        all_layer_similarities = np.concatenate(all_layer_similarities, axis=0)
        return all_layer_similarities.mean(axis=0)
    
    else:
        layer_similarities = cosine_similarity(x[:, layer, :], y[:, layer, :])
        return layer_similarities

def compute_distances(x, y, layer=None):
    x = torch.cat([elem.unsqueeze(0) for elem in x])
    y = torch.cat([elem.unsqueeze(0) for elem in y])
        

    if layer == None:
        num_layers = x.shape[1]
        all_layer_distances = []
        for layer in range(num_layers):
            layer_dist = cosine_distances(x[:, layer, :], y[:, layer, :])
            all_layer_distances.append([layer_dist])

        all_layer_distances = np.concatenate(all_layer_distances, axis=0)
        return all_layer_distances.mean(axis=0)
    
    else:
        layer_dist = cosine_distances(x[:, layer, :], y[:, layer, :])
        return layer_dist

def compute_mrr(sim_matrix, labels):
    mrr_scores = []
    for idx, sen_similarities in enumerate(sim_matrix):
        sen_label = labels[idx]
        positive_sentences_indices = np.where(np.array(labels) == sen_label)[0]
        if len(positive_sentences_indices) == 1:
            continue
        ranked_sentences = np.argsort(sen_similarities)[::-1].tolist()
        mrr_scores.append(np.max([1/(ranked_sentences.index(i)+1) for i in positive_sentences_indices if i !=idx]))
    avg_mrr_score = round(np.mean(mrr_scores), 3)
    return avg_mrr_score

def merge_texts_to_authors_per_label(style_distance_dataset, num_authors_per_label):

    author_labels = set(style_distance_dataset['feature'])
    style_authors = [style_distance_dataset.filter(lambda row: row['feature'] == l) for l in author_labels]
    style_authors = [a.shard(num_authors_per_label, index=i) for a in style_authors for i in range(num_authors_per_label)] # spit the author into 3 sub-authors
    
    merged_dataset = []
    for a in style_authors:
        merged_dataset.append({
            'positive': '\n'.join(a['positive']),
            'negative': '\n'.join(a['negative']),
            'feature':  a['feature'][0],
            'feature_clean': a['feature_clean'][0],
        })
    
    author_ds = Dataset.from_list(merged_dataset)
    return author_ds

def load_aa_data(data_path, groundtruth_path):
    import glob
    print('Loading: ', data_path)
    def q_c_mapping(c_author):
        c_author_idx = candidate_authors.index(c_author)
        found_assignment = np.where(ground_truth_assignment[:,candidate_authors.index(c_author)] == 1)
        if len(found_assignment[0]) > 0:
            q_author_idx = found_assignment[0][0]    
            return query_authors[q_author_idx]
        else:
            return c_author
            
    queries_df = pd.read_json(data_path + '_queries.jsonl', lines=True)
    candidates_df = pd.read_json(data_path + '_candidates.jsonl', lines=True)
    queries_df['authorID']  = queries_df['authorIDs'].apply(lambda x : x[0])
    candidates_df['authorSetID']  = candidates_df['authorSetIDs'].apply(lambda x : x[0])
    
    ground_truth_assignment = np.load(open(groundtruth_path + '_groundtruth.npy', 'rb'))
    candidate_authors = [a[2:-3] for a in  open(groundtruth_path + '_candidate-labels.txt').read().split('\n')][:-1]
    query_authors = [a[2:-3] for a in  open(groundtruth_path + '_query-labels.txt').read().split('\n')][:-1]

    candidates_df['authorID'] = candidates_df.authorSetID.apply(lambda a: q_c_mapping(a))
    candidates_df['source'] = ['candidates'] * len(candidates_df)
    queries_df['source'] = ['queries'] * len(queries_df)
                                                     
    all_df = pd.concat([candidates_df, queries_df])[["authorID", "fullText", "documentID", "source"]]

    return all_df, candidates_df, queries_df

def extract_sig_pairs_for_layer(hiatus_data_texts, muti_luar_layers_sims, labels, layer):
    pairs_of_sim_index = []
    for label in set(labels):
        label_indices = np.where(np.array(labels) == label)[0]

        if len(label_indices) == 1:
            continue

        label_matrix  = np.take(muti_luar_layers_sims, label_indices, axis=1)
        label_matrix  = np.take(label_matrix, label_indices, axis=2)
        zscore_matrix = zscore(label_matrix, axis=0)
        for i in range(len(label_indices)):
            for j in range(i+1, len(label_indices)):
                zscore_vector = zscore_matrix[:, i, j]
                if zscore_vector[layer] > 2.5: #np.argmax(zscore_vector) == layer:
                    # Extract the two texts, and their similarities across the n layers
                    author1_text = hiatus_data_texts[label_indices[i]]
                    author2_text = hiatus_data_texts[label_indices[j]]
                    pairs_of_sim_index.append((author1_text, author2_text, zscore_vector, label_matrix[:, i, j]))
    return pairs_of_sim_index

def det(probs, true):
    """Build the DET curve."""
    true_flat = true.flatten()
    probs_flat = probs.flatten()

    fpr, fnr, thresh = det_curve(true_flat, probs_flat)
    return fpr, fnr, thresh
    
def eer(prob, true):
    """Compute the EER."""
    
    fpr, fnr, thresh_det = det(prob, true)
    
    idx = np.nanargmin(np.absolute(fnr - fpr))
    EER = np.mean([fpr[idx], fnr[idx]])

    return round(EER, 3)