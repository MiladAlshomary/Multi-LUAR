from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset, Dataset
import numpy as np
from einops import rearrange, reduce, repeat
import torch
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
import math

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

def compute_mrr(sim_matrix, labels):
    mrr_scores = []
    for idx, sen_similarities in enumerate(sim_matrix):
        sen_label = labels[idx]
        positive_sentences_indices = np.where(np.array(labels) == sen_label)[0]
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
            'feature':  a['feature'][0]
        })
    
    author_ds = Dataset.from_list(merged_dataset)
    return author_ds