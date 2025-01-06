from SIVs.utils_cpy import get_features, get_file_paths
from SIVs.siv_baseline_luar_cpy import SIV_Baseline_Luar
import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import tensorflow as tf

class Luar_Weg:


    def get_file_names(input_dir:str):
        queries_fname = glob(os.path.join(input_dir, "*queries*"))[0]
        candidates_fname = glob(os.path.join(input_dir, "*candidates*"))[0]
        return queries_fname, candidates_fname

    def get_prefix(input_dir):
        queries_fname, candidates_fname = Luar_Weg.get_file_names(input_dir)
        prefix = queries_fname.split(os.sep)[-1]
        return prefix[:prefix.find('_TA2')]

    def modify_df(data, author_identifier, ratio):

        raw_data = []
        for _, row in data.iterrows():
            document_id = row['documentID']
            author_id = row[author_identifier]
            text = row['fullText'].replace('<PERSON>','')
            start = 0
            modifier = len(text)//ratio
            while True:
                end = start + modifier
                if end > len(text):
                    break
                raw_data.append({'documentID': document_id, author_identifier: author_id, 'fullText': text[start:end]})
                start = end
            
        return pd.DataFrame.from_dict(raw_data)
    

    def get_doc_map(data, identifier):
        doc_map = {}
        for _, row in data.iterrows():        
            doc_map[row['documentID']] = row[identifier][0]
        return doc_map
    

    def get_wegmann(data, author_labels, doc_labels):

        texts = [row['fullText'] for _, row in data.iterrows()]
        model = SentenceTransformer('AnnaWegmann/Style-Embedding')
        doc_embeddings = model.encode(texts)
        print(doc_embeddings.shape)

        embed_map = {}
        for i in range(len(doc_labels)):
            label = doc_labels[i]
            embedding = doc_embeddings[i, :]

            if label not in embed_map:
                embed_map[label] = [embedding]
            else:
                embed_map[label].append(embedding)
        
        author_embeddings = np.zeros((len(author_labels), doc_embeddings.shape[1]))
        for i in range(len(author_labels)):
            label = author_labels[i]
            embeddings = embed_map[label]
            if len(embeddings) == 1:
                embeddings = embeddings[0]
            else:
                embeddings  = np.array(embeddings)
                embeddings = np.mean(embeddings, axis=0)
        
            author_embeddings[i, :] = embeddings
        
        return doc_embeddings, author_embeddings




    def get_luar(luar, data, identifier, doc_map):

        luar.set_author_level(True)
        extract = luar.extract_embeddings(luar.model, luar.tokenizer, data=data, identifier=identifier)
        author_embeddings = np.array(extract['features'])
        author_labels = [label[2:-3] for label in extract[identifier]]
        label_map = {author_labels[i]: i for i in range(len(author_labels))}
        
        luar.set_author_level(False)
        extract = luar.extract_embeddings(luar.model, luar.tokenizer, data=data, identifier=identifier)
        doc_embeddings = np.array(extract['features'])
        doc_labels = [doc_map[doc_id] for doc_id in extract['documentID']]

        return doc_embeddings, author_embeddings, doc_labels, author_labels, label_map



    def __init__(self, input_dir, output_dir, run_id, query_identifier, candidate_identifier, ratio):

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.run_id = run_id

        queries_fname, candidates_fname = get_file_paths(input_dir)
        query_df = pd.read_json(queries_fname, lines=True)
        candidates_df = pd.read_json(candidates_fname, lines=True)
        query_df = Luar_Weg.modify_df(data=query_df, author_identifier='authorIDs', ratio=ratio)

        query_doc_map = Luar_Weg.get_doc_map(data=query_df, identifier='authorIDs')
        candidate_doc_map = Luar_Weg.get_doc_map(data=candidates_df, identifier='authorSetIDs')
        
        luar = SIV_Baseline_Luar(self.input_dir, query_identifier, candidate_identifier)
        luar.load_model()

        query_embeddings_luar, author_embeddings_luar, self.query_labels, self.author_labels, self.author_label_map = Luar_Weg.get_luar(
            luar=luar, data=query_df, identifier='authorIDs', doc_map=query_doc_map)

        candidate_embeddings_luar, author_set_embeddings_luar, self.candidate_labels, self.author_set_labels, self.author_set_label_map = Luar_Weg.get_luar(
            luar=luar, data=candidates_df, identifier='authorSetIDs', doc_map=candidate_doc_map)
        
        query_embeddings_weg, author_embeddings_weg = Luar_Weg.get_wegmann(
            data=query_df, author_labels=self.author_labels, doc_labels=self.query_labels)
        
        candidate_embeddings_weg, author_set_embeddings_weg = Luar_Weg.get_wegmann(
            data=candidates_df, author_labels=self.author_set_labels, doc_labels=self.candidate_labels)
    
        self.query_embeddings = [query_embeddings_luar, query_embeddings_weg]
        self.candidate_embeddings = [candidate_embeddings_luar, candidate_embeddings_weg]
        self.author_embeddings = [author_embeddings_luar, author_embeddings_weg]
        self.author_set_embeddings = [author_set_embeddings_luar, author_set_embeddings_weg]
    
        self.row_count = int((len(self.author_set_labels)/len(self.author_labels))//10) + 1
 

    def build_model(self, num_labels, sim_length):

       
        inp = tf.keras.layers.Input(shape=(sim_length,), dtype=tf.float32, name="cossim")
        out = tf.keras.layers.Dense(200, activation='relu')(inp)
        out = tf.keras.layers.Dropout(0.1)(out)
        out = tf.keras.layers.Dense(150,activation = 'relu')(out)
        y = tf.keras.layers.Dense(num_labels,activation = 'sigmoid')(out)
        self.model = tf.keras.Model(inputs=inp, outputs=y)

        optimizer = tf.keras.optimizers.Adam(
        learning_rate=5e-05, # this learning rate is for bert model , taken from huggingface website
        epsilon=1e-08,
        clipnorm=1.0
        )
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True)
        metric = [tf.keras.metrics.CategoricalAccuracy('balanced_accuracy'), 'accuracy']

        self.model.compile(
            optimizer = optimizer,
            loss = loss,
            metrics = metric
        )


    def save_output_files(prefix, output_dir, run_id, output_matrix, authorIDs, authorSetIDs):
        output_array_path = os.path.join(output_dir , f"{prefix}_TA2_query_candidate_attribution_scores_{run_id}.npy")
        output_query_labels_path = os.path.join(output_dir , f"{prefix}_TA2_query_candidate_attribution_query_labels_{run_id}.txt")
        output_candidate_labels_path = os.path.join(output_dir , f"{prefix}_TA2_query_candidate_attribution_candidate_labels_{run_id}.txt")
        with open(output_array_path, 'wb') as f:
            np.save(f, output_matrix)
        
        with open(output_query_labels_path, 'w') as f:
            for line in authorIDs:
                f.write(f"('{line}',)\n")
        
        with open(output_candidate_labels_path, 'w') as f:
            for line in authorSetIDs:
                f.write(f"('{line}',)\n")

    def train(self):

        
        X = [cosine_similarity(doc_e, aut_e) for doc_e, aut_e in zip(self.query_embeddings, self.author_embeddings)]
        X = np.hstack(X)
        print('X', X.shape)
        self.build_model(sim_length=X.shape[1], num_labels=len(self.author_labels))
        Y = np.zeros((len(self.query_labels), len(self.author_labels)))
        for i in range(len(self.query_labels)):
            Y[i, self.author_label_map[self.query_labels[i]]] = 1
        print('Y', np.sum(Y))
        for i in range(1):
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.15, stratify=Y)
            self.model.fit(
                x ={'cossim': X_train} ,
                y = Y_train,
                validation_data = (
                {'cossim': X_val}, Y_val
                ),
                epochs=1000,
                batch_size=16
            )


    def predict(self):
        sim_matrices = [cosine_similarity(self.author_embeddings[i], self.author_set_embeddings[i]) for i in range(len(self.author_embeddings))]
        sim_matrix = np.mean(np.array(sim_matrices), axis=0)
        indices =  np.unravel_index(np.argsort(sim_matrix, axis=None), shape=sim_matrix.shape)

        curr = 1
        selected_rows = {i: 0 for i in range(len(self.author_labels))}
        selected_cols = set()
        selected_labels = set()
        selected_count = len(self.author_labels) * self.row_count
        while(len(selected_cols) < selected_count):
            row = indices[0][-curr]
            col = indices[1][-curr]
            curr += 1

            if selected_rows[row] >= self.row_count or col in selected_cols:
                continue
            
            selected_rows[row] += 1
            selected_cols.add(col)
            selected_labels.add(self.author_set_labels[col])
        
        print('Number Selected', len(selected_labels))
       
        X = [cosine_similarity(doc_e, aut_e) for doc_e, aut_e in zip(self.candidate_embeddings, self.author_embeddings)]
        X = np.hstack(X)
        print('X_pred', X.shape)
        Y = self.model.predict(x ={'cossim': X}, batch_size=2)
        print('Y_pred',Y.shape)
        
        output_matrix =  np.zeros((len(self.author_labels), len(self.author_set_labels)))
        print('OM', output_matrix.shape)
        results = {}
        for i in range(len(self.candidate_labels)):
            label = self.candidate_labels[i]
            if label not in results:
                results[label] = [Y[i, :]]
            else:
                results[label].append(Y[i, :])

        for i in range(len(self.author_set_labels)):
            label = self.author_set_labels[i]
            if label not in selected_labels:
                continue
            curr  = np.array(results[label])
            if curr.shape[0] > 1:
                curr = np.mean(curr, axis=0)
                curr = np.reshape(curr, (1, len(self.author_labels)))
            output_matrix[:, i] = curr
        print('Sum',np.sum(output_matrix))

            
        
        
        prefix = Luar_Weg.get_prefix(input_dir=self.input_dir)

        Luar_Weg.save_output_files(prefix, self.output_dir, self.run_id, output_matrix, self.author_labels, self.author_set_labels)
