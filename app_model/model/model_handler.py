# import torch

# model/model_handler.py
# import sys
import os
# sys.path.append(os.path.abspath('/Users/ranjeetnagarkar/Desktop/Deep_learning_model/recommender_system_deploy/app_model/model'))

import json
import numpy as np
import pandas as pd
import io
import tempfile
from google.cloud import storage
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
# from model.model_definition import TwoTowerModel
import tensorflow as tf
from keras.layers import Input, Dense, Dot, Lambda
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from model.config import SearchConfig

# QUERY_COLUMNS = [
#         'q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8',
#         'q9', 'q10', 'q11', 'q12', 'q13', 'q14', 'q15', 'q16',
#         'q17', 'q18', 'q19', 'q20', 'q21', 'q22', 'q23', 'q24',
#         'q25', 'q26', 'q27', 'q28', 'q29', 'q30', 'q31'
#     ]

def l2_normalize(x):
    return tf.keras.backend.l2_normalize(x, axis=-1)

class ModelHandler:
    def __init__(self, bucket_name='models-deep-learning', \
                #  model='model_dict_path.pth',\
                 model = 'recommendation_model.h5',
                 product_model='product_model.tree',\
                 mp_product_dict='product_dictionary.json',                 
                 embedding_model='distilbert-base-uncased'):
        model_data = self.read_blob_to_memory(bucket_name, model)
        self.load_model_tf(model_data)
        self.transformer_model = SentenceTransformer(embedding_model)
        self.product_tree = self.load_annoy_index(bucket_name, product_model)
        self.product_dict = self.load_product_dictionary(bucket_name, mp_product_dict)
       
    # def load_model(self, byte_stream):
    #     """Load a complete PyTorch model from a byte stream."""
    #     byte_stream.seek(0) 
    #     state_dict = torch.load(byte_stream, map_location=torch.device('cpu'))
    #     self.model.load_state_dict(state_dict)
    #     self.model.eval()  
    #     print("Model state dictionary loaded and model set to evaluation mode.")

    def save_blob_to_tempfile(self, byte_stream):
        """Saves the byte stream to a temporary file and returns the file path."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
            tmp_file.write(byte_stream.read())
            return tmp_file.name 

    def load_model_tf(self, byte_stream):
        """Load a complete TensorFlow model from a byte stream."""
        temp_model_path = self.save_blob_to_tempfile(byte_stream)
        try:
            self.model = load_model(temp_model_path, custom_objects={'l2_normalize': l2_normalize})
            print("Model loaded successfully from", temp_model_path)
        finally:
            os.remove(temp_model_path)

    def read_blob_to_memory(self, bucket_name, blob_name):
        """Reads a blob and returns it as a byte string."""
        # storage_client = storage.Client.from_service_account_json('model/deep-learning-project-422602-5cc0079b56e8.json')
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        byte_stream = io.BytesIO()
        blob.download_to_file(byte_stream)
        byte_stream.seek(0)  # Move to the beginning of the byte stream
        return byte_stream
    
    def load_annoy_index(self, bucket_name, index_file):
        """Load an Annoy index from a blob into a temporary file."""
        byte_stream = self.read_blob_to_memory(bucket_name, index_file)
        with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
            tmp_file.write(byte_stream.read())
            tmp_file.flush()
            index = AnnoyIndex(16, 'dot')  # Ensure you know the vector length and distance metric
            index.load(tmp_file.name)
            return index
        
    def load_product_dictionary(self, bucket_name, dictionary_file):
        """Load a JSON dictionary from a blob."""
        byte_stream = self.read_blob_to_memory(bucket_name, dictionary_file)
        # Convert byte stream to a dictionary
        byte_stream.seek(0)  # Ensure the stream is at the start
        return json.load(byte_stream)

    # def preprocess_query(self, query_text):
    #     embeddings = self.transformer_model.encode(query_text, convert_to_tensor=True)
    #     embeddings = self.reshape_array(embeddings.cpu().numpy(), 32)
    #     query_embedding = self.model.forward_query(torch.tensor(embeddings, dtype=torch.float32))
    #     return query_embedding
    
    def preprocess_query_tf(self, query_text):
        embeddings = self.transformer_model.encode(query_text, convert_to_tensor=True)
        embeddings = self.reshape_array(embeddings.cpu().numpy(), 32)
        df_query_test_actual = pd.DataFrame([embeddings])
        print("Shape of df_query_test_actual:", df_query_test_actual.shape)
        print("Expected number of columns:", len(SearchConfig.QUERY_COLUMNS))

        query_tower_cols = SearchConfig.QUERY_COLUMNS
        df_query_test_actual.columns = query_tower_cols
        input_data_query = [
            np.array(df_query_test_actual[query_tower_cols]),  # Query features
            np.zeros((1, 160))  # Placeholder for product features if necessary
        ]

        query_model = Model(inputs=self.model.input,
                      outputs=self.model.get_layer('normalize_query').output)
        query_embedding = query_model.predict(input_data_query)
        return query_embedding
    
    def reshape_array(self, embedding, d):
        original_dim = embedding.shape[0]
        new_embedding = np.zeros(d)
        for j in range(d):
            start_idx = j * (original_dim // d)
            end_idx = (j + 1) * (original_dim // d) if j < (d - 1) else original_dim
            chunk = embedding[start_idx:end_idx]
            new_embedding[j] = np.mean(chunk)
        return new_embedding       

    def find_similar_products(self, query_text, k=10):
        query_embedding = self.preprocess_query_tf(query_text=query_text)
        nearest_ids = self.product_tree.get_nns_by_vector(query_embedding[0], k, include_distances=False)
        mp_product_dict = self.product_dict
        nearest_products = [mp_product_dict[str(pid)] for pid in nearest_ids]
        return nearest_products
    