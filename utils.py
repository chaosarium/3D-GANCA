# Embeddings

import numpy as np
import torch
import torch.nn as nn
import pickle
import pandas as pd

def get_embedding_info(data_path):
    '''
    returns the pytorch embedding object, mcid2block dict, block2embeddingidx dict, and embeddingidx2block dict
    '''
    
    with open(data_path + "representations.pkl", 'rb') as f:
        embeddings_dict = pickle.load(f)
    
    embeddings_array = np.load(data_path + "embeddings.npy")
    embeddings_array.shape

    embedding = nn.Embedding.from_pretrained(torch.tensor(embeddings_array), freeze=True)

    with open(data_path + "block2idx.pkl", 'rb') as f:
        block2embeddingidx = pickle.load(f)
    with open(data_path + "idx2block.pkl", 'rb') as f:
        embeddingidx2block = pickle.load(f)
        
    mc_block_database = pd.read_csv('block_ids_alt.tsv', sep='\t')
    mc_block_database = mc_block_database.filter(items=['numerical id', 'item id'])
    mc_block_database = mc_block_database.dropna(subset=["numerical id"])
    mcid2block = mc_block_database.set_index('numerical id').to_dict()['item id']

    return embedding, mcid2block, block2embeddingidx, embeddingidx2block