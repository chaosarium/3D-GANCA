# %%
# Embeddings

import loguru
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

# %%

import math

def make_seed_state(
        batch_size,
        num_channels = 128, 
        alpha_channel_index = 0,
        seed_dim = (4, 4, 4), 
        world_size = (32, 32, 32)
    ):
    
    x_missing = world_size[0] - seed_dim[0]
    y_missing = world_size[1] - seed_dim[1]
    z_missing = world_size[2] - seed_dim[2]
    
    seeds = torch.rand(batch_size, num_channels, *seed_dim)
    
    # make them all very alive
    
    seeds[:, alpha_channel_index, :, :, :] = 1.0
    
    # pad worlds
    seeds = torch.nn.functional.pad(seeds, (
        math.floor(x_missing/2), math.ceil(x_missing/2), # pad x
        math.floor(y_missing/2), math.ceil(y_missing/2), # pad y
        math.floor(z_missing/2), math.ceil(z_missing/2), # pad z
        0,0,0,0 # don't pad batch and channels
    ), mode='constant', value=0)
    
    return seeds

# %%

def make_real_labels(batch_size):
    return torch.ones(batch_size, 1)
def make_fake_labels(batch_size):
    return torch.zeros(batch_size, 1)

# %%

from einops import rearrange

def examples2embedding(worlds, embedding):
    embedded = embedding(worlds)
    embedded = rearrange(embedded, 'N x y z c -> N c x y z')
    return embedded