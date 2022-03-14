import loguru
import numpy as np
import torch
import torch.nn as nn
import pickle
import pandas as pd

def get_embedding_info(data_path):
    '''
    returns the pytorch embedding object, mcid2block dict, block2embeddingidx dict, and embeddingidx2block dict, and block2mcid dict
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
    
    block2mcid = mc_block_database.set_index('item id').to_dict()['numerical id']
    for key in block2mcid:
        processed_val = block2mcid[key].split(':')
        # getting rid of meta if there is one. For example 'minecraft:stone': '1:6' will be turned into 'minecraft:stone': '1'
        block2mcid[key] = int(processed_val[0])
    
    return embedding, mcid2block, block2embeddingidx, embeddingidx2block, block2mcid

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

# %%

from sklearn.neighbors import NearestNeighbors

def nearest_neighbors(values, all_values, n_neighbors=1, metric='euclidean'):
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm='brute').fit(all_values)
    dists, idxs = nn.kneighbors(values)
    return dists, idxs

# %%

class DataConverter():
    def __init__(self,
            embedding, 
            mcid2block, 
            block2embeddingidx, 
            embeddingidx2block, 
            block2mcid
        ) -> None:
        self.embedding = embedding
        self.mcid2block = mcid2block
        self.block2embeddingidx = block2embeddingidx
        self.embeddingidx2block = embeddingidx2block
        self.block2mcid = block2mcid
        
    def embeddingidx2blockidx(self, embeddingidx):
        block_name = self.embeddingidx2block[embeddingidx]
        blockid = self.block2mcid[block_name]
        return blockid

# %%

def extract_embedding_channels(states, num_embedding_channels):
    # in comes (N, c, x, y, z) with c containing alpha and hidden channels
    return states[:, 1:num_embedding_channels+1, :, :, :]

    
def extract_alpha_channels(states):
    # in comes (N, c, x, y, z) with c containing alpha and hidden channels
    return states[:, 0:1, :, :, :]


# insert nothing between voxels to make middle ones visible (https://terbium.io/2017/12/matplotlib-3d/)
def explode_voxels(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3]*2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded
