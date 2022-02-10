# %%
import numpy as np
from einops import rearrange
import pandas as pd
from tqdm.notebook import tqdm
DATA_DIR = "dataset/house_data/houses"
WORLD_SIZE = 32
import math

# %%

def rearrange_sample(house):
    # this rearranges the training sample into an x y z m shape
    world = rearrange(house, 'z y x b -> x y z b')
    return world

def trim_world_empty(world):
    # this removes empty voxels sorrounding a world
    arr_slices = tuple(np.s_[curr_arr.min():curr_arr.max() + 1] for curr_arr in world[:,:,:,0].nonzero())
    return world[arr_slices]

def pad_world(world):
    # this pads a world to the smallest cubic volume that it can fit in
    (x, y, z, m) = world.shape
    min_dim = max(world.shape[0:3])
    world = np.pad(world, ((0, min_dim - x),(0, min_dim - y),(0, min_dim - z), (0, 0)), 'constant')
    return world

def something():
    return "hi"

def center_and_pad_world(world, min_size=None):
    world = trim_world_empty(world)
    # this pads a world to the smallest cubic volume that it can fit in
    (x, y, z, m) = world.shape
    if min_size:
        min_dim = min_size
    else:
        min_dim = max(world.shape[0:3])
    
    x_missing = min_dim - x
    y_missing = min_dim - y
    z_missing = min_dim - z
    
    world = np.pad(world, ((math.floor(x_missing/2), math.ceil(x_missing/2)),(math.floor(y_missing/2), math.ceil(y_missing/2)),(math.floor(z_missing/2), math.ceil(z_missing/2)), (0, 0)), 'constant')
    return world

def get_filtered_stats_df():
    # load filtered houses
    filtered_df = pd.read_pickle("dataset/filtered_houses_stats.pkl")
    return filtered_df

def all_trainx_as_df():
    # load all training examples
    filtered_df = get_filtered_stats_df()
    filtered_houses_df = pd.DataFrame() # new df
    for i, row in tqdm(filtered_df.iterrows(), total=filtered_df.shape[0]):
        filtered_houses_df = filtered_houses_df.append({
            'world': center_and_pad_world(trim_world_empty(rearrange_sample(np.load(DATA_DIR + "/" + row['dir'] + "/schematic.npy"))), WORLD_SIZE),
            'dir': row['dir'],
            'Size': row['Size'],
            'Num unique IDs': row['Num unique IDs'],
            'Num unique blocks': row['Num unique blocks'],
            'Percentage air untrimmed': row['Percentage air untrimmed'],
            'Percentage air trimmed': row['Percentage air trimmed'],
        }, ignore_index=True)

    print('loaded', len(filtered_houses_df), 'houses')
    return filtered_houses_df
