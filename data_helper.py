import numpy as np
from einops import rearrange
import pandas as pd
from tqdm.notebook import tqdm
import os.path as osp, os, requests, tarfile
DATA_DIR = "dataset/house_data/houses"
WORLD_SIZE = 32
import math

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
    try:
        filtered_df = pd.read_pickle("dataset/filtered_houses_stats.pkl")
    except:
        filtered_df = pd.read_pickle("dataset/filtered_houses_stats.pkl4")
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

def houses_dataset():
    filtered_df = get_filtered_stats_df()
    out = []
    
    for i, row in tqdm(filtered_df.iterrows(), total=filtered_df.shape[0]):
        world = center_and_pad_world(trim_world_empty(rearrange_sample(np.load(DATA_DIR + "/" + row['dir'] + "/schematic.npy"))), WORLD_SIZE)
        out.append(world)
        
    print('loaded', len(out), 'houses')
    return np.array(out)

def download_dataset():
    data_dir = 'dataset'
    url = "https://craftassist.s3-us-west-2.amazonaws.com/pubr/house_data.tar.gz"
    os.makedirs(data_dir, exist_ok=True)

    tar_path = osp.join(data_dir, "houses.tar.gz")
    extracted_dir = osp.join(data_dir, "house_data")

    if not (osp.isfile(tar_path) or osp.isdir(extracted_dir)):
        print(f"Downloading dataset from {url}")
        response = requests.get(url, allow_redirects=True)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to retrieve image from url: {url}. "
                f"Status: {response.status_code}"
            )
        with open(tar_path, "wb") as f:
            f.write(response.content)
    else: 
        print("Dataset already exists")

    if not osp.isdir(extracted_dir):
        print(f"Extracting dataset to {extracted_dir}")
        tar = tarfile.open(tar_path, "r")
        tar.extractall(data_dir)

from torch.utils.data.dataset import Dataset
import torch    
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class GANCA3DDataModule(pl.LightningDataModule):
    def __init__(self, 
            batch_size = 16,
            num_workers = 0,
            mcid2block = [], 
            block2embeddingid = [],
        ):
        super().__init__()
        
        self.batch_size = batch_size
        self.dims = (32, 32, 32) # this will be returned when calling this.size()
        self.num_workers = num_workers
        self.mcid2block = mcid2block
        self.block2embeddingid = block2embeddingid
                
    def prepare_data(self):
        # download data
        download_dataset()

    def setup(self, stage=None):
        # splitting data and process stuff
        full_dataset = houses_dataset()[:,:,:,:,0]
        
        # Replace all the MC block ids with embedding ids
        
        def blockidx2embeddingidx(blockidx):
            block_name = self.mcid2block[str(blockidx)]
            embeddingid = self.block2embeddingid[block_name]
            return embeddingid

        vectorised_blockidx2embeddingidx = np.vectorize(blockidx2embeddingidx)
        
        print('Turning MC id into embedding idx. This could take up to a minute.')
        full_dataset = vectorised_blockidx2embeddingidx(full_dataset)
        
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(full_dataset, [1600, 192, 185])
        
    # these funcs can also be placed directly inside a LightningModule
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        # return DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=NUM_WORKERS)
        pass

    def test_dataloader(self):
        # return DataLoader(self.test_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=NUM_WORKERS)
        pass

    def predict_dataloader(self):
        pass