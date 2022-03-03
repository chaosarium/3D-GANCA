# import stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

BLOCK2VEC_OUT_PATH = 'output/block2vec saves/block2vec 64 dim/'

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import sys
from loguru import logger as gurulogger
gurulogger.level("DEBUG", color="<yellow><bold>")

class VoxelPerceptionNet(nn.Module):
    # Essentially running a trainable perceptor on each layer. This expands the number of channels by a factor of num_perceptions and gives us the visual features to do NCA updates
    
    def __init__(self, 
        num_in_channels=128, 
        num_perceptions=3, 
        normal_std=0.02, 
        use_normal_init=True, 
        zero_bias=True
    ):
        super().__init__()
        
        self.num_in_channels = num_in_channels
        self.normal_std = normal_std
        
        self.sequence = nn.Sequential(
            nn.Conv3d(
                self.num_in_channels, # incoming channels
                self.num_in_channels * num_perceptions, # expand num_in_channels by factor of 3
                3, # kernal size of 3, which means neighbour radius of 1
                stride=1, # stride of 1 so look at each voxel
                padding=1, # make sure to look at edge voxels
                groups=self.num_in_channels, # disconnect the perceptions so that multiple percpetions run on each channel; essentially we have num_perceptions convolutional layers side by side
                bias=False, # no bias
            ),
        )
        
        def init_weights(m):
            if isinstance(m, nn.Conv3d):
                # init weights for Conv3d layers
                nn.init.normal_(m.weight, std=normal_std)
                
                # if bias exist, init bias
                if getattr(m, "bias", None) is not None:
                    if zero_bias:
                        nn.init.zeros_(m.bias)
                    else:
                        nn.init.normal_(m.bias, std=normal_std)

        # weight initialisation
        if use_normal_init:
            with torch.no_grad():
                self.apply(init_weights)
                
    def forward(self, input):
        return self.sequence(input)
    
class VoxelUpdateNet(nn.Module):
    # This is essentially running dense nets in parallel for each voxel. It takes visual features and predict the update for each feature. In comes the output from a VoxelPerceptionNet, which is of the shape (N, num_in_channels * num_perceptions, channel_dims[0], channel_dims[1], channel_dims[2])
    
    def __init__(self,
        num_channels = 16,
        num_perceptions=3,
        channel_dims=[64, 64],
        normal_std=0.02,
        use_normal_init=True,
        zero_bias=True,
    ):
        super().__init__()
        
        def make_sequental(num_channels, channel_dims):
                
            # make first layer. 
            sequence = [
                # visual_feature_channels, x, y, z -> channel_dims[0], x, y, z
                nn.Conv3d(num_channels * num_perceptions, channel_dims[0], kernel_size=1), 
                nn.LeakyReLU(0.2, inplace=True) # trying leakyrelu here
            ]
            
            # loop through dims[1:] and make Conv3d
            for i in range(1, len(channel_dims)):
                sequence.extend([
                    nn.Conv3d(channel_dims[i - 1], channel_dims[i], kernel_size=1), 
                    nn.LeakyReLU(0.2, inplace=True) # trying leakyrelu here
                ])
                
            # make final layer
            sequence.extend([
                    nn.Conv3d(channel_dims[-1], num_channels, kernel_size=1, bias=False)
                ])
                        
            return nn.Sequential(*sequence)
        
        self.update_net = make_sequental(num_channels, channel_dims)

        def init_weights(m):
            if isinstance(m, nn.Conv3d):
                # init weights for Conv3d layers
                nn.init.normal_(m.weight, std=normal_std)
                
                # if bias exist, init bias
                if getattr(m, "bias", None) is not None:
                    if zero_bias:
                        nn.init.zeros_(m.bias)
                    else:
                        nn.init.normal_(m.bias, std=normal_std)

        # weight initialisation
        if use_normal_init:
            with torch.no_grad():
                self.apply(init_weights)

    def forward(self, x):
        # gurulogger.debug(f"it's shape {x.shape} coming in to the update net")
        return self.update_net(x)

class VoxelNCAModel(nn.Module):
    def __init__(self,
        alpha_living_threshold: float = 0.1, # level below which the cell would be dead
        cell_fire_rate: float = 0.5, # how often do cells update
        num_perceptions = 3, # num of filters
        perception_requires_grad: bool = True, # if perception filters are trainable
        num_embedding_channels: int = 64, # number of channels for block embeddings
        num_hidden_channels: int = 63, # num hidden channels
        normal_std: float = 0.0002, # for initialisation
        use_normal_init: bool = True, # whether to init
        zero_bias: bool = True, # whether to init bias as 0s
        update_net_channel_dims: List[int] = [32, 32], # channel sizes for hidden layers in VoxelUpdateNet
    ):
        super().__init__()
        self.alpha_living_threshold = alpha_living_threshold
        self.cell_fire_rate = cell_fire_rate
        self.num_perceptions = num_perceptions
        self.perception_requires_grad = perception_requires_grad
        self.num_embedding_channels = num_embedding_channels
        self.num_hidden_channels = num_hidden_channels
        self.normal_std = normal_std
        self.use_normal_init = use_normal_init
        self.zero_bias = zero_bias
        self.update_net_channel_dims = update_net_channel_dims
        
        # let's have 1 alpha channel
        self.alpha_channel_index = 0
        # the channels will be like [alpha, embeddings ... , hiddens ...]
        self.num_channels = 1 + self.num_embedding_channels + self.num_hidden_channels
        
        self.perception_net = VoxelPerceptionNet(
            num_in_channels = self.num_channels, 
            num_perceptions = self.num_perceptions, 
            normal_std = self.normal_std, 
            use_normal_init = self.use_normal_init, 
            zero_bias = self.zero_bias
        )
        if not self.perception_requires_grad:
            for p in self.perception_net.parameters():
                p.requires_grad = False

        self.update_network = VoxelUpdateNet(
            num_channels = self.num_channels,
            num_perceptions = self.num_perceptions,
            channel_dims = self.update_net_channel_dims,
            normal_std = self.normal_std, 
            use_normal_init = self.use_normal_init, 
            zero_bias = self.zero_bias
        )
        
    def check_alive(self, state):
        # scan the alpha channel and do a max pooling to get the maximum alpha for the cell's neighbourhood
        return F.max_pool3d(
            # cut out the one-hot block channels from the world (N, channels, x, y, z)
            state[:, self.alpha_channel_index : self.alpha_channel_index + 1, :, :, :],
            kernel_size = 3,
            stride = 1,
            padding = 1,
        )
    
    def perceive(self, state):
        return self.perception_net(state)
    
    def update(self, state):
        # this is going to result in a boolean tensor indicating cells that are alive
        pre_update_mask = self.check_alive(state) > self.alpha_living_threshold
        
        # extract features using the perception net
        perception = self.perceive(state)
        # calculate update deltas using the update net
        delta = self.update_network(perception)
        
        # mask out some cells. We take out (N, 1, x, y, z)
        rand_mask = torch.rand_like(state[:, 0:1, :, :, :]) < self.cell_fire_rate
        # multiply with the delta tensor to mask changes. This will broadcast to all channels
        delta = delta * rand_mask.float()
        
        # now we apply the changes
        state = state + delta
        
        # now get another boolean tensor of cells that are alive after update
        post_update_mask = self.check_alive(state) > self.alpha_living_threshold
        
        # cells are alive if they are alive both before and after update
        life_mask = (pre_update_mask & post_update_mask).float()
        # make all the dead cells everything zero # TODO train block2vec again so that air block is torch.zeros
        state = state * life_mask
                
        return state, life_mask
        
    def forward(self, 
            state, # the world state before update
            steps = 1, # how many steps to run the NCA
            get_final_mask = False
        ):
        # in comes a batch of worlds (N, channels, x, y, z)
        for step in range(steps):
            state, life_mask = self.update(state)
        
        # return accordingly
        if get_final_mask: 
            return state, life_mask
        else:
            return state
    
class VoxelDiscriminator(nn.Module):
    def __init__(self,
            num_in_channels: int = 64, # number f channels to represent a world
            use_sigmoid: bool = False,
        ):
        super().__init__()
        
        self.num_in_channels = num_in_channels
        self.use_sigmoid = use_sigmoid
                
        self.model = nn.Sequential(
            nn.Identity(),
            
            # num_in_channels, 32, 32, 32 --> 64, 16, 16, 16; downsample by factor of 2
            nn.Conv3d(num_in_channels, 64, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 64, 16, 16, 16 --> 128, 8, 8, 8; another downsample by factor of 2
            nn.Conv3d(64, 128, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128, 8, 8, 8 --> 256, 4, 4, 4; downsample by factor of 2
            nn.Conv3d(128, 256, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 256, 4, 4, 4 --> 512, 2, 2, 2; downsample by factor of 2
            nn.Conv3d(256, 512, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),            
            
            # 512, 2, 2, 2 --> 1, 1, 1, 1; whatever to scalar
            nn.Conv3d(512, 1, kernel_size=2, padding=0, stride=1, bias=False),

            # # sigmoid for binary classification
            nn.Flatten(), # make its shape N, 1
        )

    def forward(self, world):
        validity = self.model(world)
        if self.use_sigmoid:
            # gurulogger.debug('using sigmoid for discriminator')
            validity = torch.sigmoid(validity)
        return validity