import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import math
from tqdm.notebook import tqdm
import numpy as np
import copy


def get_color_dict(unique_vals):
    state = np.random.RandomState(0)
    color_arr = list(state.uniform(0, 1, (len(unique_vals), 3)))
    color_arr = [rgb2hex(color) for color in color_arr]
    color_arr = [None] + color_arr # set all to 0 for air block (block id 0)
    colors = color_arr[: len(unique_vals)] # chop off the last random
    color_dict = {str(unique_vals[i]): colors[i] for i in range(len(unique_vals))} # make string
    return color_dict # return

def convert_to_color(arr, color_dict):
    new_arr = copy.deepcopy(arr).astype(object)
    for k in color_dict:
        new_arr[new_arr == int(k)] = color_dict[k]
    return new_arr

def visualise_world(worlds_data, plots_per_row = 4, figsize = (20, 20)):
    '''
    worlds_data: a pd dataframe with column 'world' and column 'dir'
    plots_per_row: how many plots in a row, default 4
    figsize: default (20, 20)
    '''
    
    worlds = worlds_data['world'].to_list()
    dirs = worlds_data['dir'].to_list()
    
    fontsize = 2 * (figsize[0] / plots_per_row)
    
    try:
        _ = worlds[0][0][0][0][0]
    except IndexError:
        worlds = [worlds] 

    fig, axs = plt.subplots(math.ceil(len(worlds)/plots_per_row), plots_per_row, figsize = figsize, subplot_kw={"projection": '3d', "adjustable": 'box'})
    
    try:
        _ = axs[0][0]
    except:
        axs = [axs] 
        
    r, c = 0, 0
    for i, world in enumerate(tqdm(worlds)):

        (x, y, z, b) = world.shape
        blockidarray, blockmetaarray = world[:,:,:,0], world[:,:,:,1]

        color_dict = get_color_dict(np.unique(blockidarray))
        colors = convert_to_color(blockidarray, color_dict)
        
        meta_color_dict = get_color_dict(np.unique(blockmetaarray))
        edge_colors = convert_to_color(blockmetaarray, meta_color_dict)
        
        axs[r][c].set_title(dirs[i], fontsize=fontsize)
        axs[r][c].voxels(blockidarray, facecolors=colors, edgecolors=edge_colors)
        c += 1
        if c == plots_per_row:
            c = 0
            r += 1

    plt.show()

def visualise_single_world_tensor(world, ax = None):
    '''
    visualizes a single world tensor; returns an axis
    worlds_data: a long tensor of shape (x, y, z)
    '''
    
    if ax == None:
        ax = plt.axes(projection='3d', adjustable= 'box')
            
    (x, y, z) = world.shape
    blockidarray = world

    color_dict = get_color_dict(np.unique(blockidarray))
    colors = convert_to_color(blockidarray, color_dict)
            
    # ax.set_title(f'generated house')
    ax.voxels(blockidarray, facecolors=colors)
    
    return ax

import matplotlib as mpl

# reference: https://terbium.io/2017/12/matplotlib-3d/
def visualise_world_alpha(alpha_channel, ax = None):
    '''
    visualizes a single world tensor; returns an axis
    alpha_channel: a float tensor of shape (x, y, z)
    '''
    mycolormap = plt.get_cmap('coolwarm')
    transparency=0.2

    # adapted https://blog.csdn.net/weixin_39771351/article/details/111293632
    colorsvalues = np.empty(alpha_channel.shape, dtype=object)
    for i in range(0,alpha_channel.shape[0]): 
        for j in range(0,alpha_channel.shape[1]): 
            for k in range(0,alpha_channel.shape[2]): 
                relative_value = (alpha_channel[i][j][k]+1)/2 # normalize to [0,1] range
                tempc = mycolormap(relative_value)
                colorreal=(tempc[0],tempc[1],tempc[2],transparency)
                colorsvalues[i][j][k]=colorreal
    
    if ax == None:
        ax = plt.axes(projection='3d', adjustable= 'box')
        
    ax.voxels(alpha_channel, facecolors=colorsvalues, edgecolor=None, shade=True,)
    

    return ax

from einops import rearrange
import utils

def states_to_graphs(world_states, embedding_tensor, n_cols = 1, n_rows = 1, converter_class = None, size_multiplier=4, explode = False, trim = False, no_air = False, metric='euclidean'):
    # input worlds (N, c, x, y ,z); N worlds will be put left-to-right, top-to-bottom on the figure
    input_dims = world_states.shape # save dims for restoration
    print(f"n cols {n_cols}, n rows {n_rows}, input dims {input_dims}")
    assert n_cols * n_rows == input_dims[0] # make sure col and rows align
    
    # flatten
    world_states = rearrange(world_states, 'N c x y z -> (N x y z) c')
    
    if no_air:
        embedding_tensor = embedding_tensor[1:]
    
    _, idxs = utils.nearest_neighbors(
        values=world_states, 
        all_values=embedding_tensor,
        n_neighbors=1,
        metric = metric
    )
    
    # restore
    idxs = idxs.reshape(input_dims[0], input_dims[2], input_dims[3], input_dims[4])
    
    # turn embedding idxs to mc idxs
    vectorised_embeddingidx2blockidx = np.vectorize(converter_class.embeddingidx2blockidx)
    idxs = vectorised_embeddingidx2blockidx(idxs)
    
    # visualize
    fig, axs = plt.subplots(n_rows, n_cols, figsize = (n_cols*size_multiplier, n_rows*size_multiplier), dpi=100, subplot_kw={"projection": '3d', "adjustable": 'box'},)
    
    for i in range(2):
        try:
            _ = axs[0][0]
        except:
            axs = [axs] 

    r, c = 0, 0
    for world in tqdm(idxs, desc="3D plots", colour = 'pink', ncols = 1000, leave=False):
        
        if explode:
            world = utils.explode_voxels(world)
        if trim:
            arr_slices = tuple(np.s_[curr_arr.min() - 1:curr_arr.max() + 2] for curr_arr in world.nonzero())
            world = world[arr_slices]

        visualise_single_world_tensor(world, ax=axs[r][c])
        
        c += 1
        if c == n_cols:
            c = 0
            r += 1

    return fig

def alpha_states_to_graphs(alpha_channels, n_cols = 1, n_rows = 1, size_multiplier=4, explode = False, trim = False):
    # input worlds (N, 1, x, y ,z); N worlds will be put left-to-right, top-to-bottom on the figure
    input_dims = alpha_channels.shape # save dims for restoration
    print(f"n cols {n_cols}, n rows {n_rows}, input dims {input_dims}")
    assert n_cols * n_rows == input_dims[0] # make sure col and rows align
        
    # visualize
    fig, axs = plt.subplots(n_rows, n_cols, figsize = (n_cols*size_multiplier, n_rows*size_multiplier), dpi=100, subplot_kw={"projection": '3d', "adjustable": 'box'},)
    
    for i in range(2):
        try:
            _ = axs[0][0]
        except:
            axs = [axs] 

    r, c = 0, 0
    for world in tqdm(alpha_channels, desc="3D plots", colour = 'pink', ncols = 1000, leave=False):
    
        world = world[0,:,:,:] # turn (1, x, y, z) to (x, y, z)

        if explode:
            world = utils.explode_voxels(world)
        if trim:
            arr_slices = tuple(np.s_[curr_arr.min() - 1:curr_arr.max() + 2] for curr_arr in world.nonzero())
            world = world[arr_slices]

        
        visualise_world_alpha(world, ax=axs[r][c])
        
        c += 1
        if c == n_cols:
            c = 0
            r += 1

    return fig


def fig2rgb_array(fig):
    canvas = fig.canvas
    """Adapted from: https://stackoverflow.com/a/21940031/959926"""
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    ncols, nrows = canvas.get_width_height()
    scale = round(math.sqrt(buf.size / 3 / nrows / ncols))
    return buf.reshape(scale * nrows, scale * ncols, 3)

