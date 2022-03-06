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

def visualise_world_tensor(worlds, plots_per_row = 4, figsize = (20, 20)):
    '''
    worlds_data: a long tensor of shape (N, x, y, z)
    plots_per_row: how many plots in a row, default 4
    figsize: default (20, 20)
    '''
        
    fontsize = 2 * (figsize[0] / plots_per_row)
    
    fig, axs = plt.subplots(math.ceil(len(worlds)/plots_per_row), plots_per_row, figsize = figsize, subplot_kw={"projection": '3d', "adjustable": 'box'})
    
    try:
        _ = axs[0][0]
    except:
        axs = [axs] 
        
    r, c = 0, 0
    for i, world in enumerate(tqdm(worlds)):

        (x, y, z) = world.shape
        blockidarray = world

        color_dict = get_color_dict(np.unique(blockidarray))
        colors = convert_to_color(blockidarray, color_dict)
                
        axs[r][c].set_title(f'generated house {i}', fontsize=fontsize)
        axs[r][c].voxels(blockidarray, facecolors=colors)
        c += 1
        if c == plots_per_row:
            c = 0
            r += 1

    return fig

def state_to_graph():
    pass