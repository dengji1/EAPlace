from placedb import PlaceDB
from canvas import Canvas
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import os

def greedy_place(chip:Canvas, module_list):
    for i, module_name in enumerate(module_list):
        overlap = chip.get_overlap(module_name)
        wl_icre = chip.get_hpwl_increment(module_name)
        wl_icre[overlap > overlap.min()] = np.inf
        min_indices = np.where(wl_icre == np.min(wl_icre))
        idx = random.randint(0, len(min_indices[0])-1)
        chip.place_module(module_name, min_indices[0][idx], min_indices[1][idx])
    chip.overlap = np.maximum(0, chip.overlap - 1)
    chip.hpwl *= chip.ratio
    return chip

# def rand_place(chip:Canvas, module_list):
#     for i, module_name in enumerate(module_list):
#         overlap = chip.get_overlap(module_name)
#         min_indices = np.where(overlap == np.min(overlap))
#         idx = random.randint(0, len(min_indices[0])-1)
#         chip.place_module(module_name, min_indices[0][idx], min_indices[1][idx])
#     chip.overlap = np.maximum(0, chip.overlap - 1)
#     chip.hpwl *= chip.ratio
#     return chip

def place(chip:Canvas, module_list, position):
    for module_name in module_list:
        x = position[module_name]['x']
        y = position[module_name]['y']
        chip.place_module(module_name, x, y)
    return chip