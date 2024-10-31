#
import torch
import numpy as np

__all__ = [
    'get_zmax_list'
]

#======================================================================#
def get_zmax_list(dataset):
    zmax = []
    for graph in dataset:
        zm = graph.pos[:,2].max().item()
        zmax.append(zm)
    return zmax

#======================================================================#
