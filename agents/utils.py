import numpy as np
import random

from collections import namedtuple, deque
from typing import *



Transition = namedtuple('Transition', ('state', 'action', 'reward', 'done', 'next_state'))



def extract_cell(rgb_arr :np.ndarray, x :int, y :int, cell_size_x :int, cell_size_y :int|None=None) -> np.ndarray:
    """
    Extracts a specific cell from a given RGB image array.

    Parameters
    ----------
    rgb_arr : np.ndarray of shape (Y, X, C)
        The input RGB image represented as a 3D NumPy array with shape (height, width, channels),
        where Y is the height, X is the width, and C is the number of color channels (typically 3 for RGB).
    x : int
        The horizontal index (column) of the top-left corner of the cell to extract.
    y : int
        The vertical index (row) of the top-left corner of the cell to extract.
    cell_size_x : int
        The width of the cell to extract, in pixels.
    cell_size_y : int | None, optional
        The height of the cell to extract, in pixels. If ``None``, the height will be the same as `cell_size_x`.
        Default is ``None``.
        
    Returns
    -------
    : np.ndarray
        A 3D NumPy array representing the extracted cell from the original image. The shape of the array
        will be (cell_size_y, cell_size_x, C), where C is the number of color channels in the original image.
        
    Notes
    -----
    - If ``cell_size_y`` is not provided (None), it will default to the value of ``cell_size_x``, creating a square cell.
    - The coordinates (x, y) correspond to the top-left corner of the cell within the original image array.
    """
    if cell_size_y is None:
        cell_size_y = cell_size_x
    cell_img = rgb_arr[y*cell_size_y:(y+1)*cell_size_y, x*cell_size_x:(x+1)*cell_size_x, :]
    return cell_img



class Memory():
    """
    This class implements a memory for storing and sampling ``Transition``. 
    """

    def __init__(self, capacity :int=1024):
        """
        Parameters
        ----------
        capacity : int, optional
            Capacity of the memory in number of transitions, by default ``1024``.
        """
        self.memory = deque([], maxlen=capacity)
        return


    def push(self, transition :Transition) -> None:
        """
        Save a new transition.
        If the memory is full, the oldest transaction is overwritten.

        Parameters
        ----------
        transition : Transition
            New transition to store.
        """
        self.memory.append(transition)
        return None


    def sample(self, batch_size :int=16) -> Transition:
        """
        Draw randomly batch_size samples from the memory.

        Parameters
        ----------
        batch_size : int
            Number of sample to draw by default ``16``.

        Returns
        -------
        : Transition
            Transaction samples.
        """
        return random.sample(self.memory, batch_size)


    def __len__(self) -> int:
        return len(self.memory)