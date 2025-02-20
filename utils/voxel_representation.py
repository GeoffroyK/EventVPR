import numpy as np
import matplotlib.pyplot as plt
from utils import recalltw
from utils.visualisation import plot_voxel_grids, plot_3d_voxel_grid

def convert_voxel_grid(events_seq, T, dim=None):
    """
    Voxel grid representation, each event distributes its polarity p to the two closest spatio-temporal voxels.
    Algorithm:
     1. Scales the timestamps to the range [0, T] with T the number of bins.
     2. Generates event frames V with dimension (T, H, W)
    
    Args:
        events_seq (list): List of events, each event is a tuple (t, x, y, p) where:
            - t is the timestamp
            - x, y are the pixel coordinates
            - p is the polarity (-1 or 1)
        T (int): Number of time bins for the voxel grid

    Returns:
        numpy.ndarray: Voxel grid representation of shape (T, H, W) where:
            - T is the number of time bins
            - H, W are the height and width of the event camera frame
            - dim, dimension of the event frames    
    """
    # Set dimension of the event frames
    if dim is None:
        H = int(np.max(events_seq[:,:, 1])) + 1  # Max x coordinate + 1
        W = int(np.max(events_seq[:,:, 2])) + 1  # Max y coordinate + 1
    else:
        H, W = dim
        
    # Scale the timestamps to the range [0, T]
    t_min = events_seq[0][0][0]
    t_max = events_seq[0][-1][0]
    t_scaled = []
    for t, x, y, p in events_seq[0]:
        scale = (T-1) / (t_max - t_min) * (t - t_min)
        t_scaled.append(scale)

    volume = np.zeros((T, H, W))
    for t_bin in range(T):
        for idx, (t, x, y, p) in enumerate(events_seq[0]):
            x = int(x)
            y = int(y)
            # Fix polarity to -1 instead of 0
            if p == 0:
                p = -1
            volume[t_bin][x][y] += p * max(0, 1 - abs(t_bin - t_scaled[idx]))
    return volume
    
if __name__ == "__main__":
    traverses = ["sunset1", "sunset2", "daytime", "morning", "sunrise"]
    voxel_grids = []
    for traverse in traverses:
        # Traverse 1
        event_seq = recalltw.get_event_seq(traverse, 1, 1.0, "pickle")
        event_seq = event_seq[:,:15000,:]   
        voxel_grid = convert__voxel_grid(event_seq, 5)
        voxel_grids.append(voxel_grid)
    plot_voxel_grids(voxel_grids, traverses)
    plot_3d_voxel_grid(voxel_grids[0])
    