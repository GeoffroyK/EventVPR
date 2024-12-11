import numpy as np
import matplotlib.pyplot as plt
from utils import recalltw

def convert__voxel_grid(events_seq, T, dim=None):
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
            volume[t_bin][x][y] = p * max(0, 1 - abs(t - t_scaled[idx]))
    return volume

if __name__ == "__main__":
    event_seq = recalltw.get_event_seq("sunset1", 1, 1.0, "pickle")
    voxel_grid = convert__voxel_grid(event_seq, 6)
    
    # Create interactive plot
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)  # Make room for slider
    
    # Initial plot
    frame_idx = 0
    im = ax.imshow(voxel_grid[frame_idx], cmap='RdBu', interpolation='nearest')
    plt.colorbar(im)
    ax.set_title(f'Time bin: {frame_idx}')
    
    # Create slider
    ax_slider = plt.axes([0.1, 0.1, 0.65, 0.03])  # [left, bottom, width, height]
    slider = plt.Slider(
        ax=ax_slider,
        label='Time bin',
        valmin=0,
        valmax=voxel_grid.shape[0]-1,
        valinit=0,
        valstep=1
    )
    
    # Update function for slider
    def update(val):
        frame_idx = int(slider.val)
        im.set_array(voxel_grid[frame_idx])
        ax.set_title(f'Time bin: {frame_idx}')
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    plt.show()
