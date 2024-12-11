'''
Visualisation module
@GeoffroyK
'''

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def plotHistogram(ax, img):
    '''
    Plot 2 Histograms for ON/OFF channels
    '''
    channel_names = ["ON", "OFF"]
    channel_index = [0, 2]
    index = 0

    for channel in channel_index:
        plt.subplot(2,1,index+1)
        counts, bins = np.histogram(img[:,:,channel], bins=img.shape[0] * img.shape[1])
        plt.hist(counts, bins)
        ax.title(f"Channel {channel_names[index]}")
        index += 1
    ax.suptitle("Histograms of events divivided by channel")
    return ax

def plotCorrMatrix(ax, traverse1, traverse2, name1, name2):
    '''
    Compute the pearson Correlation matrix an output both the matrix and the associated plot.
    '''
    assert len(traverse1) == len(traverse2) , "Matrices should have the same shape"

    t1, t2 = {}, {}

    for index, place in enumerate(traverse1):
        t1[name1+str(index)] = place.flatten()
    for index, place in enumerate(traverse2):
        t2[name2+str(index)] = place.flatten()

    t1, t2 = pd.DataFrame(t1), pd.DataFrame(t2)
    result = pd.concat([t1, t2], axis=1).corr()
    result = result[[key for key in t2]].loc[[key for key in t1]]
    sn.heatmap(result, cmap="Blues", annot=True, ax=ax)
    return result, ax

def computeRecallAt(y_pred, n):
    assert n>0 and n<=len(y_pred[0]), f"N should be superior to 0 and inferior to the number of labels (n={n})"
    # The correct guess should be always in the matrix' diagonal: ie the correct answer is an identity matrix 
    y_true = np.identity(y_pred.shape[0])

    correctMatches = 0
    index = 0
    for pred in y_pred:
        ind = np.argpartition(pred, -n)[-n:]
        if np.argmax(y_true[index]) in ind:
            correctMatches += 1

        index += 1

    return correctMatches / y_pred[0].shape[-1]

def plot_chance_recall(ax, length):
    '''
    Plot the chance level (y=x) for a Recall@N graph.
    '''
    step = (10 / length) / 10
    #ax.plot(np.arange(0,length), np.arange(0,1,step), '--', label='chance level', color='g')
    ax.plot(np.arange(0,length), np.linspace(0, 1, length), label='chance level', linestyle='--', linewidth=1)


def plotRecallAtN(ax, correlationMatrices: list, labelsNumber: int, label="", color=None):
    '''
    Compute Recall@N from the correlation Cnk correlat matrionices among all the sampled places 
    '''
    recallScore = 0
    recallAtN = []
    recallAtN.append(0) # Initial value
    for n in range(1, labelsNumber+1):
        recallScore = 0

        for y_pred in correlationMatrices:
            recallScore += computeRecallAt(y_pred, n)

        recallAtN.append(recallScore/len(correlationMatrices)) # Mean value among all the possibilites
    if color==None:
        ax.plot(np.arange(0,labelsNumber+1), recallAtN,  label=label, linewidth=2)
    else:
        ax.plot(np.arange(0,labelsNumber+1), recallAtN,  label=label, linewidth=2, color=color)
    ax.set_title("Recall@N")
    ax.set_xlabel("N - Number of top correlated candidates")
    ax.set_ylabel("Average Recall@N (%)")
    return ax

def plot_voxel_grids(voxel_grids, traverses):
    """
    Plot multiple voxel grid representations in an interactive grid layout.

    The function creates a figure with subplots arranged in a grid, where each subplot shows
    a voxel grid representation for a different traverse. The plot is interactive - use left/right
    arrow keys to step through time bins.

    Args:
        voxel_grids (list): List of voxel grid representations, each of shape (T, H, W) where:
            - T is the number of time bins
            - H, W are the height and width of the event camera frame
        traverses (list): List of traverse names/identifiers corresponding to each voxel grid

    Returns:
        None. Displays the interactive plot.

    Controls:
        - Right arrow: Move forward one time bin
        - Left arrow: Move backward one time bin
    """
    # Create interactive plot
    num_grids = len(voxel_grids)
    # Calculate number of rows and columns for a more square arrangement
    ncols = int(np.ceil(np.sqrt(num_grids)))
    nrows = int(np.ceil(num_grids / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    
    # Convert axes to 2D array if it's 1D or single plot
    if num_grids == 1:
        axes = np.array([[axes]])
    elif len(axes.shape) == 1:
        axes = axes.reshape(1, -1)
    
    # Initial plots
    ims = []
    current_frame = [0]  # Use list to allow modification in nested function
    max_frames = voxel_grids[0].shape[0] - 1
    
    # Hide extra subplots if any
    for idx in range(num_grids, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')
    
    # Plot the voxel grids
    for i in range(num_grids):
        row = i // ncols
        col = i % ncols
        im = axes[row, col].imshow(voxel_grids[i][0].T, cmap='RdBu', 
                                 interpolation='nearest', vmin=-1, vmax=1)
        ims.append(im)
        axes[row, col].set_title(f'Time bin: 0, {traverses[i]}')
    
    def on_key(event):
        if event.key == 'right' and current_frame[0] < max_frames:
            current_frame[0] += 1
        elif event.key == 'left' and current_frame[0] > 0:
            current_frame[0] -= 1
        else:
            return
        
        # Update all plots
        for i, im in enumerate(ims):
            row = i // ncols
            col = i % ncols
            im.set_array(voxel_grids[i][current_frame[0]].T)
            axes[row, col].set_title(f'Time bin: {current_frame[0]}, {traverses[i]}')
        fig.canvas.draw_idle()
    
    # Connect the key press event
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show()

def plot_3d_voxel_grid(voxel_grid):
    """
    Plots a single voxel grid in 3D with dimensions: time (bins) and x axis.
    
    Args:
        voxel_grid (numpy.ndarray): Voxel grid of shape (T, H, W)
    """
    T, H, W = voxel_grid.shape
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a grid of coordinates
    t, h, w = np.meshgrid(np.arange(T), np.arange(H), np.arange(W), indexing='ij')
    
    # Flatten the arrays
    t = t.flatten()
    h = h.flatten()
    w = w.flatten()
    values = voxel_grid.flatten()
    
    # Separate positive and negative events
    positive_indices = values > 0
    negative_indices = values < 0
    
    # Plot positive events in red with smaller size
    ax.scatter(t[positive_indices], h[positive_indices], w[positive_indices], c='r', marker='o', s=0.1)
    
    # Plot negative events in blue with smaller size
    ax.scatter(t[negative_indices], h[negative_indices], w[negative_indices], c='b', marker='o', s=0.1)
    
    # Add title
    ax.set_title(f'Voxel Grid Representation with {T} Bins')
    
    plt.show()