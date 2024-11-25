import os
import math
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.visualisation import plotRecallAtN, plot_chance_recall, plotCorrMatrix, plotHistogram

def argmedian(x):
    median_value = np.median(x)
    median_index = np.argmin(np.abs(x - median_value))
    return median_index


def raw_events_around(x, tw, n) -> list:
    '''
    Return the raw events around withing the time window around the selected place
    '''
    xt = x[:,0]

    present = get_time_window(x, tw)
    passed_ind, future_ind = get_time_window_indices(x, tw)
    past = []
    future = []
    for n_hist in range(n):
        if n_hist % 2 == 0:
            past_val = xt[passed_ind]
            passed_ind = np.argmin(np.abs(xt - (past_val - tw / 2)))
            past.append(get_centered_time_window(x, tw, passed_ind))
        else:
            future_val = xt[future_ind]
            future_ind = np.argmin(np.abs(xt - (future_val + tw / 2)))
            future.append(get_centered_time_window(x, tw, future_ind))
    past.reverse()
    out_events = []
    for events in past:
        out_events.append(events)
    out_events.append(present)
    for event in future:
        out_events.append(event)
    return np.asarray(out_events, dtype=object)

def time_windows_around(x, tw, n):
    '''
    Return n histograms of duration tw of the event sequence x
    '''
    xt = x[:,0]

    present = get_time_window(x, tw)
    passed_ind, future_ind = get_time_window_indices(x, tw)
    past = []
    future = []

    if n == 1:
        return np.array([present])

    for n_hist in range(n):
        if n_hist % 2 == 0:
            past_val = xt[passed_ind]
            passed_ind = np.argmin(np.abs(xt - (past_val - tw / 2)))
            past.append(get_centered_time_window(x, tw, passed_ind))
        else:
            future_val = xt[future_ind]
            future_ind = np.argmin(np.abs(xt - (future_val + tw / 2)))
            future.append(get_centered_time_window(x, tw, future_ind))
    past.reverse()
    out_hists = []
    for hist in past:
        out_hists.append(event_histogram(hist))
    out_hists.append(event_histogram(present))
    for hist in future:
        out_hists.append(event_histogram(hist))
    return np.asarray(out_hists)

def get_centered_time_window(x, tw, center):
    xt = x[:,0]
    center_value = xt[center]
    beginning  = center_value - tw / 2
    end = center_value + tw / 2
    beginning_index = np.argmin(np.abs(xt - beginning))
    end_index = np.argmin(np.abs(xt - end))
    return x[beginning_index:end_index]

def get_time_window_indices(x, tw):
    xt = x[:,0]
    center = argmedian(xt)
    center_value = xt[center]
    beginning  = center_value - tw / 2
    end = center_value + tw / 2
    beginning_index = np.argmin(np.abs(xt - beginning))
    end_index = np.argmin(np.abs(xt - end))
    return beginning_index, end_index

def get_time_window(x, tw):
    xt = x[:,0]
    center = argmedian(xt)
    center_value = xt[center]
    beginning  = center_value - tw / 2
    end = center_value + tw / 2
    beginning_index = np.argmin(np.abs(xt - beginning))
    end_index = np.argmin(np.abs(xt - end))
    return x[beginning_index:end_index]

def plot_chance_recall(ax, length):
    '''
    Plot the chance level (y=x) for a Recall@N graph.
    '''
    step = (10 / length) / 10
    #ax.plot(np.arange(0,length), np.arange(0,1,step), '--', label='chance level', color='g')
    ax.plot(np.arange(0,length), np.linspace(0, 1, length), label='chance level', linestyle='--', linewidth=1)

def compute_recall_at(y_pred, n):
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
            recallScore += compute_recall_at(y_pred, n)

        recallAtN.append(recallScore/len(correlationMatrices)) # Mean value among all the possibilites
    if color==None:
        ax.plot(np.arange(0,labelsNumber+1), recallAtN,  label=label, linewidth=2)
    else:
        ax.plot(np.arange(0,labelsNumber+1), recallAtN,  label=label, linewidth=2, color=color)
    ax.set_title("Recall@N")
    ax.set_xlabel("N - Number of top correlated candidates")
    ax.set_ylabel("Average Recall@N (%)")
    return ax, recallAtN

def save_time_windows(traverse, timewindows):
    timestamps = {}
    for traverse in traverses:
        print(traverse)
        event_seq = np.load(f"event_sliding_window_40k_{traverse}.npy", allow_pickle=True).item()
        places = event_seq.keys()
        for place in places:
            
            currentKey = traverse + str(place)

            event_seq[place] = np.array(event_seq[place], dtype=object)            
            event_seq[place] = event_seq[place].flatten()

            reshaped = event_seq[place].reshape(-1, 4)
       
            for timewindow in timewindows:
                events = get_time_window(reshaped, timewindow)
                filename = f"output/timewindows/{traverse}_{place}_{timewindow}.npy"
                np.save(filename, events)

def event_histogram(patternList, dimension=[260,346,2]):
    outVector = np.zeros((dimension[0], dimension[1], 2)) # 2 Channels, ON & OFF 
    for event in patternList:
        # Timestamp, X, Y, Polarity
        x = int(event[2])
        y = int(event[1])
        polarity = int(event[3])
        
        # Ensure x and y are within bounds
        if 0 <= x < dimension[0] and 0 <= y < dimension[1]:
            outVector[x][y][polarity] += 1
    return outVector
def recall_n_timewindow(timewindows, traverses, event_data, places=25):
    duets = list(itertools.combinations(traverses, 2))
    corr_matrices = {}
    results = []
    fig, ax = plt.subplots()

    for duet in duets:
        for timewindow in timewindows:
            corr_matrices[timewindow] = []
            comparison = []
            reference = []

            for place in range(places):

                reference.append(event_data[duet[0]][timewindow][place])
                comparison.append(event_data[duet[1]][timewindow][place])

            fig2, ax2 = plt.subplots(figsize=(15,15))
            result, ax2 = plotCorrMatrix(ax2, reference, comparison, duet[0], duet[1])
            plt.close(fig2)
            corr_matrices[timewindow].append(result.to_numpy())
            
    # Color map for visualisation      
    cmap = plt.get_cmap('viridis', len(timewindows))
    for index, timewindow in enumerate(timewindows):
        ax, result = plotRecallAtN(ax, corr_matrices[timewindow], 25, label=f"event tw = {timewindow}", color=cmap(index / len(timewindows)))
        results.append(result)
    plot_chance_recall(ax, 26)

    norm = plt.Normalize(vmin=0, vmax=len(timewindows)-1)  # Normalize color values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # We set an empty array because we don't use scatter or image
    fig.colorbar(sm, ax=ax)  # Add the colorbar to the current axis


    plt.legend()
    plt.grid()
    plt.show()
    return results

def recall_n_histograms(ax, timewindows, traverses, event_data, bin, places=25) -> list:
    duets = list(itertools.combinations(traverses, 2))
    corr_matrices = {}
    results = []

    for duet in duets:
        for timewindow in timewindows:
            corr_matrices[timewindow] = []
            comparison = []
            reference = []

            for place in range(places):

                reference.append(event_data[duet[0]][place])
                comparison.append(event_data[duet[1]][place])

            fig2, ax2 = plt.subplots(figsize=(15,15))
            result, ax2 = plotCorrMatrix(ax2, reference, comparison, duet[0], duet[1])
            plt.close(fig2)
            corr_matrices[timewindow].append(result.to_numpy())
            
    # Color map for visualisation      
    cmap = plt.get_cmap('viridis', len(timewindows))
    for index, timewindow in enumerate(timewindows):
        ax, result = plotRecallAtN(ax, corr_matrices[timewindow], 25, label=f"tw:{timewindow},bin:{bin}")
        results.append(result)

    plt.legend()
    plt.grid()
    #plt.show()
    return results, ax 

def get_event_seq(traverse, n_places, timewindow, format='pickle') -> list:
    '''
    Get the event sequence of each places on selectected traverse at a fixed timewindow
    (must be saved before with the save_time_window function)
    '''
    event_seq = []
    for place in range(n_places):
        if format == 'pickle':
            event_seq.append(np.load(f"/home/geoffroyk/EventVPR/data/{traverse}_{place}_{timewindow}.npy", allow_pickle=True))
        elif format == 'txt':
            event_seq.append(np.loadtxt(f"/home/geoffroyk/EventVPR/data/{traverse}_{place}_{timewindow}.txt"))
    return np.array(event_seq, dtype=object)

def construct_histogram(traverses, timewindows, n_places=25):
    event_data = {}
    for traverse in traverses:
        event_data[traverse] = {}
    for traverse in traverses:
        for timewindow in timewindows:
            event_data[traverse][timewindow] = []

    for traverse in traverses:
        for timewindow in timewindows:
            for place in range(n_places):
                event_data[traverse][timewindow].append(event_histogram(np.load(f"output/timewindows/{traverse}_{place}_{timewindow}.npy", allow_pickle=True)))
    return event_data

def histogram_division(event_sequence: np.array, bins: int) -> np.array:
    '''
    Create N (bins) sub histograms of the provided event sequence.
    '''
    
    # Get all timestamps in the sequence
    xt = event_sequence[:,0]
    # Calculate the length of the sequence (t1 - t0)
    tw_size = xt[-1] - xt[0]
    size_bin = tw_size / bins
    subsequences = []
    last_ind = 0
    for bin in range(bins):
        bmax = xt[last_ind] + size_bin
        bmax_index = np.argmin(np.abs(xt - bmax))
        subsequences.append(event_sequence[last_ind:bmax_index,:])
        last_ind = bmax_index
    out_hist = []
    for subsequence in subsequences:
        out_hist.append(event_histogram(subsequence))
    return np.asarray(out_hist)

def plot_auc(recalls, timewindows):
    '''
    Plot the AuC as a bar plot for each Recall@n
    '''
    # Chance level for the Recall@N
    yy = np.linspace(0, 1, 26)
    # Calculate AUC for each time window
    aucs = []
    for result in recalls:
        aucs.append(auc(result, yy))
    # np.save("auc_tw.npy", aucs)

    cmap = plt.cm.inferno
    coloredge = ['r' if (bar == max(aucs)) else 'black' for bar in aucs]
    colors = ['r' if (bar == max(aucs)) else 'grey' for bar in aucs]
    
    timewindows = [str(tw) for tw in timewindows]
    fig, ax = plt.subplots(figsize=(10,8))
    ax.bar(timewindows, aucs, color=colors, edgecolor=coloredge, linewidth=1)
    ax.set_xlabel("Time window length")
    ax.set_ylabel("AUC") 
    ax.set_title("AUC for each time window on Brisbane QVPR")
    plt.show()

def plot_heatmap(traverses: list, timewindow: float, bins: int, s_place: int) -> None:
    '''
    Plot the event window as an image.
    The heatmap of the number of events per pixel
    traverses = name of the traverses 
    timewindow = size of the event sequence
    bins = number of histograms (must be > 1)
    s_place = selected place for the plot    
    '''


    for traverse in traverses:

        event_seq = get_event_seq(traverse, 25, timewindow)
        hs = histogram_division(event_seq[s_place], bins)

        fig, axs = plt.subplots(bins, 2, figsize=(18, 25), gridspec_kw={'width_ratios': [1, 3]})

        for i, image in enumerate(hs):
            on_events = image[:,:,0]
            off_events = image[:,:,-1]
            # Cumulative histogram by adding ON and OFF events
            cumulated_events = on_events + off_events

            # --- Left: Plot the image ---
            axs[i, 0].imshow(image, cmap='gray')
            axs[i, 0].axis('off')
            axs[i, 0].set_title(f'Image {i + 1}')
            
            # --- Center: Plot the 2D ON event histogram ---
            ax_hist2d = axs[i, 1]
            
            # Use a heatmap-like 2D histogram for ON events
            heatmap_combined = ax_hist2d.imshow(cumulated_events * 255, cmap='hot', aspect='auto', vmin=0, vmax=np.max(cumulated_events))
            # Optionally, overlay contours to emphasize intensity (similar to the 1σ, 2σ, 3σ lines)
            #ax_hist2d.contour(cumulated_events, colors='white', alpha=0.5)

            plt.colorbar(heatmap_combined, ax=ax_hist2d, orientation='vertical', fraction=0.046, pad=0.04)

            # --- Right: Plot the marginal histograms ---
            divider = make_axes_locatable(ax_hist2d)
            # ---- Top marginal histogram (size 260, summing along rows for cumulative events) ----
            ax_top_hist = divider.append_axes("top", size="20%", pad=0.1, sharex=ax_hist2d)
            # Sum across rows (axis=0) to get a 260-sized histogram (sum of each column)
            col_sums = cumulated_events.sum(axis=0)  # Size (346,)
            ax_top_hist.bar(np.arange(346), col_sums, color='purple', alpha=0.7)
            ax_top_hist.set_ylabel('Sum')
            ax_top_hist.set_xticks([])  # Remove x-ticks for the top histogram
            
            # ---- Right marginal histogram (size 346, summing along columns for cumulative events) ----
            ax_right_hist = divider.append_axes("right", size="20%", pad=0.1, sharey=ax_hist2d)
            # Sum across columns (axis=1) to get a 346-sized histogram (sum of each row)
            row_sums = cumulated_events.sum(axis=-1)  # Size (260,)
            ax_right_hist.barh(np.arange(260), row_sums, color='orange', alpha=0.7)
            ax_right_hist.set_xlabel('Sum')
            ax_right_hist.set_yticks([])  # Remove y-ticks for the right histogram
            

        # Adjust the layout for better spacing
        plt.suptitle(f"Place {s_place} on traverse {traverse}")
        plt.tight_layout()
        plt.show()

def init_dict(traverses, n_places) -> dict:
    mydict = {}
    for traverse in traverses:
        mydict[traverse] = {}
    for place in range(n_places):
        mydict[traverse][place] = None
    return mydict

if __name__ == "__main__":
    traverses = ["sunset1", "sunset2", "daytime", "morning", "sunrise"]
    timewindows = [0.18,0.09,0.08,0.07,0.06,0.05,0.4,0.3,0.2,0.01]
    
    # save_time_windows(traverses, timewindows)
    # event_data = construct_histogram(traverses, timewindows)
    # print("Beginning recall@n")
    #np.save("recalltw.npy", event_data)
    #recall_n(timewindows, traverses, event_data)

    timewindow = 0.06
    bins = [2]
    n_places = 25
    sequences = init_dict(traverses,n_places)
    fig, ax = plt.subplots()
    recalls = []
    for bin in bins:
        for traverse in traverses:
            event_seq = get_event_seq(traverse, n_places, timewindow)
            for place in range(n_places):
                # hs = histogram_division(event_seq[place], bin)
                # =sequences[traverse][place] = np.asarray(hs)
                hs = time_windows_around(event_seq[place], n_places, bin)
                sequences[traverse][place] = np.asarray(hs)
        res, ax = recall_n_histograms(ax, [timewindow], traverses, sequences,bin, n_places)
        recalls.append(res)
    plot_chance_recall(ax, 26)
    #plot_auc(recalls, timewindow)
    plt.show()


    # n_places = 25
    # traverse = "sunset1"
    # timewindow = 0.06
    # event_seq = get_event_seq(traverse, n_places, 0.9)
    # a = time_windows_around(event_seq[0], timewindow, 4)
    # a = np.asarray(a, dtype=object)

    