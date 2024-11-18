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

if __name__ == "__main__":
    pass