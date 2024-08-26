'''
Visualisation module
@GeoffroyK
'''

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def plotHistogram(img):
    '''
    Plot 2 Histograms for ON/OFF channels
    '''
    channel_names = ["ON", "OFF"]
    channel_index = [0, 2]
    index = 1

    plt.figure(figsize=(8,8))
    for channel in channel_index:
        plt.subplot(2,1,index)
        print("oui")
        counts, bins = np.histogram(img[:,:,channel], bins=img.shape[0] * img.shape[1])
        print(counts)
        print(bins)
        #plt.stairs(counts, bins, fill=True)
        plt.hist(counts, bins)
        print("oui")

        plt.title(f"Channel {channel_names[index-1]}")
        index += 1
    plt.suptitle("Histograms of events divivided by channel")
    plt.show()

def corrMatrix(traverse1, traverse2, name1, name2) -> np.array:
    '''
    Correlation matrix 
    '''
    assert len(traverse1) == len(traverse2) 
    
    t1, t2 = {}, {}

    for index, place in enumerate(traverse1):
        t1[name1+str(index)] = place.flatten()
    for index, place in enumerate(traverse2):
        t2[name2+str(index)] = place.flatten()

    t1, t2 = pd.DataFrame(t1), pd.DataFrame(t2)
    result = pd.concat([t1, t2], axis=1).corr()
    
    result = result[[key for key in t2]].loc[[key for key in t1]]
    plt.figure(figsize=(20,20))
    sn.heatmap(result, cmap="Blues", annot=True, )
    plt.show()
    return result

def computeRecallAt(y_pred, n):
    assert n>0 and n<=len(y_pred[0]), "N should be superior to 0 and inferior to the number of labels"
    # The correct guess should be always in the matrix' diagonal: ie the correct answer is an identity matrix 
    y_true = np.identity(y_pred.shape[0])

    correctMatches = 0
    index = 0
    for pred in y_pred:
        ind = np.argpartition(pred, -n)[-n:]
        if np.argmax(y_true[index]) in ind:
            correctMatches += 1

        # print(f"Values of true at index {index}: {y_true[index]}")
        # print(f"Index max of true {np.argmax(y_true[index])}")
        # print(f"Predicted values {pred}")
        # print(f"Print {n} highest elements in array: {ind}")
        # print("================================================")
        index += 1

    return correctMatches / y_pred[0].shape[-1]

def plotRecallAtN(correlationMatrices: list, labelsNumber: int) -> list:
    '''
    Compute Recall@N from the correlation Cnk correlation matrices among all the sampled places 
    TODO For the beginning, it should be more interesting to remove the night dataset as it will bias negatively the recall score.
    '''
    recallScore = 0
    recallAtN = []

    for n in range(1, labelsNumber+1):
        recallScore = 0

        for y_pred in correlationMatrices:
            print(n)
            recallScore += computeRecallAt(y_pred, n)

        recallAtN.append(recallScore/len(correlationMatrices)) # Mean value among all the possibilites
    
    plt.plot(np.arange(1,labelsNumber+1), recallAtN)
    plt.title("Recall@N")
    plt.xlabel("N - Number of top correlated candidates")
    plt.ylabel("Average Recall@N (%)")
    plt.show()

if __name__ == "__main__":
    t1, t2 = [], []
    correlatedMatrix = []
    for i in range(12):
        t1.append(np.random.randint(255, size=(35,35)))
        t2.append(np.random.randint(255, size=(35,35)))

    
    correlatedMatrix.append(corrMatrix(t1,t2,"oui","non").to_numpy())
    t1, t2 = [], []
    print(correlatedMatrix[0].shape)
    plotRecallAtN(correlatedMatrix, correlatedMatrix[0].shape[0])