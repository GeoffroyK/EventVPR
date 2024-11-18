import torch
import numpy as np
import pandas as pd
from eventvpr import EventVPR

filepath = "/home/keimeg/jAER-events.txt"
index_name = {0: "timestamp", 1: "x", 2:"y", 3:"Polarity"}

def checkValidity(bin, aggregation_tensor):
    assert bin == aggregation_tensor.sum()


if __name__=="__main__":

    eventdf = pd.read_csv(filepath, delimiter=' ', header=3, usecols=[0, 1, 2, 3])
    eventdf = eventdf.set_axis(["timestamp","x","y","polarity"], axis=1)

    # Convert X and Y columns in integer
    eventdf["x"] = eventdf["x"].astype(int)
    print(eventdf.dtypes)

    print(eventdf["x"].max())
    print(eventdf["y"].max())

    event_in = torch.zeros((2,eventdf["y"].max()+1, eventdf["x"].max()+1))

    ### Test spike binning
    bin = 100000
    sIndex = 0

    img = np.zeros((2, eventdf["y"].max()+1, eventdf["x"].max()+1))

    xMax = eventdf["x"].max()
    yMax = eventdf["y"].max()

    net = EventVPR(n_in = (xMax + 1) * (yMax + 1), n_out = 1)


    while sIndex < bin:
        # Which channel ie polarity
        polarity = eventdf["polarity"].loc[sIndex]
        yPos = eventdf["y"].loc[sIndex]
        xPos = eventdf["x"].loc[sIndex]
        
        event_in[polarity][yMax - yPos][xMax - xPos] += 1
        img[polarity][yMax - yPos][xMax - xPos] += 1
        sIndex += 1

    checkValidity(bin, event_in)

    net(event_in)

    import matplotlib.pyplot as plt
    img = img.astype('uint8')

    plt.figure("OFF")
    plt.imshow(img[0])
    plt.figure("ON")
    plt.imshow(img[1])
    
    plt.show()
