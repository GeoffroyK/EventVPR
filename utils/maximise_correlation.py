'''
Maximise intercorelation of frames
@GeoffroyK
'''
import numpy as np
import pandas as pd
from scipy.linalg import norm

def getCorrelationValue(reference: np.array, comparison: np.array) -> float:
    reference = pd.DataFrame(reference.flatten())
    comparison = pd.DataFrame(comparison.flatten())
    result = reference.corrwith(comparison)
    return result[0]

def maximiseSequenceWithReference(reference: np.array, frameSequence: list) -> np.array:
    '''
    Ideally, images should be grayscaled
    '''
    for index, frame in enumerate(frameSequence):
        comparisonScore = getCorrelationValue(reference, frame)
        if index == 0:
            # Maximum value is the first one 
            maxScore = comparisonScore
            maxFrame = frame
            maxIndex = index

        elif comparisonScore > maxScore:
            # Compare with all frames to get the maximum score value
            maxScore = comparisonScore
            maxFrame = frame
            maxIndex = index
    return maxFrame, maxScore, maxIndex

def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng

def compare_images(img1, img2):
    # normalize to compensate for exposure difference, this may be unnecessary
    # consider disabling it
    img1 = normalize(img1)
    img2 = normalize(img2)
    # calculate the difference and its norms
    diff = img1 - img2  # elementwise for scipy arrays
    z_norm = norm(diff.ravel(), 0)  # Zero norm
    return z_norm

def maximiseSequenceWithReferencePixelwise(reference: np.array, frameSequence: list) -> np.array:
    '''
    Ideally, images should be grayscaled
    '''
    for index, frame in enumerate(frameSequence):
        n_0 = compare_images(reference, frame)
        comparisonScore = getCorrelationValue(reference, frame)

        if index == 0:
            # Maximum value is the first one 
            maxFrame = frame
            maxIndex = index
            nMin = compare_images(reference, frameSequence[-1])
            maxScore = comparisonScore

        elif n_0 < nMin:
            # Compare with all frames to get the maximum score value
            maxScore = comparisonScore
            maxFrame = frame
            maxIndex = index
            nMin = n_0
    return maxFrame, maxScore, maxIndex

if __name__ == "__main__":
    reference = np.random.random((5,5))
    frameSequence = [np.random.random((5,5)) for i in range(10)]
    frameSequence.append(reference)
    maxFrame, maxScore, frameIndex = maximiseSequenceWithReference(reference, frameSequence)

    import matplotlib.pyplot as plt
    plt.subplot(1,2,1)
    plt.imshow(reference)
    plt.title("Reference")

    plt.subplot(1,2,2)
    plt.imshow(maxFrame)
    plt.title(f"Highest Correlated frame with corr = {maxScore}")

    plt.tight_layout()
    plt.show()

    print(compare_images(reference, reference))