import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# Parameters
tau = 20


def create_spike_train(event_sequence: np.array, dims: tuple) -> np.array:
    """
    Create a spike train from a list of spike times, each spike train is defined as the list of indices (discrete time) of the spike
    """ 
    lastTimestamp = None
    event_index = 0

    event_tensor_on = np.empty((dims[0], dims[1]), dtype=object)
    event_tensor_off = np.empty((dims[0], dims[1]), dtype=object)

    # Initialize the event tensors
    for i in range(dims[0]):
        for j in range(dims[1]):
            event_tensor_on[i][j] = []
            event_tensor_off[i][j] = []

    for subsequence in event_sequence:
        for event in subsequence:
            timestamp = float(event[0])
            x = int(event[1])
            y = int(event[2])
            p = int(event[3])

            # Initialize the last timestamp
            if lastTimestamp is None:
                lastTimestamp = timestamp

            # Check if the timestamp is the same as the last timestamp
            if timestamp != lastTimestamp:
                event_index += 1
                lastTimestamp = timestamp

            if p == 1:
                event_tensor_on[x,y].append(event_index)
            else:
                event_tensor_off[x,y].append(event_index)
    return event_tensor_on, event_tensor_off, event_index

def exp_kernel(t, tau):
    """
    Exponential kernel, with heavyside step function
    """
    return np.exp(-t / tau) * (t >= 0)

def exponential_decay_kernel(size, decay_rate):
    """Generate an exponential decay kernel."""
    return np.exp(-decay_rate * np.arange(size))

def fft_convolution(signal, kernel):
    signal_len = len(signal)
    kernel_len = len(kernel)
    conv_len = signal_len + kernel_len - 1

    signal_fft = np.fft.fft(signal, n=conv_len)
    kernel_fft = np.fft.fft(kernel, n=conv_len)

    conv_signal = np.fft.ifft(signal_fft * kernel_fft)
    return np.real(conv_signal[:signal_len])

def convoluted_signal(time:np.array, signal:np.array) -> np.array:
    """
    Convolute the signal with the exponential kernel
    """
    convoluted_signal = np.zeros_like(time, dtype=np.float64)
    for i, t in enumerate(time):
        # Accumulate contributions from past events
            if i + 1  < len(signal):
                for j in range(i+1):
                    convoluted_signal[i] += signal[j] * exp_kernel(i - j, tau)
            # else:
            #     for j in range(len(signal)):
            #         convoluted_signal[i] += signal[j] * exp_kernel(i - j, tau)
    return convoluted_signal

def heaviside(t):
    return np.heaviside(t,1)

def signal_distance(xt: np.array, yt: np.array) -> np.array:
    return np.abs(xt - yt)

def van_rossum_distance(signal_distance, tau):
    """
    Compute the van Rossum distance
    """
    integral = trapezoid(signal_distance, dx=1)
    D_R = np.sqrt((1 / tau) * integral)
    return D_R


# def exp_kernel(spikes: np.array, tau: float) -> np.array:
#     conv = np.zeros_like(spikes)
#     spiked = False
#     for index, spike in enumerate(spikes):
#         if spike == 1.:
#             print("spike")
#             conv[index] = 1
#             spiked = True
#         elif spiked:
#             #conv[index] = np.exp(- conv[index-1] + (conv[index-1] - 1) / tau)
#             conv[index] = conv[index-1] + -conv[index-1] / tau
#             # if conv[index] == 0:
#             #     print("resetting")
#             #     spiked = False
#     return conv 
        


if __name__ == "__main__":
    time = np.arange(0, 100, 1)  # Time vector

    # Création des signaux x et y
    x = np.zeros_like(time)
    y = np.zeros_like(time)
    x[21] = 1
    x[50] = 1
    y[20] = 1
    y[51] = 1

    # Calcul des signaux convolués
    xt = convoluted_signal(time, x)
    yt = convoluted_signal(time, y)
    # xt = exp_kernel(x, 2.)
    # yt = exp_kernel(y, 2.)

    # Création des sous-graphiques
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Signaux originaux, convolués et distance de van Rossum")

    # Signal x
    axs[0, 0].plot(time, x, drawstyle="steps-post")
    axs[0, 0].set_title("Signal x")
    axs[0, 0].set_xlabel("Temps")
    axs[0, 0].set_ylabel("Amplitude")

    # Signal y
    axs[0, 1].plot(time, y, drawstyle="steps-post")
    axs[0, 1].set_title("Signal y")
    axs[0, 1].set_xlabel("Temps")
    axs[0, 1].set_ylabel("Amplitude")

    # Signaux convolués xt et yt
    axs[1, 0].plot(time, xt, label="xt")
    axs[1, 0].plot(time, yt, label="yt")
    axs[1, 0].set_title("Signaux convolués xt et yt")
    axs[1, 0].set_xlabel("Temps")
    axs[1, 0].set_ylabel("Amplitude")
    axs[1, 0].legend()

    # Distance de van Rossum
    axs[1, 1].plot(time, signal_distance(xt, yt))
    axs[1, 1].set_title("(xt - yt)^2")
    axs[1, 1].set_xlabel("Temps")
    axs[1, 1].set_ylabel("Amplitude")

    plt.tight_layout()
    
    print(van_rossum_distance(signal_distance(xt, yt), tau))
    plt.show()

    event_seq = np.asarray(np.load(f"../event_sliding_window_40k_sunset1.npy", allow_pickle=True).item()[0])

    event_tensor_on, event_tensor_off, time_index = create_spike_train(event_sequence=event_seq, dims=(346,260))
    time = np.arange(0, time_index, 1)
    print(np.asarray(event_tensor_on[0][0]))
    print(time_index)

    xt = np.zeros_like(time)
    for event in event_tensor_on[0][0]:
        xt[event] = 1

    xt = convoluted_signal(time, xt)
    # Créer une nouvelle figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Tracer les événements ON dans le premier sous-graphique
    ax1.eventplot(event_tensor_on[0][0], colors='r')
    ax1.set_title("Événements ON")
    ax1.set_xlim(0, time_index)
    ax1.set_xlabel("Temps")
    ax1.set_ylabel("Amplitude")
    
    # Tracer xt dans le deuxième sous-graphique
    ax2.plot(np.arange(0, len(event_tensor_on[0][0]),1), xt, color='b')
    ax2.set_title("Signal convolué xt")
    ax2.set_xlabel("Temps")
    ax2.set_ylabel("Amplitude")
    
    # Ajuster l'espacement entre les sous-graphiques
    plt.tight_layout()
    plt.show()


    # time = np.arange(0, time_index, 1)
    # #onconv = convoluted_signal(time, event_tensor_on[0][0])
    # #offconv = convoluted_signal(time, np.asarray(event_tensor_off[0][0]))
    # offconv = fft_convolution(np.asarray(event_tensor_off[0][0]), kernel=exponential_decay_kernel(0.05,1))

    # # Création des sous-graphiques
    # fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    # fig.suptitle("Événements OFF et signal convolué")

    # # Tracé des événements OFF
    # axs[0].eventplot(event_tensor_off[0][0], colors='b')
    # axs[0].set_title("Événements OFF")
    # axs[0].set_xlabel("Temps")
    # axs[0].set_ylabel("Amplitude")

    # # Tracé du signal convolué OFF
    # axs[1].plot(time, 1, offconv, color='b')
    # axs[1].set_title("Signal convolué OFF")
    # axs[1].set_xlabel("Temps")
    # axs[1].set_ylabel("Amplitude")

    # plt.tight_layout()
    # plt.show()


    # dist = van_rossum_distance(signal_distance(onconv, offconv), tau)
    # print(dist)
    # plt.subplot(2,1,1)
    # plt.eventplot(event_tensor_on.T.flatten(), colors='r')
    # plt.title("Spikes ON")
    # plt.subplot(2,1,2)
    # plt.eventplot(event_tensor_off.T.flatten(), colors='b')
    # plt.title("Spikes OFF")
    # plt.show()
        
