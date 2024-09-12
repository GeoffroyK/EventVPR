'''
Masking Layer for K-winner take all 
In order to adress the problem of the neuron's membrane potential being reseted on spike with spikingjelly.
GABA Interneuron inhibition PyTorch Implementation from S. Thorpe, 1990.

@GeoffroyK
'''

import torch
import torch.nn as nn

class InhibitionLayer(nn.Module):
    def __init__(self, tau: float, v_threshold: float = 1.0, k: int = 1, refractory_period: int = 0):
        super().__init__()

        self.tau = tau
        self.v_threshold = v_threshold
        self.k = k
        self.v = 0.0

        # According to the litterature 2-3 ms for the refractory period, so we can consider 2 or 3 timesteps, passes
        self.refractory_period = refractory_period
        self.resting_state = None

    def forward(self, x: torch.Tensor):
        y = torch.zeros_like(x)
        self.v = self.v + (x - self.v) / self.tau
        
        # Initialize the resting state of the neurons
        if self.resting_state == None:
            self.resting_state = torch.zeros_like(self.v)
        
        # Get the k_winners neurons
        k_winners = torch.topk(self.v, self.k).indices

        for winner in k_winners:
            if self.v[winner] > self.v_threshold and self.resting_state[winner] == 0:
                y[winner] = 1.
        # Set the output of the k_winners neurons
        #y[k_winners] = torch.where(self.v[k_winners] > self.v_threshold and self.resting_state[k_winners] == torch.zeros(1), 1.0, 0.0)

        # Set the refractory period of the k_winners neurons that have spiked
        #self.resting_state[k_winners] = torch.where(self.v[k_winners] > self.v_threshold, self.refractory_period, 0.0)

        if torch.max(self.v) > self.v_threshold:
            # Reset potentials of the layer to 0
            self.v = torch.zeros_like(self.v)

        # Decrement the refractory period of the neurons that spiked previously
        self.resting_state = torch.where(self.resting_state > 0, self.resting_state - 1, 0)

        return y
