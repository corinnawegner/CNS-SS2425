import matplotlib.pyplot as plt
import numpy as np

class Neuron:
    def __init__(self, constant_current, mode = "periodic", total_spiking_time = 10, delay = 0):
        self.constant_current = constant_current
        self.rate = constant_current
        # Needed for ISO rule, generated via generate_spike_train():
        self.spike_train = None
        self.convolved_spike_train = None
        self.derived_spike_train = None
        self.mode = mode
        self.total_spiking_time = total_spiking_time
        self.delay = delay

    def update_rate(self, presynaptic_weights, input_rates):
        self.rate = np.sum(np.array(presynaptic_weights)*np.array(input_rates)) + self.constant_current

    def generate_spike_train(self, delta_t):
        spike_train = []
        if self.mode == "periodic":
            # Calculate the interval between spikes in terms of steps
            spike_interval = int(1 / self.rate / delta_t)  # interval in number of steps

            for i in np.arange(0, self.total_spiking_time, delta_t):
                if (i // delta_t) % spike_interval + self.delay == 0:  # Check if it's time for a spike
                    spike_train.append(1)
                else:
                    spike_train.append(0)
        self.spike_train = np.array(spike_train)
        input_trace = np.array([double_exponential(t) for t in np.linspace(0, 2, int(2/delta_t))])
        self.convolved_spike_train = np.convolve(self.spike_train, input_trace)
        self.derived_spike_train = np.gradient(self.spike_train)

class Synapse:
    def __init__(self, in_neuron, out_neuron, learning_rate, weight=0):
        self.weight = weight
        self.learning_rate = learning_rate
        self.in_neuron = in_neuron
        self.out_neuron = out_neuron
        self.mean_rate_in = 0 # Only needed for Covariance rule
        self.mean_rate_out = 0
        #self.convolved_input = self.in_neuron.spike_train
        self.output_spike_train = self.out_neuron.spike_train

    def update_weight(self, learning_rule="hebb", delta_t=0.1, t=0):
        rate_in = self.in_neuron.rate
        rate_out = self.out_neuron.rate
        if learning_rule == "hebb":
            self.weight += self.hebb_rule(rate_in, rate_out, delta_t)
        elif learning_rule == "bcm":
            self.weight += self.bcm_rule(rate_in, rate_out, delta_t)
        elif learning_rule == "oja":
            self.weight += self.oja_rule(rate_in, rate_out, delta_t)
        elif learning_rule == "cov":
            self.weight += self.covariance_rule(rate_in, rate_out, delta_t)
        elif learning_rule == "iso":
            self.weight += self.iso_rule(t, delta_t)
        else:
            raise ValueError(f"Unknown learning rule: {learning_rule}")

    def hebb_rule(self, rate_in, rate_out, delta_t):
        delta_weight = self.learning_rate * rate_in * rate_out * delta_t
        return delta_weight

    def bcm_rule(self, rate_in, rate_out, delta_t):
        delta_weight = self.learning_rate * rate_in * rate_out * delta_t * (rate_out - 0.3)
        return delta_weight

    def oja_rule(self, rate_in, rate_out, delta_t):
        delta_weight = self.learning_rate*(rate_in * rate_out - self.weight * rate_out**2) * delta_t
        return delta_weight

    def covariance_rule(self, rate_in, rate_out, delta_t):
        q = 0.1
        # Assuming the new mean rate has already been changed this time step
        mean_rate_in_old = self.mean_rate_in
        mean_rate_out_old = self.mean_rate_out
        rate_i = q * self.out_neuron.rate + (1-q) * mean_rate_out_old
        rate_j = q * self.in_neuron.rate + (1-q) * mean_rate_in_old
        self.mean_rate_in = rate_j
        self.mean_rate_out = rate_i
        delta_weight = self.learning_rate * (rate_out - rate_i) * (rate_in - rate_j) * delta_t
        #print(mean_rate_in_old, mean_rate_out_old, rate_i, rate_j, delta_weight, self.out_neuron.rate, self.in_neuron.rate)
        return delta_weight

    def iso_rule(self, t, delta_t):
        if self.in_neuron.convolved_spike_train is None:
            self.in_neuron.generate_spike_train(delta_t)
        if self.out_neuron.derived_spike_train is None:
            self.out_neuron.generate_spike_train(delta_t)
        u = self.in_neuron.convolved_spike_train
        dv = self.out_neuron.derived_spike_train

        t_int = int(t/delta_t) - 1
        print("t_int", t_int)

        delta_weight = self.learning_rate * dv[t_int] * u[t_int] * delta_t
        return delta_weight

def double_exponential(t):
    H = 0.2
    alpha = 1 / 20 * 10 ** 2
    beta = 1 / 2 * 10 ** 2
    #h_sequence = [double_exponential(t) for t in np.arange(0, 10, 100)]
    return H * (np.exp(-alpha*t)-np.exp(-beta*t)) if t > 0 else 0#, h_sequence

"""
time = np.linspace(0,2, 1000)

list_de = [double_exponential(t) for t in time]

print(time,list_de)
plt.plot(time, list_de)
plt.show()
"""