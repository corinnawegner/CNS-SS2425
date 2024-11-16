import numpy as np

class Neuron:
    def __init__(self, constant_current):
        self.constant_current = constant_current
        self.rate = constant_current

    def update_rate(self, presynaptic_weights, input_rates):
        self.rate = np.sum(np.array(presynaptic_weights)*np.array(input_rates)) + self.constant_current

    def generate_spike_train(self, total_time, delta_t, delay = 0, mode = "periodic"):
        spike_train = []
        for i in range(0, total_time, delta_t):
            if mode == "periodic":
                # Calculate the interval between spikes in terms of steps
                spike_interval = int(1 / self.rate / delta_t)  # interval in number of steps

                for i in range(0, total_time, delta_t):
                    if (i // delta_t) % spike_interval + delay == 0:  # Check if it's time for a spike
                        spike_train.append(1)
                    else:
                        spike_train.append(0)

        return spike_train



class Synapse:
    def __init__(self, in_neuron, out_neuron, learning_rate, weight=0):
        self.weight = weight
        self.learning_rate = learning_rate
        self.in_neuron = in_neuron
        self.out_neuron = out_neuron
        self.mean_rate_in = 0 # Only needed for Covariance rule
        self.mean_rate_out = 0

    def update_weight(self, learning_rule="hebb", delta_t=0.1):
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

    def iso_rule(self, input_spike_train, output_spike_train, t, delta_t):
        H = 0.2
        alpha = 1/20 * 10**2
        beta = 1/2 * 10**2
        h = lambda t: H * (np.exp(-alpha*t)-np.exp(-beta*t)) if t > 0 else 0

        t_tot = len(input_spike_train)/delta_t

        h_sequence = [h(t) for t in range(0, t_tot, delta_t)]
        u = np.convolve(input_spike_train, h_sequence)
        h_out = np.convolve(output_spike_train, h_sequence) #Todo: is this necessary?

        dv = np.gradient(h_out)

        delta_weight = self.learning_rate * dv[t*delta_t] * u[t*delta_t] * delta_t
        return delta_weight