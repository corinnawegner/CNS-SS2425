import numpy as np

class Neuron:
    def __init__(self, constant_current):
        self.constant_current = constant_current
        self.rate = constant_current
        self.all_rates = [constant_current]

    def update_rate(self, presynaptic_weights, input_rates):
        self.rate = np.sum(np.array(presynaptic_weights)*np.array(input_rates)) + self.constant_current
        self.all_rates.append(self.rate)

class Synapse:
    def __init__(self, in_neuron, out_neuron, learning_rate, weight=0):
        self.weight = weight
        self.learning_rate = learning_rate
        self.in_neuron = in_neuron
        self.out_neuron = out_neuron

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
        rate_i = q * self.out_neuron.all_rates[-1] + (1-q) * self.out_neuron.all_rates[-2]
        rate_j = q * self.in_neuron.all_rates[-1] + (1-q) * self.in_neuron.all_rates[-2]
        delta_weight = self.learning_rate * (rate_out - rate_i) * (rate_in - rate_j) * delta_t
        return delta_weight

