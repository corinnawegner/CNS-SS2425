import matplotlib.pyplot as plt
import numpy as np


class Neuron:
    def __init__(self, constant_current, mode="periodic", total_spiking_time=10, delay=0, stop_after=None):
        """
        Initializes a neuron with given parameters.

        :param constant_current: The base rate of the neuron without any inputs.
        :param mode: The mode of spiking behavior ("periodic" or "single").
        :param total_spiking_time: Total duration of spiking activity.
        :param delay: Delay before spiking begins (in seconds).
        :param stop_after: Time after which spiking stops (optional).
        """
        self.constant_current = constant_current
        self.rate = constant_current
        self.in_synapsis = [] #List of synapses preceding the Neuron
        self.rates = [constant_current]
        # Needed for ISO and ICO rule, generated via generate_spike_train():
        self.spike_train = None
        self.convolved_spike_train = None
        self.derived_spike_train = None
        self.mode = mode
        self.total_spiking_time = total_spiking_time
        self.delay = delay
        self.stop_after = stop_after
        self.inputs = []  # input spike trains coming from preceding Neurons

    def update_rate(self, presynaptic_weights, input_rates):
        """
        Updates the rate of the neuron based on weighted inputs.

        :param presynaptic_weights: Weights of the presynaptic inputs.
        :param input_rates: Rates of the presynaptic neurons.
        """
        self.rate = np.sum(np.array(presynaptic_weights) * np.array(input_rates)) + self.constant_current
        self.rates.append(self.rate)

    def generate_output_spike_train(self, presynaptic_weights, input_spike_trains):
        """
        Creates a spike train as a response of input spike trains weighted by the synaptical weight.
        :param presynaptic_weights: weights of input synapses
        :param input_spike_trains: convolved spike trains from the input neurons
        :return: Updates the spike train and derived spike train
        """
        output = np.sum(np.array([input_spike_trains[i] * presynaptic_weights[i] for i in range(len(input_spike_trains))]), axis=0)
        self.spike_train = output
        self.derived_spike_train = [output[i] - output[i-1] for i in range(1, len(output))]

    def generate_spike_train(self, delta_t, plot_spike = False):
        """
        Generates a spike train for the neuron based on its rate and mode (without preceding neurons).

        :param delta_t: Time step for simulation.
        :param plot_spike: Whether to plot the generated spike train.
        """
        len_train = int(self.total_spiking_time / delta_t) # Compute the length of the array for the spike train given dt and total simulation time
        if self.rate != 0: # Avoid infinity in case the rate is 0
            spike_interval = int(1 / (self.rate * delta_t))
        else:
            spike_interval = 0
        spike_train = np.zeros(len_train)
        if self.mode == "periodic":
            for i in range(0, len_train, spike_interval):
                i_delay = i + int(self.delay / delta_t) #
                if self.stop_after is not None and self.stop_after < i_delay * delta_t:
                    break
                spike_train[i_delay] = 1
        elif self.mode == "single":
            spike_train[0] = 1
        else:
            raise ValueError(f"Unknown spike mode {self.mode}")

        # print(self.rate*self.total_spiking_time==np.sum(spike_train))
        self.spike_train = spike_train
        input_trace = np.array([double_exponential(t) for t in np.linspace(0, 2, int(2 / delta_t))])
        self.convolved_spike_train = np.convolve(self.spike_train, input_trace)

        if plot_spike == True:
            # Plotting the spike train
            plt.figure(figsize=(10, 4))
            time = np.arange(0, self.total_spiking_time, delta_t)
            plt.plot(time, self.spike_train, drawstyle='steps-post', label='Spike Train')
            plt.title(f'Spike Train for delay: {self.delay}')
            plt.xlabel('Time (s)')
            plt.ylabel('Spike (binary)')
            plt.legend()
            plt.grid(True)
            plt.show()


class Synapse:
    def __init__(self, in_neuron, out_neuron, learning_rate, weight=0):
        """
        Initializes a synapse connecting two neurons.

        :param in_neuron: The presynaptic neuron.
        :param out_neuron: The postsynaptic neuron.
        :param learning_rate: The rate at which the synapse learns.
        :param weight: Initial synaptic weight.
        """
        self.weight = weight
        self.learning_rate = learning_rate
        self.in_neuron = in_neuron # Neuron preceding the synapse
        self.out_neuron = out_neuron # Neuron following the synapse
        self.mean_rate_in = 0  # Only needed for Covariance rule
        self.mean_rate_out = 0
        out_neuron.in_synapsis.append(self) # Add itself to the list of incoming synapses from the Neuron following self


    def parse_spike_train(self):
        """
        Parses the spike train from the input neuron and adds it to the output neuron's inputs.
        """
        self.out_neuron.inputs.append(self.in_neuron.convolved_spike_train)

    def update_weight(self, learning_rule="hebb", delta_t=0.1, t=0):
        """
        Updates the synaptic weight using a specified learning rule.

        :param learning_rule: The rule to use for weight updates (e.g., "hebb", "bcm").
        :param delta_t: Time step for simulation.
        :param t: Current simulation time.
        """
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
        elif learning_rule == "ico":
            self.weight += self.ico_rule(t, delta_t)
        elif learning_rule == "ico2":
            self.weight += self.ico_with_rates(rate_in, delta_t)
        else:
            raise ValueError(f"Unknown learning rule: {learning_rule}")

    def hebb_rule(self, rate_in, rate_out, delta_t):
        """
        Hebbian learning rule.

        :param rate_in: Input rate from presynaptic neuron.
        :param rate_out: Output rate from postsynaptic neuron.
        :param delta_t: Time step for simulation.
        :return: Weight change.
        """
        delta_weight = self.learning_rate * rate_in * rate_out * delta_t
        return delta_weight

    def bcm_rule(self, rate_in, rate_out, delta_t):
        """
        BCM (Bienenstock, Cooper, Munro) learning rule.

        :param rate_in: Input rate from presynaptic neuron.
        :param rate_out: Output rate from postsynaptic neuron.
        :param delta_t: Time step for simulation.
        :return: Weight change.
        """
        delta_weight = self.learning_rate * rate_in * rate_out * delta_t * (rate_out - 0.3)
        return delta_weight

    def oja_rule(self, rate_in, rate_out, delta_t):
        """
        Oja's learning rule.

        :param rate_in: Input rate from presynaptic neuron.
        :param rate_out: Output rate from postsynaptic neuron.
        :param delta_t: Time step for simulation.
        :return: Weight change.
        """
        delta_weight = self.learning_rate * (rate_in * rate_out - self.weight * rate_out ** 2) * delta_t
        return delta_weight

    def covariance_rule(self, rate_in, rate_out, delta_t):
        """
        Covariance rule
        :param rate_in: Input rate from presynaptic neuron.
        :param rate_out: Output rate from postsynaptic neuron.
        :param delta_t: Time step for simulation.
        :return: Weight change.
        """
        q = 0.1
        # Assuming the new mean rate has already been changed this time step
        mean_rate_in_old = self.mean_rate_in
        mean_rate_out_old = self.mean_rate_out
        rate_i = q * self.out_neuron.rate + (1 - q) * mean_rate_out_old
        rate_j = q * self.in_neuron.rate + (1 - q) * mean_rate_in_old
        self.mean_rate_in = rate_j
        self.mean_rate_out = rate_i
        delta_weight = self.learning_rate * (rate_out - rate_i) * (rate_in - rate_j) * delta_t
        # print(mean_rate_in_old, mean_rate_out_old, rate_i, rate_j, delta_weight, self.out_neuron.rate, self.in_neuron.rate)
        return delta_weight

    def iso_rule(self, t, delta_t):
        """
        ISO rule
        :param t: Current simulation time (absolute)
        :param delta_t: Time step for simulation.
        :return: Weight change.
        """
        if self.in_neuron.convolved_spike_train is None: # Generate a spike train for the input neuron if it is not existing
            self.in_neuron.generate_spike_train(delta_t)
        u = self.in_neuron.convolved_spike_train
        dv = self.out_neuron.derived_spike_train

        t_int = int(t / delta_t) - 1 # Compute index of the current simulation time within the array

        delta_weight = self.learning_rate * dv[t_int] * u[t_int]
        return delta_weight

    def ico_rule(self, t, delta_t):
        """
        ICO rule
        :param t: Current simulation time (absolute)
        :param delta_t: Time step for simulation.
        :return: Weight change.
        """
        if self.in_neuron.convolved_spike_train is None: # Generate a spike train for the input neuron if it is not existing
            self.in_neuron.generate_spike_train(delta_t)
        if not self.out_neuron.inputs: # Get the spike train from the parallel input Neuron (with fixed weights)
            for synapse in self.out_neuron.in_synapsis:
                if synapse is not self:
                    synapse.parse_spike_train()

        u = self.in_neuron.convolved_spike_train

        t_int = int(t / delta_t) - 1 # Compute index of the current simulation time within the array

        u_const = self.out_neuron.inputs[0]
        d_u_const = [u_const[i] - u_const[i-1] for i in range(1, len(u_const))] # Derive spike train from parallel input Neuron

        delta_weight = self.learning_rate * d_u_const[t_int] * u[t_int]
        return delta_weight

    def ico_with_rates(self, rate_in, delta_t):
        """
        ICO rule using the firing rates (not spike trains). Used in 2.4

        :param rate_in: Input rate from presynaptic neuron.
        :param delta_t: Time step for simulation.
        :return: Weight change.
        """
        for synapse in self.out_neuron.in_synapsis:
            if synapse is not self:
                rate_in_other_new = synapse.in_neuron.rate
                rate_in_other_old = synapse.in_neuron.rates[-2]

        dt_rate_other = rate_in_other_new - rate_in_other_old

        delta_weight = self.learning_rate * dt_rate_other * rate_in * delta_t
        return delta_weight

def double_exponential(t):
    """
    Eligibility trace for spikes
    :param t: time
    :return: Double exponential function value
    """
    H = 0.2
    alpha = 1 / 20 * 10 ** 3
    beta = 1 / 2 * 10 ** 3
    return H * (np.exp(-alpha * t) - np.exp(-beta * t)) if t > 0 else 0