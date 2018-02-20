# Importing some libraries.
from scipy.stats import itemfreq
from collections import Counter
from random import randrange

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os.path
import argparse
import pickle
import random
import copy
import sys


def generate_network(kind, n, m=1):
    """
    Generates the network.

    :param kind: The kind of graph to create (complete, Barabasi, Erdos).
    :param n: The number of nodes in the graph.
    :param m: Either the number of edges in the Barabasi graph or the probability of creating a
              new edge in the Erdos graph.
    """
    # Create the graph based on the given parameters.
    if kind == "complete":
        graph = nx.complete_graph(n)
    elif kind == "barabasi":
        graph = nx.barabasi_albert_graph(n, m)
    elif kind == "er":
        graph = nx.erdos_renyi_graph(n, m)
    else:
        raise ValueError("Invalid graph type '%s' selected. " % kind)
    return graph


class Agent:
    """
        This object represents an agent and its properties such as its opinion e.t.c.
        Each agent is represented by a node in the network.
    """
    def __init__(self, opinion):
        """ Initialises the agent with an initial opinion. """
        self.opinion = opinion


class Simulation:

    network_types = ['complete', 'barabasi', 'er']
    dynamic_methods = ['random', 'degree', 'closeness', 'betweeness', 'flow']

    """ This object is used as the simulation controller. """

    def __init__(self, opinion_states, colors, graph):

        # The network of the simulation.
        self.network = {"graph": graph, "dynamics": None}

        self.opinion_states = opinion_states 	# The available opinion states of the simulation.
        self.colors = colors					# The color for each opinion state.
        self.p = None                           # The fraction of committed agents.
        self.method = None                      # The method of selecting committed agents.
        self.smart = None                       # Flag for committed agents to cleverly choose their next listener.
        self.committed = None                   # List of committed nodes.
        self.centralities = None                # List of centralities.

    @staticmethod
    def get_network_types():
        return Simulation.network_types

    @staticmethod
    def get_dynamic_methods():
        return Simulation.dynamic_methods

    def get_random_speaker(self):
        """ Returns a random speaker from the network. """
        return random.choice(nx.nodes(self.network["graph"]))

    def get_random_listener(self, speaker):
        """
        Returns a random listener for the speaker.

        :param speaker: A node within the network considered to be the speaker.

        """
        return random.choice(self.network["graph"].neighbors(speaker))

    def get_end_point(self, data):
        """ Returns the time point at which the frequency of B drops to zero. """
        t = -1
        for t in range(0, len(data)):
            if data[t, self.opinion_states["B"]] == 0:
                break
        return t

    def get_next_listener(self, speaker):
        """ Returns a next listener for the speaker. """
        if self.smart:
            if self.network["dynamics"][speaker].opinion == "P":
                neighbors = self.network["graph"].neighbors(speaker)
                tobechosen = []
                for neighbour in neighbors:
                    if self.network["dynamics"][neighbour].opinion == "AB" or \
                                    self.network["dynamics"][neighbour].opinion == "B":
                        tobechosen.append(neighbour)

                if len(tobechosen) != 0:
                    centr = self.centralities[tobechosen]
                    index = min(enumerate(centr), key=operator.itemgetter(1))[0]
                    return tobechosen[index]

        return random.choice(self.network["graph"].neighbors(speaker))

    def generate_dynamics(self, p, method, smart=False):
        """ Generates the dynamics of the networks, such as the influence of each node e.t.c. """
        # A list of agents which describe the dynamics of the network.
        agents = []

        # Some more variables.
        graph = self.network["graph"]   # The graph of the network.
        n = nx.number_of_nodes(graph)   # The total number of nodes (agents).
        committed = max(1, int(n * p))  # The total number of committed agents.
        self.p = p                      # The fraction of committed agents.
        self.method = method            # The method of selecting the committed agents.
        self.committed = committed      # Saving the committed to the simulation.
        self.smart = smart              # Flag for committed agents to cleverly choose their next listener.

        # Set the centralities.
        self.centralities = np.asarray(nx.closeness_centrality(graph).values())

        if method == "random":
            # Selecting randomly a fraction of p nodes (agents) to be in the state of P.
            committed_nodes = random.sample(range(0, n), committed)
        elif method == "degree":
            # Selecting the top committed nodes with the highest degree centrality.
            committed_nodes = []
            dc = np.argsort(list(nx.degree_centrality(graph).values()))
            for i in range(0, committed):
                committed_nodes.append(dc[len(dc)-1-i])
        elif method == "closeness":
            # Selecting the top committed nodes with the highest closeness centrality.
            committed_nodes = []
            dc = np.argsort(list(nx.closeness_centrality(graph).values()))
            for i in range(0, committed):
                committed_nodes.append(dc[len(dc)-1-i])
        elif method == "betweenness":
            # Selecting the top committed nodes with the highest betweenness centrality.
            committed_nodes = []
            dc = np.argsort(list(nx.betweenness_centrality(graph).values()))
            for i in range(0, committed):
                committed_nodes.append(dc[len(dc)-1-i])
        elif method == "flow":
            # Selecting the top committed nodes with the highest betweenness centrality.
            committed_nodes = []
            dc = np.argsort(list(nx.current_flow_betweenness_centrality(graph).values()))
            for i in range(0, committed):
                committed_nodes.append(dc[len(dc)-1-i])
        else:
            raise ValueError("Invalid method '%s' for generating the dynamics of the network." % method)

        # For each node of the network...
        for node in range(0, n):
            # If that node has been chosen to be a committed node...
            if node in committed_nodes:
                # Then set its agent to the committed state.
                a = Agent("P")
            else:
                # Else set its agent to the default opinion state B.
                a = Agent("B")
            agents.append(a)

        self.network["dynamics"] = agents

    @staticmethod
    def get_opinions(dynamics):
        """ Returns a list including the opinion of each node. """
        opinions = []
        for agent in dynamics:
            opinions.append(agent.opinion)
        return opinions

    def start(self, t, k_max=1):
        """ Starts the simulation for T time unit steps and then makes the plots. """

        # First make sure that a network is loaded.
        if self.network is None:
            raise ValueError("A network has be generated first. Use the function load_network() for that.")

        # Some variables for simplicity.
        n = nx.number_of_nodes(self.network["graph"])   # Number of nodes.
        s = len(self.opinion_states)					# The total number of opinion states.
        data = np.zeros([t, s])							# Matrix containing frequency at each time step for each opinion.
        diff = np.zeros([t, 1])							# Matrix of two columns (metrics) at each time step.

        # Execute the simulation for K times.
        k = 0
        for k in range(0, k_max):

            sys.stdout.flush()
            print("Completed %d out of %d simulations." % (k, k_max), end='\r')

            # A deep copy of the network.
            dynamics_copy = copy.deepcopy(self.network["dynamics"])

            # Execute T unit time steps.
            for t in range(0, t):

                # Make N simulation time steps.
                for st in range(0, n):
                    self.step(dynamics_copy)

                # Compute some stats.
                """ The frequency of each opinion. """
                opinions = self.get_opinions(dynamics_copy)
                opinion_freq = Counter(opinions)
                for i in self.opinion_states.keys():
                    if i not in opinion_freq:
                        opinion_freq[i] = 0
                    data[t, self.opinion_states[i]] = opinion_freq[i] + data[t, self.opinion_states[i]]

                """ The average difference from the opinion with the greatest number of supporters. """
                diff[t, 0] += np.abs(opinion_freq["A"]-opinion_freq["B"]-opinion_freq["AB"]) / (n-opinion_freq["P"])

            # If we randomly choose the committed nodes, we do it again in order to get
            # a valid result.
            if self.method == "random":
                self.generate_dynamics(self.p, self.method, self.smart)

        print("Completed %d out of %d simulations." % (k+1, k_max))

        # Divide by the number of simulations in order to average the results.
        data = 1.0 * data / k_max
        diff = 1.0 * diff / k_max

        # Get the critical point.
        cp = self.get_critical_point(data)

        # Make the plots.
        self.show(t, data, diff, cp)

    def get_critical_point(self, data):
        """ Returns the critical point at which #A becomes greater than #B. """
        t = None
        for t in range(0, len(data)):
            if data[t, self.opinion_states["A"]] > data[t, self.opinion_states["B"]]:
                break
        return t

    def apply_rule(self, speaker, listener):
        """ Computes the next state of the speaker and the listener. """

        # Get the opinion of the speaker.
        opinion = Simulation.get_opinion(speaker)

        # Get the opinion of the listener as a vector.
        listeners_vector = Simulation.get_vector(listener)

        # Compute the next state of the speaker.
        if speaker != 3:
            if listeners_vector[opinion] == 1:
                speakers_vector = Simulation.get_vector(speaker)
                speakers_vector[1-opinion] = 0
                speakers_vector[opinion] = 1
                new_speaker = Simulation.get_status(speakers_vector)
            else:
                new_speaker = speaker
        else:
            new_speaker = speaker

        # Compute the next state of the listener.
        if listener != 3:
            listeners_vector[1-opinion] = 1 - listeners_vector[opinion]
            listeners_vector[opinion] = 1
            new_listener = Simulation.get_status(listeners_vector)
        else:
            new_listener = listener

        # Map the integer values to the corresponding keys.
        new_speaker = self.get_key_by_value(new_speaker)
        new_listener = self.get_key_by_value(new_listener)

        return new_speaker, new_listener

    @staticmethod
    def get_opinion(status):
        """ Returns the expressed opinion of the agent. """
        if status == 3: 				# P
            return 0
        if status == 2: 				# AB
            return random.randint(0, 1)
        return status 					# Uncommitted A or B

    @staticmethod
    def get_vector(status):
        """ Returns the expressed opinion of the agent as a vector. """
        if status == 1: 				# B
            return [0, 1]
        if status == 2: 				# AB
            return [1, 1]
        return [1, 0] 					# A or P

    @staticmethod
    def get_status(vec):
        """ Maps the vector opinion to a standard opinion. """
        if vec == [1, 0]: 				# A
            return 0
        if vec == [0, 1]: 				# B
            return 1
        if vec == [1, 1]: 				# AB
            return 2

    def step(self, dynamics):
        """
        Makes a step in the simulation.

        :param dynamics: The generated dynamics.

        """
        # Select a random speaker.
        speaker = self.get_random_speaker()

        # Select a random listener.
        listener = self.get_next_listener(speaker)

        # Update the state of the listener and the speaker.
        speaker_ns, listener_ns = self.apply_rule(self.opinion_states[dynamics[speaker].opinion],
                                                  self.opinion_states[dynamics[listener].opinion])

        # Set the new states.
        dynamics[speaker].opinion = speaker_ns
        dynamics[listener].opinion = listener_ns

    @staticmethod
    def get_key_by_value(x):
        """ Does the mapping from the value to the key. """
        if x == 0:
            return "A"
        elif x == 1:
            return "B"
        elif x == 2:
            return "AB"
        return "P"

    def show(self, t, data, diff, cp):
        """ Creates interesting plots of the simulation's data. """
        # Get the number of nodes.
        n = nx.number_of_nodes(self.network["graph"])  # Number of nodes.

        # Create the frequency plot of each party.
        plt.subplot(221)
        plt.axis('on')
        plt.xlim([0, t+1])
        for state in self.opinion_states:
            plt.plot(np.arange(t + 1), data[0:t + 1, int(1.0 * self.opinion_states[state])] / n, color=self.colors[state],
                     label=state)
            plt.ylabel('frequency')
            plt.xlabel('time step')
        plt.legend()

        # Create the phase space plot.
        plt.subplot(212)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.plot(1.0 * data[0:t+1, self.opinion_states['A']] / (n - self.committed),
                 1.0 * data[0:t+1, self.opinion_states['B']] / (n - self.committed))
        plt.ylabel('n_B')
        plt.xlabel('n_A')

        # Create the decisiveness plot.
        plt.subplot(222)
        plt.ylim([0, 1])
        plt.xlim([0, t+1])
        plt.axis('on')
        plt.plot(np.arange(cp), diff[0:cp, 0], 'blue', label='#B > #A')
        plt.plot(np.arange(start=cp - 1, stop=t), diff[cp - 1:t, 0], 'red', label='#B < #A')
        plt.legend()
        plt.ylabel('decisiveness')
        plt.xlabel('time step')
        plt.show()


if __name__ == "__main__":
    # Parse the arguments (if any).
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', type=int, default=250, help="The total number of unit time steps.")
    parser.add_argument('-N', type=int, default=100, help="The total number of nodes.")
    parser.add_argument('-k', type=int, default=50, help="Number of times to repeat the simulation.")
    parser.add_argument('-m', type=int, default=4, help="The average number of neighbours to each node.")
    parser.add_argument('-p', type=float, default=0.07, help="The fraction of the nodes which are committed.")
    parser.add_argument('-kind', type=str, default='complete', choices=Simulation.get_network_types(),
                        help="The kind of graph structure to use.")
    parser.add_argument('-method', type=str, default='degree', choices=Simulation.get_dynamic_methods(),
                        help="The method by which the selection of committed nodes is performed.")
    parser.add_argument('-s', default=False, action='store_true', help="Flag for speakers to be smart.")
    parser.add_argument('-w', action='store_true')
    args = parser.parse_args()

    # Define the available colors that can be used for each opinion state.
    c = {"RED": [1, 0, 0], "BLUE": [0, 0, 1], "GREEN": [0, 1, 0], "PURPLE": [1, 0, 1]}

    # Define the possible opinion states by an integer.
    opinions_states = {"A": 0, "B": 1, "AB": 2, "P": 3}

    # Define the color of each opinion state.
    opinion_colors = {"A": c["RED"], "B": c["BLUE"], "AB": c["GREEN"], "P": c["PURPLE"]}

    # Generate the graph of the network.
    g = generate_network(args.kind, args.N, args.m)

    # Initialise the simulation.
    simulation = Simulation(opinions_states, opinion_colors, g)

    # Generate a network for the simulation given its graph parameters.
    simulation.generate_dynamics(args.p, args.method, args.s)

    # Execute the simulation for T steps for k times.
    simulation.start(args.T, args.k)
