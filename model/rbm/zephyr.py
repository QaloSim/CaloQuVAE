"""
PyTorch implementation of a quadripartite Boltzmann machine with a 
Zephyr/Advantage2 QPU topology
"""
import dwave_networkx as dnx
import networkx as nx
from dwave.system import DWaveSampler
import random
# from hybrid.decomposers import _chimeralike_to_zephyr

import itertools
import math

import torch
from torch import nn
import os
import pickle

from CaloQVAE import logging
logger = logging.getLogger(__name__)

class ZephyrRBM(nn.Module):
    def __init__(self, cfg=None):
        super(ZephyrRBM, self).__init__()
        self._config=cfg

        # RBM constants
        self._n_partitions = self._config.rbm.partitions
        self._nodes_per_partition = self._config.rbm.latent_nodes_per_p

        # Emtpy RBM parameters
        self._weight_dict = nn.ParameterDict()
        self._bias_dict = nn.ParameterDict()
        self._weight_mask_dict = nn.ParameterDict()

        # Dict of RBM weights for different partition combinations
        for i,key in enumerate(itertools.combinations(range(self._n_partitions), 2)):
            str_key = ''.join([str(key[i]) for i in range(len(key))])
            self._weight_dict[str_key] = nn.Parameter(
                torch.randn(self._nodes_per_partition,
                            self._nodes_per_partition)*self._config.rbm.w_std[i], requires_grad=True)

        # Dict of RBM biases for each partition
        for i in range(self._n_partitions):
            self._bias_dict[str(i)] = nn.Parameter(
                torch.randn(self._nodes_per_partition)*self._config.rbm.b_std[i], requires_grad=True)
            
        if self._config.rbm.fullyconnected:
            for key in itertools.combinations(range(self._n_partitions), 2):
                str_key = ''.join([str(key[i]) for i in range(len(key))])
                self._weight_mask_dict[str_key] = nn.Parameter(torch.ones(self._nodes_per_partition, self._nodes_per_partition), requires_grad=False)
        else:
            for key in itertools.combinations(range(self._n_partitions), 2):
                str_key = ''.join([str(key[i]) for i in range(len(key))])
                self._weight_mask_dict[str_key] = nn.Parameter(torch.zeros(self._nodes_per_partition, self._nodes_per_partition), requires_grad=False)
            self.gen_qubit_idx_dict()
            self.gen_weight_mask_dict()

    @property
    def nodes_per_partition(self):
        """Accessor method for a protected variable

        :return: no. of latent nodes per partition
        """
        return self._nodes_per_partition

    @property
    def weight_dict(self):
        """Accessor method for a protected variable

        :return: dict with partition combinations as str keys ('01', '02', ...)
                 and partition weight matrices as values (w_01, w_02, ...)
        """
        # if self._qpu:
        for key in self._weight_dict.keys():
            self._weight_dict[key] = self._weight_dict[key] \
                * self._weight_mask_dict[key]
        return self._weight_dict


    @property
    def bias_dict(self):
        """Accessor method for a protected variable

        :return: dict with partition no.s as str keys ('0', '1', ...)
                 and partition biases as values (b_0, b_1, ...)
        """
        return self._bias_dict

    def gen_qubit_idx_dict(self):
        """Partition the qubits on the device into 4-partite BM

        Uses a Greedy Edge Addition strategy to optimize connectivity

        :return: dict with partition no.s as str keys ('0', '1', ...)
                 and list of qubit idxs for each partition as values
        """
        self.load_coordinates()

        # Maps qubit coordinates to partition idxs
        qubit_to_partition_map = {}
        for q in self.coordinated_graph.nodes:
            partition_idx = str((2*q[0] + q[1] + 2*q[4] + q[3]) % 4)
            qubit_idx = self.coordinates_to_idx(q, self.m, self.t)
            qubit_to_partition_map[qubit_idx] = partition_idx
        
        # Create list of cross-partition edges
        cross_partition_edges = []
        for u, v in self._qpu_sampler.edgelist:
            if u in qubit_to_partition_map and v in qubit_to_partition_map:
                if qubit_to_partition_map[u] != qubit_to_partition_map[v]:
                    cross_partition_edges.append((u, v))
        random.shuffle(cross_partition_edges)

        # Greedy Edge Addition
        selected_qubits_sets = {str(i): set() for i in range(self._n_partitions)}

        for u, v in cross_partition_edges:
            p_u = qubit_to_partition_map[u]
            p_v = qubit_to_partition_map[v]

            if len(selected_qubits_sets[p_u]) < self._nodes_per_partition and len(selected_qubits_sets[p_v]) < self._nodes_per_partition:
                selected_qubits_sets[p_u].add(u)
                selected_qubits_sets[p_v].add(v)
        
        # Convert sets to lists and create idx_dict
        self.idx_dict = {p_idx: sorted(list(qubits)) for p_idx, qubits in selected_qubits_sets.items()}


    def gen_weight_mask_dict(self):
        """Generate the weight mask for each partition-pair

        :param qubit_idx_dict (dict): Dict with partition no.s as keys and
        list of qubit idxs for each partition as values
        ;param device (DWaveSampler): QPU device containing list of nodes and 
        edges

        :return weight_mask_dict (dict): Dict with partition combinations as
        keys and weight mask for each combination as values
        """

        for i, partition_a in enumerate(self.idx_dict.keys()):
            for qubit_a in self.idx_dict[partition_a]:
                for qubit_b in self.adjacency[qubit_a]:
                    for partition_b in list(self.idx_dict.keys())[i:]:
                        if qubit_b in self.idx_dict[partition_b]:
                            weight_idx = partition_a + partition_b
                            idx_a = self.idx_dict[partition_a].index(qubit_a)
                            idx_b = self.idx_dict[partition_b].index(qubit_b)
                            self._weight_mask_dict[weight_idx][idx_a,idx_b] = 1.0


    def load_coordinates(self):
        try:
            self._qpu_sampler = DWaveSampler(solver={'topology__type': 'zephyr', 'chip_id':'Advantage2_system1.3'})
            self.m, self.t = self._qpu_sampler.properties['topology']['shape']
            graph = dnx.zephyr_graph(m=self.m, t=self.t,
                                node_list=self._qpu_sampler.nodelist, edge_list=self._qpu_sampler.edgelist)
            self.coordinated_graph = nx.relabel_nodes(
                    graph,
                    {q: dnx.zephyr_coordinates(self.m,self.t).linear_to_zephyr(q)
                    for q in graph.nodes})
            self.adjacency = self._qpu_sampler.adjacency
        except:
            logger.warn("QPU is offline. Setting a hard-coded zephyr. " \
                    "Check to see you're pinging the correct chip_id"
                    )
            self.m, self.t = 12,4
            print(os.getcwd())
            with open(os.getcwd() + '/model/rbm/nodelist.pickle', 'rb') as handle:
                self.nodelist = pickle.load(handle)
            with open(os.getcwd() + '/model/rbm/edgelist.pickle', 'rb') as handle:
                self.edgelist = pickle.load(handle)
            with open(os.getcwd() + '/model/rbm/adjacency.pickle', 'rb') as handle:
                self.adjacency = pickle.load(handle)
            graph = dnx.zephyr_graph(m=self.m, t=self.t,
                                node_list=self.nodelist, edge_list=self.edgelist)
            self.coordinated_graph = nx.relabel_nodes(
                    graph,
                    {q: dnx.zephyr_coordinates(self.m,self.t).linear_to_zephyr(q)
                    for q in graph.nodes})
            
        
    def coordinates_to_idx(self, q, m, t):
        return q[4] + m*(q[3] + 2*(q[2] + t*(q[1]+(2*m+1)*q[0])))

if __name__ == "__main__":
    rbm = ZephyrRBM()