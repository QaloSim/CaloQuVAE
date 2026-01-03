"""
PyTorch implementation of a quadripartite Boltzmann machine with a 
Zephyr/Advantage2 QPU topology.

This module defines the ZephyrRBM class, which implements a Restricted Boltzmann Machine (RBM)
with a topology inspired by the D-Wave Zephyr architecture. The RBM is partitioned into multiple
groups, and the connectivity between partitions is determined by the Zephyr topology or set to fully
connected, depending on configuration.

Classes:
    ZephyrRBM: Main class implementing the Zephyr topology RBM.

Typical usage example:
    config = ...  # Load or define your configuration object
    rbm = ZephyrRBM(config)
"""
import dwave_networkx as dnx
import networkx as nx
from dwave.system import DWaveSampler
import random
import numpy as np
# from hybrid.decomposers import _chimeralike_to_zephyr

import itertools
import math

import torch
from torch import nn
import os
import pickle

import CaloQuVAE
from CaloQuVAE import logging
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
            logger.info("RBM is configured to be fully connected.")
            for key in itertools.combinations(range(self._n_partitions), 2):
                str_key = ''.join([str(key[i]) for i in range(len(key))])
                self._weight_mask_dict[str_key] = nn.Parameter(torch.ones(self._nodes_per_partition, self._nodes_per_partition), requires_grad=False)
        elif hasattr(self._config.rbm, 'no_weights') and self._config.rbm.no_weights:
            for key in itertools.combinations(range(self._n_partitions), 2):
                str_key = ''.join([str(key[i]) for i in range(len(key))])
                self._weight_mask_dict[str_key] = nn.Parameter(torch.zeros(self._nodes_per_partition, self._nodes_per_partition), requires_grad=False)
            logger.info("RBM is configured to have no weights.")
        else:
            logger.info("RBM is configured to use Zephyr topology.")
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

    
        for p_idx, qubits in selected_qubits_sets.items():
            if len(qubits) < self._nodes_per_partition:
                logger.warn(f"Partition {p_idx} was under-filled: "
                            f"{len(qubits)}/{self._nodes_per_partition} qubits selected.")
            logger.info(f"Partition {p_idx} contains {len(qubits)} qubits")

    def gen_qubit_idx_dict(self, max_iterations=50):
        """
        Selects qubits using an iterative refinement strategy.
        Starts with a naive guess and improves it by swapping nodes.
        """
        self.load_coordinates()
        
        # Partition ALL qubits first based on their coordinates.
        # This is the common starting point for both the optimized and non-optimized cases.
        full_partitions = {str(i): [] for i in range(self._n_partitions)}
        for q_coord in self.coordinated_graph.nodes:
            partition_id = str((2 * q_coord[0] + q_coord[1] + 2 * q_coord[4] + q_coord[3]) % 4)
            qubit_idx = self.coordinates_to_idx(q_coord, self.m, self.t)
            full_partitions[partition_id].append(qubit_idx)
        
        selected_qubits = {}
        discarded_qubits = {}
            
        try:
            # Build a graph of the active hardware to get 2D positions
            node_list = getattr(self._qpu_sampler, 'nodelist', self.nodelist)
            edge_list = getattr(self._qpu_sampler, 'edgelist', self.edgelist)
            G_all = dnx.zephyr_graph(self.m, self.t, node_list=node_list, edge_list=edge_list)
            pos = dnx.zephyr_layout(G_all)

            # Chip center (use bbox midpoint to be robust to holes)
            xs, ys = zip(*[pos[n] for n in G_all.nodes])
            cx = 0.5 * (min(xs) + max(xs))
            cy = 0.5 * (min(ys) + max(ys))

            def dist2(n):
                x, y = pos[n]
                return (x - cx) ** 2 + (y - cy) ** 2

            # For each partition, pick the qubits closest to the chip's center
            for p_id, q_list in full_partitions.items():
                if len(q_list) <= self._nodes_per_partition:
                    selected_qubits[p_id] = list(q_list)
                    discarded_qubits[p_id] = []
                else:
                    centered = sorted(q_list, key=dist2)  # uses layout-based distance
                    selected_qubits[p_id] = centered[:self._nodes_per_partition]
                    discarded_qubits[p_id] = centered[self._nodes_per_partition:]

        except Exception as _e:
            # Fallback: pure "middle slice" by index in each partition's list
            logger.warn(f"Could not use chip layout for qubit selection. Falling back to index-based selection. Error: {_e}")
            for p_id, q_list in full_partitions.items():
                k = min(self._nodes_per_partition, len(q_list))
                start = max(0, (len(q_list) - k) // 2)
                end = start + k
                selected_qubits[p_id] = q_list[start:end]
                discarded_qubits[p_id] = q_list[:start] + q_list[end:]
        
        # --- Conditional Optimization ---
        # If optimization is enabled, perform the iterative refinement to maximize connectivity.
        # Otherwise, we just use the centrally-located qubits selected above.
        if self._config.rbm.optimize_partition:
            # For faster lookups, create a single set of all selected qubits
            selected_set = set(q for q_list in selected_qubits.values() for q in q_list)

            logger.info("Starting iterative refinement to maximize connectivity...")
            for i in range(max_iterations):
                # Calculate realized connectivity for all selected nodes
                realized_connectivity = {}
                current_total_edges = 0
                for qubit in selected_set:
                    neighbors_in_set = sum(1 for neighbor in self._qpu_sampler.adjacency[qubit] if neighbor in selected_set)
                    realized_connectivity[qubit] = neighbors_in_set
                    current_total_edges += neighbors_in_set
                
                current_total_edges /= 2 # Divide by 2 as each edge is counted twice
                logger.info(f"Iteration {i+1}/{max_iterations} | Current Edges: {current_total_edges:.0f}")

                best_swap = {'gain': 0, 'out_qubit': None, 'in_qubit': None, 'partition': None}

                # Find the best possible swap
                for p_id in selected_qubits.keys():
                    for out_qubit in selected_qubits[p_id]:
                        for in_qubit in discarded_qubits[p_id]:
                            # Calculate connectivity loss and gain
                            loss = realized_connectivity[out_qubit]
                            gain = sum(1 for neighbor in self._qpu_sampler.adjacency[in_qubit] if neighbor in selected_set)
                            net_gain = gain - loss
                            
                            if net_gain > best_swap['gain']:
                                best_swap.update({
                                    'gain': net_gain,
                                    'out_qubit': out_qubit,
                                    'in_qubit': in_qubit,
                                    'partition': p_id
                                })
                
                # Perform the swap if it's beneficial
                if best_swap['gain'] > 2: # Using a threshold to ensure meaningful improvement
                    p_id, out_q, in_q = best_swap['partition'], best_swap['out_qubit'], best_swap['in_qubit']
                    
                    # Update tracking variables
                    selected_qubits[p_id].remove(out_q)
                    selected_qubits[p_id].append(in_q)
                    discarded_qubits[p_id].remove(in_q)
                    discarded_qubits[p_id].append(out_q)
                    selected_set.remove(out_q)
                    selected_set.add(in_q)
                    
                    logger.info(f"  > Swapped {out_q} for {in_q} in partition {p_id} for a net gain of {best_swap['gain']} edges.")
                else:
                    logger.info("No further improvement found. Stopping optimization.")
                    break
        
        # Final assignment of the selected qubit indices.
        self.idx_dict = {p_id: sorted(q_list) for p_id, q_list in selected_qubits.items()}

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
            self._qpu_sampler = DWaveSampler(solver={'topology__type': 'zephyr', 'chip_id':'Advantage2_system1.6'})
            self.m, self.t = self._qpu_sampler.properties['topology']['shape']
            graph = dnx.zephyr_graph(m=self.m, t=self.t,
                                node_list=self._qpu_sampler.nodelist, edge_list=self._qpu_sampler.edgelist)
            self.coordinated_graph = nx.relabel_nodes(
                    graph,
                    {q: dnx.zephyr_coordinates(self.m,self.t).linear_to_zephyr(q)
                    for q in graph.nodes})
            self.adjacency = self._qpu_sampler.adjacency
        except Exception as e:
            logger.warning(f"QPU is offline. Setting a hard-coded zephyr. "
                           f"Check to see you're pinging the correct chip_id: {e}"
                           )
            self.m, self.t = 12,4
            print(os.getcwd())
            repo_root = os.path.dirname(CaloQuVAE.__file__)  # /.../CaloQuVAE

            nodelist_path = os.path.join(repo_root, 'model', 'rbm', 'nodelist.pickle')
            edgelist_path = os.path.join(repo_root, 'model', 'rbm', 'edgelist.pickle')
            adjacency_path = os.path.join(repo_root, 'model', 'rbm', 'adjacency.pickle')

            # Load the pickle files
            with open(nodelist_path, 'rb') as handle:
                self.nodelist = pickle.load(handle)

            with open(edgelist_path, 'rb') as handle:
                self.edgelist = pickle.load(handle)

            with open(adjacency_path, 'rb') as handle:
                self.adjacency = pickle.load(handle)
            graph = dnx.zephyr_graph(m=self.m, t=self.t,
                                node_list=self.nodelist, edge_list=self.edgelist)
            self.coordinated_graph = nx.relabel_nodes(
                    graph,
                    {q: dnx.zephyr_coordinates(self.m,self.t).linear_to_zephyr(q)
                    for q in graph.nodes})
            
        
    def coordinates_to_idx(self, q, m, t):
        return q[4] + m*(q[3] + 2*(q[2] + t*(q[1]+(2*m+1)*q[0])))

class ZephyrRBM_Old(ZephyrRBM):
    # override the method to use the original implementation
    def gen_qubit_idx_dict(self):
        self.load_coordinates()
        idx_dict = {}
        for partition in range(self._n_partitions):
            idx_dict[str(partition)] = []
        for q in self.coordinated_graph.nodes:
            _idx = (2*q[0]+q[1] + 2*q[4]+q[3])%4
            idx_dict[str(_idx)].append(self.coordinates_to_idx(q, self.m,self.t))
        # The original truncation logic
        for partition, idxs in idx_dict.items():
            idx_dict[partition] = idx_dict[partition][:self._nodes_per_partition]
        self.idx_dict = idx_dict

if __name__ == "__main__":
    rbm = ZephyrRBM()