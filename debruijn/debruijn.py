#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""Perform assembly based on debruijn graph."""

import argparse
import os
import sys
import networkx as nx
import matplotlib
from operator import itemgetter
import random
random.seed(9001)
from random import randint
import statistics
import textwrap
import matplotlib.pyplot as plt
matplotlib.use("Agg")

__author__ = "Nicolas Salaun"
__copyright__ = "Universite Paris Diderot"
__credits__ = ["Nicolas Salaun"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Nicolas Salaun"
__email__ = "nicolas.salaun@orange.fr"
__status__ = "Developpement"

def isfile(path): # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file
    
    :raises ArgumentTypeError: If file doesn't exist
    
    :return: (str) Path 
    """
    if not os.path.isfile(path):
        if os.path.isdir(path):
            msg = "{0} is a directory".format(path)
        else:
            msg = "{0} does not exist.".format(path)
        raise argparse.ArgumentTypeError(msg)
    return path


def get_arguments(): # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, usage=
                                     "{0} -h"
                                     .format(sys.argv[0]))
    parser.add_argument('-i', dest='fastq_file', type=isfile,
                        required=True, help="Fastq file")
    parser.add_argument('-k', dest='kmer_size', type=int,
                        default=22, help="k-mer size (default 22)")
    parser.add_argument('-o', dest='output_file', type=str,
                        default=os.curdir + os.sep + "contigs.fasta",
                        help="Output contigs in fasta file (default contigs.fasta)")
    parser.add_argument('-f', dest='graphimg_file', type=str,
                        help="Save graph as an image (png)")
    return parser.parse_args()


def read_fastq(fastq_file):
    """Extract reads from fastq files.

    :param fastq_file: (str) Path to the fastq file.
    :return: A generator object that iterate the read sequences. 
    """
    with open(fastq_file, "r") as fh:
        for line in fh:
            if line.startswith("@"):
                yield next(fh).strip()


def cut_kmer(read, kmer_size):
    """Cut read into kmers of size kmer_size.
    
    :param read: (str) Sequence of a read.
    :return: A generator object that iterate the kmers of of size kmer_size.
    """
    yield from (read[i:i+kmer_size] for i in range(len(read) - kmer_size + 1))


def build_kmer_dict(fastq_file, kmer_size):
    """Build a dictionnary object of all kmer occurrences in the fastq file

    :param fastq_file: (str) Path to the fastq file.
    :return: A dictionnary object that identify all kmer occurrences.
    """
    kmer_dict = {}
    for read in read_fastq(fastq_file):
        for kmer in cut_kmer(read, kmer_size):
            if kmer in kmer_dict:
                kmer_dict[kmer] += 1
            else:
                kmer_dict[kmer] = 1
    return kmer_dict


def build_graph(kmer_dict):
    """Build the debruijn graph

    :param kmer_dict: A dictionnary object that identify all kmer occurrences.
    :return: A directed graph (nx) of all kmer substring and weight (occurrence).
    """
    graph = nx.DiGraph()
    for kmer, weight in kmer_dict.items():
        graph.add_edge(kmer[:-1], kmer[1:], weight=weight)
    return graph

def remove_paths(graph, path_list, remove_first_node, remove_last_node):
    """
    Remove a list of paths from a directed graph.

    :param graph: A directed graph object.
    :param path_list: A list of paths, where each path is a list of node IDs.
    :param remove_first_node: A boolean indicating whether to remove the first node of each path.
    :param remove_last_node: A boolean indicating whether to remove the last node of each path.
    :return: A modified directed graph object.
    """
    for path in path_list:
        if remove_first_node and remove_last_node:
            graph.remove_nodes_from(path)
        elif remove_first_node:
            graph.remove_nodes_from(path[:-1])
        elif remove_last_node:
            graph.remove_nodes_from(path[1:])
        else:
            graph.remove_nodes_from(path[1:-1])
    return graph


def select_best_path(graph, path_list, path_length, weight_avg_list, delete_entry_node=False, delete_sink_node=False):
    """Select the best path between different paths

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param path_length_list: (list) A list of length of each path
    :param weight_avg_list: (list) A list of average weight of each path
    :param delete_entry_node: (boolean) True->We remove the first node of a path 
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    if statistics.stdev(weight_avg_list) > 0:
        ind = weight_avg_list.index(max(weight_avg_list))
    elif statistics.stdev(path_length) > 0:
        ind = path_length.index(max(path_length))
    else:
        ind = random.randint(0, len(path_list))
    path_list.pop(ind)
    return remove_paths(graph, path_list, delete_entry_node, delete_sink_node)

def path_average_weight(graph, path):
    """Compute the weight of a path

    :param graph: (nx.DiGraph) A directed graph object
    :param path: (list) A path consist of a list of nodes
    :return: (float) The average weight of a path
    """
    return statistics.mean([d["weight"] for (u, v, d) in graph.subgraph(path).edges(data=True)])

def solve_bubble(graph, ancestor_node, descendant_node):
    """Explore and solve bubble issue

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph 
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    path_list = list(nx.all_simple_paths(graph, source=ancestor_node, target=descendant_node))
    path_length = [len(i) for i in path_list]
    weight_avg_list = [statistics.mean([j[2]['weight'] for j in list(graph.subgraph(i).edges(data=True))]) for i in path_list]
    solved_bubble = select_best_path(graph, path_list, path_length, weight_avg_list, delete_entry_node=False, delete_sink_node=False)
    return solved_bubble

def simplify_bubbles(graph):
    """Detect and explode bubbles

    :param graph: (nx.DiGraph) A directed graph object
    # :return: (nx.DiGraph) A directed graph object
    """
    # if not nx.is_directed_acyclic_graph(graph):
    #     raise ValueError("Input graph is not a directed acyclic graph (DAG).")
    
    bubble = False
    for node in graph.nodes():
        predecessors = list(graph.predecessors(node))
        if len(predecessors) > 1:
            combinations = [(predecessors[i], predecessors[j]) for i in range(0, len(predecessors) - 1) for j in range(i + 1, len(predecessors)) if i != j]
            for combination in combinations:
                print("here")
                ancestor = nx.lowest_common_ancestor(graph, combination[0], combination[1])
                if ancestor:
                    bubble = True
                    break
        if bubble:
            break

    if bubble:
        graph = simplify_bubbles(solve_bubble(graph, ancestor, node))
    return graph

# Usage example
graph = nx.DiGraph()  # Create your directed graph here
if nx.is_directed_acyclic_graph(graph):
    graph = simplify_bubbles(graph)
else:
    print("The input graph is not a DAG. Please modify your graph or algorithm to make it a DAG.")


def solve_entry_tips(graph, starting_nodes):
    """Remove entry tips and returns a graph without useless entry paths on the simplify bubble basis

    :param graph: (nx.DiGraph) A directed graph object
    :starting_nodes: (list) A list of nodes without predecessors
    :return: (nx.DiGraph) A directed graph object
    """
    path_list = [list(nx.all_simple_paths(graph, start, node)) for node in graph.nodes for start in starting_nodes if len(list(graph.predecessors(node))) > 1]
    if path_list:
        path_length = [len(path) for paths in path_list for path in paths]
        weight_avg = [path_average_weight(graph, path) for paths in path_list for path in paths]
        graph = select_best_path(graph, [path for paths in path_list for path in paths], path_length, weight_avg, delete_entry_node=True, delete_sink_node=False)
    return graph


def solve_out_tips(graph, ending_nodes):
    """Remove out tips

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    path_list = [list(nx.all_simple_paths(graph, node, end)) for node in graph.nodes for end in ending_nodes if len(list(graph.successors(node))) > 1]
    if path_list:
        path_length = [len(path) for paths in path_list for path in paths]
        weight_avg = [path_average_weight(graph, path) for paths in path_list for path in paths]
        graph = select_best_path(graph, [path for paths in path_list for path in paths], path_length, weight_avg, delete_entry_node=False, delete_sink_node=True)
    return graph

def get_starting_nodes(graph):
    """Get nodes without predecessors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without predecessors
    """
    starting_nodes = []
    for node in graph.nodes():
        if graph.in_degree(node) == 0:
            starting_nodes.append(node)
    return starting_nodes 

def get_sink_nodes(graph):
    """Get nodes without successors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without successors
    """
    sink_nodes = []
    for node in graph.nodes():
        if graph.out_degree(node) == 0:
            sink_nodes.append(node)
    return sink_nodes

def get_contigs(graph, starting_nodes, ending_nodes):
    """Extract the contigs from the graph

    :param graph: (nx.DiGraph) A directed graph object 
    :param starting_nodes: (list) A list of nodes without predecessors
    :param ending_nodes: (list) A list of nodes without successors
    :return: (list) List of [contiguous sequence and their length]
    """
    contigs = []
    for starting_node in starting_nodes:
        for ending_node in ending_nodes:
            if nx.has_path(graph, starting_node, ending_node):
                for path in nx.all_simple_paths(graph, starting_node, ending_node):
                    sequence = str(path[0])
                    for node in path[1:]:
                        sequence += node[-1]
                    contigs.append([sequence, len(sequence)])
    return contigs

def save_contigs(contigs_list, output_file):
    """Write all contigs in fasta format 

    :param contig_list: (list) List of [contiguous sequence and their length]
    :param output_file: (str) Path to the output file
    """
    with open(output_file, "w") as fh:
        for i, contig in enumerate(contigs_list):
            fh.write(f">contig_{i} len={contig[1]}\n")
            fh.write(textwrap.fill(contig[0], width=80))
            fh.write("\n")

                


def draw_graph(graph, graphimg_file): # pragma: no cover
    """Draw the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param graphimg_file: (str) Path to the output file
    """                                   
    fig, ax = plt.subplots()
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d['weight'] > 3]
    #print(elarge)
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d['weight'] <= 3]
    #print(elarge)
    # Draw the graph with networkx
    #pos=nx.spring_layout(graph)
    pos = nx.random_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=6)
    nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(graph, pos, edgelist=esmall, width=6, alpha=0.5, 
                           edge_color='b', style='dashed')
    #nx.draw_networkx(graph, pos, node_size=10, with_labels=False)
    # save image
    plt.savefig(graphimg_file)


#==============================================================
# Main program
#==============================================================


if __name__ == '__main__': # pragma: no cover
    random.seed(1)
    args = get_arguments()
    dict_kmer = build_kmer_dict(args.fastq_file, args.kmer_size)
    graph = build_graph(dict_kmer)
    start_nodes = get_starting_nodes(graph)
    end_nodes = get_sink_nodes(graph)
    graph = simplify_bubbles(graph)
    graph = solve_entry_tips(graph, get_starting_nodes(graph))
    graph = solve_out_tips(graph, get_sink_nodes(graph))
    contigs = get_contigs(graph, get_starting_nodes(graph), get_sink_nodes(graph))
    save_contigs(contigs, args.output_file)
    if args.graphimg_file:
        draw_graph(graph, args.graphimg_file)
