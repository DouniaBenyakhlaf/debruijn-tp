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

import statistics
import textwrap
import matplotlib.pyplot as plt
from typing import Iterator, Dict, List
import argparse
import itertools
import os
import sys
from pathlib import Path
import networkx as nx
from networkx import (
    DiGraph,
    all_simple_paths,
    lowest_common_ancestor,
    has_path,
)
import matplotlib
import random
random.seed(9001)

matplotlib.use("Agg")

__author__ = "Dounia BENYAKHLAF"
__copyright__ = "Universite Paris Diderot"
__credits__ = ["Dounia BENYAKHLAF"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Dounia BENYAKHLAF"
__email__ = "benyakhlaf.dounia@gmail.com"
__status__ = "Developpement"


def isfile(path: str) -> Path:  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments():  # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description=__doc__, usage=f"{sys.argv[0]} -h"
    )
    parser.add_argument(
        "-i", dest="fastq_file", type=isfile, required=True, help="Fastq file"
    )
    parser.add_argument(
        "-k", dest="kmer_size", type=int, default=22, help="k-mer size (default 22)"
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        default=Path(os.curdir + os.sep + "contigs.fasta"),
        help="Output contigs in fasta file (default contigs.fasta)",
    )
    parser.add_argument(
        "-f", dest="graphimg_file", type=Path, help="Save graph as an image (png)"
    )
    return parser.parse_args()


def read_fastq(fastq_file: Path) -> Iterator[str]:
    """Extract reads from fastq files.

    :param fastq_file: (Path) Path to the fastq file.
    :return: A generator object that iterate the read sequences.
    """
    if isfile(fastq_file):
        with open(fastq_file, 'r', encoding="utf-8") as fq_file:
            iter_file = iter(fq_file)
            for _ in iter_file:
                sequence = next(iter_file)
                next(iter_file) # separateur
                next(iter_file) # evaluation
                yield sequence.strip()




def cut_kmer(read: str, kmer_size: int) -> Iterator[str]:
    """Cut read into kmers of size kmer_size.

    :param read: (str) Sequence of a read.
    :return: A generator object that provides the kmers (str) of size kmer_size.
    """
    n = len(read)
    for i in range(n):
        if i+kmer_size <= n:
            kmer = ""
            for l in range(i, i+kmer_size):
                kmer += read[l]
            yield kmer




def build_kmer_dict(fastq_file: Path, kmer_size: int) -> Dict[str, int]:
    """Build a dictionnary object of all kmer occurrences in the fastq file

    :param fastq_file: (str) Path to the fastq file.
    :return: A dictionnary object that identify all kmer occurrences.
    """
    kmers_dict = {}
    sequences = read_fastq(fastq_file)
    for seq in sequences:
        kmers = cut_kmer(seq, kmer_size)
        for kmer in kmers:
            if kmer not in kmers_dict:
                kmers_dict[kmer] = 0
            kmers_dict[kmer] += 1
    return kmers_dict


def build_graph(kmer_dict: Dict[str, int]) -> DiGraph:
    """Build the debruijn graph

    :param kmer_dict: A dictionnary object that identify all kmer occurrences.
    :return: A directed graph (nx) of all kmer substring and weight (occurrence).
    """
    graph = DiGraph()

    for kmer, weight in kmer_dict.items():
        prefix = kmer[:-1]
        suffix = kmer[1:]
        graph.add_edge(prefix, suffix, weight=weight)

    return graph


def remove_paths(
    graph: DiGraph,
    path_list: List[List[str]],
    delete_entry_node: bool,
    delete_sink_node: bool,
) -> DiGraph:
    """Remove a list of path in a graph. A path is set of connected node in
    the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    for path in path_list:
        if(delete_entry_node and delete_sink_node):
            graph.remove_nodes_from(path)
        elif delete_entry_node:
            graph.remove_nodes_from(path[:-1])
        elif delete_sink_node:
            graph.remove_nodes_from(path[1:])
        graph.remove_nodes_from(path[1:-1])
    return graph


def select_best_path(
    graph: DiGraph,
    path_list: List[List[str]],
    path_length: List[int],
    weight_avg_list: List[float],
    delete_entry_node: bool = False,
    delete_sink_node: bool = False,
) -> DiGraph:
    """Select the best path between different paths

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param path_length_list: (list) A list of length of each path
    :param weight_avg_list: (list) A list of average weight of each path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    indice_best_path = 0
    std_weight = statistics.stdev(weight_avg_list)
    if std_weight > 0:
        indice_best_path = weight_avg_list.index(max(weight_avg_list))
    else :
        std_length = statistics.stdev(path_length)
        if std_length > 0:
            indice_best_path = path_length.index(max(path_length))
        else:
            indice_best_path = random.randint(0, len(path_list))
    del path_list[indice_best_path]
    new_graph = remove_paths(graph, path_list,
                            delete_entry_node, delete_sink_node)
    return new_graph



def path_average_weight(graph: DiGraph, path: List[str]) -> float:
    """Compute the weight of a path

    :param graph: (nx.DiGraph) A directed graph object
    :param path: (list) A path consist of a list of nodes
    :return: (float) The average weight of a path
    """
    return statistics.mean(
        [d["weight"] for (u, v, d) in graph.subgraph(path).edges(data=True)]
    )


def solve_bubble(graph: DiGraph, ancestor_node: str, descendant_node: str) -> DiGraph:
    """Explore and solve bubble issue

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    path_list = list(all_simple_paths(graph, ancestor_node, descendant_node))
    path_length = [len(l) for l in path_list]
    weight_avg_list = []
    for path in path_list:
        weight_avg_list.append(path_average_weight(graph, path))
    return select_best_path(graph, path_list, path_length, weight_avg_list, False, False)
    

def simplify_bubbles(graph: DiGraph) -> DiGraph:
    """Detect and explode bubbles

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    bubble = False
    node_n = None
    for node in graph.nodes:
        predecessors_list = list(graph.predecessors(node))
        if len(predecessors_list) > 1:
            combinaisons = list(itertools.combinations(predecessors_list, 2))
            for comb in combinaisons:
                ancestor_node = lowest_common_ancestor(graph, comb[0], comb[1])
                if ancestor_node is not None :
                    bubble = True
                    node_n = node
                    break
    if bubble == True:
        graph = simplify_bubbles(solve_bubble(graph, ancestor_node, node_n))
    return graph


def solve_tip_entry_aux(graph: DiGraph, ancestors: List[str], descendant_node: str
                        ) -> DiGraph:
    """Explore and solve tip issue

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    path_list = []
    for ancestor_node in ancestors:
        path_list.extend(all_simple_paths(graph, ancestor_node, descendant_node))
    path_length = [len(l) for l in path_list]
    weight_avg_list = []
    for path in path_list:
        weight_avg_list.append(path_average_weight(graph, path))
    return select_best_path(graph, path_list, path_length, weight_avg_list,
                            True, False)

def solve_tip_out_aux(graph: DiGraph, successors: List[str], node: str
            ) -> DiGraph:
    """Explore and solve tip issue

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    path_list = []
    for suc_node in successors:
        path_list.extend(all_simple_paths(graph, node, suc_node))
    path_length = [len(l) for l in path_list]
    weight_avg_list = []
    for path in path_list:
        weight_avg_list.append(path_average_weight(graph, path))
    return select_best_path(graph, path_list, path_length, weight_avg_list,
                            False, True)


def solve_entry_tips(graph: DiGraph, starting_nodes: List[str]) -> DiGraph:
    """Remove entry tips

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of starting nodes
    :return: (nx.DiGraph) A directed graph object
    """
    node_n = None
    ancestors = []
    for node in graph.nodes:
        pred = list(graph.predecessors(node))
        if len(pred) > 1:
            for start in starting_nodes:
                if( (start in graph) and (has_path(graph, start, node))):
                    ancestors.append(start)
            if len(ancestors) >= 2:
                node_n = node
                break
    if len(ancestors) >= 2:
        graph = solve_entry_tips(solve_tip_entry_aux(graph, ancestors, node_n),
                                 starting_nodes)
    return graph


def solve_out_tips(graph: DiGraph, ending_nodes: List[str]) -> DiGraph:
    """Remove out tips

    :param graph: (nx.DiGraph) A directed graph object
    :param ending_nodes: (list) A list of ending nodes
    :return: (nx.DiGraph) A directed graph object
    """
    node_n = None
    successors = []
    for node in graph.nodes:
        succ = list(graph.successors(node))
        if len(succ) > 1:
            for end in ending_nodes:
                if( (end in graph) and (has_path(graph, node, end))):
                    successors.append(end)
            if len(successors) >= 2:
                node_n = node
                break
    if len(successors) >= 2:
        graph = solve_out_tips(solve_tip_out_aux(graph, successors, node_n), ending_nodes)
    return graph


def get_starting_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without predecessors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without predecessors
    """
    starting_nodes = []
    for node in graph.nodes:
        if next(graph.predecessors(node), None) is None:
            starting_nodes.append(node)
    return starting_nodes


def get_sink_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without successors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without successors
    """
    sink_nodes = []
    for node in graph.nodes:
        if next(graph.successors(node), None) is None:
            sink_nodes.append(node)
    return sink_nodes


def get_contigs(
    graph: DiGraph, starting_nodes: List[str], ending_nodes: List[str]
) -> List:
    """Extract the contigs from the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of nodes without predecessors
    :param ending_nodes: (list) A list of nodes without successors
    :return: (list) List of [contiguous sequence and their length]
    """
    list_all_contigs = []
    for start in starting_nodes:
        for end in ending_nodes:
            all_paths = all_simple_paths(graph, start, end)
            for path in all_paths:
                contig = path[0]
                for i in range(1, len(path)):
                    contig += path[i][-1]
                list_all_contigs.append((contig, len(contig)))
    return list_all_contigs


def save_contigs(contigs_list: List[str], output_file: Path) -> None:
    """Write all contigs in fasta format

    :param contig_list: (list) List of [contiguous sequence and their length]
    :param output_file: (Path) Path to the output file
    """
    with open(output_file, 'w', encoding="utf-8") as contig_fasta:
        for i, contig in enumerate(contigs_list):
            contig_fasta.write(f">contig_{i} len={contig[1]}\n")
            contig_fasta.write(f"{textwrap.fill(contig[0], width=80)}\n")


def draw_graph(graph: DiGraph, graphimg_file: Path) -> None:  # pragma: no cover
    """Draw the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param graphimg_file: (Path) Path to the output file
    """
    fig, ax = plt.subplots()
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] > 3]
    # print(elarge)
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] <= 3]
    # print(elarge)
    # Draw the graph with networkx
    # pos=nx.spring_layout(graph)
    pos = nx.random_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=6)
    nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        graph, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )
    #nx.draw_networkx(graph, pos, node_size=10, with_labels=False)
    # save image
    plt.savefig(graphimg_file.resolve())


# ==============================================================
# Main program
# ==============================================================
def main() -> None:  # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()

    # Test
    kmer_dict = build_kmer_dict(args.fastq_file, args.kmer_size)
    graph = build_graph(kmer_dict)
    if args.graphimg_file:
        draw_graph(graph, args.graphimg_file)
    graph = simplify_bubbles(graph)

    starting_nodes = get_starting_nodes(graph)
    ending_nodes = get_sink_nodes(graph)
    graph = solve_entry_tips(graph, starting_nodes)
    graph = solve_out_tips(graph, ending_nodes)
    contigs_list = get_contigs(graph, starting_nodes, ending_nodes)
    save_contigs(contigs_list, args.output_file)

    # Fonctions de dessin du graphe
    # A decommenter si vous souhaitez visualiser un petit
    # graphe
    # Plot the graph
    # if args.graphimg_file:
    #     draw_graph(graph, args.graphimg_file)


if __name__ == "__main__":  # pragma: no cover
    main()
