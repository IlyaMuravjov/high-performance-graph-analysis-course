import pathlib

import networkx as nx
import pygraphblas as pgb

__all__ = ["read_graph", "digraph_to_ajd_matrix", "read_adj_matrix_for_digraph"]


def read_graph(name: str) -> nx.Graph:
    """
    Reads specified test graph from `test/data/graphs` folder
    :param name: the name of the graph
    :return: the graph read
    """
    return nx.nx_agraph.read_dot(
        pathlib.Path(__file__).parent / "data" / "graphs" / f"{name}.dot"
    )


def digraph_to_ajd_matrix(graph: nx.Graph) -> pgb.Matrix:
    """
    Creates boolean adjacency matrix for a directed given graph
    :param graph: input (unlabeled) graph
    :return: boolean adjacency matrix
    """
    adj_matrix = pgb.Matrix.sparse(
        pgb.BOOL, graph.number_of_nodes(), graph.number_of_nodes()
    )
    for source, target in graph.edges():
        adj_matrix[int(source), int(target)] = True
    return adj_matrix


def read_adj_matrix_for_digraph(name: str) -> pgb.Matrix:
    """
    Creates boolean adjacency matrix for a specified test graph from `test/data/graphs` folder
    :param name: the name of the graph
    :return: boolean adjacency matrix
    """
    return digraph_to_ajd_matrix(read_graph(name))
