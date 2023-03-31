import pathlib

import networkx as nx
import pygraphblas as pgb

__all__ = [
    "read_graph",
    "digraph_to_ajd_matrix",
    "read_adj_matrix_for_digraph",
    "undirected_graph_to_symmetric_adj_matrix",
    "read_adj_matrix_for_undirected_graph",
]


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


def undirected_graph_to_symmetric_adj_matrix(graph: nx.Graph) -> pgb.Matrix:
    """
    Creates symmetric adjacency matrix for a given undirected graph
    :param graph: input (unlabeled) undirected graph
    :return: symmetric adjacency matrix
    """
    adj_matrix = pgb.Matrix.sparse(
        pgb.BOOL, graph.number_of_nodes(), graph.number_of_nodes()
    )
    for source, target in graph.edges():
        adj_matrix[int(source), int(target)] = True
        adj_matrix[int(target), int(source)] = True
    return adj_matrix


def read_adj_matrix_for_undirected_graph(name: str) -> pgb.Matrix:
    """
    Creates symmetric adjacency matrix for a specified test undirected graph from `test/data/graphs` folder
    :param name: the name of the graph
    :raises ValueError: if input graph is directed
    :return: symmetric adjacency matrix
    """
    graph = read_graph(name)
    if graph.is_directed():
        raise ValueError("Input graph is directed. Expected undirected graph.")
    return undirected_graph_to_symmetric_adj_matrix(graph)
