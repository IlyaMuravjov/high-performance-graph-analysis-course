from typing import List
from typing import Tuple

import pygraphblas as pgb
import numpy as np

from project.checks import check_adj_matrix
from project.checks import check_start_in_range
from project.pygraphblas_utils import matrix_to_list_of_key_and_value_list_tuples

__all__ = [
    "multi_source_bellman_ford",
    "single_source_bellman_ford",
    "all_pairs_floyd_warshall",
]

_FLOAT_TYPE = pgb.FP64


def multi_source_bellman_ford(
    adj_matrix: pgb.Matrix, starts: List[int]
) -> List[Tuple[int, List[float]]]:
    """
    Finds the shortest paths from multiple source vertices to all vertices in the given weighted graph.

    :param adj_matrix: floating point adjacency matrix for a weighted graph to perform multi-source
                       Bellman-Ford algorithm on
    :param starts: a list of indices of starting vertices
    :return: list of tuples, where each tuple contains a starting vertex index and a list of the distances
             from the starting vertex for each vertex in the graph. If a vertex is not reachable from
             a corresponding starting vertex, its shortest distance is `np.inf`.
    :raises ValueError: if there is a reachable negative cycle in the graph
                        or if the `adj_matrix` is not a square matrix
                        or if the `adj_matrix` is not of `FP64` data type
    :raises IndexError: if any index in `starts` is out of range
    """
    check_adj_matrix(adj_matrix, _FLOAT_TYPE)
    num_vertices = adj_matrix.nrows
    for start in starts:
        check_start_in_range(start, num_vertices)
    if num_vertices == 0:
        return []
    adj_matrix = adj_matrix.eadd(
        pgb.Matrix.identity(_FLOAT_TYPE, num_vertices, 0.0), _FLOAT_TYPE.MIN
    )
    dist = pgb.Matrix.sparse(_FLOAT_TYPE, len(starts), num_vertices)
    for i, start in enumerate(starts):
        dist[i, start] = 0.0
    for _ in range(num_vertices):
        old_dist = dist
        dist = dist.mxm(adj_matrix, _FLOAT_TYPE.MIN_PLUS)
        if old_dist.iseq(dist):
            return matrix_to_list_of_key_and_value_list_tuples(
                dist, row_keys=starts, default_value=np.inf
            )
    raise ValueError("Negative cycle detected")


def single_source_bellman_ford(adj_matrix: pgb.Matrix, start: int) -> List[float]:
    """
    Finds the shortest paths from a single source vertex to all other vertices in the given weighted graph.

    :param adj_matrix: floating point adjacency matrix for a weighted graph to perform single-source
                       Bellman-Ford algorithm on
    :param start: index of the starting vertex
    :return: list of the distances from the starting vertex for each vertex in the graph.
             If the current vertex is not reachable from the starting vertex, its shortest distance is `np.inf`.
    :raises ValueError: if there is a reachable negative cycle in the graph
                        or if the `adj_matrix` is not a square matrix
                        or if the `adj_matrix` is not of `FP64` data type
    :raises IndexError: if any index in `starts` is out of range
    """
    _, dist = multi_source_bellman_ford(adj_matrix, [start])[0]
    return dist


def all_pairs_floyd_warshall(adj_matrix: pgb.Matrix) -> List[Tuple[int, List[float]]]:
    """
    Finds the shortest paths between all pairs of vertices in the given weighted graph.

    :param adj_matrix: floating point adjacency matrix for a weighted graph to perform all-pairs
                       Floyd-Warshall algorithm on
    :return: list of tuples, where each tuple contains a starting vertex index and a list of the distances
             from the starting vertex for each vertex in the graph. If a vertex is not reachable from
             a corresponding starting vertex, its shortest distance is `np.inf`.
    :raises ValueError: if there is a negative cycle in the graph
                        or if the `adj_matrix` is not a square matrix
                        or if the `adj_matrix` is not of `FP64` data type
    """
    check_adj_matrix(adj_matrix, _FLOAT_TYPE)
    num_vertices = adj_matrix.nrows
    dist = adj_matrix.eadd(
        pgb.Matrix.identity(_FLOAT_TYPE, num_vertices, 0.0), _FLOAT_TYPE.MIN
    )
    for k in range(num_vertices):
        dist.extract_matrix(col_index=k).mxm(
            dist.extract_matrix(row_index=k),
            _FLOAT_TYPE.MIN_PLUS,
            accum=_FLOAT_TYPE.MIN,
            out=dist,
        )
    if dist.diag().reduce_float(_FLOAT_TYPE.min_monoid) < 0:
        raise ValueError("Negative cycle detected")
    return matrix_to_list_of_key_and_value_list_tuples(
        dist, row_keys=list(range(num_vertices)), default_value=np.inf
    )
