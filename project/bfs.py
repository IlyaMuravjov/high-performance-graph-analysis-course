import pygraphblas as pgb

from typing import List
from typing import Tuple

from project.checks import check_adj_matrix
from project.checks import check_start_in_range
from project.pygraphblas_utils import matrix_to_list_of_key_and_value_list_tuples
from project.pygraphblas_utils import vector_to_list

__all__ = ["bfs", "multi_source_bfs"]


def bfs(adj_matrix: pgb.Matrix, start: int) -> List[int]:
    """
    Performs breadth-first search algorithm on the given unweighted graph to
    find the distance to each vertex from the starting vertex.

    :param adj_matrix: adjacency matrix of `BOOL` data type for a graph to perform bfs on
    :param start: index of the starting vertex
    :return: a list of integers representing distance to each vertex from the starting vertex or -1 if the vertex is unreachable
    :raises ValueError: if the `adj_matrix` is not a square matrix or if it is not of `BOOL` data type
    :raises IndexError: if the `start` is out of range
    """
    check_adj_matrix(adj_matrix, pgb.BOOL)
    num_vertices = adj_matrix.nrows
    check_start_in_range(start, num_vertices)
    front = pgb.Vector.sparse(pgb.BOOL, num_vertices)
    front[start] = True
    res = pgb.Vector.sparse(pgb.INT64, num_vertices)
    i = 0
    while front.nvals != 0:
        res[front] = i
        front.vxm(adj_matrix, out=front, mask=res, desc=pgb.descriptor.RSC)
        i += 1
    return vector_to_list(res, default_value=-1)


def multi_source_bfs(
    adj_matrix: pgb.Matrix, starts: List[int]
) -> List[Tuple[int, List[int]]]:
    """
    Performs multi-source breadth-first search algorithm on the given unweighted graph to
    find the parents of each vertex on the shortest path from each starting vertex.

    :param adj_matrix: adjacency matrix of `BOOL` data type for a graph to perform multi-source bfs on
    :param starts: a list of indices of starting vertices
    :return: a list of tuples, where each tuple contains a starting vertex index and a list of parent indices
             for each vertex in the graph. The parent index indicates the previous vertex on the shortest path
             to the current vertex starts. If the current vertex is the starting vertex, its parent index is -1.
             If the current vertex is not reachable from any of the starting vertices, its parent index is -2.
             If there are multiple possible parent indices for a vertex, the smallest one is chosen.
    :raises ValueError: if the `adj_matrix` is not a square matrix or if it is not of `BOOL` data type
    :raises IndexError: if any index in `starts` is out of range

    >>> adj_matrix = pgb.Matrix.from_lists([0, 0, 1, 2], [1, 2, 3, 3], [True, True, True, True], nrows=4, ncols=4)
    >>> starts = [0, 2]
    >>> parents = multi_source_bfs(adj_matrix, starts)
    >>> print(parents)
    [(0, [-1, 0, 0, 1]), (2, [-2, -2, -1, 2])]
    """
    check_adj_matrix(adj_matrix, pgb.BOOL)
    num_vertices = adj_matrix.nrows
    for start in starts:
        check_start_in_range(start, num_vertices)
    front = pgb.Matrix.sparse(pgb.INT64, len(starts), num_vertices)
    for i, start in enumerate(starts):
        front[i, start] = -1
    res = pgb.Matrix.sparse(pgb.INT64, len(starts), num_vertices)
    i = -1
    while front.nvals != 0:
        res.assign(front, mask=front, desc=pgb.descriptor.S)
        front.apply(pgb.INT64.POSITIONJ, out=front)
        front.mxm(
            adj_matrix,
            semiring=pgb.INT64.MIN_FIRST,
            out=front,
            mask=res,
            desc=pgb.descriptor.RSC,
        )
        i += 1
    return matrix_to_list_of_key_and_value_list_tuples(
        res, row_keys=starts, default_value=-2
    )
