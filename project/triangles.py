from typing import List

import pygraphblas as pgb

from project.checks import check_bool_adj_matrix

__all__ = [
    "count_triangles_for_each_vertex",
    "count_triangles_cohen",
    "count_triangles_sandia",
]


def count_triangles_for_each_vertex(adj_matrix: pgb.Matrix) -> List[int]:
    """
    Counts the number of triangles each vertex is a part of in the given undirected graph
    represented as an adjacency matrix.

    :param adj_matrix: adjacency matrix of `BOOL` data type for an undirected graph, expected to be symmetric, but not checked
    :return: a list of integers representing the number of triangles each vertex is a part of
    :raises ValueError: if the `adj_matrix` is not a square matrix or if it is not of `BOOL` data type
    """
    check_bool_adj_matrix(adj_matrix)
    num_vertices = adj_matrix.nrows
    identity_matrix = pgb.Matrix.identity(pgb.BOOL, num_vertices)
    zero_matrix = pgb.Matrix.sparse(pgb.BOOL, num_vertices, num_vertices)

    # get rid of loops
    adj_matrix = adj_matrix.eadd(
        zero_matrix, mask=identity_matrix, desc=pgb.descriptor.C
    )

    res = adj_matrix.mxm(
        adj_matrix, pgb.UINT64.PLUS_TIMES, mask=adj_matrix
    ).reduce_vector()
    res.assign_scalar(0, mask=res, desc=pgb.descriptor.C)
    return [num // 2 for num in res.vals]


def count_triangles_cohen(adj_matrix: pgb.Matrix) -> int:
    """
    Counts the number of triangles in an undirected graph using Cohen's algorithm.

    :param adj_matrix: adjacency matrix of `BOOL` data type for an undirected graph, expected to be symmetric, but not checked
    :return: the number of triangles in the graph
    :raises ValueError: if the `adj_matrix` is not a square matrix or if it is not of `BOOL` data type
    """
    check_bool_adj_matrix(adj_matrix)
    adj_lower = adj_matrix.tril(-1)
    adj_upper = adj_matrix.triu(1)
    return (
        adj_lower.mxm(adj_upper, pgb.UINT64.PLUS_TIMES, mask=adj_matrix).reduce_int()
        // 2
    )


def count_triangles_sandia(adj_matrix: pgb.Matrix) -> int:
    """
    Counts the number of triangles in an undirected graph using Sandia algorithm.

    :param adj_matrix: adjacency matrix of `BOOL` data type for an undirected graph, expected to be symmetric, but not checked
    :return: the number of triangles in the graph
    :raises ValueError: if the `adj_matrix` is not a square matrix or if it is not of `BOOL` data type
    """
    check_bool_adj_matrix(adj_matrix)
    adj_lower = adj_matrix.tril(-1)
    return adj_lower.mxm(
        adj_lower.transpose(), pgb.UINT64.PLUS_TIMES, mask=adj_lower
    ).reduce_int()
