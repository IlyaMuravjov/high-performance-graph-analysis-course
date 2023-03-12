from typing import List

import pygraphblas as pgb

__all__ = ["bfs"]


def bfs(adj_matrix: pgb.Matrix, start: int) -> List[int]:
    """
    Performs breadth-first search algorithm on the given unweighted graph to
    find the distance to each vertex from the starting vertex.

    :param adj_matrix: adjacency matrix of `BOOL` data type for a graph to perform bfs on
    :param start: index of the starting vertex
    :return: a list of integers representing distance to each vertex from the starting vertex
    :raises ValueError: if the `adj_matrix` is not a square matrix or if it is not of `BOOL` data type
    :raises IndexError: if the `start` is out of range
    """
    if not adj_matrix.square:
        raise ValueError(
            f"Adjacency matrix must be square, provided shape: {adj_matrix.shape}"
        )
    num_vertices = adj_matrix.nrows
    if not adj_matrix.type == pgb.BOOL:
        raise ValueError(
            f"Adjacency matrix must have bool type, provided type: {adj_matrix.type}"
        )
    if not (0 <= start < num_vertices):
        raise IndexError(f"Start {start} is out of range [0, {num_vertices})")
    front = pgb.Vector.sparse(pgb.BOOL, num_vertices)
    front[start] = True
    res = pgb.Vector.dense(pgb.INT64, num_vertices, fill=-1)
    visited = pgb.Vector.sparse(pgb.BOOL, num_vertices)
    visited_nvals = visited.nvals
    iter = 0
    while True:
        old_visited_nvals = visited_nvals
        res[front] = iter
        visited += front
        visited_nvals = visited.nvals
        if visited_nvals == old_visited_nvals:
            return list(res.vals)
        front.vxm(adj_matrix, out=front, mask=visited, desc=pgb.descriptor.RC)
        iter += 1
