import pygraphblas as pgb

__all__ = ["check_bool_adj_matrix", "check_start_in_range"]


def check_bool_adj_matrix(adj_matrix: pgb.Matrix):
    if not adj_matrix.square:
        raise ValueError(
            f"Adjacency matrix must be square, provided shape: {adj_matrix.shape}"
        )
    if not adj_matrix.type == pgb.BOOL:
        raise ValueError(
            f"Adjacency matrix must have bool type, provided type: {adj_matrix.type}"
        )


def check_start_in_range(start: int, num_vertices: int):
    if not (0 <= start < num_vertices):
        raise IndexError(f"Start {start} is out of range [0, {num_vertices})")
