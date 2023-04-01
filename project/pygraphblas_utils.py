import pygraphblas as pgb

__all__ = ["vector_to_list", "matrix_to_list_of_key_and_value_list_tuples"]


def vector_to_list(vector: pgb.Vector, default_value) -> list:
    res = [default_value for _ in range(vector.size)]
    for i, v in vector:
        res[i] = v
    return res


def matrix_to_list_of_key_and_value_list_tuples(
    matrix: pgb.Matrix, row_keys: list, default_value
) -> list:
    if len(row_keys) != matrix.nrows:
        raise ValueError("Number of keys isn't equal to number of rows")
    res = [
        (row_keys[i], [default_value for _ in range(matrix.ncols)])
        for i in range(matrix.nrows)
    ]
    for i, j, v in matrix:
        res[i][1][j] = v
    return res
