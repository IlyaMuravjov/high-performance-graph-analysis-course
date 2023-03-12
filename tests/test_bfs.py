import pytest
import pygraphblas as pgb

from project.bfs import bfs
from tests.utils import read_adj_matrix_for_graph


def test_bfs(config_data: dict):
    adj_matrix = read_adj_matrix_for_graph(config_data["graph"])
    start = int(config_data["start"])
    expected_result = [int(dist) for dist in config_data["expected-result"]]
    actual_result = bfs(adj_matrix, start)
    assert actual_result == expected_result


def test_bfs_non_bool_matrix():
    with pytest.raises(ValueError):
        bfs(pgb.Matrix.sparse(pgb.INT64, 1, 1), 0)


def test_bfs_non_square_matrix():
    with pytest.raises(ValueError):
        bfs(pgb.Matrix.sparse(pgb.BOOL, 1, 2), 0)


def test_bfs_negative_start():
    with pytest.raises(IndexError):
        bfs(pgb.Matrix.sparse(pgb.BOOL, 1, 1), -1)


def test_bfs_too_large_start():
    with pytest.raises(IndexError):
        bfs(pgb.Matrix.sparse(pgb.BOOL, 1, 1), 1)
