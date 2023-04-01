import pytest

from project.shortest_path import all_pairs_floyd_warshall
from project.shortest_path import multi_source_bellman_ford
from project.shortest_path import single_source_bellman_ford
from tests.utils import read_adj_matrix_for_weighted_digraph


def test_multi_source_bellman_ford(config_data: dict):
    adj_matrix = read_adj_matrix_for_weighted_digraph(config_data["graph"])
    starts = [int(start) for start in config_data["starts"]]
    if bool(config_data["has-reachable-negative-cycle"]):
        with pytest.raises(ValueError):
            multi_source_bellman_ford(adj_matrix, starts)
    else:
        expected_result = [
            (int(start), [float(dist) for dist in parents])
            for (start, parents) in config_data["expected-result"].items()
        ]
        actual_result = multi_source_bellman_ford(adj_matrix, starts)
        assert actual_result == expected_result


def test_single_source_bellman_ford(config_data: dict):
    adj_matrix = read_adj_matrix_for_weighted_digraph(config_data["graph"])
    start = int(config_data["start"])
    if bool(config_data["has-reachable-negative-cycle"]):
        with pytest.raises(ValueError):
            single_source_bellman_ford(adj_matrix, start)
    else:
        expected_result = [float(dist) for dist in config_data["expected-result"]]
        actual_result = single_source_bellman_ford(adj_matrix, start)
        assert actual_result == expected_result


def test_all_pairs_floyd_warshall(config_data: dict):
    adj_matrix = read_adj_matrix_for_weighted_digraph(config_data["graph"])
    if bool(config_data["has-negative-cycle"]):
        with pytest.raises(ValueError):
            all_pairs_floyd_warshall(adj_matrix)
    else:
        expected_result = [
            (int(start), [float(dist) for dist in parents])
            for (start, parents) in config_data["expected-result"].items()
        ]
        actual_result = all_pairs_floyd_warshall(adj_matrix)
        assert actual_result == expected_result
