import pytest
import project.triangles

from project.triangles import count_triangles_for_each_vertex
from tests.utils import read_adj_matrix_for_undirected_graph


def test_count_triangles_for_each_vertex(config_data: dict):
    adj_matrix = read_adj_matrix_for_undirected_graph(config_data["graph"])
    expected_result = [int(num) for num in config_data["expected-result"]]
    actual_result = count_triangles_for_each_vertex(adj_matrix)
    assert actual_result == expected_result


@pytest.mark.parametrize(
    "tested_fun", ["count_triangles_cohen", "count_triangles_sandia"]
)
def test_count_triangles(config_data: dict, tested_fun: str):
    adj_matrix = read_adj_matrix_for_undirected_graph(config_data["graph"])
    expected_result = int(config_data["expected-result"])
    actual_result = getattr(project.triangles, tested_fun)(adj_matrix)
    assert actual_result == expected_result
