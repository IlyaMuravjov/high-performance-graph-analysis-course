import json
import pathlib
import textwrap


def pytest_generate_tests(metafunc):
    """
    Configures `config_data` fixture

    Reads an array of test data from `{test_folder}/config/{test_file}/{test_function}.json`
    and uses each element of that array as a `config_data` argument for the test function

    :param metafunc: test function to parametrize
    :return: None
    """
    if "config_data" not in metafunc.fixturenames:
        return
    test_file_path = pathlib.Path(metafunc.module.__file__)
    test_config = (
        test_file_path.parent
        / "config"
        / test_file_path.with_suffix("").name
        / metafunc.function.__name__
    ).with_suffix(".json")
    if not test_config.exists():
        test_config = (
            test_file_path.parent / "config" / test_file_path.with_suffix("").name
        ).with_suffix(".json")
    test_config_content = json.loads(test_config.read_text())
    if not isinstance(test_config_content, list):
        raise Exception(
            f"config_data for {metafunc.function.__name__} in {metafunc.module.__file__} isn't a list"
        )
    metafunc.parametrize(
        "config_data",
        test_config_content,
        ids=(
            textwrap.shorten(config_data.__str__(), 100, placeholder="...")
            for config_data in test_config_content
        ),
    )
