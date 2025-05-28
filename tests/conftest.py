from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def data_dir():
    return Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def aion_model_path():
    return "polymathic-ai/aion-base"
