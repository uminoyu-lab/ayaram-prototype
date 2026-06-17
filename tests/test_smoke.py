"""M0 smoke test: every module in the package imports cleanly.

This is intentionally minimal -- M0 ships stubs only. Real behavioral tests
will land alongside M1 implementation.
"""

import importlib


def test_ayaram_imports():
    for name in (
        "ayaram",
        "ayaram.core",
        "ayaram.modes",
        "ayaram.learning",
        "ayaram.memory",
    ):
        importlib.import_module(name)


def test_demos_import():
    for name in (
        "demos",
        "demos.kanji_memory",
        "demos.ising_solver",
        "demos.attention_test",
    ):
        importlib.import_module(name)


def test_version_string():
    import ayaram

    assert isinstance(ayaram.__version__, str)
    assert ayaram.__version__
