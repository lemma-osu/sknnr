from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    """Remove tests marked with `uncollect_if` from parametrization.

    See https://github.com/pytest-dev/pytest/issues/3730#issuecomment-567142496
    for discussion of how to filter parametrized tests using a custom hook
    and marker."""
    removed = []
    kept = []
    for item in items:
        if m := item.get_closest_marker("uncollect_if"):
            func = m.kwargs["func"]
            if func(**item.callspec.params):
                removed.append(item)
                continue
        kept.append(item)
    if removed:
        config.hook.pytest_deselected(items=removed)
        items[:] = kept
