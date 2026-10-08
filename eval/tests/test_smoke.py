def test_packages_import():
    import eval  # noqa: F401
    import model  # noqa: F401
    from shared import jsonlog  # noqa: F401
