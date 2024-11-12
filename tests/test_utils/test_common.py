from clisops.utils.common import parse_size


def test_parse_size():
    tests = [
        ("1000000.0b", 1000000),
        ("1MiB", 1048576),
        ("1.0MB", 1000000),
        ("0.001Gb", 1000000),
        ("500Mb", 500000000),
    ]

    for test, check in tests:
        assert parse_size(test) == check
