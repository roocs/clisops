from clisops import testing


def load_test_data_single(worker_id):
    """
    This fixture ensures that the required test data repository
    has been cloned to the cache directory within the home directory.

    This is a helper function for Windows builds that do not have
    access to the locking mechanism used by macOS and Linux.
    """
    repositories = {
        "stratus": {
            "repo": testing.ESGF_TEST_DATA_REPO_URL,
            "branch": testing.ESGF_TEST_DATA_VERSION,
            "cache_dir": testing.ESGF_TEST_DATA_CACHE_DIR,
        },
        "nimbus": {
            "repo": testing.XCLIM_TEST_DATA_REPO_URL,
            "branch": testing.XCLIM_TEST_DATA_VERSION,
            "cache_dir": testing.XCLIM_TEST_DATA_CACHE_DIR,
        },
    }

    for name, repo in repositories.items():
        testing.gather_testing_data(worker_id=worker_id, **repo)


if __name__ == "__main__":
    load_test_data_single(worker_id="master")
