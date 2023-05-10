# Testing With Pytest

In order to run the tests first install the requirements in your environment
with `pip install -r requirements_tests.txt`.

Then you can run the test suite with
```bash
pytest -v -m fast
```

This will run all the tests tagged as "fast", which is the default for CI.

## Tagging The Tests
These tags are currently in use:

| Tag          | Description                             |
|--------------|-----------------------------------------|
| `fast`       | Marks test as fast for CI.              |
| `slow`       | Marks test as slow for local testing.   |

To specify any of the tags use the `-m` tag:
```bash
pytest -v -m $TAG
```

New tests should be tagged according to whether they should be tested in CI or
only on local machines (e.g. because of performance constraints.)

## Running Specific Tests
Specific tests can be run with the `-k` flag. For example if we want to run 
tests that contain "NerDL" we can run these tests like so: 

```bash
pytest -v -k "NerDL"
```

### JetBrains IDEs

To run individual tests in a JetBrains IDE (i.e. IntelliJ or PyCharm) there might be issues when using to interface to 
run the tests. To fix this, go to _Project (Alt+1)_, choose the `python` folder and mark the directory with _Mark 
Directory as > Test Sources Root_.

# Tests with unittest
Third party app integration still uses the old unittests, as these require
additional dependencies that are not needed for the other tests.

To run these, run the python file `../run-tests-thirdparty.py`.