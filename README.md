# scikit-learn-knn

This package is in active development.

## Developer Guide

### Setup

After cloning the repository, install the package in editable mode with the development dependencies using:

```bash
$ pip install -e .[dev]
```

### Pre-commit

This project uses [pre-commit](https://pre-commit.com/) to run testing, linting, type-checking, and formatting. You can run pre-commit manually with:

```bash
$ pre-commit run --all-files
```

...or install it to run automatically before every commit with:
    
```bash
$ pre-commit install
```

### Testing

Unit tests are *not* run by `pre-commit`, but can be run manually with:

```bash
$ pytest
```

Measure test coverage with:

```bash
$ pytest --cov=sklearn_knn
```