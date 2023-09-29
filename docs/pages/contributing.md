# Contributing

## Developer Guide

### Setup

This project uses [hatch](https://hatch.pypa.io/latest/) to manage the development environment and build and publish releases. Make sure `hatch` is [installed](https://hatch.pypa.io/latest/install/) first:

```bash
$ pip install hatch
```

Now you can [enter the development environment](https://hatch.pypa.io/latest/environment/#entering-environments) using:

```bash
$ hatch shell
```

This will install development dependencies in an isolated environment and drop you into a shell (use `exit` to leave).

### Pre-commit

Use [pre-commit](https://pre-commit.com/) to run linting, type-checking, and formatting:

```bash
$ pre-commit run --all-files
```

...or install it to run automatically before every commit with:

```bash
$ pre-commit install
```

You can run pre-commit hooks separately and pass additional arguments to them. For example, to run `black` on a single file:

```bash
$ pre-commit run black --files=src/sknnr/_base.py
```

### Testing

Unit tests are *not* run by `pre-commit`, but can be run manually using `hatch` [scripts](https://hatch.pypa.io/latest/config/environment/overview/#scripts):

```bash
$ hatch run test:all
```

Measure test coverage with:

```bash
$ hatch run test:coverage
```

Any additional arguments are passed to `pytest`. For example, to run a subset of tests matching a keyword:

```bash
$ hatch run test:all -k gnn
```

### Documentation

Documentation is built with [mkdocs](https://www.mkdocs.org/). During development, you can run a live-reloading server with:

```bash
$ hatch run docs:serve
```

To build the documentation locally, use:

```bash
$ hatch run docs:build
```

!!! warning

    The built docs do not need to be committed, as they are automatically built upon deployment to ReadTheDocs.

The API reference is generated from Numpy-style docstrings using [mkdocstrings](https://mkdocstrings.github.io/). New classes can be added to the API reference by creating a new markdown file in the `docs/pages/api` directory, adding that file to the [`nav` tree](https://www.mkdocs.org/user-guide/configuration/#nav) in `docs/mkdocs.yml`, and [including the docstring](https://mkdocstrings.github.io/python/usage/#injecting-documentation) in the markdown file:

```markdown
::: sknnr.module.class
```

### Releasing

First, use `hatch` to [update the version number](https://hatch.pypa.io/latest/version/#updating).

```bash
$ hatch version [major|minor|patch]
```

Then, [build](https://hatch.pypa.io/latest/build/#building) and [publish](https://hatch.pypa.io/latest/publish/#publishing) the release to PyPI with:

```bash
$ hatch clean
$ hatch build
$ hatch publish
```
