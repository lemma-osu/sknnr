from setuptools import find_packages, setup

from src.sklearn_knn import __version__


def setup_package():
    metadata = dict(
        name="scikit-learn-knn",
        author="LEMMA group @ Oregon State University",
        author_email="matt.gregory@oregonstate.edu",
        description="Scikit-learn estimators for kNN methods",
        url="http://github.com/lemma-osu/scikit-learn-knn/",
        version=__version__,
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        python_requires=">=3.8",
        install_requires=[
            "numpy",
            "scikit-learn",
        ],
        extras_require={
            "dev": [
                "black",
                "bumpversion",
                "flake8",
                "isort",
                "pytest",
                "pytest-cov",
                "pandas",
                "pre-commit",
                "mypy",
                "twine",
            ],
        },
    )
    setup(**metadata)


if __name__ == "__main__":
    setup_package()
