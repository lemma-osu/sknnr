from setuptools import setup, find_packages

VERSION = "0.1.0"


def setup_package():
    metadata = dict(
        name="scikit-learn-knn",
        author="LEMMA group @ Oregon State University",
        author_email="matt.gregory@oregonstate.edu",
        description="Scikit-learn estimators for kNN methods",
        url="http://github.com/lemma-osu/scikit-learn-knn/",
        version=VERSION,
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        install_requires=[
            "numpy",
            "scikit-learn",
        ],
        extras_require={
            "dev": [
                "pytest",
                "pandas",
            ],
        }
    )
    setup(**metadata)


if __name__ == "__main__":
    setup_package()
