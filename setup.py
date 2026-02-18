from setuptools import setup

setup(
    name="tranquillyzer",
    version="0.2.1",
    packages=[".", "scripts", "utils", "wrappers"],
    py_modules=["main"],
    include_package_data=True,
    package_data={
        "": ["models/*", "utils/*.tsv"],
    },
    install_requires=[
        "numpy",
        "pandas",
        "polars",
        "matplotlib",
        "seaborn",
        "tqdm",
        "filelock",
        "tensorflow",
        "tensorflow-addons",
        "rapidfuzz",
        "pysam",
        "numba",
        "typer",
        "biopython",
        "python-Levenshtein",
        "scikit-learn",
    ],
    entry_points={
        "console_scripts": [
            "tranquillyzer=main:app",
        ],
    },
    python_requires=">=3.10,<3.12",
)
