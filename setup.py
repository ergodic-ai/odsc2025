from setuptools import setup, find_packages

setup(
    name="ergodic_cd",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "networkx>=3.4.2",
        "numpy>=1.22",
        "pandas>=2.2.2",
        "pydantic>=2.11.4",
        "scikit-learn>=1.6.1",
        "seaborn>=0.13.2",
        "tqdm>=4.67.1",
    ],
)
