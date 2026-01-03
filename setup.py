from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="RobustPortfolio",
    version="0.1.0",
    description="Advanced quantitative tools for robust portfolio optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ton Vossen",
    url="https://github.com/KoperSloper/RobustPortfolio",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "cvxpy",
        "matplotlib",
        "seaborn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)