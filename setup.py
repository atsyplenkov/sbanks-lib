# -*- coding: utf-8 -*-
"""Setup script for sbanks-core library."""

from setuptools import setup, find_packages

setup(
    name="sbanks-core",
    version="0.1.0",
    author="Anatoly Tsyplenkov",
    author_email="atsyplenkov@geogr.msu.ru",
    description="Pure Python geometry smoothing algorithms",
    long_description=open("README.md").read()
    if __import__("os").path.exists("README.md")
    else "",
    long_description_content_type="text/markdown",
    url="https://github.com/atsyplenkov/sbanks-lib",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: GIS",
    ],
)
