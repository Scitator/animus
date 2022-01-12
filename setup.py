#!/usr/bin/env python
import os

from setuptools import setup, find_packages

NAME = "animus"
DESCRIPTION = "Animus is a minimalistic framework to run machine learning experiments."
URL = "https://github.com/Scitator/animus"
EMAIL = "scitator@gmail.com"
AUTHOR = "Sergey Kolesnikov"
REQUIRES_PYTHON = ">=3.6.0"
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
here = os.path.dirname(os.path.abspath(__file__))

setup(
    name=NAME,
    version="0.0.1",
    url=URL,
    download_url=URL,
    description=DESCRIPTION,
    license="Apache License 2.0",
    author=AUTHOR,
    author_email=EMAIL,
    long_description=open(os.path.join(here, "README.md")).read(),
    long_description_content_type="text/markdown",
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(exclude=("tests", "examples",)),
    install_requires=[],
    include_package_data=True,
    keywords=[
        "Machine Learning",
        "Deep Learning",
    ],
    classifiers=[
        "Natural Language :: English",
        "Development Status :: 4 - Beta",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
        # Audience
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        # Topics
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
        # Programming
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
