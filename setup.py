#!/usr/bin/env python

import pathlib
from setuptools import find_packages, setup


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="qtn-design",  # Replace with your own username
    version="0.0.1",
    author="Matt Thibodeau, Nicolas Sawaya",
    author_email="nicolas.sawaya@intel.com",
    description="Automated quantum tensor network design.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/intel-sandbox/applications.quantum.pando-tn",
    license="Apache 2",
    # package_dir={'': 'src'},
    packages=["qtn-design"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Apache 2",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7", # required for ordered dicts.
    install_requires=["quimb>=1.4", "scipy>=1.1", "torch>=1.12"],
    extras_require={
        "dev": [
        ]
    },
)
