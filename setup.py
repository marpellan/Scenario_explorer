# -*- coding: utf-8 -*-

import pathlib
import setuptools

about = dict()
directory_path = pathlib.Path(__file__).parent
with open(directory_path / "trajectoire" / "__project_information__.py", "r", encoding="utf-8") as _file_descriptor:
    exec(_file_descriptor.read(), about)

with open(directory_path / "README.md", "r", encoding="utf-8") as _file_descriptor:
    readme = _file_descriptor.read()

setuptools.setup(
    name=about["__package_name__"],
    version=about["__version__"],
    description=about["__description__"],
    keywords=["jddp", "cometh", "simulation"],
    long_description=readme,
    long_description_content_type="text/markdown",
    author = about["__author__"],
    author_email=about["__author_email__"],
    url=about["__url__"],
    platforms=["Linux", "Windows"],
    packages=["trajectoire"],
    include_package_data=True,
    python_requires='>=3.9',
    install_requires=[
            "matplotlib",
            "numpy",
            "pandas",
            "pyam-iamc",
    ],
    zip_safe=False,
    classifiers=[
            "Development Status :: 2 - Pre-Alpha",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: Implementation :: CPython",
            "Topic :: Scientific/Engineering",
            "Topic :: Software Development :: Libraries",
    ],
    extras_require={
        "test": [
            "pytest-cov",
        ],
        "doc": [
            "IPython",
            "nbsphinx",
            "numpydoc",
            "Sphinx",
            "sphinx-autodoc-typehints",
            "sphinxcontrib-napoleon",
            "sphinx_rtd_theme",
        ],
    },
    project_urls={
        "Documentation": "https://dee.scm-pages.cstb.fr/dee/shape/trajectoire",
        "Source": "https://scm.cstb.fr/dee/shape/trajectoire",
    },
)
