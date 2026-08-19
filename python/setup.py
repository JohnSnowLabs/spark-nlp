"""Compatibility shim.

All project metadata now lives in pyproject.toml (PEP 621). This file remains
only so that legacy `python setup.py ...` invocations keep working; prefer
`python -m build`.
"""

from setuptools import setup

setup()
