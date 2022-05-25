"""
Welcome to pyGNMF !
---------------------
pyGNMF is a Python package for implementation of Generalised Non-negative Matrix Factorisation method introduced in "Generalised Non-negative Matrix Factorisation for Air Pollution Source Apportionment" published in "Science of Total Environment"

Help us develop the package further - we appreciate any feedback !

Resources
---------
The original article:
https://---
"""

__version__ = "1.0.0"
__author__ = "Nirav L. Lekinwala & Mani Bhushan"
__all__ = [
    "multiplicative",
    "projectedGradient",
    "__version__",
    "__author__",
]

## Importing necessary libraries
from pyGNMF import multiplicative, projectedGradient