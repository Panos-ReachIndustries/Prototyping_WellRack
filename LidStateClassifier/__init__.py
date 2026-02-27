"""
LidStateClassifier module for deterministic lid state classification in rack columns.

This module provides a non-deep-learning approach to classify whether rack columns
have open or closed lids based on image features extracted from vial patches.
"""

from .classifier import LidStateClassifier

__all__ = ['LidStateClassifier']
