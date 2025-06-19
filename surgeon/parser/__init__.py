"""
Parser package for Git-Surgeon.

Contains modules for natural language processing, intent classification,
and command generation for Git operations.
"""

from .classifier import IntentClassifier

# Export key classes
__all__ = ['IntentClassifier']
