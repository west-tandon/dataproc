"""Provides standard IR measures."""

def precision_at(k):
    """Returns a function operating on a data frame, which calculates P@k"""
    return lambda s: s[:k].sum() / s[:k].count()
