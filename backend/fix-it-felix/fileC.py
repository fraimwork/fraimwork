# fileC contains a bug, depended on by both fileA and fileB

def validate_positive_integer(value):
    # Bug: 0 should be considered valid but isn't
    return isinstance(value, int) and value > 0