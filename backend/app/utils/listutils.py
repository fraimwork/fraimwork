def flatten(lst):
    """Flattens a list of arbitrary depth into a single list.

    Args:
        lst: The list to flatten.

    Returns:
        A flattened list.
    """

    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

def compact(lst):
    """Removes all falsy values from a list.

    Args:
        lst: The list to compact.

    Returns:
        A list with all falsy values removed.
    """

    return [x for x in lst if x]