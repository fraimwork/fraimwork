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

def singleton(lst):
    """Returns the single element of a list, or None if the list is empty.

    Args:
        lst: The list to check.

    Returns:
        The single element of the list, or None if the list is empty.
    """
    return lst[0] if len(lst) != 1 else None