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

def partition(lst, percentage_breakdown):
    """Splits a list into two lists based on a percentage breakdown.

    Args:
        lst: The list to split.
        percentage_breakdown: An array of floats that sum to 1

    Returns:
        A tuple of two lists, where the first list contains the first

    """
    assert sum(percentage_breakdown) == 1

    total = len(lst)
    partitioned = []
    for percentage in percentage_breakdown:
        partitioned.append(lst[:int(total * percentage)])
        lst = lst[int(total * percentage):]

    return partitioned