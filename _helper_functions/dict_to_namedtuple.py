from collections import namedtuple


def dict_to_namedtuple(tuple_name: str, dictionary: dict) -> namedtuple:
    """
    Recursively converts all dictionaries into named tuples, provided that all keys are strings
    """
    tuple = namedtuple(tuple_name, dictionary.keys())

    def unpack(item):
        key, value = item

        if isinstance(value, dict):
            return dict_to_namedtuple(key, value)
        else:
            return value

    values = map(unpack, dictionary.items())
    return tuple(*values)
