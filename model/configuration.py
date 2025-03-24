import json
import os

from model import Error


def load_configuration(descriptor):
    """
    Load configuration from the given descriptor.
    Args:
        descriptor:
    Returns:
    """
    if not os.path.exists(descriptor):
        raise Error(f'Configuration file {descriptor} '
                          f'not found')
    with open(descriptor, 'r') as stream:
        return json.load(stream)
