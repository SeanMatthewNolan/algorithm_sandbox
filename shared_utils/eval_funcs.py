from shared_utils.constants import EPS


def is_close_absolute(value: float, reference: float, tol=1e-5):
    """
    Function to compare value against reference for testing
    :param value: value genrated by code to test
    :param reference: referencxe value to compare against
    :param tol: relative tolerance
    :return: boolean whether value is close to reference
    """
    return abs(value - reference) < tol


def is_close_relative(value: float, reference: float, tol=1e-5):
    """
    Function to compare value against reference for testing
    :param value: value genrated by code to test
    :param reference: referencxe value to compare against
    :param tol: relative tolerance
    :return: boolean whether value is close to reference
    """
    if reference > EPS:
        return abs(value - reference) / reference < tol
    else:
        return abs(value) / EPS < tol


def is_close(value: float, reference: float, abs_tol=1e-5, rel_tol=1e-5):
    """
    Function to compare value against reference for testing
    :param value: value genrated by code to test
    :param reference: referencxe value to compare against
    :param abs_tol: absolute tolerance
    :param rel_tol: relative tolerance
    :return: boolean whether value is close to reference
    """
    return is_close_absolute(value, reference, tol=abs_tol) or is_close_relative(value, reference, tol=rel_tol)
