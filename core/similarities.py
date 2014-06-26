# coding: utf-8

import logging
from math import sqrt
from core.exeptions import InputDataException

__author__ = 'dioexul'

logger = logging.getLogger(__name__)


def _correct_type_parameters(attr_space_1, attr_space_2, name):
    """ Validation of input data by type.
    :param attr_space_1: first list with attributes
    :param attr_space_2: second list with attributes
    :param name: metric name
    :return: True if parameters are correct, else raise error.
    """
    if not (isinstance(attr_space_1, list) and isinstance(attr_space_2, list)):
        logger.error('%s: attr_space is not list.', name)
        raise InputDataException('%s: attr_space is not list.', name)

    return True


def _correct_len_parameters(attr_space_1, attr_space_2, name):
    """ Validation of input data by list length.
    :param attr_space_1: first list with attributes
    :param attr_space_2: second list with attributes
    :param name: metric name
    :return: True if parameters are correct, else raise error.
    """
    if len(attr_space_1) != len(attr_space_2):
        logger.error('%s: length attr_spaces is different.', name)
        raise InputDataException('%s: length attr_spaces is different.', name)

    return True


def euclidean_distance(attr_space_1, attr_space_2):
    """Calculated Euclidean Distance of two lists with attributes.
    :param attr_space_1: first list with attributes
    :param attr_space_2: second list with attributes
    :return: Value (float) between 0 and 1, where 1 means that the objects are identical.
    """

    _correct_type_parameters(attr_space_1, attr_space_2, 'Euclidean distance')
    _correct_len_parameters(attr_space_1, attr_space_2, 'Euclidean distance')

    result = 0.0

    for attr in xrange(0, len(attr_space_1)):
        result += (attr_space_1[attr] - attr_space_2[attr]) ** 2

    result **= 0.5

    return 1 / (1 + result)


def pearson_correlation(attr_space_1, attr_space_2):
    """Calculated Pearson Correlation Coefficient of two lists with attributes.
    :param attr_space_1: first list with attributes
    :param attr_space_2: second list with attributes
    :return: Value (float) between 0 and 1, where similarity:
            None (-0.09:0.0 or 0.00:0.09)
            Small (-0.3:-0.1 or 0.1:0.3)
            Medium (-0.5:-0.3 or 0.3:0.5)
            Large (-1.0:-0.5 or 0.5:1.0)
    """

    _correct_type_parameters(attr_space_1, attr_space_2, 'Pearson correlation')
    _correct_len_parameters(attr_space_1, attr_space_2, 'Pearson correlation')

    attr_len = len(attr_space_1)

    # sum
    sum_as1 = sum([attr_space_1[attr] for attr in xrange(0, attr_len)])
    sum_as2 = sum([attr_space_2[attr] for attr in xrange(0, attr_len)])
    # sum the squares
    square_sum_as1 = sum([attr_space_1[attr] ** 2 for attr in xrange(0, attr_len)])
    square_sum_as2 = sum([attr_space_2[attr] ** 2 for attr in xrange(0, attr_len)])
    # sum the multiplication results
    multiplication_sum = sum([attr_space_1[attr] * attr_space_2[attr] for attr in xrange(0, attr_len)])

    #Pearson correlation
    numerator = multiplication_sum - (sum_as1 * sum_as2 / attr_len)
    denominator = ((square_sum_as1 - pow(sum_as1, 2)/attr_len) * (square_sum_as2 - pow(sum_as2, 2)/attr_len)) ** 0.5
    if denominator == 0:
        return 0

    result = numerator/denominator
    return result


def jaccard_index(attr_space_1, attr_space_2):
    """Calculated Jaccard/Tanimoto index of two lists with attributes.
    The Jaccard/Tanimoto Coefficient uses the ratio of the intersecting set to the union set as the measure of similarity.
    :param attr_space_1: first list with attributes
    :param attr_space_2: second list with attributes
    :return: Value (float) between 0 and 1, where 1 means that the objects are identical.
    """

    _correct_type_parameters(attr_space_1, attr_space_2, 'Jaccard index')

    # intersection = [common_item for common_item in attr_space_1 if common_item in attr_space_2]
    # result = float(len(intersection))/(len(attr_space_1) + len(attr_space_2) - len(intersection))

    # set interpretation
    set_as1 = set(attr_space_1)
    set_as2 = set(attr_space_2)
    numerator = float(len(set.intersection(set_as1, set_as2)))
    denominator = len(set.union(set_as1, set_as2))
    if denominator == 0:
        return 0
    result = numerator / denominator
    return result


def ochiai_coefficient(attr_space_1, attr_space_2):
    """Calculated Ochiai coefficient of two lists with attributes.
    :param attr_space_1: first list with attributes
    :param attr_space_2: second list with attributes
    :return: value (float)
    """

    _correct_type_parameters(attr_space_1, attr_space_2, 'Ochiai coefficient')

    set_as1 = set(attr_space_1)
    set_as2 = set(attr_space_2)
    numerator = float(len(set.intersection(set_as1, set_as2)))
    denominator = len(set_as1) * len(set_as2)

    if denominator == 0:
        return 0

    result = numerator / denominator
    return result


def cosine_similarity(attr_space_1, attr_space_2):
    """Calculated Ochiai coefficient of two lists with attributes.
    The resulting similarity ranges from −1 meaning exactly opposite, to 1 meaning exactly the same,
    with 0 usually indicating independence, and in-between values indicating intermediate similarity or dissimilarity.
    :param attr_space_1: first list with attributes
    :param attr_space_2: second list with attributes
    :return: value (float) between [-1, 1]
    """

    _correct_type_parameters(attr_space_1, attr_space_2, 'Cosine similarity')
    _correct_len_parameters(attr_space_1, attr_space_2, 'Pearson correlation')

    attr_len = len(attr_space_1)

    numerator = sum([attr_space_1[i] * attr_space_2[i] for i in xrange(0, attr_len)])

    # Normalize the first vector
    sum_as1 = sum([attr_space_1[attr] ** 2 for attr in xrange(0, attr_len)])
    norm_as1 = sqrt(sum_as1)

    # Normalize the second vector
    sum_as2 = sum([attr_space_2[i]*attr_space_2[i] for i in xrange(0, attr_len)])
    norm_as2 = sqrt(sum_as2)
    denominator = norm_as1 * norm_as2

    if denominator == 0:
        return 0

    result = numerator/denominator
    return result


# if __name__ == "__main__":
#     pass

# Yes, I know the scipy's functions:
# braycurtis(u, v)	Computes the Bray-Curtis distance between two 1-D arrays.
# canberra(u, v)	Computes the Canberra distance between two 1-D arrays.
# chebyshev(u, v)	Computes the Chebyshev distance.
# cityblock(u, v)	Computes the City Block (Manhattan) distance.
# correlation(u, v)	Computes the correlation distance between two 1-D arrays.
# cosine(u, v)	Computes the Cosine distance between 1-D arrays.
# dice(u, v)	Computes the Dice dissimilarity between two boolean 1-D arrays.
# euclidean(u, v)	Computes the Euclidean distance between two 1-D arrays.
# hamming(u, v)	Computes the Hamming distance between two 1-D arrays.
# jaccard(u, v)	Computes the Jaccard-Needham dissimilarity between two boolean 1-D arrays.
# kulsinski(u, v)	Computes the Kulsinski dissimilarity between two boolean 1-D arrays.
# mahalanobis(u, v, VI)	Computes the Mahalanobis distance between two 1-D arrays.
# matching(u, v)	Computes the Matching dissimilarity between two boolean 1-D arrays.
# minkowski(u, v, p)	Computes the Minkowski distance between two 1-D arrays.
# rogerstanimoto(u, v)	Computes the Rogers-Tanimoto dissimilarity between two boolean 1-D arrays.
# russellrao(u, v)	Computes the Russell-Rao dissimilarity between two boolean 1-D arrays.
# seuclidean(u, v, V)	Returns the standardized Euclidean distance between two 1-D arrays.
# sokalmichener(u, v)	Computes the Sokal-Michener dissimilarity between two boolean 1-D arrays.
# sokalsneath(u, v)	Computes the Sokal-Sneath dissimilarity between two boolean 1-D arrays.
# sqeuclidean(u, v)	Computes the squared Euclidean distance between two 1-D arrays.
# wminkowski(u, v, p, w)	Computes the weighted Minkowski distance between two 1-D arrays.
# yule(u, v)	Computes the Yule dissimilarity between two boolean 1-D arrays.