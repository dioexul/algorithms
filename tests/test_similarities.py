# coding: utf-8

import logging
from nose.tools import assert_equal, raises
from core.exeptions import InputDataException
from core.similarities import euclidean_distance, pearson_correlation, jaccard_index, ochiai_coefficient, \
    cosine_similarity

__author__ = 'dioexul'

logger = logging.getLogger(__name__)


def test_euclidean_distance():
    attr_space_1 = [5, 7, 8, 10]
    attr_space_2 = [10, 9, 8, 4]
    result = euclidean_distance(attr_space_1, attr_space_2)
    logger.debug('function result: %s', result)
    assert_equal(result, 0.11034777731716484)


@raises(InputDataException)
def test_euclidean_distance_type_error():
    result = euclidean_distance('abs', [1, 2, 3])


@raises(InputDataException)
def test_euclidean_distance_len_error():
    euclidean_distance([1, 2, 3, 4, 5, 6], [1, 2, 3])


def test_pearson_correlation():
    attr_space_1 = [5, 7, 8, 10]
    attr_space_2 = [10, 9, 8, 4]
    result = pearson_correlation(attr_space_1, attr_space_2)
    logger.debug('function result: %s', result)
    assert_equal(result, -0.9078412990032038)


@raises(InputDataException)
def test_pearson_correlation_type_error():
    result = pearson_correlation('abs', [1, 2, 3])


@raises(InputDataException)
def test_pearson_correlation_len_error():
    pearson_correlation([1, 2, 3, 4, 5, 6], [1, 2, 3])


def test_jaccard_index():
    attr_space_1 = [5, 7, 8, 10]
    attr_space_2 = [10, 9, 8, 4, 6]
    result = jaccard_index(attr_space_1, attr_space_2)
    logger.debug('function result: %s', result)
    assert_equal(result, 0.2857142857142857)


@raises(InputDataException)
def test_jaccard_index_type_error():
    result = jaccard_index('abs', [1, 2, 3])


def test_ochiai_coefficient():
    attr_space_1 = [5, 7, 8, 10]
    attr_space_2 = [10, 9, 8, 4, 6]
    result = ochiai_coefficient(attr_space_1, attr_space_2)
    logger.debug('function result: %s', result)
    assert_equal(result, 0.1)


@raises(InputDataException)
def test_ochiai_coefficient_type_error():
    result = ochiai_coefficient('abs', [1, 2, 3])


def test_cosine_similarity():
    attr_space_1 = [5, 7, 8, 10, 7]
    attr_space_2 = [10, 9, 8, 4, 6]
    result = cosine_similarity(attr_space_1, attr_space_2)
    logger.debug('function result: %s', result)
    assert_equal(result, 0.8871163652599533)


@raises(InputDataException)
def test_cosine_similarity_type_error():
    result = cosine_similarity('abs', [1, 2, 3])


@raises(InputDataException)
def test_cosine_similarity_len_error():
    cosine_similarity([1, 2, 3, 4, 5, 6], [1, 2, 3])
