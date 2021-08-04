"""
test module
"""
import unittest
import numpy as np
from src import Polynomial


class PolynomialTets(unittest.TestCase):
    """
    class docstring
    """

    def test_square_of_three_should_be_nince(self):
        """
        test method
        """
        # given
        poli = Polynomial(np.array([1, 0, 0]))
        x_value = 3
        expected = 9
        # when
        actual = poli.evaluate(x_value)
        # then
        self.assertEqual(expected, actual)
