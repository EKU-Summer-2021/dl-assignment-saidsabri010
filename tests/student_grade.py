"""
test module
"""
import unittest
import pandas as pd
from src.student_grade import KerasModel


class MyTestCase(unittest.TestCase):
    """
    test class
    """

    def test_build_model(self):
        """
        test method
        """
        data = pd.read_csv('https://raw.githubusercontent.com/saidsabri010/dataset/main/Concrete_Data_Yeh.csv')

        instance = KerasModel(data.drop(columns=['csMPa']),
                              data['csMPa'])  # pylint: disable=E1136

        actual = instance.run_model()
        expected = [40, 40]
        self.assertLess(actual, expected)
