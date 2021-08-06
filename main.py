"""
this is the main module
"""
import pandas as pd
from src.student_grade import KerasModel

data = pd.read_csv('https://raw.githubusercontent.com/saidsabri010/dataset/main/Concrete_Data_Yeh.csv')

instance = KerasModel(data.drop(columns=['csMPa']),
                      data['csMPa']) # pylint: disable=E1136
print(instance.run_model())
