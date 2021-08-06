"""
module for student grade prediction
"""
import keras_tuner as kt # pylint: disable=E0401
from tensorflow import keras # pylint: disable=E0401
from keras.utils.vis_utils import plot_model # pylint: disable=E0401
from sklearn.model_selection import train_test_split # pylint: disable=E0401


class KerasModel:
    """
    class for building the model
    """

    def __init__(self, data_x, data_y):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data_x, data_y, test_size=0.2)

    def build_model(self, hparameter):
        """
        function for building the model
        """
        model = keras.Sequential()
        for i in range(hparameter.Int('num_layers', 2, 20)):
            model.add(keras.layers.Dense(units=hparameter.Int('units_' + str(i), min_value=16, max_value=512, step=32),
                                         activation='relu'))
        model.add(keras.layers.Dense(1, activation='softmax'))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hparameter.Choice('learning_rate',
                                  values=[1e-2, 1e-3, 1e-4])),
            loss='mean_absolute_error', metrics=['mean_absolute_error'])
        model.fit(self.x_train, self.y_train, epochs=3)
        filename = 'EKU-2021/model_plot.png'
        plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)
        return model

    def run_model(self):
        """
        function to get the results
        """
        tuner = kt.RandomSearch(
            self.build_model,
            objective='val_mean_absolute_error',
            max_trials=5,
            project_name='EKU-2021'
        )
        tuner.search(self.x_train, self.y_train, epochs=5, validation_data=(self.x_test, self.y_test))
        best_model = tuner.get_best_models(num_models=1)[0]
        return best_model.evaluate(self.x_test, self.y_test)
