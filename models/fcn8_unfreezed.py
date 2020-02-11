import keras

from models import fcn8


class Model(fcn8.Model):

    def get_model(self, **kwargs) -> keras.Model:
        model = super().get_model()

        # Unfreeze layers
        for layer in model.layers:
            if "block" in layer.name:
                layer.trainable = True

        return model
