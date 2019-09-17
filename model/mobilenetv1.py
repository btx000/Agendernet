import numpy as np
import os
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.utils import plot_model


class AgenderNetMobileNetV1(Model):
    """Classification model based on MobileNetV1 with 2 outputs, each for age and gender
    """

    def __init__(self):
        self.input_size = 128
        base = MobileNet(input_shape=(128, 128, 3), alpha=0.5, include_top=False, weights=os.path.dirname(os.path.abspath(__file__))+'/weight/mobilenetv1/mobilenet_5_0_128_tf_no_top.h5')
        top_layer = GlobalAveragePooling2D()(base.output)
        gender_layer = Dense(2, activation='softmax', name='gender_prediction')(top_layer)
        age_layer = Dense(101, activation='softmax', name='age_prediction')(top_layer)
        super().__init__(inputs=base.input, outputs=[gender_layer, age_layer], name='AgenderNetMobileNetV1')
    
    def prep_phase1(self):
        for layer in model.layers[:86]:
            layer.trainable=False
        for layer in model.layers[86:]:
            layer.trainable=True
    
    def prep_phase2(self):
        pass

    @staticmethod
    def decode_prediction(prediction):
        """
        Decode prediction to age and gender prediction.
        Use softmax regression for age and argmax for gender.
        Parameters
        ----------
        prediction : list of numpy array
            Result from model prediction [gender, age]
        Return
        ----------
        gender_predicted : numpy array
            Decoded gender 1 male, 0 female
        age_predicted : numpy array
            Age from softmax regression
        """
        gender_predicted = np.argmax(prediction[0], axis=1)
        age_predicted = prediction[1].dot(np.arange(0, 101).reshape(101, 1)).flatten()
        return gender_predicted, age_predicted

    @staticmethod
    def prep_image(data):
        """Preproces image specific to model

        Parameters
        ----------
        data : numpy ndarray
            Array of N images to be preprocessed

        Returns
        -------
        numpy ndarray
            Array of preprocessed image
        """
        data = data.astype('float16')
        data /= 128.
        data -= 1.
        return data


if __name__ == '__main__':
    model = AgenderNetMobileNetV1()
    print(model.summary())
    for (i, layer) in enumerate(model.layers):
        print(i, layer.name)