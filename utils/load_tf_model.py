from tensorflow.keras.applications import  DenseNet201
from tensorflow.keras.models import Model
def load_tf_model():
    model = DenseNet201()
    fe = Model(inputs=model.input, outputs=model.layers[-2].output)
    return fe
