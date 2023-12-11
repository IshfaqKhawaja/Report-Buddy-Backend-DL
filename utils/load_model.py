from tensorflow.keras.models import load_model

def load():
    model = load_model('static/model.h5')
    return model