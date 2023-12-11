import tensorflow as tf

def preprocess_image(image, fe):
    """
    Preprocess the image to be ready for the model.
    :param image: The image to preprocess.
    :return: The preprocessed image.
    """
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image /= 255.0  # normalize to [0,1] range
    image = tf.reshape(image, (1, 224, 224, 3))
    image = fe.predict(image, verbose=0)
    return image