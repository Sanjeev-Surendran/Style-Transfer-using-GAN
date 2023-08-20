# Import numpy
import numpy as np
# Import tensorflow
import tensorflow as tf
# Import load image
from tensorflow.keras.preprocessing.image import load_img
# Import load model
from tensorflow.keras.models import load_model
# Import image to array
from tensorflow.keras.utils import img_to_array
# Import app specific functions
from constants import *

# Class for Instance Normalization
class InstanceNormalization(tf.keras.layers.Layer):
    # Initialization of Objects
    def __init__(self, epsilon=1e-5, name=None, **kwargs):
        # Calling parent's init
        super(InstanceNormalization, self).__init__(name=name)
        self.epsilon = epsilon
        super(InstanceNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        # Compute Mean and Variance, Axes=[1,2] ensures Instance Normalization
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset
    
    def get_config(self):
        config = super(InstanceNormalization, self).get_config()
        config.update({"epsilon": self.epsilon})
        return config

def process_image(file):
    img_height = IMG_HEIGHT
    img_width = IMG_WIDTH
    
    # Load the image
    # Resize the image to (img_height, img_width)
    # Choose color mode as grayscale
    img = load_img(file, color_mode = "grayscale", target_size=(img_height, img_width))
    
    # Convert image to array
    img = img_to_array(img)
    
    # Normalize the image in range of [-1, 1]
    # Grayscale values range from [0, 255], hence dividing by 127.5
    # and subtracting by -1.0 will bring the range to [-1, 1]
    img = ((img / 127.5) - 1.0)
    
    # Reshape
    img = img.reshape(1, img_height, img_width, 1)
    
    # Expand dimensions
    img = np.expand_dims(img, axis=0)
    
    # Typecast to float32
    img = img.astype('float32')
    # Convert to tensor and return it
    return tf.data.Dataset.from_tensor_slices(img)

def generate_image(uploaded_file):
    # Load the generators and also load custom layer - InstanceNormalization
    model_g = load_model(MODEL_G_NAME, custom_objects={'InstanceNormalization': InstanceNormalization})
    model_f = load_model(MODEL_F_NAME, custom_objects={'InstanceNormalization': InstanceNormalization})
    
    # Process Image
    input_image_tslice = process_image(uploaded_file)
    
    # Generate Images
    for input_image in tf.data.Dataset.zip((input_image_tslice)):
        prediction1 = model_g(input_image)
        prediction2 = model_f(input_image)
    
    t2_image = prediction1[0].numpy()[:, :, 0] * 0.5 + 0.5
    t1_image = prediction2[0].numpy()[:, :, 0] * 0.5 + 0.5
    
    return t1_image, t2_image