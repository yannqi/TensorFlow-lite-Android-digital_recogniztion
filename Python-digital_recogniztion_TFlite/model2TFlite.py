import tensorflow as tf 
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt 
import random

model = keras.models.load_model('/workspace/tensorflow-lite/digital_recogniztion_TFlite/my_model.h5')
# Convert Keras model to TF Lite format.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_float_model = converter.convert()

# Show model size in KBs.
float_model_size = len(tflite_float_model) / 1024
print('Float model size = %dKBs.' % float_model_size)

# Re-convert the model to TF Lite using quantization.
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

# Show model size in KBs.
quantized_model_size = len(tflite_quantized_model) / 1024
print('Quantized model size = %dKBs,' % quantized_model_size)
print('which is about %d%% of the float model size.'\
      % (quantized_model_size * 100 / float_model_size))
#生成tflite文件
open("tflite_quantized_model.tflite", "wb").write(tflite_quantized_model)
open("tflite_float_model.tflite", "wb").write(tflite_float_model)