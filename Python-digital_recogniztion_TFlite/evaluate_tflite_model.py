import tensorflow as tf 
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt 
import random

# A helper function to evaluate the TF Lite model using "test" dataset.
def evaluate_tflite_model(tflite_model_path):
  # Initialize TFLite interpreter using the model.
  interpreter = tf.lite.Interpreter(model_path = tflite_model_path)
  interpreter.allocate_tensors()
  input_tensor_index = interpreter.get_input_details()[0]["index"]
  output = interpreter.tensor(interpreter.get_output_details()[0]["index"])

  # Run predictions on every image in the "test" dataset.
  prediction_digits = []
  for test_image in test_images:
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_tensor_index, test_image)

    # Run inference.
    interpreter.invoke()

    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    digit = np.argmax(output()[0])
    prediction_digits.append(digit)

  # Compare prediction results with ground truth labels to calculate accuracy.
  accurate_count = 0
  for index in range(len(prediction_digits)):
    if prediction_digits[index] == test_labels[index]:
      accurate_count += 1
  accuracy = accurate_count * 1.0 / len(prediction_digits)

  return accuracy

# Evaluate the TF Lite float model. You'll find that its accurary is identical
# to the original TF (Keras) model because they are essentially the same model
# stored in different format.
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images/255.0
test_images = test_images/255.0

tflite_float_model_path='/workspace/tensorflow-lite/digital_recogniztion_TFlite/tflite_float_model.tflite'

float_accuracy = evaluate_tflite_model(tflite_float_model_path)
print('Float model accuracy = %.4f' % float_accuracy)

# Evalualte the TF Lite quantized model.
# Don't be surprised if you see quantized model accuracy is higher than
# the original float model. It happens sometimes :)
tflite_quantized_model_path='/workspace/tensorflow-lite/digital_recogniztion_TFlite/tflite_quantized_model.tflite'
quantized_accuracy = evaluate_tflite_model(tflite_quantized_model_path)
print('Quantized model accuracy = %.4f' % quantized_accuracy)
print('Accuracy drop = %.4f' % (float_accuracy - quantized_accuracy))
