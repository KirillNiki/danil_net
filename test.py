from network import compile_model, save_path
from utils import convert
import tensorflow as tf
import numpy as np

file_name = './source/01/0_01_0.wav'

model = compile_model()
model = tf.keras.models.load_model(save_path)
input = convert(file_name)
input = tf.reshape(input, [1, input.shape[0], input.shape[1], 1])
out_vals = model.predict(input)[0]

output = np.where(out_vals == np.max(out_vals, axis=0))[0][0]
print(f'\npredicted class >>>>>>>>: {output}\n')
