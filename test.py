from network import compile_model, save_path
from utils import convert, show_spectro
import tensorflow as tf
import numpy as np


file_name = './tests/0_01_0.wav'

model = compile_model()
model = tf.keras.models.load_model(save_path)
spectro = convert(file_name)

input = tf.reshape(spectro, [1, spectro.shape[0], spectro.shape[1], 1])
out_vals = model.predict(input)[0]

output = np.where(out_vals == np.max(out_vals, axis=0))[0][0]
print(f'\npredicted class >>>>>>>>: {output}\n')
show_spectro(spectro)
