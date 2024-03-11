from tensorflow import keras
import tensorflow as tf
from dataset_builder import Dataset
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
from utils import convert

save_path = './model.keras'
save_sub_path = './model_sub.keras'

# class Network(keras.Model):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = Conv2D(64, (7, 7), padding='valid', activation='relu')
#         self.pool1 = MaxPooling2D((2, 2), strides=2)
#         self.conv2 = Conv2D(96, (5, 5), padding='valid', activation='relu')
#         self.pool2 = MaxPooling2D((2, 2), strides=2)
#         self.conv3 = Conv2D(256, (3, 3), padding='valid', activation='relu')
#         self.conv4 = Conv2D(256, (2, 2), padding='valid', activation='relu')
#         self.flat = Flatten()
#         self.dense1 = Dense(4096, activation='relu')
#         self.dense2 = Dense(4096, activation='relu')
#         self.dense3 = Dense(10,  activation='softmax')

#     def call(self, input):
#         x = self.conv1(input)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.pool2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.flat(x)
#         x = self.dense1(x)
#         x = self.dense2(x)
#         output = self.dense3(x)

#         return output


def get_model(input):
    x = Conv2D(64, (7, 7), padding='valid', activation='relu')(input)
    x = MaxPooling2D((2, 2), strides=2)(x)
    x = Conv2D(96, (5, 5), padding='valid', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=2)(x)
    x = Conv2D(256, (3, 3), padding='valid', activation='relu')(x)
    x = Conv2D(256, (2, 2), padding='valid', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    output = Dense(10,  activation='softmax')(x)

    model = tf.keras.Model(inputs=input, outputs=output)
    return model


def compile_model():
    input = tf.keras.Input(shape=(90, 33, 1))
    model = get_model(input)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    model = compile_model()
    try:
        model = tf.keras.models.load_model(save_path)
    except:
        model = compile_model()

    train_dataset = Dataset()
    x_train, y_train = train_dataset.get_data()
    randomize = np.arange(len(x_train))
    np.random.shuffle(randomize)
    x_train = x_train[randomize]
    y_train = y_train[randomize]

    history = model.fit(x_train, y_train, batch_size=32,
                        epochs=100, validation_split=0.2)
    print(history.history)
    model.save(save_path)
    model.save(save_sub_path)

    test_dataset = Dataset(mode='test')
    x_test, y_test = test_dataset.get_data()
    results = model.evaluate(x_test, y_test, batch_size=32)
    print("test loss, test acc:", results)
