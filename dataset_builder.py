
import tensorflow as tf
import numpy
import utils

class_count = 10

class Dataset():
    def __init__(self, mode='train', size=(90, 33)):
        self.data_path = utils.dataset_folder
        self.descript_file_path = f'{self.data_path}/{mode}/{mode}.txt'
        self.mode = mode
        self.data_index = 0
        self.size = size

    def get_data(self):
        f = open(self.descript_file_path, 'r')
        files = f.readlines()
        f.close()
        
        dataset_size = len(files)

        x_data = numpy.empty([dataset_size, self.size[0], self.size[1], 1])
        y_data = numpy.empty([dataset_size, class_count])
        for i in range(dataset_size):
            file = files[i][:-1]
            x = numpy.load(f'{self.data_path}/{self.mode}/elements/{file}.npy')
            x_data[i] = tf.reshape(x, [self.size[0], self.size[1], 1])

            f = open(f'{self.data_path}/{self.mode}/text/{file}.txt', 'r')
            class_index = int(f.read())
            f.close()
            class_predictions = numpy.zeros(class_count)
            class_predictions[class_index] = 1
            y_data[i] = class_predictions
        return x_data, y_data
