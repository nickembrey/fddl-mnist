from mnist import MNIST
import numpy as np

# https://pypi.org/project/python-mnist/
path = './mnist'
mndata = MNIST(path)
training_images, training_labels = mndata.load_training()
testing_images, testing_labels = mndata.load_testing()
training_labels, training_images = list(zip(*sorted(zip(training_labels, training_images), key=lambda x: x[0])))
training_images = np.array(training_images).T
testing_images = np.array(testing_images).T

def get_training_set(n_per_class):
    all_images = []
    all_labels = []
    for i in range(10):
        class_images = []
        class_labels = []
        for j in range(len(training_labels)):
            if training_labels[j] == i:
                class_images.append(training_images[:, j])
                class_labels.append(training_labels[j])
            if len(class_images) == n_per_class:
                break
        all_images += class_images
        all_labels += class_labels
    return np.array(all_images).T, all_labels