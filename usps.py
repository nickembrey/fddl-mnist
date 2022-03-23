import numpy as np
import h5py

path = './usps.h5'

with h5py.File(path, 'r') as hf:
    training_set = hf.get('train')
    training_images = training_set.get('data')[:]
    training_labels = training_set.get('target')[:]
    testing_set = hf.get('test')
    testing_images = testing_set.get('data')[:]
    testing_labels = testing_set.get('target')[:]

all_images = []
all_labels = []
for i in range(10):
    class_images = []
    class_labels = []
    for j in range(len(training_labels)):
        if training_labels[j] == i:
            class_images.append(training_images[j])
            class_labels.append(training_labels[j])
        if len(class_images) == 500:
            break
    all_images += class_images
    all_labels += class_labels

training_images = all_images
training_labels = all_labels

training_images = np.array(training_images).T

testing_images = testing_images[:1000]
testing_labels = testing_labels[:1000]

testing_images = np.array(testing_images).T
