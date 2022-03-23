import csv
import numpy as np

path = '/Users/nick/dev/mfml/project/1k-data-sets/'

csvfile = path + '1k-Labels.csv'
n1000_labels = np.genfromtxt(csvfile, delimiter=',')

csvfile = path + '1k-Clean.csv'
n1000_clean = np.genfromtxt(csvfile, delimiter=',')

csvfile = path + '1k-Cor-0_5ENR.csv'
n1000_cor0_5ENR = np.genfromtxt(csvfile, delimiter=',')

csvfile = path + '1k-Cor-1ENR.csv'
n1000_cor1ENR = np.genfromtxt(csvfile, delimiter=',')

csvfile = path + '1k-Cor-2ENR.csv'
n1000_cor2ENR = np.genfromtxt(csvfile, delimiter=',')

csvfile = path + '1k-SinCor-0_5ENR.csv'
n1000_sincor0_5ENR = np.genfromtxt(csvfile, delimiter=',')

csvfile = path + '1k-SinCor-1ENR.csv'
n1000_sincor1ENR = np.genfromtxt(csvfile, delimiter=',')

csvfile = path + '1k-SinCor-2ENR.csv'
n1000_sincor2ENR = np.genfromtxt(csvfile, delimiter=',')

csvfile = path + '1k-WGN-0_5ENR.csv'
n1000_WGN0_5ENR = np.genfromtxt(csvfile, delimiter=',')

csvfile = path + '1k-WGN-1ENR.csv'
n1000_WGN1ENR = np.genfromtxt(csvfile, delimiter=',')

csvfile = path + '1k-WGN-2ENR.csv'
n1000_WGN2ENR = np.genfromtxt(csvfile, delimiter=',')

path = '/Users/nick/dev/mfml/project/3k-data-sets/'

csvfile = path + '3k-Labels.csv'
n3000_labels = np.genfromtxt(csvfile, delimiter=',')

csvfile = path + '3k-Clean.csv'
n3000_clean = np.genfromtxt(csvfile, delimiter=',')

csvfile = path + '3k-Cor-0_5ENR.csv'
n3000_cor0_5ENR = np.genfromtxt(csvfile, delimiter=',')

csvfile = path + '3k-Cor-1ENR.csv'
n3000_cor1ENR = np.genfromtxt(csvfile, delimiter=',')

csvfile = path + '3k-Cor-2ENR.csv'
n3000_cor2ENR = np.genfromtxt(csvfile, delimiter=',')

csvfile = path + '3k-SinCor-0_5ENR.csv'
n3000_sincor0_5ENR = np.genfromtxt(csvfile, delimiter=',')

csvfile = path + '3k-SinCor-1ENR.csv'
n3000_sincor1ENR = np.genfromtxt(csvfile, delimiter=',')

csvfile = path + '3k-SinCor-2ENR.csv'
n3000_sincor2ENR = np.genfromtxt(csvfile, delimiter=',')

csvfile = path + '3k-WGN-0_5ENR.csv'
n3000_WGN0_5ENR = np.genfromtxt(csvfile, delimiter=',')

csvfile = path + '3k-WGN-1ENR.csv'
n3000_WGN1ENR = np.genfromtxt(csvfile, delimiter=',')

csvfile = path + '3k-WGN-2ENR.csv'
n3000_WGN2ENR = np.genfromtxt(csvfile, delimiter=',')

def get_training_set(n_per_class, training_images):
    all_images = []
    all_labels = []
    for i in range(10):
        class_images = []
        class_labels = []
        for j in range(len(n1000_labels)):
            if n1000_labels[j] == i:
                class_images.append(training_images[:, j])
                class_labels.append(n1000_labels[j])
            if len(class_images) == n_per_class:
                break
        all_images += class_images
        all_labels += class_labels
    return np.array(all_images).T, all_labels