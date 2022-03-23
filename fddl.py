import numpy as np
import scipy.optimize as opt
import scipy.linalg as linalg
import mnist_data
import matplotlib.pyplot as plt


# derivative of the discriminative function with respecto to Xi
def dQ(Ai, D, Xi, k, class_index, lambda2, eta):
    first_term = -2 * D.T @ (Ai - D @ Xi)
    index = k * class_index
    Di = D[:, index:index + k]

    # https://math.stackexchange.com/a/4005213/357361
    # select a subset of rows from Xi
    t2_row_selector = np.identity(Xi.shape[0])[:, index:index + k].T
    second_term = -2 * (Di @ t2_row_selector).T @ (Ai - (Di @ t2_row_selector @ Xi))

    third_term = np.zeros(second_term.shape)
    for j in range(int(D.shape[1] / k)):
        if j != class_index:
            j_index = k * j
            Dj = D[:, j_index:j_index + k]
            t3_row_selector = np.identity(Xi.shape[0])[:, j_index:j_index + k].T
            third_term += 2 * (Dj @ t3_row_selector).T @ Dj @ t3_row_selector @ Xi

    Mi = np.tile(np.mean(Xi, axis=1), (Xi.shape[1], 1)).T
    f_term = 2 * (Xi - Mi) + 2 * lambda2 * eta * Xi
    return first_term + second_term + third_term + f_term

# randomly initialize the dictionary with unit norms
def initialize_dictionary(dataset_shape, n_classes, k):
    # https: // stackoverflow.com / a / 60022349 / 5374021
    D = None
    for n in range(n_classes * k):
        v = np.random.rand(dataset_shape[0])
        v_hat = v / linalg.norm(v)
        if D is None:
            D = v_hat
        else:
            D = np.vstack((D, v_hat))
    return D.T

# thresholding function
def S(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, np.zeros(x.shape))

# function to be minimized in testing
def get_ahat(a, current_Di, test_image, m_ii, gamma1, gamma2):
    return np.square(linalg.norm(test_image - current_Di @ a, ord=2)) \
           + gamma1 * linalg.norm(a, ord=1) \
           + gamma2 * np.square(linalg.norm(a - m_ii, ord=2))

# main algorithm function
def FDDL(A, Y, TA, TY, k, n_classes, lambda1, lambda2, gamma1, gamma2, eta):
    D = initialize_dictionary((A.shape[0], A.shape[1]), n_classes, k)
    X = np.zeros((n_classes * k, A.shape[1]))

    for iteration in range(20):

        for i in range(n_classes):
            i_index = int(A.shape[1] / n_classes) * i

            for _ in range(5):
                Xi = X[:, i_index:i_index + int(A.shape[1] / n_classes)]

                Ai = A[:, i_index:i_index + int(A.shape[1] / n_classes)]
                change = 0.01 * dQ(Ai, D, Xi, k, i, lambda2, eta)
                Xi = S(Xi - change, tau=(lambda1 / 2))
                X[:, i_index:i_index + int(A.shape[1] / n_classes)] = Xi
        for i in range(n_classes):
            i_index = int(A.shape[1] / n_classes) * i
            Ai = A[:, i_index:i_index + int(A.shape[1] / n_classes)]
            Di = D[:, i * k:i * k + k]
            Xi = X[:, i_index:i_index + int(A.shape[1] / n_classes)]
            for column in range(Di.shape[1]):
                x_row = Xi[column, :]
                if (x_row != 0).any():
                    Di[:, column] = (Ai @ x_row.T) / linalg.norm(Ai @ x_row.T, ord=2)
            D[:, i * k:i * k + k] = Di

    # testing
    correct = 0
    total = TA.shape[1]

    for i in range(total):
        test_sample = TA[:, i]
        errors = []

        print(f"Image {i} of {total}...\n")

        running_accuracy = 0
        if i != 0:
            running_accuracy = correct / i

        print(f"Accuracy so far: {running_accuracy}...\n")
        for current_class in range(n_classes):
            index = int(A.shape[1] / n_classes) * current_class
            Di = D[:, current_class * k:current_class * k + k]
            Xi = X[:, index:index + int(A.shape[1] / n_classes)]
            Xii = Xi[current_class * k:current_class * k + k, :]
            m_ii = np.mean(Xii, axis=1)

            guess = np.random.rand(k)
            guess /= linalg.norm(guess)
            ahat = opt.minimize(get_ahat, guess, args=(Di, test_sample, m_ii, gamma1, gamma2)).x
            error = np.square(linalg.norm(test_sample - Di @ ahat, ord=2)) \
                    + gamma1 * linalg.norm(ahat, ord=1) \
                    + gamma2 * np.square(linalg.norm(ahat - m_ii, ord=2))

            errors.append(error)

        min_error = min(errors)
        prediction = errors.index(min_error)
        print(f"Predicted as: {prediction}")
        if prediction == TY[i]:
            correct += 1

    return correct / total


n_per_class = [1000]

for n in n_per_class:
    training_images, training_labels = mnist_data.get_training_set(n)

    accuracy = FDDL(
        A=training_images,
        Y=training_labels,
        TA=mnist_data.testing_images[:, :1000],
        TY=mnist_data.testing_labels[:1000],
        k=5,
        n_classes=10,
        lambda1=0.1,
        gamma1=0.1,
        lambda2=0.001,
        gamma2=0.005,
        eta=0.0001
    )

    plt.scatter(n * 10, accuracy)

plt.title(f"Accuracy by training set size")
plt.xlabel('Training set size')
plt.ylabel('Accuracy')

plt.show()