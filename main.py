import numpy as np

from mnist import MNIST
mndata = MNIST('./python-mnist/data')
glob_images, glob_labels = mndata.load_training()
glob_images_test, glob_labels_test = mndata.load_testing()

batches_sizes = 10
nb_neurons = 30
alpha = 0.01
tol = 90


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def splitInRandomGroupsOfSize(n):
    indexes = range(0, len(glob_images))
    np.random.shuffle(indexes)
    return np.array([[glob_images[indexes[j]] for j in range(i, i+n)] for i in range(0, len(glob_images), n)]), \
           np.array([[glob_labels[indexes[j]] for j in range(i, i+n)] for i in range(0, len(glob_labels), n)])


ims, labs = splitInRandomGroupsOfSize(batches_sizes)
ims /= 255
glob_images_test = np.array(glob_images_test)
glob_images_test /= 255


def outputFromGroup(group):
    return np.transpose(np.array(group))


def initializeWeights1(nb_neur):
    return np.array([np.random.normal(0, 1./np.sqrt(784), 784) for i in range(0, nb_neur)])


def initializeWeights2(nb_neur):
    return np.array([np.random.normal(0, 1./np.sqrt(nb_neur), nb_neur) for i in range(0, 10)])


def newWeights(w, learn_rate, batch_size, delta, al):
    return w - learn_rate/batch_size * np.dot(delta, np.transpose(al))


def newBiais(b, learn_rate, batch_size, delta):
    inter = np.dot(delta, np.ones(batch_size))
    return b - learn_rate/batch_size * inter


def calculateZl(wl, a, b, batch_size):
    inter = np.dot(wl, a)
    new_b = np.transpose([b for i in range(0, batch_size)])
    return inter + new_b


def backpropag(w, delta, z):
    return np.dot(np.transpose(w), delta)*sigmoid(z)*(1-sigmoid(z))


def expectedResultFromLabels(group_labels, batches_sizes):
    r = np.zeros((10, batches_sizes))
    for i, b in enumerate(group_labels):
        r[i][b] += 1
    return r


def calculateSuccessRate(w1, w2):
    w = np.dot(w2, w1)
    suc = 0.
    for i, img in enumerate(glob_images_test):
        a = np.dot(w, img)
        if (np.argmax(a) == glob_labels_test[i]):
            suc += 1
    return suc/10000*100

# init
wl1 = initializeWeights1(nb_neurons)
wl2 = initializeWeights2(nb_neurons)

continuer = True
ep = 0
print "Starting Calculation. Stopping when success_rate >= " + str(tol) + "%"
while continuer:
    ep += 1
    print "Epoch " + str(ep)
    nb_success = 0.
    for i in range(1, ims.shape[0]):

        al0 = outputFromGroup(ims[i])
        b1 = np.zeros(nb_neurons)
        b2 = np.zeros(10)

        # main loop
        zl1 = calculateZl(wl1, al0, b1, batches_sizes)
        al1 = sigmoid(zl1)
        zl2 = calculateZl(wl2, al1, b2, batches_sizes)
        al2 = sigmoid(zl2)

        deltaL2 = al2 - expectedResultFromLabels(labs[0], batches_sizes)
        deltaL1 = backpropag(wl2, deltaL2, zl1)

        wl2 = newWeights(wl2, alpha, batches_sizes, deltaL2, al1)
        wl1 = newWeights(wl1, alpha, batches_sizes, deltaL1, al0)
	print al0.shape
        b2 = newBiais(b2, alpha, batches_sizes, deltaL2)
        b1 = newBiais(b1, alpha, batches_sizes, deltaL1)

    suc = calculateSuccessRate(wl1, wl2)
    print "\tsuccess_rate : " + str(np.round(suc, 2)) + "%"
    continuer = suc < tol

