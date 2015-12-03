import numpy as np
import os
import gzip
import pickle


def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h


#def load_data(dataset):
def mnist(ntrain=60000,ntest=10000,onehot=True):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''
    path = "data-mnist/mnist.pkl.gz"
    f = gzip.open(path, 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is an numpy.ndarray of 2 dimensions (a matrix)
    # which row's correspond to an example. target is a
    # numpy.ndarray of 1 dimensions (vector)) that have the same length as
    # the number of rows in the input. It should give the target
    # target to the example with the same index in the input.
    print("Done.")
    #return (train_set, valid_set, test_set)
    if onehot:
        trY = one_hot(train_set[1], 10)
        teY = one_hot(valid_set[1], 10)
    else:
        trY = np.asarray(train_set[1])
        teY = np.asarray(valid_set[1])
    trX = train_set[0]
    teX = valid_set[0]
    print(trX.shape)
    trX = trX.reshape(trX.shape[0], 1, 28, 28)
    teX = teX.reshape(teX.shape[0], 1, 28, 28)
    return trX, trY, teX, teY, 1, 28




import scipy
import theano
def caltech(onehot=True, num_imgs_to_load=1000):
    path = "data-caltech/101_ObjectCategories/"

    # load file names
    fnames = []
    cats = {}
    for path, subdirs, files in os.walk(path):
        if len(files) > 0:
            cats[path] = []
            #print(path)
        for name in files:
            fnames.append(os.path.join(path, name))
            cats[path].append(os.path.join(path, name))


    # load images
    i=0
    for k in cats.keys():
        cats[i] = cats[k]
        del cats[k]
        i += 1
    #print(cats.keys())
    
    pics = []
    imgshape = []
    #rand = np.array(fnames)
    count = 0
    # for img in fnames:
    #     if count >= num_imgs_to_load:
    #         break
    #     count += 1
    #     pic = scipy.misc.imread(img)
    #     if len(pic.shape)==3:
    #         if pic.shape[0] >= 220 and pic.shape[1] >= 220:
    #             pic = pic[20:20+200, 20:20+200, :]
    #             pic = scipy.misc.imresize(pic, size=(60,60))
    #             imgshape = pic.shape
    #             pics.append(pic)
    #             labels.append()
    
    labels = []
    for label, vals in cats.items():
        for val in vals:
            if count >= num_imgs_to_load:
                break
            count += 1
            pic = scipy.misc.imread(val)
            if len(pic.shape)==3:
                if pic.shape[0] >= 220 and pic.shape[1] >= 220:
                    pic = pic[20:20+200, 20:20+200, :]
                    pic = scipy.misc.imresize(pic, size=(60,60))
                    imgshape = pic.shape
                    pics.append(pic)
                    labels.append(label)
    
        
    X = np.array(pics)
    Y = np.array(labels)
    print(X.shape)
    print(Y.shape)
    if onehot:
        Y = one_hot(Y, i)
    print(Y.shape)
    return X, Y

# data will be reshaped as (num_imgs, num_channels, height, width)
# in this case:            (50000, 3, 32, 32)
def cifar10(onehot=True):
    path = "data-cifar10"
    X = np.zeros((50000, 3072))
    Y = np.zeros((50000, 1)).astype(int)
    for i in range(1,6):
        f = open(os.path.join(path, "data_batch_" + str(i)), 'rb')
        dict = pickle.load(f, encoding="latin1")
        X[(i-1) * 10000: i * 10000,:] = dict['data']
        Y[(i-1) * 10000: i * 10000,:] = np.array(dict['labels']).reshape((10000,1)).astype(int)
    print(Y.dtype)
    Y = one_hot(Y, 10)
    print(Y.shape)
    print(Y[:10, :])
    X = X.reshape(50000, 3, 32, 32)

    print("test")
    f = open(os.path.join(path, "test_batch"), 'rb')
    dict = pickle.load(f, encoding="latin1")
    X_test = dict['data'].reshape(10000, 3, 32, 32)
    Y_test = np.array(dict['labels']).reshape((10000,1)).astype(int)
    print(X_test.shape)
    print(Y_test.shape)
    Y_test = one_hot(Y_test, 10)
    return X, Y, X_test, Y_test, 3, 32


def load_data(dataset):
    if dataset == "mnist":
        return mnist()
    if dataset == "cifar10":
        return cifar10()
