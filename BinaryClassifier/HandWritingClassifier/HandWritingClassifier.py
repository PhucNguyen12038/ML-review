import numpy as np
import mnist
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import accuracy_score

N = 1000
train_imgs = mnist.train_images()[:N]
train_labels = mnist.train_labels()[:N]
Xtrain_all = np.asarray(train_imgs)
ytrain_all = np.array(train_labels.tolist())
dimension = Xtrain_all.shape[1] * Xtrain_all.shape[2]
print(dimension)

test_imgs = mnist.test_images()[:N]
test_labels = mnist.test_labels()[:N]
Xtest_all = np.asarray(test_imgs)
ytest_all = np.array(test_labels.tolist())

cls = [[0], [1]]
img_size = Xtrain_all.shape[1] * Xtrain_all.shape[2]


def extract_data(X, y, classes):
    """
    X: numpy array, matrix of size (N, d), d is data dim
    y: numpy array, size (N, )
    cls: two lists of labels. For example:
        cls = [[1, 4, 7], [5, 6, 8]]
    return:
        X: extracted data
        y: extracted label
            (0 and 1, corresponding to two lists in cls)
    """
    y_res_id = np.array([])
    for i in cls[0]:
        y_res_id = np.hstack((y_res_id, np.where(y == i)[0]))
    n0 = len(y_res_id)

    for i in cls[1]:
        y_res_id = np.hstack((y_res_id, np.where(y == i)[0]))
    n1 = len(y_res_id) - n0

    y_res_id = y_res_id.astype(int)
    X_res = X[y_res_id, :] / 255.0
    y_res = np.asarray([0] * n0 + [1] * n1)
    return (X_res, y_res)


def extract_feature(train_imgs, train_labels, dimension):
    imgs = np.zeros([train_imgs.shape[0], dimension])
    for i in range(train_imgs.shape[0]):
        for j in range(train_imgs.shape[1]):
            for k in range(train_imgs.shape[2]):
                pixel = train_imgs[i][j][k]
                imgs[i][j * k] = pixel / 255.0

    return imgs, train_labels


(trn_imgs, trn_labels) = extract_feature(Xtrain_all, ytrain_all, dimension)
print(trn_imgs.shape)
print(trn_labels.shape)
# extract data for training
# (X_train, y_train) = extract_data(Xtrain_all, ytrain_all, cls)

# extract data for test
# (X_test, y_test) = extract_data(Xtest_all, ytest_all, cls)

# train the logistic regression model
# logreg = linear_model.LogisticRegression(C=1e5) # just a big number
# logreg.fit(trn_imgs, trn_labels)

# predict
# y_pred = logreg.predict(Xtest_all)
# print("Accuracy: %.2f %%" %(100*accuracy_score(ytest_all, y_pred.tolist())))
