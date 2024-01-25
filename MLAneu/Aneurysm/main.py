import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils import data
from data_process import fd_pass_process, aneurist_process
from models import LogisticRegression, SVM
from sklearn import svm, linear_model
import joblib


def get_fd_train_data(seed=42):
    '''
    get and save FD-PASS training set
    '''
    process_flag = os.path.exists(".\processed\\fd\\train_features.csv")
    if not process_flag:
        train_features, valid_features, test_features, train_nlabel, train_hlabel, \
            valid_nlabel, valid_hlabel, test_nlabel, test_hlabel = fd_pass_process(seed)
        np.savetxt(".\processed\\fd\\train_features.csv", train_features, delimiter=",", fmt='%s')
        np.savetxt(".\processed\\fd\\valid_features.csv", valid_features, delimiter=",", fmt='%s')
        np.savetxt(".\processed\\fd\\test_features.csv", test_features, delimiter=",", fmt='%s')
        np.savetxt(".\processed\\fd\\train_nlabel.csv", train_nlabel, delimiter=",", fmt='%s')
        np.savetxt(".\processed\\fd\\train_hlabel.csv", train_hlabel, delimiter=",", fmt='%s')
        np.savetxt(".\processed\\fd\\valid_nlabel.csv", valid_nlabel, delimiter=",", fmt='%s')
        np.savetxt(".\processed\\fd\\valid_hlabel.csv", valid_hlabel, delimiter=",", fmt='%s')
        np.savetxt(".\processed\\fd\\test_nlabel.csv", test_nlabel, delimiter=",", fmt='%s')
        np.savetxt(".\processed\\fd\\test_hlabel.csv", test_hlabel, delimiter=",", fmt='%s')
    else:
        train_features = pd.read_csv(".\processed\\fd\\train_features.csv", header=None).values
        valid_features = pd.read_csv(".\processed\\fd\\valid_features.csv", header=None).values
        train_nlabel = pd.read_csv(".\processed\\fd\\train_nlabel.csv", header=None).values
        train_hlabel = pd.read_csv(".\processed\\fd\\train_hlabel.csv", header=None).values
        valid_nlabel = pd.read_csv(".\processed\\fd\\valid_nlabel.csv", header=None).values
        valid_hlabel = pd.read_csv(".\processed\\fd\\valid_hlabel.csv", header=None).values
    return train_features[:, 1:], valid_features[:, 1:], train_nlabel, train_hlabel, valid_nlabel, valid_hlabel


def get_fd_test_data():
    '''
    get and save FD-PASS test set
    '''
    test_features = pd.read_csv(".\processed\\fd\\test_features.csv", header=None).values
    test_nlabel = pd.read_csv(".\processed\\fd\\test_nlabel.csv", header=None).values
    test_hlabel = pd.read_csv(".\processed\\fd\\test_hlabel.csv", header=None).values
    return test_features[:, 1:], test_nlabel, test_hlabel


def get_aneu_train_data(seed=42):
    '''
    get and save AneurIST training set
    '''
    process_flag = os.path.exists(".\processed\AneurIST\\train_features.csv")
    if not process_flag:
        train_features, valid_features, test_features, train_label, valid_label, test_label = aneurist_process(seed)
        np.savetxt(".\processed\AneurIST\\train_features.csv", train_features, delimiter=",", fmt='%s')
        np.savetxt(".\processed\AneurIST\\valid_features.csv", valid_features, delimiter=",", fmt='%s')
        np.savetxt(".\processed\AneurIST\\test_features.csv", test_features, delimiter=",", fmt='%s')
        np.savetxt(".\processed\AneurIST\\train_label.csv", train_label, delimiter=",", fmt='%s')
        np.savetxt(".\processed\AneurIST\\valid_label.csv", valid_label, delimiter=",", fmt='%s')
        np.savetxt(".\processed\AneurIST\\test_label.csv", test_label, delimiter=",", fmt='%s')
    else:
        train_features = pd.read_csv(".\processed\AneurIST\\train_features.csv", header=None).values
        valid_features = pd.read_csv(".\processed\AneurIST\\valid_features.csv", header=None).values
        train_label = pd.read_csv(".\processed\AneurIST\\train_label.csv", header=None).values
        valid_label = pd.read_csv(".\processed\AneurIST\\valid_label.csv", header=None).values
    return train_features[:, 1:], valid_features[:, 1:], train_label, valid_label


def get_aneu_test_data():
    '''
    get and save AneurIST test set
    '''
    test_features = pd.read_csv(".\processed\AneurIST\\test_features.csv", header=None).values
    test_label = pd.read_csv(".\processed\AneurIST\\test_label.csv", header=None).values
    return test_features[:, 1:], test_label


def get_acc(out, labels):
    '''
    Calculate accuracy
    '''
    out = (out > 0.5) + 0
    res = (out == labels) + 0
    return res.sum() / res.shape[0]


def load_array(data_arrays, batch_size):
    '''
    process dataloader for training and test
    '''
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size)


def train_lr(train_features, valid_features, train_nlabel, train_hlabel, valid_nlabel, valid_hlabel,
             lr=0.01, epochs=10, bs=2, hyper_tension=False):
    '''
    train logisticRegression model
    '''
    LR = LogisticRegression(train_features.shape[1])
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(LR.parameters(), lr)
    input_data = torch.from_numpy(train_features.astype(np.float32))
    if hyper_tension:
        labels = torch.from_numpy(train_hlabel.astype(np.float32))
    else:
        labels = torch.from_numpy(train_nlabel.astype(np.float32))
    data_iter = load_array((input_data, labels), bs)

    for epoch in range(epochs):
        for X, y in data_iter:
            out = LR(X)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        out = LR(input_data)
        loss = loss_fn(out, labels)
        acc = get_acc(out, labels)
        print("\tepoch:" + str(epoch) + "\n\tloss: " + str(loss) + "\n\tacc: " + str(acc))

    valid_data = torch.from_numpy(valid_features.astype(np.float32))
    if hyper_tension:
        v_labels = torch.from_numpy(valid_hlabel.astype(np.float32))
    else:
        v_labels = torch.from_numpy(valid_nlabel.astype(np.float32))
    v_out = LR(valid_data)
    loss = loss_fn(v_out, v_labels)
    acc = get_acc(v_out, v_labels)
    print("\tvalid:" + "\tloss: " + str(loss) + "\n\tacc: " + str(acc))
    if hyper_tension:
        torch.save(LR, ".\\trained\\fd\lrh.pth")
    elif train_hlabel is None:
        torch.save(LR, ".\\trained\AneurIST\lr.pth")
    else:
        torch.save(LR, ".\\trained\\fd\lrn.pth")


def test_lr(test_features, test_nlabel, test_hlabel, hyper_tension=False):
    '''
    test logisticRegression model
    '''
    if hyper_tension:
        LR = torch.load(".\\trained\\fd\lrh.pth")
    elif test_hlabel is None:
        LR = torch.load(".\\trained\AneurIST\lr.pth")
    else:
        LR = torch.load(".\\trained\\fd\lrn.pth")
    test_data = torch.from_numpy(test_features.astype(np.float32))
    if hyper_tension:
        test_labels = torch.from_numpy(test_hlabel.astype(np.float32))
    else:
        test_labels = torch.from_numpy(test_nlabel.astype(np.float32))
    test_out = LR(test_data)
    loss_fn = nn.BCELoss()
    loss = loss_fn(test_out, test_labels)
    acc = get_acc(test_out, test_labels)
    if hyper_tension:
        print("\ttest lrh fd:\n" + "\tloss: " + str(loss) + "\n\tacc: " + str(acc) + "\n")
    elif test_hlabel is None:
        print("\ttest lr aneu:\n" + "\tloss: " + str(loss) + "\n\tacc: " + str(acc) + "\n")
    else:
        print("\ttest lrn fd:\n" + "\tloss: " + str(loss) + "\n\tacc: " + str(acc) + "\n")


def train_svm(train_features, valid_features, train_nlabel, train_hlabel, valid_nlabel, valid_hlabel,
             lr=0.01, epochs=10, bs=2, hyper_tension=False):
    '''
    train SVM model
    '''
    svm = SVM(train_features.shape[1])
    loss_fn = nn.HingeEmbeddingLoss()
    optimizer = torch.optim.Adam(svm.parameters(), lr)
    input_data = torch.from_numpy(train_features.astype(np.float32))
    if hyper_tension:
        labels = torch.from_numpy(train_hlabel.astype(np.float32))
    else:
        labels = torch.from_numpy(train_nlabel.astype(np.float32))
    data_iter = load_array((input_data, labels), bs)

    for epoch in range(epochs):
        for X, y in data_iter:
            out = svm(X)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        out = torch.sign(svm(input_data))
        loss = loss_fn(out, labels)
        acc = get_acc(out, labels)
        print("\tepoch:" + str(epoch) + "\n\tloss: " + str(loss) + "\n\tacc: " + str(acc))

    valid_data = torch.from_numpy(valid_features.astype(np.float32))
    if hyper_tension:
        v_labels = torch.from_numpy(valid_hlabel.astype(np.float32))
    else:
        v_labels = torch.from_numpy(valid_nlabel.astype(np.float32))
    v_out = svm(valid_data)
    loss = loss_fn(v_out, v_labels)
    acc = get_acc(v_out, v_labels)
    print("\tvalid:" + "\tloss: " + str(loss) + "\n\tacc: " + str(acc))
    if hyper_tension:
        torch.save(svm, ".\\trained\\fd\svmh.pth")
    elif train_hlabel is None:
        torch.save(svm, ".\\trained\AneurIST\svm.pth")
    else:
        torch.save(svm, ".\\trained\\fd\svmn.pth")


def test_svm(test_features, test_nlabel, test_hlabel, hyper_tension=False):
    '''
    test SVM model
    '''
    if hyper_tension:
        svm = torch.load(".\\trained\\fd\svmh.pth")
    elif test_hlabel is None:
        svm = torch.load(".\\trained\AneurIST\svm.pth")
    else:
        svm = torch.load(".\\trained\\fd\svmn.pth")
    test_data = torch.from_numpy(test_features.astype(np.float32))
    if hyper_tension:
        test_labels = torch.from_numpy(test_hlabel.astype(np.float32))
    else:
        test_labels = torch.from_numpy(test_nlabel.astype(np.float32))
    test_out = svm(test_data)
    loss_fn = nn.HingeEmbeddingLoss()
    loss = loss_fn(test_out, test_labels)
    acc = get_acc(test_out, test_labels)
    if hyper_tension:
        print("\ttest svmh fd:\n" + "\tloss: " + str(loss) + "\n\tacc: " + str(acc) + "\n")
    elif test_hlabel is None:
        print("\ttest svm aneu:\n" + "\tloss: " + str(loss) + "\n\tacc: " + str(acc) + "\n")
    else:
        print("\ttest svmn fd:\n" + "\tloss: " + str(loss) + "\n\tacc: " + str(acc) + "\n")


if __name__ == '__main__':
    # process FD-PASS data, get training and test set
    train_features, valid_features, train_nlabel, train_hlabel, valid_nlabel, valid_hlabel = get_fd_train_data()
    test_features, test_nlabel, test_hlabel = get_fd_test_data()
    # train LR and SVM
    if not os.path.exists(".\\trained\\fd\lrn.pth"):
        train_lr(train_features, valid_features, train_nlabel, train_hlabel, valid_nlabel, valid_hlabel,
                 lr=0.009, epochs=30, bs=4, hyper_tension=False)
    if not os.path.exists(".\\trained\\fd\lrh.pth"):
        train_lr(train_features, valid_features, train_nlabel, train_hlabel, valid_nlabel, valid_hlabel,
                 lr=0.009, epochs=30, bs=4, hyper_tension=True)
    if not os.path.exists(".\\trained\\fd\svmn.pth"):
        train_svm(train_features, valid_features, train_nlabel, train_hlabel, valid_nlabel, valid_hlabel,
                  lr=0.009, epochs=50, bs=4, hyper_tension=False)
    if not os.path.exists(".\\trained\\fd\svmh.pth"):
        train_svm(train_features, valid_features, train_nlabel, train_hlabel, valid_nlabel, valid_hlabel,
                  lr=0.009, epochs=50, bs=4, hyper_tension=True)
    # test LR and SVM
    test_lr(test_features, test_nlabel, test_hlabel, hyper_tension=False)
    test_lr(test_features, test_nlabel, test_hlabel, hyper_tension=True)
    test_svm(test_features, test_nlabel, test_hlabel, hyper_tension=False)
    test_svm(test_features, test_nlabel, test_hlabel, hyper_tension=True)

    # train SVM(rbf kernel)
    if not os.path.exists(".\\trained\\fd\sk_svmn.pkl"):
        sk_svm = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
        sk_svm.fit(train_features, train_hlabel.ravel())
        joblib.dump(sk_svm, ".\\trained\\fd\sk_svmn.pkl")
    else:
        sk_svm = joblib.load(".\\trained\\fd\sk_svmn.pkl")
    # test SVM(rbf kernel)
    results = sk_svm.predict(test_features)
    print("\ttest sk_svmn fd:\n" + "\tacc: " + str(get_acc(results, test_nlabel.ravel())) + "\n")
    # train SVM(rbf kernel) hypertension
    if not os.path.exists(".\\trained\\fd\sk_svmh.pkl"):
        sk_svm = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
        sk_svm.fit(train_features, train_hlabel.ravel())
        joblib.dump(sk_svm, ".\\trained\\fd\sk_svmh.pkl")
    else:
        sk_svm = joblib.load(".\\trained\\fd\sk_svmh.pkl")
    # test SVM(rbf kernel) hypertension
    results = sk_svm.predict(test_features)
    print("\ttest sk_svmh fd:\n" + "\tacc: " + str(get_acc(results, test_hlabel.ravel())) + "\n")

    # train lasso
    if not os.path.exists(".\\trained\\fd\lasson.pkl"):
        lasso = linear_model.LassoCV()
        lasso.fit(train_features, train_hlabel.ravel())
        joblib.dump(lasso, ".\\trained\\fd\lasson.pkl")
    else:
        lasso = joblib.load(".\\trained\\fd\lasson.pkl")
    # test lasso
    results = lasso.predict(test_features)
    print("\ttest lasson fd:\n" + "\tacc: " + str(get_acc(results, test_hlabel.ravel())) + "\n")

    # train lasso hypertension
    if not os.path.exists(".\\trained\\fd\lassoh.pkl"):
        lasso = linear_model.LassoCV()
        lasso.fit(train_features, train_hlabel.ravel())
        joblib.dump(lasso, ".\\trained\\fd\lassoh.pkl")
    else:
        lasso = joblib.load(".\\trained\\fd\lassoh.pkl")
    # test lasso hypertension
    results = lasso.predict(test_features)
    print("\ttest lassoh fd:\n" + "\tacc: " + str(get_acc(results, test_hlabel.ravel())) + "\n")

    # process FD-AneurIST data, get training and test set
    train_features, valid_features, train_label, valid_label = get_aneu_train_data()
    test_features, test_label, = get_aneu_test_data()
    # train LR and SVM
    if not os.path.exists(".\\trained\AneurIST\lr.pth"):
        train_lr(train_features, valid_features, train_label, None, valid_label, None,
                 lr=0.009, epochs=100, bs=16, hyper_tension=False)
    if not os.path.exists(".\\trained\AneurIST\svm.pth"):
        train_svm(train_features, valid_features, train_label, None, valid_label, None,
                  lr=0.01, epochs=50, bs=2, hyper_tension=False)
    # test LR and SVM
    test_lr(test_features, test_label, None, hyper_tension=False)
    test_svm(test_features, test_label, None, hyper_tension=False)

    # train SVM(rbf)
    if not os.path.exists(".\\trained\AneurIST\sk_svm.pkl"):
        sk_svm = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
        sk_svm.fit(train_features, train_label.ravel())
        joblib.dump(sk_svm, ".\\trained\AneurIST\sk_svm.pkl")
    else:
        sk_svm = joblib.load(".\\trained\AneurIST\sk_svm.pkl")
    # test SVM(rbf)
    results = sk_svm.predict(test_features)
    print("\ttest sk_svm aneu:\n" + "\tacc: " + str(get_acc(results, test_label.ravel())) + "\n")

    # train lasso
    if not os.path.exists(".\\trained\AneurIST\lassoh.pkl"):
        lasso = linear_model.LassoCV()
        lasso.fit(train_features, train_label.ravel())
        joblib.dump(lasso, ".\\trained\AneurIST\lassoh.pkl")
    else:
        lasso = joblib.load(".\\trained\AneurIST\lassoh.pkl")
    # test lasso
    results = lasso.predict(test_features)
    print("\ttest lassoh fd:\n" + "\tacc: " + str(get_acc(results, test_label.ravel())) + "\n")

