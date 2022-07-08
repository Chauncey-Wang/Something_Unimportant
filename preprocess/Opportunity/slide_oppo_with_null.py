from utils import *


n_channels = 113 # number of sensor channels
# window_size = 40 # Sliding window length
window_size = 30 # Sliding window length
# stride = 20  # Sliding window step
stride = 15  # Sliding window step


def load_data2array(name,len_seq,stride):
    Xs = np.empty(shape=[0, window_size, 113])
    ys = np.empty(shape=[0])
    #  Use glob module and wildcard to build a list of files to load from data directory
    path = "data/{}_data_*".format(name)
    data = glob.glob(path)

    for file in data:
        X, y = load_dataset(file)  # X是'numpy.ndarray'  (27825, 113)的strides是(4, 111300)即原来的距离乘4
        X, y = slide(X, y, len_seq, stride, save=False)
        print("X.shape", X.shape)
        # print(X[0])
        print("y.shape", y.shape)
        Xs = np.vstack((Xs, X))
        # print(Xs[0])
        print("Xs.shape", Xs.shape)
        ys = np.hstack((ys, y))
        print("ys.shape", ys.shape)
    return Xs, ys


if __name__ == '__main__':
    print('==HYPERPARAMETERS==')

    X_train,y_train = load_data2array('train', window_size, stride)  # X_train是列表，但里面的元素是array
    print(y_train.shape)
    X_valid, y_valid = load_data2array('val', window_size, stride)  # X_train是列表，但里面的元素是array
    print(y_valid.shape)
    X_test, y_test = load_data2array('test', window_size, stride)  # X_train是列表，但里面的元素是array
    print(y_test.shape)

    print(X_train.shape)
    print(y_train.shape)
    print(X_valid.shape)
    print(y_valid.shape)
    print(X_test.shape)
    print(y_test.shape)

    np.save("./x_train.npy", X_train)
    np.save("./y_train.npy", y_train)
    np.save("./x_valid.npy", X_valid)
    np.save("./y_valid.npy", y_valid)
    np.save("./x_test.npy", X_test)
    np.save("./y_test.npy", y_test)







