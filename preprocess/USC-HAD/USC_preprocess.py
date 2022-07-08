import numpy as np
import scipy.io as scio
import torch
import os


def z_score(data, tensor_mean, tensor_std):
    result = (data - tensor_mean)/tensor_std
    return result


def window_slide(List, window_size, stride):
    if type(List) != list:
        List = List.permute(1, 0).tolist()
    hight = len(List)
    width = len(List[0])
    times = (width - window_size) // stride + 1
    result = []
    for i in range(hight):
        temp = []
        for j in range(times):
            temp.append(List[i][j*stride: j*stride+window_size])
        result.append(temp)
    return torch.tensor(result).permute(1, 2, 0)


def split_data(data, ratio=(8, 2), if_save=False):
    train_ratio = ratio[0] / sum(ratio)
    X_train, X_test, Y_train, Y_test = torch.tensor([]), torch.tensor([]), [], []
    for i, each_tensor in enumerate(data):
        train_part = int(each_tensor.size(0)*train_ratio)
        np.random.seed(1)
        array_index = np.random.permutation(each_tensor)
        # print("array_index", array_index)
        array = torch.from_numpy(array_index)
        X_train = torch.cat((X_train, array[:train_part]), 0)
        X_test = torch.cat((X_test, array[train_part:]), 0)
        Y_train += [i for j in range(train_part)]
        Y_test += [i for j in range(each_tensor.size(0) - train_part)]
    print(X_train.size(), X_test.size(), torch.tensor(Y_train).size(), torch.tensor(Y_test).size())
    if if_save:
        np.save('./output/x_train', np.array(X_train, dtype=np.float32))
        np.save('./output/x_test', np.array(X_test, dtype=np.float32))
        np.save('./output/y_train', np.array(Y_train, dtype=np.int64))
        np.save('./output/y_test', np.array(Y_test, dtype=np.int64))


def start_processing(WINDOW_SIZE=256, STRIDE=256, IF_ZSCORE=False, IF_SAVE=False):
    subject_list = os.listdir('USC-HAD')
    os.chdir('USC-HAD')
    result = []
    for i in range(12):
        result.append(torch.tensor([]))
    for each_subject in subject_list:
        if os.path.isdir(each_subject):
            # file_list = sorted(os.listdir(each_subject))
            file_list = ['a1t1.mat', 'a1t2.mat', 'a1t3.mat', 'a1t4.mat', 'a1t5.mat',
                         'a2t1.mat', 'a2t2.mat', 'a2t3.mat', 'a2t4.mat', 'a2t5.mat',
                         'a3t1.mat', 'a3t2.mat', 'a3t3.mat', 'a3t4.mat', 'a3t5.mat',
                         'a4t1.mat', 'a4t2.mat', 'a4t3.mat', 'a4t4.mat', 'a4t5.mat',
                         'a5t1.mat', 'a5t2.mat', 'a5t3.mat', 'a5t4.mat', 'a5t5.mat',
                         'a6t1.mat', 'a6t2.mat', 'a6t3.mat', 'a6t4.mat', 'a6t5.mat',
                         'a7t1.mat', 'a7t2.mat', 'a7t3.mat', 'a7t4.mat', 'a7t5.mat',
                         'a8t1.mat', 'a8t2.mat', 'a8t3.mat', 'a8t4.mat', 'a8t5.mat',
                         'a9t1.mat', 'a9t2.mat', 'a9t3.mat', 'a9t4.mat', 'a9t5.mat',
                         'a10t1.mat', 'a10t2.mat', 'a10t3.mat', 'a10t4.mat', 'a10t5.mat',
                         'a11t1.mat', 'a11t2.mat', 'a11t3.mat', 'a11t4.mat', 'a11t5.mat',
                         'a12t1.mat', 'a12t2.mat', 'a12t3.mat', 'a12t4.mat', 'a12t5.mat',]

            os.chdir(each_subject)
            for i, each_mat in enumerate(file_list):
                seq = i // 5
                data = scio.loadmat(each_mat)
                new_tensor = torch.tensor(data['sensor_readings'])
                result[seq] = torch.cat((result[seq], new_tensor), 0)
            os.chdir('../')
    os.chdir('../')
    if IF_ZSCORE:
        all_tensor = torch.tensor([])
        for i in range(len(result)):
            all_tensor = torch.cat((all_tensor, result[i]), 0)
        all_tensor_mean = np.array(torch.mean(all_tensor, 0, keepdim=True))
        all_tensor_std = np.array(torch.std(all_tensor, 0, keepdim=True))
        for i in range(len(result)):
            result[i] = window_slide(z_score(result[i], all_tensor_mean, all_tensor_std), window_size=WINDOW_SIZE, stride=STRIDE)
    else:
        for i in range(len(result)):
            result[i] = window_slide(result[i], window_size=WINDOW_SIZE, stride=STRIDE)
    split_data(result, if_save=IF_SAVE)


if __name__ == "__main__":
    start_processing(WINDOW_SIZE=512, STRIDE=256, IF_ZSCORE=True, IF_SAVE=True)


