import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from scipy import stats


def read_data(file_path):
    column_names = ['user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    data = pd.read_csv(file_path, header=None, names=column_names)
    return data


def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma


dataset = read_data('./WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt')
dataset.dropna(axis=0, how='any', inplace=True)
dataset['x-axis'] = feature_normalize(dataset['x-axis'])
dataset['y-axis'] = feature_normalize(dataset['y-axis'])
dataset['z-axis'] = pd.to_numeric(dataset['z-axis'].str.replace(';', ''))  
dataset['z-axis'] = feature_normalize(dataset['z-axis'])


def windows(data, size):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (size / 2)


def segment_signal(data, window_size=90):
    segments = np.empty((0, window_size, 3))
    labels = np.empty((0))
    for (start, end) in windows(data["timestamp"], window_size):
        x = data["x-axis"][start:end]
        y = data["y-axis"][start:end]
        z = data["z-axis"][start:end]
        if len(dataset["timestamp"][start:end]) == window_size:
            segments = np.vstack([segments, np.dstack([x, y, z])])
            labels = np.append(labels, stats.mode(data["activity"][start:end])[0][0])
    return segments, labels


segments, labels = segment_signal(dataset)
labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)  # one-hot labels
# labels = np.argmax(labels, axis=-1)  # one-hot to number
reshaped_segments = segments.reshape(len(segments), 1, 90, 3)  # len(segments)=24403
# reshaped_segments的数据类型<class 'numpy.ndarray'>  结构(24403, 1, 90, 3)

train_test_split = np.random.rand(len(reshaped_segments)) < 0.70
train_x = reshaped_segments[train_test_split]
train_y = labels[train_test_split]
test_x = reshaped_segments[~train_test_split]
test_y = labels[~train_test_split]

np.save('./WISDM/train_x.npy', train_x)
np.save('./WISDM/train_y.npy', train_y)
np.save('./WISDM/test_x.npy', test_x)
np.save('./WISDM/test_y.npy', test_y)

