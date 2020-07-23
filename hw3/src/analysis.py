from util import readfile
import collections
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.manifold import TSNE

eleven_colors = ['black', 'brown', 'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'grey', 'gold', 'silver']

def label_dist(train_y, val_y):
    ctr = collections.Counter(train_y)
    cval = collections.Counter(val_y)
    tr_dist = np.zeros(11)
    for key, value in sorted(ctr.items()):
        tr_dist[key] = value
    val_dist = np.zeros(11)
    for key, value in sorted(cval.items()):
        val_dist[key] = value
    width = 0.4
    x_tr = np.arange(11) - width / 2
    x_val = np.arange(11) + width / 2
    plt.bar(x_tr, tr_dist / np.sum(tr_dist), width=width, label='Training Set Dist.')
    plt.bar(x_val, val_dist / np.sum(val_dist), width=width, label='Validation Set Dist.')
    plt.legend()
    plt.savefig('graphic/label_dist.jpg')

def mae(x, y):
    return np.mean(np.abs(x - y)) / len(x)

def mse(x, y):
    return np.mean(np.power(x - y, 2)) / len(x)

def cos_sim(x, y):
    return np.sum(x * y) / np.sqrt(np.sum(x ** 2)) / np.sqrt(np.sum(y ** 2))

def mahanttan_distance(x, y):
    return np.sum(np.abs(x - y))

def distance(x, y):
    return np.sum(np.power(x - y, 2))

def distance_dist(train_x, val_x, n_cols = 40):
    t = train_x.reshape(train_x.shape[0], -1)
    v = val_x.reshape(val_x.shape[0], -1)
    mean_t = np.mean(t, axis=0)
    mean_v = np.mean(v, axis=0)
    t_l2 = np.array([distance(mean_t, img) for img in t])
    v_l2 = np.array([distance(mean_v, img) for img in v])
    # t_bucket = np.zeros(1001)
    tmax = np.max(t_l2)
    vmax = np.max(v_l2)
    max_dist = np.max((tmax, vmax))
    d_t =  np.floor(t_l2 / max_dist * (n_cols - 1)).astype(int)
    d_t = collections.Counter(d_t)
    t_dist = np.zeros(n_cols)
    for key, value in d_t.items():
        t_dist[key] = value
    t_dist /= np.sum(t_dist)
    d_v = np.floor(v_l2 / max_dist * (n_cols - 1)).astype(int)
    d_v = collections.Counter(d_v)
    v_dist = np.zeros(n_cols)
    for key, value in d_v.items():
        v_dist[key] = value
    v_dist /= np.sum(v_dist)
    width = 0.4
    plt.bar(np.arange(n_cols), t_dist, width=width, label='Train Image Distance dist.')
    # plt.bar(np.arange(40) + width / 2, v_dist, width=width, label='Val Image Distance dist.')
    plt.legend()
    plt.savefig('graphic/train_image_dist.jpg')
    plt.close('all')
    plt.bar(np.arange(n_cols) + width / 2, v_dist, width=width, label='Val Image Distance dist.')
    plt.legend()
    plt.savefig('graphic/val_image_dist.jpg')
    plt.close('all')
    plt.bar(np.arange(n_cols) - width / 2, t_dist, width=width, label='Train Image Distance dist.')
    plt.bar(np.arange(n_cols) + width / 2, v_dist, width=width, label='Val Image Distance dist.')
    plt.legend()
    plt.savefig('graphic/image_dist.jpg')
    plt.close('all')

def dist_over_distance(train_x, train_y, val_x, val_y, n_cols=20, norm=True):
    t = train_x.reshape(train_x.shape[0], -1)
    v = val_x.reshape(val_x.shape[0], -1)
    mean_t = np.mean(t, axis=0)
    mean_v = np.mean(v, axis=0)
    t_l2 = np.array([distance(mean_t, img) for img in t])
    v_l2 = np.array([distance(mean_v, img) for img in v])
    tmax = np.max(t_l2)
    vmax = np.max(v_l2)
    max_dist = np.max((tmax, vmax))
    discrete_t =  np.floor(t_l2 / max_dist * 19).astype(int)
    t_label = np.zeros((20, 11))
    for dist, label in zip(discrete_t, train_y):
        t_label[dist][label] += 1
    if norm:
        for i in range(20):
            if np.sum(t_label[i]) > 0:
                t_label[i] = t_label[i] / np.sum(t_label[i])
    else:
        t_label /= len(train_x)
    
    discrete_v = np.floor(v_l2 / max_dist * 19).astype(int)
    v_label = np.zeros((20, 11))
    for dist, label in zip(discrete_v, val_y):
        v_label[dist][label] += 1
    if norm:
        for i in range(20):
            if np.sum(v_label[i]) > 0:
                v_label[i] = v_label[i] / np.sum(v_label[i])
    else:
        v_label /= len(val_x)

    width = 0.4
    plt.bar(np.arange(20), t_label[:, 0], label='{}'.format(0))
    accu = t_label[:, 0]
    for i in range(10):
        plt.bar(np.arange(20), t_label[:, i + 1], bottom=accu, label='{}'.format(i + 1))
        accu += t_label[:, i + 1]
    plt.legend()
    plt.xticks(np.linspace(0, 20, 11, dtype=np.int))
    if norm:
        plt.savefig('graphic/train_dist_over_distance_norm.jpg')
    else:
        plt.savefig('graphic/train_dist_over_distance.jpg')
    plt.close('all')
    plt.bar(np.arange(20), v_label[:, 0], label='{}'.format(0))
    accu = v_label[:, 0]
    for i in range(10):
        plt.bar(np.arange(20), v_label[:, i + 1], bottom=accu, label='{}'.format(i + 1))
        accu += v_label[:, i + 1]
    plt.legend()
    plt.xticks(np.linspace(0, 20, 11, dtype=np.int))
    if norm:
        plt.savefig('graphic/val_dist_over_distance_norm.jpg')
    else:
        plt.savefig('graphic/val_dist_over_distance.jpg')
    plt.close('all')

def visualize_tsne(train_x, train_y, val_x, val_y):
    train_feat = TSNE(n_jobs=8, verbose=1).fit_transform(train_x.reshape(train_x.shape[0], -1))
    val_feat = TSNE(n_jobs=8, verbose=1).fit_transform(val_x.reshape(val_x.shape[0], -1))
    train_bucket = [[] for i in range(11)]
    val_bucket = [[] for i in range(11)]
    for pos, label in zip(train_feat, train_y):
        train_bucket[label].append(pos)
    for pos, label in zip(val_feat, val_y):
        val_bucket[label].append(pos)
    for i in range(11):
        positions = np.array(train_bucket[i])
        plt.scatter(positions[:, 0], positions[:, 1], color=eleven_colors[i], label='{}'.format(i))
    plt.legend()
    plt.savefig('graphic/train-tsne.jpg')
    plt.close('all')
    for i in range(11):
        positions = np.array(val_bucket[i])
        plt.scatter(positions[:, 0], positions[:, 1], color=eleven_colors[i], label='{}'.format(i))
    plt.legend()
    plt.savefig('graphic/val-tsne.jpg')
    plt.close('all')

if __name__ == '__main__':
    train_x, train_y, val_x, val_y = pickle.load(open('data/data.pkl', 'rb'))
    visualize_tsne(train_x, train_y, val_x, val_y)


