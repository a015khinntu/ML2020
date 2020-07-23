from sklearn.decomposition import PCA
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def recon_loss(x, y):
    return np.mean(np.sqrt(np.mean(np.power(x - y, 2), axis=0)))

def main(train_x, train_y, val_x, val_y, dim=500):
    # x_train = train_x.reshape(train_x.shape[0], -1)
    # x_val = val_x.reshape(val_x.shape[0], -1)
    # flatten_pca = PCA(n_components=dim * 3, whiten=True).fit(x_train)
    # train_feat = flatten_pca.transform(x_train)
    # train_rec = flatten_pca.inverse_transform(train_feat)
    # print(recon_loss(x_train, train_rec))
    x_train = train_x.reshape(train_x.shape[0], -1, 3)
    x_val = val_x.reshape(val_x.shape[0], -1, 3)
    # print(x_train[:, :, 0].shape)
    # x_train = np.transpose(x_train, axis=[0, 2, 1])
    r_pca = PCA(n_components=dim, whiten=True).fit(x_train[:, :, 0])
    g_pca = PCA(n_components=dim, whiten=True).fit(x_train[:, :, 1])
    b_pca = PCA(n_components=dim, whiten=True).fit(x_train[:, :, 2])
    r_feat = r_pca.transform(x_train[:, :, 0])
    g_feat = g_pca.transform(x_train[:, :, 1])
    b_feat = b_pca.transform(x_train[:, :, 2])
    r_rec = r_pca.inverse_transform(r_feat)
    g_rec = g_pca.inverse_transform(g_feat)
    b_rec = b_pca.inverse_transform(b_feat)
    train_rec = np.zeros_like(x_train)
    train_rec[:, :, 0] = r_rec
    train_rec[:, :, 1] = g_rec
    train_rec[:, :, 2] = b_rec
    print(recon_loss(x_train, train_rec))
    train_feat = np.concatenate((r_feat, g_feat, b_feat), axis=1)
    r_feat = r_pca.transform(x_val[:, :, 0])
    g_feat = r_pca.transform(x_val[:, :, 1])
    b_feat = r_pca.transform(x_val[:, :, 2])
    val_feat = np.concatenate((r_feat, g_feat, b_feat), axis=1)
    # print(train_feat.shape)
    model = SVC().fit(train_feat, train_y)
    train_acc = accuracy_score(train_y, model.predict(train_feat))
    val_acc = accuracy_score(val_y, model.predict(val_feat))
    print('Train acc: {}, Val acc: {}'.format(train_acc, val_acc))

if __name__ == '__main__':
    train_x, train_y, val_x, val_y = pickle.load(open('data/data.pkl', 'rb'))
    main(train_x, train_y, val_x, val_y)
