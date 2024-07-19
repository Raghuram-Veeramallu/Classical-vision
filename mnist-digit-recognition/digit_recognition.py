import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main


def get_mini_batch(im_train, label_train, batch_size):
    # plt.imshow(im_train[:, 0:5].reshape((14, 14), order='F'), cmap='gray')
    # plt.show()

    np.random.seed(10)

    if im_train.shape[1] != label_train.shape[1]:
        raise ValueError('The sample size of both training images and labels must be same')

    train_size = im_train.shape[1]
    n_labels = len(np.unique(label_train))
    n_batches = train_size//batch_size
    # flag to denote that the last batch will be partially filled
    batches_overflow_flag = train_size%batch_size > 0

    # converting the train labels to onehot vectors
    onehot_encoded_train_labels = np.zeros((n_labels, train_size))
    onehot_encoded_train_labels[label_train.T.flatten(), np.arange(train_size)] = 1

    train_im_indexes = list(np.arange(train_size))
    # shuffling the indexes
    np.random.shuffle(train_im_indexes)

    # splitting the im_train and label_train into batches
    mini_batch_x = [im_train[:, train_im_indexes[(i*batch_size):((i+1)*batch_size)]] for i in range(0, n_batches)]
    mini_batch_y = [onehot_encoded_train_labels[:, train_im_indexes[(i*batch_size):((i+1)*batch_size)]] for i in range(0, n_batches)]

    # if there is an overflow, add the partially filled batch
    if batches_overflow_flag:
        mini_batch_x.append(im_train[:, train_im_indexes[batch_size*n_batches:]])
        mini_batch_y.append(onehot_encoded_train_labels[:, train_im_indexes[batch_size*n_batches:]])

    return mini_batch_x, mini_batch_y


def fc(x, w, b):
    y = np.matmul(w, x) + b
    return y


def fc_backward(dl_dy, x, w, b, y):
    dl_dw =  np.outer(dl_dy.reshape(-1), x).reshape(1, (w.shape[0] * w.shape[1]))
    dl_db = dl_dy
    dl_dx = np.matmul(dl_dy, w)

    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):
    l = math.pow(np.linalg.norm(y - y_tilde), 2)
    dl_dy = - 2 * (y - y_tilde)
    return l, dl_dy


def loss_cross_entropy_softmax(x, y):
    exp_x = np.exp(x)
    y_til = exp_x / np.sum(exp_x)
    l = np.sum(y * np.log(y_til))
    dl_dy =  - 2 * (y - y_til)
    return l, dl_dy


def relu(x):
    # leaky relu
    err = 0.01
    y = np.maximum(err*x, x)
    return y


def relu_backward(dl_dy, x, y):
    # leaky relu
    err = 0.01
    dl_dx = dl_dy * np.where(x > 0, 1, err)
    return dl_dx


def im2col(x, F_h, F_w, stride):
    H, W, C1 = x.shape
    H_bar = (H-F_h) // stride + 1
    W_bar = (W-F_w) // stride + 1
    shp = F_h, F_w, H_bar, W_bar
    s0, s1, _ = x.strides
    strd = s0, s1, s0, s1
    col = np.lib.stride_tricks.as_strided(x, shape=shp, strides=strd).reshape((F_h * F_w * C1, H_bar * W_bar))
    return col


def col2im(x, H, W):
    C2, M = x.shape
    return x.reshape(H, W, C2, order='F')


def conv(x, w_conv, b_conv, stride=1):
    # Image dimensions - H: Height, W: Width, C1: number of channels
    H, W, C1 = x.shape
    # Filter dimensions - F_h: Height, F_w: Width, C1: number of channels, C2: number of filters
    F_h, F_w, C1, C2 = w_conv.shape

    # to match the output shape with input shape, we will have to use a padding of (F_h - 1)/2
    # H_out = 1 + (H_in + 2 * pad - F_h) / stride
    # since the stride = 1, we can find that pad = (F_h - 1)/2 when H_out = H_in
    padding_n = int((F_h - 1)//2)

    # define the output shape (H x W x C2)
    y = np.zeros((H, W, C2))

    # padding the input image only in H and W axis
    padded_x = np.pad(x, pad_width=((padding_n, padding_n), (padding_n, padding_n), (0, 0)), mode='constant', constant_values=0)
    
    im_col = im2col(padded_x, F_h, F_w, stride)
    w_reshaped = w_conv.reshape((F_h * F_w * C1, C2))
    
    mul = np.matmul(w_reshaped.T, im_col)

    y = mul + b_conv

    y = col2im(y, H, W)

    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y, stride=1):
    
    # Image dimensions - H: Height, W: Width, C1: number of channels
    H, W, C1 = x.shape
    # Filter dimensions - F_h: Height, F_w: Width, C1: number of channels, C2: number of filters
    F_h, F_w, C1, C2 = w_conv.shape

    padding_n = int((F_h - 1)//2) 
    
    dl_dw = np.zeros(w_conv.shape)
    dl_db = np.zeros(b_conv.shape)

    padded_x = np.pad(x, pad_width=((padding_n, padding_n), (padding_n, padding_n), (0, 0)), mode='constant', constant_values=0)

    # calculating bias
    dl_dy_reshaped = np.reshape(dl_dy, (C2, -1))
    dl_db = np.sum(dl_dy_reshaped, axis=1).reshape(-1, 1)

    x_im_col = im2col(padded_x, F_h, F_w, stride)
    dl_dw = np.matmul(x_im_col, dl_dy_reshaped.T)

    dl_dw = dl_dw.reshape((F_h, F_w, C1, C2))

    return dl_dw, dl_db


def pool2x2(x):
    # pool window height and width
    H_w = 2
    W_w = 2
    # stide of 2
    window_stride = 2

    H, W, C = x.shape

    H_out = int(1 + (H - H_w) / window_stride)
    W_out = int(1 + (W - W_w) / window_stride)

    # out shape
    y = np.zeros((H_out, W_out, C))

    for k in range(0, C):
        for i in range(0, H_out):
            for j in range(0, W_out):
                y[j, i, k] = np.max(x[window_stride*j:window_stride*(j+1), window_stride*i:window_stride*(i+1), k])

    return y


def pool2x2_backward(dl_dy, x, y):
    H, W, C = x.shape

    H_w = 2
    W_w = 2
    window_stride = 2

    H_out, W_out, C2 = dl_dy.shape

    dl_dx = np.zeros(x.shape)

    for k in range(0, C):
        for i in range(0, H_out):
            for j in range(0, W_out):
                window_view = np.max(x[i*window_stride:i*window_stride+H_w, j*window_stride:j*window_stride+W_w, k])
                mask_on_window = (window_view == np.max(window_view))
                dl_dx[i*window_stride:i*window_stride+H_w, j*window_stride:j*window_stride+W_w, k] = mask_on_window * dl_dy[i, j, k]

    return dl_dx


def flattening(x):
    y = np.matrix.flatten(x, order='F').reshape((-1, 1), order='F')
    return y


def flattening_backward(dl_dy, x, y):
    dl_dx = dl_dy.reshape(x.shape, order='F')
    return dl_dx


def train_slp_linear(mini_batch_x, mini_batch_y):
    
    if len(mini_batch_x) != len(mini_batch_y):
        raise ValueError('The number of batches must be same for images and labels')
    
    if len(mini_batch_x) <= 0:
        raise ValueError('There should be atleast 1 mini batch')
    
    n_size = mini_batch_x[0].shape[0]
    out_size = mini_batch_y[0].shape[0]

    n_batches = len(mini_batch_x)
    batch_size = mini_batch_x[0].shape[1]
    
    # Set the learning rate
    learning_rate = 0.09
    # Set the decay rate
    decay_rate = 0.89

    # Initialize the weights with a Gaussian noise
    w = np.random.normal(0, 1, size=(out_size, n_size))

    # initialize the bias
    b = np.random.normal(0, 1, size=(out_size, 1))

    num_iter = 2500
    
    losses = []

    for iter_n in range(0, num_iter):
        if ((iter_n+1)%1000 == 0):
            learning_rate = learning_rate * decay_rate
        
        batch_dL_dw = np.zeros(out_size * n_size)
        batch_dL_db = np.zeros(out_size)

        batch_idx = iter_n % n_batches
        batch_x = mini_batch_x[batch_idx]
        batch_y = mini_batch_y[batch_idx]

        batch_loss = 0

        for i in range(0, batch_x.shape[1]):
            img_x = batch_x[:, i]
            label_y = batch_y[:, i]
            # predict label
            y_tilde = fc(img_x.reshape(n_size, 1), w, b).reshape(-1)
            # compute loss
            l, dl_dy = loss_euclidean(y_tilde, label_y)
            batch_loss = batch_loss + abs(l)
            # gradient backpropagation
            _, dl_dw, dl_db = fc_backward(dl_dy.reshape(1, out_size), img_x, w, b, label_y.reshape(out_size, 1))

            batch_dL_dw = batch_dL_dw + dl_dw
            batch_dL_db = batch_dL_db + dl_db
        
        w = w - (batch_dL_dw.reshape(w.shape) * learning_rate / batch_size)
        b = b - (batch_dL_db.reshape(b.shape) * learning_rate / batch_size)
        losses.append(batch_loss)

    # plot losses
    # plt.plot(losses)
    # plt.show()

    return w, b


def train_slp(mini_batch_x, mini_batch_y):
    if len(mini_batch_x) != len(mini_batch_y):
        raise ValueError('The number of batches must be same for images and labels')
    
    if len(mini_batch_x) <= 0:
        raise ValueError('There should be atleast 1 mini batch')
    
    n_size = mini_batch_x[0].shape[0]
    out_size = mini_batch_y[0].shape[0]

    n_batches = len(mini_batch_x)
    batch_size = mini_batch_x[0].shape[1]
    
    # Set the learning rate
    learning_rate = 0.18
    # Set the decay rate
    decay_rate = 0.89

    # Initialize the weights with a Gaussian noise
    w = np.random.normal(0, 1, size=(out_size, n_size))

    # initialize the bias
    b = np.random.normal(0, 1, size=(out_size, 1))

    num_iter = 7500
    
    losses = []

    for iter_n in range(0, num_iter):
        if ((iter_n+1)%1000 == 0):
            learning_rate = learning_rate * decay_rate
        
        batch_dL_dw = np.zeros(out_size * n_size)
        batch_dL_db = np.zeros(out_size)

        batch_idx = iter_n % n_batches
        batch_x = mini_batch_x[batch_idx]
        batch_y = mini_batch_y[batch_idx]

        batch_loss = 0

        for i in range(0, batch_x.shape[1]):
            img_x = batch_x[:, i]
            label_y = batch_y[:, i]
            # predict label
            y_tilde = fc(img_x.reshape(n_size, 1), w, b).reshape(-1)
            # compute loss
            l, dl_dy = loss_cross_entropy_softmax(y_tilde, label_y)
            batch_loss = batch_loss + abs(l)
            # gradient backpropagation
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy.reshape(1, out_size), img_x, w, b, label_y.reshape(out_size, 1))

            batch_dL_dw = batch_dL_dw + dl_dw
            batch_dL_db = batch_dL_db + dl_db
        
        w = w - (batch_dL_dw.reshape(w.shape) * learning_rate / batch_size)
        b = b - (batch_dL_db.reshape(b.shape) * learning_rate / batch_size)
        losses.append(batch_loss)

    # plot losses
    # plt.plot(losses)
    # plt.show()

    return w, b


def train_mlp(mini_batch_x, mini_batch_y):
    if len(mini_batch_x) != len(mini_batch_y):
        raise ValueError('The number of batches must be same for images and labels')
    
    if len(mini_batch_x) <= 0:
        raise ValueError('There should be atleast 1 mini batch')
    
    n_size = mini_batch_x[0].shape[0]
    out_size = mini_batch_y[0].shape[0]

    n_batches = len(mini_batch_x)
    batch_size = mini_batch_x[0].shape[1]

    hidden_layer_units = 30
    
    # Set the learning rate
    learning_rate = 0.16
    # Set the decay rate
    decay_rate = 0.99

    # Initialize the weights with a Gaussian noise
    w1 = np.random.normal(0, 1, size=(hidden_layer_units, n_size))
    w2 = np.random.normal(0, 1, size=(out_size, hidden_layer_units))

    # initialize the bias
    b1 = np.random.normal(0, 1, size=(hidden_layer_units, 1))
    b2 = np.random.normal(0, 1, size=(out_size, 1))

    num_iter = 15000
    
    losses = []

    for iter_n in range(0, num_iter):
        if ((iter_n+1)%1000 == 0):
            learning_rate = learning_rate * decay_rate
        
        batch_dL_dw_1 = np.zeros(hidden_layer_units * n_size)
        batch_dL_dw_2 = np.zeros(out_size * hidden_layer_units)
        batch_dL_db_1 = np.zeros(hidden_layer_units)
        batch_dL_db_2 = np.zeros(out_size)

        batch_idx = iter_n % n_batches
        batch_x = mini_batch_x[batch_idx]
        batch_y = mini_batch_y[batch_idx]

        batch_loss = 0

        for i in range(0, batch_x.shape[1]):
            img_x = batch_x[:, i]
            label_y = batch_y[:, i]
            # predict label
            act_hidden = fc(img_x.reshape(n_size, 1), w1, b1).reshape(-1)
            hidden_out = relu(act_hidden)
            y_tilde = fc(hidden_out.reshape(hidden_layer_units, 1), w2, b2).reshape(-1)

            # compute loss
            l, dl_dy_2 = loss_cross_entropy_softmax(y_tilde, label_y)
            batch_loss = batch_loss + abs(l)
            # # gradient backpropagation
            dl_dx_2, dl_dw_2, dl_db_2 = fc_backward(dl_dy_2.reshape(1, out_size), hidden_out, w2, b2, y_tilde)

            dl_dy_1 = relu_backward(dl_dx_2, act_hidden, hidden_out)

            _, dl_dw_1, dl_db_1 = fc_backward(dl_dy_1.reshape(1, hidden_layer_units), img_x, w1, b1, act_hidden)

            batch_dL_dw_1 = batch_dL_dw_1 + dl_dw_1
            batch_dL_dw_2 = batch_dL_dw_2 + dl_dw_2
            batch_dL_db_1 = batch_dL_db_1 + dl_db_1
            batch_dL_db_2 = batch_dL_db_2 + dl_db_2
        
        w1 = w1 - (batch_dL_dw_1.reshape(w1.shape) * learning_rate / batch_size)
        w2 = w2 - (batch_dL_dw_2.reshape(w2.shape) * learning_rate / batch_size)
        b1 = b1 - (batch_dL_db_1.reshape(b1.shape) * learning_rate / batch_size)
        b2 = b2 - (batch_dL_db_2.reshape(b2.shape) * learning_rate / batch_size)
        losses.append(batch_loss)

    # plot losses
    # plt.plot(losses)
    # plt.show()

    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y):
    if len(mini_batch_x) != len(mini_batch_y):
        raise ValueError('The number of batches must be same for images and labels')
    
    if len(mini_batch_x) <= 0:
        raise ValueError('There should be atleast 1 mini batch')
    
    n_size = mini_batch_x[0].shape[0]
    out_size = mini_batch_y[0].shape[0]

    n_batches = len(mini_batch_x)
    batch_size = mini_batch_x[0].shape[1]

    flattened_size = 147
    
    # Set the learning rate
    learning_rate = 0.09
    # Set the decay rate
    decay_rate = 0.9

    filter_shape = (3, 3)
    input_channels = 1

    filter_channels = 3

    # # Initialize the weight with a Gaussian noise
    w_conv = np.random.normal(0, 1, size=(filter_shape[0], filter_shape[1], input_channels, filter_channels))
    w_fc = np.random.normal(0, 1, size=(out_size, flattened_size))

    # initialize the bias
    b_conv = np.random.normal(0, 1, size=(filter_channels, 1))
    b_fc = np.random.normal(0, 1, size=(out_size, 1))

    num_iter = 10000
    
    losses = []

    for iter_n in range(0, num_iter):
        if ((iter_n+1)%1000 == 0):
            learning_rate = learning_rate * decay_rate

        batch_dL_dw_conv = np.zeros((filter_shape[0], filter_shape[1], input_channels, filter_channels))
        batch_dL_dw_fc = np.zeros(out_size * flattened_size)
        batch_dL_db_conv = np.zeros((filter_channels, 1))
        batch_dL_db_fc = np.zeros(out_size)

        batch_idx = iter_n % n_batches
        batch_x = mini_batch_x[batch_idx]
        batch_y = mini_batch_y[batch_idx]

        batch_loss = 0

        for i in range(0, batch_x.shape[1]):
            # (14 x 14) input with 1 channel
            img_x = batch_x[:, i].reshape((14, 14, 1), order='F')
            label_y = batch_y[:, i]

            conv_x = conv(img_x, w_conv, b_conv)

            hidden_out = relu(conv_x)

            pooled_out = pool2x2(hidden_out)

            flattened = flattening(pooled_out)
            
            y_tilde = fc(flattened, w_fc, b_fc)

            # compute loss
            l, dl_dy = loss_cross_entropy_softmax(y_tilde.reshape(-1), label_y)
            batch_loss = batch_loss + abs(l)

            dl_dy_flattened, dl_dw_fc, dl_db_fc = fc_backward(dl_dy, flattened, w_fc, b_fc, y_tilde)

            dl_dy_pooled = flattening_backward(dl_dy_flattened, pooled_out, flattened)

            dl_df = pool2x2_backward(dl_dy_pooled, hidden_out, pooled_out)

            dl_conv_x = relu_backward(dl_df, conv_x, hidden_out)

            dl_dw_conv, dl_db_conv = conv_backward(dl_conv_x, img_x, w_conv, b_conv, conv_x)

            batch_dL_dw_conv = batch_dL_dw_conv + dl_dw_conv
            batch_dL_dw_fc = batch_dL_dw_fc + dl_dw_fc
            batch_dL_db_conv = batch_dL_db_conv + dl_db_conv
            batch_dL_db_fc = batch_dL_db_fc + dl_db_fc
        
        w_conv = w_conv - (batch_dL_dw_conv.reshape(w_conv.shape) * learning_rate / batch_size)
        w_fc = w_fc - (batch_dL_dw_fc.reshape(w_fc.shape) * learning_rate / batch_size)
        b_conv = b_conv - (batch_dL_db_conv.reshape(b_conv.shape) * learning_rate / batch_size)
        b_fc = b_fc - (batch_dL_db_fc.reshape(b_fc.shape) * learning_rate / batch_size)
        losses.append(batch_loss)

    # plot losses
    # plt.plot(losses)
    # plt.show()

    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    main.main_slp_linear()
    main.main_slp()
    main.main_mlp()
    main.main_cnn()
