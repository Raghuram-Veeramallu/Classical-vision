import os
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from itertools import product
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from scipy import stats
from pathlib import Path, PureWindowsPath


def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list


def compute_confusion_matrix(true_values, predicted_values, label_classes):

    confusion_matrix = np.zeros((len(label_classes), len(label_classes)), dtype=int)

    for true_label, prediction in zip(true_values ,predicted_values):
        confusion_matrix[true_label][prediction] += 1
    
    return confusion_matrix


def calculate_accuracy(confusion_matrix, n_predicted_values):
    # trace of the confusion matrix has the correct predictions
    correct_predictions = np.sum(np.trace(confusion_matrix))
    return correct_predictions/n_predicted_values


def compute_dsift(img, stride=8, size=8):
    # using opencv2 to obtain SIFT keypoints and descriptors
    sift = cv2.SIFT_create()

    # defining all the possible keypoints to calculate the dense sift at
    key_points = [cv2.KeyPoint(x=(h+size/2), y=(w+size/2), size=size) for w, h in product(np.arange(0, img.shape[1], stride), np.arange(0, img.shape[0], stride))]
    _, dense_feature = sift.compute(img, key_points)

    return np.array(dense_feature)


def get_tiny_image(img, output_size):
    w, h = output_size
    im_w, im_h = img.shape

    px_conv_width = math.floor(im_w / w)
    px_conv_height = math.floor(im_h / h)

    feature = np.zeros(output_size)

    for y in range(0, h):
        for x in range(0, w):
            feature[x, y] = np.average(img[(x * px_conv_width):((x + 1) * px_conv_width), (y * px_conv_height):((y + 1) * px_conv_height)])

    # normalize the image by having zero mean and unit length
    # zero mean
    feature_mean = np.mean(feature)

    # unit length
    feature = feature/np.linalg.norm(feature)

    feature = feature - feature_mean

    return feature


def predict_knn(feature_train, label_train, feature_test, k):

    knn_model = NearestNeighbors(n_neighbors=k)
    knn_model.fit(feature_train, label_train)
    pred_nearest_points = knn_model.kneighbors(feature_test, return_distance=False)
    pred_potential_classes = label_train[pred_nearest_points]
    label_test_pred = stats.mode(pred_potential_classes, axis=1, keepdims=False).mode
    return label_test_pred


def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):

    tiny_img_size = [7, 7]
    k_neighbors = 6

    # 1: Load training and testing images
    # 2: Build image prepresentation
    img_train_tiny = []
    for img_path in img_train_list:
        img = cv2.imread(img_path, 0)
        tiny_img = get_tiny_image(img, tiny_img_size)
        # plt.imshow(img, cmap='gray')
        # plt.show()
        # plt.imshow(tiny_img, cmap='gray')
        # plt.show()
        img_train_tiny.append(tiny_img.reshape(-1))

    img_test_tiny = []
    for img_path in img_test_list:
        img = cv2.imread(img_path, 0)
        tiny_img = get_tiny_image(img, tiny_img_size)
        img_test_tiny.append(tiny_img.reshape(-1))

    # Encoding images to integers for KNN to be able to recognize them
    label_encoder = LabelEncoder()
    label_encoder.fit(label_classes)
    encoded_training_labels = label_encoder.transform(label_train_list)

    # 3: Train a classifier using the representations of the training images
    label_test_pred = predict_knn(np.asarray(img_train_tiny), encoded_training_labels, np.asarray(img_test_tiny), k_neighbors)
    # 4: Classify the testing data.
    encoded_test_labels = label_encoder.transform(label_test_list)
    # 5: Compute accuracy of testing data classification.
    confusion = compute_confusion_matrix(encoded_test_labels, label_test_pred, label_classes)
    # confusion = confusion_matrix(encoded_test_labels, label_test_pred)
    accuracy = calculate_accuracy(confusion, len(label_test_pred))

    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def build_visual_dictionary(dense_feature_list, dict_size):
    kmeans = KMeans(n_clusters=dict_size, n_init=10, max_iter=250)
    kmeans.fit(dense_feature_list)
    vocab = kmeans.cluster_centers_

    return vocab


def compute_bow(feature, vocab):

    d_size = vocab.shape[0]
    train_labels = np.array(list(range(0, d_size)))

    near_neigh_model = NearestNeighbors(n_neighbors=1)
    near_neigh_model.fit(vocab, train_labels)
    pred_nearest_points = near_neigh_model.kneighbors(feature, return_distance=False)
    pred_potential_classes = train_labels[pred_nearest_points]
    classification = stats.mode(pred_potential_classes, axis=1, keepdims=False).mode
    bow = np.zeros(d_size)
    for each_class in classification:
        bow[each_class] += 1
    bow_feature = np.asarray(bow)
    bow_feature = bow / np.linalg.norm(bow)

    return bow_feature


def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):

    k_neighbors = 9
    d_size = 50

    dsift_size = 20
    dsift_stride = 20
    
    # 1: Load training and testing images
    img_train_dense_features_stacked = None
    img_train_dense_features = []
    for img_path in img_train_list:
        img = cv2.imread(img_path, 0)
        dense_features = compute_dsift(img, dsift_stride, dsift_size)
        img_train_dense_features.append(dense_features)
        if (img_train_dense_features_stacked is None):
            img_train_dense_features_stacked = dense_features
        else:
            img_train_dense_features_stacked = np.vstack([img_train_dense_features_stacked, dense_features])
    
    vocab_knn_bow = build_visual_dictionary(img_train_dense_features_stacked, d_size)
    np.savetxt('./scene-recognition/vocab_knn_bow.txt', vocab_knn_bow)
    # vocab_knn_bow = np.loadtxt('vocab_knn_bow.txt')

    # 2: Build image prepresentation
    img_train_bow_features = []
    for dfeature in img_train_dense_features:
        bow = compute_bow(dfeature, vocab_knn_bow)
        img_train_bow_features.append(bow)


    img_test_dense_features = []
    img_test_bow_features = []
    for img_path in img_test_list:
        img = cv2.imread(img_path, 0)
        dense_features = compute_dsift(img, dsift_stride, dsift_size)
        img_test_dense_features.append(dense_features)
        bow = compute_bow(dense_features, vocab_knn_bow)
        img_test_bow_features.append(bow)

    # Encoding images to integers for KNN to be able to recognize them
    label_encoder = LabelEncoder()
    label_encoder.fit(label_classes)
    encoded_training_labels = label_encoder.transform(label_train_list)

    # 3: Train a classifier using the representations of the training images
    label_test_pred = predict_knn(np.asarray(img_train_bow_features), encoded_training_labels, np.asarray(img_test_bow_features), k_neighbors)
    # 4: Classify the testing data.
    encoded_test_labels = label_encoder.transform(label_test_list)
    # 5: Compute accuracy of testing data classification.
    confusion = compute_confusion_matrix(encoded_test_labels, label_test_pred, label_classes)
    accuracy = calculate_accuracy(confusion, len(label_test_pred))

    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def predict_svm(feature_train, label_train, feature_test, n_classes):
    all_label_test_pred = []

    for i in range(0, n_classes):
        binary_label_train = list(map(int, label_train == i))
        SVM = LinearSVC(C=0.837)
        SVM.fit(feature_train, binary_label_train)
        each_label_test_pred = SVM.decision_function(feature_test)
        all_label_test_pred.append(each_label_test_pred)
    
    # Maximum probability among the n_classes is the class to which the test feature belongs to
    label_test_pred = [np.argmax(np.array(all_label_test_pred)[:, i]) for i in range(0, len(feature_test))]

    return label_test_pred


def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):

    d_size = 50
    
    dsift_size = 20
    dsift_stride = 20

    # 1: Load training and testing images
    img_train_dense_features_stacked = None
    img_train_dense_features = []
    for img_path in img_train_list:
        img = cv2.imread(img_path, 0)
        dense_features = compute_dsift(img, dsift_stride, dsift_size)
        img_train_dense_features.append(dense_features)
        if (img_train_dense_features_stacked is None):
            img_train_dense_features_stacked = dense_features
        else:
            img_train_dense_features_stacked = np.vstack([img_train_dense_features_stacked, dense_features])
    
    vocab_svm_bow = build_visual_dictionary(img_train_dense_features_stacked, d_size)
    np.savetxt('./scene-recognition/vocab_svm_bow.txt', vocab_svm_bow)
    # vocab_svm_bow = np.loadtxt('vocab_knn_bow.txt')

    # 2: Build image prepresentation
    img_train_bow_features = []
    for dfeature in img_train_dense_features:
        bow = compute_bow(dfeature, vocab_svm_bow)
        img_train_bow_features.append(bow)


    img_test_dense_features = []
    img_test_bow_features = []
    for img_path in img_test_list:
        img = cv2.imread(img_path, 0)
        dense_features = compute_dsift(img, dsift_stride, dsift_size)
        img_test_dense_features.append(dense_features)
        bow = compute_bow(dense_features, vocab_svm_bow)
        img_test_bow_features.append(bow)

    # Encoding images to integers for SVM to be able to recognize them
    label_encoder = LabelEncoder()
    label_encoder.fit(label_classes)
    encoded_training_labels = label_encoder.transform(label_train_list)

    # 3: Train a classifier using the representations of the training images
    label_test_pred = predict_svm(np.asarray(img_train_bow_features), encoded_training_labels, np.asarray(img_test_bow_features), len(label_classes))
    # 4: Classify the testing data.
    encoded_test_labels = label_encoder.transform(label_test_list)
    # 5: Compute accuracy of testing data classification.
    confusion = compute_confusion_matrix(encoded_test_labels, label_test_pred, label_classes)
    accuracy = calculate_accuracy(confusion, len(label_test_pred))

    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info("./scene-recognition/data/")

    classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
    
    classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
