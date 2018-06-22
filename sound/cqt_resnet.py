from __future__ import print_function
import numpy as np
import pandas as pd
from os import path
import gc

import sys
sys.path.append('G:/kaggle/dnn_model/')
sys.setrecursionlimit(50000)


class label2any:
    def __init__(self, labels):
        self.u_label = np.unique(labels)
        self.num_class = len(self.u_label)
        self.c2i = {}
        self.i2c = {}
        for i, c in enumerate(self.u_label):
            self.c2i[c] = i
            self.i2c[i] = c
    def label2(self, labels, _type):
        if "num" == _type:
            return [self.c2i[label] for label in labels]
        if "onehot" == _type:
            onehots = np.zeros((len(labels), self.num_class), dtype=np.int32)
            for i, label in enumerate(labels):
                onehots[i][self.c2i[label]] = 1
            return onehots
    def num2(self, nums):
        return [self.i2c[num] for num in nums]
    def onehot2(self, onehots):
        return [idx for onehot in onehots for idx, val in enumerate(onehot) if 1 == val]


gc.enable()

path_img = "G:/kaggle/sound/cqt_img/png_train64/"
train = pd.read_csv('../_dl/train.csv')
ilabel2any = label2any(train.label.values)

img_rows, img_cols = 64, 64
img_channels = 3

fname_trainset = []
label_trainset = []
fname_valset = []
label_valset = []

for label in ilabel2any.u_label:
    tmp_set = train[train["label"] == label]
    tmp_set.reset_index(drop=True, inplace=True)
    tmp_nb_train = int(np.trunc(len(tmp_set) * 0.7))
    tmp_train = tmp_set[tmp_set.index < tmp_nb_train]
    tmp_val = tmp_set[tmp_set.index >= tmp_nb_train]
    fname_trainset.append(tmp_train.fname.values)
    label_trainset.append(tmp_train.label.values)
    fname_valset.append(tmp_val.fname.values)
    label_valset.append(tmp_val.label.values)

X_train = np.concatenate(fname_trainset, axis=0)
y_train = np.concatenate(label_trainset, axis=0)
X_val = np.concatenate(fname_valset, axis=0)
y_val = np.concatenate(label_valset, axis=0)

trainY = ilabel2any.label2(y_train, _type="onehot")
testY = ilabel2any.label2(y_val, _type="onehot")

"""
trainX = []
testX = []

from keras.preprocessing.image import load_img, img_to_array, array_to_img, random_rotation

for fname in X_train:  
    trainX.append(img_to_array(load_img(path.join(path_img, fname.replace(".wav", ".png")))))
    gc.collect()
for fname in X_val:  
    testX.append(img_to_array(load_img(path.join(path_img, fname.replace(".wav", ".png")))))
    gc.collect()

trainX = np.array(trainX)
testX = np.array(testX)
np.save("G:/kaggle/sound/cqt_img/png_train64/trainX.npy", trainX)
np.save("G:/kaggle/sound/cqt_img/png_train64/testX.npy", testX)
"""

trainX = np.load("G:/kaggle/sound/cqt_img/png_train64/trainX.npy")
testX = np.load("G:/kaggle/sound/cqt_img/png_train64/testX.npy")


from thirdparty.keras import resnet_groupnorm as resnet
import numpy as np
import sklearn.metrics as metrics

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K

batch_size = 100
nb_classes = ilabel2any.num_class
nb_epoch = 30

img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
bottleneck = False
width = 1
weight_decay = 1e-3

model = resnet.ResNet18(img_dim, width=width, bottleneck=bottleneck, weight_decay=weight_decay,
                        classes=nb_classes)
print("Model created")

model.summary()
optimizer = Adam(lr=1e-3) # Using Adam instead of SGD to speed up training
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
print("Finished compiling")
print("Building model...")
"""
(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX = trainX.astype('float32')
testX = testX.astype('float32')

trainX = resnet.preprocess_input(trainX)
testX = resnet.preprocess_input(testX)

Y_train = np_utils.to_categorical(trainY, nb_classes)
Y_test = np_utils.to_categorical(testY, nb_classes)
"""

# Load model
weights_file="ResNet18-cqt.h5"
if path.exists(weights_file):
    model.load_weights(weights_file, by_name=True)
    print("Model loaded.")

out_dir="weights/"

callbacks_list=[
    #EarlyStopping(monitor="val_loss", mode="min", patience=5),
    ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=1e-5),
    ModelCheckpoint("Y:/cqt_resnet18_best_loss.h5", monitor='val_loss', verbose=1, save_best_only=True)
]

model.fit(
    trainX, trainY,
    callbacks=callbacks_list,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=1,
    validation_data=(testX, testY)
)

model.save_weights('Y:/cqt64_resnet18.h5')
"""
yPreds = model.predict(testX)
yPred = np.argmax(yPreds, axis=1)
yTrue = testY

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)

"""