from __future__ import print_function
import numpy as np
np.random.seed(777)

import pandas as pd
from os import path

import gc
gc.enable()

import sys
sys.path.append('G:/kaggle/dnn_model/')
sys.setrecursionlimit(50000)


path_img = "G:/kaggle/sound/cqt_img/png_train64/"
path_img_test = "G:/kaggle/sound/cqt_img/png_test64/"

train = pd.read_csv('G:/kaggle/sound/_dl/train.csv')
test = pd.read_csv("G:/kaggle/sound/_dl/sample_submission.csv")


from keras.utils import Sequence, to_categorical
from keras.preprocessing.image import load_img, img_to_array, array_to_img, random_rotation

X_train = []
for fname in train.fname.values:  
    X_train.append(img_to_array(load_img(path.join(path_img, fname.replace(".wav", ".png")))))

X_train = np.array(X_train)
#y_train = ilabel2any.label2(train.label.values, _type="onehot")
np.save("G:/kaggle/sound/cqt_img/png_train64/trainX.npy", X_train)
#np.save("G:/kaggle/sound/cqt_img/png_train64/y_train.npy", y_train)

X_test = []
for fname in test.fname.values:  
    X_test.append(img_to_array(load_img(path.join(path_img_test, fname.replace(".wav", ".png")))))

X_test = np.array(X_test)
np.save("G:/kaggle/sound/cqt_img/png_test64/X_test.npy", X_test)

"""
X_train = np.load("G:/kaggle/sound/cqt_img/png_train64/trainX.npy")
y_train = np.load("G:/kaggle/sound/cqt_img/png_train64/y_train.npy")
"""

LABELS = list(train.label.unique())
label_idx = {label: i for i, label in enumerate(LABELS)}
train.set_index("fname", inplace=True)
test.set_index("fname", inplace=True)
train["label_idx"] = train.label.apply(lambda x: label_idx[x])

#X_train = np.load("G:/kaggle/sound/cqt_img/png_train64/trainX.npy")
#X_test = np.load("G:/kaggle/sound/cqt_img/png_test64/X_test.npy")
y_train = to_categorical(train.label_idx, num_classes=len(LABELS))

gc.collect()


import os
from thirdparty.keras import resnet_groupnorm as resnet
import numpy as np
import sklearn.metrics as metrics

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from sklearn.cross_validation import StratifiedKFold

img_rows, img_cols = 64, 64
img_channels = 3

batch_size = 100
nb_classes = len(LABELS)
nb_epoch = 30

img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
bottleneck = False
width = 1
weight_decay = 1e-3

PREDICTION_FOLDER = "fix_predictions"

if not path.exists(PREDICTION_FOLDER):
    os.mkdir(PREDICTION_FOLDER)

if path.exists('logs/' + PREDICTION_FOLDER):
    shutil.rmtree('logs/' + PREDICTION_FOLDER)


skf = StratifiedKFold(train.label_idx, n_folds=10)
for i, (train_split, val_split) in enumerate(skf):
    X, y, X_val, y_val = X_train[train_split], y_train[train_split], X_train[val_split], y_train[val_split]

    model = resnet.ResNet18(img_dim, width=width, bottleneck=bottleneck, weight_decay=weight_decay, classes=nb_classes)
    
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=Adam(lr=8e-4), 
        metrics=["accuracy"]
    )

    print("#"*50)
    print("Fold: ", i)
    
    callbacks_list=[
        EarlyStopping(monitor="val_loss", mode="min", patience=5),
        ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=1e-5),
        ModelCheckpoint(f"fix_cqt_resnet18_best_loss_{i}.h5", monitor='val_loss', verbose=1, save_best_only=True)
    ]

    model.fit(
        X, y,
        callbacks=callbacks_list,
        batch_size=batch_size,
        epochs=nb_epoch,
        verbose=1,
        validation_data=(X_val, y_val)
    )

    model.load_weights(f"fix_cqt_resnet18_best_loss_{i}.h5")

     # Save test predictions
    predictions = model.predict(X_test, batch_size=64, verbose=1)
    np.save(PREDICTION_FOLDER + "/fix_test_predictions_%d.npy"%i, predictions)

    top_3 = np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(x)) for x in top_3]
    test['label'] = predicted_labels
    test[['label']].to_csv(PREDICTION_FOLDER + "/fix_predictions_%d.csv"%i)

    del model
    gc.collect()

"""
for i in range(10):
    model = resnet.ResNet18(img_dim, width=width, bottleneck=bottleneck, weight_decay=weight_decay, classes=nb_classes)
    
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=Adam(lr=8e-4), 
        metrics=["accuracy"]
    )

    model.load_weights(f"fix_cqt_resnet18_best_loss_{i}.h5")

    # Save test predictions
    predictions = model.predict(X_test, batch_size=64, verbose=1)
    np.save(PREDICTION_FOLDER + "/fix_test_predictions_%d.npy"%i, predictions)

    top_3 = np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(x)) for x in top_3]
    test['label'] = predicted_labels
    test[['label']].to_csv(PREDICTION_FOLDER + "/fix_predictions_%d.csv"%i)





pred_list = []
for i in range(10):
    pred_list.append(np.load("G:/kaggle/sound/CQT-resnet18/predictions/test_predictions_%d.npy"%i))

prediction = np.ones_like(pred_list[0])
for pred in pred_list:
    prediction = prediction*pred

prediction = prediction**(1./len(pred_list))
# Make a submission file
top_3 = np.array(LABELS)[np.argsort(-prediction, axis=1)[:, :3]]
predicted_labels = [' '.join(list(x)) for x in top_3]
test = pd.read_csv("G:/kaggle/sound/_dl/sample_submission.csv")
test['label'] = predicted_labels
test[['fname', 'label']].to_csv("cqtsubmission.csv", index=False)




"""