from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.estimator as learn
import os, shutil
from sklearn import metrics

# Get a new directory to hold checkpoints from a neural network.  This allows the neural network to be
# loaded later.  If the erase param is set to true, the contents of the directory will be cleared.
def get_model_dir(name,erase):
    base_path = os.path.join(".","dnn")
    model_dir = os.path.join(base_path,name)
    os.makedirs(model_dir,exist_ok=True)
    if erase and len(model_dir)>4 and os.path.isdir(model_dir):
        shutil.rmtree(model_dir,ignore_errors=True) # be careful, this deletes everything below the specified path
    return model_dir

print("\nNeural Network")

train = np.zeros((5,5))

#User 0
train[0][0] = 4
train[4][0] = 3

#Others
train[0][1] = 4
train[4][1] = 3
train[1][1] = 2

train[0][2] = 4
train[4][2] = 3
train[1][2] = 2

train[0][3] = 4
train[4][3] = 3
train[1][3] = 2

train[0][4] = 4
train[4][4] = 3
train[1][4] = 2



classifier = MLPRegressor(hidden_layer_sizes=(20), activation='relu', solver='adam',
                          alpha=1e-3, batch_size='auto', learning_rate='constant',
                          learning_rate_init=0.001, power_t=0.5, max_iter=200,
                          shuffle=True, random_state=7, tol=0.0001,
                          verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                          early_stopping=False, validation_fraction=0.1, beta_1=0.9,
                          beta_2=0.999, epsilon=1e-08)

classifier.fit(train, train)
test = np.array([4,0,0,0,3]).reshape(1,-1)
pred = classifier.predict(test)

print(pred)

cv_y = []
cv_pred = []

cv_y.append(test)
cv_pred.append(pred)

cv_y = np.concatenate(cv_y)
cv_pred = np.concatenate(cv_pred)
score = np.sqrt(metrics.mean_squared_error(cv_y, cv_pred))
print("\n Average RMSE: {}".format(score))


# kf = KFold(5, random_state=7, shuffle=True)
# cv_y = []
# cv_pred = []
# fold = 0
#
# for training, test in kf.split(x_train):
#     fold += 1
#     pred = []
#
#     scaler = preprocessing.StandardScaler()
#     x_train_fold = scaler.fit_transform(x_train[training])
#     x_test_fold = scaler.transform(x_train[test])
#
#     y_train_fold = y_train[training]
#     y_test_fold = y_train[test]
#
#     classifier.fit(x_train_fold, y_train_fold)
#     pred = classifier.predict(x_test_fold)
#     cv_y.append(y_test_fold)
#     cv_pred.append(pred)
#
# cv_y = np.concatenate(cv_y)
# cv_pred = np.concatenate(cv_pred)
# score = np.sqrt(metrics.mean_squared_error(cv_y, cv_pred))
# print("\n Average RMSE: {}".format(score))
#
