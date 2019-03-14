from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.estimator as learn
import os, shutil
from sklearn import metrics

# Get a new directory to hold checkpoints from a neural network.  This allows the neural network to be
# loaded later.  If the erase param is set to true, the contents of the directory will be cleared.
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasRegressor

def get_model_dir(name,erase):
    base_path = os.path.join(".","dnn")
    model_dir = os.path.join(base_path,name)
    os.makedirs(model_dir,exist_ok=True)
    if erase and len(model_dir)>4 and os.path.isdir(model_dir):
        shutil.rmtree(model_dir,ignore_errors=True) # be careful, this deletes everything below the specified path
    return model_dir


print("Tensor Flow Version: {}".format(tf.__version__))

tf.enable_eager_execution()
sess = tf.Session()

def pack_features_vector(features):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, features

batch_size = 32

#train_dataset = tf.contrib.data.make_csv_dataset(
train_dataset = tf.data.experimental.make_csv_dataset(
    ['data_1_sample.csv'],
    #predictors,
    batch_size,
    column_names=None,
    label_name=None,
    num_epochs=1,
    header=True)

train_dataset = train_dataset.map(pack_features_vector)

#features, labels = next(iter(train_dataset))



def get_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(train_dataset.output_shapes[0][1],) ))  # input shape required)
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(train_dataset.output_shapes[0][1]))
    model.compile(loss='mean_squared_error', optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

model = get_model()
np.random.seed(999)
estimators = []
estimators.append(('mlp', KerasRegressor(build_fn=model, epochs=1, batch_size=32, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=1, random_state=999)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))

model.save('my_model.h5')