from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.estimator as learn
import os, shutil
from sklearn import metrics
import csv
from scipy import sparse


tf.enable_eager_execution()

# Get a new directory to hold checkpoints from a neural network.  This allows the neural network to be
# loaded later.  If the erase param is set to true, the contents of the directory will be cleared.
def get_model_dir(name,erase):
    base_path = os.path.join(".","dnn")
    model_dir = os.path.join(base_path,name)
    os.makedirs(model_dir,exist_ok=True)
    if erase and len(model_dir)>4 and os.path.isdir(model_dir):
        shutil.rmtree(model_dir,ignore_errors=True) # be careful, this deletes everything below the specified path
    return model_dir


# Load a Netflix file in the format USER x MOVIE
# Return: the sparse matrix, number of lines and number of columns
def loadCSVasSparse(fileName):

    filehandle = open(fileName,'r')
    csvreader = csv.reader(filehandle)
    rows = sum(1 for row in csvreader)
    filehandle.seek(0)
    row = next(csvreader)
    columns = len(row)
    matrix = sparse.lil_matrix((rows, columns))
    filehandle.seek(0)
    for i,line in enumerate(csvreader):
        matrix[i,] = np.array(line,dtype=int)

    numberOfColumns = len(line)

    return matrix, i, numberOfColumns

def normaliseSparse(A,axis=1):
    if (not sparse.issparse(A)):
        raise Exception("A is not a sparse matrix")
    #A = A.tocsr()
    # A = csr_matrix(A)
    #columns = A.shape[0]
    aux = (A.multiply(A)) # A**2 element wise
    aux = aux.sum(axis=axis)
    #aux = np.array(aux).reshape(1,columns)
    lengths = np.sqrt(aux)
    lengths = 1/lengths # It is not possible to do A/lenght, so workaround
    lengths = sparse.csr_matrix(lengths)
    return A.multiply(lengths)  # Here should be A/lengths if sparse division works well


print("\nNeural Network")


print("Tensor Flow Version: {}".format(tf.__version__))

#from keras import backend
#backend.set_floatx('int32')

def pack_features_vector(features):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)

  return features

#predictors = tf.contrib.data.CsvDataset(['data_1_sample.csv'],[tf.int32]*1000,header=True)
#iterator = dataset.make_initializable_iterator()
#next_element = iterator.get_next()
#labels = tf.contrib.data.CsvDataset( ['data_1_sample.csv'],[tf.int32]*1000,header=True)

#dataset = tf.data.Dataset.zip((predictors, labels))

batch_size = 1

#train_dataset = tf.contrib.data.make_csv_dataset(
# train_dataset = tf.data.experimental.make_csv_dataset(
#     ['data_1_user_movie.csv'],
#     #predictors,
#     batch_size,
#     column_names=None,
#     label_name=None,
#     num_epochs=1,
#     header=False)
#
#
# train_dataset = train_dataset.map(pack_features_vector)

#features, labels = next(iter(train_dataset))


# iterator = train_dataset.make_initializable_iterator()
# sess.run(iterator.initializer)
# features, labels = iterator.get_next()

def get_model(input, classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu, input_shape=(input,) ))  # input shape required)
    model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(classes))
    #model.compile(loss='mean_squared_error', optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
    #              metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

# Function to read a specifc line from NetFlix file
def getLine(lineToRead, file):
 with open(file,'rb') as file:
    #next(file) # skip header  - no header anymore
    if (lineToRead == 1):
        line = file.readline().decode('utf-8')
    else:
        #go to the line
        line = file.readline().decode('utf-8')
        length = len(line) # size of all rows except for the heather
        file.seek(0,0)
        #next(file)  - no header anymore
        # -1 because file pointer begins in 0
        file.seek((lineToRead-1)*length,1) # access line 30878 through the current position
        line = file.readline().decode('utf-8')
        return line

#Build the sparse matrix of all movies
dataset = ['data_1_user_movie.csv','data_2_user_movie.csv','data_3_user_movie.csv']#,'data_4_user_movie.csv']
normalizedSparseMatrix = sparse.lil_matrix((0, 0))
for file in dataset:
    sparseMatrix, numberOfLines, numberOfColumns = loadCSVasSparse(file)
    sparseMatrix = normaliseSparse(sparseMatrix)
    normalizedSparseMatrix = sparse.hstack([normalizedSparseMatrix,sparseMatrix])

normalizedSparseMatrix = normalizedSparseMatrix.asformat('csr')
# Train for user 30878
#lineToRead = 30878
#user_30878 = np.array(getLine(lineToRead, file).split(','), dtype=int)  # movies rated from user
#user_30878 = normaliseSparse(sparse.csr_matrix(user_30878[0:4000]))
user_30878 = normalizedSparseMatrix[30878-1,]
x = tf.cast(user_30878.toarray(), dtype=tf.float32)


optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
global_step = tf.Variable(0)
#sess = tf.Session()

# keep results for plotting
train_loss_results = []
train_accuracy_results = []
from tensorflow import contrib
tfe = contrib.eager
num_epochs = 5
model = get_model(normalizedSparseMatrix.shape[1],normalizedSparseMatrix.shape[1])


max_cossine = 0.0
count = 0
for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

    # Training loop
    #for x in train_dataset:
      #for i in range(0,numberOfLines):
    for y in normalizedSparseMatrix:
        # Optimize the model
        #y = sparseMatrix[i,0:4000]
        cosine_similarity = float((user_30878 @  y.transpose()).toarray())
        if round((cosine_similarity),2) > max_cossine and round((cosine_similarity),2) < 0.99:
            max_cossine = cosine_similarity

        if (cosine_similarity < 0.6):
            continue

        count += 1
        y = tf.cast(y.todense(), dtype=tf.float32)

        with tf.GradientTape() as tape:
            pred = model(x)
            loss_value = tf.losses.cosine_distance(labels=y, predictions=pred,axis=0)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)

        # Track progress
        epoch_loss_avg(loss_value)  # add current batch loss
        # compare predicted label to actual label
        epoch_accuracy(pred, y)

    # end epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 1 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))

model.save('my_model.h5')
print("Max cossine: ", max_cossine)
print("Number of neighboors: ", count)


#-144816456.647