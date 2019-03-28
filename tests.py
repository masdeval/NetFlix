from collections import defaultdict
import csv
import pandas as pd
import numpy as np


data = defaultdict(lambda:defaultdict(lambda: int))

dd = [2,32,53,64]

for a in dd[1:3]:
    print(a)


user1 = np.array([2,3,3,4,4,3,5,0,0,0,0,0,0])
user2 = np.array([0,0,0,0,0,0,2,3,3,4,4,3,5])

#cosine similarity
from sklearn.metrics.pairwise import cosine_similarity as cosine
print(cosine(user1.reshape(1,-1),user2.reshape(1,-1)))

#correlation
#print(np.corrcoef(user1, user2))

user1 = np.array([2,3,3,4,4,3,5,0,0,0,0,0,0])
user2 = np.array([5,4,5,1,1,1,1,0,0,0,0,0,0])
#cosine similarity
from sklearn.metrics.pairwise import cosine_similarity as cosine
print(cosine(user1.reshape(1,-1),user2.reshape(1,-1)))
#correlation
#print(np.corrcoef(user1, user2))


user1 = np.array([2,3,3,4,4,3,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
user2 = np.array([1,3,3,4,4,3,5,1,3,2,1,4,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
#cosine similarity
from sklearn.metrics.pairwise import cosine_similarity as cosine
print(cosine(user1.reshape(1,-1),user2.reshape(1,-1)))
#correlation
#print(np.corrcoef(user1, user2))


## Como posicionar ponteiro de arquivo numa linha especifica
lineToRead = 2
with open('data_1_user_movie.csv','rb') as file:
    next(file) # skip heather
    if (lineToRead == 1):
        line = file.readline().decode('utf-8')
    else:
        #go to the line
        line = file.readline().decode('utf-8')
        length = len(line) # size of all rows except for the heather
        file.seek(0,0)
        next(file)
        file.seek((lineToRead-1)*length,1) # access line 30878 through the current position
        line = file.readline().decode('utf-8')


print('\n\n')
## Entendendo sparse matrix

from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve, norm


def normaliseSparse(A,axis=1):
    if (not issparse(A)):
        raise Exception("A is not a sparse matrix")

    #A = A.tocsr()
    # A = csr_matrix(A)
    columns = A.shape[0]
    aux = (A.multiply(A)) # A**2 element wise
    aux = aux.sum(axis=axis)
    #aux = np.array(aux).reshape(1,columns)
    lengths = np.sqrt(aux)
    lengths = 1/lengths
    lengths = csr_matrix(lengths)
    return A.multiply(lengths)

A = np.array([[1, 2, 0], [0, 0, 3], [4, 0, 5]])

vector1 = A[0,]
vector2 = A[0,]
print(cosine(vector1.reshape(1,-1),vector2.reshape(1,-1)))
#print(A @ A) # matrix multiplication
#print(A * A) # element multiplication
#print(normaliseSparse(A))

print('\n\n')

A = csr_matrix(A)
vector1 = A[0,]
vector2 = A[0,]

print(cosine(vector1.reshape(1,-1),vector2.reshape(1,-1)))
#print(A.multiply(A).todense()) # element multiplication
#print((A*A).todense()) # matrix multiplication
#print((A@A).todense()) # matrix multiplication
#print((A*2).todense())
A_normalized = normaliseSparse(A)
vector1 = A_normalized[0,]
vector2 = A_normalized[0,].transpose()
print(vector1 @ vector2)

print('\n\n')

import tensorflow as tf
tf.enable_eager_execution()
y = tf.cast(user1, dtype=tf.float32)
pred = tf.cast(user2, dtype=tf.float32)
with tf.GradientTape() as tape:
 loss_value = tf.losses.cosine_distance(labels=y, predictions=pred,axis=0)
print(loss_value)


import os

files = os.listdir("./neighbors")
if files.__contains__("439\n.txt"):
    print('coco');