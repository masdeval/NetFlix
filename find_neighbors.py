#
# For each user in the probe dataset, find those users that are similar in rating movies.
#

import numpy as np
import tensorflow as tf
import csv
from scipy import sparse
import os


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

dataset = ['data_1_user_movie.csv','data_2_user_movie.csv','data_3_user_movie.csv','data_4_user_movie.csv']
normalizedSparseMatrix = sparse.lil_matrix((0, 0))
for file in dataset:
    sparseMatrix, numberOfLines, numberOfColumns = loadCSVasSparse(file)
    sparseMatrix = normaliseSparse(sparseMatrix)
    normalizedSparseMatrix = sparse.hstack([normalizedSparseMatrix,sparseMatrix])

normalizedSparseMatrix = normalizedSparseMatrix.asformat('csr')
users = set()
files = os.listdir("./neighbors")
#read the users from probe.txt and for each one find its neighbors
with open('netflix-prize-data/probe.txt','r') as file:
    for line in file:
        if line.__contains__(':'): # id of a movie
            continue
        if (int(line)>100000): # I selected only 100.000 users
            continue
        name = str(int(line)) + '.txt'
        if files.__contains__(name):
            continue
        if users.__contains__(line):
            continue
        else:
            users.add(line)

        user = normalizedSparseMatrix[int(line)-1,]
        neighbors = open(name, 'w')
        for neighbor,y in enumerate(normalizedSparseMatrix):
            cosine_similarity = float((user @ y.transpose()).toarray())

            if (cosine_similarity < 0.6):
                continue

            neighbors.write(str(neighbor+1)+'\n')

        neighbors.close()